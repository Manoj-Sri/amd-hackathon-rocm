"""parse_config tool — extract WorkloadConfig from a user's training script/args.

Three input shapes are supported, dispatched by file extension:

  * ``.py`` — Python source. Walks the AST for HuggingFace
    ``TrainingArguments(...)`` calls, ``DataLoader(...)`` calls,
    ``torch.compile(...)`` calls, ``model.gradient_checkpointing_enable()``
    invocations, and ``os.environ["X"] = "..."`` assignments.
  * ``.json`` — ``json.load`` of a dict shaped like ``TrainingArguments`` kwargs.
  * ``.yaml`` / ``.yml`` — ``yaml.safe_load`` of the same shape.

Before storing the source in ``WorkloadConfig.raw_source`` we run a regex
redaction pass for common secret shapes (HF/OpenAI/GitHub/Bearer tokens,
``$HOME`` paths, ``s3://``/``ws[s]://`` URIs). Each scrubbed pattern is
recorded in ``WorkloadConfig.redactions`` so the report can flag what was
removed.

Anything that doesn't map cleanly onto a ``WorkloadConfig`` field is shoved
into ``extras`` so we don't lose information.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import yaml

from agent.schemas import ToolResult, WorkloadConfig
from agent.tools import Tool


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

# (label, compiled pattern). Order matters: more specific tokens first so we
# don't accidentally swallow them inside a generic Bearer rule.
_REDACTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("hf_token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("github_token", re.compile(r"gho_[A-Za-z0-9]{20,}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._\-]{8,}")),
    ("home_path", re.compile(r"/home/[^/\s\"'`]+")),
    ("s3_uri", re.compile(r"s3://[^\s\"'`]+")),
    ("ws_uri", re.compile(r"wss?://[^\s\"'`]+")),
]


def _redact(source: str) -> tuple[str, list[str]]:
    """Apply every redaction pattern. Returns (clean_source, labels_hit)."""
    labels: list[str] = []
    cleaned = source
    for label, pattern in _REDACTION_PATTERNS:
        if pattern.search(cleaned):
            labels.append(label)
            cleaned = pattern.sub(f"<REDACTED:{label}>", cleaned)
    return cleaned, labels


def _redact_extras(extras: dict[str, Any]) -> list[str]:
    """Mutate ``extras`` in place to scrub secret-shaped string values.

    Returns the labels that fired so we can merge them into ``redactions``.
    Lists/dicts inside extras are walked recursively. Non-string leaves are
    untouched.
    """
    labels: list[str] = []

    def _scrub(value: Any) -> Any:
        if isinstance(value, str):
            cleaned, hits = _redact(value)
            labels.extend(hits)
            return cleaned
        if isinstance(value, dict):
            return {k: _scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_scrub(v) for v in value]
        return value

    for k, v in list(extras.items()):
        extras[k] = _scrub(v)
    return labels


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _literal(node: ast.AST) -> Any:
    """Best-effort literal evaluation. Returns ``None`` for non-literal expressions."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _attr_chain(node: ast.AST) -> str:
    """Render ``a.b.c`` from a Name/Attribute chain. Empty string for everything else."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _call_name(node: ast.Call) -> str:
    """Resolve the dotted name of a Call's target (e.g. ``torch.compile``)."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return _attr_chain(func)
    return ""


def _kwargs_to_dict(
    node: ast.Call, dict_constants: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Pull literal kwargs out of a Call.

    Also resolves ``**dict_var`` splats when the splat target was assigned a
    literal dict elsewhere in the module (the resolved dict is in
    ``dict_constants``). This is defensive against the common refactor pattern
    ``_ta = dict(...); TrainingArguments(**_ta)`` — if the parser doesn't
    follow the splat, every TrainingArguments field disappears and the agent
    reasons over HF defaults instead of the script's actual values.
    """
    out: dict[str, Any] = {}
    dict_constants = dict_constants or {}
    for kw in node.keywords:
        if kw.arg is None:
            # `**something` splat. If something is a Name that resolves to a
            # dict literal we collected in pass 1, lift those entries in.
            if isinstance(kw.value, ast.Name):
                resolved = dict_constants.get(kw.value.id)
                if resolved:
                    for k, v in resolved.items():
                        # Don't override anything an explicit kwarg already set.
                        out.setdefault(k, v)
            continue
        val = _literal(kw.value)
        if val is not None or isinstance(kw.value, ast.Constant):
            out[kw.arg] = val
    return out


def _collect_dict_constants(tree: ast.AST) -> dict[str, dict[str, Any]]:
    """Find module-level ``NAME = dict(k=v, ...)`` and ``NAME = {k: v, ...}``
    assignments where every value is a literal. Returns name → resolved dict.

    Used by ``_kwargs_to_dict`` to follow ``**NAME`` splats.
    """
    constants: dict[str, dict[str, Any]] = {}
    for stmt in getattr(tree, "body", []):
        if not isinstance(stmt, ast.Assign):
            continue
        targets = [t for t in stmt.targets if isinstance(t, ast.Name)]
        if not targets:
            continue
        resolved: dict[str, Any] | None = None
        # `dict(k=v, ...)` form
        if (
            isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "dict"
        ):
            resolved = {}
            for kw in stmt.value.keywords:
                if kw.arg is None:
                    continue
                val = _literal(kw.value)
                if val is not None or isinstance(kw.value, ast.Constant):
                    resolved[kw.arg] = val
        # `{"k": v, ...}` form
        elif isinstance(stmt.value, ast.Dict):
            resolved = {}
            for k_node, v_node in zip(stmt.value.keys, stmt.value.values):
                if not isinstance(k_node, ast.Constant) or not isinstance(k_node.value, str):
                    continue
                val = _literal(v_node)
                if val is not None or isinstance(v_node, ast.Constant):
                    resolved[k_node.value] = val
        if resolved:
            for t in targets:
                constants[t.id] = resolved
    return constants


# ---------------------------------------------------------------------------
# Field mapping
# ---------------------------------------------------------------------------

# HF TrainingArguments → WorkloadConfig field name
_TRAINING_ARGS_MAP: dict[str, str] = {
    "per_device_train_batch_size": "batch_size",
    "gradient_accumulation_steps": "grad_accum_steps",
    "max_seq_length": "seq_len",
    "model_max_length": "seq_len",
    "optim": "optimizer",
    "gradient_checkpointing": "gradient_checkpointing",
    "torch_compile": "torch_compile",
    "learning_rate": "lr",
    "warmup_steps": "warmup_steps",
    "dataloader_num_workers": "dataloader_workers",
    "dataloader_pin_memory": "dataloader_pin_memory",
    "dataloader_prefetch_factor": "dataloader_prefetch_factor",
    "dataloader_persistent_workers": "dataloader_persistent_workers",
    "model_name_or_path": "model_name",
    "model_name": "model_name",
}

_DATALOADER_MAP: dict[str, str] = {
    "num_workers": "dataloader_workers",
    "pin_memory": "dataloader_pin_memory",
    "prefetch_factor": "dataloader_prefetch_factor",
    "persistent_workers": "dataloader_persistent_workers",
}

_VALID_PRECISIONS = {"fp32", "fp16", "bf16", "fp8"}
_VALID_ATTENTION = {"sdpa", "flash", "flash_rocm", "eager", "unknown"}


def _coerce_precision(args: dict[str, Any]) -> str | None:
    """Translate HF-style precision flags (bf16=True/fp16=True) into a Precision literal."""
    explicit = args.get("precision")
    if isinstance(explicit, str) and explicit in _VALID_PRECISIONS:
        return explicit
    if args.get("bf16") is True:
        return "bf16"
    if args.get("fp16") is True:
        return "fp16"
    if args.get("tf32") is True:
        return "fp32"
    return None


def _coerce_attention(args: dict[str, Any]) -> str | None:
    impl = args.get("attn_implementation") or args.get("attention_impl")
    if not isinstance(impl, str):
        return None
    impl = impl.lower()
    if impl in _VALID_ATTENTION:
        return impl
    if "flash" in impl and "rocm" in impl:
        return "flash_rocm"
    if "flash" in impl:
        return "flash"
    if impl == "eager":
        return "eager"
    if impl == "sdpa":
        return "sdpa"
    return None


def _apply_kwargs(
    payload: dict[str, Any],
    raw: dict[str, Any],
    extras: dict[str, Any],
    field_map: dict[str, str],
) -> None:
    """Push ``raw`` kwargs into ``payload`` using ``field_map``; unknown keys go to ``extras``."""
    for key, value in raw.items():
        if value is None:
            continue
        target = field_map.get(key)
        if target is None:
            extras[key] = value
            continue
        payload[target] = value


# ---------------------------------------------------------------------------
# Python AST extraction
# ---------------------------------------------------------------------------


def _extract_from_python(source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Walk the AST. Returns (mapped_fields, extras_dict).

    Two-pass: first pass collects module-level ``NAME = "literal"`` assignments
    so we can resolve identifiers passed positionally to ``from_pretrained``
    (the common ``MODEL_ID = "..."; from_pretrained(MODEL_ID)`` shape).
    """
    payload: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    env_vars: dict[str, str] = {}
    raw_training_args: dict[str, Any] = {}
    saw_torch_compile_call = False

    tree = ast.parse(source)

    # Pass 1: harvest top-level constants for identifier resolution.
    constants: dict[str, Any] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            val = _literal(stmt.value)
            if val is None:
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = val

    # Pass 1b: harvest dict-shaped constants so `**name` splats into Calls
    # can be resolved (e.g. `_ta = dict(...); TrainingArguments(**_ta)`).
    dict_constants = _collect_dict_constants(tree)

    def _arg_value(node: ast.AST) -> Any:
        """Literal eval, falling back to the constants table for bare Names."""
        val = _literal(node)
        if val is not None:
            return val
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        return None

    # Pass 2: walk every node, including nested calls, for tool/library calls.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            short = name.rsplit(".", 1)[-1] if name else ""

            if short in ("TrainingArguments", "Seq2SeqTrainingArguments"):
                kw = _kwargs_to_dict(node, dict_constants)
                raw_training_args.update(kw)

            elif short == "DataLoader":
                kw = _kwargs_to_dict(node, dict_constants)
                _apply_kwargs(payload, kw, extras, _DATALOADER_MAP)

            elif name == "torch.compile" or (
                short == "compile" and isinstance(node.func, ast.Attribute)
            ):
                # ``model = torch.compile(model, ...)`` is concrete evidence the
                # workload uses compile, regardless of any later TrainingArguments
                # kwarg. Record it as sticky-True.
                saw_torch_compile_call = True

            elif short == "gradient_checkpointing_enable":
                payload["gradient_checkpointing"] = True

            elif short == "from_pretrained":
                if node.args:
                    val = _arg_value(node.args[0])
                    if isinstance(val, str) and "model_name" not in payload:
                        payload["model_name"] = val
                kw = _kwargs_to_dict(node)
                attn = _coerce_attention(kw)
                if attn:
                    payload["attention_impl"] = attn

            elif short == "LoraConfig":
                kw = _kwargs_to_dict(node)
                if "r" in kw and isinstance(kw["r"], int):
                    payload["lora_rank"] = kw["r"]
                for k, v in kw.items():
                    if k != "r":
                        extras[f"lora.{k}"] = v

        elif isinstance(node, ast.Assign):
            # Catch ``os.environ["FOO"] = "bar"``.
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and _attr_chain(target.value) == "os.environ"
                ):
                    key = _literal(target.slice)
                    val = _literal(node.value)
                    if isinstance(key, str) and val is not None:
                        env_vars[key] = str(val)

    if raw_training_args:
        # precision/attention have to be derived after all kwargs are gathered.
        prec = _coerce_precision(raw_training_args)
        if prec:
            payload["precision"] = prec
        attn = _coerce_attention(raw_training_args)
        if attn:
            payload["attention_impl"] = attn
        _apply_kwargs(payload, raw_training_args, extras, _TRAINING_ARGS_MAP)
        # Drop keys we already consumed for precision/attention.
        for k in ("bf16", "fp16", "tf32", "attn_implementation"):
            extras.pop(k, None)

    # An explicit ``torch.compile(...)`` call wins over a False kwarg in
    # TrainingArguments — the call is concrete evidence; the kwarg is just a
    # default the user may have left at False.
    if saw_torch_compile_call:
        payload["torch_compile"] = True

    if env_vars:
        payload["env_vars"] = env_vars

    return payload, extras


# ---------------------------------------------------------------------------
# JSON / YAML extraction
# ---------------------------------------------------------------------------


def _extract_from_dict(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate a flat TrainingArguments-shaped dict into our schema."""
    payload: dict[str, Any] = {}
    extras: dict[str, Any] = {}

    prec = _coerce_precision(data)
    if prec:
        payload["precision"] = prec
    attn = _coerce_attention(data)
    if attn:
        payload["attention_impl"] = attn

    _apply_kwargs(payload, data, extras, _TRAINING_ARGS_MAP)

    # Pull DataLoader-shaped keys too, in case the config nests them.
    dl = data.get("dataloader") if isinstance(data.get("dataloader"), dict) else None
    if dl:
        _apply_kwargs(payload, dl, extras, _DATALOADER_MAP)
        extras.pop("dataloader", None)

    # env_vars block, if present.
    env = data.get("env_vars") or data.get("env")
    if isinstance(env, dict):
        payload["env_vars"] = {str(k): str(v) for k, v in env.items()}
        extras.pop("env_vars", None)
        extras.pop("env", None)

    # Same precision/attention scrub as the AST path.
    for k in ("bf16", "fp16", "tf32", "attn_implementation"):
        extras.pop(k, None)

    return payload, extras


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


def _build_config(
    payload: dict[str, Any],
    extras: dict[str, Any],
    raw_source: str,
    redactions: list[str],
) -> WorkloadConfig:
    """Construct a WorkloadConfig from the extracted payload + extras + redacted source."""
    # Schema requires model_name; fall back if neither the script nor the
    # config supplied one.
    payload.setdefault("model_name", "unknown")
    # Coerce types defensively — JSON/YAML can hand us strings where ints belong.
    int_fields = (
        "batch_size",
        "grad_accum_steps",
        "seq_len",
        "warmup_steps",
        "dataloader_workers",
        "lora_rank",
        "dataloader_prefetch_factor",
    )
    for field in int_fields:
        if field in payload and payload[field] is not None:
            try:
                payload[field] = int(payload[field])
            except (TypeError, ValueError):
                extras[field] = payload.pop(field)
    if "lr" in payload and payload["lr"] is not None:
        try:
            payload["lr"] = float(payload["lr"])
        except (TypeError, ValueError):
            extras["lr"] = payload.pop("lr")
    bool_fields = (
        "gradient_checkpointing",
        "torch_compile",
        "dataloader_pin_memory",
        "dataloader_persistent_workers",
    )
    for field in bool_fields:
        if field in payload and payload[field] is not None:
            payload[field] = bool(payload[field])

    payload["raw_source"] = raw_source
    payload["redactions"] = redactions
    payload["extras"] = extras
    return WorkloadConfig(**payload)


def _parse_config_full(file_path: str) -> WorkloadConfig | ToolResult:
    """Inner parser that returns the **complete** WorkloadConfig (including
    ``raw_source``). Tests use this to verify redaction. The public
    ``_parse_config`` wraps this and trims ``raw_source`` from the LLM-
    facing tool result.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return ToolResult(ok=False, error=f"File not found: {file_path}")

        raw = path.read_text(encoding="utf-8")
        clean, redactions = _redact(raw)
        suffix = path.suffix.lower()

        if suffix == ".py":
            payload, extras = _extract_from_python(raw)
        elif suffix == ".json":
            data = json.loads(raw)
            if not isinstance(data, dict):
                return ToolResult(
                    ok=False, error="JSON config must be an object at the top level."
                )
            payload, extras = _extract_from_dict(data)
        elif suffix in (".yaml", ".yml"):
            data = yaml.safe_load(raw)
            if not isinstance(data, dict):
                return ToolResult(
                    ok=False, error="YAML config must be a mapping at the top level."
                )
            payload, extras = _extract_from_dict(data)
        else:
            return ToolResult(
                ok=False,
                error=f"Unsupported config extension '{suffix}'. Expected .py/.json/.yaml/.yml.",
            )

        # Scrub any secret-shaped strings that survived in extras (mostly a
        # JSON/YAML concern — values come straight from the dict, not the
        # textual source, so the source-level redact pass missed them).
        extra_labels = _redact_extras(extras)
        for label in extra_labels:
            if label not in redactions:
                redactions.append(label)

        # env_vars come straight from the AST as literal strings — the
        # source-level redactor matched the *literal expression* in the source
        # but the value lives in our payload too, so scrub it again.
        env_vars = payload.get("env_vars")
        if isinstance(env_vars, dict):
            env_labels: list[str] = []
            cleaned_env: dict[str, str] = {}
            for k, v in env_vars.items():
                if isinstance(v, str):
                    cleaned, hits = _redact(v)
                    cleaned_env[k] = cleaned
                    env_labels.extend(hits)
                else:
                    cleaned_env[k] = v
            payload["env_vars"] = cleaned_env
            for label in env_labels:
                if label not in redactions:
                    redactions.append(label)

        return _build_config(payload, extras, clean, redactions)
    except SyntaxError as exc:
        return ToolResult(ok=False, error=f"Python parse error: {exc}")
    except json.JSONDecodeError as exc:
        return ToolResult(ok=False, error=f"JSON parse error: {exc}")
    except yaml.YAMLError as exc:
        return ToolResult(ok=False, error=f"YAML parse error: {exc}")
    except Exception as exc:
        return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")


def _parse_config(file_path: str) -> ToolResult:
    """Parse a training-config artefact into a WorkloadConfig and return it
    as a tool result.

    Detects format from the path suffix; ``.py`` runs through the AST visitor,
    ``.json`` and ``.yaml``/``.yml`` go through the dict path. Anything else is
    rejected up front rather than guessed.

    The returned dict **omits** ``raw_source`` to keep the LLM's audit
    conversation small enough to fit Qwen's context window. The redacted
    source still lives in ``WorkloadConfig.raw_source`` server-side; tests
    inspect it via ``_parse_config_full``.
    """
    cfg_or_err = _parse_config_full(file_path)
    if isinstance(cfg_or_err, ToolResult):
        return cfg_or_err  # error path
    return ToolResult(ok=True, result=cfg_or_err.model_dump(exclude={"raw_source"}))


PARSE_CONFIG = Tool(
    name="parse_config",
    description=(
        "Parse a user-uploaded training script or HF TrainingArguments JSON/YAML "
        "into a normalized WorkloadConfig. Redacts tokens, keys, and filesystem "
        "paths before returning."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the uploaded training script or config file.",
            }
        },
        "required": ["file_path"],
    },
    fn=_parse_config,
)
