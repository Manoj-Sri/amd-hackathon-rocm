#!/usr/bin/env python
"""Iterative auto-tuner for AMD MI300X / ROCm 7.0 workloads.

Two modes, picked with `--mode`:

  hardcoded (default)
    Walks through a curated list of MI300X-specific tuning changes one
    at a time. Deterministic, no LLM required — experiments are
    derived from the rules in kb/rocm_rules.yaml.

  llm
    On each iteration, asks the same LLM backend the agent uses
    (qwen-hf via HF_TOKEN, or qwen-vllm via GOBLIN_QWEN_VLLM_URL) for
    the next single experiment to try, given the live waste_budget,
    history of what's been tried, and the KB rules as context. The
    LLM's response is parsed as JSON with the same shape as the
    hardcoded Experiment dataclass.

After each change, runs a real benchmark via goblin_runner.sh and keeps
the change only if tokens/sec improved meaningfully (>1% by default —
the threshold cuts measurement noise). Stops when N consecutive
experiments produce no improvement, or when the source of experiments
is exhausted.

Usage:
    # hardcoded mode (default):
    python scripts/auto_tune.py workloads/train_qwen_lora.py --steps 20

    # LLM-driven mode:
    python scripts/auto_tune.py workloads/train_qwen_lora.py \\
        --mode llm --steps 20 --max-iterations 10

Output:
  - A row-by-row log of each experiment attempted, accepted or rejected
  - A final summary with cumulative speedup
  - A pointer to a temp file containing the best workload script for
    diff-against-baseline inspection

Extending hardcoded mode: add an Experiment to EXPERIMENTS. The
substitutions field is a list of (regex_pattern, replacement) tuples
applied with re.subn against the workload source. env_vars are exported
into the goblin_runner.sh subprocess and persist on every accepted
iteration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOBLIN_RUNNER = REPO_ROOT / "runner" / "goblin_runner.sh"
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Experiment:
    name: str
    description: str
    rationale: str
    substitutions: list[tuple[str, str]] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)


# Curated for ROCm 7.0 + MI300X (CDNA3, 192 GB HBM3). Ordered roughly by
# typical impact on Qwen-shaped LoRA fine-tuning workloads. Each
# experiment stacks on top of any previously accepted ones.
EXPERIMENTS: list[Experiment] = [
    Experiment(
        name="bf16_over_fp16",
        description="Switch precision from fp16 to bf16",
        rationale=(
            "MI300X (CDNA3) prefers bf16: same throughput, larger numeric "
            "range, no loss-scaler needed. fp16 underutilizes the matrix "
            "engine on this arch."
        ),
        substitutions=[
            (r"torch_dtype=torch\.float16", "torch_dtype=torch.bfloat16"),
            (r"\bfp16=True\b", "bf16=True"),
        ],
    ),
    Experiment(
        name="batch_size_8",
        description="Increase per_device_train_batch_size 4 → 8",
        rationale="MI300X has 192 GB HBM; batch=4 leaves it on the floor.",
        substitutions=[
            (r"per_device_train_batch_size=4\b", "per_device_train_batch_size=8"),
        ],
    ),
    Experiment(
        name="batch_size_16",
        description="Further increase per_device_train_batch_size to 16",
        rationale="If batch=8 fit and improved, try doubling again.",
        substitutions=[
            (r"per_device_train_batch_size=\d+", "per_device_train_batch_size=16"),
        ],
    ),
    Experiment(
        name="sdpa_attention",
        description="Switch attention from eager to SDPA",
        rationale=(
            "Eager attention is the slowest path. SDPA dispatches to the "
            "best available kernel (flash on ROCm 7.x where supported, "
            "memory-efficient elsewhere)."
        ),
        substitutions=[
            (r'attn_implementation="eager"', 'attn_implementation="sdpa"'),
        ],
    ),
    Experiment(
        name="dataloader_workers_4",
        description="Bump dataloader_num_workers 0 → 4",
        rationale=(
            "0 workers means the GPU sits idle while the host loads the "
            "next batch. 4 is a safe value across most CPU configs."
        ),
        substitutions=[
            (r"dataloader_num_workers=0", "dataloader_num_workers=4"),
            (r"num_workers=0", "num_workers=4"),
        ],
    ),
    Experiment(
        name="pin_memory",
        description="Enable dataloader_pin_memory",
        rationale=(
            "Pinned host buffers make H2D copies async and overlap with "
            "the GPU. Worth it once you have >0 dataloader workers."
        ),
        substitutions=[
            (r"dataloader_pin_memory=False", "dataloader_pin_memory=True"),
            (r"\bpin_memory=False\b", "pin_memory=True"),
        ],
    ),
    Experiment(
        name="env_hipblaslt",
        description="Set TORCH_BLAS_PREFER_HIPBLASLT=1",
        rationale=(
            "hipBLASLt is significantly faster than rocBLAS for the GEMM "
            "shapes Qwen produces (LoRA-projected attention)."
        ),
        env_vars={"TORCH_BLAS_PREFER_HIPBLASLT": "1"},
    ),
    Experiment(
        name="env_tunable_op",
        description="Set PYTORCH_TUNABLEOP_ENABLED=1",
        rationale=(
            "Enables runtime kernel auto-tuning. Pays a first-run "
            "warmup cost in exchange for a steady-state win on every "
            "subsequent step."
        ),
        env_vars={"PYTORCH_TUNABLEOP_ENABLED": "1"},
    ),
    Experiment(
        name="env_miopen_find",
        description="Set MIOPEN_FIND_MODE=3",
        rationale=(
            "MIOpen FAST mode picks already-tuned kernels without on-the-"
            "fly search. Reduces per-step variance."
        ),
        env_vars={"MIOPEN_FIND_MODE": "3"},
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def apply_substitutions(source: str, subs: list[tuple[str, str]]) -> str | None:
    """Apply each (pattern, replacement) in order. Returns the new source,
    or None if any pattern matched zero times (already applied or N/A for
    this workload)."""
    out = source
    for pattern, replacement in subs:
        new, n = re.subn(pattern, replacement, out)
        if n == 0:
            return None
        out = new
    return out


def benchmark(
    workload_path: Path,
    steps: int,
    env_overrides: dict[str, str],
    timeout: int = 600,
) -> dict | None:
    """Run goblin_runner.sh on the workload, return parsed RunMetrics dict
    or None on failure."""
    with tempfile.TemporaryDirectory(prefix="auto_tune_run_") as out_dir_str:
        out_dir = Path(out_dir_str)
        env = os.environ.copy()
        env["USER_SCRIPT"] = str(workload_path)
        env["OUT_DIR"] = str(out_dir)
        env["STEPS"] = str(steps)
        # Candidate workload lives in /tmp, so its self-bootstrap line
        # `sys.path.insert(0, dirname(dirname(__file__)))` resolves to /tmp
        # — which has no `workloads/` package. Inject the real repo root via
        # PYTHONPATH so `from workloads._runtime import ...` succeeds.
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(REPO_ROOT) + (os.pathsep + existing_pp if existing_pp else "")
        )
        env.update(env_overrides)

        try:
            proc = subprocess.run(
                [str(GOBLIN_RUNNER)],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {timeout}s")
            return None

        if proc.returncode != 0:
            print(f"  goblin_runner.sh failed (exit {proc.returncode})")
            tail = (proc.stderr or "").strip().splitlines()[-8:]
            for line in tail:
                print(f"    | {line}")
            return None

        try:
            from runner import profile_parser

            metrics = profile_parser.parse(out_dir, steps=steps)
            return metrics.model_dump()
        except Exception as exc:  # parser is defensive but be safe
            print(f"  profile_parser raised: {type(exc).__name__}: {exc}")
            return None


def _delta_pct(new: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return (new - baseline) / baseline * 100.0


# ---------------------------------------------------------------------------
# LLM-driven experiment generator
# ---------------------------------------------------------------------------


_LLM_SYSTEM_PROMPT = """\
You are an expert at tuning AMD MI300X (ROCm 7.0, CDNA3 arch, 192 GB
HBM3) training workloads. The user is iteratively benchmarking changes
to a transformers/peft fine-tuning script. On each turn you suggest ONE
specific parameter change to try next, targeting the largest non-useful
waste bucket in the most recent benchmark.

Your output MUST be a single JSON object with this exact shape (no
prose, no markdown fences, just the object):

{
  "name": "short_snake_case_name",
  "rationale": "1-3 sentences on why this change addresses the worst waste bucket",
  "substitutions": [["regex_pattern", "replacement"]],
  "env_vars": {"VAR_NAME": "value"}
}

CRITICAL output rules — read carefully:

1. env_vars keys are LITERAL shell environment variable names. NEVER
   prefix them with "env_vars." or any other dotted path. The KB rules
   shown to you use dotted paths like `env_vars.MIOPEN_FIND_MODE`
   because they describe a config dict — you must STRIP that prefix and
   use just `MIOPEN_FIND_MODE` as the env var key.
   Wrong:  {"env_vars.MIOPEN_FIND_MODE": "3"}
   Right:  {"MIOPEN_FIND_MODE": "3"}

2. substitutions are (regex_pattern, replacement) pairs applied with
   re.subn against the current workload source. Patterns must match at
   least one occurrence in the source — if zero matches, the experiment
   is auto-skipped (counted as no improvement).

3. When the previous change for a parameter improved tokens/sec, push
   that parameter further in the same direction next time. E.g. if
   batch_size 4 → 8 won, try 8 → 16. If 16 won and HBM is still under
   ~150 GB, try 32. Don't be timid — MI300X has 192 GB HBM3.

4. Don't repeat any (name OR substitution OR env_var combo) from
   history. If a change was rejected, don't propose the same numerical
   value again — try a different one.

5. If you cannot think of a productive next change, output:
     {"name": "STOP", "rationale": "<why>", "substitutions": [], "env_vars": {}}

CONCRETE OUTPUT EXAMPLES — match this shape exactly:

Switch fp16 → bf16 (precision_path bucket):
  {"name": "bf16_over_fp16",
   "rationale": "MI300X CDNA3 matrix cores prefer bf16: same throughput, larger numeric range, no loss-scaler.",
   "substitutions": [["fp16=True", "bf16=True"], ["torch_dtype=torch\\\\.float16", "torch_dtype=torch.bfloat16"]],
   "env_vars": {}}

Increase batch size to 16 (memory_headroom bucket):
  {"name": "batch_size_16",
   "rationale": "Current HBM peak is well under 192 GB; bigger batch saturates the GPU.",
   "substitutions": [["per_device_train_batch_size=\\\\d+", "per_device_train_batch_size=16"]],
   "env_vars": {}}

Switch attention to SDPA (kernel_shape bucket):
  {"name": "sdpa_attention",
   "rationale": "Eager attention is the slowest path; SDPA dispatches to a tuned kernel.",
   "substitutions": [["attn_implementation=\\"eager\\"", "attn_implementation=\\"sdpa\\""]],
   "env_vars": {}}

Bump dataloader workers (data_wait bucket):
  {"name": "dataloader_workers_4",
   "rationale": "0 workers starves the GPU between batches.",
   "substitutions": [["dataloader_num_workers=0", "dataloader_num_workers=4"]],
   "env_vars": {}}

Set MIOpen FAST mode (kernel_shape bucket, env-only):
  {"name": "miopen_find_fast",
   "rationale": "FAST mode picks already-tuned kernels without on-the-fly search.",
   "substitutions": [],
   "env_vars": {"MIOPEN_FIND_MODE": "3"}}

Prefer hipBLASLt (kernel_shape bucket, env-only):
  {"name": "prefer_hipblaslt",
   "rationale": "hipBLASLt is faster than rocBLAS for Qwen GEMM shapes on MI300X.",
   "substitutions": [],
   "env_vars": {"TORCH_BLAS_PREFER_HIPBLASLT": "1"}}
"""


_LLM_USER_TEMPLATE = """\
KB rules (one-liner per rule, for grounding):
{kb_summary}

Tunable parameters detected in the current workload — current literal
values, and the regex pattern you'd use to change each:
{tunables}

Latest benchmark:
- tokens_per_sec: {tps:.1f}
- gpu_util_pct:   {util:.1f}
- hbm_peak_gb:    {hbm:.2f}
- waste_budget (seconds/step):
{waste_lines}

Sorted recoverable waste (largest first — go after these):
{recoverable_sorted}

History of changes already tried this run (newest first):
{history_lines}

Suggest ONE next change targeting the largest recoverable bucket. JSON only.
"""


def _kb_summary(rules_yaml_path: Path, max_chars: int = 6000) -> str:
    """Return a compact one-line-per-rule summary of kb/rocm_rules.yaml.

    Notably we DO NOT show the raw `transform` field — earlier versions
    did and the LLM ended up copying its dotted-path notation literally
    (`env_vars.MIOPEN_FIND_MODE` as the env var name, not as a dict
    accessor). The system prompt's CONCRETE EXAMPLES section is the
    canonical source of truth for output shape; this summary just
    grounds the LLM's reasoning in the catalog of known issues.
    """
    if not rules_yaml_path.exists():
        return "(KB rules file not found)"
    try:
        import yaml

        rules = yaml.safe_load(rules_yaml_path.read_text()) or []
    except Exception as exc:
        return f"(failed to parse KB: {exc})"

    lines = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        rid = r.get("id", "?")
        bucket = r.get("targets_bucket", "?")
        sym = (r.get("symptom") or "").strip().replace("\n", " ")
        if len(sym) > 110:
            sym = sym[:107] + "..."
        lines.append(f"- {rid:55s} [{bucket}]  {sym}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


# Map of (substring-in-source) → (parameter description, example regex
# pattern, example replacement template). Each entry is a hint shown to
# the LLM so it has a concrete target to point its substitutions at —
# instead of guessing what the workload's literal config text looks like.
_TUNABLE_HINTS: list[tuple[str, str, str, str]] = [
    # (token to detect, description, regex_for_substitution, replacement_template)
    ("torch_dtype=torch.float16",
     "model precision (matches `torch_dtype=torch.float16`)",
     r"torch_dtype=torch\.float16",
     "torch_dtype=torch.bfloat16"),
    ("torch_dtype=torch.bfloat16",
     "model precision (already bf16)",
     r"torch_dtype=torch\.bfloat16",
     "torch_dtype=torch.float16"),
    ("fp16=True",
     "TrainingArguments fp16 (matches `fp16=True`)",
     r"\bfp16=True\b",
     "bf16=True"),
    ("bf16=True",
     "TrainingArguments bf16 (already bf16)",
     r"\bbf16=True\b",
     "fp16=True"),
    ("attn_implementation=\"eager\"",
     "attention impl (matches `attn_implementation=\"eager\"`)",
     r'attn_implementation="eager"',
     'attn_implementation="sdpa"'),
    ("attn_implementation=\"sdpa\"",
     "attention impl (currently sdpa; could try flash_attention_2)",
     r'attn_implementation="sdpa"',
     'attn_implementation="flash_attention_2"'),
    ("per_device_train_batch_size=",
     "per-device batch size (matches `per_device_train_batch_size=<N>`)",
     r"per_device_train_batch_size=\d+",
     "per_device_train_batch_size=<NEW_VALUE>"),
    ("dataloader_num_workers=",
     "dataloader workers (matches `dataloader_num_workers=<N>`)",
     r"dataloader_num_workers=\d+",
     "dataloader_num_workers=<NEW_VALUE>"),
    ("dataloader_pin_memory=",
     "dataloader pin_memory (matches `dataloader_pin_memory=<bool>`)",
     r"dataloader_pin_memory=(True|False)",
     "dataloader_pin_memory=True"),
    ("gradient_checkpointing=",
     "gradient checkpointing toggle",
     r"gradient_checkpointing=(True|False)",
     "gradient_checkpointing=True"),
    ("torch_compile=",
     "torch.compile toggle (use cautiously on ROCm 7.x)",
     r"torch_compile=(True|False)",
     "torch_compile=True"),
    ("optim=\"adamw_torch\"",
     "optimizer choice (currently adamw_torch)",
     r'optim="adamw_torch"',
     'optim="adamw_torch_fused"'),
]


def _tunables_summary(source: str) -> str:
    """Detect which tunable parameters are present in the workload source
    and surface their current literal values + ready-to-use regex patterns
    so the LLM has concrete substitution targets.

    Skips comment lines when reporting the "current" value — many workloads
    document expected findings in a top-of-file comment block, and we want
    the LLM to see the live config line, not the doc string.
    """
    lines: list[str] = []
    source_lines = source.splitlines()
    for token, desc, pattern, replacement in _TUNABLE_HINTS:
        live_line: str | None = None
        for raw in source_lines:
            stripped = raw.lstrip()
            if stripped.startswith("#"):
                continue
            if token in raw:
                live_line = raw.strip()
                break
        if live_line is None:
            continue
        lines.append(
            f"  • {desc}\n"
            f"    current: {live_line}\n"
            f"    pattern: {pattern!r}    replacement template: {replacement!r}"
        )
    if not lines:
        return "  (no recognized tunables — substitutions will need to match other text)"
    return "\n".join(lines)


def _recoverable_sorted(waste: dict) -> str:
    """List the non-useful_gpu buckets sorted by size, so the LLM can
    explicitly target the biggest one first."""
    if not waste:
        return "  (no waste_budget available)"
    items = [
        (name, value)
        for name, value in waste.items()
        if name != "useful_gpu" and isinstance(value, (int, float))
    ]
    items.sort(key=lambda kv: kv[1], reverse=True)
    if not items:
        return "  (no recoverable buckets)"
    return "\n".join(f"  {i + 1}. {name:18s} = {value:.4f}" for i, (name, value) in enumerate(items))


def _config_snippet(source: str, max_lines: int = 80) -> str:
    """Return the lines around `TrainingArguments(` and `from_pretrained(` so
    the LLM sees the actual config it's modifying without us shipping the
    whole script. Gives ~max_lines of context.
    """
    lines = source.splitlines()
    keep: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(
            tok in lower
            for tok in (
                "trainingarguments(",
                "from_pretrained(",
                "loraconfig(",
                "dataloader(",
                "torch_dtype",
                "attn_implementation",
                "fp16=",
                "bf16=",
                "per_device_train_batch_size",
                "dataloader_num_workers",
                "dataloader_pin_memory",
                "gradient_checkpointing",
                "torch_compile",
                "optim=",
            )
        ):
            keep.append((i, line))
    if not keep:
        return source[:2000]
    # Coalesce nearby line indices into windows for readability
    windows: list[list[str]] = []
    last_idx = -10
    cur: list[str] = []
    for i, line in keep:
        if i - last_idx > 3:
            if cur:
                windows.append(cur)
            cur = []
        cur.append(f"{i + 1:4d}: {line}")
        last_idx = i
    if cur:
        windows.append(cur)
    out = "\n\n".join("\n".join(w) for w in windows)
    if out.count("\n") > max_lines:
        out_lines = out.splitlines()[:max_lines]
        out = "\n".join(out_lines) + "\n... (truncated)"
    return out


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none yet — this is the first iteration)"
    lines = []
    for h in reversed(history[-12:]):  # last 12 newest-first
        outcome = h.get("outcome", "?")
        delta = h.get("delta_pct")
        delta_s = f"{delta:+.2f}%" if delta is not None else "n/a"
        subs = h.get("substitutions") or []
        envs = h.get("env_vars") or {}
        change_repr = f"subs={subs} env={envs}"
        lines.append(f"- {h['name']:25s} {outcome:9s} Δ {delta_s:8s}  {change_repr}")
    return "\n".join(lines)


def _format_waste(waste: dict) -> str:
    keys = (
        "useful_gpu",
        "data_wait",
        "host_gap",
        "comm_excess",
        "memory_headroom",
        "precision_path",
        "kernel_shape",
    )
    return "\n".join(f"    {k:18s} = {waste.get(k, 0.0):.4f}" for k in keys)


def _build_llm_backend():
    """Construct the same backend the agent loop uses. Surfaces a clear
    message if neither HF_TOKEN nor a vLLM URL is configured."""
    has_hf = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    has_vllm = bool(os.environ.get("GOBLIN_QWEN_VLLM_URL"))
    backend_kind = os.environ.get("GOBLIN_AGENT_BACKEND", "qwen-hf").lower()
    if backend_kind in ("qwen-hf", "qwen", "hf", "") and not has_hf:
        raise SystemExit(
            "LLM mode requires HF_TOKEN (qwen-hf backend) or "
            "GOBLIN_AGENT_BACKEND=qwen-vllm + GOBLIN_QWEN_VLLM_URL."
        )
    if backend_kind in ("qwen-vllm", "qwen_vllm", "vllm", "local") and not has_vllm:
        raise SystemExit(
            "LLM mode with qwen-vllm backend requires GOBLIN_QWEN_VLLM_URL."
        )
    from agent.backends import make_backend

    return make_backend(system_prompt=_LLM_SYSTEM_PROMPT, max_tokens=1024)


async def _ask_llm_for_experiment(
    backend,
    *,
    kb_summary: str,
    source: str,
    metrics: dict,
    history: list[dict],
) -> Experiment | None:
    """One LLM turn → one Experiment (or None for STOP / parse failure)."""
    waste = metrics.get("waste_budget") or {}
    prompt = _LLM_USER_TEMPLATE.format(
        kb_summary=kb_summary,
        tunables=_tunables_summary(source),
        tps=metrics.get("tokens_per_sec", 0.0),
        util=metrics.get("gpu_util_pct", 0.0),
        hbm=metrics.get("hbm_peak_gb", 0.0),
        waste_lines=_format_waste(waste),
        recoverable_sorted=_recoverable_sorted(waste),
        history_lines=_format_history(history),
    )
    backend.add_user_message(prompt)
    turn = await backend.next_turn(tool_schemas=[])
    raw = " ".join(turn.text_blocks).strip()

    obj = _extract_json_object(raw)
    if obj is None:
        print(f"  LLM response was not parseable JSON. Raw: {raw[:300]!r}")
        return None

    name = (obj.get("name") or "").strip()
    if not name or name.upper() == "STOP":
        print(f"  LLM signaled STOP: {obj.get('rationale', '(no rationale)')}")
        return None

    subs_raw = obj.get("substitutions") or []
    envs = obj.get("env_vars") or {}
    if not subs_raw and not envs:
        print(f"  LLM returned an empty experiment ({name}); skipping")
        return None

    subs: list[tuple[str, str]] = []
    for entry in subs_raw:
        if isinstance(entry, list) and len(entry) == 2:
            subs.append((str(entry[0]), str(entry[1])))
        elif isinstance(entry, dict) and "pattern" in entry and "replacement" in entry:
            subs.append((str(entry["pattern"]), str(entry["replacement"])))

    # Defensive: strip dotted-path prefixes from env var keys (e.g. the LLM
    # emits `env_vars.MIOPEN_FIND_MODE` because it's mimicking the KB
    # transform shape). The actual env var name is always the last segment.
    cleaned_envs: dict[str, str] = {}
    for k, v in envs.items():
        key = str(k)
        if "." in key:
            stripped = key.rsplit(".", 1)[-1]
            print(f"  [warn] LLM emitted dotted env key {key!r}; using {stripped!r} instead")
            key = stripped
        cleaned_envs[key] = str(v)

    return Experiment(
        name=name,
        description=obj.get("description") or name,
        rationale=str(obj.get("rationale") or ""),
        substitutions=subs,
        env_vars=cleaned_envs,
    )


def _extract_json_object(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM response, tolerating
    markdown fences / leading prose."""
    if not text:
        return None
    # strip ```json ... ``` fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    # otherwise grab the first balanced { ... }
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blob = text[start : i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    start = -1
                    continue
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("workload", type=Path, help="Path to workload script")
    p.add_argument(
        "--mode",
        choices=("hardcoded", "llm"),
        default="hardcoded",
        help=(
            "hardcoded (default): walk through the priority-ordered EXPERIMENTS list. "
            "llm: ask the agent's configured LLM backend (qwen-hf or qwen-vllm) for "
            "the next experiment based on the live waste_budget + history."
        ),
    )
    p.add_argument("--steps", type=int, default=20, help="Steps per benchmark")
    p.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help=(
            "Cap on experiments to try. Default: len(EXPERIMENTS) for hardcoded mode, "
            "10 for llm mode."
        ),
    )
    p.add_argument(
        "--early-stop-after",
        type=int,
        default=3,
        help="Stop after N consecutive non-improvements",
    )
    p.add_argument(
        "--improvement-threshold",
        type=float,
        default=1.0,
        help="Min %% improvement over current best to accept (default 1.0)",
    )
    args = p.parse_args()
    if args.max_iterations <= 0:
        args.max_iterations = len(EXPERIMENTS) if args.mode == "hardcoded" else 10

    workload = args.workload.resolve()
    if not workload.exists():
        sys.stderr.write(f"workload not found: {workload}\n")
        return 1
    if not GOBLIN_RUNNER.exists():
        sys.stderr.write(f"goblin_runner.sh not found at {GOBLIN_RUNNER}\n")
        return 1

    workspace = Path(tempfile.mkdtemp(prefix="auto_tune_workloads_"))
    print(f"Auto-tune workspace: {workspace}")
    print(f"Mode:                {args.mode}")
    print(f"Workload:            {workload}")
    print(f"Steps per benchmark: {args.steps}")
    print(f"Max iterations:      {args.max_iterations}")
    print(f"Early stop after:    {args.early_stop_after} non-improvements")
    print(f"Accept threshold:    {args.improvement_threshold:.1f}%\n")

    # LLM mode setup happens before the baseline so we fail fast on missing
    # credentials rather than after burning a baseline benchmark.
    llm_backend = None
    kb_summary = ""
    if args.mode == "llm":
        llm_backend = _build_llm_backend()
        kb_summary = _kb_summary(REPO_ROOT / "kb" / "rocm_rules.yaml")
        print("LLM backend ready. KB summary loaded.\n")

    baseline_source = workload.read_text()
    baseline_path = workspace / "00_baseline.py"
    baseline_path.write_text(baseline_source)

    print("=" * 60)
    print("Baseline benchmark")
    print("=" * 60)
    baseline = benchmark(baseline_path, args.steps, {})
    if baseline is None:
        sys.stderr.write("Baseline benchmark failed; cannot continue.\n")
        return 1

    baseline_tps = baseline["tokens_per_sec"]
    print(f"  tokens/sec:    {baseline_tps:.1f}")
    print(f"  hbm_peak_gb:   {baseline['hbm_peak_gb']:.2f}")
    print(f"  gpu_util_pct:  {baseline['gpu_util_pct']:.1f}")
    print(
        "  waste_budget:  "
        + ", ".join(f"{k}={v:.3f}" for k, v in baseline["waste_budget"].items() if v > 0)
    )

    best_source = baseline_source
    best_tps = baseline_tps
    best_env: dict[str, str] = {}
    last_metrics = baseline
    accepted: list[tuple[str, float, float]] = []  # (name, tps, delta_pct)
    rejected: list[tuple[str, str]] = []  # (name, reason)
    history: list[dict] = []  # for LLM context
    consecutive_no_improvement = 0

    for i in range(args.max_iterations):
        # ---- Get next experiment (mode-dependent) ----
        if args.mode == "hardcoded":
            if i >= len(EXPERIMENTS):
                print("\nReached end of EXPERIMENTS list.")
                break
            exp = EXPERIMENTS[i]
        else:  # llm
            print(f"\n[asking LLM for next experiment, iteration {i + 1}...]")
            try:
                exp = asyncio.run(
                    _ask_llm_for_experiment(
                        llm_backend,
                        kb_summary=kb_summary,
                        source=best_source,
                        metrics=last_metrics,
                        history=history,
                    )
                )
            except Exception as exc:
                print(f"  LLM call failed: {type(exc).__name__}: {exc}")
                exp = None
            if exp is None:
                print("LLM produced no experiment — stopping.")
                break

        print()
        print("=" * 60)
        print(f"Iteration {i + 1}: {exp.name}")
        print("=" * 60)
        print(f"  description: {exp.description}")
        print(f"  rationale:   {exp.rationale}")

        # ---- Build candidate source ----
        if exp.substitutions:
            try:
                candidate_source = apply_substitutions(best_source, exp.substitutions)
            except re.error as exc:
                print(f"  SKIPPED — invalid regex from LLM: {exc}")
                rejected.append((exp.name, f"bad regex: {exc}"))
                history.append({
                    "name": exp.name, "outcome": "rejected",
                    "delta_pct": None,
                    "substitutions": exp.substitutions, "env_vars": exp.env_vars,
                })
                consecutive_no_improvement += 1
                if consecutive_no_improvement >= args.early_stop_after:
                    print(f"\nNo improvement for {args.early_stop_after} consecutive iterations — early stopping.")
                    break
                continue
            if candidate_source is None:
                print("  SKIPPED — substitution patterns didn't match (already applied or N/A)")
                rejected.append((exp.name, "patterns didn't match"))
                history.append({
                    "name": exp.name, "outcome": "skipped",
                    "delta_pct": None,
                    "substitutions": exp.substitutions, "env_vars": exp.env_vars,
                })
                consecutive_no_improvement += 1
                if consecutive_no_improvement >= args.early_stop_after:
                    print(f"\nNo improvement for {args.early_stop_after} consecutive iterations — early stopping.")
                    break
                continue
        else:
            candidate_source = best_source

        # Slugify name for the filename
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", exp.name)[:40] or "exp"
        candidate_path = workspace / f"{i + 1:02d}_{safe_name}.py"
        candidate_path.write_text(candidate_source)
        candidate_env = {**best_env, **exp.env_vars}
        if exp.env_vars:
            print(f"  env vars:    {exp.env_vars}")

        m = benchmark(candidate_path, args.steps, candidate_env)
        if m is None:
            rejected.append((exp.name, "benchmark failed"))
            history.append({
                "name": exp.name, "outcome": "failed",
                "delta_pct": None,
                "substitutions": exp.substitutions, "env_vars": exp.env_vars,
            })
            consecutive_no_improvement += 1
        else:
            tps = m["tokens_per_sec"]
            delta = _delta_pct(tps, best_tps)
            print(f"  tokens/sec:  {tps:.1f}  (Δ {delta:+.2f}% vs current best)")
            print(f"  hbm_peak_gb: {m['hbm_peak_gb']:.2f}")
            print(f"  gpu_util_pct:{m['gpu_util_pct']:.1f}")

            if delta >= args.improvement_threshold:
                print(f"  ACCEPTED — {exp.name} is the new baseline")
                best_source = candidate_source
                best_tps = tps
                best_env = candidate_env
                last_metrics = m
                accepted.append((exp.name, tps, delta))
                history.append({
                    "name": exp.name, "outcome": "accepted",
                    "delta_pct": delta,
                    "substitutions": exp.substitutions, "env_vars": exp.env_vars,
                })
                consecutive_no_improvement = 0
            else:
                print("  REJECTED — improvement below threshold")
                rejected.append((exp.name, f"{delta:+.2f}%"))
                history.append({
                    "name": exp.name, "outcome": "rejected",
                    "delta_pct": delta,
                    "substitutions": exp.substitutions, "env_vars": exp.env_vars,
                })
                # In LLM mode the latest metrics are still useful context
                # even on a rejected experiment — let the LLM see what
                # happened.
                if args.mode == "llm":
                    last_metrics = m
                consecutive_no_improvement += 1

        if consecutive_no_improvement >= args.early_stop_after:
            print(
                f"\nNo improvement for {args.early_stop_after} consecutive iterations — early stopping."
            )
            break

    # Save best
    best_path = workspace / "best.py"
    best_path.write_text(best_source)

    # Summary
    print()
    print("=" * 60)
    print("AUTO-TUNE SUMMARY")
    print("=" * 60)
    print(f"Baseline tokens/sec: {baseline_tps:.1f}")
    print(
        f"Best tokens/sec:     {best_tps:.1f}  "
        f"({_delta_pct(best_tps, baseline_tps):+.2f}% vs baseline)"
    )
    print()
    print(f"Accepted ({len(accepted)}):")
    for name, tps, delta in accepted:
        print(f"  + {name:25s}  {tps:8.1f} tok/s  (Δ {delta:+.2f}%)")
    print()
    print(f"Rejected ({len(rejected)}):")
    for name, reason in rejected:
        print(f"  - {name:25s}  {reason}")
    print()

    if best_env:
        print("Required env vars for best config:")
        for k, v in best_env.items():
            print(f"  export {k}={v}")
        print()

    print(f"Best workload script:  {best_path}")
    print(f"Diff vs baseline:      diff {workload} {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
