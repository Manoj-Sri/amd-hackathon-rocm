#!/usr/bin/env python
"""Iterative auto-tuner for AMD MI300X / ROCm 7.0 workloads.

Three modes, picked with `--mode`:

  hardcoded (default)
    Walks through a curated list of MI300X-specific tuning changes one
    at a time. Deterministic, no LLM required — experiments are
    derived from the rules in kb/rocm_rules.yaml.

  llm
    On each iteration, asks the LLM backend (qwen-hf via HF_TOKEN, or
    qwen-vllm via GOBLIN_QWEN_VLLM_URL) for ONE next experiment given
    the live waste_budget, history, and KB rules. Greedy coordinate
    descent — accept changes that beat the current best by the
    improvement threshold, otherwise revert.

  llm-explore
    On each iteration, asks the LLM for K candidate experiments at
    once (--candidates-per-iteration, default 3). Benchmarks all K,
    picks the one with the highest tokens/sec, and accepts only if it
    beats the current best. Higher GPU cost (~Kx benchmarks per
    iteration) but better at finding interaction effects that greedy
    one-at-a-time can miss.

After each change, runs a real benchmark via goblin_runner.sh and keeps
the change only if tokens/sec improved meaningfully (>1% by default —
the threshold cuts measurement noise). Stops when N consecutive
experiments produce no improvement, or when the source of experiments
is exhausted.

Usage:
    # hardcoded mode (default):
    python scripts/auto_tune.py workloads/train_qwen_lora.py --steps 20

    # LLM-driven greedy mode:
    python scripts/auto_tune.py workloads/train_qwen_lora.py \\
        --mode llm --steps 20

    # LLM-driven multi-candidate exploration:
    python scripts/auto_tune.py workloads/train_qwen_lora.py \\
        --mode llm-explore --candidates-per-iteration 3 --steps 20

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

# Default workload template — used when the user passes --model instead
# of an explicit workload path. We just substitute MODEL_ID and reuse all
# the other defaults (fp16, batch=4, eager attention, LoRA r=16, …).
_DEFAULT_WORKLOAD_TEMPLATE = REPO_ROOT / "workloads" / "train_qwen_lora.py"


def _generate_workload_from_model(model_id: str, dest: Path) -> Path:
    """Build a baseline workload by substituting MODEL_ID into the demo
    template (`workloads/train_qwen_lora.py`). Writes to `dest`, returns
    the path.

    Caveats:
    - Uses the demo's LoRA target_modules (`q_proj`, `v_proj`) which work
      for the major decoder-only LLM families (Qwen, Llama, Mistral,
      Gemma). MoE / GPT-2-style architectures will need a custom workload.
    - The template overwrites HF_TOKEN with a redactable fake. Public
      models load fine; gated models (Llama, etc.) need the user to edit
      the generated workload or use a custom one.
    """
    if not _DEFAULT_WORKLOAD_TEMPLATE.exists():
        raise SystemExit(
            f"--model needs the template at {_DEFAULT_WORKLOAD_TEMPLATE}, but it's missing"
        )
    template_src = _DEFAULT_WORKLOAD_TEMPLATE.read_text()
    new_src, n = re.subn(
        r'MODEL_ID = "[^"]*"',
        f'MODEL_ID = "{model_id}"',
        template_src,
    )
    if n == 0:
        raise SystemExit(
            f"Couldn't find `MODEL_ID = \"...\"` in {_DEFAULT_WORKLOAD_TEMPLATE} "
            "to substitute. Has the template format changed?"
        )
    dest.write_text(new_src)
    return dest


# POSIX env var name: leading letter or underscore, then alnum/underscore.
# subprocess.run() raises ValueError if any key in the env dict violates
# this. We validate up-front rather than letting the subprocess crash.
_VALID_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _sanitize_env_vars(envs: dict, context: str = "") -> dict[str, str]:
    """Clean an env_vars dict from the LLM:
      1. Strip dotted prefixes (`env_vars.X` → `X`) the LLM mimics from the
         KB transform notation.
      2. Drop any key that still isn't a valid POSIX env var name. Warns
         instead of crashing — the LLM occasionally embeds shell syntax
         (e.g. `'NUMACTL_INTERLEAVE=1'` as a key) which would make
         subprocess.run raise ValueError.
    """
    cleaned: dict[str, str] = {}
    for k, v in envs.items():
        key = str(k)
        if "." in key:
            stripped = key.rsplit(".", 1)[-1]
            tag = f" [{context}]" if context else ""
            print(f"  [warn]{tag} dotted env key {key!r}; using {stripped!r}")
            key = stripped
        if not _VALID_ENV_NAME.match(key):
            tag = f" [{context}]" if context else ""
            print(
                f"  [warn]{tag} dropping invalid env var name {key!r} "
                "(must match [A-Za-z_][A-Za-z0-9_]*)"
            )
            continue
        cleaned[key] = str(v)
    return cleaned


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
        name="batch_size_32",
        description="Push per_device_train_batch_size to 32",
        rationale=(
            "MI300X has 192 GB HBM3 — batch 16 typically peaks ~130 GB. "
            "If 16 fit, 32 likely fits too and reduces step overhead per "
            "token. Reverts cleanly via OOM-as-crash if not."
        ),
        substitutions=[
            (r"per_device_train_batch_size=\d+", "per_device_train_batch_size=32"),
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
        except ValueError as exc:
            # subprocess.run validates env var names and raises ValueError
            # for malformed keys (e.g. names containing '=' or spaces). The
            # LLM has occasionally emitted those; we sanitize earlier but
            # this is the last-resort backstop so a single bad candidate
            # doesn't crash the whole tuning run.
            print(f"  REJECTED — illegal env var name(s): {exc}")
            print(f"  env keys offered: {list(env_overrides.keys())}")
            return None
        except OSError as exc:
            print(f"  REJECTED — could not spawn goblin_runner.sh: {exc}")
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

1. env_vars keys are LITERAL POSIX shell environment variable names.
   They MUST match the regex [A-Za-z_][A-Za-z0-9_]* — letters, digits,
   underscores only, starting with a letter or underscore.
   - NEVER prefix them with "env_vars." or any other dotted path.
   - NEVER include "=" or shell syntax in the key — env var names are
     identifiers, NOT assignments and NOT commands.
   - If you want to invoke a command-line tool like `numactl` or
     `taskset`, that CANNOT be expressed as an env_var. Don't try.
     Either propose a `substitutions` change to the script, or skip.
   Wrong:  {"env_vars.MIOPEN_FIND_MODE": "3"}
   Wrong:  {"NUMACTL_INTERLEAVE=1": "numactl --interleave=all"}
   Wrong:  {"export FOO": "bar"}
   Right:  {"MIOPEN_FIND_MODE": "3"}
   Right:  {"TORCH_BLAS_PREFER_HIPBLASLT": "1"}

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
Hardware facts (use these — do not contradict):
- AMD MI300X, CDNA3 architecture, 192 GB HBM3
- bf16 throughput on CDNA3 ≈ same as fp16, > fp32 (matrix engine is fp16/bf16/fp8 native)
- fp32 is the SLOWEST option on this arch — never suggest it as an improvement

Known incompatibilities for THIS workload (peft + LoRA on transformers Trainer):
{incompatibilities}

KB rules (one-liner per rule, for grounding):
{kb_summary}

Current accepted workload state — these are the literal values in the
script after every change accepted so far. The next change you propose
should mutate one of these (or set an env var). DO NOT propose a value
that's already present here.
{tunables}

Latest benchmark (this is the result of the most recent ACCEPTED state):
- tokens_per_sec: {tps:.1f}
- mfu_pct:        {mfu:.2f}   (% of MI300X dense bf16 peak; healthy LoRA ranges 30-50%)
- gpu_util_pct:   {util:.1f}
- hbm_peak_gb:    {hbm:.2f}
- waste_budget (seconds/step):
{waste_lines}

Sorted recoverable waste (largest first — go after these):
{recoverable_sorted}

History of changes already tried this run (newest first; outcomes are
"accepted" / "rejected" / "crashed" / "skipped"):
{history_lines}

If the latest entry is "crashed", the change you propose next must be
STRUCTURALLY different (different parameter, not just a different value
of the same one).

Suggest ONE next change targeting the largest recoverable bucket. JSON only.
"""


# Workload-specific incompatibilities the LLM otherwise wastes iterations on.
# Keep this list short and concrete — it goes into every prompt.
_KNOWN_INCOMPATIBILITIES = [
    "gradient_checkpointing=True requires `model.enable_input_require_grads()`"
    " before peft wrapping for LoRA models. Setting it via a single substitution"
    " WILL CRASH the workload. Don't propose it.",
    "bitsandbytes-based optimizers (`adamw_8bit`, `paged_adamw_8bit`) and"
    " `load_in_8bit=True` are NOT supported on ROCm 7.x. Don't propose them.",
    "torch_compile=True with peft/LoRA on ROCm 7.x triggers compile-time"
    " errors with the current PyTorch nightly (2.9.x). Don't propose it"
    " unless you have specific evidence it works on this version.",
    "flash_attention_2 may not be installed (try `attn_implementation=\"sdpa\"`"
    " before `\"flash_attention_2\"`).",
    "persistent_workers=True requires num_workers > 0. PyTorch raises"
    " `ValueError: persistent_workers option needs num_workers > 0` if you"
    " enable it while num_workers=0. If the current workload has"
    " dataloader_num_workers=0, do NOT propose persistent_workers=True"
    " alone — pair it with `dataloader_num_workers=4` (or higher) in the"
    " SAME experiment via two substitutions, or wait until a previous"
    " experiment has bumped num_workers above 0.",
    "dataloader_prefetch_factor only works when num_workers > 0 (same"
    " constraint as persistent_workers). Same rule: bump num_workers in"
    " the same experiment, or skip.",
]


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


def _build_llm_backend(system_prompt: str = _LLM_SYSTEM_PROMPT, max_tokens: int = 1024):
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

    return make_backend(system_prompt=system_prompt, max_tokens=max_tokens)


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
        incompatibilities="\n".join(f"- {line}" for line in _KNOWN_INCOMPATIBILITIES),
        kb_summary=kb_summary,
        tunables=_tunables_summary(source),
        tps=metrics.get("tokens_per_sec", 0.0),
        mfu=metrics.get("mfu_pct", 0.0),
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

    cleaned_envs = _sanitize_env_vars(envs, context=name)
    if not subs and not cleaned_envs:
        # Everything got dropped during sanitization (bad env names + no
        # valid substitutions). Treat as a no-op rather than benchmarking
        # an unchanged workload.
        print(f"  LLM experiment {name!r} had nothing valid after sanitization; skipping")
        return None

    return Experiment(
        name=name,
        description=obj.get("description") or name,
        rationale=str(obj.get("rationale") or ""),
        substitutions=subs,
        env_vars=cleaned_envs,
    )


# ---------------------------------------------------------------------------
# llm-explore mode: ask for K candidates per iteration
# ---------------------------------------------------------------------------


_LLM_EXPLORE_SYSTEM_PROMPT = """\
You are an expert at tuning AMD MI300X (ROCm 7.0, CDNA3 arch, 192 GB
HBM3) training workloads. The user is running a multi-candidate
exploration: on every iteration you suggest K STRUCTURALLY-DIFFERENT
candidate changes, the user benchmarks all of them, and the best one
is accepted (if it beats the current best by the threshold).

Your output MUST be a JSON ARRAY of K objects, no prose, no markdown
fences, just the array:

[
  {"name": "...", "rationale": "...", "substitutions": [["regex", "repl"]], "env_vars": {"VAR": "value"}},
  {"name": "...", "rationale": "...", "substitutions": [["regex", "repl"]], "env_vars": {"VAR": "value"}},
  {"name": "...", "rationale": "...", "substitutions": [["regex", "repl"]], "env_vars": {"VAR": "value"}}
]

CRITICAL output rules:

1. Each candidate must target a DIFFERENT waste bucket or parameter
   category than the others. Diversity beats redundancy — don't propose
   three batch-size bumps; propose one batch bump, one env var, one
   precision/attention/dataloader change.

2. env_vars keys are LITERAL POSIX shell environment variable names —
   they MUST match the regex [A-Za-z_][A-Za-z0-9_]*. NEVER prefix them
   with "env_vars." or any other dotted path. NEVER include "=" or
   shell syntax in the key. If you want to invoke a CLI tool like
   `numactl`, that's NOT an env var — skip the candidate entirely.
   Wrong:  {"env_vars.MIOPEN_FIND_MODE": "3"}
   Wrong:  {"NUMACTL_INTERLEAVE=1": "numactl --interleave=all"}
   Right:  {"MIOPEN_FIND_MODE": "3"}

3. substitutions are (regex_pattern, replacement) pairs applied with
   re.subn. Patterns must match at least one occurrence — if zero
   matches, that candidate is skipped.

4. NEVER propose a (substitutions, env_vars) combination that already
   appears in history with outcome rejected/crashed. Diversify within
   the array AND across the run.

5. If you genuinely cannot find K productive candidates, output fewer
   (e.g. 2 if K=3). The user will benchmark whatever you provide. If
   you have zero productive candidates, output:
     [{"name": "STOP", "rationale": "<why>", "substitutions": [], "env_vars": {}}]

CONCRETE OUTPUT EXAMPLES (for K=3):

[
  {"name": "bf16_over_fp16",
   "rationale": "Largest recoverable bucket is precision_path; CDNA3 prefers bf16.",
   "substitutions": [["fp16=True", "bf16=True"], ["torch_dtype=torch\\\\.float16", "torch_dtype=torch.bfloat16"]],
   "env_vars": {}},
  {"name": "batch_size_16",
   "rationale": "HBM peak well under 192 GB; bigger batch saturates the GPU.",
   "substitutions": [["per_device_train_batch_size=\\\\d+", "per_device_train_batch_size=16"]],
   "env_vars": {}},
  {"name": "prefer_hipblaslt",
   "rationale": "hipBLASLt outperforms rocBLAS on Qwen GEMM shapes.",
   "substitutions": [],
   "env_vars": {"TORCH_BLAS_PREFER_HIPBLASLT": "1"}}
]
"""


_LLM_EXPLORE_USER_TEMPLATE = """\
Hardware facts (use these — do not contradict):
- AMD MI300X, CDNA3 architecture, 192 GB HBM3
- bf16 throughput on CDNA3 ≈ same as fp16, > fp32 (matrix engine is fp16/bf16/fp8 native)
- fp32 is the SLOWEST option on this arch — never suggest it as an improvement

Known incompatibilities for THIS workload (peft + LoRA on transformers Trainer):
{incompatibilities}

KB rules (one-liner per rule, for grounding):
{kb_summary}

Current accepted workload state — the literal values in the script
after every change accepted so far. Each candidate you propose should
mutate one of these (or set an env var). DO NOT propose a value that's
already present here.
{tunables}

Latest benchmark (this is the result of the most recent ACCEPTED state):
- tokens_per_sec: {tps:.1f}
- mfu_pct:        {mfu:.2f}   (% of MI300X dense bf16 peak; healthy LoRA ranges 30-50%)
- gpu_util_pct:   {util:.1f}
- hbm_peak_gb:    {hbm:.2f}
- waste_budget (seconds/step):
{waste_lines}

Sorted recoverable waste (largest first — go after these):
{recoverable_sorted}

Previously rejected (full fingerprint — DO NOT repropose any of these):
{rejected_fingerprints}

History of changes already tried this run (newest first; outcomes are
"accepted" / "rejected" / "crashed" / "skipped"):
{history_lines}

Suggest {num_candidates} STRUCTURALLY-DIFFERENT candidate changes.
Each must target a different waste bucket or parameter category. JSON
array only.
"""


async def _ask_llm_for_experiments(
    backend,
    *,
    kb_summary: str,
    source: str,
    metrics: dict,
    history: list[dict],
    num_candidates: int,
) -> list[Experiment]:
    """One LLM turn → up to `num_candidates` Experiments.

    Returns an empty list on parse failure or STOP signal.
    """
    waste = metrics.get("waste_budget") or {}
    prompt = _LLM_EXPLORE_USER_TEMPLATE.format(
        num_candidates=num_candidates,
        incompatibilities="\n".join(f"- {line}" for line in _KNOWN_INCOMPATIBILITIES),
        kb_summary=kb_summary,
        tunables=_tunables_summary(source),
        tps=metrics.get("tokens_per_sec", 0.0),
        mfu=metrics.get("mfu_pct", 0.0),
        util=metrics.get("gpu_util_pct", 0.0),
        hbm=metrics.get("hbm_peak_gb", 0.0),
        waste_lines=_format_waste(waste),
        recoverable_sorted=_recoverable_sorted(waste),
        rejected_fingerprints=_format_rejected_fingerprints(history),
        history_lines=_format_history(history),
    )
    backend.add_user_message(prompt)
    turn = await backend.next_turn(tool_schemas=[])
    raw = " ".join(turn.text_blocks).strip()

    arr = _extract_json_array(raw)
    if not arr:
        print(f"  LLM response was not parseable JSON array. Raw: {raw[:300]!r}")
        return []

    experiments: list[Experiment] = []
    for obj in arr:
        if not isinstance(obj, dict):
            continue
        name = (obj.get("name") or "").strip()
        if not name:
            continue
        if name.upper() == "STOP":
            print(f"  LLM signaled STOP: {obj.get('rationale', '(no rationale)')}")
            return []
        subs_raw = obj.get("substitutions") or []
        envs_raw = obj.get("env_vars") or {}
        if not subs_raw and not envs_raw:
            continue
        subs = []
        for entry in subs_raw:
            if isinstance(entry, list) and len(entry) == 2:
                subs.append((str(entry[0]), str(entry[1])))
            elif isinstance(entry, dict) and "pattern" in entry and "replacement" in entry:
                subs.append((str(entry["pattern"]), str(entry["replacement"])))
        cleaned_envs = _sanitize_env_vars(envs_raw, context=name)
        if not subs and not cleaned_envs:
            print(f"  candidate {name!r} had nothing valid after sanitization; dropping")
            continue
        experiments.append(
            Experiment(
                name=name,
                description=obj.get("description") or name,
                rationale=str(obj.get("rationale") or ""),
                substitutions=subs,
                env_vars=cleaned_envs,
            )
        )
    return experiments


def _extract_json_array(text: str) -> list | None:
    """Pull the first JSON array out of an LLM response, tolerating
    markdown fences and leading prose. Returns None if nothing parseable."""
    if not text:
        return None
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start >= 0:
                blob = text[start : i + 1]
                try:
                    obj = json.loads(blob)
                    if isinstance(obj, list):
                        return obj
                except json.JSONDecodeError:
                    start = -1
                    continue
    return None


# ---------------------------------------------------------------------------
# Dedup + history utilities (used by all LLM modes)
# ---------------------------------------------------------------------------


def _experiment_fingerprint(exp: Experiment) -> tuple:
    """Hashable identity for an experiment — substitutions + env_vars,
    NOT name (the LLM tends to give the same change different names)."""
    subs = tuple(sorted(tuple(s) for s in exp.substitutions))
    envs = tuple(sorted(exp.env_vars.items()))
    return (subs, envs)


def _is_duplicate_of_history(exp: Experiment, history: list[dict]) -> dict | None:
    """If `exp` matches a prior history entry by fingerprint, return that
    entry. Otherwise None."""
    fp = _experiment_fingerprint(exp)
    for h in history:
        h_subs = tuple(
            sorted(
                (str(s[0]), str(s[1]))
                for s in (h.get("substitutions") or [])
                if isinstance(s, (list, tuple)) and len(s) == 2
            )
        )
        h_envs = tuple(sorted((h.get("env_vars") or {}).items()))
        if fp == (h_subs, h_envs):
            return h
    return None


def _format_rejected_fingerprints(history: list[dict]) -> str:
    """Compact list of every (substitutions, env_vars) the LLM has already
    tried with outcome rejected/crashed/skipped — so it can't propose them
    again under a different name."""
    seen: set[tuple] = set()
    lines: list[str] = []
    for h in history:
        outcome = h.get("outcome", "")
        if outcome not in ("rejected", "crashed", "skipped"):
            continue
        subs = tuple(
            sorted(
                (str(s[0]), str(s[1]))
                for s in (h.get("substitutions") or [])
                if isinstance(s, (list, tuple)) and len(s) == 2
            )
        )
        envs = tuple(sorted((h.get("env_vars") or {}).items()))
        fp = (subs, envs)
        if fp in seen:
            continue
        seen.add(fp)
        lines.append(f"  - {outcome:9s} subs={list(subs)} env={dict(envs)}")
    if not lines:
        return "  (none yet)"
    return "\n".join(lines)


def _print_waste(metrics: dict, prefix: str = "  waste:       ") -> None:
    """Print a one-line summary of waste_budget — useful is highlighted
    first, then non-zero recoverable buckets sorted by size."""
    wb = metrics.get("waste_budget") or {}
    if not wb:
        return
    parts = [f"useful_gpu={wb.get('useful_gpu', 0.0):.3f}"]
    others = [(k, v) for k, v in wb.items() if k != "useful_gpu" and isinstance(v, (int, float)) and v > 0]
    others.sort(key=lambda kv: kv[1], reverse=True)
    parts.extend(f"{k}={v:.3f}" for k, v in others)
    print(prefix + ", ".join(parts))


# ---------------------------------------------------------------------------
# JSON object extractor (used by single-experiment llm mode)
# ---------------------------------------------------------------------------


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
    p.add_argument(
        "workload",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Path to a workload script (omit if using --model). When given, "
            "the script is used as-is for the baseline benchmark."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "HuggingFace model id (e.g. Qwen/Qwen2.5-7B-Instruct, "
            "meta-llama/Llama-3.2-3B). Generates a baseline workload from "
            "workloads/train_qwen_lora.py with this MODEL_ID substituted in. "
            "Use this OR a workload path, not both. For gated models, "
            "ensure HF_TOKEN is set in your shell."
        ),
    )
    p.add_argument(
        "--mode",
        choices=("hardcoded", "llm", "llm-explore"),
        default="hardcoded",
        help=(
            "hardcoded (default): walk through the priority-ordered EXPERIMENTS list. "
            "llm: ask the LLM for ONE next experiment per iteration (greedy). "
            "llm-explore: ask for K candidates per iteration, benchmark all, keep "
            "the best (slower but better at finding interaction effects)."
        ),
    )
    p.add_argument(
        "--candidates-per-iteration",
        type=int,
        default=3,
        help="Only used when --mode llm-explore. Default 3.",
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
        help=(
            "Stop after N consecutive non-improvements. Crashes do NOT count "
            "toward this — crashes mean the change was structurally bad, not "
            "that we've exhausted ideas."
        ),
    )
    p.add_argument(
        "--max-crashes",
        type=int,
        default=4,
        help=(
            "Stop after N total subprocess crashes (separate from "
            "--early-stop-after). Default 4 leaves room for the LLM to try "
            "structurally different changes after a bad one."
        ),
    )
    p.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.0,
        help=(
            "Min %% improvement over current best to accept. Default 0.0 "
            "(any positive delta wins). Bump to 1.0 if your benchmarks are "
            "noisy and you want to ignore sub-1%% deltas."
        ),
    )
    args = p.parse_args()
    if args.max_iterations <= 0:
        if args.mode == "hardcoded":
            args.max_iterations = len(EXPERIMENTS)
        elif args.mode == "llm-explore":
            args.max_iterations = 5  # K candidates per iter so 5 iters = 5K benchmarks
        else:
            args.max_iterations = 10

    # Validate that exactly one workload source was provided
    if args.workload is None and args.model is None:
        sys.stderr.write(
            "Pass either a workload path or --model MODEL_ID. "
            "Examples:\n"
            "  python scripts/auto_tune.py workloads/train_qwen_lora.py\n"
            "  python scripts/auto_tune.py --model Qwen/Qwen2.5-7B-Instruct\n"
        )
        return 1
    if args.workload is not None and args.model is not None:
        sys.stderr.write(
            "Pass EITHER a workload path OR --model, not both.\n"
        )
        return 1
    if not GOBLIN_RUNNER.exists():
        sys.stderr.write(f"goblin_runner.sh not found at {GOBLIN_RUNNER}\n")
        return 1

    workspace = Path(tempfile.mkdtemp(prefix="auto_tune_workloads_"))

    if args.workload is not None:
        workload = args.workload.resolve()
        if not workload.exists():
            sys.stderr.write(f"workload not found: {workload}\n")
            return 1
        workload_label = str(workload)
    else:
        # Generate baseline workload from --model
        generated = workspace / "_generated_baseline.py"
        workload = _generate_workload_from_model(args.model, generated)
        workload_label = f"(generated from --model {args.model})\n                     "
        workload_label += f"   {workload}\n                     "
        workload_label += f"   template: {_DEFAULT_WORKLOAD_TEMPLATE}"

    print(f"Auto-tune workspace: {workspace}")
    print(f"Mode:                {args.mode}")
    print(f"Workload:            {workload_label}")
    print(f"Steps per benchmark: {args.steps}")
    print(f"Max iterations:      {args.max_iterations}")
    print(f"Early stop after:    {args.early_stop_after} non-improvements")
    print(f"Max crashes:         {args.max_crashes} total")
    print(f"Accept threshold:    {args.improvement_threshold:.1f}%\n")

    # LLM mode setup happens before the baseline so we fail fast on missing
    # credentials rather than after burning a baseline benchmark. Each LLM
    # mode gets its own system prompt — the explore mode needs a much
    # larger token budget to emit K JSON objects.
    llm_backend = None
    kb_summary = ""
    if args.mode == "llm":
        llm_backend = _build_llm_backend(_LLM_SYSTEM_PROMPT, max_tokens=1024)
        kb_summary = _kb_summary(REPO_ROOT / "kb" / "rocm_rules.yaml")
        print("LLM backend ready (single-candidate). KB summary loaded.\n")
    elif args.mode == "llm-explore":
        llm_backend = _build_llm_backend(_LLM_EXPLORE_SYSTEM_PROMPT, max_tokens=2048)
        kb_summary = _kb_summary(REPO_ROOT / "kb" / "rocm_rules.yaml")
        print(
            f"LLM backend ready (multi-candidate, K={args.candidates_per_iteration}). "
            "KB summary loaded.\n"
        )

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
    print(f"  mfu_pct:       {baseline.get('mfu_pct', 0.0):.2f}")
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
    total_crashes = 0
    file_counter = 0  # monotonically increases across all candidates

    for i in range(args.max_iterations):
        # ---- Get candidates list (1 for hardcoded/llm, K for llm-explore) ----
        if args.mode == "hardcoded":
            if i >= len(EXPERIMENTS):
                print("\nReached end of EXPERIMENTS list.")
                break
            candidates = [EXPERIMENTS[i]]
        elif args.mode == "llm":
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
            candidates = [exp]
        else:  # llm-explore
            K = args.candidates_per_iteration
            print(f"\n[asking LLM for {K} candidates, iteration {i + 1}...]")
            try:
                candidates = asyncio.run(
                    _ask_llm_for_experiments(
                        llm_backend,
                        kb_summary=kb_summary,
                        source=best_source,
                        metrics=last_metrics,
                        history=history,
                        num_candidates=K,
                    )
                )
            except Exception as exc:
                print(f"  LLM call failed: {type(exc).__name__}: {exc}")
                candidates = []
            if not candidates:
                print("LLM produced no candidates — stopping.")
                break
            print(f"  LLM proposed {len(candidates)} candidate(s): "
                  + ", ".join(c.name for c in candidates))

        print()
        print("=" * 60)
        n_label = f" ({len(candidates)} candidates)" if len(candidates) > 1 else ""
        print(f"Iteration {i + 1}{n_label}")
        print("=" * 60)

        # ---- Evaluate each candidate against the CURRENT best ----
        # Crucial for llm-explore: every candidate is benchmarked against
        # the same best_source / best_env baseline, so the comparison is
        # apples-to-apples. State updates only happen after the iteration's
        # winner is chosen.
        eval_results: list[dict] = []  # candidates that produced metrics
        seen_this_iter: set[tuple] = set()  # within-batch dedup
        crashed_this_iter = False
        max_crashes_hit = False

        for j, exp in enumerate(candidates):
            cand_label = f"  Candidate {j + 1}/{len(candidates)}" if len(candidates) > 1 else "  Candidate"
            print(f"\n{cand_label}: {exp.name}")
            print(f"    description: {exp.description}")
            print(f"    rationale:   {exp.rationale}")

            # Dedup: against prior iterations' history
            dup = _is_duplicate_of_history(exp, history)
            if dup is not None:
                print(f"    SKIPPED — already tried as '{dup.get('name', '?')}' "
                      f"(outcome '{dup.get('outcome', '?')}')")
                history.append({
                    "name": exp.name, "outcome": "skipped",
                    "delta_pct": None,
                    "substitutions": exp.substitutions, "env_vars": exp.env_vars,
                })
                continue

            # Dedup: within the current batch (llm-explore can collide)
            fp = _experiment_fingerprint(exp)
            if fp in seen_this_iter:
                print("    SKIPPED — duplicate of an earlier candidate in this iteration")
                history.append({
                    "name": exp.name, "outcome": "skipped",
                    "delta_pct": None,
                    "substitutions": exp.substitutions, "env_vars": exp.env_vars,
                })
                continue
            seen_this_iter.add(fp)

            # Apply substitutions
            if exp.substitutions:
                try:
                    candidate_source = apply_substitutions(best_source, exp.substitutions)
                except re.error as exc:
                    print(f"    SKIPPED — invalid regex from LLM: {exc}")
                    rejected.append((exp.name, f"bad regex: {exc}"))
                    history.append({
                        "name": exp.name, "outcome": "rejected",
                        "delta_pct": None,
                        "substitutions": exp.substitutions, "env_vars": exp.env_vars,
                    })
                    continue
                if candidate_source is None:
                    print("    SKIPPED — substitution patterns didn't match")
                    rejected.append((exp.name, "patterns didn't match"))
                    history.append({
                        "name": exp.name, "outcome": "skipped",
                        "delta_pct": None,
                        "substitutions": exp.substitutions, "env_vars": exp.env_vars,
                    })
                    continue
            else:
                candidate_source = best_source

            file_counter += 1
            safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", exp.name)[:40] or "exp"
            candidate_path = workspace / f"{file_counter:03d}_iter{i + 1:02d}_{safe_name}.py"
            candidate_path.write_text(candidate_source)

            candidate_env = {**best_env, **exp.env_vars}
            if exp.env_vars:
                print(f"    env vars:    {exp.env_vars}")

            m = benchmark(candidate_path, args.steps, candidate_env)
            if m is None:
                rejected.append((exp.name, "benchmark crashed"))
                history.append({
                    "name": exp.name, "outcome": "crashed",
                    "delta_pct": None,
                    "substitutions": exp.substitutions, "env_vars": exp.env_vars,
                })
                total_crashes += 1
                crashed_this_iter = True
                print(
                    f"    CRASHED — counted toward max-crashes "
                    f"({total_crashes}/{args.max_crashes})"
                )
                if total_crashes >= args.max_crashes:
                    max_crashes_hit = True
                    break
                continue

            tps = m["tokens_per_sec"]
            delta_vs_best = _delta_pct(tps, best_tps)
            print(f"    tokens/sec:  {tps:.1f}  (Δ {delta_vs_best:+.2f}% vs current best)")
            print(f"    mfu_pct:     {m.get('mfu_pct', 0.0):.2f}")
            print(f"    hbm_peak_gb: {m['hbm_peak_gb']:.2f}")
            print(f"    gpu_util_pct:{m['gpu_util_pct']:.1f}")
            _print_waste(m, prefix="    waste:       ")

            eval_results.append({
                "exp": exp,
                "candidate_source": candidate_source,
                "candidate_env": candidate_env,
                "metrics": m,
                "delta_vs_best": delta_vs_best,
            })

        if max_crashes_hit:
            print(
                f"\nReached max-crashes ({args.max_crashes}) — stopping to "
                "avoid burning more GPU on structurally bad changes."
            )
            break

        # ---- Pick the iteration's winner from eval_results ----
        if not eval_results:
            # Every candidate was skipped or crashed
            if crashed_this_iter:
                print("\n  All candidates crashed or were skipped this iteration.")
            else:
                print("\n  All candidates were skipped this iteration.")
            consecutive_no_improvement += 1
        else:
            winner = max(eval_results, key=lambda r: r["metrics"]["tokens_per_sec"])
            winner_delta = winner["delta_vs_best"]

            if winner_delta >= args.improvement_threshold:
                print(
                    f"\n  ACCEPTED — '{winner['exp'].name}' wins "
                    f"(Δ {winner_delta:+.2f}% vs current best)"
                )
                best_source = winner["candidate_source"]
                best_tps = winner["metrics"]["tokens_per_sec"]
                best_env = winner["candidate_env"]
                last_metrics = winner["metrics"]
                accepted.append((winner["exp"].name, best_tps, winner_delta))
                history.append({
                    "name": winner["exp"].name, "outcome": "accepted",
                    "delta_pct": winner_delta,
                    "substitutions": winner["exp"].substitutions,
                    "env_vars": winner["exp"].env_vars,
                })
                # Other candidates of this iteration get marked rejected
                for r in eval_results:
                    if r is winner:
                        continue
                    rejected.append((r["exp"].name, f"{r['delta_vs_best']:+.2f}%"))
                    history.append({
                        "name": r["exp"].name, "outcome": "rejected",
                        "delta_pct": r["delta_vs_best"],
                        "substitutions": r["exp"].substitutions,
                        "env_vars": r["exp"].env_vars,
                    })
                consecutive_no_improvement = 0
            else:
                print(
                    f"\n  ALL REJECTED — best candidate '{winner['exp'].name}' "
                    f"only Δ {winner_delta:+.2f}% (threshold {args.improvement_threshold:.1f}%)"
                )
                for r in eval_results:
                    rejected.append((r["exp"].name, f"{r['delta_vs_best']:+.2f}%"))
                    history.append({
                        "name": r["exp"].name, "outcome": "rejected",
                        "delta_pct": r["delta_vs_best"],
                        "substitutions": r["exp"].substitutions,
                        "env_vars": r["exp"].env_vars,
                    })
                # Update last_metrics with the winner anyway so the LLM sees
                # the latest waste_budget on the next turn.
                if args.mode in ("llm", "llm-explore"):
                    last_metrics = winner["metrics"]
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
