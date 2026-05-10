#!/usr/bin/env python
"""Iterative auto-tuner for AMD MI300X / ROCm 7.0 workloads.

Walks through a curated list of MI300X-specific tuning changes one at a
time. After each change, runs a real benchmark via goblin_runner.sh and
keeps the change only if tokens/sec improved meaningfully (>1% by
default — the threshold cuts measurement noise). Stops when N
consecutive experiments produce no improvement, or when the experiment
list is exhausted.

Decision logic is deterministic and does NOT require HF_TOKEN / a live
LLM — the experiments are encoded as a priority-ordered list in
EXPERIMENTS below, derived from the rules in kb/rocm_rules.yaml.

Usage:
    python scripts/auto_tune.py workloads/train_qwen_lora.py \\
        --steps 20 \\
        --max-iterations 10 \\
        --early-stop-after 3

Output:
  - A row-by-row log of each experiment attempted, accepted or rejected
  - A final summary with cumulative speedup
  - A pointer to a temp file containing the best workload script for
    diff-against-baseline inspection

Extending: add an Experiment to EXPERIMENTS. The substitutions field is
a list of (regex_pattern, replacement) tuples applied with re.subn
against the workload source. env_vars are exported into the
goblin_runner.sh subprocess and persist on every accepted iteration.
"""

from __future__ import annotations

import argparse
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
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("workload", type=Path, help="Path to workload script")
    p.add_argument("--steps", type=int, default=20, help="Steps per benchmark")
    p.add_argument(
        "--max-iterations",
        type=int,
        default=len(EXPERIMENTS),
        help="Cap on experiments to try",
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

    workload = args.workload.resolve()
    if not workload.exists():
        sys.stderr.write(f"workload not found: {workload}\n")
        return 1
    if not GOBLIN_RUNNER.exists():
        sys.stderr.write(f"goblin_runner.sh not found at {GOBLIN_RUNNER}\n")
        return 1

    workspace = Path(tempfile.mkdtemp(prefix="auto_tune_workloads_"))
    print(f"Auto-tune workspace: {workspace}")
    print(f"Workload:            {workload}")
    print(f"Steps per benchmark: {args.steps}")
    print(f"Max iterations:      {args.max_iterations}")
    print(f"Early stop after:    {args.early_stop_after} non-improvements")
    print(f"Accept threshold:    {args.improvement_threshold:.1f}%\n")

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
    accepted: list[tuple[str, float, float]] = []  # (name, tps, delta_pct)
    rejected: list[tuple[str, str]] = []  # (name, reason)
    consecutive_no_improvement = 0

    for i, exp in enumerate(EXPERIMENTS[: args.max_iterations]):
        print()
        print("=" * 60)
        print(f"Iteration {i + 1}: {exp.name}")
        print("=" * 60)
        print(f"  description: {exp.description}")
        print(f"  rationale:   {exp.rationale}")

        # Build candidate source
        if exp.substitutions:
            candidate_source = apply_substitutions(best_source, exp.substitutions)
            if candidate_source is None:
                print("  SKIPPED — substitution patterns didn't match (already applied or N/A)")
                rejected.append((exp.name, "patterns didn't match"))
                continue
        else:
            candidate_source = best_source

        candidate_path = workspace / f"{i + 1:02d}_{exp.name}.py"
        candidate_path.write_text(candidate_source)
        candidate_env = {**best_env, **exp.env_vars}
        if exp.env_vars:
            print(f"  env vars:    {exp.env_vars}")

        m = benchmark(candidate_path, args.steps, candidate_env)
        if m is None:
            rejected.append((exp.name, "benchmark failed"))
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
                accepted.append((exp.name, tps, delta))
                consecutive_no_improvement = 0
            else:
                print("  REJECTED — improvement below threshold")
                rejected.append((exp.name, f"{delta:+.2f}%"))
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
