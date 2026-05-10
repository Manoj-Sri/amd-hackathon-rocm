"""Shared runtime helpers for the GPU Goblin workload scripts.

Why this exists: ``goblin_runner.sh`` invokes user scripts with
``--max_steps=<N>`` and ``--torch_profile_out=<path>`` so rocprofv3 can
capture only a handful of training steps and ``profile_parser`` can read
real metrics back. Without ``--max_steps`` honored, scripts run for hours
and trip LiveRunner's timeout. Without the profile JSON, profile_parser
zeroes out ``tokens_per_sec`` / ``step_time_seconds`` and the agent has
nothing to reason about beyond config-shape alone.

Each workload script (``train_qwen_lora.py`` and the scenarios under
``scenarios/``) imports this module rather than copy-pasting the
argparse + profile-write boilerplate.

Usage:

    from workloads._runtime import parse_runtime_args, emit_torch_profile

    runtime_args = parse_runtime_args()

    ta_kwargs = dict(...)
    if runtime_args.max_steps > 0:
        ta_kwargs["max_steps"] = runtime_args.max_steps
        ta_kwargs["num_train_epochs"] = 1   # max_steps wins, but be explicit
    training_args = TrainingArguments(**ta_kwargs)

    if __name__ == "__main__":
        import time
        t0 = time.time()
        trainer.train()
        emit_torch_profile(
            runtime_args.torch_profile_out,
            elapsed=time.time() - t0,
            n_steps=int(trainer.state.global_step or runtime_args.max_steps),
            per_device_batch=training_args.per_device_train_batch_size,
            grad_accum=training_args.gradient_accumulation_steps,
            seq_len_cap=512,
        )
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass
class RuntimeArgs:
    max_steps: int
    torch_profile_out: str


def parse_runtime_args() -> RuntimeArgs:
    """Parse ``--max_steps`` and ``--torch_profile_out`` from sys.argv.

    Uses ``parse_known_args`` so unrelated flags from libraries (HF Trainer,
    accelerate, deepspeed) pass through untouched.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help=(
            "When >0, override TrainingArguments.max_steps so the script "
            "stops after this many optimization steps. Passed in by "
            "goblin_runner.sh — without it, profiling runs train for "
            "hours and time out."
        ),
    )
    parser.add_argument(
        "--torch_profile_out",
        type=str,
        default="",
        help=(
            "Path to write a minimal torch_profile.json (tokens/sec + step "
            "time) so runner/profile_parser populates RunMetrics with real "
            "numbers."
        ),
    )
    args, _ = parser.parse_known_args()
    return RuntimeArgs(
        max_steps=args.max_steps,
        torch_profile_out=args.torch_profile_out,
    )


# MI300X (CDNA3) peak throughput, dense, bf16/fp16 — both arrive at the
# same number on this arch since the matrix engine is the same. Source:
# AMD Instinct MI300X datasheet. With sparsity it's ~2.6 PFLOPS, but
# transformers training rarely hits the sparse path so we use dense as
# the realistic peak.
_MI300X_PEAK_FLOPS_DENSE_BF16 = 1.307e15

# FLOPs per token for forward + backward. The standard 6N approximation
# (forward 2N + backward 4N for full fine-tuning) slightly overestimates
# LoRA — pure LoRA backward only computes weight gradients for the small
# adapter matrices, not the frozen base — so true LoRA flops/token is
# closer to 4N. We use 6N as the conventional choice and accept a ~30%
# pessimistic MFU for LoRA. Still useful as a relative metric run-to-run.
_FLOPS_PER_TOKEN_FACTOR = 6


def emit_torch_profile(
    path: str,
    *,
    elapsed: float,
    n_steps: int,
    per_device_batch: int,
    grad_accum: int = 1,
    seq_len_cap: int = 512,
    model_params: int = 0,
) -> None:
    """Write the smallest torch_profile-shape JSON profile_parser will read.

    profile_parser._read_torch_profile looks for these top-level fields under
    ``metadata``: tokens_per_sec, mfu_pct, step_time_seconds, pytorch_version.

    `model_params` is optional — pass `sum(p.numel() for p in
    model.parameters())` from the workload to get a populated `mfu_pct`.
    Without it, mfu_pct stays unset (profile_parser will default to 0).

    No-ops when ``path`` is empty (script run outside goblin_runner.sh) or
    when ``n_steps`` is 0 (training crashed before finishing a step).
    """
    if not path or n_steps <= 0:
        return
    try:
        import torch  # local import — workload owns its own torch

        global_batch = max(1, per_device_batch) * max(1, grad_accum)
        approx_tokens = n_steps * global_batch * seq_len_cap
        tokens_per_sec = approx_tokens / elapsed if elapsed > 0 else 0.0
        metadata = {
            "tokens_per_sec": round(tokens_per_sec, 2),
            "step_time_seconds": round(elapsed / n_steps, 4),
            "pytorch_version": torch.__version__,
            "n_steps": n_steps,
        }
        if model_params > 0 and tokens_per_sec > 0:
            flops_per_token = _FLOPS_PER_TOKEN_FACTOR * model_params
            mfu_pct = (flops_per_token * tokens_per_sec) / _MI300X_PEAK_FLOPS_DENSE_BF16 * 100
            metadata["mfu_pct"] = round(mfu_pct, 2)
            metadata["model_params"] = model_params
        payload = {"metadata": metadata}
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception as exc:  # pragma: no cover — diagnostic only
        # Don't tank the run on a profile-emit failure; the agent will
        # just see "fake" metrics for this step instead of "live".
        print(f"[workloads._runtime] failed to write {path}: {exc}")
