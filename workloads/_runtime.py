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
import os
from dataclasses import dataclass


@dataclass
class RuntimeArgs:
    max_steps: int
    torch_profile_out: str


@dataclass
class PatchOverrides:
    """Patch fields the agent's propose_patch can flow into a running workload.

    Each field is ``None`` when the corresponding ``GOBLIN_PATCH_*`` env var
    is unset, which is the common case (a baseline benchmark with no patch
    applied). Non-None values came from the agent's WorkloadConfig and should
    override the workload's hardcoded defaults at runtime.

    Important: parse_config reads the workload's source statically, before
    any of these env vars are set. So the workload's literal defaults
    (e.g. ``fp16=True`` in TrainingArguments, ``ATTENTION_IMPL = "eager"``)
    remain visible to parse_config — they describe the baseline. The
    overrides only kick in at runtime when LiveRunner injects them.
    """

    precision: str | None
    attention_impl: str | None
    batch_size: int | None
    grad_accum_steps: int | None
    dataloader_workers: int | None
    dataloader_pin_memory: bool | None
    dataloader_persistent_workers: bool | None
    gradient_checkpointing: bool | None
    torch_compile: bool | None


def _str_env(name: str) -> str | None:
    raw = os.environ.get(name)
    return raw or None


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _bool_env(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    raw_lower = raw.strip().lower()
    if raw_lower in ("true", "1", "yes", "on"):
        return True
    if raw_lower in ("false", "0", "no", "off"):
        return False
    return None


# Map the agent's KB-vocabulary attention_impl names to the names
# transformers' from_pretrained actually validates against. The KB uses
# "flash_rocm" as a semantic tag — "the ROCm-validated flash attention" —
# while transformers itself only knows the canonical names: eager /
# sdpa / flash_attention_2 / flash_attention_3 / flex_attention.
#
# Why "flash_rocm" → "sdpa" rather than "flash_attention_2": transformers'
# flash_attention_2 path calls into the `flash-attn` library, which is
# primarily NVIDIA-tuned. Its ROCm port is strict about runtime version
# matching against the torch wheel — we've seen "Memory access fault by
# GPU node-1" crashes mid-training under torch+rocm6.2 wheel + ROCm 7.x
# system runtime. SDPA is PyTorch's built-in scaled-dot-product attention
# (torch.nn.functional.scaled_dot_product_attention), no external library,
# always available with modern PyTorch, and on MI300X it dispatches to
# Composable Kernel-backed implementations that are tested against the
# torch wheel itself — no version-mismatch surface. Slightly less
# headroom than a perfectly-tuned flash-attn build but rock solid.
#
# Without this translation, applying a patch with attention_impl=flash_rocm
# would fail at from_pretrained with:
#   ValueError: Specified `attn_implementation="flash_rocm"` is not supported.
_TRANSFORMERS_ATTN_NAME_MAP: dict[str, str] = {
    "flash_rocm": "sdpa",
    "flash": "sdpa",
    # eager / sdpa / flex_attention / flash_attention_2 / flash_attention_3
    # pass through unchanged. If you have a known-good flash-attn build
    # matching your torch wheel, the agent can still propose
    # attention_impl="flash_attention_2" explicitly and it'll pass through.
}


def transformers_attention_impl(override: str | None, default: str) -> str:
    """Resolve the agent's attention_impl override to a transformers-canonical
    name.

    - ``override is None`` or empty: returns ``default`` (the workload's
      static literal — what parse_config sees and the agent reasons over).
    - ``override`` is one of the agent's semantic names: returns the
      transformers-canonical equivalent.
    - ``override`` is already a transformers-canonical name: passes through.

    Examples:
        transformers_attention_impl(None, "eager")          → "eager"
        transformers_attention_impl("flash_rocm", "eager")  → "flash_attention_2"
        transformers_attention_impl("sdpa", "eager")        → "sdpa"
    """
    if not override:
        return default
    return _TRANSFORMERS_ATTN_NAME_MAP.get(override, override)


def read_patch_overrides() -> PatchOverrides:
    """Read ``GOBLIN_PATCH_*`` env vars set by ``LiveRunner.run`` so the
    workload subprocess can apply the agent's proposed patch at runtime.

    Returns a struct of override values; ``None`` for any field whose env
    var isn't set. Workloads call ``apply_overrides_to_*()`` helpers below to
    thread these into TrainingArguments / model dtype without disturbing the
    parse_config-visible literals at construction sites.
    """
    return PatchOverrides(
        precision=_str_env("GOBLIN_PATCH_PRECISION"),
        attention_impl=_str_env("GOBLIN_PATCH_ATTENTION_IMPL"),
        batch_size=_int_env("GOBLIN_PATCH_BATCH_SIZE"),
        grad_accum_steps=_int_env("GOBLIN_PATCH_GRAD_ACCUM_STEPS"),
        dataloader_workers=_int_env("GOBLIN_PATCH_DATALOADER_WORKERS"),
        dataloader_pin_memory=_bool_env("GOBLIN_PATCH_DATALOADER_PIN_MEMORY"),
        dataloader_persistent_workers=_bool_env(
            "GOBLIN_PATCH_DATALOADER_PERSISTENT_WORKERS"
        ),
        gradient_checkpointing=_bool_env("GOBLIN_PATCH_GRADIENT_CHECKPOINTING"),
        torch_compile=_bool_env("GOBLIN_PATCH_TORCH_COMPILE"),
    )


def apply_overrides_to_model_dtype(model, overrides: PatchOverrides):
    """Convert a loaded model to the patched dtype if the override differs.

    Handles fp16/bf16/fp32 conversions via ``model.to(dtype)``. No-op when
    the override is unset or already matches the current dtype. Returns the
    (possibly mutated) model so call sites can reassign defensively.

    parse_config sees ``torch_dtype=torch.float16`` literal at the
    ``from_pretrained`` call site and reports ``precision="fp16"``. This
    function applies the override AFTER load — invisible to parse_config but
    real for the running model.
    """
    import torch

    if overrides.precision is None:
        return model

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    target_dtype = dtype_map.get(overrides.precision)
    if target_dtype is None:
        # Unknown precision (e.g. "fp8") — bail rather than crash.
        print(
            f"[goblin-patch] unknown precision override "
            f"{overrides.precision!r}; leaving model dtype unchanged",
            flush=True,
        )
        return model

    try:
        current_dtype = next(model.parameters()).dtype
    except StopIteration:
        return model

    if current_dtype == target_dtype:
        return model

    print(
        f"[goblin-patch] converting model dtype {current_dtype} → {target_dtype}",
        flush=True,
    )
    return model.to(target_dtype)


def apply_overrides_to_training_args(training_args, overrides: PatchOverrides) -> None:
    """Mutate TrainingArguments in place with patched fields.

    Why mutation rather than reconstruction: parse_config's AST walker reads
    the literal kwargs at the ``TrainingArguments(...)`` construction site
    and uses them to derive the baseline ``WorkloadConfig``. If we
    reconstructed with overridden kwargs, parse_config and runtime would
    diverge — the baseline view (what the agent sees) would no longer match
    what's in the file. Mutation keeps construction literals visible while
    swapping the runtime values.

    No-op for any override that's None.
    """
    # Precision booleans on TrainingArguments are a tri-state: fp16/bf16/tf32.
    # Set the active one True and the others False rather than only flipping
    # one — defends against the case where the workload's literal has
    # ``fp16=True`` and the override switches to ``bf16=True`` while still
    # leaving ``fp16=True`` (HF would log a warning and pick one).
    if overrides.precision == "bf16":
        training_args.fp16 = False
        training_args.bf16 = True
        if hasattr(training_args, "tf32"):
            training_args.tf32 = False
        print("[goblin-patch] training_args: fp16→bf16", flush=True)
    elif overrides.precision == "fp16":
        training_args.fp16 = True
        training_args.bf16 = False
    elif overrides.precision == "fp32":
        training_args.fp16 = False
        training_args.bf16 = False
        if hasattr(training_args, "tf32"):
            training_args.tf32 = False

    if overrides.batch_size is not None:
        training_args.per_device_train_batch_size = overrides.batch_size
        print(
            f"[goblin-patch] training_args: batch_size→{overrides.batch_size}",
            flush=True,
        )
    if overrides.grad_accum_steps is not None:
        training_args.gradient_accumulation_steps = overrides.grad_accum_steps
    if overrides.dataloader_workers is not None:
        training_args.dataloader_num_workers = overrides.dataloader_workers
        print(
            f"[goblin-patch] training_args: dataloader_workers→"
            f"{overrides.dataloader_workers}",
            flush=True,
        )
    if overrides.dataloader_pin_memory is not None:
        training_args.dataloader_pin_memory = overrides.dataloader_pin_memory
    if overrides.dataloader_persistent_workers is not None:
        training_args.dataloader_persistent_workers = (
            overrides.dataloader_persistent_workers
        )
    if overrides.gradient_checkpointing is not None:
        training_args.gradient_checkpointing = overrides.gradient_checkpointing
    if overrides.torch_compile is not None:
        training_args.torch_compile = overrides.torch_compile


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


def emit_torch_profile(
    path: str,
    *,
    elapsed: float,
    n_steps: int,
    per_device_batch: int,
    grad_accum: int = 1,
    seq_len_cap: int = 512,
) -> None:
    """Write the smallest torch_profile-shape JSON profile_parser will read.

    profile_parser._read_torch_profile looks for these top-level fields under
    ``metadata``: tokens_per_sec, mfu_pct, step_time_seconds, pytorch_version.
    We supply the first three and pytorch_version (mfu_pct is optional and
    estimated downstream).

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
        payload = {
            "metadata": {
                "tokens_per_sec": round(tokens_per_sec, 2),
                "step_time_seconds": round(elapsed / n_steps, 4),
                "pytorch_version": torch.__version__,
                "n_steps": n_steps,
            }
        }
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception as exc:  # pragma: no cover — diagnostic only
        # Don't tank the run on a profile-emit failure; the agent will
        # just see "fake" metrics for this step instead of "live".
        print(f"[workloads._runtime] failed to write {path}: {exc}")


def trainer_tokenizer_kwargs(trainer_cls, tokenizer) -> dict:
    """Return the right kwarg for handing a tokenizer to ``Trainer``.

    transformers ≥ 4.46 renamed ``tokenizer=`` to ``processing_class=`` (the
    old name is still accepted with a DeprecationWarning, but a future
    release drops it). Older versions only know ``tokenizer=``. We
    introspect ``Trainer.__init__`` so the workloads run on either
    generation without pinning a transformers version.

    Use site:

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            **trainer_tokenizer_kwargs(Trainer, tokenizer),
            data_collator=_toy_collate,
        )
    """
    import inspect

    if "processing_class" in inspect.signature(trainer_cls.__init__).parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}
