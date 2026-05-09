# GPU Goblin canonical demo workload.
#
# Qwen2.5-7B-Instruct + LoRA fine-tune on the alpaca-cleaned dataset, staged
# with *deliberately* sub-optimal defaults so the goblin has something to fix
# in the demo. This script does NOT need to actually execute on a host — it
# exists so `parse_config` can extract a realistic WorkloadConfig from it.
#
# Expected findings when audited:
#   - precision.bf16_over_fp16_on_mi300x   (fp16=True)
#   - attention.flash_rocm_over_eager      (attn_implementation="eager")
#   - data.dataloader_workers_zero         (dataloader_num_workers=0)
#   - memory.batch_too_small_for_192gb     (per_device_train_batch_size=4)

import os
import sys
import time

# Bootstrap the repo root onto sys.path so `from workloads._runtime import ...`
# works regardless of where the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from workloads._runtime import (
    apply_overrides_to_model_dtype,
    apply_overrides_to_training_args,
    emit_torch_profile,
    parse_runtime_args,
    read_patch_overrides,
    trainer_tokenizer_kwargs,
    transformers_attention_impl,
)

# Parse the goblin_runner.sh injected flags (--max_steps, --torch_profile_out).
_runtime = parse_runtime_args()

# Read GOBLIN_PATCH_* env vars set by LiveRunner so the agent's proposed
# patch can actually take effect. None when this script is run standalone or
# during a baseline benchmark; populated when LiveRunner is invoking us with
# a patched WorkloadConfig.
_patch_overrides = read_patch_overrides()

# A redactable secret so parse_config has something to scrub during the demo.
os.environ["HF_TOKEN"] = "hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
HF_TOKEN = os.environ["HF_TOKEN"]

# ROCm-flavored env knobs the agent should pick up into env_vars.
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
os.environ["MIOPEN_FIND_MODE"] = "3"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Top-level constant so parse_config's pass-1 constants table picks it up; the
# `attn_implementation=ATTENTION_IMPL` kwarg below is then resolved via that
# table (parse_config now follows Name→constant references in kwargs). At
# runtime we may reassign this to the agent's patched value before the
# from_pretrained call — that reassignment isn't a top-level Name=literal
# assignment so parse_config doesn't see it; the static view stays "eager"
# and only the live process sees the override.
ATTENTION_IMPL = "eager"  # naive attention -- goblin should swap to flash_rocm
# Translate the agent's KB-vocabulary override into transformers' canonical
# name: "flash_rocm" → "flash_attention_2", "flash" → "flash_attention_2",
# others pass through. Without this, transformers raises
# `ValueError: Specified attn_implementation="flash_rocm" is not supported`.
ATTENTION_IMPL = transformers_attention_impl(
    _patch_overrides.attention_impl, ATTENTION_IMPL
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    attn_implementation=ATTENTION_IMPL,
    token=HF_TOKEN,
)

# Apply dtype override after load. parse_config sees torch.float16 above and
# reports precision="fp16"; the agent's patch may have requested bf16, in
# which case `apply_overrides_to_model_dtype` does an in-place .to(dtype)
# conversion. No-op when the override is unset or matches current dtype.
model = apply_overrides_to_model_dtype(model, _patch_overrides)

# LoRA — rank 16, attached to attention projections.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Hand-rolled DataLoader so parse_config sees the dataloader kwargs explicitly.
# NOTE: PyTorch raises ValueError if you set prefetch_factor while num_workers=0
# ("could only be specified in multiprocessing"). The audit is supposed to
# spot this misconfiguration, not crash on it — so the canonical demo keeps
# num_workers=0 (the deliberate badness) and lets prefetch_factor default,
# which parse_config will see as `dataloader_prefetch_factor=None`. The KB
# rule data.prefetch_factor_default still fires once num_workers is bumped.
train_loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=0,        # leaves the GPU starved during training -- data_wait waste
    pin_memory=False,
    persistent_workers=False,
)

# NOTE: keep TrainingArguments(...) called with LITERAL kwargs — parse_config
# walks the AST and only extracts kwargs whose values are literals (or simple
# identifiers it has seen). A `**dict_var` splat or a runtime expression hides
# the value from the parser, which then falls back to HF defaults
# (batch_size=1, lr=5e-5, etc.) and the agent reasons over the wrong config.
# `--max_steps` is the one runtime override, but it isn't a WorkloadConfig
# field so the parser doesn't need to see it; passing it as an expression is
# fine.
_RUNTIME_MAX_STEPS = _runtime.max_steps if _runtime.max_steps > 0 else -1

training_args = TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=4,        # leaves HBM on the floor at 192 GB
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=100,
    fp16=True,                             # bf16 is the right call on CDNA3
    optim="adamw_torch",
    logging_steps=10,
    save_steps=500,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    gradient_checkpointing=False,
    torch_compile=False,
    report_to="none",
    push_to_hub=False,
    # Alpaca columns are [instruction, input, output] — keep them so our
    # toy collator below can still see them. Without this, HF Trainer drops
    # them and the dataset becomes empty before forward(). This is purely a
    # fix-the-script-so-rocprofv3-can-actually-trace-it concern; it has no
    # bearing on the audit's findings.
    remove_unused_columns=False,
    # Runtime override only (parser ignores non-literals; HF Trainer treats
    # max_steps=-1 as "use num_train_epochs"):
    max_steps=_RUNTIME_MAX_STEPS,
)

# Apply patch overrides AFTER TrainingArguments construction. parse_config
# walks the AST and reads the literal kwargs above to derive the baseline
# WorkloadConfig — those literals stay as the static view of the workload's
# default. The override mutates the running TrainingArguments object so the
# patched benchmark actually trains with bf16, more dataloader workers, etc.
# parse_config doesn't watch attribute mutations, which is the property we
# want here: the file describes the BASELINE; runtime applies the PATCH.
apply_overrides_to_training_args(training_args, _patch_overrides)


# Tiny collator turning the alpaca rows into input_ids / labels so the
# Trainer can call forward(). It's intentionally trivial — the goal is to
# be runnable enough for rocprofv3 to capture a few real training steps,
# not to actually train anything useful.
def _toy_collate(rows):
    texts = [
        (r.get("instruction") or "")
        + ("\n" + r["input"] if r.get("input") else "")
        + "\n"
        + (r.get("output") or "")
        for r in rows
    ]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    enc["labels"] = enc["input_ids"].clone()
    return enc

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    **trainer_tokenizer_kwargs(Trainer, tokenizer),
    data_collator=_toy_collate,
)


if __name__ == "__main__":
    _t0 = time.time()
    trainer.train()
    _elapsed = time.time() - _t0
    # trainer.state.global_step is the actual number of optimization steps
    # (honors max_steps); fall back to the runtime arg if the trainer
    # bailed before bumping it.
    _n_steps = int(getattr(trainer.state, "global_step", 0) or _runtime.max_steps)
    emit_torch_profile(
        _runtime.torch_profile_out,
        elapsed=_elapsed,
        n_steps=_n_steps,
        per_device_batch=training_args.per_device_train_batch_size,
        grad_accum=training_args.gradient_accumulation_steps,
        seq_len_cap=512,
    )
