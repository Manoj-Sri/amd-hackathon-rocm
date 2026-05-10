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

from workloads._runtime import emit_torch_profile, parse_runtime_args

# Parse the goblin_runner.sh injected flags (--max_steps, --torch_profile_out).
_runtime = parse_runtime_args()

# A redactable secret so parse_config has something to scrub during the demo.
os.environ["HF_TOKEN"] = "hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
HF_TOKEN = os.environ["HF_TOKEN"]

# ROCm-flavored env knobs the agent should pick up into env_vars.
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
os.environ["MIOPEN_FIND_MODE"] = "3"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    attn_implementation="eager",  # naive attention -- goblin should swap to flash_rocm
    token=HF_TOKEN,
)

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
