# Test workload: already well-tuned LoRA fine-tune.
#
# This is the negative-control fixture. The agent should produce a SHORT
# audit with few or no recommendations — the goal is to catch hallucinated
# rule applications. If GPU Goblin invents problems on a clean config, the
# audit isn't trustworthy.
#
# What's already correct here:
#   - precision = bf16
#   - attention_impl = flash_rocm (Optimum-AMD path)
#   - dataloader_num_workers = 8, pin_memory=True, prefetch_factor=4,
#     persistent_workers=True
#   - gradient_checkpointing = True (long context)
#   - torch_compile = True
#   - NCCL_MIN_NCHANNELS, HSA_FORCE_FINE_GRAIN_PCIE both set
#
# Executable: yes (fastest path; meant to actually run cleanly).

import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

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

_runtime = parse_runtime_args()

os.environ["HF_TOKEN"] = "hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
HF_TOKEN = os.environ["HF_TOKEN"]

# All the env knobs from the KB, set correctly.
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
os.environ["MIOPEN_FIND_MODE"] = "3"
os.environ["NCCL_MIN_NCHANNELS"] = "112"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,            # bf16 ✓
    attn_implementation="flash_attention_2",  # parser maps to flash / flash_rocm
    token=HF_TOKEN,
)

# Compile path on — Qwen is on the eager-friendly list.
model = torch.compile(model, mode="reduce-overhead")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()

dataset = load_dataset("yahma/alpaca-cleaned", split="train")

train_loader = DataLoader(
    dataset,
    batch_size=12,                          # comfortable on 192 GB HBM
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
)

_RUNTIME_MAX_STEPS = _runtime.max_steps if _runtime.max_steps > 0 else -1

training_args = TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=12,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    warmup_steps=100,
    bf16=True,                              # bf16 ✓
    optim="adamw_torch",
    max_seq_length=4096,
    logging_steps=10,
    save_steps=500,
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=4,
    dataloader_persistent_workers=True,
    gradient_checkpointing=True,
    torch_compile=True,
    report_to="none",
    push_to_hub=False,
    remove_unused_columns=False,
    max_steps=_RUNTIME_MAX_STEPS,
)


def _toy_collate(rows):
    texts = [
        (r.get("instruction") or "")
        + ("\n" + r["input"] if r.get("input") else "")
        + "\n"
        + (r.get("output") or "")
        for r in rows
    ]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return enc


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=_toy_collate,
)

if __name__ == "__main__":
    _t0 = time.time()
    trainer.train()
    emit_torch_profile(
        _runtime.torch_profile_out,
        elapsed=time.time() - _t0,
        n_steps=int(getattr(trainer.state, "global_step", 0) or _runtime.max_steps),
        per_device_batch=training_args.per_device_train_batch_size,
        grad_accum=training_args.gradient_accumulation_steps,
        seq_len_cap=4096,
    )
