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
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
