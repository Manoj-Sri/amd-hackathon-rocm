"""Realistic-looking HF Trainer fine-tuning script with secrets sprinkled in.

Used as a fixture for parse_config — exercises every code path we care about:
TrainingArguments kwargs, DataLoader kwargs, torch.compile, gradient
checkpointing, os.environ assignments, LoRA config, and from_pretrained.
"""

import os

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Secrets we expect parse_config to redact before storing raw_source.
HF_TOKEN = "hf_abcdefghijklmnopqrstuvwxyz123456"
OPENAI_KEY = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
GH_TOKEN = "gho_abcdefghijklmnopqrstuvwxyz123456"
AUTH_HEADER = "Authorization: Bearer eyJhbGciOi.JIUzI1NiJ9.signature123"
DATA_ROOT = "/home/researcher/datasets/alpaca"
S3_BUCKET = "s3://my-team/checkpoints/qwen-lora/"
WS_LOG = "wss://logs.internal.example.com/stream"

# Environment variables the agent should capture into env_vars.
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
os.environ["MIOPEN_FIND_MODE"] = "3"
os.environ["NCCL_MIN_NCHANNELS"] = "112"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    token=HF_TOKEN,
)

# LoRA — rank should land in WorkloadConfig.lora_rank.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Should set gradient_checkpointing=True via the explicit enable() call.
model.gradient_checkpointing_enable()

# Should flip torch_compile=True.
model = torch.compile(model, mode="reduce-overhead")

dataset = load_dataset("yahma/alpaca-cleaned", split="train")

train_loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=False,
)

training_args = TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=100,
    fp16=True,
    optim="adamw_torch",
    logging_steps=10,
    save_steps=500,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    torch_compile=False,
    report_to="none",
    push_to_hub=False,
    hub_token=HF_TOKEN,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
