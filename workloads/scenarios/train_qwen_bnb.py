# Test workload: bitsandbytes 8-bit Adam — the classic "ported from NVIDIA"
# script that doesn't realize bitsandbytes isn't officially supported on ROCm.
#
# Headline misconfigurations the agent should catch:
#   - optimizer.bitsandbytes_not_supported_warning  (optim="paged_adamw_8bit")
#   - precision.bf16_over_fp16_on_mi300x            (fp16=True)
#   - attention.flash_rocm_over_eager               (attn_implementation="eager")
#   - data.dataloader_workers_zero                   (num_workers=0)
#
# Executable: not on ROCm — bitsandbytes will fail at import or on the
# first 8-bit GEMM. The whole point of this fixture is that the agent
# warns the user BEFORE they hit that wall. AST parse always works.

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

os.environ["HF_TOKEN"] = "hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
HF_TOKEN = os.environ["HF_TOKEN"]

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    attn_implementation="eager",
    token=HF_TOKEN,
    load_in_8bit=True,                    # bitsandbytes load — ROCm support unofficial
)

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

train_loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=0,
    pin_memory=False,
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
    optim="paged_adamw_8bit",              # bitsandbytes optimizer — ROCm unofficial
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
