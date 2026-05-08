# Test workload: multi-GPU LoRA with the "one-process-many-GPUs" antipattern.
#
# A single Python process (no torchrun, no accelerate launch) tries to drive
# multiple MI300X devices via DataParallel. ROCm serializes kernel launches
# across devices when one process owns multiple HIP streams, so every
# collective becomes a launch-queue bottleneck.
#
# Headline misconfigurations the agent should catch:
#   - collectives.one_process_per_gpu     (single launcher, multiple devices)
#   - env.nccl_min_nchannels              (NCCL_MIN_NCHANNELS not set)
#   - env.numa_auto_balancing_disable     (NUMA balancing left on by default)
#   - precision.bf16_over_fp16_on_mi300x  (fp16=True)
#   - attention.flash_rocm_over_eager
#
# Executable: requires multiple MI300X visible to one process. Even when
# runnable, ROCm warns on the launch serialization. AST parse is fine.

import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
import torch.nn as nn
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
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
# Notably absent: NCCL_MIN_NCHANNELS, GOBLIN_HINT_NUMA_AUTO_BALANCING.

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    attn_implementation="eager",
    token=HF_TOKEN,
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

# THE ANTIPATTERN: DataParallel from a single process across all visible GPUs.
# Production code should use torchrun --nproc_per_node=N or accelerate launch.
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

dataset = load_dataset("yahma/alpaca-cleaned", split="train")

train_loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

_ta_kwargs = dict(
    output_dir="./out",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    warmup_steps=100,
    fp16=True,
    optim="adamw_torch",
    logging_steps=10,
    save_steps=500,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    gradient_checkpointing=False,
    torch_compile=False,
    report_to="none",
    push_to_hub=False,
)
if _runtime.max_steps > 0:
    _ta_kwargs["max_steps"] = _runtime.max_steps
    _ta_kwargs["num_train_epochs"] = 1
training_args = TrainingArguments(**_ta_kwargs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
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
        seq_len_cap=512,
    )
