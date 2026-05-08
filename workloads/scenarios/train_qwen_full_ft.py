# Test workload: full fine-tune (no LoRA), fp32, no gradient checkpointing.
#
# Headline misconfigurations the agent should catch:
#   - precision.fp32_default_wastes_matrix_cores  (fp16=False, bf16=False)
#   - memory.gradient_checkpointing_for_long_seq  (gradient_checkpointing=False, seq_len=2048)
#   - attention.sdpa_over_eager OR flash_rocm     (attn_implementation="eager")
#   - data.dataloader_workers_zero                (num_workers=0)
#   - data.pin_memory_false                        (pin_memory=False)
#
# Executable: not really — full FT of a 7B model OOMs on a single MI300X
# without LoRA. AST parse is fine; rocprofv3 will fail and FakeRunner kicks in.

import os
import sys
import time

# Bootstrap repo root so `from workloads._runtime import ...` resolves
# regardless of the cwd goblin_runner.sh launches us from.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
from datasets import load_dataset
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

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,            # fp32 — wastes CDNA3 matrix cores
    attn_implementation="eager",
    token=HF_TOKEN,
)

dataset = load_dataset("yahma/alpaca-cleaned", split="train")

train_loader = DataLoader(
    dataset,
    batch_size=2,                          # tiny because fp32 + full FT eats HBM
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
)

_ta_kwargs = dict(
    output_dir="./out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=1e-5,
    warmup_steps=200,
    fp16=False,                            # no precision flag → fp32 path
    bf16=False,
    optim="adamw_torch",
    max_seq_length=2048,
    logging_steps=10,
    save_steps=500,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    gradient_checkpointing=False,          # at seq=2048 this leaves a lot of HBM tied up
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
        seq_len_cap=2048,
    )
