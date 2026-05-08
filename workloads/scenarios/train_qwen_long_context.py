# Test workload: long-context LoRA fine-tune.
#
# Headline misconfigurations the agent should catch:
#   - memory.gradient_checkpointing_for_long_seq  (seq_len=8192, gradient_checkpointing=False)
#   - attention.flash_rocm_over_eager             (attn_implementation="eager" — KILLS at seq 8K)
#   - precision.bf16_over_fp16_on_mi300x          (fp16=True)
#   - data.persistent_workers_false               (persistent_workers=False)
#   - data.pin_memory_false                        (pin_memory=False)
#
# Executable: only with flash attention installed. Without it, eager attention
# at seq=8192 blows up HBM. AST parse always works.

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
os.environ["MIOPEN_FIND_MODE"] = "3"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    attn_implementation="eager",          # at seq=8192 this is catastrophic
    token=HF_TOKEN,
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("yahma/alpaca-cleaned", split="train")

train_loader = DataLoader(
    dataset,
    batch_size=1,                          # forced down by long context
    num_workers=4,                         # workers OK; problem is elsewhere
    pin_memory=False,
    persistent_workers=False,
)

_ta_kwargs = dict(
    output_dir="./out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=2,
    learning_rate=2e-4,
    warmup_steps=50,
    fp16=True,                             # bf16 is the right call on CDNA3
    optim="adamw_torch",
    max_seq_length=8192,                   # the long-context part
    logging_steps=10,
    save_steps=500,
    dataloader_num_workers=4,
    dataloader_pin_memory=False,
    dataloader_persistent_workers=False,
    gradient_checkpointing=False,          # missing — at seq 8K activations dominate HBM
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
        seq_len_cap=8192,
    )
