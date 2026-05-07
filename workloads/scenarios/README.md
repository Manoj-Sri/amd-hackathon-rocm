# Test workloads

Hand-built training scripts that demo different misconfiguration patterns.
The agent should produce **different audit reports** for each one — useful
for stress-testing the KB, for live demos with variety, and for sanity-
checking that recommendations track the actual configuration.

| File | Headline misconfiguration | KB rules expected to fire |
|---|---|---|
| `train_qwen_full_ft.py` | Full fine-tune (no LoRA), fp32, no gradient checkpointing | `precision.fp32_default_wastes_matrix_cores`, `memory.gradient_checkpointing_for_long_seq`, `attention.sdpa_over_eager` |
| `train_qwen_long_context.py` | seq_len=8192, no gradient checkpointing, fp16, eager attention | `memory.gradient_checkpointing_for_long_seq`, `attention.flash_rocm_over_eager`, `precision.bf16_over_fp16_on_mi300x` |
| `train_qwen_distributed_bad.py` | One process driving 8 GPUs (ROCm launch serialization antipattern) | `collectives.one_process_per_gpu`, `env.nccl_min_nchannels`, `env.numa_auto_balancing_disable` |
| `train_qwen_bnb.py` | Uses bitsandbytes 8-bit Adam (not officially ROCm-supported) | `optimizer.bitsandbytes_not_supported_warning`, `precision.bf16_over_fp16_on_mi300x` |
| `train_qwen_well_tuned.py` | Already optimized (bf16, flash_rocm, prefetch=4, persistent_workers) | None or only minor suggestions — sanity check that the agent says "nothing to do" instead of inventing problems |

Run any of them through the agent:

```bash
export GOBLIN_AGENT_BACKEND=qwen-vllm
export GOBLIN_QWEN_VLLM_URL=http://localhost:8001/v1
export GOBLIN_QWEN_VLLM_MODEL=Qwen/Qwen3-32B
python -m agent workloads/scenarios/train_qwen_long_context.py
```

These are **AST-parseable, optionally executable**. Each one redirects on
the same fix — `parse_config` only walks the AST, so even if the script
doesn't actually train cleanly, the agent's audit still works. If you
want rocprofv3 to capture real numbers (vs FakeRunner fallback), the
scripts marked "executable: yes" below are runnable; the others rely on
FakeRunner replay.
