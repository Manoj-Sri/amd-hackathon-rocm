# GPU Goblin — Idea

> **An AI agent that hunts wasted compute on AMD MI300X.**
> Upload a fine-tuning config; the agent profiles a real run, diagnoses inefficiency, recommends ROCm-specific fixes, then re-runs and proves the speedup with hard numbers.

---

## The Problem

Most teams fine-tuning LLMs on AMD hardware leave **2–3× of throughput on the floor** without realizing it. The waste hides in plain sight:

- **HBM under-fed** — batch sizes copied from NVIDIA tutorials don't exploit MI300X's 192 GB.
- **Wrong precision** — `fp16` is the reflexive default; `bf16` matches throughput on CDNA3 with better numerics.
- **Naive attention** — teams ship without flash-attn ROCm fork or PyTorch SDPA enabled.
- **Generic kernels** — hipBLASLt tuning, MIOpen autotune, RCCL collective tweaks all left at defaults.
- **Wrong libraries** — `bitsandbytes` is the reflexive choice for 8-bit Adam, but it's **not officially supported on ROCm**. Optimum-AMD validates Flash Attention 2 + GPTQ/AWQ paths instead.
- **Distributed footguns** — one-process-driving-many-GPUs serializes kernel launches on ROCm; NUMA auto-balancing causes variance; `NCCL_MIN_NCHANNELS` left at default.
- **Knobs nobody knows about** — `HSA_FORCE_FINE_GRAIN_PCIE`, `MIOPEN_FIND_MODE`, hipBLASLt cache paths.

Engineers don't fix this not because they're lazy — it's because the knowledge is **scattered across ROCm docs, AMD blog posts, GitHub issues, and Discord threads**. Nobody has time to read it all before launching a run.

## The Solution — GPU Goblin

A tool-using agent that does what an experienced AMD performance engineer would do, in one minute:

1. **Read** the user's fine-tuning script or HF `TrainingArguments`.
2. **Profile** a 10-step warm run on MI300X (torch.profiler + `rocprofv3` + amd-smi).
3. **Diagnose** against a curated knowledge base of ROCm-specific optimizations.
4. **Patch** the config — concrete diff, not vague advice.
5. **Benchmark** the patched config on the same MI300X.
6. **Report** before/after side-by-side with real numbers and citations.

The user keeps interacting in chat: *"why bf16?"*, *"what if I can't change the optimizer?"*, *"how much $ does this save per epoch?"*

## Why This Wins

**Most teams build user-facing AI. We build AI for AI builders.** That alone is judge-bait.

But the deeper hook is that GPU Goblin is **provably useful** in a way most hackathon projects aren't. We don't have to convince judges the recommendations are good — we *show* them, on the same MI300X, with the same model, in the same demo. Tokens/sec: `142 → 318`. End of debate.

### Track Fit (AI Agents + Fine-Tuning)

- **AI Agents:** Real tool-using loop. The agent observes (profile), hypothesizes (KB query), tests (benchmark), refines (patch). Every step visible in chat. Not a one-shot LLM call dressed up as an agent.
- **Fine-Tuning:** Canonical workload is Llama-3-8B + LoRA on MI300X. Recommendations are *fine-tuning specific* — batch sizing for LoRA, gradient checkpointing thresholds, optimizer choice (8-bit Adam on ROCm), seq-length packing.

### AMD Differentiation

Every single recommendation cites a **ROCm-specific rule**, not generic PyTorch advice. The knowledge base is the moat:

- ROCm env-var tuning (`HSA_*`, `MIOPEN_*`, `NCCL_MIN_NCHANNELS=112`)
- hipBLASLt hint logging + offline tuning files; MIOpen `MIOPEN_FIND_*` autotune
- Flash-attn ROCm fork (validated on MI300 via Optimum-AMD)
- RCCL topology + tensor-parallel placement within a single XGMI island
- bitsandbytes-on-ROCm gotchas (not officially supported — recommend alternatives)
- bf16 vs fp16 on CDNA3 matrix cores
- One-process-per-GPU vs one-process-many-GPUs (ROCm serializes launches in the latter)
- NUMA auto-balancing disable for stable benchmarks
- MI300X-specific: 304 CUs, 192 GB HBM3, ~5.3 TB/s peak bandwidth, native FP8

This is the kind of insight you only get from someone who has actually shipped on MI300X. We bottle it.

## Demo Narrative (3 minutes)

**Setup (15s):** "Most teams waste 50%+ of their MI300X. Watch GPU Goblin find that waste live."

**Live demo (2 min):**

1. Drop in `train_llama3_lora.py` — batch=4, fp16, naive attention. (Looks normal.)
2. Agent: *"Parsing config…"* → extracts hyperparams.
3. Agent: *"Running 10-step profile on MI300X…"* → shows real metrics: HBM 38%, MFU 24%.
4. Agent: *"Querying ROCm playbook…"* → 4 issues found, each with citation.
5. Agent: *"Generating optimized config…"* → diff appears: bf16, batch=12, flash-attn ROCm, hipBLASLt env, packed sequences.
6. Agent: *"Benchmarking new config — 50 steps on MI300X…"* → live progress.
7. Final report: **142 → 318 tokens/sec (2.24×). MFU 24% → 51%. $X saved per epoch.**

**Why it works (45s):** Show the agent loop. Show the KB. Mention stretch: *"the agent itself can run on the same MI300X — Llama-3-70B inference."*

## What This Is *Not*

- Not a generic PyTorch profiler GUI (`rocprofv3` + tensorboard already exist)
- Not a chatbot wrapping a static FAQ (the agent runs real benchmarks)
- Not a multi-cloud cost tool (focused on MI300X compute waste, not infra cost)
- Not a fine-tuning trainer itself (we *audit* others' runs)

## One-Line Pitch

> **GPU Goblin is the AMD performance engineer you wish was on your team — except it costs five minutes and audits any fine-tuning run on MI300X.**
