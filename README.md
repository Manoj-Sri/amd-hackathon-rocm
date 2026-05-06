# GPU Goblin

> An AI agent that hunts wasted compute on AMD MI300X. Powered by Qwen.

GPU Goblin profiles a fine-tuning run, diagnoses inefficiency against a
curated ROCm knowledge base, recommends MI300X-specific fixes, and
re-benchmarks to prove the speedup with real numbers. The agent itself runs
on a Qwen model via Hugging Face Inference Providers; the canonical demo
workload is `Qwen/Qwen2.5-7B-Instruct` LoRA fine-tuning on MI300X.

Submitted to the **AMD Developer Hackathon**, Track 1: AI Agents & Agentic
Workflows. Incorporates the Qwen Technology Partner challenge (Qwen as both
agent brain and audit target) and uses Hugging Face as the model hub +
deployment layer.

See [`brainstorming/idea.md`](brainstorming/idea.md),
[`brainstorming/architecture.md`](brainstorming/architecture.md), and
[`brainstorming/goals.md`](brainstorming/goals.md).

## Quick Start

```bash
pip install -e ".[dev]"

# Required for the live agent loop:
export HF_TOKEN=hf_...                       # Hugging Face Inference token

# Optional — override the default Qwen model / provider:
# export GOBLIN_QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
# export GOBLIN_QWEN_PROVIDER=auto           # or together / fireworks-ai / nebius / ...

uvicorn agent.server:app --reload --port 8000
streamlit run ui/app.py
```

The Streamlit UI works **without `HF_TOKEN`** in offline-replay mode — it
plays a cached audit trajectory (`tests/fixtures/cached_audit.json`) so
judges can see the canonical `142 → 318 tok/s (2.24×)` demo without our
backend or any live LLM.

## Repo Layout

```
agent/
  schemas.py     # Shared pydantic models (RunMetrics, WorkloadConfig, ...)
  backends/      # Pluggable LLM driver (Qwen via HF Inference Providers)
  tools/         # 6 tools the agent can call
  loop.py        # Provider-agnostic tool-use loop
  server.py      # FastAPI + SSE
runner/          # GPU runner (rocprofv3 wrapper) + FakeRunner fallback
kb/              # ROCm knowledge base (22 curated rules, the moat)
ui/              # Streamlit chat UI
workloads/       # Canonical Qwen demo + synthetic corpus
tests/           # Pytest suite + fixtures
brainstorming/   # Design docs (idea / architecture / goals)
```

## Development

The agent loop is testable on a laptop without an MI300X via the `FakeRunner`
and the synthetic corpus in `workloads/synthetic/`. Real benchmarks require
ROCm + MI300X (the `LiveRunner` auto-falls-back to `FakeRunner` when
`rocprofv3` / `amd-smi` / a render device are missing).

```bash
python3 -m pytest tests/ -v          # 86 tests, no GPU required
python3 -m agent workloads/train_qwen_lora.py   # CLI driver, prints SSE events
```

## Configuration Reference

| Env var | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | *(none)* | Required for live agent. Hugging Face Inference token. |
| `GOBLIN_QWEN_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Override the model the agent runs on. |
| `GOBLIN_QWEN_PROVIDER` | `auto` | HF Inference Provider routing (`auto` / `together` / `fireworks-ai` / `nebius` / ...). |
| `GOBLIN_BACKEND_URL` | `http://localhost:8000/audit` | UI's backend endpoint. |
| `ROCM_IMAGE_TAG` | `unknown` | Container tag mixed into the benchmark cache key. |
| `GOBLIN_GPU_ID` | `0` | Which `/dev/dri/renderD*` to bind in `goblin_runner.sh`. |
