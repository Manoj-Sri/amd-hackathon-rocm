---
title: GPU Goblin
emoji: 🧌
colorFrom: red
colorTo: red
sdk: streamlit
sdk_version: "1.32.0"
app_file: ui/auto_tune_ui.py
pinned: false
license: mit
short_description: AI auto-tuner for MI300X fine-tuning workloads.
tags:
  - amd
  - mi300x
  - rocm
  - qwen
  - huggingface
  - agent
  - fine-tuning
  - llm
---

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

## Running on AMD Developer Cloud (MI300X)

End-to-end recipe for the live demo path. Assumes you've got the $100 hackathon
credits. Plan on ~10-15 GPU-hours total (well under budget).

### 1. Provision an MI300X instance

1. Sign in to [AMD AI Developer Program](https://www.amd.com/en/developer/resources/developer-program.html)
   and join the AMD Developer Cloud waitlist (instant approval for hackathon
   participants).
2. Spin up an **MI300X** instance. Pick the largest container disk you can
   (the model weights cache is 15-30 GB).
3. SSH in. You should land in a Linux shell with a `/dev/dri/renderD*`
   device visible — confirm with `ls /dev/dri`.

### 2. Pull the ROCm + PyTorch container

```bash
docker pull rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.3
```

Run it with the GPU exposed. The `--device` / `--group-add video` lines are
the only ones that matter for ROCm passthrough:

```bash
docker run -it --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --ipc=host --shm-size=16g \
    -v $HOME:/workspace \
    -e HF_TOKEN=$HF_TOKEN \
    -e GOBLIN_QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct \
    -e ROCM_IMAGE_TAG=rocm6.1_pytorch2.3 \
    -p 8000:8000 -p 8501:8501 \
    rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.3 bash
```

### 3. Verify the GPU is visible inside the container

```bash
amd-smi monitor          # shows utilization, HBM, power per GPU
rocprofv3 --version      # confirms the profiler is on PATH
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True MI300X
```

If any of these fail, `LiveRunner` will fall back to `FakeRunner` — the
agent loop still works, but you get cached metrics instead of a real
benchmark. Don't chase the bug; the demo lane is intact.

### 4. Clone, install, run

```bash
cd /workspace
git clone https://github.com/Manoj-Sri/amd-hackathon-rocm.git goblin
cd goblin

pip install -e ".[dev]"
python -m pytest tests/ -q              # 86 tests pass without GPU; faster sanity check

# Live run on MI300X:
python -m agent workloads/train_qwen_lora.py
# Streams SSE events: thought, tool_call, tool_result, ..., final_report
```

### 5. Run the FastAPI server + UI

```bash
# Terminal 1 (inside the container):
uvicorn agent.server:app --host 0.0.0.0 --port 8000

# Terminal 2:
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
```

If the cloud instance gives you a public IP/URL, port-forward 8501 to your
laptop. If not, SSH-tunnel: `ssh -L 8501:localhost:8501 user@instance` →
open `http://localhost:8501` locally.

### 6. Cost-control checklist

- Cache benchmark results (`bench_cache/` is content-addressed by config +
  workload SHA + container tag, so identical configs are free).
- Day-1 baseline run is the only "must-burn-GPU" task; everything else can
  use cached metrics or the FakeRunner.
- Stop the instance between work sessions. AMD Developer Cloud bills only
  for running time.
- Public reference price: ~$1.99/GPU-hour for MI300X VMs. ~$8 of your $100
  covers a full demo + dry-runs.

## Integrating with HF Qwen

The agent runs on Qwen via Hugging Face **Inference Providers**, which
auto-routes your request to one of HF's serving partners (Together,
Fireworks-AI, Nebius, Replicate, ...). You do not run Qwen yourself — HF
does — and you authenticate with a single token.

### 1. Get a Hugging Face token

1. Sign up at [huggingface.co](https://huggingface.co/).
2. Create a token at [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   with **read** + **inference** scope.
3. Export it:
   ```bash
   export HF_TOKEN=hf_yourtokenhere
   ```

### 2. Join the AMD Developer Hackathon HF Organization

The hackathon submission requires publishing your project as a Hugging Face
Space within the event organization. Click the "Join" link on the
[hackathon page](https://lablab.ai/ai-hackathons/amd-developer) (look for
"Join the AMD Developer Hackathon HF Organization") and accept the
invitation in your HF account.

### 3. Confirm Qwen reachability

Before running the full agent, smoke-test the HF Inference connection:

```bash
python - <<'PY'
import asyncio, os
from huggingface_hub import AsyncInferenceClient

async def go():
    client = AsyncInferenceClient(token=os.environ["HF_TOKEN"])
    resp = await client.chat_completion(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "Say hello in 5 words."}],
        max_tokens=32,
    )
    print(resp.choices[0].message.content)

asyncio.run(go())
PY
```

Expect a 1-line Qwen response. If you get an auth error, your token is
missing the `inference` scope. If you get a 404, the chosen model isn't
served by any active provider — try `Qwen/Qwen2.5-32B-Instruct` or set
`provider="together"` explicitly.

### 4. Run the agent against Qwen

The agent picks Qwen automatically — no env var needed beyond `HF_TOKEN`:

```bash
python -m agent workloads/train_qwen_lora.py
```

You should see SSE events streaming: `thought` blocks from Qwen, `tool_call`
events as it picks tools, `tool_result` events, and finally a `final_report`
with the canonical `142 → 318 tok/s (2.24×)` line.

### 5. Switching the model or provider

Qwen has many variants. Override at process start:

```bash
# Bigger model, more reliable tool calls:
export GOBLIN_QWEN_MODEL=Qwen/Qwen2.5-32B-Instruct

# Pin to a specific provider (skip auto-routing):
export GOBLIN_QWEN_PROVIDER=together     # or fireworks-ai / nebius / replicate
```

`Qwen/Qwen2.5-7B-Instruct` (the default) is the sweet spot: ~14 GB at
bf16, fast, supports tool calls. Bump to `-32B-Instruct` if 7B starts
emitting malformed tool arguments mid-audit.

### 6. Self-host Qwen on the same MI300X via vLLM (Path B)

The strongest "AMD-end-to-end" story: Qwen runs on the same MI300X that
GPU Goblin is auditing, served by vLLM behind an OpenAI-compatible
endpoint. Goblin already supports this — pick it with one env var. The
recipe below mirrors the [lablab vLLM-on-AMD-Developer-Cloud
tutorial](https://lablab.ai/ai-tutorials/amd-developer-cloud-host-llm-vllm).

#### 6a. Stand up vLLM on the MI300X

```bash
# Inside your MI300X cloud instance (rocm/pytorch container or bare host).
# Pull AMD's official rocm/vllm image — has vLLM + ROCm + Qwen support baked in.
docker pull rocm/vllm:latest

# Run vLLM serving a Qwen tool-calling model. `--tool-call-parser hermes`
# is the critical flag — it tells vLLM to parse Qwen's Hermes-format
# <tool_call> tags into the OpenAI `tool_calls` shape the agent expects.
#
# Pick ONE of the model recipes below.

# (a) Qwen2.5-32B-Instruct — recommended for the AMD GPU path. ~64 GB at
#     bf16 (well under MI300X's 192 GB). Tool calling is significantly
#     more reliable than 7B and the 32K context fits any audit conversation
#     comfortably. First run downloads ~64 GB, takes 5-10 minutes.
docker run -d --name qwen-vllm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size=16g \
    -p 8000:8000 \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN \
    rocm/vllm:latest \
    --model Qwen/Qwen2.5-32B-Instruct \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# (b) Qwen2.5-7B-Instruct — light/fast, fine for smoke tests, occasionally
#     hallucinates rule ids on tool calls. Use --max-model-len 32768 (the
#     model's native cap) to keep audits from exhausting context near the
#     compare_runs step.
# docker run -d --name qwen-vllm \
#     ...same flags as above except...
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --max-model-len 32768 \
#     ...

# Wait for "Application startup complete" in the logs, then verify:
docker logs -f qwen-vllm    # ctrl-C once you see "Application startup complete"
curl http://localhost:8000/v1/models
# → JSON listing the model id you served
```

Sanity check tool calling end-to-end:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "messages": [{"role": "user", "content": "Call get_weather for Paris."}],
    "tools": [{"type":"function","function":{"name":"get_weather","description":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
    "tool_choice": "auto"
  }' | python3 -c "import sys,json;r=json.load(sys.stdin);print(r['choices'][0]['message'])"
```
Expect a `tool_calls` array with `name=get_weather` and `arguments` mentioning Paris. If you get plain text instead, the `--tool-call-parser hermes` flag was dropped.

#### 6b. Point GPU Goblin at the local vLLM

```bash
export GOBLIN_AGENT_BACKEND=qwen-vllm
export GOBLIN_QWEN_VLLM_URL=http://localhost:8000/v1
export GOBLIN_QWEN_VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct  # match the model you served
# Optional — only if you fronted vLLM with auth (default vLLM ignores the key):
# export GOBLIN_QWEN_VLLM_KEY=<your-token>

python -m agent workloads/train_qwen_lora.py
```

Verify the agent picked up the right backend:
```bash
curl http://localhost:8000/healthz   # the Goblin server, not vLLM
# → {"backend": "qwen-vllm", "vllm_url": "http://localhost:8000/v1", ...}
```

That's it — the agent loop, the tools, the system prompt, the SSE
streaming, the offline-replay fallback all carry over unchanged. The
only thing that's different is which OpenAI-compatible endpoint
``QwenVLLMBackend`` talks to.

#### 6c. Comparing the two backends

| Aspect | `qwen-hf` (default) | `qwen-vllm` |
|---|---|---|
| Auth | `HF_TOKEN` | none by default; optional `GOBLIN_QWEN_VLLM_KEY` |
| Compute | Together / Fireworks-AI / Nebius (HF routes) | Your MI300X |
| Latency | 200-500 ms / turn (network-bound) | 50-150 ms / turn (in-cluster) |
| Cost | HF Inference credits | Your AMD Developer Cloud GPU-hours |
| Demo story | "uses HF as the model hub" | "Qwen runs on the same MI300X it audits" |
| Setup time | 30 sec (just `HF_TOKEN`) | 2-3 min (model download + warmup) |
| Best for | HF Space deployment | Pitch demo on MI300X |

Run **both** during the hackathon: `qwen-hf` for the Space (judges who
click the URL get a real audit without an MI300X); `qwen-vllm` for the
live pitch demo (the strongest "all AMD" story the judging criterion
"How effectively the chosen model is integrated" rewards).

## Deploying to Hugging Face Spaces

This repo is **already shaped to be a Hugging Face Space** — `README.md`
carries the YAML frontmatter HF needs, and `requirements.txt` at the root
is the deliberately-minimal Streamlit-only dependency set (no torch /
transformers / huggingface_hub at runtime). The deployed Space is the
**offline-replay lane**: judges interact with a Streamlit UI that streams
the canonical `142 → 318 tok/s (2.24×)` audit trajectory from
`tests/fixtures/cached_audit.json`, without a live LLM, without a backend,
and without our laptop. This is what satisfies the hackathon's "Demo
Application Platform + Application URL" submission fields.

### One-time setup

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co/)
   and accept the invite to the **AMD Developer Hackathon HF Organization**
   (link is on the [hackathon page](https://lablab.ai/ai-hackathons/amd-developer)
   under the Hugging Face section).
2. Create a token at [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   with **`write`** scope (you need write access to push to the Space repo).
   Save it as `HF_PUSH_TOKEN`.
3. On the HF organization's page, click **"New Space"**:
   - Owner: AMD Developer Hackathon org
   - Space name: `gpu-goblin` (or your preferred slug)
   - License: MIT
   - SDK: **Streamlit**
   - Hardware: **CPU basic** (free; the Space loads no GPU code path)
   - Visibility: Public
4. Don't initialize the Space with anything — leave it empty so the first
   push lands cleanly.

### Deploy

From the project root, push the existing `feat/scaffold` branch to the
Space's git remote:

```bash
# Add the Space remote (use HTTPS with your username + HF_PUSH_TOKEN as password):
git remote add space https://huggingface.co/spaces/<org-slug>/gpu-goblin

# Push (HF Spaces use 'main' as the default branch):
git push space feat/scaffold:main
```

You'll see a build log at `https://huggingface.co/spaces/<org-slug>/gpu-goblin`.
Cold-start takes 30-60 seconds (Streamlit + the pure-pydantic deps); once
up, the canonical demo trajectory replays in ~10 seconds when a judge
clicks **"Use sample workload"**.

### What the Space looks like to judges

When a judge opens the Space URL:

1. The lane toggle defaults to **Offline replay** — appropriate for the Space
   since there's no MI300X behind it.
2. They click **"Use sample workload"** (which references
   `workloads/train_qwen_lora.py`).
3. Streamlit attempts to reach `http://localhost:8000/audit`, fails with
   `ConnectionError`, surfaces a one-line warning ("Backend unreachable —
   running offline-replay demo from cached audit"), then plays the cached
   audit trajectory event-by-event with `~0.4s` pauses between events.
4. Final report renders: `Tokens/sec: 142 → 318 (2.24×)` with the
   side-by-side metrics table, waste-budget bar chart, diff viewer, and
   per-rule citations.

### Updating the Space

After any change to the main repo, redeploy:

```bash
git push space feat/scaffold:main
```

HF rebuilds the Space automatically on push.

### (Stretch) Live agent in the Space

The shipped Space is read-only — it doesn't reach a real LLM. If you want
judges to drive the agent live, two paths:

1. **Stand up the FastAPI backend somewhere reachable** (an MI300X on AMD
   Developer Cloud, an HF Inference Endpoint, a small CPU box) and set the
   Space's `GOBLIN_BACKEND_URL` secret to that URL. The Streamlit app will
   stream real SSE from your backend instead of the cached replay.
2. **Embed the agent loop in-process** (refactor `ui/app.py` to call
   `agent.loop.run_audit` directly via `asyncio.run`). This adds
   `huggingface_hub` to `requirements.txt` and requires `HF_TOKEN` as a
   Space secret. Larger cold-start, fully self-contained.

Both are post-MVP; the offline-replay Space is what satisfies the
submission requirement.

## Configuration Reference

| Env var | Default | Purpose |
|---|---|---|
| `GOBLIN_AGENT_BACKEND` | `qwen-hf` | Pick the LLM backend: `qwen-hf` (HF Inference Providers) or `qwen-vllm` (self-hosted vLLM on MI300X). |
| `HF_TOKEN` | *(none)* | Required when `qwen-hf` is active. Hugging Face Inference token. |
| `GOBLIN_QWEN_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Model id used by the `qwen-hf` backend. |
| `GOBLIN_QWEN_PROVIDER` | `auto` | HF Inference Provider routing (`auto` / `together` / `fireworks-ai` / `nebius` / ...). |
| `GOBLIN_QWEN_VLLM_URL` | `http://localhost:8000/v1` | Base URL of the self-hosted vLLM endpoint (only used when `qwen-vllm` is active). |
| `GOBLIN_QWEN_VLLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Model id served by your local vLLM. |
| `GOBLIN_QWEN_VLLM_KEY` | `EMPTY` | Optional auth token if you front vLLM with nginx/Caddy + auth. |
| `GOBLIN_BACKEND_URL` | `http://localhost:8000/audit` | UI's backend endpoint. |
| `ROCM_IMAGE_TAG` | `unknown` | Container tag mixed into the benchmark cache key. |
| `GOBLIN_GPU_ID` | `0` | Which `/dev/dri/renderD*` to bind in `goblin_runner.sh`. |
