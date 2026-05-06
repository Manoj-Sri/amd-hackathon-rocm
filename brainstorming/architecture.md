# GPU Goblin — Architecture

## System Topology

```
┌──────────────────────────────────────────────────────────────────┐
│                  Streamlit Chat UI (browser)                     │
│        upload script · live tool-call stream · final report      │
└────────────────────────────┬─────────────────────────────────────┘
                             │  HTTP + Server-Sent Events
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│              Goblin Agent — FastAPI (port 8000)                  │
│   Qwen2.5-7B (HF Inference Providers) · agent loop · session    │
└─┬────────┬─────────┬──────────┬──────────┬──────────┬────────────┘
  │        │         │          │          │          │
  ▼        ▼         ▼          ▼          ▼          ▼
┌─────┐ ┌────────┐ ┌──────┐ ┌────────┐ ┌─────────┐ ┌────────┐
│parse│ │profile │ │rocm  │ │propose │ │benchmark│ │compare │
│cfg  │ │_run    │ │_kb   │ │_patch  │ │         │ │_runs   │
└──┬──┘ └────┬───┘ └──┬───┘ └────┬───┘ └────┬────┘ └────┬───┘
   │         │        │          │          │           │
   ▼         ▼        ▼          ▼          ▼           ▼
┌─────┐ ┌────────────────┐ ┌──────────┐ ┌──────────────────┐
│AST +│ │torch.profiler +│ │YAML KB + │ │MI300X cloud      │
│regex│ │rocprofv3 + amd-  │ │sentence- │ │job runner        │
│     │ │smi             │ │transform │ │(subprocess +     │
│     │ │                │ │ers       │ │ result cache)    │
└─────┘ └────────────────┘ └──────────┘ └──────────────────┘
```

## Components

### 1. Streamlit Chat UI

- Single-page app: file upload + chat panel + report panel
- Streams tool calls as cards: *"calling profile_run…"* with live status
- Final report: side-by-side metrics, diff viewer, kernel waterfall chart
- Backup mode: replay a cached "golden run" if MI300X unreachable

### 2. Goblin Agent (FastAPI)

- Single endpoint: `POST /audit` — takes uploaded script, returns SSE stream of agent steps
- Session state in-memory only (hackathon, not production)
- Hard cap: **max 8 tool calls per audit** (prevents loops)
- LLM: **Qwen2.5-7B-Instruct** via Hugging Face Inference Providers (`huggingface_hub.AsyncInferenceClient.chat_completion`, OpenAI-shape tool calls). Pluggable via `agent/backends/`; a future `LiveQwenBackend` swaps to a self-hosted vLLM-on-MI300X endpoint with no other code changes.
- System prompt establishes ROCm expert persona + tool-call etiquette + report format

### 3. The Six Tools

#### `parse_config(file_path: str) -> ConfigDict`
Inputs: HF `TrainingArguments` (Python or JSON), raw PyTorch training script, YAML config.
Implementation: AST parsing for `.py` (extract `TrainingArguments(...)` kwargs), `json.load` / `yaml.safe_load` for configs, regex fallback.
Output schema:
```python
{
  "model_name": str,
  "batch_size": int,
  "grad_accum_steps": int,
  "seq_len": int,
  "precision": "fp16" | "bf16" | "fp32",
  "optimizer": str,
  "attention_impl": "sdpa" | "flash" | "eager" | "unknown",
  "gradient_checkpointing": bool,
  "lora_rank": int | None,
  "dataloader_workers": int,
  "lr": float,
  "warmup_steps": int,
  "raw_source": str,
}
```

#### `profile_run(config: ConfigDict, steps: int = 10) -> RunMetrics`
Wraps the user's training command in `torch.profiler` + `rocprofv3`. Runs for N steps after a 2-step warmup. Captures and **summarizes** (never returns raw traces — too big for LLM context).
Output is a `RunMetrics` (the same type returned by `benchmark` — see schemas below):
```python
RunMetrics(
  steps=10,
  tokens_per_sec=...,
  mfu_pct=...,
  hbm_peak_gb=..., hbm_avg_gb=...,
  gpu_util_pct=...,
  top_kernels=[KernelEntry(name=..., pct_time=...), ...],  # top 5
  attention_kernel_loaded=...,
  waste_budget=WasteBudget(...),  # see below
  warnings=[...],
)
```

##### Waste Budget Decomposition

Inside `RunMetrics.waste_budget`, total step time is decomposed into interpretable buckets:

```
T_total = T_useful_gpu
        + T_data_wait        # dataloader / host→device stalls
        + T_host_gap         # CPU→GPU launch latency, eager-mode kernel gaps
        + T_comm_excess      # collectives, all-reduce, RCCL overhead
        + T_memory_headroom  # HBM left idle relative to model needs
        + T_precision_path   # throughput lost vs ideal precision (fp32 where bf16 OK)
        + T_kernel_shape     # GEMM tile mismatch, untuned hipBLASLt/MIOpen
```

This decomposition is the basis for the "where time was lost" chart in the final report and for ranking which rules to apply first. Each rule in the KB declares which bucket it targets (`targets_bucket: data_wait`), so `propose_patch` can avoid stacking redundant fixes for the same bucket.

#### `query_rocm_kb(symptom: str, top_k: int = 5) -> List[Rule]`
Semantic search over the YAML KB. Embeds the symptom string with `sentence-transformers/all-MiniLM-L6-v2`, cosine-similarity against pre-embedded rules.
Output: list of rules sorted by relevance.

#### `propose_patch(config: ConfigDict, rules: List[Rule], metrics: RunMetrics) -> Patch`
Deterministic — no LLM call. Applies rule-to-config transforms (each rule has a `transform` field describing the diff) and computes a per-rule and aggregate uplift estimate from the waste budget.
Output:
```python
Patch(
  new_config=ConfigDict(...),
  diff=str,                            # unified diff
  rationale=[RuleApplication(...)],    # one entry per applied rule
  expected_speedup_low=float,          # conservative end of range
  expected_speedup_high=float,         # optimistic end of range
  confidence=float,                    # 0..1, see formula below
)
```

##### Uplift Estimate

For each applied rule with `targets_bucket=B`, predicted recovery is `recovery_fraction × T_B / T_total`. Aggregate predicted recoverable waste = sum across applied rules (capped per bucket). Predicted speedup = `1 / (1 - recoverable_fraction)`. Reported as a range, not a point estimate.

##### Confidence Score

```
confidence = evidence_coverage × rule_consistency
```

- `evidence_coverage` — fraction of waste-budget buckets that had real measurement (vs. defaulted-to-zero because the profiler couldn't observe it). Lower if `profile_run` produced partial traces.
- `rule_consistency` — 1.0 if all applied rules target distinct buckets and don't have conflicting `transform` fields; lower if rules overlap or conflict.

(The report's `historical_calibration` term is intentionally dropped — we have no historical data in a hackathon timeframe. Document this honestly in the system prompt.)

#### `benchmark(config: ConfigDict, steps: int = 50) -> RunMetrics`
Same pipeline as `profile_run` but runs longer and at full quality. Returns the same `RunMetrics` type. Result is cached by `sha256((canonical_config_json, workload_script_sha, rocm_image_tag, runner_script_sha))` — re-running the same config is free, and the version-tagged hash prevents stale cache hits when the container or runner changes.

#### `compare_runs(before: RunMetrics, after: RunMetrics) -> Report`
Pure function. Builds the side-by-side report dict the UI renders. Includes the side-by-side waste-budget bar chart that visually shows which buckets shrank.

### 4. ROCm Knowledge Base

Single file: `kb/rocm_rules.yaml`. ~25 rules at MVP. Schema:

```yaml
- id: precision.bf16_over_fp16_on_mi300x
  category: precision
  symptom: "fp16 used on MI300X"
  detect:
    config.precision: fp16
  fix:
    config.precision: bf16
  expected_impact: "Same throughput, +numerical stability. Reduces NaN risk."
  rocm_version_min: "6.0"
  citation: "AMD ROCm Best Practices Guide §3.2"
```

Categories:
- `precision` (bf16 over fp16 on CDNA3 matrix cores; FP8 for stretch inference scenarios)
- `attention` (flash-attn ROCm fork via Optimum-AMD, PyTorch SDPA, packed sequences)
- `memory` (batch-size sweetspot for 192 GB HBM3, gradient checkpoint thresholds, activation offloading)
- `kernels` (hipBLASLt hint logging + offline tuning files; MIOpen `MIOPEN_FIND_*` autotune)
- `env_vars` (`HSA_FORCE_FINE_GRAIN_PCIE`, `MIOPEN_FIND_MODE`, `NCCL_MIN_NCHANNELS=112`, NUMA auto-balancing disable)
- `optimizer` (8-bit Adam on ROCm — **warn that bitsandbytes is not officially supported on ROCm**; recommend Optimum-AMD-validated alternatives or CPU-offload optimizers)
- `data` (`num_workers`, `pin_memory=True`, `prefetch_factor`, `persistent_workers=True`, IterableDataset sharding correctness)
- `compile` (`torch.compile` when graph-break evidence is low; `torch_compile=True` in `TrainingArguments`)
- `collectives` (RCCL channel count, one-process-per-GPU vs one-process-many-GPUs)
- `topology` (tensor parallelism within a single XGMI island; prefer TP over EP on single-node)

Rules are hand-curated on Day 1 from ROCm docs + AMD blog posts. **This is the moat.** No LLM-generated rules. Each rule entry includes a `citation` field linking back to a ROCm doc page or AMD blog post — every recommendation in the final report carries this citation.

**Footgun guardrail — `ROCPROFSYS_*` are NOT tuning vars.** ROCm Systems Profiler env vars (`ROCPROFSYS_MODE`, `ROCPROFSYS_USE_SAMPLING`, etc.) configure how the *profiler* observes a run; they do not affect how work is dispatched to the GPU. The KB **excludes** all `ROCPROFSYS_*` vars from optimization rules. Additionally, `parse_config` flags any user config that sets `ROCPROFSYS_*` as if it were a perf knob and surfaces a warning in the report ("These configure the profiler, not the workload — they will not change throughput").

**Workload-validity disclaimer.** Every recommendation is valid for the specific tuple `(workload script, model, GPU=MI300X, ROCm version, framework version, batch/seq pattern)` observed during profiling. The system prompt instructs the agent to state this explicitly when uncertainty is high, and the final report includes a footer line: *"Recommendations validated against MI300X with ROCm <version> and PyTorch <version>. Re-run audit if you change model, hardware, or framework version."*

### 5. Profiling Pipeline

A wrapper script `goblin_runner.sh` invokes the user's training command inside the ROCm container, with:

```bash
# isolate to a single MI300X to keep concurrent benchmark runs sane
export ROCR_VISIBLE_DEVICES=${GOBLIN_GPU_ID:-0}

rocprofv3 --hsa-trace --kernel-trace -o trace.csv -- \
  python -m torch.profiler.scheduler ... \
  python <user_script> --max_steps=10
```

Post-processing (`profile_parser.py`):
- parse `trace.csv` for kernel breakdown
- parse `torch.profiler` JSON for tokens/sec + MFU
- parse `amd-smi` polling output for HBM + util
- merge into single `RunMetrics` with populated `WasteBudget`

### 6. Benchmark Result Cache

`bench_cache/` directory. Key: `sha256((canonical_config_json, workload_script_sha, rocm_image_tag, runner_script_sha))`. Value: full `RunMetrics` plus the unhashed key tuple for debuggability. Lets us re-run the demo without burning cloud time. The Streamlit UI exposes a `--no-cache` toggle for the Day-3 dry-run pass that confirms cached results haven't gone stale.

### 7. Synthetic Corpus (`workloads/synthetic/`)

A directory of pre-generated, deliberately misconfigured fine-tuning runs. Each entry is `{config.py, expected_findings.yaml, cached_metrics.json}`. Generated on Day 1 by ROCm Lead by perturbing the canonical Qwen2.5-7B LoRA workload along single dimensions: `num_workers=0`, FP32 instead of BF16, naive attention, `torch.compile=False`, untuned hipBLASLt, etc.

Two purposes:
1. **Backend Lead can develop on a laptop** — test the agent loop, KB queries, and patch generation against `cached_metrics.json` without touching the GPU.
2. **Demo lane 1 (offline replay)** — judges can audit any synthetic scenario in <30 seconds without live MI300X. If cloud is unreachable on demo day, this is the safety net.

Each scenario's `expected_findings.yaml` lists which rules *should* fire — gives us a regression test for KB recall.

### 8. Demo Lanes

The system runs in one of two modes:

| Lane | Source of metrics | Use |
|---|---|---|
| **Offline replay** | Synthetic corpus + cached `RunMetrics` JSON | Dev, regression testing, demo backup if MI300X unreachable |
| **Live MI300X** | `goblin_runner.sh` → real GPU | Canonical demo, "before/after with real numbers" pitch moment |

The agent loop is identical in both lanes — only the `RunnerProtocol` implementation differs. This is the seam introduced to address the audit's testability finding.

### 9. Secrets & Privacy

Uploaded scripts and logs may contain API tokens, dataset paths, internal URLs. Before any artefact is persisted to disk or sent to the LLM:

1. **Regex redaction pass** in `parse_config` for common patterns: `Bearer [A-Za-z0-9._-]+`, `sk-[A-Za-z0-9]{20,}`, `hf_[A-Za-z0-9]{20,}`, `/home/<user>/`, `s3://`, `wss?://`.
2. **Ephemeral storage** — `bench_cache/` is gitignored; uploaded scripts are deleted at session end.
3. **No model weights or training data required** — we audit configs and metrics only. The system prompt explicitly tells the agent it cannot ask for weights or datasets.

This is a one-screen redaction pass, not a full PII pipeline. Sufficient for hackathon; would need expansion for production.

## Data Flow — One Audit

```
[user uploads script]  ──► [redaction pass]
        │
        ▼
parse_config ──► ConfigDict ────────────────────────────┐
        │                                                │
        ▼                                                │
profile_run ──► RunMetrics + WasteBudget ──► query_rocm_kb │
        │                                       │        │
        │                                       ▼        │
        │                              List[Rule] ──────►│
        │                                                ▼
        │                                       propose_patch (uplift + confidence)
        │                                                │
        │                                                ▼
        │                                       Patch (new ConfigDict + diff)
        │                                                │
        ▼                                                ▼
benchmark(original) ◄──── cache ─────► benchmark(patched)
        │                                                │
        └────────► compare_runs ◄────────────────────────┘
                          │
                          ▼
                       Report ──► UI
                       (waste-budget bar chart + metrics + diff)
```

## Tech Stack (frozen)

| Layer | Choice | Why |
|---|---|---|
| Agent LLM | Qwen2.5-7B-Instruct via HF Inference Providers | Tool calling via OpenAI-compatible API; routes through Together / Fireworks-AI / Nebius (auto). Stretch: self-host on MI300X via vLLM. |
| Backend | Python 3.11 + FastAPI + anthropic SDK | Standard, async-friendly |
| Frontend | Streamlit | Ships in hours, not days |
| Container | `rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.3` | Official AMD image |
| Profiling | torch.profiler + rocprofv3 + amd-smi | Native ROCm tools |
| KB index | sentence-transformers + numpy cosine | No vector DB needed for 25 rules |
| Workload | Qwen2.5-7B + LoRA + alpaca-cleaned | Canonical, reproducible |
| Hardware | MI300X cloud (single GPU, 192 GB HBM) | Hackathon constraint |

## Repo Layout

```
amd-hackathon/
├── brainstorming/          # this folder
├── kb/
│   └── rocm_rules.yaml     # the moat
├── agent/
│   ├── server.py           # FastAPI app
│   ├── loop.py             # agent loop driver
│   ├── prompts.py          # system prompt
│   └── tools/
│       ├── parse_config.py
│       ├── profile_run.py
│       ├── query_rocm_kb.py
│       ├── propose_patch.py
│       ├── benchmark.py
│       └── compare_runs.py
├── runner/
│   ├── goblin_runner.sh    # rocprofv3 wrapper
│   └── profile_parser.py
├── ui/
│   └── app.py              # Streamlit
├── workloads/
│   ├── train_qwen_lora.py    # canonical demo workload
│   └── synthetic/            # pre-generated misconfigured runs + cached metrics
│       ├── 01_no_workers/
│       ├── 02_fp32_default/
│       ├── 03_naive_attention/
│       └── ...
├── bench_cache/            # gitignored
└── docker/
    └── Dockerfile          # rocm/pytorch + our deps
```

## Agent Loop — Pseudocode

The loop is provider-agnostic. It talks to a `Backend` (see `agent/backends/`); today's only concrete is `QwenHFBackend`.

```python
def run_audit(user_script_path: str) -> SSEStream:
    backend = make_backend(system_prompt=SYSTEM_PROMPT)  # QwenHFBackend
    backend.add_user_message(f"Audit this fine-tuning workload: {user_script_path}")

    for step in range(MAX_STEPS):  # MAX_STEPS = 8
        turn = await backend.next_turn(tool_schemas())
        for text in turn.text_blocks:
            yield {"type": "thought", "text": text}
        for tc in turn.tool_calls:
            yield {"type": "tool_call", "name": tc.name, "input": tc.input}
            result = call_tool(tc.name, **tc.input)
            yield {"type": "tool_result", "name": tc.name, "result": result}
            backend.add_tool_result(tc.id, tc.name, result.content, is_error=not result.ok)
        if turn.stop_reason == "end_turn":
            break

    yield {"type": "final_report", "report": extract_final_report(tool_results)}
```

The Backend protocol (`agent/backends/base.py`) is a 3-method contract:
`add_user_message`, `next_turn`, `add_tool_result`. `QwenHFBackend` translates
between this neutral shape and OpenAI-compatible chat-completion calls
through HF Inference Providers.

## Boundaries & Interfaces

- **Agent ↔ Tools:** Each tool is a pure function with typed input/output. No shared state. Tools never call other tools — only the agent orchestrates.
- **Tools ↔ MI300X:** Only `profile_run` and `benchmark` touch the GPU. Both go through `goblin_runner.sh` for consistency.
- **Backend ↔ UI:** SSE stream of typed events (`thought`, `tool_call`, `tool_result`, `final_report`). UI just renders events; no agent logic on frontend.
- **KB ↔ Code:** YAML is the single source of truth. Code consumes; never generates rules at runtime.

## What's Deliberately Excluded (YAGNI)

- ❌ Multi-GPU / distributed training analysis (single MI300X is enough; collective rules in KB still apply when user uploads multi-GPU configs, we just don't benchmark them)
- ❌ Pretraining workloads (fine-tuning only)
- ❌ vLLM / inference serving as a primary mode (Day-4 stretch only; vLLM-specific rules NOT in MVP KB)
- ❌ Live job watching with intervention (offline audit only — much simpler)
- ❌ User accounts / persistence / DB (in-memory session, hackathon)
- ❌ Vector DB (25 rules + numpy is fine)
- ❌ Polars/DuckDB for run storage (Pydantic + JSON is enough at this scale)
- ❌ ML model for uplift prediction (deterministic waste-budget calculus is enough)
- ❌ Auto-applying patches to the user's repo (we output a diff, user applies)
- ❌ Sliders/what-if interactive panel (chat layer answers what-if conversationally; sliders are stretch only)
- ❌ Cost calculator beyond a static "$ per training run" line (stretch goal only)
