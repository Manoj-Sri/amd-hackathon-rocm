# GPU Goblin — Goals & Implementation Plan

## North Star

Win the AMD hackathon (**Track 1: AI Agents & Agentic Workflows**) by demonstrating a **real, reproducible 2×+ throughput improvement on a Qwen2.5-7B LoRA fine-tune on MI300X**, driven end-to-end by a Qwen-powered tool-using agent. Hits the Qwen Technology Partner challenge with end-to-end Qwen-on-AMD; deploys as a Hugging Face Space within the hackathon HF Organization.

Everything else is in service of that demo.

## Success Criteria

| # | Criterion | Measurable As |
|---|---|---|
| 1 | Real MI300X speedup | ≥ 2.0× tokens/sec, before vs. after, on the canonical workload |
| 2 | Agent is actually agentic | ≥ 4 tool calls per audit, visible in UI |
| 3 | AMD differentiation | 100% of recommendations cite a ROCm-specific KB rule |
| 4 | Uplift estimate honesty | Predicted speedup range (low–high) brackets the measured speedup on ≥ 8 of 10 synthetic corpus scenarios |
| 5 | Offline parse coverage | ≥ 90% of synthetic corpus parsed without manual edits |
| 6 | End-to-end reliability | Demo runs cleanly 5× in a row without manual intervention |
| 7 | Pitch quality | 3-min demo + 1-min architecture + 1-min impact, rehearsed |
| 8 | Backup safety | Offline-replay lane works fully when MI300X disconnected |

## Mapping to Judging Criteria

| Hackathon criterion | Where we score |
|---|---|
| **Application of Technology** | Tool-using agent loop on MI300X; rocprofv3 + torch.profiler integration; deterministic uplift via waste-budget calculus; ROCm-specific KB |
| **Presentation** | Live before/after benchmark on real MI300X; waste-budget bar chart; chat-based "why?" interaction; 90-sec backup video |
| **Business Value** | Each audit reports `$ saved per training run` and `time saved per epoch`, framed against $1.99/hr public MI300X pricing reference |
| **Originality** | "AI for AI builders" — meta-positioning. Real benchmarks, not just LLM advice. Waste-budget decomposition is novel framing |

## Team Roles (3 people × 4 days)

### ROCm / ML Lead
- Day 1: MI300X cloud env, ROCm container, baseline workload (Qwen2.5-7B LoRA on alpaca)
- Day 1-2: Hand-curate 20-25 KB rules from ROCm docs + AMD blog
- Day 2: `profile_run` + `benchmark` tools (rocprofv3 wrapper, parser)
- Day 3: Validate end-to-end speedup on canonical demo, generate cached results
- **Owns:** anything that touches the GPU

### Agent / Backend Lead
- Day 1: FastAPI skeleton, tool schemas, Qwen-via-HF tool-use plumbing
- Day 1-2: `parse_config`, `propose_patch`, `query_rocm_kb`, `compare_runs` tools
- Day 2: Full agent loop, system prompt, SSE streaming
- Day 3: Hardening, error handling, max-steps cap, fallback behaviors
- **Owns:** agent reasoning quality + tool wiring

### Frontend / Demo Lead
- Day 1: Streamlit skeleton, file upload, message log
- Day 2: Tool-call cards (live status), report renderer, diff viewer
- Day 3: Demo polish, charts, golden-run replay mode
- Day 4: Pitch deck, 90-second backup video, dry runs
- **Owns:** what the judges actually see

Roles overlap on integration days — pair-program when blocked.

## Day-by-Day Plan

### Day 1 (May 5–6) — Foundations

**ROCm Lead**
- [ ] Provision MI300X cloud instance via AMD Developer Cloud ($100 credits), SSH access, persistent storage
- [ ] Pull `rocm/pytorch:rocm6.1_*` image, verify GPU visible inside container
- [ ] Run baseline `train_qwen_lora.py` (batch=4, fp16, naive attention, alpaca, 100 steps)
- [ ] Capture baseline tokens/sec, MFU, HBM peak — *this is our "before"*
- [ ] Generate **synthetic corpus** — 5-8 misconfigured variants of the canonical workload (FP32, num_workers=0, naive attention, etc.) with cached `RunMetrics` JSON for each
- [ ] Start drafting KB rules (target 10 rules by EOD), each tagged with `targets_bucket` matching the waste-budget decomposition

**Backend Lead**
- [ ] Repo scaffold per architecture.md layout
- [ ] `pip install fastapi anthropic sentence-transformers pyyaml pydantic`
- [ ] Define `agent/schemas.py` with `RunMetrics`, `WasteBudget`, `ConfigDict`, `Patch`, `Rule`, `Report` as pydantic models — **Day-1 priority** (blocks all tools)
- [ ] Define `RunnerProtocol` interface; build `MockRunner` that loads cached metrics from `workloads/synthetic/` (lets backend dev without MI300X)
- [ ] FastAPI `POST /audit` skeleton with SSE
- [ ] `parse_config` tool — handle HF `TrainingArguments` first; include regex redaction pass for tokens/paths
- [ ] Qwen tool-use hello-world (one tool, one round-trip via HF Inference Providers)

**Frontend Lead**
- [ ] Streamlit skeleton with file upload + chat panel
- [ ] Hardcoded "fake audit" — render canned tool calls + report
- [ ] Pick chart library (Altair recommended for Streamlit)

**Day 1 Exit Criteria**
- Baseline benchmark numbers in hand
- Synthetic corpus has ≥ 3 cached scenarios (Backend Lead can now dev without GPU)
- Schemas (`RunMetrics`, `WasteBudget`, etc.) frozen
- `RunnerProtocol` + `MockRunner` working end-to-end
- Backend can call Qwen with one tool via HF Inference Providers
- UI renders a fake audit
- 10 KB rules drafted

### Day 2 (May 6–7) — Core Build

**ROCm Lead**
- [ ] Finish KB to 20-25 rules; hand-tag categories + `targets_bucket`; pre-embed with sentence-transformers
- [ ] Include the high-impact MI300X rules: BF16-over-FP16, AITER-flash-attn-via-Optimum-AMD, `NCCL_MIN_NCHANNELS=112`, NUMA disable, one-process-per-GPU, hipBLASLt hint logging, MIOpen `MIOPEN_FIND_*`, **bitsandbytes-not-supported-on-ROCm warning**, `num_workers`/`pin_memory`/`prefetch_factor`/`persistent_workers`
- [ ] `profile_run` tool: rocprofv3 wrapper + torch.profiler + amd-smi → `RunMetrics` with `WasteBudget`
- [ ] `benchmark` tool: same pipeline, longer run, with version-tagged cache
- [ ] Validate on baseline workload: profile output makes sense, MFU is plausible, waste budget sums to ~T_total

**Backend Lead**
- [ ] All 6 tools wired and individually tested with fixtures from synthetic corpus
- [ ] Full agent loop with max-steps cap, SSE event types finalized, error envelope per tool (`{ok, result, error}`)
- [ ] System prompt iterated against test workloads — include MI300X hardware specs (304 CUs, 192 GB HBM3, ~5.3 TB/s, FP8 native) so the agent reasons quantitatively
- [ ] `propose_patch` deterministic transformer + uplift estimator (waste-budget × bucket recovery) + confidence formula

**Frontend Lead**
- [ ] Live tool-call cards consuming real SSE stream
- [ ] Final report layout: side-by-side metrics + diff + kernel chart + **waste-budget bar chart** (where time was lost, before vs after)
- [ ] Lane toggle: `Live MI300X` vs `Offline replay (synthetic corpus)` — judges can pick either
- [ ] First end-to-end run through the actual backend

**Day 2 Exit Criteria**
- Real audit runs end-to-end on a toy workload
- Profile + benchmark return real MI300X numbers
- KB has 20+ rules, embeddings pre-computed
- UI streams real agent activity

### Day 3 (May 7–8) — Demo Day Prep

**All hands**
- [ ] Run canonical demo (Qwen2.5-7B LoRA) end-to-end → confirm ≥ 2× speedup
- [ ] Cache the demo benchmark results — don't burn cloud time on every rehearsal
- [ ] Build "golden run" replay mode (read cached SSE events, replay timing)
- [ ] Validate uplift accuracy on synthetic corpus: predicted range should bracket measured speedup on ≥ 8 of 10 scenarios
- [ ] Polish system prompt for demo-friendly narration ("I'll start by…")
- [ ] Tighten error handling — agent should never panic in front of judges; tool failures degrade gracefully with `{ok:false}` envelopes
- [ ] Run with `--no-cache` once to verify cached results aren't masking real bugs
- [ ] 5× clean dry runs, fix anything flaky

**Day 3 Exit Criteria**
- Canonical demo: cleanly runs in ≤ 4 minutes, ≥ 2× speedup, no manual fixes
- Cached results enable offline replay
- Offline-replay lane works fully when MI300X is disconnected (proven by unplugging cloud)
- Golden-run video recorded as backup

### Day 4 (May 8–9) — Pitch & Stretch

**Frontend Lead**
- [ ] Pitch deck (5 slides): problem, agent loop diagram, demo (live), KB rules sample, impact + ask
- [ ] Cover image for the submission listing
- [ ] Final 90-second backup video
- [ ] Submission form filled (title, short + long description, tags), repo public, README crisp
- [ ] **Build-in-Public bonus track:** at least one public update post (X/LinkedIn/Discord) showing the agent's first audit; one ROCm/Optimum-AMD feedback note based on what was rough during build

**ROCm + Backend Leads (in parallel, optional stretch)**
- [ ] vLLM inference workload as second demo (only if rock-solid on Day 3)
- [ ] Cost calculator: `$ saved per training run` line, anchored on $1.99/hr public MI300X reference
- [ ] What-if slider panel for batch / precision / attention (chat already does this conversationally, sliders are visual icing)
- [ ] Stretch dream: self-host Qwen on MI300X via vLLM (replacing the HF-Inference-Providers path) — closes the loop entirely on AMD silicon. Mention in pitch even if not running live.

**Day 4 Exit Criteria**
- Submission complete by deadline
- 5+ rehearsed dry runs of the pitch
- Cover image, video, slides, repo all linked from submission form
- Backup video on USB stick, in cloud, on phone

## Scope Discipline (YAGNI)

If we're behind schedule, **cut in this order**:

1. ❌ All stretch goals (vLLM, cost calc, self-hosted agent)
2. ❌ Live tool-call UI animations — static cards work
3. ❌ Some KB categories (keep precision/attention/memory; drop env_vars/collectives)
4. ❌ Multi-file script parsing — single-file only
5. ❌ Dynamic batch-size search — hardcode the recommended batch for demo workload

**Never cut:**
- Real MI300X benchmark (that's the entire pitch)
- Tool-using agent loop (that's the track fit)
- ROCm-specific KB citations (that's the differentiation)

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MI300X cloud quota burns out | Medium | High | Cache every benchmark, dev with 10-step traces, full benchmarks only for canonical demo |
| `rocprofv3` flaky / version mismatch | Medium | High | Always run `torch.profiler` in parallel as fallback. Pre-record golden trace. Use `rocprofv3` not deprecated `rocprof`/`rocprofv2` |
| MI300X cloud unreachable on demo day | Low | Critical | **Offline-replay lane** (synthetic corpus + cached metrics) provides full demo without cloud — judges can't tell the difference for the agent-reasoning portion |
| Uploaded scripts contain user secrets / tokens | Medium | Medium | Regex redaction pass in `parse_config` before any persistence or LLM call; ephemeral storage; explicit "we never store weights or datasets" line in UI |
| Agent loops infinitely | Low | Medium | Hard cap of 8 tool calls, fall back to "best effort" report after cap |
| Recommendations are generic, not ROCm-specific | Medium | Critical | Hand-curate KB on Day 1 *before* wiring agent — KB is the moat |
| Demo crashes during pitch | Low | Critical | Pre-recorded video backup. Golden-run replay mode. USB + cloud + phone copies |
| Qwen2.5-7B doesn't fit a 12-batch on MI300X | Low | Medium | Have a fallback config in hand (batch=8 + grad_accum=2) |
| LoRA on alpaca too easy — speedup looks staged | Low | High | Measure on a non-trivial seq_len (1024+), include MFU not just tokens/sec, show kernel breakdown to prove it's real |
| HF Inference Provider rate limit / outage during demo | Low | Medium | Offline-replay UI lane plays cached_audit.json without any backend; pre-cache a full recorded session; have a backup HF token; `provider="auto"` already routes around individual provider outages |
| Team member unavailable (illness, etc.) | Low | High | Pair on critical path (agent loop, KB) so no single point of failure |

## Definition of Done — MVP

GPU Goblin is "done enough to ship" when **all** of these are true:

- ✅ A judge can upload `train_qwen_lora.py` (we provide it) and get a real audit
- ✅ Agent makes ≥ 4 visible tool calls
- ✅ Final report shows ≥ 2× tokens/sec, real numbers from MI300X
- ✅ Every recommendation in the report cites a ROCm KB rule by ID
- ✅ User can ask follow-up questions in chat ("why bf16?") and get cited answers
- ✅ Full audit completes in ≤ 4 minutes
- ✅ 5 consecutive dry runs succeed without manual intervention
- ✅ Backup video and cached replay both work

## Stretch Definition of Done

- 🎯 Second canonical workload (vLLM inference) audited end-to-end
- 🎯 Cost calculator: "you save $X per training run, $Y per epoch"
- 🎯 Agent backed by self-hosted Qwen via vLLM on the same MI300X (the ultimate AMD story — replaces today's HF Inference Providers path with on-cluster serving)

## Compute Budget — AMD Developer Cloud Credits

Eligible participants get **$100 in AMD Developer Cloud credits**. Public reference price for MI300X is around $1.99/hr (single VM) or $3.39/hr (8× bare metal). Plan accordingly:

| Activity | GPU-hours | Notes |
|---|---|---|
| Day-1 baseline + synthetic corpus generation | 4–6 | One-shot, results cached |
| Day-2 KB validation runs | 2–3 | Sanity-check the rules fire on synthetic scenarios |
| Day-3 canonical demo dry runs (cached) | 2–4 | Cache hits after the first run |
| Day-3 `--no-cache` cold validation | 1 | Confirms nothing's stale |
| Day-4 final dry runs + record video | 2–3 | Lock the demo |
| **Total estimate** | **~12–17 hrs** | Well within $100 even at bare-metal rates |

Backend Lead spends zero MI300X time after Day 1 — develops against synthetic corpus + `MockRunner`.

## Submission Checklist

- [ ] Public GitHub repo with clear README + architecture diagram + setup instructions
- [ ] 90-second demo video (live agent run, real MI300X numbers)
- [ ] Pitch deck (PDF or slides URL)
- [ ] Cover image (project listing visual)
- [ ] Architecture diagram (PNG)
- [ ] Sample audit report PDF (one canonical run, before/after, including waste-budget chart)
- [ ] Short description (1-2 sentences) + long description (paragraph) + technology/category tags
- [ ] At least one Build-in-Public post + one ROCm/Optimum-AMD feedback note (bonus track)
- [ ] Hackathon submission form filled
- [ ] Team member credits + contact
