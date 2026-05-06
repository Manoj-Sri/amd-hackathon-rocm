# GPU Goblin

> An AI agent that hunts wasted compute on AMD MI300X.

GPU Goblin profiles a fine-tuning run, diagnoses inefficiency against a curated ROCm knowledge base, recommends MI300X-specific fixes, and re-benchmarks to prove the speedup with real numbers.

See [`brainstorming/idea.md`](brainstorming/idea.md), [`brainstorming/architecture.md`](brainstorming/architecture.md), and [`brainstorming/goals.md`](brainstorming/goals.md).

## Quick Start

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn agent.server:app --reload --port 8000
streamlit run ui/app.py
```

## Repo Layout

```
agent/        # Agent loop, FastAPI server, tool implementations
  schemas.py  # Shared pydantic models (RunMetrics, ConfigDict, ...)
  tools/      # 6 tools the agent can call
runner/       # GPU runner (rocprofv3 wrapper) + FakeRunner
kb/           # ROCm knowledge base (YAML rules, the moat)
ui/           # Streamlit chat UI
workloads/    # Canonical demo + synthetic corpus
tests/        # Pytest suite + fixtures
```

## Development

The agent loop is testable on a laptop without an MI300X via the `FakeRunner` and synthetic corpus in `workloads/synthetic/`. Real benchmarks require ROCm + MI300X.
