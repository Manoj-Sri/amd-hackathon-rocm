"""Agent loop driver — provider-agnostic tool-use loop for one audit.

`run_audit(file_path)` is an async generator that yields `SSEEvent` objects in
the order the UI should render them: thoughts, tool calls, tool results, and
finally either a `final_report` event (extracted from the most recent
successful `compare_runs` tool result) or an `error` event.

The loop itself doesn't know about Anthropic or Hugging Face — it talks to
whichever `Backend` `make_backend()` returns. The backend (Claude or Qwen-HF
today) handles all per-API translation. See `agent/backends/__init__.py`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from agent import tools as tools_module
from agent.backends import Backend, ToolCall, make_backend
from agent.prompts import SYSTEM_PROMPT
from agent.schemas import SSEEvent

MAX_STEPS = 10
"""Hard cap on tool calls per audit. The canonical trajectory is six calls
(parse → profile → query_kb → patch → benchmark×2 → compare). The extra
4 calls of headroom let the model recover from common mistakes (JSON
nesting glitches, retry on ToolResult(ok=False)) without exhausting the
budget before compare_runs. Was 8; bumped after a live run hit a wall when
two misnested-arg benchmark retries ate the slack meant for compare_runs.
"""
MAX_TOKENS = 2048


def _extract_final_report(
    tool_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Walk tool results in reverse and return the most recent successful
    compare_runs payload, or None if there isn't one."""
    for entry in reversed(tool_results):
        if entry["name"] == "compare_runs" and entry["ok"]:
            return entry["result"]
    return None


def _auto_compare(
    tool_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Synthesize a Report from whatever the audit produced when the model
    didn't reach `compare_runs` cleanly. Three recovery tiers, in order of
    fidelity:

    Tier 1 — full data: ≥2 benchmarks + ≥1 propose_patch.
        Treat first benchmark as baseline, last as patched run. Highest
        fidelity since both numbers are real.

    Tier 2 — patch but only one benchmark: ≥1 patch + 1 benchmark.
        Use the single benchmark as baseline. For the "after" side, run
        MockRunner on the patched config to get a deterministic projection.
        Marks the report as projected so the demo is honest about it.

    Tier 3 — no patch ran but we have rules from query_rocm_kb + ≥1 benchmark.
        We *could* deterministically apply propose_patch ourselves here, but
        that's over-reaching. Return None and let the caller surface a
        clean error instead.

    Returns the Report dict, or None when no tier applies.
    """
    benchmarks = [
        e for e in tool_results if e["name"] == "benchmark" and e["ok"]
    ]
    patches = [
        e for e in tool_results if e["name"] == "propose_patch" and e["ok"]
    ]

    # Tier 1: full data path.
    if len(benchmarks) >= 2 and patches:
        latest_patch = patches[-1]["result"]
        before = benchmarks[0]["result"]
        after = benchmarks[-1]["result"]
        return _call_compare_runs(latest_patch, before, after, " (auto-synthesized compare_runs)")

    # Tier 2: patch + 1 benchmark — fill in the patched-side metrics from
    # MockRunner so the demo still produces a Report with a clear note.
    if patches and len(benchmarks) == 1:
        latest_patch = patches[-1]["result"]
        before = benchmarks[0]["result"]
        # Project the patched run via MockRunner. The synthetic corpus has
        # a `02_optimized` scenario the patched config typically matches.
        from agent.schemas import WorkloadConfig
        from runner.protocol import MockRunner

        try:
            patched_cfg = WorkloadConfig.model_validate(latest_patch["new_config"])
            after_metrics = MockRunner().run(patched_cfg, steps=before.get("steps", 50))
            after = after_metrics.model_dump()
        except Exception:
            return None
        return _call_compare_runs(
            latest_patch,
            before,
            after,
            " (auto-synthesized; patched-side projected via MockRunner)",
        )

    return None


def _call_compare_runs(
    patch: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    suffix: str,
) -> dict[str, Any] | None:
    workload_name = (
        patch.get("new_config", {}).get("model_name")
        or "Audited Workload"
    ) + suffix
    result = tools_module.call(
        "compare_runs",
        workload_name=workload_name,
        before=before,
        after=after,
        patch=patch,
    )
    return result.result if result.ok else None


def _safe_json(value: Any) -> str:
    """Serialize a tool result for inclusion in a tool_result content block.

    Falls back to ``str(value)`` if json can't represent the value (e.g. a
    Pydantic model already coerced upstream — shouldn't happen, but defensive).
    """
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


# Trim Tensile-generated rocBLAS / kernel-trace names to a leading identifier.
# 60 chars keeps `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT256x192x64_M…`
# enough to identify the kernel family but drops the 150+ chars of compiled
# tile-config suffixes that don't help the LLM reason and balloon its context.
_LLM_KERNEL_NAME_MAX = 60
# Top-N kernels the LLM gets to see. The UI keeps all five via the
# tool_result SSE event; this cap is purely about LLM context size.
_LLM_TOP_KERNELS_MAX = 3


def _compact_for_llm(value: Any) -> Any:
    """Strip verbose fields from a tool result before sending it to the LLM.

    Why this exists: ``benchmark`` and ``profile_run`` return RunMetrics with
    a ``top_kernels`` list whose entries carry full Tensile-generated rocBLAS
    GEMM kernel names. Each name is ~200 chars; five per result is ~1KB of
    pure noise the LLM doesn't extract anything actionable from. Worse, the
    LLM tends to *regurgitate* them into downstream tool calls
    (``compare_runs`` especially carries before+after RunMetrics, doubling the
    bloat), where they collide with vLLM's ``max_tokens`` ceiling and get
    truncated mid-JSON — surfacing as garbled raw text in the agent
    reasoning panel.

    This compactor only affects the LLM's view. The UI's ``tool_result`` SSE
    event still carries the full ``result.result`` (so judges see the real
    kernel names in the expandable card), and ``tool_results_log`` keeps the
    full data so auto-synth Tier 1 can reconstruct an honest Report.

    Operates on a shallow copy — does not mutate the original.
    """
    if not isinstance(value, dict):
        return value
    out = dict(value)
    kernels = out.get("top_kernels")
    if isinstance(kernels, list):
        compact_kernels = []
        for entry in kernels[:_LLM_TOP_KERNELS_MAX]:
            if not isinstance(entry, dict):
                compact_kernels.append(entry)
                continue
            name = entry.get("name", "")
            if isinstance(name, str) and len(name) > _LLM_KERNEL_NAME_MAX:
                name = name[:_LLM_KERNEL_NAME_MAX] + "…"
            compact_kernels.append({**entry, "name": name})
        out["top_kernels"] = compact_kernels
    return out


async def _drive(backend: Backend) -> AsyncIterator[SSEEvent]:
    """Pure orchestration loop. Backend handles per-API state; we yield events."""
    tool_results_log: list[dict[str, Any]] = []

    for _step in range(MAX_STEPS):
        turn = await backend.next_turn(tools_module.tool_schemas())

        for text in turn.text_blocks:
            if text:
                yield SSEEvent(type="thought", data={"text": text})

        for tc in turn.tool_calls:
            async for ev in _execute_tool_call(backend, tc, tool_results_log):
                yield ev

        if turn.stop_reason == "end_turn":
            break

    report = _extract_final_report(tool_results_log)
    if report is not None:
        yield SSEEvent(type="final_report", data={"report": report})
        return

    # Fallback: the model didn't call compare_runs (or its tool_call landed
    # inside a thinking block where the parser couldn't extract it).
    # Synthesize the report deterministically from the tool log if we have
    # enough material. See _auto_compare for the prerequisites.
    auto = _auto_compare(tool_results_log)
    if auto is not None:
        yield SSEEvent(
            type="thought",
            data={
                "text": (
                    "Note: model did not emit a compare_runs tool call (likely "
                    "left it inside a <think> block). Synthesizing the final "
                    "report from the latest propose_patch + two benchmarks."
                )
            },
        )
        yield SSEEvent(type="final_report", data={"report": auto})
        return

    yield SSEEvent(
        type="error",
        data={
            "message": (
                "Audit completed without producing a final report (and "
                "auto-synthesis fallback couldn't run — need at least one "
                "successful propose_patch and two successful benchmarks)."
            )
        },
    )


async def _execute_tool_call(
    backend: Backend,
    tc: ToolCall,
    tool_results_log: list[dict[str, Any]],
) -> AsyncIterator[SSEEvent]:
    """Yield the tool_call/tool_result event pair and record the outcome."""
    yield SSEEvent(
        type="tool_call",
        data={"id": tc.id, "name": tc.name, "input": tc.input},
    )

    result = tools_module.call(tc.name, **tc.input)

    yield SSEEvent(
        type="tool_result",
        data={
            "id": tc.id,
            "name": tc.name,
            "ok": result.ok,
            "result": result.result,
            "error": result.error,
        },
    )

    tool_results_log.append(
        {
            "id": tc.id,
            "name": tc.name,
            "ok": result.ok,
            "result": result.result,
            "error": result.error,
        }
    )

    # Compact `result.result` before handing it to the LLM — drops verbose
    # rocBLAS Tensile kernel names that bloat context without helping
    # reasoning. The UI's tool_result event (yielded above) and
    # `tool_results_log` (appended above) both keep the full result, so
    # auto-synth and the side-by-side metrics table are unaffected.
    content = (
        _safe_json(_compact_for_llm(result.result))
        if result.ok
        else (result.error or "tool failed")
    )
    backend.add_tool_result(
        tool_call_id=tc.id,
        name=tc.name,
        content=content,
        is_error=not result.ok,
    )


async def run_audit(file_path: str) -> AsyncIterator[SSEEvent]:
    """Run one audit and yield SSE events as they happen.

    Selects the LLM backend from the `GOBLIN_AGENT_BACKEND` env var (defaults
    to `claude`; `qwen` routes through HF Inference Providers). On any
    backend or loop exception, yields a single `error` SSE event and stops.
    """
    try:
        backend = make_backend(system_prompt=SYSTEM_PROMPT, max_tokens=MAX_TOKENS)
    except Exception as exc:
        yield SSEEvent(type="error", data={"message": str(exc)})
        return

    backend.add_user_message(f"Audit this fine-tuning workload: {file_path}")

    try:
        async for ev in _drive(backend):
            yield ev
    except Exception as exc:
        yield SSEEvent(type="error", data={"message": str(exc)})
