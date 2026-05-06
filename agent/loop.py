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

MAX_STEPS = 8
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


def _safe_json(value: Any) -> str:
    """Serialize a tool result for inclusion in a tool_result content block.

    Falls back to ``str(value)`` if json can't represent the value (e.g. a
    Pydantic model already coerced upstream — shouldn't happen, but defensive).
    """
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


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
    else:
        yield SSEEvent(
            type="error",
            data={"message": "Audit completed without producing a final report"},
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

    content = (
        _safe_json(result.result) if result.ok else (result.error or "tool failed")
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
