"""Agent loop driver — the Claude tool-use loop for one audit.

`run_audit(file_path)` is an async generator that yields `SSEEvent` objects in
the order the UI should render them: thoughts, tool calls, tool results, and
finally either a `final_report` event (extracted from the most recent
successful `compare_runs` tool result) or an `error` event.

This file owns the protocol between the agent and the rest of the system. The
tools live behind `agent.tools.call(name, **input)` and return a `ToolResult`
that we forward into the conversation as a `tool_result` content block per the
Anthropic tool-use protocol. The Anthropic client itself is constructed lazily
so the FastAPI app can start up without an API key (offline mode).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from agent import tools as tools_module
from agent.prompts import SYSTEM_PROMPT
from agent.schemas import SSEEvent

MAX_STEPS = 8
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048


def _build_client() -> AsyncAnthropic:
    """Construct the Anthropic client.

    Raises RuntimeError if ANTHROPIC_API_KEY is missing — the server catches
    this and converts it into an SSE error event so the process keeps running
    in offline mode.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; agent loop cannot reach Claude. "
            "Set the env var or use the offline replay UI lane."
        )
    return AsyncAnthropic(api_key=api_key)


def _content_block_to_dict(block: Any) -> dict[str, Any]:
    """Re-serialize a Claude content block back into the dict shape the API
    expects when we echo it inside an `assistant` message in the next request.
    """
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": block.text}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    # Fallback: best-effort dump for forward compatibility (e.g. thinking blocks).
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return dict(block) if isinstance(block, dict) else {"type": btype or "unknown"}


def _extract_final_report(tool_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Walk tool results in reverse and return the most recent successful
    compare_runs payload, or None if there isn't one."""
    for entry in reversed(tool_results):
        if entry["name"] == "compare_runs" and entry["ok"]:
            return entry["result"]
    return None


async def run_audit(file_path: str) -> AsyncIterator[SSEEvent]:
    """Run one audit and yield SSE events as they happen.

    The loop runs at most MAX_STEPS turns. Each turn:
      * sends the conversation to Claude with the system prompt + tool schemas,
      * yields a `thought` event for every text block in the response,
      * for each tool_use block: yields `tool_call`, dispatches via
        `agent.tools.call`, yields `tool_result`, and appends the tool result
        to the conversation per Claude's tool-use protocol,
      * stops when stop_reason == "end_turn".

    After the loop, yields a `final_report` event sourced from the last
    successful compare_runs result, or an `error` event if none exists.
    """
    try:
        client = _build_client()
    except Exception as exc:
        yield SSEEvent(type="error", data={"message": str(exc)})
        return

    conversation: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": f"Audit this fine-tuning workload: {file_path}",
        }
    ]
    tool_results_log: list[dict[str, Any]] = []

    try:
        for _step in range(MAX_STEPS):
            response = await client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=tools_module.claude_tool_schemas(),
                messages=conversation,
            )

            # Echo the assistant turn back into the conversation so the next
            # request has the tool_use blocks to pair with our tool_result blocks.
            assistant_blocks = [_content_block_to_dict(b) for b in response.content]
            conversation.append({"role": "assistant", "content": assistant_blocks})

            tool_result_blocks: list[dict[str, Any]] = []
            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "text" and block.text:
                    yield SSEEvent(type="thought", data={"text": block.text})
                elif btype == "tool_use":
                    yield SSEEvent(
                        type="tool_call",
                        data={
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        },
                    )

                    tool_input = block.input if isinstance(block.input, dict) else {}
                    result = tools_module.call(block.name, **tool_input)

                    yield SSEEvent(
                        type="tool_result",
                        data={
                            "id": block.id,
                            "name": block.name,
                            "ok": result.ok,
                            "result": result.result,
                            "error": result.error,
                        },
                    )
                    tool_results_log.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "ok": result.ok,
                            "result": result.result,
                            "error": result.error,
                        }
                    )
                    # Claude needs a tool_result block (string content is fine
                    # — tools return JSON-serializable dicts via Pydantic, but
                    # the API just needs a string). On error we surface the
                    # error message so the agent can adapt.
                    if result.ok:
                        content_str = _safe_json(result.result)
                    else:
                        content_str = f"ERROR: {result.error or 'tool failed'}"
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content_str,
                            "is_error": not result.ok,
                        }
                    )

            if tool_result_blocks:
                conversation.append({"role": "user", "content": tool_result_blocks})

            if response.stop_reason == "end_turn":
                break

        report = _extract_final_report(tool_results_log)
        if report is not None:
            yield SSEEvent(type="final_report", data={"report": report})
        else:
            yield SSEEvent(
                type="error",
                data={"message": "Audit completed without producing a final report"},
            )
    except Exception as exc:
        yield SSEEvent(type="error", data={"message": str(exc)})
        return


def _safe_json(value: Any) -> str:
    """Serialize a tool result for inclusion in a tool_result content block.

    Falls back to ``str(value)`` if json can't represent the value (e.g. a
    Pydantic model already coerced upstream — shouldn't happen, but defensive).
    """
    import json

    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)
