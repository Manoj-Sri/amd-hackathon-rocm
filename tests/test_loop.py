"""Tests for the agent loop driver.

We never touch the real Anthropic API. Every test injects a fake
`AsyncAnthropic` client whose `messages.create` returns a queued sequence of
fake responses. Tools are also stubbed so we can drive specific control-flow
paths (success, ok=False, missing-final-report, mid-run exception).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest

from agent import loop as loop_module
from agent.schemas import SSEEvent, ToolResult


# ---------------------------------------------------------------------------
# Fakes for Claude responses
# ---------------------------------------------------------------------------


@dataclass
class FakeText:
    text: str
    type: str = "text"


@dataclass
class FakeToolUse:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class FakeResponse:
    content: list[Any]
    stop_reason: str  # "tool_use" or "end_turn"


class FakeMessages:
    def __init__(self, queued: list[FakeResponse]) -> None:
        self._queued = list(queued)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeResponse:
        # Snapshot the messages list — the loop mutates it in place across
        # turns, so without copy(), every recorded call would point at the
        # same final list.
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            import copy

            snapshot["messages"] = copy.deepcopy(snapshot["messages"])
        self.calls.append(snapshot)
        if not self._queued:
            raise AssertionError(
                "FakeMessages exhausted — loop made more API calls than expected"
            )
        return self._queued.pop(0)


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.messages = FakeMessages(responses)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_fake_client(monkeypatch, responses: list[FakeResponse]) -> FakeClient:
    """Replace `_build_client` with one that returns our scripted FakeClient."""
    # The loop's _build_client requires ANTHROPIC_API_KEY to be set; satisfy
    # that even though we never use it.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    fake = FakeClient(responses)
    monkeypatch.setattr(loop_module, "_build_client", lambda: fake)
    return fake


def _install_fake_tools(monkeypatch, tool_responses: dict[str, ToolResult]) -> list[str]:
    """Replace `tools_module.call` with a stub returning preset ToolResults.

    Returns a list which records the order tools were invoked.
    """
    invoked: list[str] = []

    def fake_call(name: str, **_kwargs: Any) -> ToolResult:
        invoked.append(name)
        if name in tool_responses:
            return tool_responses[name]
        return ToolResult(ok=False, error=f"no fake registered for {name}")

    # Stub claude_tool_schemas so the loop doesn't depend on real tool imports.
    monkeypatch.setattr(loop_module.tools_module, "call", fake_call)
    monkeypatch.setattr(loop_module.tools_module, "claude_tool_schemas", lambda: [])
    return invoked


async def _collect(stream) -> list[SSEEvent]:
    out: list[SSEEvent] = []
    async for event in stream:
        out.append(event)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emits_thought_then_tool_call_then_tool_result(monkeypatch) -> None:
    responses = [
        FakeResponse(
            content=[
                FakeText(text="I'll start by parsing the config."),
                FakeToolUse(id="tu_1", name="parse_config", input={"file_path": "/x.py"}),
            ],
            stop_reason="tool_use",
        ),
        FakeResponse(content=[FakeText(text="Done.")], stop_reason="end_turn"),
    ]
    _install_fake_client(monkeypatch, responses)
    invoked = _install_fake_tools(
        monkeypatch,
        {"parse_config": ToolResult(ok=True, result={"model_name": "x"})},
    )

    events = await _collect(loop_module.run_audit("/x.py"))
    types = [e.type for e in events]

    # Thought before tool_call before tool_result; loop ends with error event
    # because no compare_runs ran.
    assert types[0] == "thought"
    assert types[1] == "tool_call"
    assert types[2] == "tool_result"
    # Final event is "error" because we never ran compare_runs.
    assert types[-1] == "error"
    assert "without producing a final report" in events[-1].data["message"]
    assert invoked == ["parse_config"]

    # Tool_call carries id/name/input; tool_result mirrors that plus ok/result/error.
    assert events[1].data == {
        "id": "tu_1",
        "name": "parse_config",
        "input": {"file_path": "/x.py"},
    }
    assert events[2].data["ok"] is True
    assert events[2].data["result"] == {"model_name": "x"}
    assert events[2].data["error"] is None


@pytest.mark.asyncio
async def test_final_report_extracted_from_compare_runs(monkeypatch) -> None:
    fake_report = {"workload_name": "test", "speedup_actual": 2.0}
    responses = [
        FakeResponse(
            content=[
                FakeText(text="Wrapping up."),
                FakeToolUse(
                    id="tu_compare",
                    name="compare_runs",
                    input={"workload_name": "t", "before": {}, "after": {}, "patch": {}},
                ),
            ],
            stop_reason="end_turn",
        ),
    ]
    _install_fake_client(monkeypatch, responses)
    _install_fake_tools(monkeypatch, {"compare_runs": ToolResult(ok=True, result=fake_report)})

    events = await _collect(loop_module.run_audit("/x.py"))

    assert events[-1].type == "final_report"
    assert events[-1].data["report"] == fake_report


@pytest.mark.asyncio
async def test_tool_error_passes_through_does_not_crash(monkeypatch) -> None:
    responses = [
        FakeResponse(
            content=[
                FakeText(text="Trying parse."),
                FakeToolUse(id="tu_1", name="parse_config", input={"file_path": "/bogus"}),
            ],
            stop_reason="tool_use",
        ),
        FakeResponse(content=[FakeText(text="Giving up.")], stop_reason="end_turn"),
    ]
    _install_fake_client(monkeypatch, responses)
    _install_fake_tools(
        monkeypatch,
        {"parse_config": ToolResult(ok=False, error="file not found")},
    )

    events = await _collect(loop_module.run_audit("/bogus"))
    tool_result_events = [e for e in events if e.type == "tool_result"]
    assert len(tool_result_events) == 1
    assert tool_result_events[0].data["ok"] is False
    assert tool_result_events[0].data["error"] == "file not found"
    # The loop kept running rather than bailing out.
    assert events[-1].type == "error"  # no compare_runs ⇒ "no final report" error


@pytest.mark.asyncio
async def test_missing_api_key_yields_error_event(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Don't replace _build_client — we want the real one to raise.
    events = await _collect(loop_module.run_audit("/x.py"))
    assert len(events) == 1
    assert events[0].type == "error"
    assert "ANTHROPIC_API_KEY" in events[0].data["message"]


@pytest.mark.asyncio
async def test_mid_loop_exception_is_caught(monkeypatch) -> None:
    class BoomMessages:
        async def create(self, **_kwargs: Any) -> Any:
            raise RuntimeError("boom")

    class BoomClient:
        messages = BoomMessages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(loop_module, "_build_client", lambda: BoomClient())
    monkeypatch.setattr(loop_module.tools_module, "claude_tool_schemas", lambda: [])

    events = await _collect(loop_module.run_audit("/x.py"))
    assert events[-1].type == "error"
    assert "boom" in events[-1].data["message"]


@pytest.mark.asyncio
async def test_loop_caps_at_max_steps(monkeypatch) -> None:
    """Even if the model never says end_turn, we bail after MAX_STEPS."""
    # Build MAX_STEPS responses, all stop_reason=tool_use, all calling one tool.
    responses = [
        FakeResponse(
            content=[
                FakeText(text=f"step {i}"),
                FakeToolUse(id=f"tu_{i}", name="parse_config", input={"file_path": "/x.py"}),
            ],
            stop_reason="tool_use",
        )
        for i in range(loop_module.MAX_STEPS + 2)  # extra so we'd overrun if uncapped
    ]
    fake = _install_fake_client(monkeypatch, responses)
    _install_fake_tools(monkeypatch, {"parse_config": ToolResult(ok=True, result={})})

    events = await _collect(loop_module.run_audit("/x.py"))
    # API was called exactly MAX_STEPS times.
    assert len(fake.messages.calls) == loop_module.MAX_STEPS
    # Last event is the "no final report" error (not crash).
    assert events[-1].type == "error"


@pytest.mark.asyncio
async def test_tool_use_id_is_preserved_in_conversation_for_next_turn(monkeypatch) -> None:
    """Critical for Claude's tool-use protocol: the tool_use id must be echoed
    back as `tool_use_id` in the next user message's tool_result block.
    """
    responses = [
        FakeResponse(
            content=[
                FakeText(text="parse"),
                FakeToolUse(id="tu_abc", name="parse_config", input={"file_path": "/x"}),
            ],
            stop_reason="tool_use",
        ),
        FakeResponse(content=[FakeText(text="done")], stop_reason="end_turn"),
    ]
    fake = _install_fake_client(monkeypatch, responses)
    _install_fake_tools(monkeypatch, {"parse_config": ToolResult(ok=True, result={"a": 1})})

    await _collect(loop_module.run_audit("/x"))

    # Second API call should include the tool_result block referencing tu_abc.
    second_call = fake.messages.calls[1]
    last_msg = second_call["messages"][-1]
    assert last_msg["role"] == "user"
    blocks = last_msg["content"]
    assert any(
        b.get("type") == "tool_result" and b.get("tool_use_id") == "tu_abc"
        for b in blocks
    )
