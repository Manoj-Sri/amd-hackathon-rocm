"""Tests for the agent loop driver.

We never hit a real LLM. The loop talks to a `Backend` (see
`agent/backends/`); each test injects a `FakeBackend` whose `next_turn`
returns a queued sequence of scripted `AgentTurn` objects. Tools are
stubbed so we can drive specific control-flow paths.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent import loop as loop_module
from agent.backends.base import AgentTurn, Backend, ToolCall
from agent.schemas import SSEEvent, ToolResult


# ---------------------------------------------------------------------------
# Fake backend
# ---------------------------------------------------------------------------


class FakeBackend(Backend):
    """A scripted Backend for testing the loop in isolation.

    Each test queues a list of `AgentTurn`s; calling `next_turn` pops the
    next one. We also record every tool result the loop hands back so tests
    can assert that error / id / content were threaded through correctly.
    """

    name = "fake"

    def __init__(
        self,
        scripted_turns: list[AgentTurn] | None = None,
        next_turn_raises: BaseException | None = None,
    ) -> None:
        self._scripted = list(scripted_turns or [])
        self._raise_on_next = next_turn_raises
        self.user_messages: list[str] = []
        self.tool_results: list[dict[str, Any]] = []
        self.turn_count = 0

    def add_user_message(self, content: str) -> None:
        self.user_messages.append(content)

    def add_tool_result(
        self,
        tool_call_id: str,
        name: str,
        content: str,
        is_error: bool,
    ) -> None:
        self.tool_results.append(
            {
                "id": tool_call_id,
                "name": name,
                "content": content,
                "is_error": is_error,
            }
        )

    async def next_turn(self, tool_schemas: list[dict[str, Any]]) -> AgentTurn:
        self.turn_count += 1
        if self._raise_on_next is not None:
            exc = self._raise_on_next
            self._raise_on_next = None
            raise exc
        if not self._scripted:
            raise AssertionError(
                "FakeBackend exhausted — loop made more turns than expected"
            )
        return self._scripted.pop(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_backend(monkeypatch, backend: Backend) -> Backend:
    """Replace `make_backend` so the loop sees our fake."""
    monkeypatch.setattr(loop_module, "make_backend", lambda **_kwargs: backend)
    return backend


def _install_make_backend_raises(monkeypatch, exc: BaseException) -> None:
    def boom(**_kwargs: Any) -> Backend:
        raise exc

    monkeypatch.setattr(loop_module, "make_backend", boom)


def _install_fake_tools(
    monkeypatch, tool_responses: dict[str, ToolResult]
) -> list[str]:
    """Replace `tools_module.call` and `tool_schemas`. Returns a list
    that records the order tools were invoked.
    """
    invoked: list[str] = []

    def fake_call(name: str, **_kwargs: Any) -> ToolResult:
        invoked.append(name)
        return tool_responses.get(
            name, ToolResult(ok=False, error=f"no fake registered for {name}")
        )

    monkeypatch.setattr(loop_module.tools_module, "call", fake_call)
    monkeypatch.setattr(loop_module.tools_module, "tool_schemas", lambda: [])
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
    backend = FakeBackend(
        scripted_turns=[
            AgentTurn(
                text_blocks=["I'll start by parsing the config."],
                tool_calls=[
                    ToolCall(id="tu_1", name="parse_config", input={"file_path": "/x.py"})
                ],
                stop_reason="tool_use",
            ),
            AgentTurn(text_blocks=["Done."], tool_calls=[], stop_reason="end_turn"),
        ]
    )
    _install_backend(monkeypatch, backend)
    invoked = _install_fake_tools(
        monkeypatch,
        {"parse_config": ToolResult(ok=True, result={"model_name": "x"})},
    )

    events = await _collect(loop_module.run_audit("/x.py"))
    types = [e.type for e in events]

    assert types[0] == "thought"
    assert types[1] == "tool_call"
    assert types[2] == "tool_result"
    # No compare_runs ⇒ final event is the "no final report" error.
    assert types[-1] == "error"
    assert "without producing a final report" in events[-1].data["message"]
    assert invoked == ["parse_config"]

    # tool_call carries id/name/input; tool_result mirrors that plus ok/result/error.
    assert events[1].data == {
        "id": "tu_1",
        "name": "parse_config",
        "input": {"file_path": "/x.py"},
    }
    assert events[2].data["ok"] is True
    assert events[2].data["result"] == {"model_name": "x"}
    assert events[2].data["error"] is None

    # The user message and tool result were threaded into the backend.
    assert backend.user_messages == ["Audit this fine-tuning workload: /x.py"]
    assert backend.tool_results == [
        {
            "id": "tu_1",
            "name": "parse_config",
            "content": '{"model_name": "x"}',
            "is_error": False,
        }
    ]


@pytest.mark.asyncio
async def test_final_report_extracted_from_compare_runs(monkeypatch) -> None:
    fake_report = {"workload_name": "test", "speedup_actual": 2.0}
    backend = FakeBackend(
        scripted_turns=[
            AgentTurn(
                text_blocks=["Wrapping up."],
                tool_calls=[
                    ToolCall(
                        id="tu_compare",
                        name="compare_runs",
                        input={
                            "workload_name": "t",
                            "before": {},
                            "after": {},
                            "patch": {},
                        },
                    )
                ],
                stop_reason="end_turn",
            ),
        ]
    )
    _install_backend(monkeypatch, backend)
    _install_fake_tools(
        monkeypatch, {"compare_runs": ToolResult(ok=True, result=fake_report)}
    )

    events = await _collect(loop_module.run_audit("/x.py"))

    assert events[-1].type == "final_report"
    assert events[-1].data["report"] == fake_report


@pytest.mark.asyncio
async def test_tool_error_passes_through_does_not_crash(monkeypatch) -> None:
    backend = FakeBackend(
        scripted_turns=[
            AgentTurn(
                text_blocks=["Trying parse."],
                tool_calls=[
                    ToolCall(id="tu_1", name="parse_config", input={"file_path": "/bogus"})
                ],
                stop_reason="tool_use",
            ),
            AgentTurn(text_blocks=["Giving up."], tool_calls=[], stop_reason="end_turn"),
        ]
    )
    _install_backend(monkeypatch, backend)
    _install_fake_tools(
        monkeypatch,
        {"parse_config": ToolResult(ok=False, error="file not found")},
    )

    events = await _collect(loop_module.run_audit("/bogus"))
    tool_result_events = [e for e in events if e.type == "tool_result"]
    assert len(tool_result_events) == 1
    assert tool_result_events[0].data["ok"] is False
    assert tool_result_events[0].data["error"] == "file not found"
    # The loop kept iterating rather than bailing.
    assert events[-1].type == "error"  # no compare_runs ⇒ "no final report"

    # Backend received an is_error=True tool result with the error message.
    assert backend.tool_results[-1]["is_error"] is True
    assert backend.tool_results[-1]["content"] == "file not found"


@pytest.mark.asyncio
async def test_backend_construction_failure_yields_error_event(monkeypatch) -> None:
    _install_make_backend_raises(
        monkeypatch, RuntimeError("HF_TOKEN is not set; Qwen backend cannot run.")
    )
    events = await _collect(loop_module.run_audit("/x.py"))
    assert len(events) == 1
    assert events[0].type == "error"
    assert "HF_TOKEN" in events[0].data["message"]


@pytest.mark.asyncio
async def test_mid_loop_exception_is_caught(monkeypatch) -> None:
    backend = FakeBackend(next_turn_raises=RuntimeError("boom"))
    _install_backend(monkeypatch, backend)
    monkeypatch.setattr(loop_module.tools_module, "tool_schemas", lambda: [])

    events = await _collect(loop_module.run_audit("/x.py"))
    assert events[-1].type == "error"
    assert "boom" in events[-1].data["message"]


@pytest.mark.asyncio
async def test_loop_caps_at_max_steps(monkeypatch) -> None:
    """Even if the model never says end_turn, we bail after MAX_STEPS."""
    backend = FakeBackend(
        scripted_turns=[
            AgentTurn(
                text_blocks=[f"step {i}"],
                tool_calls=[
                    ToolCall(id=f"tu_{i}", name="parse_config", input={"file_path": "/x.py"})
                ],
                stop_reason="tool_use",
            )
            for i in range(loop_module.MAX_STEPS + 2)  # extra so we'd overrun
        ]
    )
    _install_backend(monkeypatch, backend)
    _install_fake_tools(monkeypatch, {"parse_config": ToolResult(ok=True, result={})})

    events = await _collect(loop_module.run_audit("/x.py"))
    # Backend's next_turn was called exactly MAX_STEPS times.
    assert backend.turn_count == loop_module.MAX_STEPS
    # Last event is the "no final report" error (not a crash).
    assert events[-1].type == "error"


@pytest.mark.asyncio
async def test_tool_call_id_is_threaded_to_backend(monkeypatch) -> None:
    """The loop must hand the tool_call id back to the backend so the next
    turn's request can correlate the tool_result with the originating call.
    """
    backend = FakeBackend(
        scripted_turns=[
            AgentTurn(
                text_blocks=["parse"],
                tool_calls=[
                    ToolCall(id="tu_abc", name="parse_config", input={"file_path": "/x"})
                ],
                stop_reason="tool_use",
            ),
            AgentTurn(text_blocks=["done"], tool_calls=[], stop_reason="end_turn"),
        ]
    )
    _install_backend(monkeypatch, backend)
    _install_fake_tools(
        monkeypatch, {"parse_config": ToolResult(ok=True, result={"a": 1})}
    )

    await _collect(loop_module.run_audit("/x"))

    # Backend got exactly one tool_result with id=tu_abc.
    assert len(backend.tool_results) == 1
    assert backend.tool_results[0]["id"] == "tu_abc"
    assert backend.tool_results[0]["name"] == "parse_config"
    assert backend.tool_results[0]["is_error"] is False
