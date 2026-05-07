"""Tests for QwenVLLMBackend.

Mocks the openai AsyncClient at the chat.completions level so no real
network call ever happens. Verifies the OpenAI-shape conversation
threading, tool-call extraction, and finish_reason mapping.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from agent.backends import active_backend_name, make_backend
from agent.backends.base import AgentTurn
from agent.backends.qwen_vllm import (
    QwenVLLMBackend,
    _normalize_finish_reason,
    _to_openai_tools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeChatCompletions:
    """Stands in for ``client.chat.completions``. Records every call and
    returns scripted responses one-by-one."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError(
                "FakeChatCompletions exhausted — backend made more calls than expected"
            )
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(responses))


def _scripted_response(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> Any:
    """Build the SimpleNamespace shape the openai SDK returns from
    chat.completions.create(...).
    """
    tcs = []
    if tool_calls:
        for tc in tool_calls:
            tcs.append(
                SimpleNamespace(
                    id=tc["id"],
                    function=SimpleNamespace(
                        name=tc["name"],
                        arguments=tc.get("arguments", "{}"),
                    ),
                )
            )
    msg = SimpleNamespace(content=content, tool_calls=tcs or None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _backend_with(responses: list[Any]) -> QwenVLLMBackend:
    """Construct a QwenVLLMBackend with a scripted client."""
    backend = QwenVLLMBackend.__new__(QwenVLLMBackend)
    backend._system = "you are a test agent"
    backend._model = "Qwen/Qwen2.5-7B-Instruct"
    backend._base_url = "http://fake-vllm:8000/v1"
    backend._api_key = "EMPTY"
    backend._max_tokens = 1024
    backend._client = _FakeClient(responses)
    backend._conversation = [{"role": "system", "content": backend._system}]
    return backend


# ---------------------------------------------------------------------------
# Tool-schema translation
# ---------------------------------------------------------------------------


def test_to_openai_tools_translates_neutral_shape() -> None:
    neutral = [
        {
            "name": "parse_config",
            "description": "Parse the file.",
            "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
        },
    ]
    out = _to_openai_tools(neutral)
    assert out == [
        {
            "type": "function",
            "function": {
                "name": "parse_config",
                "description": "Parse the file.",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                },
            },
        }
    ]


def test_to_openai_tools_handles_missing_input_schema() -> None:
    out = _to_openai_tools([{"name": "x", "description": "y"}])
    assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_finish_reason_normalization() -> None:
    assert _normalize_finish_reason("stop") == "end_turn"
    assert _normalize_finish_reason("tool_calls") == "tool_use"
    assert _normalize_finish_reason("length") == "max_tokens"
    assert _normalize_finish_reason(None) == "other"
    assert _normalize_finish_reason("weird") == "weird"


# ---------------------------------------------------------------------------
# next_turn behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_next_turn_emits_text_block_and_end_turn() -> None:
    backend = _backend_with([_scripted_response(content="hello there", finish_reason="stop")])
    backend.add_user_message("audit /tmp/x.py")
    turn = await backend.next_turn(tool_schemas=[])
    assert isinstance(turn, AgentTurn)
    assert turn.text_blocks == ["hello there"]
    assert turn.tool_calls == []
    assert turn.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_next_turn_emits_tool_calls_with_parsed_args() -> None:
    backend = _backend_with(
        [
            _scripted_response(
                content="calling parse_config",
                tool_calls=[
                    {
                        "id": "tc-1",
                        "name": "parse_config",
                        "arguments": '{"file_path": "/tmp/x.py"}',
                    }
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    backend.add_user_message("audit /tmp/x.py")
    turn = await backend.next_turn(tool_schemas=[])
    assert turn.stop_reason == "tool_use"
    assert len(turn.tool_calls) == 1
    tc = turn.tool_calls[0]
    assert tc.id == "tc-1"
    assert tc.name == "parse_config"
    assert tc.input == {"file_path": "/tmp/x.py"}


@pytest.mark.asyncio
async def test_next_turn_handles_malformed_tool_arguments() -> None:
    """vLLM occasionally emits unparseable JSON in arguments — don't crash."""
    backend = _backend_with(
        [
            _scripted_response(
                tool_calls=[
                    {"id": "tc-1", "name": "parse_config", "arguments": "{not-json"}
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    backend.add_user_message("x")
    turn = await backend.next_turn(tool_schemas=[])
    # We get the call but with empty args rather than raising.
    assert turn.tool_calls[0].input == {}


@pytest.mark.asyncio
async def test_tool_result_is_threaded_into_next_request() -> None:
    backend = _backend_with(
        [
            _scripted_response(
                tool_calls=[
                    {"id": "tc-1", "name": "parse_config", "arguments": "{}"}
                ],
                finish_reason="tool_calls",
            ),
            _scripted_response(content="done", finish_reason="stop"),
        ]
    )
    backend.add_user_message("audit")
    await backend.next_turn(tool_schemas=[])
    backend.add_tool_result("tc-1", "parse_config", '{"ok": true}', is_error=False)
    await backend.next_turn(tool_schemas=[])

    # The second create() call should include role="tool" referencing tc-1.
    second_call = backend._client.chat.completions.calls[1]
    msgs = second_call["messages"]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "tc-1"
    assert tool_msgs[0]["content"] == '{"ok": true}'


@pytest.mark.asyncio
async def test_is_error_prefix_added_to_failed_tool_results() -> None:
    backend = _backend_with([_scripted_response(content="adapting", finish_reason="stop")])
    backend.add_tool_result("tc-1", "parse_config", "file not found", is_error=True)
    await backend.next_turn(tool_schemas=[])
    msgs = backend._client.chat.completions.calls[0]["messages"]
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert tool_msg["content"].startswith("ERROR:")


# ---------------------------------------------------------------------------
# Factory selection via env var
# ---------------------------------------------------------------------------


def test_make_backend_picks_vllm_when_env_var_set(monkeypatch) -> None:
    monkeypatch.setenv("GOBLIN_AGENT_BACKEND", "qwen-vllm")
    monkeypatch.setenv("GOBLIN_QWEN_VLLM_URL", "http://test:8000/v1")
    assert active_backend_name() == "qwen-vllm"
    backend = make_backend(system_prompt="x")
    assert isinstance(backend, QwenVLLMBackend)
    assert backend._base_url == "http://test:8000/v1"


def test_active_backend_name_aliases(monkeypatch) -> None:
    for alias in ("qwen-vllm", "qwen_vllm", "vllm", "local", "QWEN-VLLM", "Vllm"):
        monkeypatch.setenv("GOBLIN_AGENT_BACKEND", alias)
        assert active_backend_name() == "qwen-vllm", alias
    for alias in ("qwen-hf", "qwen", "hf", ""):
        monkeypatch.setenv("GOBLIN_AGENT_BACKEND", alias)
        assert active_backend_name() == "qwen-hf", alias


def test_make_backend_default_is_hf(monkeypatch) -> None:
    monkeypatch.delenv("GOBLIN_AGENT_BACKEND", raising=False)
    monkeypatch.setenv("HF_TOKEN", "fake-test-token")
    assert active_backend_name() == "qwen-hf"


# ---------------------------------------------------------------------------
# Construction-time error: openai SDK missing
# ---------------------------------------------------------------------------


def test_constructor_raises_when_openai_not_installed(monkeypatch) -> None:
    """If openai isn't on PYTHONPATH, the backend's _build_client raises a
    clear RuntimeError instead of an opaque ImportError."""
    import sys

    saved = sys.modules.pop("openai", None)
    sys.modules["openai"] = None  # block re-import
    try:
        with pytest.raises(RuntimeError, match="openai"):
            QwenVLLMBackend(system_prompt="x")
    finally:
        if saved is not None:
            sys.modules["openai"] = saved
        else:
            sys.modules.pop("openai", None)
