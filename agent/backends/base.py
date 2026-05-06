"""Backend protocol — what the agent loop needs from any LLM provider.

The loop is provider-agnostic. Each backend owns its own conversation state
and per-API translation; the loop only sees the unified `AgentTurn` shape.

Two backends ship:
  * `ClaudeBackend` — anthropic.AsyncAnthropic, system as a top-level field,
    tool_result blocks batched in one user message.
  * `QwenHFBackend` — huggingface_hub.AsyncInferenceClient.chat_completion,
    system as the first message, tool_result as one role="tool" message per
    call. Routes to whichever provider serves the chosen Qwen model.

A future LiveQwenBackend talking to a self-hosted vLLM-on-MI300X endpoint
slots in identically — it just speaks the OpenAI-compatible shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """One tool call requested by the model in a turn."""

    id: str
    """Provider-assigned identifier. Used to correlate the eventual tool_result."""
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTurn:
    """One turn's response, normalized across providers."""

    text_blocks: list[str] = field(default_factory=list)
    """Free-text the model produced this turn (rendered as `thought` SSE events)."""

    tool_calls: list[ToolCall] = field(default_factory=list)

    stop_reason: str = "end_turn"
    """One of: 'end_turn', 'tool_use', 'max_tokens', 'other'.

    The loop breaks on 'end_turn'. Other values keep iterating up to MAX_STEPS.
    """


class Backend(ABC):
    """Pluggable LLM driver for the agent loop.

    Lifecycle:
        backend = SomeBackend(system_prompt=...)
        backend.add_user_message("Audit this workload: ...")
        for step in range(MAX_STEPS):
            turn = await backend.next_turn(tool_schemas)
            ... yield events ...
            for tc in turn.tool_calls:
                result = call_tool(tc)
                backend.add_tool_result(tc.id, tc.name, result.content, is_error=...)
            if turn.stop_reason == "end_turn":
                break
    """

    name: str = "base"
    """Short label used in /healthz and logs (e.g. 'claude', 'qwen-hf')."""

    @abstractmethod
    def add_user_message(self, content: str) -> None:
        """Append a user message to the internal conversation."""

    @abstractmethod
    async def next_turn(self, tool_schemas: list[dict[str, Any]]) -> AgentTurn:
        """Run one turn against the provider. Updates internal state so the
        next call to `next_turn` already has the assistant's last response in
        context. Tool schemas are in the loop's neutral shape:
            {"name": str, "description": str, "input_schema": json-schema dict}
        Each backend translates to the provider's own tool format.
        """

    @abstractmethod
    def add_tool_result(
        self,
        tool_call_id: str,
        name: str,
        content: str,
        is_error: bool,
    ) -> None:
        """Record a tool result for the model to consume on the next turn.

        Different providers want different shapes (Claude batches into one
        user message; OpenAI/Qwen wants one role='tool' message per call).
        Implementations buffer or append as appropriate.
        """
