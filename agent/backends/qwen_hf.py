"""QwenHFBackend — drives Qwen via huggingface_hub.AsyncInferenceClient.

Routes through Hugging Face Inference Providers (Together / Fireworks /
Replicate / Nebius / etc., with `provider="auto"` doing the picking).
The chat_completion API matches OpenAI shape, which is what Qwen tool
calling speaks natively.

Conversation shape (OpenAI-compatible):
  * system as the first message: {"role": "system", "content": "..."}
  * user / assistant alternation
  * tool_calls live on the assistant message: {"role": "assistant",
        "content": "...", "tool_calls": [{"id": ..., "type": "function",
        "function": {"name": ..., "arguments": "<json string>"}}]}
  * tool results: one message each: {"role": "tool", "tool_call_id": ...,
        "content": "..."}

Tool schemas (in this codebase's neutral shape) translate to:
  {"type": "function", "function": {"name": ..., "description": ...,
   "parameters": <JSON-schema>}}

If a tool result is an error, we prefix the content with `ERROR: ` so the
model sees that the call failed (OpenAI shape has no `is_error` flag).
"""

from __future__ import annotations

import json
import os
from typing import Any

from huggingface_hub import AsyncInferenceClient

from agent.backends.base import AgentTurn, Backend, ToolCall

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_PROVIDER = "auto"


class QwenHFBackend(Backend):
    """Qwen-via-HF-Inference-Providers driver."""

    name = "qwen-hf"

    def __init__(
        self,
        system_prompt: str,
        model: str | None = None,
        provider: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._system = system_prompt
        self._model = model or os.environ.get("GOBLIN_QWEN_MODEL", DEFAULT_MODEL)
        self._provider = provider or os.environ.get(
            "GOBLIN_QWEN_PROVIDER", DEFAULT_PROVIDER
        )
        self._max_tokens = max_tokens
        self._client = self._build_client()
        # System message lives at the head of the conversation for OpenAI-shape
        # APIs. We seed it once at construction.
        self._conversation: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

    @staticmethod
    def _build_client() -> AsyncInferenceClient:
        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGINGFACEHUB_API_TOKEN"
        )
        if not token:
            raise RuntimeError(
                "HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) is not set; Qwen backend "
                "cannot reach HF Inference Providers. Set the env var, switch to "
                "GOBLIN_AGENT_BACKEND=claude, or use the offline replay UI lane."
            )
        # `provider` is set per-call rather than on the client to keep the
        # constructor stable across huggingface_hub versions.
        return AsyncInferenceClient(token=token)

    # ------------------------------------------------------------------
    # Backend API
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self._conversation.append({"role": "user", "content": content})

    def add_tool_result(
        self,
        tool_call_id: str,
        name: str,  # noqa: ARG002 — OpenAI shape correlates by tool_call_id only
        content: str,
        is_error: bool,
    ) -> None:
        if is_error and not content.startswith("ERROR:"):
            content = f"ERROR: {content}"
        self._conversation.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    async def next_turn(self, tool_schemas: list[dict[str, Any]]) -> AgentTurn:
        oai_tools = _to_openai_tools(tool_schemas)

        response = await self._client.chat_completion(
            model=self._model,
            messages=self._conversation,
            tools=oai_tools,
            max_tokens=self._max_tokens,
            tool_choice="auto",
        )

        choice = response.choices[0]
        msg = choice.message
        text = (msg.content or "").strip()

        # Echo assistant turn so the next request preserves tool_calls.
        echoed: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            echoed["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in msg.tool_calls
            ]
        self._conversation.append(echoed)

        # Translate to neutral AgentTurn.
        text_blocks = [text] if text else []
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except (TypeError, json.JSONDecodeError):
                args = {}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, input=args)
            )

        return AgentTurn(
            text_blocks=text_blocks,
            tool_calls=tool_calls,
            stop_reason=_normalize_finish_reason(choice.finish_reason),
        )


def _to_openai_tools(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate this codebase's neutral tool schema (the `tool_schemas()`
    shape with `name`/`description`/`input_schema`) into OpenAI's
    `{type: function, function: {...}}` shape that vLLM and HF chat_completion
    consume.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": s.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for s in tool_schemas
    ]


def _normalize_finish_reason(reason: str | None) -> str:
    """Map OpenAI finish_reason to our neutral set."""
    if reason == "stop":
        return "end_turn"
    if reason == "tool_calls":
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    return reason or "other"
