"""QwenVLLMBackend — drives Qwen via a self-hosted vLLM-on-MI300X endpoint.

This is the "all AMD silicon" path: Qwen runs on the same MI300X that's
auditing the user's workload, served by vLLM behind an OpenAI-compatible
``/v1/chat/completions`` endpoint. We talk to it with the standard ``openai``
SDK pointed at a custom ``base_url``.

Stand it up with the lablab tutorial recipe — TL;DR:

    docker run -d --name qwen-vllm \\
        --device=/dev/kfd --device=/dev/dri --group-add video \\
        --ipc=host --shm-size=16g \\
        -p 8000:8000 \\
        -v $HF_HOME:/root/.cache/huggingface \\
        rocm/vllm:latest \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --dtype bfloat16 \\
        --max-model-len 8192 \\
        --enable-auto-tool-choice \\
        --tool-call-parser hermes

The ``--enable-auto-tool-choice --tool-call-parser hermes`` flags are the
ones that matter — Qwen2.5 uses Hermes-format tool tags and vLLM needs to
parse them into the OpenAI ``tool_calls`` shape on the way out.

Configuration (env vars):
    GOBLIN_QWEN_VLLM_URL    Base URL ending in /v1. Default http://localhost:8000/v1.
    GOBLIN_QWEN_VLLM_MODEL  Served model id. Default Qwen/Qwen2.5-7B-Instruct.
    GOBLIN_QWEN_VLLM_KEY    Optional auth header. vLLM ignores it by default;
                            useful if you put nginx/Caddy in front with auth.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agent.backends.base import AgentTurn, Backend, ToolCall

DEFAULT_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MAX_TOKENS = 2048


class QwenVLLMBackend(Backend):
    """OpenAI-compatible client pointed at a self-hosted vLLM endpoint.

    Same OpenAI conversation shape ``QwenHFBackend`` uses — the only
    difference is the transport: we hit a URL we control instead of HF's
    Inference Providers router. That means tool-call latency drops to
    in-cluster network (good) and we burn MI300X cycles instead of HF
    credits (also good — it's what the AMD credits are for).
    """

    name = "qwen-vllm"

    def __init__(
        self,
        system_prompt: str,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._system = system_prompt
        self._model = model or os.environ.get("GOBLIN_QWEN_VLLM_MODEL", DEFAULT_MODEL)
        self._base_url = base_url or os.environ.get(
            "GOBLIN_QWEN_VLLM_URL", DEFAULT_URL
        )
        self._api_key = api_key or os.environ.get(
            "GOBLIN_QWEN_VLLM_KEY", DEFAULT_API_KEY
        )
        self._max_tokens = max_tokens
        self._client = self._build_client()
        # System message at head of conversation (OpenAI shape).
        self._conversation: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

    def _build_client(self) -> Any:
        # Lazy import — pulls openai SDK only when this backend is selected.
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "QwenVLLMBackend requires the 'openai' package. "
                "Install with `pip install openai>=1.30` (or `pip install -e \".[dev]\"` "
                "for the full dev extras)."
            ) from exc
        return AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)

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

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._conversation,
            tools=oai_tools,
            max_tokens=self._max_tokens,
            tool_choice="auto",
        )

        choice = response.choices[0]
        msg = choice.message
        text = (msg.content or "").strip()

        # Echo the assistant turn back so the next request carries any
        # pending tool_calls forward (vLLM enforces this in strict mode).
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
    """Translate the codebase's neutral tool schema (the ``tool_schemas()``
    shape with ``name``/``description``/``input_schema``) into OpenAI's
    ``{type: function, function: {...}}`` shape that vLLM consumes.

    Same translation as ``qwen_hf._to_openai_tools`` — kept duplicated rather
    than shared because the two backends are independently importable and we
    don't want one to drag in the other's dependencies.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": (
                    s.get("input_schema") or {"type": "object", "properties": {}}
                ),
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
