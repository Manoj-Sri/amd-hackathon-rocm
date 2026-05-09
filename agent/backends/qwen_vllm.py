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
import logging
import os
import re
import uuid
from typing import Any

from agent.backends.base import AgentTurn, Backend, ToolCall

DEFAULT_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MAX_TOKENS = 2048

_LOG = logging.getLogger(__name__)

# Matches the opening `<tool_call>` marker (with optional whitespace after).
# We deliberately don't try to balance braces with regex — `raw_decode` does
# that correctly. See `_extract_tool_calls_from_text` for usage.
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>\s*", re.DOTALL)
# Sweep at the end to remove any orphan `<tool_call>` / `</tool_call>` markers
# left over from blocks whose JSON didn't parse cleanly.
_TOOL_CALL_TAG_RE = re.compile(r"</?tool_call>")
# Chat-template special tokens that occasionally leak through when Qwen2.5-7B
# loses role discipline (the `<|im_start|>user` injection we saw on
# train_qwen_distributed_bad.py). Strip from cleaned text so the UI doesn't
# render them as thought.
_IM_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>(?:\s*\w+)?")


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
        stop_reason = _normalize_finish_reason(choice.finish_reason)

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

        # Defensive fallback: vLLM's hermes parser intermittently misses tool
        # calls when Qwen2.5-7B's response wraps them in malformed scaffolding
        # (chat-template special tokens like `<|im_start|>`, nested
        # `<tool_call>` markers, role-confused dialogue). When the parser
        # comes back empty but the text contains well-formed `<tool_call>`
        # JSON blocks, recover them so the loop dispatches them normally.
        # Without this, the audit silently misses propose_patch / compare_runs
        # and the catch-all "no final report produced" error fires.
        if not tool_calls and text:
            recovered, cleaned_text = _extract_tool_calls_from_text(text)
            if recovered:
                _LOG.warning(
                    "qwen-vllm: recovered %d tool_call(s) from raw text "
                    "after vLLM hermes parser missed them. Names: %s",
                    len(recovered),
                    ", ".join(tc.name for tc in recovered),
                )
                tool_calls = recovered
                text = cleaned_text
                text_blocks = [cleaned_text] if cleaned_text else []
                # Override finish_reason — vLLM said `stop` because no
                # tool_calls were extracted, but we just recovered some.
                stop_reason = "tool_use"

        # Echo the assistant turn back so the next request carries any
        # pending tool_calls forward (vLLM enforces this in strict mode —
        # tool-role messages must be preceded by an assistant turn that
        # has matching tool_calls). We source from the local `tool_calls`
        # list so any recovered calls end up in the echoed message too;
        # otherwise vLLM rejects the next request with a 400.
        echoed: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            echoed["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input),
                    },
                }
                for tc in tool_calls
            ]
        self._conversation.append(echoed)

        return AgentTurn(
            text_blocks=text_blocks,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
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


def _extract_tool_calls_from_text(text: str) -> tuple[list[ToolCall], str]:
    """Defensive fallback: scan raw assistant text for `<tool_call>{...}</tool_call>`
    blocks whose JSON parses cleanly, and reconstruct ToolCall objects.

    Why this exists: vLLM's hermes parser is the canonical extractor — when
    it succeeds, this function isn't called. But Qwen2.5-7B intermittently
    emits malformed scaffolding around its tool calls (chat-template special
    tokens like ``<|im_start|>``, nested ``<tool_call>`` markers,
    role-confused dialogue) that fool the parser, leaving the actual JSON
    tool call buried in the response's text content. We've seen this trip up
    `propose_patch` and `compare_runs` on the AMD MI300X live audit path,
    leading to the agent loop's "no final report produced" catch-all error.

    Implementation note: we use ``json.JSONDecoder.raw_decode`` rather than a
    regex with brace-balancing because the recovered JSON contains arbitrary
    nesting (``arguments`` typically has nested dicts and arrays for
    benchmark metrics, top_kernels, etc.). Regex with non-greedy matching
    would clip after the first ``}``, regex with greedy matching would
    swallow multiple blocks at once. ``raw_decode`` parses one JSON value
    and tells us where it ended — exactly what we need.

    Returns ``(tool_calls, cleaned_text)``. ``cleaned_text`` has the matched
    ``<tool_call>...</tool_call>`` blocks (and any orphan tags / leaked
    chat-template tokens) stripped, so the UI doesn't double-render the raw
    markup as a thought block.
    """
    tool_calls: list[ToolCall] = []
    seen_signatures: set[str] = set()
    decoder = json.JSONDecoder()

    output_parts: list[str] = []
    last_end = 0
    pos = 0

    while True:
        match = _TOOL_CALL_OPEN_RE.search(text, pos)
        if match is None:
            output_parts.append(text[last_end:])
            break

        json_start = match.end()
        try:
            payload, json_consumed = decoder.raw_decode(text[json_start:])
        except json.JSONDecodeError:
            # Malformed JSON inside this <tool_call> — skip past the opening
            # tag and keep scanning. The orphan opening tag will be cleaned
            # up by the final tag-strip pass below.
            pos = json_start
            continue

        absolute_end = json_start + json_consumed

        # Eat an optional trailing `</tool_call>` (the LLM almost always
        # emits it, but we don't require it).
        trailing = text[absolute_end:absolute_end + 30]
        close_match = re.match(r"\s*</tool_call>", trailing)
        if close_match:
            absolute_end += close_match.end()

        if isinstance(payload, dict):
            name = payload.get("name")
            args = payload.get("arguments")
            if isinstance(name, str) and isinstance(args, dict):
                # Dedupe identical recovered calls — Qwen2.5-7B sometimes
                # emits the same tool call twice in one response when it's
                # confused.
                signature = f"{name}:{json.dumps(args, sort_keys=True)}"
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    tool_calls.append(
                        ToolCall(
                            id=f"recovered_{uuid.uuid4().hex[:12]}",
                            name=name,
                            input=args,
                        )
                    )

        # Append everything before this <tool_call> to the cleaned output;
        # the block itself (tags + JSON) is dropped.
        output_parts.append(text[last_end:match.start()])
        last_end = absolute_end
        pos = absolute_end

    cleaned = "".join(output_parts)
    # Final sweep — orphan tags from blocks whose JSON didn't parse, and any
    # chat-template special tokens that leaked through.
    cleaned = _TOOL_CALL_TAG_RE.sub("", cleaned)
    cleaned = _IM_TOKEN_RE.sub("", cleaned)
    cleaned = cleaned.strip()

    return tool_calls, cleaned
