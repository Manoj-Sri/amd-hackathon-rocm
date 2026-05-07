"""LLM backends for the GPU Goblin agent loop.

Two backends ship today, both Qwen, both speaking OpenAI-shape tool calls:

  * ``QwenHFBackend``    — Qwen via Hugging Face Inference Providers.
                           HF auto-routes to Together / Fireworks-AI /
                           Nebius / etc. Needs ``HF_TOKEN``. The default
                           and the path the public HF Space uses.

  * ``QwenVLLMBackend``  — Qwen self-hosted on YOUR MI300X via vLLM,
                           OpenAI-compatible at ``http://host:8000/v1``.
                           "All AMD silicon" path. Stand it up with
                           the lablab tutorial recipe, then point Goblin
                           at it with ``GOBLIN_AGENT_BACKEND=qwen-vllm``.

Pick one with the env var ``GOBLIN_AGENT_BACKEND``:

    export GOBLIN_AGENT_BACKEND=qwen-hf       # default — uses HF_TOKEN
    export GOBLIN_AGENT_BACKEND=qwen-vllm     # uses GOBLIN_QWEN_VLLM_URL

Backend-specific knobs:

    # qwen-hf
    HF_TOKEN                       # required
    GOBLIN_QWEN_MODEL              # default Qwen/Qwen2.5-7B-Instruct
    GOBLIN_QWEN_PROVIDER           # default auto

    # qwen-vllm
    GOBLIN_QWEN_VLLM_URL           # default http://localhost:8000/v1
    GOBLIN_QWEN_VLLM_MODEL         # default Qwen/Qwen2.5-7B-Instruct
    GOBLIN_QWEN_VLLM_KEY           # optional auth header (vLLM ignores it)
"""

from __future__ import annotations

import os

from agent.backends.base import AgentTurn, Backend, ToolCall

__all__ = [
    "AgentTurn",
    "Backend",
    "ToolCall",
    "make_backend",
    "active_backend_name",
]


_VLLM_ALIASES = {"qwen-vllm", "qwen_vllm", "vllm", "local", "qwen-local"}
_HF_ALIASES = {"qwen-hf", "qwen_hf", "qwen", "hf"}


def active_backend_name() -> str:
    """The backend name selected by env, normalized to its canonical id.

    Anything not recognised falls through to ``qwen-hf`` (the safe default).
    """
    raw = (os.environ.get("GOBLIN_AGENT_BACKEND") or "qwen-hf").strip().lower()
    if raw in _VLLM_ALIASES:
        return "qwen-vllm"
    if raw in _HF_ALIASES:
        return "qwen-hf"
    return "qwen-hf"


def make_backend(system_prompt: str, **kwargs) -> Backend:
    """Construct the agent backend selected by ``GOBLIN_AGENT_BACKEND``.

    Constructor kwargs (``model``, ``provider``, ``base_url``, ``api_key``,
    ``max_tokens``) are forwarded to whichever backend is chosen; irrelevant
    kwargs are silently dropped so callers can stay backend-agnostic.

    Imports the chosen backend module lazily — neither the openai SDK nor
    huggingface_hub is loaded unless the corresponding backend is actually
    in use.
    """
    name = active_backend_name()
    if name == "qwen-vllm":
        from agent.backends.qwen_vllm import QwenVLLMBackend

        return QwenVLLMBackend(
            system_prompt=system_prompt,
            model=kwargs.get("model"),
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            max_tokens=kwargs.get("max_tokens", 2048),
        )

    from agent.backends.qwen_hf import QwenHFBackend

    return QwenHFBackend(
        system_prompt=system_prompt,
        model=kwargs.get("model"),
        provider=kwargs.get("provider"),
        max_tokens=kwargs.get("max_tokens", 2048),
    )
