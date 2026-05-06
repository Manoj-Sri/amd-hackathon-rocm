"""LLM backend for the GPU Goblin agent loop — Qwen via HF Inference Providers.

Today: a single concrete backend, `QwenHFBackend`, which routes through
Hugging Face Inference Providers (Together / Fireworks / Nebius / Replicate
/ ...). The factory exists so a future `LiveQwenBackend` (talking to a
self-hosted vLLM-on-MI300X endpoint) can slot in without changing the loop.

Knobs (env vars):
    HF_TOKEN                       # required — your HF Inference token
    GOBLIN_QWEN_MODEL              # default: Qwen/Qwen2.5-7B-Instruct
    GOBLIN_QWEN_PROVIDER           # default: auto (HF picks Together/Fireworks/etc.)
"""

from __future__ import annotations

from agent.backends.base import AgentTurn, Backend, ToolCall
from agent.backends.qwen_hf import DEFAULT_MODEL, DEFAULT_PROVIDER, QwenHFBackend

__all__ = [
    "AgentTurn",
    "Backend",
    "QwenHFBackend",
    "ToolCall",
    "make_backend",
    "active_backend_name",
]


def active_backend_name() -> str:
    """Short label used in /healthz and logs."""
    return "qwen-hf"


def make_backend(system_prompt: str, **kwargs) -> Backend:
    """Construct the agent backend.

    Constructor kwargs (model, provider, max_tokens) are passed through;
    irrelevant kwargs are silently dropped so callers can stay
    backend-agnostic for when a second backend is added later.
    """
    return QwenHFBackend(
        system_prompt=system_prompt,
        model=kwargs.get("model"),
        provider=kwargs.get("provider"),
        max_tokens=kwargs.get("max_tokens", 2048),
    )
