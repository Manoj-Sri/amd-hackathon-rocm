"""Tool registry for the GPU Goblin agent loop.

Each tool file exports a `Tool` instance. The registry collects them so
`agent/loop.py` can pass `tool_schemas()` to the Anthropic API and
dispatch tool calls by name without hardcoded imports.

Phase 2 agents replacing tool implementations should NOT touch this file —
they edit the `fn` and the implementation module only. Adding a new tool
requires one edit here (the import + ALL_TOOLS append).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent.schemas import ToolResult


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    """JSON schema for the tool's input — passed to Claude's tool-use API."""
    fn: Callable[..., ToolResult]
    """Callable. Must accept the JSON-decoded input dict and return ToolResult."""


from agent.tools.benchmark import BENCHMARK  # noqa: E402
from agent.tools.compare_runs import COMPARE_RUNS  # noqa: E402
from agent.tools.parse_config import PARSE_CONFIG  # noqa: E402
from agent.tools.profile_run import PROFILE_RUN  # noqa: E402
from agent.tools.propose_patch import PROPOSE_PATCH  # noqa: E402
from agent.tools.query_rocm_kb import QUERY_ROCM_KB  # noqa: E402

ALL_TOOLS: list[Tool] = [
    PARSE_CONFIG,
    PROFILE_RUN,
    QUERY_ROCM_KB,
    PROPOSE_PATCH,
    BENCHMARK,
    COMPARE_RUNS,
]

TOOL_BY_NAME: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}


def tool_schemas() -> list[dict[str, Any]]:
    """Schemas in the shape Claude's tool-use API expects."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in ALL_TOOLS
    ]


def call(name: str, **kwargs: Any) -> ToolResult:
    """Dispatch a tool call by name with keyword args from the JSON-decoded input.

    Hardening notes (live-AMD-GPU lessons):

    1. **Hallucinated kwargs get silently dropped.** Models routinely invent
       plausible-sounding argument names (e.g. ``cache=True`` instead of the
       declared ``force_rerun``). We filter ``kwargs`` against the function's
       inspected signature so a single fabricated kwarg can't tank the call.

    2. **Missing required args become structured errors.** If the model
       forgets a required arg (e.g. ``profile_run`` with empty input), we
       return ``ToolResult(ok=False)`` with a message that names the missing
       fields and lists what the tool actually accepts. Letting the
       ``TypeError`` leak just confuses the model on the next turn.

    3. **Any other exception is wrapped, never raised.** Pydantic
       ``ValidationError``, runtime errors inside the tool, anything — they
       all surface as ``ToolResult(ok=False, error=...)`` so the agent loop
       can adapt and the SSE stream stays well-formed.

    Tools that declare ``**kwargs`` themselves are handled transparently —
    we pass everything through.
    """
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return ToolResult(
            ok=False,
            error=f"Unknown tool: {name!r}. Available: {sorted(TOOL_BY_NAME)}",
        )

    sig = inspect.signature(tool.fn)
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    declared = {
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    if accepts_var_kwargs:
        filtered = dict(kwargs)
    else:
        filtered = {k: v for k, v in kwargs.items() if k in declared}

    missing = [
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and p.default is inspect.Parameter.empty
        and p.name not in filtered
    ]
    if missing:
        return ToolResult(
            ok=False,
            error=(
                f"Tool {name!r} is missing required argument(s): "
                f"{', '.join(missing)}. Accepted arguments: "
                f"{sorted(declared)}."
            ),
        )

    try:
        return tool.fn(**filtered)
    except Exception as exc:
        return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
