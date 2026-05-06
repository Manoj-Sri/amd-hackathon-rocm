"""Tool registry for the GPU Goblin agent loop.

Each tool file exports a `Tool` instance. The registry collects them so
`agent/loop.py` can pass `tool_schemas()` to the Anthropic API and
dispatch tool calls by name without hardcoded imports.

Phase 2 agents replacing tool implementations should NOT touch this file —
they edit the `fn` and the implementation module only. Adding a new tool
requires one edit here (the import + ALL_TOOLS append).
"""

from __future__ import annotations

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

    Wraps unhandled exceptions in a ToolResult(ok=False) so the agent loop
    never crashes on a tool error — it sees a structured failure instead.
    """
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return ToolResult(ok=False, error=f"Unknown tool: {name}")
    try:
        return tool.fn(**kwargs)
    except Exception as exc:
        return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
