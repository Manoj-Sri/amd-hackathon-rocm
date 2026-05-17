"""profile_run tool — short profiling pass (default 10 steps) returning RunMetrics.

Delegates to `runner.protocol.LiveRunner`, which itself auto-falls-back to
`MockRunner` whenever the host can't actually run a live profile (no
rocprofv3, no amd-smi, no /dev/dri/renderD*, subprocess failure, parse
error). On a laptop you'll always see `runner_kind="fake"` with a clear
warning prepended to `RunMetrics.warnings`.
"""

from __future__ import annotations

from agent.schemas import ToolResult, WorkloadConfig
from agent.tools import Tool
from runner.protocol import _default_runner

_RUNNER = _default_runner()


def _profile_run(config: dict, steps: int = 10) -> ToolResult:
    # Defensive: Qwen2.5-7B occasionally nests ``steps`` *inside* the config
    # dict rather than at the top level. WorkloadConfig strict-validates so
    # this would error out and burn a tool slot for nothing. Pull it back
    # out, letting an explicit caller-supplied value win ties.
    if isinstance(config, dict) and "steps" in config:
        misnested = config.pop("steps")
        if steps == 10 and misnested is not None:
            try:
                steps = int(misnested)
            except (TypeError, ValueError):
                pass
    workload = WorkloadConfig.model_validate(config)
    metrics = _RUNNER.run(workload, steps=steps)
    return ToolResult(ok=True, result=metrics.model_dump())


PROFILE_RUN = Tool(
    name="profile_run",
    description=(
        "Run the workload for a short profiling pass (default 10 steps after a "
        "2-step warmup). Returns RunMetrics with a populated WasteBudget so the "
        "agent can see where time is being lost on MI300X. Cheap to call early."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "description": "WorkloadConfig from parse_config (as a dict).",
            },
            "steps": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
                "description": "Profile this many steps after a 2-step warmup.",
            },
        },
        "required": ["config"],
    },
    fn=_profile_run,
)
