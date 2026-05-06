"""profile_run tool — short profiling pass (default 10 steps) returning RunMetrics.

STUB IMPLEMENTATION — Phase 2 agent C replaces the body with a real rocprofv3 +
torch.profiler wrapper that calls runner/goblin_runner.sh. The current stub
delegates to FakeRunner, which loads cached metrics from
workloads/synthetic/<scenario>/cached_metrics.json based on the WorkloadConfig.
"""

from __future__ import annotations

from agent.schemas import ToolResult, WorkloadConfig
from agent.tools import Tool
from runner.protocol import FakeRunner

_RUNNER = FakeRunner()


def _profile_run(config: dict, steps: int = 10) -> ToolResult:
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
