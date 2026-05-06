"""benchmark tool — full benchmark (default 50 steps), version-tagged cached.

STUB IMPLEMENTATION — Phase 2 agent C replaces with rocprofv3 + torch.profiler
wrapper that goes through goblin_runner.sh on a real MI300X. Cache key includes
container + runner script SHA so stale results don't sneak through.

Current stub delegates to FakeRunner (same backing data as profile_run, just
more steps).
"""

from __future__ import annotations

from agent.schemas import ToolResult, WorkloadConfig
from agent.tools import Tool
from runner.protocol import FakeRunner

_RUNNER = FakeRunner()


def _benchmark(config: dict, steps: int = 50) -> ToolResult:
    workload = WorkloadConfig.model_validate(config)
    metrics = _RUNNER.run(workload, steps=steps)
    return ToolResult(ok=True, result=metrics.model_dump())


BENCHMARK = Tool(
    name="benchmark",
    description=(
        "Full benchmark (default 50 steps). Same metric shape as profile_run "
        "but at production-scale step count. Result is cached by a version-"
        "tagged hash so re-runs of the same config are free. Use this AFTER "
        "propose_patch to validate the patched config — and call it once on "
        "the original config for before/after comparison."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "config": {"type": "object", "description": "WorkloadConfig dict."},
            "steps": {
                "type": "integer",
                "default": 50,
                "minimum": 10,
                "maximum": 500,
                "description": "Number of measured steps (after a 2-step warmup).",
            },
        },
        "required": ["config"],
    },
    fn=_benchmark,
)
