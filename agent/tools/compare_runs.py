"""compare_runs tool — build the side-by-side Report from two RunMetrics + Patch.

Mostly a pure transform. The Phase 2 agent A may extend this with chart-ready
data structures, but the core math is here.
"""

from __future__ import annotations

from agent.schemas import (
    MetricDelta,
    Patch,
    Report,
    RunMetrics,
    ToolResult,
)
from agent.tools import Tool


def _compare_runs(
    workload_name: str, before: dict, after: dict, patch: dict
) -> ToolResult:
    before_m = RunMetrics.model_validate(before)
    after_m = RunMetrics.model_validate(after)
    patch_m = Patch.model_validate(patch)

    speedup = (
        after_m.tokens_per_sec / before_m.tokens_per_sec
        if before_m.tokens_per_sec
        else 0.0
    )

    deltas = [
        MetricDelta(
            name="tokens_per_sec",
            before=before_m.tokens_per_sec,
            after=after_m.tokens_per_sec,
            unit="tok/s",
        ),
        MetricDelta(
            name="mfu_pct", before=before_m.mfu_pct, after=after_m.mfu_pct, unit="%"
        ),
        MetricDelta(
            name="hbm_peak_gb",
            before=before_m.hbm_peak_gb,
            after=after_m.hbm_peak_gb,
            unit="GB",
        ),
        MetricDelta(
            name="gpu_util_pct",
            before=before_m.gpu_util_pct,
            after=after_m.gpu_util_pct,
            unit="%",
        ),
    ]

    summary = (
        f"Tokens/sec: {before_m.tokens_per_sec:.0f} → {after_m.tokens_per_sec:.0f} "
        f"({speedup:.2f}×). MFU: {before_m.mfu_pct:.0f}% → {after_m.mfu_pct:.0f}%."
    )

    report = Report(
        workload_name=workload_name,
        before=before_m,
        after=after_m,
        patch=patch_m,
        metric_deltas=deltas,
        waste_budget_before=before_m.waste_budget,
        waste_budget_after=after_m.waste_budget,
        speedup_actual=round(speedup, 2),
        speedup_predicted_low=patch_m.expected_speedup_low,
        speedup_predicted_high=patch_m.expected_speedup_high,
        confidence=patch_m.confidence,
        summary_line=summary,
    )
    return ToolResult(ok=True, result=report.model_dump())


COMPARE_RUNS = Tool(
    name="compare_runs",
    description=(
        "Build the final side-by-side Report from a baseline RunMetrics, an "
        "optimized RunMetrics, and the Patch that connects them. Pure function — "
        "always call this last."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "workload_name": {
                "type": "string",
                "description": "Human-readable workload label for the report header.",
            },
            "before": {"type": "object", "description": "Baseline RunMetrics dict."},
            "after": {"type": "object", "description": "Optimized RunMetrics dict."},
            "patch": {"type": "object", "description": "Patch dict from propose_patch."},
        },
        "required": ["workload_name", "before", "after", "patch"],
    },
    fn=_compare_runs,
)
