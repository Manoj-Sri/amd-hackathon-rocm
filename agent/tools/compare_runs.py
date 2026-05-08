"""compare_runs tool — build the side-by-side Report from two RunMetrics + Patch.

Mostly a pure transform. Resilient to one specific LLM failure mode:
sometimes the model collapses the Patch dict down to its ``new_config`` fields
when forwarding it between turns (passing
``patch={"precision": "bf16", "attention_impl": "flash_rocm", ...}`` instead of
``patch={"new_config": {...}, "diff": ..., "rationale": [...], ...}``).
``_normalize_patch`` detects that shape and either:

  1. Substitutes the cached ``propose_patch`` result if available
     (preferred — preserves rationale, diff, predicted speedup, confidence).
  2. Wraps the flat config as a minimal Patch (fallback — Report has empty
     rationale and no predicted speedup, but still ships).

Either way the audit produces a Report instead of bailing on a Pydantic
ValidationError.
"""

from __future__ import annotations

from typing import Any

from agent.schemas import (
    MetricDelta,
    Patch,
    Report,
    RunMetrics,
    ToolResult,
    WorkloadConfig,
)
from agent.tools import Tool


# Patch-shape sentinel keys: a real Patch dict has at minimum new_config + diff.
_PATCH_KEYS = {"new_config", "diff", "rationale", "expected_speedup_low"}
# WorkloadConfig sentinel keys: presence of these without _PATCH_KEYS means
# the LLM passed a flat config as the patch.
_FLAT_CONFIG_KEYS = {"model_name", "precision", "attention_impl", "batch_size"}


def _looks_like_flat_config(d: dict[str, Any]) -> bool:
    if not isinstance(d, dict):
        return False
    if any(k in d for k in _PATCH_KEYS):
        return False
    return any(k in d for k in _FLAT_CONFIG_KEYS)


def _normalize_patch(patch: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return ``(patch_dict, notes)`` — never raises on a malformed input.

    If the LLM passed a flat WorkloadConfig dict instead of a Patch envelope,
    we recover by checking ``propose_patch.latest_patch()`` first (full
    fidelity) and falling back to wrapping the flat config (low fidelity).
    """
    notes: list[str] = []
    if isinstance(patch, dict) and any(k in patch for k in _PATCH_KEYS):
        return patch, notes

    if _looks_like_flat_config(patch):
        # Try the cached patch first — preserves rationale + diff + uplift.
        from agent.tools.propose_patch import latest_patch as _latest_patch

        cached = _latest_patch()
        if cached is not None:
            notes.append(
                "patch arg looked like a flat WorkloadConfig; substituted the "
                "cached propose_patch result so rationale and diff survive."
            )
            return cached, notes

        # No cache — wrap the flat config minimally so Report still renders.
        try:
            wrapped_cfg = WorkloadConfig.model_validate(
                {"model_name": "unknown", **patch}
            ).model_dump()
        except Exception:
            wrapped_cfg = patch
        notes.append(
            "patch arg looked like a flat WorkloadConfig and no cached "
            "propose_patch result was available; synthesized a minimal "
            "Patch (rationale empty, no predicted speedup)."
        )
        return (
            {
                "new_config": wrapped_cfg,
                "diff": "(diff unavailable — patch was passed as flat config)",
                "rationale": [],
                "expected_speedup_low": 1.0,
                "expected_speedup_high": 1.0,
                "confidence": 0.0,
            },
            notes,
        )

    # Some other malformed shape — let pydantic produce a clear error.
    return patch, notes


def _compare_runs(
    workload_name: str, before: dict, after: dict, patch: dict
) -> ToolResult:
    patch_dict, patch_notes = _normalize_patch(patch)

    try:
        before_m = RunMetrics.model_validate(before)
        after_m = RunMetrics.model_validate(after)
        patch_m = Patch.model_validate(patch_dict)
    except Exception as exc:
        return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")

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
    payload = report.model_dump()
    if patch_notes:
        payload["notes"] = patch_notes
    return ToolResult(ok=True, result=payload)


COMPARE_RUNS = Tool(
    name="compare_runs",
    description=(
        "Build the final side-by-side Report from a baseline RunMetrics, an "
        "optimized RunMetrics, and the Patch that connects them. Pure function — "
        "always call this last.\n"
        "\n"
        "`patch` should be the FULL Patch dict you got back from propose_patch "
        "(with `new_config`, `diff`, `rationale`, `expected_speedup_low`, etc.) "
        "— NOT just the optimized config fields. If you forward a flat config "
        "by mistake, compare_runs will recover by looking up the cached "
        "propose_patch result, but you'll lose detail."
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
            "patch": {
                "type": "object",
                "description": (
                    "FULL Patch dict from propose_patch — must include "
                    "`new_config`, `diff`, `rationale`, `expected_speedup_low/high`, "
                    "`confidence`. Do not pass just the optimized config fields."
                ),
            },
        },
        "required": ["workload_name", "before", "after", "patch"],
    },
    fn=_compare_runs,
)
