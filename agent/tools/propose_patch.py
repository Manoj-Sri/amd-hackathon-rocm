"""propose_patch tool — apply rules to a config, estimate uplift + confidence.

STUB IMPLEMENTATION — Phase 2 agent A replaces this with the full
deterministic transformer described in architecture.md §3:
  - Apply each rule's `transform` to the config (dotted-path key/value).
  - Skip rules whose `detect` block doesn't match.
  - Sum estimated_recovery_seconds per bucket (capped per bucket).
  - Compute speedup range from waste-budget recovery: 1 / (1 - frac).
  - Compute confidence = evidence_coverage × rule_consistency.
  - Generate unified diff (use difflib).

Current stub applies up to two known transforms and produces a plausible
Patch using the canonical 142→318 demo numbers.
"""

from __future__ import annotations

from agent.schemas import (
    Patch,
    Rule,
    RuleApplication,
    ToolResult,
    WorkloadConfig,
)
from agent.tools import Tool


def _propose_patch(config: dict, rules: list[dict], metrics: dict) -> ToolResult:
    workload = WorkloadConfig.model_validate(config)
    typed_rules = [Rule.model_validate(r) for r in rules]

    new_cfg_data = workload.model_dump()
    rationale: list[RuleApplication] = []

    for rule in typed_rules:
        if not _detect_matches(workload, rule):
            continue
        for path, value in rule.transform.items():
            _set_dotted(new_cfg_data, path, value)
        # crude per-rule recovery estimate from waste budget
        budget = metrics.get("waste_budget", {})
        bucket_seconds = float(budget.get(rule.targets_bucket, 0.0))
        rationale.append(
            RuleApplication(
                rule_id=rule.id,
                rationale=rule.expected_impact,
                citation=rule.citation,
                targets_bucket=rule.targets_bucket,
                estimated_recovery_seconds=bucket_seconds * rule.expected_recovery_fraction,
            )
        )

    new_workload = WorkloadConfig.model_validate(new_cfg_data)

    total = sum(metrics.get("waste_budget", {}).values()) or 1.0
    recovered = sum(r.estimated_recovery_seconds for r in rationale)
    frac = max(0.0, min(0.85, recovered / total))
    speedup_low = 1.0 / (1.0 - max(0.0, frac - 0.10))
    speedup_high = 1.0 / (1.0 - min(0.85, frac + 0.10))

    confidence = 0.85 if rationale else 0.0

    patch = Patch(
        new_config=new_workload,
        diff=_render_diff(workload, new_workload),
        rationale=rationale,
        expected_speedup_low=round(speedup_low, 2),
        expected_speedup_high=round(speedup_high, 2),
        confidence=round(confidence, 2),
    )
    return ToolResult(ok=True, result=patch.model_dump())


def _detect_matches(cfg: WorkloadConfig, rule: Rule) -> bool:
    data = cfg.model_dump()
    for key, expected in rule.detect.items():
        if data.get(key) != expected:
            return False
    return True


def _set_dotted(data: dict, path: str, value) -> None:
    parts = path.split(".")
    cur = data
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _render_diff(before: WorkloadConfig, after: WorkloadConfig) -> str:
    bd = before.model_dump()
    ad = after.model_dump()
    lines = []
    for key in sorted(set(bd) | set(ad)):
        if bd.get(key) != ad.get(key):
            lines.append(f"- {key}: {bd.get(key)!r}")
            lines.append(f"+ {key}: {ad.get(key)!r}")
    return "\n".join(lines) if lines else "(no changes)"


PROPOSE_PATCH = Tool(
    name="propose_patch",
    description=(
        "Apply matching ROCm rules to the user's WorkloadConfig and produce a "
        "concrete Patch with a unified diff, per-rule rationale, and a "
        "predicted speedup range with confidence. Deterministic — no LLM call."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "config": {"type": "object", "description": "WorkloadConfig dict."},
            "rules": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of Rule dicts from query_rocm_kb.",
            },
            "metrics": {
                "type": "object",
                "description": "RunMetrics dict from profile_run (used for uplift math).",
            },
        },
        "required": ["config", "rules", "metrics"],
    },
    fn=_propose_patch,
)
