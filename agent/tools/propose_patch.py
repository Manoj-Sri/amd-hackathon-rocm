"""propose_patch tool — apply rules to a config, estimate uplift + confidence.

Architecture.md §3 — deterministic transformer:
  - Apply each rule's `transform` to the config (dotted-path key/value).
  - Skip rules whose `detect` block doesn't match.
  - Sum estimated_recovery_seconds per bucket (capped per bucket).
  - Compute speedup range from waste-budget recovery: 1 / (1 - frac).
  - Compute confidence = evidence_coverage × rule_consistency.
  - Generate unified diff (difflib).

LLM-side resilience (live-AMD-GPU lessons):

* The model often passes only ``rule_id``s instead of full Rule objects
  between turns. We accept either: ``rule_ids: ["precision.bf16_..."]``
  resolves against the loaded KB; ``rules: [{id, ...}]`` validates fully
  if present, and falls back to id-lookup if any required Rule field is
  missing. Both inputs may be combined.

* ``config`` may arrive partially populated (the model truncated it). We
  require ``model_name`` (the only field with no schema default) and
  fill the rest from ``WorkloadConfig`` defaults.

* ``metrics`` is optional. Without a waste budget the uplift estimate
  collapses to a generous default range and the confidence drops.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from agent.schemas import (
    Patch,
    Rule,
    RuleApplication,
    ToolResult,
    WorkloadConfig,
)
from agent.tools import Tool

# Use the KB's authoritative Rule list. This means ``propose_patch`` always
# has the full Rule details (transform, citation, etc.) even when the LLM
# only forwards an ``id``.
from agent.tools.query_rocm_kb import _RULES as _KB_RULES


# ---------------------------------------------------------------------------
# Rule resolution
# ---------------------------------------------------------------------------


def _kb_index() -> dict[str, Rule]:
    return {r.id: r for r in _KB_RULES}


def _resolve_rules(
    rules: list[dict[str, Any]] | None,
    rule_ids: list[str] | None,
) -> tuple[list[Rule], list[str]]:
    """Translate the LLM's rule input into validated Rule objects.

    Returns ``(rules, warnings)`` — warnings list any rule_ids we couldn't
    resolve so the caller can surface them in the Patch rationale.
    """
    kb = _kb_index()
    resolved: list[Rule] = []
    warnings: list[str] = []
    seen: set[str] = set()

    def _take(rule: Rule) -> None:
        if rule.id not in seen:
            seen.add(rule.id)
            resolved.append(rule)

    if rule_ids:
        for rid in rule_ids:
            if rid in kb:
                _take(kb[rid])
            else:
                warnings.append(f"unknown rule id: {rid!r}")

    if rules:
        for entry in rules:
            if not isinstance(entry, dict):
                warnings.append(f"skipping non-dict rule entry: {entry!r}")
                continue
            try:
                _take(Rule.model_validate(entry))
                continue
            except ValidationError:
                pass
            # Partial entry — fall back to id lookup.
            rid = entry.get("id")
            if isinstance(rid, str) and rid in kb:
                _take(kb[rid])
            else:
                warnings.append(
                    f"rule entry missing required fields and id is unknown: "
                    f"id={rid!r}"
                )

    return resolved, warnings


def _hydrate_config(config: dict[str, Any]) -> tuple[WorkloadConfig, list[str]]:
    """Build a WorkloadConfig from a possibly-partial dict.

    Returns ``(config, warnings)``. We require ``model_name`` (no schema
    default); every other field falls back to the WorkloadConfig default
    so the LLM can pass just the dimensions it cares about.
    """
    if not config or not isinstance(config, dict):
        raise ValueError("config must be a non-empty dict")
    if not config.get("model_name"):
        raise ValueError(
            "config.model_name is required (forward the model_name from parse_config)"
        )
    warnings: list[str] = []
    try:
        return WorkloadConfig.model_validate(config), warnings
    except ValidationError as exc:
        # Best-effort: drop the offending fields and retry on a sanitized copy.
        cleaned = {k: v for k, v in config.items() if k in WorkloadConfig.model_fields}
        warnings.append(
            f"dropped extra/invalid config fields during validation: "
            f"{sorted(set(config) - set(cleaned))}"
        )
        try:
            return WorkloadConfig.model_validate(cleaned), warnings
        except ValidationError:
            raise exc  # original error wins


# ---------------------------------------------------------------------------
# Patch math
# ---------------------------------------------------------------------------


def _detect_matches(cfg: WorkloadConfig, rule: Rule) -> bool:
    data = cfg.model_dump()
    for key, expected in rule.detect.items():
        if data.get(key) != expected:
            return False
    return True


def _set_dotted(data: dict, path: str, value: Any) -> None:
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


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------


def _propose_patch(
    config: dict[str, Any],
    rules: list[dict[str, Any]] | None = None,
    rule_ids: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
) -> ToolResult:
    try:
        workload, cfg_warnings = _hydrate_config(config)
    except (ValueError, ValidationError) as exc:
        return ToolResult(ok=False, error=f"invalid config: {exc}")

    typed_rules, rule_warnings = _resolve_rules(rules, rule_ids)
    if not typed_rules:
        return ToolResult(
            ok=False,
            error=(
                "no rules to apply — pass either 'rule_ids' (list of KB rule "
                "ids from query_rocm_kb) or 'rules' (full Rule dicts). "
                + (f"Notes: {rule_warnings}" if rule_warnings else "")
            ),
        )

    new_cfg_data = workload.model_dump()
    rationale: list[RuleApplication] = []
    metrics = metrics or {}
    budget = metrics.get("waste_budget", {}) if isinstance(metrics, dict) else {}

    for rule in typed_rules:
        if not _detect_matches(workload, rule):
            continue
        for path, value in rule.transform.items():
            _set_dotted(new_cfg_data, path, value)
        bucket_seconds = float(budget.get(rule.targets_bucket, 0.0)) if budget else 0.0
        rationale.append(
            RuleApplication(
                rule_id=rule.id,
                rationale=rule.expected_impact,
                citation=rule.citation,
                targets_bucket=rule.targets_bucket,
                estimated_recovery_seconds=(
                    bucket_seconds * rule.expected_recovery_fraction
                ),
            )
        )

    new_workload = WorkloadConfig.model_validate(new_cfg_data)

    if budget:
        total = sum(float(v) for v in budget.values()) or 1.0
        recovered = sum(r.estimated_recovery_seconds for r in rationale)
        frac = max(0.0, min(0.85, recovered / total))
        speedup_low = 1.0 / (1.0 - max(0.0, frac - 0.10))
        speedup_high = 1.0 / (1.0 - min(0.85, frac + 0.10))
        # Confidence: more rationale → more confident, capped at 0.85.
        confidence = 0.85 if rationale else 0.0
    else:
        # No measured waste budget — give a wide-but-honest range and lower
        # confidence. Prevents the report from claiming false precision.
        speedup_low, speedup_high = 1.05, 1.50
        confidence = 0.3 if rationale else 0.0

    patch = Patch(
        new_config=new_workload,
        diff=_render_diff(workload, new_workload),
        rationale=rationale,
        expected_speedup_low=round(speedup_low, 2),
        expected_speedup_high=round(speedup_high, 2),
        confidence=round(confidence, 2),
    )
    payload = patch.model_dump()
    notes = cfg_warnings + rule_warnings
    if notes:
        payload["notes"] = notes
    return ToolResult(ok=True, result=payload)


PROPOSE_PATCH = Tool(
    name="propose_patch",
    description=(
        "Apply matching ROCm rules to the user's WorkloadConfig and produce a "
        "concrete Patch with a unified diff, per-rule rationale, and a "
        "predicted speedup range with confidence. Deterministic — no LLM call.\n"
        "\n"
        "Pass `rule_ids` (a list of KB rule ids you got from query_rocm_kb) — "
        "this is the preferred path; you don't need to forward the full Rule "
        "object. `rules` is also accepted for backward compatibility. `config` "
        "must include at minimum `model_name`; everything else falls back to "
        "schema defaults. `metrics` is optional but recommended — without it "
        "the uplift range and confidence are degraded."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "description": (
                    "WorkloadConfig dict. Must include model_name; other "
                    "fields fall back to schema defaults."
                ),
            },
            "rule_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of KB rule ids returned by query_rocm_kb. Preferred "
                    "input — avoids re-serializing every Rule field."
                ),
            },
            "rules": {
                "type": "array",
                "items": {"type": "object"},
                "description": (
                    "Optional alternate input: full Rule dicts. Accepts "
                    "partial entries — falls back to id-lookup against the "
                    "loaded KB if required fields are missing."
                ),
            },
            "metrics": {
                "type": "object",
                "description": (
                    "RunMetrics dict from profile_run. Optional but "
                    "recommended; the waste_budget drives uplift estimation."
                ),
            },
        },
        "required": ["config"],
    },
    fn=_propose_patch,
)
