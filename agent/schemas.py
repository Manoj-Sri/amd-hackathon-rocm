"""Shared pydantic models for the GPU Goblin agent.

Single source of truth for tool inputs/outputs. Every tool consumes and
returns these types — no raw dicts cross tool boundaries.

Defined in `architecture.md` §3 (the six tools) and §3 (waste-budget decomposition).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared envelope: every tool returns ToolResult so the agent loop can handle
# failures uniformly without inventing per-tool error shapes.
# ---------------------------------------------------------------------------


class ToolResult(BaseModel):
    """Uniform envelope for every tool call.

    `ok=False` means the tool ran but couldn't produce a useful result. The
    agent should consult `error` and either retry with different input,
    fall back to another tool, or surface the issue in the final report.
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool
    result: Any | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Configs the user uploads, parsed into a normalized shape.
# ---------------------------------------------------------------------------


Precision = Literal["fp32", "fp16", "bf16", "fp8"]
AttentionImpl = Literal["sdpa", "flash", "flash_rocm", "eager", "unknown"]


class WorkloadConfig(BaseModel):
    """Normalized representation of a user's training script / TrainingArguments.

    Filled by `parse_config`. Intentionally narrow — only the hyperparameters
    we have rules for. Anything else lives in `extras` so we don't lose it.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str
    batch_size: int = 1
    grad_accum_steps: int = 1
    seq_len: int = 1024
    precision: Precision = "fp16"
    optimizer: str = "adamw_torch"
    attention_impl: AttentionImpl = "unknown"
    gradient_checkpointing: bool = False
    lora_rank: int | None = None
    dataloader_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_prefetch_factor: int | None = None
    dataloader_persistent_workers: bool = False
    torch_compile: bool = False
    lr: float = 5e-5
    warmup_steps: int = 0
    env_vars: dict[str, str] = Field(default_factory=dict)
    extras: dict[str, Any] = Field(default_factory=dict)
    raw_source: str = ""
    redactions: list[str] = Field(default_factory=list)
    """Labels of secret-shaped strings that were redacted during parse."""


# ---------------------------------------------------------------------------
# Run metrics — the unified output of profile_run AND benchmark.
# Pre-emptive fix from brooks-audit: don't let ProfileSummary and
# BenchmarkResult drift apart.
# ---------------------------------------------------------------------------


class WasteBudget(BaseModel):
    """Decomposition of total step time into recoverable buckets.

    Sum of fields ≈ total step time (modulo measurement noise). Each field is
    seconds (per step). Buckets a rule can target are listed in the
    `WasteBucket` literal below.
    """

    model_config = ConfigDict(extra="forbid")

    useful_gpu: float = 0.0
    data_wait: float = 0.0
    host_gap: float = 0.0
    comm_excess: float = 0.0
    memory_headroom: float = 0.0
    precision_path: float = 0.0
    kernel_shape: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.useful_gpu
            + self.data_wait
            + self.host_gap
            + self.comm_excess
            + self.memory_headroom
            + self.precision_path
            + self.kernel_shape
        )

    @property
    def recoverable(self) -> float:
        """Sum of all non-useful buckets — upper bound on what optimization can save."""
        return self.total - self.useful_gpu


WasteBucket = Literal[
    "data_wait",
    "host_gap",
    "comm_excess",
    "memory_headroom",
    "precision_path",
    "kernel_shape",
]


class KernelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    pct_time: float


class RunMetrics(BaseModel):
    """Output of profile_run (steps=10) and benchmark (steps=50+).

    Same schema both ways — only `steps` differs in convention.
    """

    model_config = ConfigDict(extra="forbid")

    steps: int
    tokens_per_sec: float
    mfu_pct: float
    hbm_peak_gb: float
    hbm_avg_gb: float
    gpu_util_pct: float
    top_kernels: list[KernelEntry] = Field(default_factory=list)
    attention_kernel_loaded: str = "unknown"
    waste_budget: WasteBudget = Field(default_factory=WasteBudget)
    warnings: list[str] = Field(default_factory=list)
    rocm_version: str = "unknown"
    pytorch_version: str = "unknown"
    runner_kind: Literal["live", "fake"] = "live"
    """Whether these metrics came from a real MI300X (live) or FakeRunner replay."""


# ---------------------------------------------------------------------------
# Knowledge base rules + patch generation.
# ---------------------------------------------------------------------------


RuleCategory = Literal[
    "precision",
    "attention",
    "memory",
    "kernels",
    "env_vars",
    "optimizer",
    "data",
    "compile",
    "collectives",
    "topology",
]


class Rule(BaseModel):
    """One curated ROCm-specific optimization rule.

    Lives in `kb/rocm_rules.yaml`. Pre-embedded by the KB build step so
    `query_rocm_kb` can do cosine similarity at query time.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    category: RuleCategory
    targets_bucket: WasteBucket
    """Which waste-budget bucket this rule recovers from. Used by propose_patch
    to avoid double-counting and by uplift estimator to weight the gain."""

    symptom: str
    """Natural-language description of when the rule applies. Embedded for search."""

    detect: dict[str, Any] = Field(default_factory=dict)
    """Optional structured precondition over ConfigDict fields. If present and
    doesn't match the user's config, propose_patch skips this rule even if
    query_rocm_kb returned it."""

    transform: dict[str, Any] = Field(default_factory=dict)
    """Concrete config mutations to apply. Keys are dotted paths into ConfigDict
    (e.g. 'precision', 'env_vars.NCCL_MIN_NCHANNELS'). Values are the target value."""

    expected_recovery_fraction: float = 0.3
    """Conservative estimate: fraction of `targets_bucket` time this rule recovers.
    Used as the multiplier in the uplift formula. Range [0, 1]."""

    expected_impact: str
    """Human-readable rationale shown in the report."""

    rocm_version_min: str = "6.0"
    citation: str
    """ROCm doc page / AMD blog post / paper that backs this rule. Required."""


class RuleApplication(BaseModel):
    """One rule applied to one config — the audit trail for the report."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    rationale: str
    citation: str
    targets_bucket: WasteBucket
    estimated_recovery_seconds: float
    """How much of `targets_bucket` we expect this rule to recover, in seconds/step."""


class Patch(BaseModel):
    """Output of propose_patch — a concrete diff plus the audit trail."""

    model_config = ConfigDict(extra="forbid")

    new_config: WorkloadConfig
    diff: str
    """Unified diff between the original and patched config (pretty-printed)."""

    rationale: list[RuleApplication] = Field(default_factory=list)
    expected_speedup_low: float = 1.0
    """Conservative end of the speedup range, multiplicative (e.g., 1.4 means +40%)."""

    expected_speedup_high: float = 1.0
    """Optimistic end of the speedup range."""

    confidence: float = 0.0
    """0..1: evidence_coverage × rule_consistency.
    See architecture.md §3 propose_patch confidence formula."""


# ---------------------------------------------------------------------------
# Final report — the side-by-side payload the UI renders.
# ---------------------------------------------------------------------------


class MetricDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    before: float
    after: float
    unit: str = ""

    @property
    def delta_pct(self) -> float:
        if self.before == 0:
            return 0.0
        return (self.after - self.before) / self.before * 100.0


class Report(BaseModel):
    """Final side-by-side audit report. Built by compare_runs."""

    model_config = ConfigDict(extra="forbid")

    workload_name: str
    before: RunMetrics
    after: RunMetrics
    patch: Patch
    metric_deltas: list[MetricDelta] = Field(default_factory=list)
    waste_budget_before: WasteBudget
    waste_budget_after: WasteBudget
    speedup_actual: float
    """Measured speedup: after.tokens_per_sec / before.tokens_per_sec."""

    speedup_predicted_low: float
    speedup_predicted_high: float
    confidence: float
    summary_line: str
    """One-sentence headline for the demo: 'Tokens/sec 142 → 318 (2.24×).'"""

    validity_footer: str = (
        "Recommendations validated against MI300X with the observed ROCm and "
        "PyTorch versions. Re-run the audit if you change model, hardware, or "
        "framework version."
    )


# ---------------------------------------------------------------------------
# SSE stream events — what the FastAPI server pushes to the UI.
# ---------------------------------------------------------------------------


class SSEEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["thought", "tool_call", "tool_result", "final_report", "error"]
    data: dict[str, Any]
