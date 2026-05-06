"""query_rocm_kb tool — semantic search over the curated ROCm rule YAML.

STUB IMPLEMENTATION — Phase 2 agent B replaces this with a real
sentence-transformers + cosine-similarity index over kb/rocm_rules.yaml.
Currently returns 2 hardcoded high-impact rules so the rest of the loop runs.
"""

from __future__ import annotations

from agent.schemas import Rule, ToolResult
from agent.tools import Tool

_STUB_RULES = [
    Rule(
        id="precision.bf16_over_fp16_on_mi300x",
        category="precision",
        targets_bucket="precision_path",
        symptom="fp16 used on MI300X / CDNA3",
        detect={"precision": "fp16"},
        transform={"precision": "bf16"},
        expected_recovery_fraction=0.85,
        expected_impact=(
            "MI300X CDNA3 matrix cores execute bf16 at the same throughput as fp16 "
            "with strictly better numerical stability. Reduces NaN risk in long runs."
        ),
        rocm_version_min="6.0",
        citation="ROCm MI300X Optimization Guide §3.2 — bf16 vs fp16",
    ),
    Rule(
        id="attention.flash_rocm_over_eager",
        category="attention",
        targets_bucket="kernel_shape",
        symptom="naive (eager) attention on MI300X — no flash kernel loaded",
        detect={"attention_impl": "eager"},
        transform={"attention_impl": "flash_rocm"},
        expected_recovery_fraction=0.7,
        expected_impact=(
            "Use the ROCm-validated flash-attention kernel (via Optimum-AMD or "
            "PyTorch SDPA backend). Eliminates O(seq_len^2) attention memory; "
            "typically 2-3x faster on MI300X for seq_len >= 1024."
        ),
        rocm_version_min="6.0",
        citation="AMD ROCm vLLM/Optimum-AMD docs — Flash Attention validated on MI300",
    ),
]


def _query_rocm_kb(symptom: str, top_k: int = 5) -> ToolResult:  # noqa: ARG001 — stub
    return ToolResult(
        ok=True,
        result={"rules": [r.model_dump() for r in _STUB_RULES[:top_k]]},
    )


QUERY_ROCM_KB = Tool(
    name="query_rocm_kb",
    description=(
        "Search the curated ROCm/MI300X optimization knowledge base by natural-"
        "language symptom. Returns up to top_k Rules with citations. Use this "
        "after profile_run to find rules matching the observed waste pattern."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "symptom": {
                "type": "string",
                "description": "Natural-language description of the observed problem.",
            },
            "top_k": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum number of rules to return.",
            },
        },
        "required": ["symptom"],
    },
    fn=_query_rocm_kb,
)
