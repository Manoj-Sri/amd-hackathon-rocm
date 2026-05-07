"""Tests for ``agent.tools.query_rocm_kb``.

Coverage:
  * The shipped ``kb/rocm_rules.yaml`` is loadable, validates against
    :class:`Rule`, and contains every required-by-spec rule id.
  * Each rule's ``targets_bucket`` is a valid :class:`WasteBucket` and its
    ``category`` is a valid :class:`RuleCategory` (caught for free by
    pydantic, but assert here for a clearer failure when the YAML drifts).
  * Semantic search returns the bf16 rule first for an "fp16 on MI300X"
    query and the flash-attn / sdpa rules first for an attention query.
  * Bad inputs (empty symptom, ``top_k <= 0``, ``top_k`` larger than the
    rule count) are handled gracefully via ``ToolResult.ok=False`` or
    clamped to the rule count.
  * Embeddings cache: the cache file is created on first import, and a
    second ``_embed_rules`` call with the same YAML bytes hits the cache
    without re-encoding.
  * The frozen ``Tool`` definition retains its ``name``, ``description``,
    and ``input_schema`` shape — the agent registry depends on these.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest
import yaml

from agent.schemas import Rule, RuleCategory, ToolResult, WasteBucket
from agent.tools.query_rocm_kb import (
    _KB_YAML,
    _RULES,
    QUERY_ROCM_KB,
    _cache_path,
    _embed_rules,
    _load_rules,
    _query_rocm_kb,
)


REQUIRED_RULE_IDS = {
    "precision.bf16_over_fp16_on_mi300x",
    "attention.flash_rocm_over_eager",
    "attention.sdpa_over_eager",
    "memory.batch_too_small_for_192gb",
    "memory.gradient_checkpointing_for_long_seq",
    "data.dataloader_workers_zero",
    "data.pin_memory_false",
    "data.prefetch_factor_default",
    "data.persistent_workers_false",
    "compile.torch_compile_off",
    "env.nccl_min_nchannels",
    "env.numa_auto_balancing_disable",
    "env.hsa_force_fine_grain_pcie",
    "kernels.hipblaslt_hint_logging",
    "kernels.miopen_find_mode_2",
    "optimizer.bitsandbytes_not_supported_warning",
    "collectives.one_process_per_gpu",
    "topology.tp_within_xgmi_island",
}


# ---------------------------------------------------------------------------
# YAML KB invariants
# ---------------------------------------------------------------------------


class TestYamlKB:
    def test_kb_yaml_exists(self) -> None:
        assert _KB_YAML.exists(), f"Missing {_KB_YAML}"

    def test_kb_loads_and_validates(self) -> None:
        rules, _raw = _load_rules(_KB_YAML)
        # Every rule pydantic-validated.
        for r in rules:
            assert isinstance(r, Rule)
        # Spec calls for 20-25 rules; allow a small wiggle room either way.
        assert 18 <= len(rules) <= 30, f"Expected 18-30 rules, got {len(rules)}"

    def test_required_rule_ids_present(self) -> None:
        ids = {r.id for r in _RULES}
        missing = REQUIRED_RULE_IDS - ids
        assert not missing, f"Required rule ids missing from KB: {sorted(missing)}"

    def test_rule_ids_unique(self) -> None:
        ids = [r.id for r in _RULES]
        dupes = {i for i in ids if ids.count(i) > 1}
        assert not dupes, f"Duplicate rule ids: {sorted(dupes)}"

    def test_categories_are_valid(self) -> None:
        valid = set(get_args(RuleCategory))
        for r in _RULES:
            assert r.category in valid, f"{r.id}: invalid category {r.category!r}"

    def test_targets_bucket_valid(self) -> None:
        valid = set(get_args(WasteBucket))
        for r in _RULES:
            assert r.targets_bucket in valid, (
                f"{r.id}: invalid targets_bucket {r.targets_bucket!r}"
            )

    def test_recovery_fraction_in_range(self) -> None:
        for r in _RULES:
            assert 0.0 <= r.expected_recovery_fraction <= 1.0, (
                f"{r.id}: expected_recovery_fraction out of [0,1] "
                f"({r.expected_recovery_fraction})"
            )

    def test_citations_non_empty(self) -> None:
        for r in _RULES:
            assert r.citation and r.citation.strip(), f"{r.id}: empty citation"

    def test_bitsandbytes_is_warning_only(self) -> None:
        rule = next(r for r in _RULES if r.id == "optimizer.bitsandbytes_not_supported_warning")
        # Warning rule has empty transform — propose_patch must not auto-fix.
        assert rule.transform == {}

    def test_categories_cover_spec(self) -> None:
        # Architecture §4 lists 10 categories. We expect rules in at least
        # the high-impact ones the spec calls out as required.
        cats = {r.category for r in _RULES}
        for required in (
            "precision",
            "attention",
            "memory",
            "data",
            "compile",
            "env_vars",
            "kernels",
            "optimizer",
            "collectives",
            "topology",
        ):
            assert required in cats, f"No rule for category {required!r}"


# ---------------------------------------------------------------------------
# Skip-and-warn behaviour for invalid entries
# ---------------------------------------------------------------------------


class TestLoadRulesResilience:
    def test_invalid_entry_is_skipped_with_warning(self, tmp_path: Path) -> None:
        bad = tmp_path / "rules.yaml"
        bad.write_text(
            yaml.safe_dump(
                [
                    {
                        "id": "good.rule",
                        "category": "precision",
                        "targets_bucket": "precision_path",
                        "symptom": "fp16 used",
                        "expected_impact": "switch to bf16",
                        "citation": "ROCm guide",
                    },
                    {"id": "bad.rule", "category": "not_a_real_category"},
                ]
            )
        )
        with pytest.warns(UserWarning):
            rules, _raw = _load_rules(bad)
        assert [r.id for r in rules] == ["good.rule"]

    def test_top_level_must_be_list(self, tmp_path: Path) -> None:
        bad = tmp_path / "rules.yaml"
        bad.write_text("not_a_list: 1\n")
        with pytest.raises(ValueError, match="top-level"):
            _load_rules(bad)


# ---------------------------------------------------------------------------
# Semantic search behaviour
# ---------------------------------------------------------------------------


class TestQuery:
    def test_returns_ok_for_real_query(self) -> None:
        result = _query_rocm_kb("fp16 used on MI300X with eager attention", top_k=5)
        assert isinstance(result, ToolResult)
        assert result.ok, result.error
        rules = result.result["rules"]
        assert 1 <= len(rules) <= 5

    def test_fp16_query_returns_bf16_rule_in_top_results(self) -> None:
        result = _query_rocm_kb("fp16 used on MI300X / CDNA3", top_k=3)
        ids = [r["id"] for r in result.result["rules"]]
        assert "precision.bf16_over_fp16_on_mi300x" in ids

    def test_eager_attention_query_returns_attention_rules(self) -> None:
        result = _query_rocm_kb(
            "eager attention with no flash kernel loaded on MI300X", top_k=3
        )
        ids = [r["id"] for r in result.result["rules"]]
        attention_ids = {
            "attention.flash_rocm_over_eager",
            "attention.sdpa_over_eager",
        }
        assert attention_ids & set(ids), (
            f"Expected at least one of {attention_ids} in top 3, got {ids}"
        )

    def test_dataloader_query_returns_data_rules(self) -> None:
        result = _query_rocm_kb(
            "DataLoader num_workers is zero, GPU starves waiting for batches",
            top_k=3,
        )
        ids = [r["id"] for r in result.result["rules"]]
        data_ids = {
            "data.dataloader_workers_zero",
            "data.pin_memory_false",
            "data.prefetch_factor_default",
            "data.persistent_workers_false",
        }
        assert data_ids & set(ids), (
            f"Expected at least one data.* rule in top 3, got {ids}"
        )

    def test_top_k_bounds_returned(self) -> None:
        result = _query_rocm_kb("anything", top_k=2)
        assert len(result.result["rules"]) == 2

    def test_top_k_clamped_to_rule_count(self) -> None:
        # Asking for more than we have should not crash; we should get every
        # rule back, ordered by score.
        result = _query_rocm_kb("anything", top_k=100)
        assert len(result.result["rules"]) == len(_RULES)

    def test_results_sorted_by_descending_score(self) -> None:
        # If two queries with different focus return different top rules,
        # ordering is real — top_1 differs by query.
        a = _query_rocm_kb("fp16 numerical instability", top_k=1).result["rules"][0]
        b = _query_rocm_kb("eager attention slow on long sequences", top_k=1).result[
            "rules"
        ][0]
        assert a["id"] != b["id"], (
            "Top rule should depend on query, but both queries returned "
            f"{a['id']} — semantic search is degenerate."
        )

    def test_rule_payload_is_lite_shape(self) -> None:
        # The LLM-facing rule payload is intentionally trimmed from the full
        # Rule schema — only id / symptom / transform / expected_impact /
        # citation make it through. This shrinks the audit conversation enough
        # to fit Qwen2.5-7B's 8K window. Full Rule lookup happens server-side
        # in propose_patch via the loaded KB.
        result = _query_rocm_kb("any query", top_k=1)
        payload = result.result["rules"][0]
        assert set(payload.keys()) == {
            "id",
            "symptom",
            "transform",
            "expected_impact",
            "citation",
        }
        # And the id resolves against the loaded KB so propose_patch can
        # reconstruct the full Rule.
        from agent.tools.query_rocm_kb import _RULES

        kb_ids = {r.id for r in _RULES}
        assert payload["id"] in kb_ids


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestErrors:
    def test_empty_symptom(self) -> None:
        result = _query_rocm_kb("", top_k=3)
        assert result.ok is False
        assert "symptom" in (result.error or "")

    def test_whitespace_only_symptom(self) -> None:
        result = _query_rocm_kb("   \t\n  ", top_k=3)
        assert result.ok is False

    def test_top_k_zero(self) -> None:
        result = _query_rocm_kb("fp16", top_k=0)
        assert result.ok is False
        assert "top_k" in (result.error or "")

    def test_top_k_negative(self) -> None:
        result = _query_rocm_kb("fp16", top_k=-3)
        assert result.ok is False


# ---------------------------------------------------------------------------
# Embeddings cache
# ---------------------------------------------------------------------------


class TestEmbeddingsCache:
    def test_cache_file_was_created_on_import(self) -> None:
        raw = _KB_YAML.read_bytes()
        cache = _cache_path(raw)
        assert cache.exists(), (
            f"Expected embeddings cache at {cache}; cache write failed silently."
        )

    def test_second_embed_call_uses_cache_without_recoding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rules, raw = _load_rules(_KB_YAML)
        # Sentinel: replace the lazy model getter so any encode() call would
        # blow up. If the cache is hit, the model is never consulted.
        from agent.tools import query_rocm_kb as kb_module

        def explode() -> None:
            raise AssertionError(
                "_get_model() called on a cache-hit path; cache is not being used."
            )

        monkeypatch.setattr(kb_module, "_get_model", explode)
        embeddings = _embed_rules(rules, raw)
        assert embeddings.shape[0] == len(rules)
        assert embeddings.dtype.kind == "f"


# ---------------------------------------------------------------------------
# Tool registry shape — the agent loop depends on these fields being stable.
# ---------------------------------------------------------------------------


class TestToolDefinition:
    def test_name_is_query_rocm_kb(self) -> None:
        assert QUERY_ROCM_KB.name == "query_rocm_kb"

    def test_description_unchanged_keywords(self) -> None:
        # Must still describe the search-by-symptom semantics; the system
        # prompt references this language.
        desc = QUERY_ROCM_KB.description
        assert "ROCm" in desc and "symptom" in desc

    def test_input_schema_shape(self) -> None:
        schema = QUERY_ROCM_KB.input_schema
        assert schema["type"] == "object"
        # Both single-query (`symptom`) and batched (`symptoms`) shapes are
        # advertised — either works at runtime, neither is strictly required
        # in the schema because the impl validates "at least one" itself.
        assert "symptom" in schema["properties"]
        assert "symptoms" in schema["properties"]
        assert "top_k" in schema["properties"]
        assert schema["properties"]["symptoms"]["type"] == "array"
        assert schema["properties"]["symptoms"]["items"] == {"type": "string"}

    def test_fn_is_module_query(self) -> None:
        assert QUERY_ROCM_KB.fn is _query_rocm_kb
