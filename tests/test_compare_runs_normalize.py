"""Tests for compare_runs's flat-Patch recovery path.

Live AMD-GPU lesson: Qwen models routinely forward a flat WorkloadConfig (or
even just the changed-fields subset) as the ``patch=`` argument to
compare_runs, instead of the full Patch envelope. ``_normalize_patch`` is
the safety net — it must:

  1. Pass real Patch dicts through unchanged.
  2. Detect any flat-config shape (full WorkloadConfig, just dataloader
     fields, just env_vars, etc.) — NOT just dicts with model_name.
  3. Recover by substituting the cached propose_patch result when one
     exists, or wrapping the flat config minimally as a last resort.
"""

from __future__ import annotations

import pytest

from agent.tools import compare_runs as cr_mod
from agent.tools import propose_patch as pp_mod


# ---------------------------------------------------------------------------
# _looks_like_flat_config detection
# ---------------------------------------------------------------------------


class TestFlatConfigDetection:
    def test_real_patch_is_not_flat(self) -> None:
        real = {
            "new_config": {"model_name": "x"},
            "diff": "(no changes)",
            "rationale": [],
            "expected_speedup_low": 1.0,
            "expected_speedup_high": 1.0,
            "confidence": 0.0,
        }
        assert cr_mod._looks_like_flat_config(real) is False

    def test_full_workload_config_is_flat(self) -> None:
        flat = {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "precision": "bf16",
            "attention_impl": "flash_rocm",
            "batch_size": 12,
        }
        assert cr_mod._looks_like_flat_config(flat) is True

    def test_dataloader_only_diff_is_flat(self) -> None:
        # The exact failure mode from the live MI300X audit: model only
        # passed the *changed* dataloader fields, no model_name in sight.
        flat = {
            "dataloader_persistent_workers": True,
            "dataloader_pin_memory": True,
            "dataloader_workers": 8,
        }
        assert cr_mod._looks_like_flat_config(flat) is True

    def test_env_vars_only_diff_is_flat(self) -> None:
        flat = {"env_vars": {"NCCL_MIN_NCHANNELS": "112"}}
        assert cr_mod._looks_like_flat_config(flat) is True

    def test_precision_only_diff_is_flat(self) -> None:
        flat = {"precision": "bf16"}
        assert cr_mod._looks_like_flat_config(flat) is True

    def test_unrelated_dict_not_flat(self) -> None:
        # Garbage dict with no WorkloadConfig fields → don't claim it's flat.
        assert cr_mod._looks_like_flat_config({"foo": 1, "bar": 2}) is False

    def test_non_dict_not_flat(self) -> None:
        assert cr_mod._looks_like_flat_config(None) is False  # type: ignore[arg-type]
        assert cr_mod._looks_like_flat_config("a string") is False  # type: ignore[arg-type]
        assert cr_mod._looks_like_flat_config([1, 2, 3]) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _normalize_patch + cached-patch recovery
# ---------------------------------------------------------------------------


@pytest.fixture
def cached_patch(monkeypatch):
    """Plant a fake `latest_patch()` so the recovery path picks it up."""
    fake = {
        "new_config": {"model_name": "Qwen/Qwen2.5-7B-Instruct", "precision": "bf16"},
        "diff": "- precision: fp16\n+ precision: bf16",
        "rationale": [
            {
                "rule_id": "precision.bf16_over_fp16_on_mi300x",
                "rationale": "r",
                "citation": "c",
                "targets_bucket": "precision_path",
                "estimated_recovery_seconds": 0.09,
            }
        ],
        "expected_speedup_low": 1.05,
        "expected_speedup_high": 1.30,
        "confidence": 0.85,
    }
    monkeypatch.setattr(pp_mod, "_LAST_PATCH", fake)
    yield fake


class TestNormalizePatch:
    def test_real_patch_passes_through(self) -> None:
        real = {
            "new_config": {"model_name": "x"},
            "diff": "...",
            "rationale": [],
            "expected_speedup_low": 1.0,
            "expected_speedup_high": 1.0,
            "confidence": 0.0,
        }
        out, notes = cr_mod._normalize_patch(real)
        assert out is real
        assert notes == []

    def test_dataloader_only_diff_recovers_via_cached(self, cached_patch) -> None:
        # The exact live-AMD-GPU failure: model forwarded only the changed
        # dataloader fields. Old code's narrow sentinel set (model_name etc.)
        # would miss this. New behavior: detected, cached patch substituted.
        flat = {
            "dataloader_persistent_workers": True,
            "dataloader_pin_memory": True,
            "dataloader_workers": 8,
        }
        out, notes = cr_mod._normalize_patch(flat)
        assert out is cached_patch  # full fidelity restored
        assert any("substituted the cached" in n for n in notes)

    def test_flat_config_falls_back_to_minimal_wrap_when_no_cache(
        self, monkeypatch
    ) -> None:
        # No cached patch — must still produce a Patch-shape dict so
        # compare_runs doesn't crash on Pydantic validation.
        monkeypatch.setattr(pp_mod, "_LAST_PATCH", None)
        flat = {"precision": "bf16"}
        out, notes = cr_mod._normalize_patch(flat)
        assert "new_config" in out
        assert "diff" in out
        assert out["expected_speedup_low"] == 1.0
        assert out["confidence"] == 0.0
        assert any("synthesized a minimal Patch" in n for n in notes)

    def test_non_flat_garbage_passes_through_for_pydantic_to_reject(
        self, monkeypatch
    ) -> None:
        # If it's neither a real Patch nor a recognizable flat config, let
        # pydantic produce the clear ValidationError — don't silently mangle.
        monkeypatch.setattr(pp_mod, "_LAST_PATCH", None)
        garbage = {"foo": 1, "bar": [2]}
        out, notes = cr_mod._normalize_patch(garbage)
        assert out is garbage
        assert notes == []
