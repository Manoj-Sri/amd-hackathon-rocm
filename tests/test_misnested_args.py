"""Tests for the defensive misnested-arg extraction in benchmark + profile_run.

Live AMD-GPU lesson: Qwen2.5-7B (and probably others) occasionally JSON-nests
``steps`` / ``cache`` *inside* the ``config`` dict instead of at the top level
alongside it. WorkloadConfig strict-validates extras, so without this defense
the call errors out and a tool slot is wasted. The well-tuned scenario run
on 2026-05-07 burned two of the eight available slots on this exact mistake;
fixing it costs nothing and saves the audit.
"""

from __future__ import annotations

import shutil

from agent.tools import call


def _baseline_config() -> dict:
    return {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "batch_size": 4,
        "precision": "fp16",
        "attention_impl": "eager",
        "dataloader_workers": 0,
    }


class TestBenchmarkMisnestedArgs:
    def setup_method(self) -> None:
        # Each test starts with an empty cache so cache-hit doesn't mask the
        # behavior under test.
        shutil.rmtree("bench_cache", ignore_errors=True)

    def test_steps_nested_in_config_is_extracted(self) -> None:
        """Old behavior: ``WorkloadConfig`` validation explodes with
        'Extra inputs are not permitted [steps]'. New behavior: defensive
        extraction pulls ``steps`` back to the top-level arg, call succeeds.
        """
        cfg = {**_baseline_config(), "steps": 25}
        result = call("benchmark", config=cfg)
        assert result.ok, result.error
        assert result.result["steps"] == 25

    def test_cache_nested_in_config_is_extracted(self) -> None:
        cfg = {**_baseline_config(), "cache": False}
        result = call("benchmark", config=cfg)
        assert result.ok, result.error

    def test_force_rerun_nested_in_config_is_extracted(self) -> None:
        cfg = {**_baseline_config(), "force_rerun": True}
        result = call("benchmark", config=cfg)
        assert result.ok, result.error

    def test_explicit_top_level_wins_over_nested(self) -> None:
        """If caller passes BOTH (config has steps + top-level steps), the
        explicit non-default top-level wins. Defensive code is for the
        accident case, not for letting nesting silently override."""
        cfg = {**_baseline_config(), "steps": 25}
        result = call("benchmark", config=cfg, steps=37)
        assert result.ok, result.error
        assert result.result["steps"] == 37

    def test_all_three_nested_at_once(self) -> None:
        """The exact failure mode from the live run: model nested three
        runtime args inside config. All three should get pulled out.
        """
        cfg = {
            **_baseline_config(),
            "steps": 30,
            "cache": False,
            "force_rerun": True,
        }
        result = call("benchmark", config=cfg)
        assert result.ok, result.error
        assert result.result["steps"] == 30


class TestProfileRunMisnestedArgs:
    def test_steps_nested_in_config_is_extracted(self) -> None:
        cfg = {**_baseline_config(), "steps": 7}
        result = call("profile_run", config=cfg)
        assert result.ok, result.error
        assert result.result["steps"] == 7

    def test_explicit_top_level_wins(self) -> None:
        cfg = {**_baseline_config(), "steps": 7}
        result = call("profile_run", config=cfg, steps=15)
        assert result.ok, result.error
        assert result.result["steps"] == 15

    def test_clean_config_unaffected(self) -> None:
        """Sanity: when nothing is misnested, behavior is unchanged."""
        result = call("profile_run", config=_baseline_config())
        assert result.ok, result.error
        assert result.result["steps"] == 10  # default
