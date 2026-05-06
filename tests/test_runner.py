"""Tests for runner/protocol.py and runner/profile_parser.py.

Two laptop-only invariants:
  1. FakeRunner still works exactly as before (the Phase 1 contract).
  2. LiveRunner gracefully falls back to FakeRunner whenever GPU/profiler
     tools are missing — this dev box has no AMD GPU, so every test here
     should exercise the fallback path.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest import mock

import pytest

from agent.schemas import RunMetrics, WorkloadConfig
from runner import profile_parser
from runner.protocol import FakeRunner, LiveRunner, _default_runner, gpu_available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _baseline_config() -> WorkloadConfig:
    return WorkloadConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        precision="fp16",
        batch_size=4,
        attention_impl="eager",
        dataloader_workers=0,
    )


# ---------------------------------------------------------------------------
# FakeRunner — unchanged contract
# ---------------------------------------------------------------------------


class TestFakeRunner:
    def test_matches_baseline_scenario(self):
        runner = FakeRunner()
        metrics = runner.run(_baseline_config(), steps=10)
        assert isinstance(metrics, RunMetrics)
        assert metrics.runner_kind == "fake"
        assert metrics.steps == 10
        # 01_baseline_bad fixture
        assert metrics.tokens_per_sec == pytest.approx(142.0)

    def test_steps_override_takes_precedence(self):
        runner = FakeRunner()
        metrics = runner.run(_baseline_config(), steps=99)
        assert metrics.steps == 99

    def test_default_metrics_when_no_match(self):
        runner = FakeRunner()
        # An unknown model_name forces the no-match path.
        cfg = _baseline_config().model_copy(update={"model_name": "unknown/model"})
        metrics = runner.run(cfg, steps=7)
        assert metrics.runner_kind == "fake"
        assert metrics.steps == 7
        assert any("FakeRunner" in w for w in metrics.warnings)

    def test_corpus_dir_missing_returns_default(self, tmp_path):
        runner = FakeRunner(corpus_dir=tmp_path / "nope")
        metrics = runner.run(_baseline_config(), steps=10)
        assert metrics.runner_kind == "fake"


# ---------------------------------------------------------------------------
# gpu_available — pure detection
# ---------------------------------------------------------------------------


class TestGpuAvailable:
    def test_no_rocprofv3(self):
        with mock.patch("runner.protocol.shutil.which", return_value=None):
            ok, reason = gpu_available()
        assert ok is False
        assert reason and "rocprofv3" in reason

    def test_no_amd_smi(self):
        def which(name):
            return "/usr/bin/rocprofv3" if name == "rocprofv3" else None

        with mock.patch("runner.protocol.shutil.which", side_effect=which):
            ok, reason = gpu_available()
        assert ok is False
        assert reason and "amd-smi" in reason

    def test_no_render_device(self):
        with mock.patch(
            "runner.protocol.shutil.which",
            side_effect=lambda name: f"/usr/bin/{name}",
        ), mock.patch("runner.protocol._has_render_device", return_value=False):
            ok, reason = gpu_available()
        assert ok is False
        assert reason and "renderD" in reason

    def test_all_present(self):
        with mock.patch(
            "runner.protocol.shutil.which",
            side_effect=lambda name: f"/usr/bin/{name}",
        ), mock.patch("runner.protocol._has_render_device", return_value=True):
            ok, reason = gpu_available()
        assert ok is True
        assert reason is None


# ---------------------------------------------------------------------------
# LiveRunner — must fall back on this no-GPU dev machine
# ---------------------------------------------------------------------------


class TestLiveRunnerFallback:
    def test_falls_back_when_gpu_unavailable(self):
        runner = LiveRunner()
        metrics = runner.run(_baseline_config(), steps=10)
        # On a laptop, gpu_available() returns False → FakeRunner path.
        assert metrics.runner_kind == "fake"
        # The warning must be the FIRST entry (LiveRunner prepends it).
        assert metrics.warnings, "LiveRunner must surface a fallback warning"
        assert "LiveRunner" in metrics.warnings[0]

    def test_falls_back_when_runner_script_missing(self, tmp_path):
        with mock.patch("runner.protocol.gpu_available", return_value=(True, None)):
            runner = LiveRunner(runner_script=tmp_path / "does_not_exist.sh")
            metrics = runner.run(_baseline_config(), steps=10)
        assert metrics.runner_kind == "fake"
        assert any("runner script not found" in w for w in metrics.warnings)

    def test_falls_back_when_runner_script_not_executable(self, tmp_path):
        script = tmp_path / "goblin_runner.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        # Deliberately don't chmod +x
        with mock.patch("runner.protocol.gpu_available", return_value=(True, None)):
            runner = LiveRunner(runner_script=script)
            metrics = runner.run(_baseline_config(), steps=10)
        assert metrics.runner_kind == "fake"
        assert any("not executable" in w for w in metrics.warnings)

    def test_falls_back_when_subprocess_returns_nonzero(self, tmp_path):
        script = tmp_path / "goblin_runner.sh"
        script.write_text("#!/usr/bin/env bash\nexit 7\n")
        script.chmod(0o755)
        with mock.patch("runner.protocol.gpu_available", return_value=(True, None)):
            runner = LiveRunner(runner_script=script)
            metrics = runner.run(_baseline_config(), steps=10)
        assert metrics.runner_kind == "fake"
        assert any("exited with code 7" in w for w in metrics.warnings)


# ---------------------------------------------------------------------------
# _default_runner — module-level factory
# ---------------------------------------------------------------------------


def test_default_runner_is_live_runner():
    runner = _default_runner()
    assert isinstance(runner, LiveRunner)


# ---------------------------------------------------------------------------
# profile_parser — graceful degradation when artefacts missing
# ---------------------------------------------------------------------------


class TestProfileParser:
    def test_empty_dir_returns_zero_metrics_with_warnings(self, tmp_path):
        metrics = profile_parser.parse(tmp_path, config=_baseline_config(), steps=10)
        assert metrics.tokens_per_sec == 0.0
        assert metrics.mfu_pct == 0.0
        assert metrics.gpu_util_pct == 0.0
        assert len(metrics.warnings) >= 3  # one warning per missing artefact

    def test_parses_synthetic_artefacts(self, tmp_path):
        # Minimal rocprofv3-shaped CSV
        trace = tmp_path / "trace.csv"
        with trace.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["KernelName", "DurationNs"])
            w.writerow(["aten::matmul (fp16)", 5_000_000])
            w.writerow(["aten::scaled_dot_product_attention", 3_000_000])
            w.writerow(["rccl_AllReduce", 1_000_000])
            w.writerow(["hipBLASLt_generic_gemm", 2_000_000])

        # Minimal torch.profiler chrome trace with embedded metadata
        torch_profile = {
            "metadata": {
                "tokens_per_sec": 142.0,
                "mfu_pct": 24.0,
                "pytorch_version": "2.3.0+rocm6.1",
                "step_time_seconds": 0.5,
                "host_busy_fraction": 0.6,
            },
            "traceEvents": [],
        }
        (tmp_path / "torch_profile.json").write_text(json.dumps(torch_profile))

        # Minimal amd-smi telemetry
        smi = tmp_path / "amd_smi.csv"
        with smi.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["VRAM_USED_GB", "GFX_ACTIVITY"])
            w.writerow(["72.0", "20.0"])  # < 30% util → triggers data_wait
            w.writerow(["75.0", "22.0"])

        metrics = profile_parser.parse(tmp_path, config=_baseline_config(), steps=10)
        assert metrics.tokens_per_sec == pytest.approx(142.0)
        assert metrics.mfu_pct == pytest.approx(24.0)
        assert metrics.hbm_peak_gb == pytest.approx(75.0)
        assert metrics.hbm_avg_gb == pytest.approx(73.5)
        # comm_excess detected (rccl kernel, 1 ms)
        assert metrics.waste_budget.comm_excess == pytest.approx(0.001)
        # data_wait triggered (gpu util < 30, host_busy > 0.5)
        assert metrics.waste_budget.data_wait > 0.0
        # precision_path triggered (config.precision='fp16' AND fp16 kernels present)
        assert metrics.waste_budget.precision_path > 0.0
        # kernel_shape: generic GEMM detected
        assert metrics.waste_budget.kernel_shape > 0.0
        # memory_headroom: 75 GB used << 70% × 192 GB = 134.4 GB → slack
        assert metrics.waste_budget.memory_headroom > 0.0

    def test_bf16_config_skips_precision_path(self, tmp_path):
        # Even with fp16-tagged kernels, a bf16 config means precision_path = 0
        # because the user is already on the optimal precision.
        trace = tmp_path / "trace.csv"
        with trace.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["KernelName", "DurationNs"])
            w.writerow(["matmul_fp16_kernel", 5_000_000])
        torch_profile = {
            "metadata": {
                "tokens_per_sec": 318.0,
                "mfu_pct": 51.0,
                "step_time_seconds": 0.3,
                "host_busy_fraction": 0.2,
            },
            "traceEvents": [],
        }
        (tmp_path / "torch_profile.json").write_text(json.dumps(torch_profile))
        smi = tmp_path / "amd_smi.csv"
        smi.write_text("VRAM_USED_GB,GFX_ACTIVITY\n168.0,86.0\n")

        bf16_config = _baseline_config().model_copy(update={"precision": "bf16"})
        metrics = profile_parser.parse(tmp_path, config=bf16_config, steps=50)
        assert metrics.waste_budget.precision_path == 0.0


# ---------------------------------------------------------------------------
# Caching — exercise the benchmark tool's cache layer
# ---------------------------------------------------------------------------


class TestBenchmarkCache:
    """The benchmark tool writes to the real bench_cache/ directory; isolate it."""

    @pytest.fixture(autouse=True)
    def _isolate_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.tools.benchmark._CACHE_DIR", tmp_path / "bench_cache")
        # Force ROCM_IMAGE_TAG to a known value so the key is reproducible.
        monkeypatch.setenv("ROCM_IMAGE_TAG", "test-tag")
        yield

    def test_cache_hit_on_second_call(self):
        from agent.tools.benchmark import _benchmark

        cfg = _baseline_config().model_dump()
        r1 = _benchmark(cfg, steps=50)
        assert r1.ok
        # Second call should HIT the cache and warn about it.
        r2 = _benchmark(cfg, steps=50)
        assert r2.ok
        assert any("cache hit" in w for w in r2.result["warnings"])

    def test_force_rerun_bypasses_cache(self):
        from agent.tools.benchmark import _benchmark

        cfg = _baseline_config().model_dump()
        _benchmark(cfg, steps=50)
        r2 = _benchmark(cfg, steps=50, force_rerun=True)
        assert r2.ok
        assert not any("cache hit" in w for w in r2.result["warnings"])

    def test_different_steps_invalidate_cache(self):
        from agent.tools.benchmark import _benchmark

        cfg = _baseline_config().model_dump()
        _benchmark(cfg, steps=50)
        r2 = _benchmark(cfg, steps=100)
        # Same config, different steps → different cache key → cold call.
        assert not any("cache hit" in w for w in r2.result["warnings"])

    def test_runner_script_change_invalidates_cache(self, tmp_path, monkeypatch):
        from agent.tools.benchmark import _benchmark

        cfg = _baseline_config().model_dump()
        _benchmark(cfg, steps=50)

        # Pretend the runner script changed by swapping the path the cache
        # key reads from. (Simulates "container/runner version bump".)
        fake_script = tmp_path / "different_runner.sh"
        fake_script.write_text("# different content\n")
        monkeypatch.setattr("agent.tools.benchmark._RUNNER_SCRIPT", fake_script)

        r2 = _benchmark(cfg, steps=50)
        assert not any("cache hit" in w for w in r2.result["warnings"])
