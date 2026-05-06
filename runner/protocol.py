"""RunnerProtocol — the seam between profile_run/benchmark and the actual GPU.

This is the testability fix from the brooks-audit (Warning #2): without this
abstraction, every change to a tool that touches profiling required an
MI300X cloud session. With it, Backend Lead develops on a laptop using
FakeRunner; Day-3 swaps in the real runner for the canonical demo.

Real implementations subclass `Runner` and call into goblin_runner.sh.
Tests and laptop dev use FakeRunner, which loads canned RunMetrics from
workloads/synthetic/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from agent.schemas import RunMetrics, WorkloadConfig


class Runner(Protocol):
    """Anything that can take a WorkloadConfig and produce RunMetrics."""

    def run(self, config: WorkloadConfig, steps: int) -> RunMetrics:  # pragma: no cover
        ...


class FakeRunner:
    """Loads pre-recorded RunMetrics from workloads/synthetic/<scenario>/cached_metrics.json.

    The scenario is selected by matching `WorkloadConfig` fields against each
    synthetic scenario's `match` block in its manifest. If multiple scenarios
    match, the most specific one wins (highest number of matched keys).
    If none match, returns a generic baseline.

    This lets us:
      1. Develop the agent loop without an MI300X.
      2. Demo when MI300X cloud is unreachable (offline-replay lane).
      3. Run integration tests deterministically.
    """

    def __init__(self, corpus_dir: Path | str = "workloads/synthetic") -> None:
        self.corpus_dir = Path(corpus_dir)

    def run(self, config: WorkloadConfig, steps: int) -> RunMetrics:
        scenario = self._match_scenario(config)
        if scenario is None:
            return self._default_metrics(steps)

        metrics_path = scenario / "cached_metrics.json"
        if not metrics_path.exists():
            return self._default_metrics(steps)

        data = json.loads(metrics_path.read_text())
        # The cached file may not have steps populated; let the caller's
        # request take precedence so profile_run vs benchmark return as expected.
        data["steps"] = steps
        data["runner_kind"] = "fake"
        return RunMetrics.model_validate(data)

    # ------------------------------------------------------------------
    # Scenario matching
    # ------------------------------------------------------------------

    def _match_scenario(self, config: WorkloadConfig) -> Path | None:
        if not self.corpus_dir.exists():
            return None
        best: tuple[int, Path] | None = None
        for scenario_dir in sorted(self.corpus_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            manifest = scenario_dir / "manifest.json"
            if not manifest.exists():
                continue
            try:
                spec = json.loads(manifest.read_text())
            except json.JSONDecodeError:
                continue
            match_block = spec.get("match", {})
            score = self._score(config, match_block)
            if score < 0:
                continue
            if best is None or score > best[0]:
                best = (score, scenario_dir)
        return best[1] if best else None

    @staticmethod
    def _score(config: WorkloadConfig, match: dict) -> int:
        """Return number of keys that match, or -1 if any key conflicts."""
        cfg = config.model_dump()
        score = 0
        for key, expected in match.items():
            if cfg.get(key) != expected:
                return -1
            score += 1
        return score

    @staticmethod
    def _default_metrics(steps: int) -> RunMetrics:
        from agent.schemas import KernelEntry, WasteBudget

        return RunMetrics(
            steps=steps,
            tokens_per_sec=120.0,
            mfu_pct=22.0,
            hbm_peak_gb=72.0,
            hbm_avg_gb=58.0,
            gpu_util_pct=45.0,
            top_kernels=[
                KernelEntry(name="aten::matmul", pct_time=42.0),
                KernelEntry(name="aten::scaled_dot_product_attention", pct_time=18.0),
                KernelEntry(name="aten::layer_norm", pct_time=7.0),
            ],
            attention_kernel_loaded="sdpa",
            waste_budget=WasteBudget(
                useful_gpu=0.55,
                data_wait=0.18,
                host_gap=0.07,
                comm_excess=0.0,
                memory_headroom=0.10,
                precision_path=0.06,
                kernel_shape=0.04,
            ),
            warnings=["FakeRunner: no matching scenario, returning generic baseline."],
            runner_kind="fake",
        )
