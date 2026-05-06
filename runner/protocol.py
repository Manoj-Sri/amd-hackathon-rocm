"""RunnerProtocol — the seam between profile_run/benchmark and the actual GPU.

This is the testability fix from the brooks-audit (Warning #2): without this
abstraction, every change to a tool that touches profiling required an
MI300X cloud session. With it, Backend Lead develops on a laptop using
FakeRunner; Day-3 swaps in the real runner for the canonical demo.

Real implementations subclass `Runner` and call into goblin_runner.sh.
Tests and laptop dev use FakeRunner, which loads canned RunMetrics from
workloads/synthetic/.

`LiveRunner` is the production path: it shells out to `goblin_runner.sh`
(which itself wraps rocprofv3 + torch.profiler), parses the resulting
artefacts via `runner.profile_parser.parse`, and on ANY failure
(missing tools, no GPU, subprocess error) falls back to FakeRunner so the
demo still works on a laptop.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from agent.schemas import RunMetrics, WorkloadConfig

_LOG = logging.getLogger(__name__)


class Runner(Protocol):
    """Anything that can take a WorkloadConfig and produce RunMetrics."""

    def run(self, config: WorkloadConfig, steps: int) -> RunMetrics:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# GPU detection — used by LiveRunner to decide live vs. fallback.
# ---------------------------------------------------------------------------


def _has_render_device() -> bool:
    """At least one /dev/dri/renderD* device is present."""
    dri = Path("/dev/dri")
    if not dri.exists():
        return False
    try:
        return any(child.name.startswith("renderD") for child in dri.iterdir())
    except OSError:
        return False


def gpu_available() -> tuple[bool, str | None]:
    """Return `(ok, reason_if_missing)`.

    LiveRunner is only safe to invoke when ALL of the following hold:
      1. `rocprofv3` is on PATH (the kernel-trace driver).
      2. `amd-smi` is on PATH (HBM/power telemetry sampler).
      3. /dev/dri has at least one `renderD*` node (a real AMD GPU).

    If any check fails we fall back to FakeRunner with a clear warning.
    """
    if shutil.which("rocprofv3") is None:
        return False, "rocprofv3 not found on PATH"
    if shutil.which("amd-smi") is None:
        return False, "amd-smi not found on PATH"
    if not _has_render_device():
        return False, "no /dev/dri/renderD* device present"
    return True, None


# ---------------------------------------------------------------------------
# FakeRunner — loads canned RunMetrics from workloads/synthetic/.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LiveRunner — production path. Spawns goblin_runner.sh under rocprofv3.
# ---------------------------------------------------------------------------


# Defaults are pinned to the repo layout. Override via env vars in tests / CI.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RUNNER_SCRIPT = _REPO_ROOT / "runner" / "goblin_runner.sh"
_DEFAULT_USER_SCRIPT = _REPO_ROOT / "workloads" / "train_llama3_lora.py"


class LiveRunner:
    """Real-MI300X path: shells out to goblin_runner.sh and parses artefacts.

    Auto-falls-back to FakeRunner whenever the host can't actually run a live
    profile (missing rocprofv3/amd-smi, no AMD GPU, subprocess error, or
    parser failure). The fallback path is the demo safety net.

    Public API matches the `Runner` Protocol: `run(config, steps) -> RunMetrics`.
    """

    def __init__(
        self,
        runner_script: Path | str = _DEFAULT_RUNNER_SCRIPT,
        user_script: Path | str = _DEFAULT_USER_SCRIPT,
        timeout_seconds: int = 1800,
        fake_fallback: FakeRunner | None = None,
    ) -> None:
        self.runner_script = Path(runner_script)
        self.user_script = Path(user_script)
        self.timeout_seconds = timeout_seconds
        self._fake = fake_fallback or FakeRunner()

    # ------------------------------------------------------------------

    def run(self, config: WorkloadConfig, steps: int) -> RunMetrics:
        ok, reason = gpu_available()
        if not ok:
            return self._fallback(
                config,
                steps,
                f"LiveRunner: GPU/profiler unavailable ({reason}); using FakeRunner.",
            )

        # Sanity-check the runner script before spawning anything.
        if not self.runner_script.exists():
            return self._fallback(
                config,
                steps,
                f"LiveRunner: runner script not found at {self.runner_script}; using FakeRunner.",
            )
        if not os.access(self.runner_script, os.X_OK):
            return self._fallback(
                config,
                steps,
                f"LiveRunner: runner script {self.runner_script} not executable; using FakeRunner.",
            )

        # Late import — only needed on the live path. Keeps laptop-only test
        # runs from importing parser dependencies (csv stdlib is fine, but
        # this also keeps the dependency direction explicit).
        from runner import profile_parser

        with tempfile.TemporaryDirectory(prefix="goblin_run_") as out_dir_str:
            out_dir = Path(out_dir_str)

            env = os.environ.copy()
            env["USER_SCRIPT"] = str(self.user_script)
            env["OUT_DIR"] = str(out_dir)
            env["STEPS"] = str(steps)

            cmd = [str(self.runner_script)]
            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return self._fallback(
                    config,
                    steps,
                    f"LiveRunner: goblin_runner.sh timed out after "
                    f"{self.timeout_seconds}s; using FakeRunner.",
                )
            except OSError as exc:
                return self._fallback(
                    config,
                    steps,
                    f"LiveRunner: failed to spawn goblin_runner.sh ({exc}); using FakeRunner.",
                )

            if proc.returncode != 0:
                tail = (proc.stderr or "").strip().splitlines()[-5:]
                return self._fallback(
                    config,
                    steps,
                    "LiveRunner: goblin_runner.sh exited with "
                    f"code {proc.returncode}; using FakeRunner. stderr tail: {tail}",
                )

            try:
                metrics = profile_parser.parse(out_dir, config=config, steps=steps)
            except Exception as exc:  # pragma: no cover — defensive
                return self._fallback(
                    config,
                    steps,
                    f"LiveRunner: profile_parser.parse failed ({type(exc).__name__}: {exc}); "
                    "using FakeRunner.",
                )

            metrics.runner_kind = "live"
            return metrics

    # ------------------------------------------------------------------

    def _fallback(self, config: WorkloadConfig, steps: int, warning: str) -> RunMetrics:
        _LOG.warning(warning)
        metrics = self._fake.run(config, steps)
        # Make the fallback observable to upstream tools — they surface
        # warnings into the final report.
        metrics.warnings = [warning, *metrics.warnings]
        metrics.runner_kind = "fake"
        return metrics


# ---------------------------------------------------------------------------
# Module-level factory used by agent/tools/{profile_run,benchmark}.py.
# ---------------------------------------------------------------------------


def _default_runner() -> Runner:
    """Return the runner profile_run / benchmark should use by default.

    Always returns a `LiveRunner` — `LiveRunner.run` itself decides whether to
    actually invoke the GPU pipeline or fall back to FakeRunner. Centralising
    this here means the live-vs-fake decision lives in exactly one place.
    """
    return LiveRunner()
