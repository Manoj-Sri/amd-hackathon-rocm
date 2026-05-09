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
_DEFAULT_USER_SCRIPT = _REPO_ROOT / "workloads" / "train_qwen_lora.py"
_FAILURE_ARCHIVE_ROOT = _REPO_ROOT / "bench_cache"


def _archive_failure(out_dir: Path, proc: subprocess.CompletedProcess) -> Path:
    """Copy a failed runner's out_dir into bench_cache/last_runner_failure_<ts>/
    along with the subprocess's captured stdout/stderr. The directory survives
    after the tempdir cleanup so the user can `tail -n 100 stderr.log` etc.
    """
    import shutil
    import time

    ts = time.strftime("%Y%m%dT%H%M%S")
    dest = _FAILURE_ARCHIVE_ROOT / f"last_runner_failure_{ts}"
    try:
        dest.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            for child in out_dir.iterdir():
                target = dest / child.name
                if child.is_dir():
                    shutil.copytree(child, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(child, target)
        # Also persist the subprocess's own captured output — these are what
        # goblin_runner.sh's failure trap dumped.
        (dest / "subprocess_stdout.log").write_text(proc.stdout or "")
        (dest / "subprocess_stderr.log").write_text(proc.stderr or "")
        (dest / "subprocess_returncode").write_text(str(proc.returncode))
    except OSError as exc:
        _LOG.warning("LiveRunner: could not archive failure logs (%s)", exc)
        return dest
    return dest


def _config_to_patch_env(config: WorkloadConfig) -> dict[str, str]:
    """Translate WorkloadConfig fields into ``GOBLIN_PATCH_*`` env vars that
    the workload subprocess reads via ``workloads._runtime.read_patch_overrides``.

    This is the mechanism that makes ``benchmark(config=patched_config)``
    actually run with the patched config instead of running the workload's
    hardcoded values twice. Each field whose value differs from the
    ``WorkloadConfig`` schema default flows through as one env var; fields
    matching the default are SKIPPED so we don't accidentally clobber a
    workload's actual literal in the case where parse_config failed to
    extract that field (config would then carry the schema default, not the
    workload's real value, and emitting it would force the workload to a
    wrong setting).

    Returned mapping is always string→string for direct insertion into
    ``subprocess.run(env=...)``.
    """
    # Schema-default reference. We compare against this to decide whether to
    # emit an override — a field at its default likely wasn't extracted, and
    # silencing it lets the workload's actual literal stand.
    schema_default = WorkloadConfig(model_name="__goblin_default_marker__")

    patch_env: dict[str, str] = {}

    if config.precision and config.precision != schema_default.precision:
        patch_env["GOBLIN_PATCH_PRECISION"] = str(config.precision)
    if (
        config.attention_impl
        and config.attention_impl != schema_default.attention_impl
    ):
        patch_env["GOBLIN_PATCH_ATTENTION_IMPL"] = str(config.attention_impl)
    if (
        config.batch_size is not None
        and config.batch_size != schema_default.batch_size
    ):
        patch_env["GOBLIN_PATCH_BATCH_SIZE"] = str(config.batch_size)
    if (
        config.grad_accum_steps is not None
        and config.grad_accum_steps != schema_default.grad_accum_steps
    ):
        patch_env["GOBLIN_PATCH_GRAD_ACCUM_STEPS"] = str(config.grad_accum_steps)
    if (
        config.dataloader_workers is not None
        and config.dataloader_workers != schema_default.dataloader_workers
    ):
        patch_env["GOBLIN_PATCH_DATALOADER_WORKERS"] = str(config.dataloader_workers)
    if (
        config.dataloader_pin_memory is not None
        and config.dataloader_pin_memory != schema_default.dataloader_pin_memory
    ):
        patch_env["GOBLIN_PATCH_DATALOADER_PIN_MEMORY"] = str(
            config.dataloader_pin_memory
        ).lower()
    if (
        config.dataloader_persistent_workers is not None
        and config.dataloader_persistent_workers
        != schema_default.dataloader_persistent_workers
    ):
        patch_env["GOBLIN_PATCH_DATALOADER_PERSISTENT_WORKERS"] = str(
            config.dataloader_persistent_workers
        ).lower()
    if (
        config.gradient_checkpointing is not None
        and config.gradient_checkpointing != schema_default.gradient_checkpointing
    ):
        patch_env["GOBLIN_PATCH_GRADIENT_CHECKPOINTING"] = str(
            config.gradient_checkpointing
        ).lower()
    if (
        config.torch_compile is not None
        and config.torch_compile != schema_default.torch_compile
    ):
        patch_env["GOBLIN_PATCH_TORCH_COMPILE"] = str(config.torch_compile).lower()
    return patch_env


def _default_runner_timeout_seconds() -> int:
    """Resolve the LiveRunner subprocess timeout from env, with safe defaults.

    Reads ``GOBLIN_RUNNER_TIMEOUT_SECONDS`` and validates it as a positive
    int. Anything missing, empty, non-numeric, or non-positive falls back to
    1800 seconds (30 minutes).

    Why this helper exists: live MI300X audits sometimes overshoot the old
    600s default — model download on a cold cache, ROCm kernel JIT on the
    first step, or torch silently running on CPU after a botched pip
    install. Operators need a knob to extend the budget without editing
    code; this turns it into a single env var.
    """
    raw = os.environ.get("GOBLIN_RUNNER_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return 1800
    try:
        val = int(raw)
    except ValueError:
        _LOG.warning(
            "LiveRunner: GOBLIN_RUNNER_TIMEOUT_SECONDS=%r is not an int; "
            "falling back to 1800s.",
            raw,
        )
        return 1800
    if val <= 0:
        _LOG.warning(
            "LiveRunner: GOBLIN_RUNNER_TIMEOUT_SECONDS=%d is not positive; "
            "falling back to 1800s.",
            val,
        )
        return 1800
    return val


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
        timeout_seconds: int | None = None,
        fake_fallback: FakeRunner | None = None,
    ) -> None:
        # `timeout_seconds=None` (the default) means "consult
        # GOBLIN_RUNNER_TIMEOUT_SECONDS, then fall back to 1800s (30 min)".
        # Explicit ints from callers/tests still win.
        #
        # 30 min is generous on a healthy MI300X — a 50-step benchmark of
        # Qwen2.5-7B LoRA at bs=1/seq=512 finishes in 2–5 min when torch is
        # correctly using ROCm. The extra headroom absorbs cold model
        # downloads, kernel JIT on first run, and slow CPU-fallback hiccups
        # if torch ends up CPU-only. If your workload needs longer, bump
        # the env var:
        #     export GOBLIN_RUNNER_TIMEOUT_SECONDS=3600
        self.runner_script = Path(runner_script)
        self.user_script = Path(user_script)
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None
            else _default_runner_timeout_seconds()
        )
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
            # Inject patch-override env vars so the workload subprocess can
            # actually apply the agent's proposed config (precision, attention
            # impl, dataloader settings, etc.) — without this, benchmark calls
            # with different `config` arguments produced identical results
            # because the workload script hardcoded its values and ignored the
            # `config` parameter entirely. See workloads/_runtime.py
            # `read_patch_overrides` for the consumer side.
            for var, value in _config_to_patch_env(config).items():
                env[var] = value

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
                # Archive the full out_dir to bench_cache/last_runner_failure_<ts>/
                # so the user can inspect stdout.log / stderr.log / amd_smi.err
                # after the tempdir cleanup. The path goes into the warning
                # message so it's surfaced through ToolResult.warnings.
                archive_path = _archive_failure(out_dir, proc)
                stderr_tail = (proc.stderr or "").strip().splitlines()[-15:]
                stdout_tail = (proc.stdout or "").strip().splitlines()[-5:]
                return self._fallback(
                    config,
                    steps,
                    "LiveRunner: goblin_runner.sh exited with "
                    f"code {proc.returncode}; using FakeRunner. "
                    f"Failure logs archived at {archive_path}. "
                    f"stderr tail: {stderr_tail}. "
                    f"stdout tail: {stdout_tail}.",
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
