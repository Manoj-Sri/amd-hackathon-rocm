"""profile_parser — turn goblin_runner.sh artefacts into RunMetrics.

Reads three files written by goblin_runner.sh (architecture.md §5):

    <out_dir>/trace.csv          rocprofv3 kernel trace
    <out_dir>/torch_profile.json torch.profiler chrome trace
    <out_dir>/amd_smi.csv        amd-smi telemetry, ~200 ms cadence

and produces one `RunMetrics` with a populated `WasteBudget`.

Each waste-budget bucket is computed with a deliberately simple, documented
heuristic — these are best-effort signals for the agent, NOT measured
ground-truth. See the docstring on `_waste_budget()` for the per-bucket
formula and its known failure modes.

This module is import-tolerant on machines without the rocprofv3 stack —
it only reads files. Missing or unparseable files degrade individual
metrics to zero and append a warning to RunMetrics.warnings rather than
raising. LiveRunner ultimately decides what to do with parse failures.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.schemas import KernelEntry, RunMetrics, WasteBudget, WorkloadConfig

_LOG = logging.getLogger(__name__)


# MI300X has 192 GB HBM3. We treat sustained <70% utilisation as headroom.
_HBM_TOTAL_GB = 192.0
_HBM_HEALTHY_TARGET = 0.70

# Kernel-name patterns used by the heuristics.
#
# We use `(?<![A-Za-z0-9])` / `(?![A-Za-z0-9])` instead of `\b` so that
# underscores act as token separators — kernel names like
# `rccl_AllReduce` and `hipBLASLt_generic_gemm` should match.
_BOUND_L = r"(?<![A-Za-z0-9])"
_BOUND_R = r"(?![A-Za-z0-9])"
_RCCL_PATTERN = re.compile(
    _BOUND_L + r"(rccl|nccl|all[_-]?reduce|broadcast|reduce[_-]?scatter)" + _BOUND_R, re.I
)
_GEMM_PATTERN = re.compile(_BOUND_L + r"(gemm|matmul|hgemm|sgemm|hipblaslt)" + _BOUND_R, re.I)
_GENERIC_GEMM_PATTERN = re.compile(
    _BOUND_L + r"(generic|fallback|naive|reference)" + _BOUND_R, re.I
)
_FP16_PATTERN = re.compile(_BOUND_L + r"(fp16|half|f16)" + _BOUND_R, re.I)
_BF16_PATTERN = re.compile(_BOUND_L + r"(bf16|bfloat16)" + _BOUND_R, re.I)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse(
    out_dir: Path,
    config: WorkloadConfig | None = None,
    steps: int = 10,
) -> RunMetrics:
    """Build a `RunMetrics` from goblin_runner.sh artefacts in `out_dir`.

    `config` is used by some heuristics (e.g. precision_path skips the
    estimate when the config is already bf16). It can be None at the cost
    of a more conservative waste-budget estimate.

    Always returns a RunMetrics object — individual metrics degrade to
    zero with a warning rather than raising, because LiveRunner ultimately
    decides whether parse failures should trigger fallback.
    """
    warnings: list[str] = []

    kernels = _read_kernels(out_dir / "trace.csv", warnings)
    torch_summary = _read_torch_profile(out_dir / "torch_profile.json", warnings)
    smi = _read_amd_smi(out_dir / "amd_smi.csv", warnings)

    top_kernels = _top_kernels(kernels, top_n=5)
    gpu_util_pct = smi.gpu_util_pct if smi.gpu_util_pct is not None else _gpu_util_from_kernels(
        kernels, torch_summary
    )

    waste_budget = _waste_budget(
        kernels=kernels,
        torch_summary=torch_summary,
        smi=smi,
        config=config,
    )

    attention_kernel_loaded = _detect_attention_kernel(kernels)

    return RunMetrics(
        steps=steps,
        tokens_per_sec=torch_summary.tokens_per_sec or 0.0,
        mfu_pct=torch_summary.mfu_pct or 0.0,
        hbm_peak_gb=smi.hbm_peak_gb or 0.0,
        hbm_avg_gb=smi.hbm_avg_gb or 0.0,
        gpu_util_pct=gpu_util_pct,
        top_kernels=top_kernels,
        attention_kernel_loaded=attention_kernel_loaded,
        waste_budget=waste_budget,
        warnings=warnings,
        rocm_version=smi.rocm_version or "unknown",
        pytorch_version=torch_summary.pytorch_version or "unknown",
        runner_kind="live",
    )


# ---------------------------------------------------------------------------
# Internal data carriers
# ---------------------------------------------------------------------------


@dataclass
class _Kernel:
    name: str
    duration_ns: int
    """Kernel duration in nanoseconds."""

    is_collective: bool = False
    is_gemm: bool = False
    is_generic_gemm: bool = False


@dataclass
class _TorchSummary:
    tokens_per_sec: float | None = None
    mfu_pct: float | None = None
    pytorch_version: str | None = None
    step_time_seconds: float | None = None
    """Average wall-clock seconds per training step (used by waste budget)."""

    host_busy_fraction: float | None = None
    """Fraction of step time the host (CPU) spent doing non-launch work.
    Heuristic: `cpu_self_time / step_time` reported by torch.profiler."""


@dataclass
class _SmiSummary:
    hbm_peak_gb: float | None = None
    hbm_avg_gb: float | None = None
    gpu_util_pct: float | None = None
    rocm_version: str | None = None


# ---------------------------------------------------------------------------
# trace.csv (rocprofv3) → list[_Kernel]
# ---------------------------------------------------------------------------


def _read_kernels(path: Path, warnings: list[str]) -> list[_Kernel]:
    """Parse rocprofv3 kernel-trace CSV into a list of `_Kernel` records.

    rocprofv3 column names vary slightly by version. We look up by header and
    accept the common aliases; if the file is missing or unparseable we
    return an empty list and append a warning so the caller can decide.
    """
    if not path.exists():
        warnings.append(f"profile_parser: kernel trace not found at {path}")
        return []

    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                warnings.append(f"profile_parser: empty kernel trace at {path}")
                return []
            name_col = _pick_column(reader.fieldnames, ["KernelName", "Kernel_Name", "kernel_name", "Name"])
            start_col = _pick_column(reader.fieldnames, ["BeginNs", "start_ns", "BeginNS", "Start"])
            end_col = _pick_column(reader.fieldnames, ["EndNs", "end_ns", "EndNS", "End"])
            duration_col = _pick_column(reader.fieldnames, ["DurationNs", "duration_ns", "Duration"])

            kernels: list[_Kernel] = []
            for row in reader:
                name = (row.get(name_col) or "").strip() if name_col else ""
                if not name:
                    continue
                duration = _row_duration_ns(row, duration_col, start_col, end_col)
                if duration <= 0:
                    continue
                kernels.append(
                    _Kernel(
                        name=name,
                        duration_ns=duration,
                        is_collective=bool(_RCCL_PATTERN.search(name)),
                        is_gemm=bool(_GEMM_PATTERN.search(name)),
                        is_generic_gemm=bool(
                            _GEMM_PATTERN.search(name) and _GENERIC_GEMM_PATTERN.search(name)
                        ),
                    )
                )
            return kernels
    except (OSError, csv.Error) as exc:
        warnings.append(f"profile_parser: failed to read kernel trace ({exc})")
        return []


def _pick_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    """Pick the first matching column name, with three fallback tiers:
        1. Exact match.
        2. Case-insensitive exact match.
        3. Substring match (case-insensitive) — every token of any candidate
           must appear somewhere in the field name. Tolerates the column-name
           drift between rocprofv3 / amd-smi versions (e.g. `VRAM_USED` vs
           `vram_used_mb` vs `VRAM USED MB`).
    """
    for c in candidates:
        if c in fieldnames:
            return c
    lower = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    # Substring tier: split each candidate on _/space, require all tokens
    # appear in the (lowercased) field name. Avoids matching too eagerly
    # by requiring every token of the candidate.
    for c in candidates:
        tokens = [t for t in c.lower().replace("_", " ").split() if t]
        if not tokens:
            continue
        for fname in fieldnames:
            fl = fname.lower()
            if all(t in fl for t in tokens):
                return fname
    return None


def _row_duration_ns(
    row: dict[str, str], duration_col: str | None, start_col: str | None, end_col: str | None
) -> int:
    if duration_col and row.get(duration_col):
        try:
            return int(float(row[duration_col]))
        except ValueError:
            return 0
    if start_col and end_col and row.get(start_col) and row.get(end_col):
        try:
            return int(float(row[end_col])) - int(float(row[start_col]))
        except ValueError:
            return 0
    return 0


def _top_kernels(kernels: list[_Kernel], top_n: int) -> list[KernelEntry]:
    if not kernels:
        return []
    total = sum(k.duration_ns for k in kernels) or 1
    by_name: dict[str, int] = {}
    for k in kernels:
        by_name[k.name] = by_name.get(k.name, 0) + k.duration_ns
    ranked = sorted(by_name.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [KernelEntry(name=name, pct_time=ns / total * 100.0) for name, ns in ranked]


def _detect_attention_kernel(kernels: list[_Kernel]) -> str:
    for k in kernels:
        n = k.name.lower()
        if "flash_attn" in n and "rocm" in n:
            return "flash_rocm"
        if "flash" in n and "attn" in n:
            return "flash"
        if "scaled_dot_product_attention" in n:
            return "sdpa"
    if kernels:
        return "eager"  # nothing flash-shaped; default conservative label
    return "unknown"


def _gpu_util_from_kernels(kernels: list[_Kernel], torch_summary: _TorchSummary) -> float:
    """Fallback GPU util when amd-smi is missing.

    util ≈ sum(kernel duration) / total wall-clock step time.
    """
    if not kernels or torch_summary.step_time_seconds in (None, 0):
        return 0.0
    total_kernel_ns = sum(k.duration_ns for k in kernels)
    wall_ns = torch_summary.step_time_seconds * 1e9
    if wall_ns <= 0:
        return 0.0
    return min(100.0, total_kernel_ns / wall_ns * 100.0)


# ---------------------------------------------------------------------------
# torch_profile.json (torch.profiler chrome trace) → _TorchSummary
# ---------------------------------------------------------------------------


def _read_torch_profile(path: Path, warnings: list[str]) -> _TorchSummary:
    """Pull tokens/sec, MFU, and step timing from a torch.profiler artefact.

    The user script (workloads/train_qwen_lora.py in Phase 3) is
    responsible for embedding `tokens_per_sec`, `mfu_pct`, `pytorch_version`
    and `step_time_seconds` in the trace as `metadata` events. If those are
    missing, we estimate `step_time_seconds` from the total trace duration.
    """
    summary = _TorchSummary()
    if not path.exists():
        warnings.append(f"profile_parser: torch profile not found at {path}")
        return summary
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"profile_parser: failed to read torch profile ({exc})")
        return summary

    # torch.profiler chrome trace is `{"traceEvents": [...], "metadata": {...}}`
    metadata = data.get("metadata") if isinstance(data, dict) else None
    if isinstance(metadata, dict):
        summary.tokens_per_sec = _coerce_float(metadata.get("tokens_per_sec"))
        summary.mfu_pct = _coerce_float(metadata.get("mfu_pct"))
        summary.pytorch_version = metadata.get("pytorch_version") or metadata.get("torch_version")
        summary.step_time_seconds = _coerce_float(metadata.get("step_time_seconds"))
        summary.host_busy_fraction = _coerce_float(metadata.get("host_busy_fraction"))

    events = data.get("traceEvents") if isinstance(data, dict) else None
    if isinstance(events, list):
        if summary.step_time_seconds is None:
            summary.step_time_seconds = _step_time_from_events(events)
        if summary.host_busy_fraction is None:
            summary.host_busy_fraction = _host_busy_from_events(events)

    return summary


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _step_time_from_events(events: list[dict]) -> float | None:
    """Estimate per-step wall-clock seconds from chrome-trace duration events.

    Looks for `name == "ProfilerStep#*"` complete events; falls back to the
    overall trace span if those aren't present.
    """
    durations: list[float] = []
    overall_start: float | None = None
    overall_end: float | None = None
    for ev in events:
        if not isinstance(ev, dict):
            continue
        name = ev.get("name", "")
        ts = ev.get("ts")
        dur = ev.get("dur")
        if isinstance(name, str) and name.startswith("ProfilerStep") and isinstance(dur, (int, float)):
            durations.append(float(dur) / 1e6)  # us → seconds
        if isinstance(ts, (int, float)) and isinstance(dur, (int, float)):
            start = float(ts)
            end = start + float(dur)
            overall_start = start if overall_start is None else min(overall_start, start)
            overall_end = end if overall_end is None else max(overall_end, end)
    if durations:
        return sum(durations) / len(durations)
    if overall_start is not None and overall_end is not None and overall_end > overall_start:
        return (overall_end - overall_start) / 1e6
    return None


def _host_busy_from_events(events: list[dict]) -> float | None:
    """Heuristic: cpu_op event time / total event span.

    Used by the data_wait waste bucket to disambiguate "GPU idle because the
    host is busy preparing the next batch" from "GPU idle because nothing
    is running anywhere".
    """
    cpu_op_us = 0.0
    span_min: float | None = None
    span_max: float | None = None
    for ev in events:
        if not isinstance(ev, dict):
            continue
        cat = ev.get("cat", "")
        ts = ev.get("ts")
        dur = ev.get("dur")
        if not (isinstance(ts, (int, float)) and isinstance(dur, (int, float))):
            continue
        ts_f = float(ts)
        dur_f = float(dur)
        end = ts_f + dur_f
        span_min = ts_f if span_min is None else min(span_min, ts_f)
        span_max = end if span_max is None else max(span_max, end)
        if isinstance(cat, str) and "cpu_op" in cat.lower():
            cpu_op_us += dur_f
    if span_min is None or span_max is None or span_max <= span_min:
        return None
    span = span_max - span_min
    if span <= 0:
        return None
    return min(1.0, cpu_op_us / span)


# ---------------------------------------------------------------------------
# amd_smi.csv → _SmiSummary
# ---------------------------------------------------------------------------


def _read_amd_smi(path: Path, warnings: list[str]) -> _SmiSummary:
    """Aggregate amd-smi polling output into HBM peak/avg + GPU util."""
    summary = _SmiSummary()
    if not path.exists():
        warnings.append(f"profile_parser: amd-smi telemetry not found at {path}")
        return summary

    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                warnings.append(f"profile_parser: empty amd-smi telemetry at {path}")
                return summary
            hbm_col = _pick_column(
                reader.fieldnames,
                [
                    # ROCm 7.x amd-smi --vram-usage emits "VRAM_USED" or
                    # "vram_used_mb" depending on minor version
                    "VRAM_USED_MB",
                    "vram_used_mb",
                    "VRAM_USED",
                    "VRAM_USED_GB",
                    "vram_used",
                    "VRAM Used",
                    # Older rocm 6.x naming
                    "MEM_USED",
                    "mem_used",
                ],
            )
            util_col = _pick_column(
                reader.fieldnames,
                [
                    # ROCm 7.x: --gfx flag → "gfx_util" / "GFX_UTIL"
                    "GFX_UTIL",
                    "gfx_util",
                    "GFX_UTILIZATION",
                    "gfx_utilization",
                    # rocm 6.x and older
                    "GFX_ACTIVITY",
                    "gfx_activity",
                    "GPU_USE",
                    "GFX %",
                    "Util",
                ],
            )
            rocm_col = _pick_column(reader.fieldnames, ["ROCM_VERSION", "rocm_version"])

            hbm_samples: list[float] = []
            util_samples: list[float] = []
            for row in reader:
                if hbm_col:
                    hbm_gb = _hbm_to_gb(row.get(hbm_col))
                    if hbm_gb is not None:
                        hbm_samples.append(hbm_gb)
                if util_col:
                    util = _coerce_float(row.get(util_col))
                    if util is not None:
                        util_samples.append(min(100.0, util))
                if rocm_col and summary.rocm_version is None:
                    val = (row.get(rocm_col) or "").strip()
                    if val:
                        summary.rocm_version = val
            if hbm_samples:
                summary.hbm_peak_gb = max(hbm_samples)
                summary.hbm_avg_gb = sum(hbm_samples) / len(hbm_samples)
            if util_samples:
                summary.gpu_util_pct = sum(util_samples) / len(util_samples)
            return summary
    except (OSError, csv.Error) as exc:
        warnings.append(f"profile_parser: failed to read amd-smi telemetry ({exc})")
        return summary


def _hbm_to_gb(raw: str | None) -> float | None:
    """amd-smi sometimes reports VRAM in MB, sometimes in GB. Heuristic: if
    the number is > 1024 we assume MB and divide."""
    if not raw:
        return None
    try:
        v = float(str(raw).strip().replace("MB", "").replace("GB", "").replace(",", ""))
    except ValueError:
        return None
    if v > 1024.0:
        return v / 1024.0
    return v


# ---------------------------------------------------------------------------
# Waste-budget heuristics
# ---------------------------------------------------------------------------


def _waste_budget(
    *,
    kernels: list[_Kernel],
    torch_summary: _TorchSummary,
    smi: _SmiSummary,
    config: WorkloadConfig | None,
) -> WasteBudget:
    """Decompose step time into the seven WasteBudget buckets (architecture.md §3).

    These are HEURISTICS, not measurements. Each bucket is in seconds-per-step
    so they can be summed against `step_time_seconds`. If we can't observe a
    bucket we set it to 0 — `evidence_coverage` in `propose_patch` then
    discounts confidence accordingly.

    Per-bucket logic:

      data_wait
        Fraction of step time where GPU util was below 30% AND the host was
        busy (host_busy_fraction > 0.5). Maps to dataloader / H2D copy stalls.

      precision_path
        If the user is already on bf16/fp8 we skip this bucket (recovery is 0).
        Otherwise we estimate from kernel names: time spent in fp16-tagged
        GEMMs is the recoverable surface; bf16-tagged kernels are not.

      kernel_shape
        Fraction of total GEMM kernel time spent on kernels whose names
        match `generic|fallback|naive|reference` — i.e. hipBLASLt/MIOpen
        couldn't pick a tuned tile size and fell back to a slow path.

      host_gap
        Time the GPU was idle while the host was NOT busy either — pure
        launch latency / eager-mode kernel gaps. We approximate as
        `(1 - gpu_util) * (1 - host_busy)` × step_time.

      comm_excess
        Sum of all collective-kernel duration (anything matching the rccl
        pattern). Treated as 100% recoverable elsewhere — the rule decides
        recovery_fraction.

      memory_headroom
        `(192 - hbm_peak) / 192 × small_constant`. Only counts the headroom
        that exceeded a healthy 70% target — running at 60% of HBM costs us
        roughly 0.07 of step time worth of optimisation surface.

      useful_gpu
        Whatever step time is left. Will be roughly the busy GPU time minus
        comm_excess and the kernel-shape penalty.
    """
    step_t = torch_summary.step_time_seconds or 0.0
    gpu_util = (smi.gpu_util_pct or 0.0) / 100.0  # 0..1
    host_busy = torch_summary.host_busy_fraction or 0.0
    if step_t <= 0:
        return WasteBudget()

    # data_wait: GPU under-utilised AND host busy → dataloader bottleneck.
    if gpu_util < 0.30 and host_busy > 0.5:
        data_wait = step_t * (1.0 - gpu_util) * host_busy
    else:
        data_wait = 0.0

    # host_gap: GPU idle while host idle too → launch latency / kernel gaps.
    host_gap = step_t * (1.0 - gpu_util) * (1.0 - host_busy)

    # comm_excess: total time in collective kernels (in seconds).
    comm_excess_ns = sum(k.duration_ns for k in kernels if k.is_collective)
    comm_excess = comm_excess_ns / 1e9

    # kernel_shape: fraction of GEMM time spent on un-tuned/generic kernels.
    gemm_total_ns = sum(k.duration_ns for k in kernels if k.is_gemm) or 1
    generic_gemm_ns = sum(k.duration_ns for k in kernels if k.is_generic_gemm)
    kernel_shape = (generic_gemm_ns / gemm_total_ns) * step_t * gpu_util  # cap at "real" GPU time

    # precision_path: only meaningful if config is fp16/fp32. Estimate from
    # kernel name tags. If config is bf16+ we leave as 0 — already optimal.
    precision_path = 0.0
    if config is None or config.precision in {"fp16", "fp32"}:
        fp16_ns = sum(k.duration_ns for k in kernels if _FP16_PATTERN.search(k.name))
        bf16_ns = sum(k.duration_ns for k in kernels if _BF16_PATTERN.search(k.name))
        denom = fp16_ns + bf16_ns
        if denom > 0:
            # Fraction of compute still on fp16. On MI300X bf16 is faster +
            # more numerically stable, so this fraction is the recoverable
            # precision_path surface.
            precision_path = (fp16_ns / denom) * step_t * gpu_util * 0.10

    # memory_headroom: headroom past the 70% healthy target × small constant.
    memory_headroom = 0.0
    hbm_peak = smi.hbm_peak_gb
    if hbm_peak is not None and hbm_peak > 0:
        utilisation = hbm_peak / _HBM_TOTAL_GB
        if utilisation < _HBM_HEALTHY_TARGET:
            slack = (_HBM_HEALTHY_TARGET - utilisation) / _HBM_HEALTHY_TARGET
            # Small constant: HBM slack is real but only enables a fraction of
            # potential gain (you still need a larger batch to use it). 0.05
            # of step time per "unit of slack" is a conservative anchor.
            memory_headroom = slack * step_t * 0.05

    # useful_gpu: everything else. Clamp to >= 0.
    spent = data_wait + host_gap + comm_excess + kernel_shape + precision_path + memory_headroom
    useful_gpu = max(0.0, step_t - spent)

    return WasteBudget(
        useful_gpu=useful_gpu,
        data_wait=data_wait,
        host_gap=host_gap,
        comm_excess=comm_excess,
        memory_headroom=memory_headroom,
        precision_path=precision_path,
        kernel_shape=kernel_shape,
    )
