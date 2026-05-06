"""benchmark tool — full benchmark (default 50 steps), version-tagged cached.

Delegates to `runner.protocol.LiveRunner` (which auto-falls-back to FakeRunner
when the host has no GPU). Adds a content-addressed cache keyed on:

    sha256(canonical_config_json
           || workload_script_sha
           || rocm_image_tag
           || runner_script_sha)

so re-running the same config is free, but a stale cache entry from before
a runner-script edit or a container bump is automatically invalidated.

Cache files live under `bench_cache/<hash>.json`. Pass `force_rerun=True` in
the tool input to bypass the cache for a single call (architecture.md §6:
the Day-3 dry-run that confirms cached results haven't gone stale).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from agent.schemas import RunMetrics, ToolResult, WorkloadConfig
from agent.tools import Tool
from runner.protocol import _default_runner

_LOG = logging.getLogger(__name__)

_RUNNER = _default_runner()

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _REPO_ROOT / "bench_cache"
_WORKLOAD_SCRIPT = _REPO_ROOT / "workloads" / "train_llama3_lora.py"
_RUNNER_SCRIPT = _REPO_ROOT / "runner" / "goblin_runner.sh"


# ---------------------------------------------------------------------------
# Cache-key helpers
# ---------------------------------------------------------------------------


def _canonical_config_json(config: dict) -> str:
    """Stable JSON for hashing — deterministic key order, no whitespace drift."""
    # Round-trip through WorkloadConfig so unknown keys don't leak into the
    # hash and defaults are filled in consistently.
    workload = WorkloadConfig.model_validate(config)
    return json.dumps(workload.model_dump(), sort_keys=True, separators=(",", ":"))


def _file_sha(path: Path) -> str:
    if not path.exists():
        return "none"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _container_tag() -> str:
    """The ROCm container tag if running in CI/cloud, else `unknown`."""
    return os.environ.get("ROCM_IMAGE_TAG", "unknown")


def _cache_key(config: dict, steps: int) -> str:
    """Compose the cache key per architecture.md §6.

    `steps` is included because a 10-step profile and a 500-step benchmark
    are not the same artefact even with the same config. (Strictly speaking
    architecture.md §6 only mentions config + workload + container + runner;
    we add steps so profile_run-style short calls don't poison long-run
    benchmark cache entries.)
    """
    payload = (
        _canonical_config_json(config)
        + "|"
        + _file_sha(_WORKLOAD_SCRIPT)
        + "|"
        + _container_tag()
        + "|"
        + _file_sha(_RUNNER_SCRIPT)
        + "|steps="
        + str(steps)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.json"


def _read_cache(key: str) -> dict[str, Any] | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("benchmark: failed to read cache %s (%s); ignoring", path, exc)
        return None


def _write_cache(key: str, metrics: RunMetrics, config: dict, steps: int) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(key)
        # Persist the raw key tuple alongside the metrics so a human can
        # debug "why did this cache hit?" without re-hashing.
        payload = {
            "metrics": metrics.model_dump(),
            "key_components": {
                "config": json.loads(_canonical_config_json(config)),
                "steps": steps,
                "workload_script_sha": _file_sha(_WORKLOAD_SCRIPT),
                "rocm_image_tag": _container_tag(),
                "runner_script_sha": _file_sha(_RUNNER_SCRIPT),
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    except OSError as exc:
        _LOG.warning("benchmark: failed to write cache (%s); continuing", exc)


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------


def _benchmark(config: dict, steps: int = 50, force_rerun: bool = False) -> ToolResult:
    key = _cache_key(config, steps)

    if not force_rerun:
        cached = _read_cache(key)
        if cached is not None:
            metrics_dict = cached.get("metrics", cached)
            metrics = RunMetrics.model_validate(metrics_dict)
            metrics.warnings = ["benchmark: cache hit (use force_rerun=True to bypass)", *metrics.warnings]
            return ToolResult(ok=True, result=metrics.model_dump())

    workload = WorkloadConfig.model_validate(config)
    metrics = _RUNNER.run(workload, steps=steps)
    _write_cache(key, metrics, config, steps)
    return ToolResult(ok=True, result=metrics.model_dump())


BENCHMARK = Tool(
    name="benchmark",
    description=(
        "Full benchmark (default 50 steps). Same metric shape as profile_run "
        "but at production-scale step count. Result is cached by a version-"
        "tagged hash so re-runs of the same config are free. Use this AFTER "
        "propose_patch to validate the patched config — and call it once on "
        "the original config for before/after comparison."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "config": {"type": "object", "description": "WorkloadConfig dict."},
            "steps": {
                "type": "integer",
                "default": 50,
                "minimum": 10,
                "maximum": 500,
                "description": "Number of measured steps (after a 2-step warmup).",
            },
            "force_rerun": {
                "type": "boolean",
                "default": False,
                "description": (
                    "If true, bypass the bench_cache for this call and re-run "
                    "against the live runner. Used by the Day-3 dry-run pass "
                    "that confirms cached results haven't gone stale."
                ),
            },
        },
        "required": ["config"],
    },
    fn=_benchmark,
)
