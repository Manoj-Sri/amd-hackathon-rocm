"""FastAPI server for GPU Goblin.

One audit endpoint plus a health probe. Streams the agent loop's `SSEEvent`s
to the UI via Server-Sent Events. CORS is wide open because Streamlit runs on
a different port — fine for a hackathon.

The agent runs on Qwen via Hugging Face Inference Providers. HF_TOKEN is
read at startup; if it's missing the server still starts (so the offline-
replay UI lane keeps working) but `/audit` yields a single error event.
We never crash on missing keys.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agent.backends import active_backend_name
from agent.loop import run_audit
from agent.schemas import SSEEvent
from agent.tools import ALL_TOOLS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AUTO_TUNE_SCRIPT = _REPO_ROOT / "scripts" / "auto_tune.py"

app = FastAPI(title="GPU Goblin Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _has_hf_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness + tool inventory + active backend. UI uses this to confirm
    the agent is reachable and configured."""
    name = active_backend_name()
    base = {
        "ok": True,
        "tools": [t.name for t in ALL_TOOLS],
        "backend": name,
    }
    if name == "qwen-vllm":
        base.update(
            {
                "model": os.environ.get(
                    "GOBLIN_QWEN_VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"
                ),
                "vllm_url": os.environ.get(
                    "GOBLIN_QWEN_VLLM_URL", "http://localhost:8000/v1"
                ),
                "has_api_key": True,  # vLLM doesn't require one by default
            }
        )
    else:
        base.update(
            {
                "model": os.environ.get(
                    "GOBLIN_QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct"
                ),
                "provider": os.environ.get("GOBLIN_QWEN_PROVIDER", "auto"),
                "has_api_key": _has_hf_token(),
            }
        )
    return base


async def _stream_audit(file_path: str) -> AsyncIterator[dict]:
    """Bridge `run_audit`'s SSEEvent generator into the dict shape that
    sse-starlette expects. Each yielded dict becomes one `data: ...\\n\\n`
    SSE message.
    """
    if not _has_hf_token():
        # Surface a clean error instead of letting the loop crash on missing key.
        yield {
            "data": SSEEvent(
                type="error",
                data={
                    "message": (
                        "HF_TOKEN not set on the server — Qwen agent loop is "
                        "unavailable. Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) "
                        "or use the offline-replay UI lane."
                    )
                },
            ).model_dump_json()
        }
        return

    try:
        async for event in run_audit(file_path):
            yield {"data": event.model_dump_json()}
    except Exception as exc:  # defence in depth — run_audit already wraps itself
        yield {
            "data": SSEEvent(
                type="error", data={"message": f"server: {exc}"}
            ).model_dump_json()
        }


@app.post("/audit")
async def audit(file: UploadFile = File(...)) -> EventSourceResponse:
    """Accept a multipart file upload and stream the agent's audit events.

    The uploaded file is saved to a tempfile (preserving the extension so
    `parse_config`'s extension-dispatched parser picks the right path) and
    handed to `run_audit`. We don't delete the temp file here — the audit
    might still be reading it; the OS reaps it eventually and `bench_cache/`
    is gitignored.
    """
    suffix = Path(file.filename or "").suffix or ".bin"
    fd, tmp_path = tempfile.mkstemp(prefix="goblin_upload_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())
    except Exception:
        # If we couldn't even land the upload, surface that immediately.
        async def _err() -> AsyncIterator[dict]:
            yield {
                "data": SSEEvent(
                    type="error",
                    data={"message": "Failed to save uploaded file."},
                ).model_dump_json()
            }

        return EventSourceResponse(_err())

    return EventSourceResponse(_stream_audit(tmp_path))


# ---------------------------------------------------------------------------
# Auto-tune endpoint — lets a UI on a CPU-only host (e.g. an HF Space) drive
# scripts/auto_tune.py running on a remote MI300X server. The endpoint
# spawns the CLI, tails its --events NDJSON stream, and re-emits each line
# as an SSE message. Subprocess output is discarded; everything the UI
# needs is in the structured events.
# ---------------------------------------------------------------------------


class AutoTuneRequest(BaseModel):
    """JSON shape the /auto-tune endpoint accepts. Mirrors the auto_tune.py
    CLI surface so the UI just sends what the user picked in the form."""

    model: str | None = Field(
        default=None,
        description="HuggingFace model id (e.g. Qwen/Qwen2.5-7B-Instruct). "
        "Mutually exclusive with `workload`.",
    )
    workload: str | None = Field(
        default=None,
        description="Path to a workload script ON THE SERVER's filesystem. "
        "Mutually exclusive with `model`.",
    )
    mode: str = Field(default="hardcoded", pattern="^(hardcoded|llm|llm-explore)$")
    candidates_per_iteration: int = Field(default=3, ge=2, le=10)
    steps: int = Field(default=20, ge=1, le=500)
    max_iterations: int = Field(default=10, ge=1, le=50)
    early_stop_after: int = Field(default=3, ge=1, le=20)
    max_crashes: int = Field(default=4, ge=1, le=20)
    improvement_threshold: float = Field(default=0.0, ge=0.0, le=20.0)


def _build_auto_tune_cmd(req: AutoTuneRequest, events_file: Path) -> list[str]:
    cmd: list[str] = [sys.executable, "-u", str(_AUTO_TUNE_SCRIPT)]
    if req.model:
        cmd.extend(["--model", req.model])
    elif req.workload:
        cmd.append(req.workload)
    cmd.extend([
        "--mode", req.mode,
        "--steps", str(req.steps),
        "--max-iterations", str(req.max_iterations),
        "--early-stop-after", str(req.early_stop_after),
        "--max-crashes", str(req.max_crashes),
        "--improvement-threshold", str(req.improvement_threshold),
        "--events", str(events_file),
    ])
    if req.mode == "llm-explore":
        cmd.extend(["--candidates-per-iteration", str(req.candidates_per_iteration)])
    return cmd


async def _stream_auto_tune(req: AutoTuneRequest) -> AsyncIterator[dict]:
    """Spawn auto_tune.py and forward its NDJSON --events stream as SSE.

    Each event is forwarded verbatim — the UI gets the same structured
    payload it would see when running auto_tune.py locally. We discard
    the subprocess's stdout/stderr; any errors are surfaced via the
    `summary` event's absence at process exit.
    """
    events_file = Path(tempfile.mktemp(prefix="auto_tune_events_", suffix=".ndjson"))
    events_file.write_text("")

    cmd = _build_auto_tune_cmd(req, events_file)

    # Validate at least one of model/workload was provided. (Pydantic
    # can't express "exactly one of A or B" cleanly, so we check here.)
    if not req.model and not req.workload:
        yield {"data": json.dumps({
            "type": "error",
            "message": "Pass either `model` or `workload`, not neither."
        })}
        return
    if req.model and req.workload:
        yield {"data": json.dumps({
            "type": "error",
            "message": "Pass either `model` or `workload`, not both."
        })}
        return

    proc = subprocess.Popen(
        cmd,
        cwd=str(_REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ},
    )

    seen_bytes = 0
    try:
        while True:
            # Poll the events file for new lines
            try:
                with events_file.open("r") as f:
                    f.seek(seen_bytes)
                    chunk = f.read()
                    new_seen = f.tell()
            except OSError:
                chunk = ""
                new_seen = seen_bytes

            if chunk:
                # Drop a trailing partial line — re-read it next tick once
                # the writer has flushed the rest.
                lines = chunk.splitlines(keepends=True)
                if lines and not lines[-1].endswith("\n"):
                    partial = lines.pop()
                    new_seen -= len(partial.encode("utf-8"))
                for line in lines:
                    line = line.strip()
                    if line:
                        yield {"data": line}
            seen_bytes = new_seen

            if proc.poll() is not None:
                # Subprocess exited. Drain whatever's left on disk.
                try:
                    with events_file.open("r") as f:
                        f.seek(seen_bytes)
                        tail = f.read()
                except OSError:
                    tail = ""
                for line in tail.splitlines():
                    line = line.strip()
                    if line:
                        yield {"data": line}
                if proc.returncode != 0:
                    yield {"data": json.dumps({
                        "type": "process_exit",
                        "returncode": proc.returncode,
                        "message": (
                            f"auto_tune.py exited with code {proc.returncode}. "
                            "Check the server's stderr or check `last_runner_failure_*` "
                            "in `bench_cache/` for goblin_runner.sh failure logs."
                        ),
                    })}
                break

            await asyncio.sleep(0.5)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            events_file.unlink()
        except OSError:
            pass


@app.post("/auto-tune")
async def auto_tune_endpoint(req: AutoTuneRequest) -> EventSourceResponse:
    """Stream auto_tune.py events back to the caller as SSE.

    Run a UI on any host (HF Spaces, local laptop), point it at this
    endpoint, and the actual GPU work happens on the server hosting the
    FastAPI app. Subprocess output is discarded — only the --events
    NDJSON stream crosses the wire, one structured event per SSE message.
    """
    if not _AUTO_TUNE_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"auto_tune.py not found at {_AUTO_TUNE_SCRIPT}",
        )
    return EventSourceResponse(_stream_auto_tune(req))


# Convenience: support `python -m uvicorn agent.server:app --reload`.
__all__ = ["app"]


def _decode_event(raw: str) -> dict:
    """Helper for the CLI driver — parse a serialized SSEEvent JSON payload.

    Lives here so __main__.py and tests can share one parser.
    """
    return json.loads(raw)
