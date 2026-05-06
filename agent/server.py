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

import json
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from agent.backends import active_backend_name
from agent.loop import run_audit
from agent.schemas import SSEEvent
from agent.tools import ALL_TOOLS

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
    return {
        "ok": True,
        "tools": [t.name for t in ALL_TOOLS],
        "backend": active_backend_name(),
        "model": os.environ.get("GOBLIN_QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "provider": os.environ.get("GOBLIN_QWEN_PROVIDER", "auto"),
        "has_api_key": _has_hf_token(),
    }


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


# Convenience: support `python -m uvicorn agent.server:app --reload`.
__all__ = ["app"]


def _decode_event(raw: str) -> dict:
    """Helper for the CLI driver — parse a serialized SSEEvent JSON payload.

    Lives here so __main__.py and tests can share one parser.
    """
    return json.loads(raw)
