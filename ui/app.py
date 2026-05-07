"""GPU Goblin — Streamlit UI.

Single-page demo app. Lets a judge:
  1. Pick a demo lane (offline-replay or live MI300X).
  2. Drop in a fine-tuning script (or use the canonical pre-shipped sample).
  3. Watch the agent stream tool calls live.
  4. Read the side-by-side audit report with a waste-budget chart.

The backend is `POST /audit` on a FastAPI server (default
``http://localhost:8000/audit``, override with ``GOBLIN_BACKEND_URL``). It
streams Server-Sent Events shaped like ``agent.schemas.SSEEvent`` using the
``data: <json>\\n\\n`` framing that ``sse-starlette`` emits.

If the backend is unreachable (or the developer is running the UI solo
during the build) we fall back to replaying ``tests/fixtures/cached_audit.json``
with small ``time.sleep`` pauses so the live-cards demo still works end to
end. This is what makes the UI demoable as a SOLO process.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

# `streamlit run ui/app.py` only adds `ui/` to sys.path, not the repo root,
# so `from agent.schemas import ...` would fail without this bootstrap.
# Same problem when the app is deployed as a Hugging Face Space — HF runs
# `streamlit run ui/app.py` from the repo root, but the script's parent dir
# is what lands on sys.path. Fix it once, here.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import altair as alt
import pandas as pd
import requests
import streamlit as st

from agent.schemas import Report, SSEEvent

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_WORKLOAD = REPO_ROOT / "workloads" / "train_qwen_lora.py"
CACHED_AUDIT = REPO_ROOT / "tests" / "fixtures" / "cached_audit.json"

DEFAULT_BACKEND = "http://localhost:8000/audit"
BACKEND_URL = os.environ.get("GOBLIN_BACKEND_URL", DEFAULT_BACKEND)

REPLAY_DELAY_S = 0.4

WASTE_BUCKETS: list[str] = [
    "useful_gpu",
    "data_wait",
    "host_gap",
    "comm_excess",
    "memory_headroom",
    "precision_path",
    "kernel_shape",
]

# Green for the only "good" bucket; warm shades for the lossy ones.
WASTE_COLORS: dict[str, str] = {
    "useful_gpu": "#2EA043",       # green
    "data_wait": "#F0A36E",        # warm orange
    "host_gap": "#E07A5F",         # terracotta
    "comm_excess": "#D9534F",      # red
    "memory_headroom": "#F4C25C",  # amber
    "precision_path": "#C7522A",   # rust
    "kernel_shape": "#B85450",     # brick
}

TOOL_DISPLAY: dict[str, str] = {
    "parse_config": "parse_config",
    "profile_run": "profile_run",
    "query_rocm_kb": "query_rocm_kb",
    "propose_patch": "propose_patch",
    "benchmark": "benchmark",
    "compare_runs": "compare_runs",
}


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _parse_sse_lines(lines: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Decode SSE ``data: <json>\\n\\n`` frames into dicts.

    sse-starlette emits one ``data:`` line per event and an empty line
    terminator. ``requests``'s ``iter_lines`` already strips the trailing
    ``\\n`` so we just look for the ``data:`` prefix.
    """
    for raw in lines:
        if raw is None:
            continue
        line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload:
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            # Malformed event — skip rather than crash the whole stream.
            continue


def _stream_from_backend(uploaded_path: Path, lane: str) -> Iterator[dict[str, Any]]:
    """Open a streaming POST to the agent backend and yield decoded SSE events."""
    files = {"file": (uploaded_path.name, uploaded_path.open("rb"), "text/plain")}
    data = {"lane": lane}
    try:
        with requests.post(
            BACKEND_URL, files=files, data=data, stream=True, timeout=(5, 600)
        ) as resp:
            resp.raise_for_status()
            yield from _parse_sse_lines(resp.iter_lines())
    finally:
        files["file"][1].close()


def _stream_from_cache() -> Iterator[dict[str, Any]]:
    """Replay tests/fixtures/cached_audit.json with small inter-event pauses."""
    events = json.loads(CACHED_AUDIT.read_text())
    for ev in events:
        time.sleep(REPLAY_DELAY_S)
        yield ev


def _has_hf_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))


def _stream_from_inproc(uploaded_path: Path) -> Iterator[dict[str, Any]]:
    """Drive `agent.loop.run_audit` in a worker thread; yield events synchronously.

    Bridges Streamlit's sync world to the loop's async generator. The agent
    runs in a background thread with its own asyncio event loop; events
    arrive on a bounded queue that this generator drains until a sentinel.

    All heavy imports happen inside this function so that `ui.app` itself
    stays light when the in-process lane isn't engaged.
    """
    import asyncio
    import queue
    import threading

    from agent.loop import run_audit

    SENTINEL: object = object()
    q: "queue.Queue[Any]" = queue.Queue(maxsize=128)

    async def _producer() -> None:
        try:
            async for event in run_audit(str(uploaded_path)):
                q.put({"type": event.type, "data": event.data})
        except Exception as exc:  # pragma: no cover — defence-in-depth
            q.put(
                {
                    "type": "error",
                    "data": {"message": f"in-proc agent: {type(exc).__name__}: {exc}"},
                }
            )
        finally:
            q.put(SENTINEL)

    def _runner() -> None:
        # Each thread needs its own event loop; asyncio.run handles setup +
        # teardown for us.
        asyncio.run(_producer())

    thread = threading.Thread(target=_runner, daemon=True, name="goblin-agent-loop")
    thread.start()

    try:
        while True:
            item = q.get()
            if item is SENTINEL:
                break
            yield item
    finally:
        thread.join(timeout=5)


def _events_for(uploaded_path: Path | None, lane: str) -> Iterator[dict[str, Any]]:
    """Return an event iterator. Three live paths in priority order:
    (1) in-process Qwen via HF Inference Providers when HF_TOKEN is set;
    (2) external FastAPI backend via GOBLIN_BACKEND_URL;
    (3) cached replay (always works).

    The two upstream paths each get one chance with a clear st.warning on
    failure. Offline lane skips straight to cache.
    """
    if uploaded_path is None or lane != "live":
        yield from _stream_from_cache()
        return

    if _has_hf_token():
        try:
            yield from _stream_from_inproc(uploaded_path)
            return
        except Exception as exc:
            st.warning(
                f"In-process agent failed ({type(exc).__name__}: {exc}) — "
                "trying external backend, then cached replay."
            )

    try:
        yield from _stream_from_backend(uploaded_path, lane)
        return
    except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
        st.warning(
            f"Backend unreachable ({type(exc).__name__}) — running offline-replay "
            "demo from cached audit."
        )

    yield from _stream_from_cache()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _status_badge(status: str) -> str:
    """Render a colored pill for a tool status."""
    palette = {
        "pending": ("#888", "pending"),
        "running": ("#3B82F6", "running…"),
        "done": ("#2EA043", "done"),
        "failed": ("#D9534F", "failed"),
    }
    color, label = palette.get(status, ("#888", status))
    return (
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:10px;font-size:12px;'>{label}</span>"
    )


def _render_tool_cards(container, cards: list[dict[str, Any]]) -> None:
    """(Re)render the live tool-call card stack."""
    container.empty()
    with container.container():
        if not cards:
            st.caption("Tool calls will stream in here as the agent runs.")
            return
        for card in cards:
            badge = _status_badge(card["status"])
            display = TOOL_DISPLAY.get(card["name"], card["name"])
            st.markdown(
                f"**`{display}`** &nbsp; {badge}",
                unsafe_allow_html=True,
            )
            if card.get("input_summary"):
                st.caption(card["input_summary"])
            if card["status"] == "done" and card.get("result_summary"):
                with st.expander("result", expanded=False):
                    st.code(card["result_summary"], language="json")
            if card["status"] == "failed" and card.get("error"):
                st.error(card["error"])
            st.divider()


def _summarize_input(name: str, payload: dict[str, Any]) -> str:
    """One-line summary of a tool input for the card."""
    if name == "parse_config":
        return f"file_path = {payload.get('file_path')!r}"
    if name == "profile_run":
        return f"steps = {payload.get('steps', '?')}, model = {(payload.get('config') or {}).get('model_name', '?')}"
    if name == "query_rocm_kb":
        return f"symptom = {payload.get('symptom', '')!r}, top_k = {payload.get('top_k', '?')}"
    if name == "propose_patch":
        rules = payload.get("rules") or []
        return f"rules = {len(rules)} candidates"
    if name == "benchmark":
        steps = payload.get("steps", "?")
        cfg = payload.get("config") or {}
        return f"steps = {steps}, precision = {cfg.get('precision', '?')}, batch = {cfg.get('batch_size', '?')}"
    if name == "compare_runs":
        return f"workload = {payload.get('workload_name', '?')!r}"
    return ""


def _summarize_result(name: str, result: Any) -> str:
    """Truncated, human-readable JSON snippet of a tool result."""
    if not isinstance(result, dict):
        return json.dumps(result)[:600]
    if name == "parse_config":
        keys = ("model_name", "batch_size", "precision", "attention_impl",
                "dataloader_workers", "redactions")
        slim = {k: result.get(k) for k in keys if k in result}
        return json.dumps(slim, indent=2)
    if name in ("profile_run", "benchmark"):
        keys = ("steps", "tokens_per_sec", "mfu_pct", "hbm_peak_gb",
                "gpu_util_pct", "attention_kernel_loaded")
        slim = {k: result.get(k) for k in keys if k in result}
        return json.dumps(slim, indent=2)
    if name == "query_rocm_kb":
        rules = result.get("rules") or []
        slim = [{"id": r.get("id"), "targets_bucket": r.get("targets_bucket")} for r in rules]
        return json.dumps(slim, indent=2)
    if name == "propose_patch":
        slim = {
            "rule_count": len(result.get("rationale") or []),
            "expected_speedup_low": result.get("expected_speedup_low"),
            "expected_speedup_high": result.get("expected_speedup_high"),
            "confidence": result.get("confidence"),
        }
        return json.dumps(slim, indent=2)
    if name == "compare_runs":
        slim = {
            "summary_line": result.get("summary_line"),
            "speedup_actual": result.get("speedup_actual"),
            "confidence": result.get("confidence"),
        }
        return json.dumps(slim, indent=2)
    blob = json.dumps(result)
    return blob if len(blob) <= 600 else blob[:600] + "…"


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _render_waste_chart(report: Report) -> None:
    """Stacked horizontal bars: 'before' / 'after' rows, segments per bucket."""
    rows: list[dict[str, Any]] = []
    for label, budget in (("before", report.waste_budget_before),
                          ("after", report.waste_budget_after)):
        for bucket in WASTE_BUCKETS:
            rows.append(
                {
                    "run": label,
                    "bucket": bucket,
                    "seconds": float(getattr(budget, bucket)),
                }
            )

    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("run:N", title="", sort=["before", "after"]),
            x=alt.X(
                "seconds:Q",
                title="Time per step (s)",
                stack="zero",
            ),
            color=alt.Color(
                "bucket:N",
                title="Waste bucket",
                scale=alt.Scale(
                    domain=WASTE_BUCKETS,
                    range=[WASTE_COLORS[b] for b in WASTE_BUCKETS],
                ),
                sort=WASTE_BUCKETS,
            ),
            order=alt.Order("bucket_order:Q"),
            tooltip=["run", "bucket", alt.Tooltip("seconds:Q", format=".3f")],
        )
        .transform_calculate(
            bucket_order=(
                "indexof("
                + json.dumps(WASTE_BUCKETS)
                + ", datum.bucket)"
            )
        )
        .properties(height=140)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_metric_table(report: Report) -> None:
    rows = []
    for d in report.metric_deltas:
        rows.append(
            {
                "metric": d.name,
                "before": f"{d.before:.2f} {d.unit}".strip(),
                "after": f"{d.after:.2f} {d.unit}".strip(),
                "delta_%": f"{d.delta_pct:+.1f}%",
            }
        )
    st.table(pd.DataFrame(rows))


def _render_rationale(report: Report) -> None:
    if not report.patch.rationale:
        st.info("No rules applied — your config already looks tuned.")
        return
    for ra in report.patch.rationale:
        with st.container(border=True):
            st.markdown(f"**`{ra.rule_id}`** &nbsp; targets `{ra.targets_bucket}`")
            st.write(ra.rationale)
            st.caption(
                f"Citation: {ra.citation}  ·  "
                f"est. recovery: {ra.estimated_recovery_seconds:.3f} s/step"
            )


def _render_final_report(report_dict: dict[str, Any]) -> None:
    report = Report.model_validate(report_dict)

    st.success(report.summary_line)

    # Headline metrics row
    tps_delta = next(
        (d for d in report.metric_deltas if d.name == "tokens_per_sec"), None
    )
    mfu_delta = next((d for d in report.metric_deltas if d.name == "mfu_pct"), None)
    hbm_delta = next(
        (d for d in report.metric_deltas if d.name == "hbm_peak_gb"), None
    )
    cols = st.columns(3)
    if tps_delta is not None:
        cols[0].metric(
            "Tokens/sec",
            f"{tps_delta.after:.0f}",
            f"{tps_delta.delta_pct:+.1f}% vs {tps_delta.before:.0f}",
        )
    cols[0].markdown(
        f"**{report.speedup_actual:.2f}× speedup** "
        f"(predicted {report.speedup_predicted_low:.2f}–"
        f"{report.speedup_predicted_high:.2f}×, conf {report.confidence:.2f})"
    )
    if mfu_delta is not None:
        cols[1].metric(
            "MFU",
            f"{mfu_delta.after:.0f}%",
            f"{mfu_delta.delta_pct:+.1f}%",
        )
    if hbm_delta is not None:
        cols[2].metric(
            "HBM peak",
            f"{hbm_delta.after:.0f} GB",
            f"{hbm_delta.delta_pct:+.1f}%",
        )

    st.subheader("Side-by-side metrics")
    _render_metric_table(report)

    st.subheader("Where time was lost")
    _render_waste_chart(report)

    st.subheader("Patch")
    with st.expander("Unified diff", expanded=False):
        st.code(report.patch.diff or "(empty diff)", language="diff")

    st.subheader("Rationale")
    _render_rationale(report)

    st.caption(report.validity_footer)

    st.download_button(
        label="Download Report (JSON)",
        data=json.dumps(report.model_dump(), indent=2),
        file_name=f"goblin_report_{report.workload_name.replace(' ', '_')}.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Audit runner — drives the live cards
# ---------------------------------------------------------------------------


def _run_audit(uploaded_path: Path | None, lane: str) -> None:
    """Stream events into the live panel and stash the final report in session state."""
    st.session_state["thoughts"] = []
    st.session_state["cards"] = []
    st.session_state["final_report"] = None
    st.session_state["error"] = None

    cards: list[dict[str, Any]] = st.session_state["cards"]
    cards_panel = st.session_state["cards_panel"]
    thoughts_panel = st.session_state["thoughts_panel"]

    def _refresh_thoughts() -> None:
        thoughts_panel.empty()
        with thoughts_panel.container():
            if not st.session_state["thoughts"]:
                st.caption("Agent reasoning will appear here.")
            for txt in st.session_state["thoughts"]:
                st.markdown(f"> {txt}")

    _refresh_thoughts()
    _render_tool_cards(cards_panel, cards)

    for ev_dict in _events_for(uploaded_path, lane):
        try:
            ev = SSEEvent.model_validate(ev_dict)
        except Exception as exc:
            st.session_state["error"] = f"Malformed SSE event: {exc}"
            break

        if ev.type == "thought":
            text = ev.data.get("text") or ev.data.get("content") or ""
            if isinstance(text, list):
                # Anthropic content blocks — stringify
                text = "\n".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in text
                )
            if text:
                st.session_state["thoughts"].append(text)
                _refresh_thoughts()

        elif ev.type == "tool_call":
            name = ev.data.get("name", "tool")
            cards.append(
                {
                    "id": ev.data.get("id") or f"{name}-{len(cards)}",
                    "name": name,
                    "status": "running",
                    "input_summary": _summarize_input(name, ev.data.get("input") or {}),
                }
            )
            _render_tool_cards(cards_panel, cards)

        elif ev.type == "tool_result":
            target_id = ev.data.get("id")
            target_name = ev.data.get("name")
            target = None
            for card in reversed(cards):
                if target_id and card["id"] == target_id:
                    target = card
                    break
                if target_name and card["name"] == target_name and card["status"] == "running":
                    target = card
                    break
            if target is None:
                target = {
                    "id": target_id or f"{target_name}-{len(cards)}",
                    "name": target_name or "tool",
                    "status": "running",
                    "input_summary": "",
                }
                cards.append(target)

            ok = ev.data.get("ok", True)
            if ok:
                target["status"] = "done"
                target["result_summary"] = _summarize_result(
                    target["name"], ev.data.get("result")
                )
            else:
                target["status"] = "failed"
                target["error"] = ev.data.get("error") or "Tool reported ok=false"
            _render_tool_cards(cards_panel, cards)

        elif ev.type == "final_report":
            st.session_state["final_report"] = ev.data.get("report")

        elif ev.type == "error":
            st.session_state["error"] = ev.data.get("message") or json.dumps(ev.data)
            break


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="GPU Goblin",
        page_icon="🧌",
        layout="wide",
    )

    st.title("🧌 GPU Goblin")
    st.caption("An AI agent that hunts wasted compute on AMD MI300X")

    # ------- Lane selector -------
    lane = st.radio(
        "Demo lane",
        options=["Offline replay (synthetic corpus)", "Live agent"],
        horizontal=True,
        index=0,
    )
    lane_token = "offline" if lane.startswith("Offline") else "live"

    if lane_token == "live":
        if _has_hf_token():
            qwen_model = os.environ.get("GOBLIN_QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
            st.caption(
                f"🟢 Live mode: agent runs **{qwen_model}** in-process via Hugging "
                "Face Inference Providers. GPU-touching tools (profile_run, "
                "benchmark) use the FakeRunner with cached MI300X metrics — "
                "this is the demo lane for the Hugging Face Space."
            )
        elif BACKEND_URL != DEFAULT_BACKEND or BACKEND_URL == DEFAULT_BACKEND:
            st.caption(
                f"🔵 Live mode: streaming SSE from `{BACKEND_URL}`. Falls back "
                "to cached replay if unreachable. Set `HF_TOKEN` to instead "
                "drive Qwen in-process without a backend."
            )

    # ------- File picker -------
    st.subheader("1. Pick a workload")
    upload_col, sample_col = st.columns([3, 1])
    with upload_col:
        uploaded = st.file_uploader(
            "Upload your training script or HF TrainingArguments",
            type=["py", "json", "yaml", "yml"],
            label_visibility="visible",
        )
    with sample_col:
        st.write("")
        st.write("")
        use_sample = st.button(
            "Use sample workload",
            help=f"Audit {SAMPLE_WORKLOAD.name} (Qwen2.5-7B-Instruct + LoRA, deliberately mis-tuned).",
        )

    if "uploaded_path" not in st.session_state:
        st.session_state["uploaded_path"] = None
        st.session_state["uploaded_label"] = None

    if uploaded is not None:
        # Persist the upload to a temp file so the backend can re-open it.
        tmp = Path(tempfile.gettempdir()) / f"goblin_{uploaded.name}"
        tmp.write_bytes(uploaded.getvalue())
        st.session_state["uploaded_path"] = tmp
        st.session_state["uploaded_label"] = uploaded.name
    elif use_sample:
        if SAMPLE_WORKLOAD.exists():
            st.session_state["uploaded_path"] = SAMPLE_WORKLOAD
            st.session_state["uploaded_label"] = SAMPLE_WORKLOAD.name
        else:
            st.error(f"Sample workload not found at {SAMPLE_WORKLOAD}")

    if st.session_state.get("uploaded_label"):
        st.success(f"Selected: `{st.session_state['uploaded_label']}`")

    # ------- Audit button -------
    st.subheader("2. Run the audit")
    audit_disabled = st.session_state.get("uploaded_path") is None
    audit_clicked = st.button(
        "Audit",
        type="primary",
        disabled=audit_disabled,
        help="Streams the agent's tool calls live, then shows the side-by-side report.",
    )

    # ------- Live panels (always present so we can stream into them) -------
    st.subheader("3. Agent activity")
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Agent reasoning**")
        st.session_state["thoughts_panel"] = st.empty()
        if not st.session_state.get("thoughts"):
            st.session_state["thoughts_panel"].caption(
                "Agent reasoning will appear here."
            )
    with right:
        st.markdown("**Tool calls**")
        st.session_state["cards_panel"] = st.empty()
        if not st.session_state.get("cards"):
            st.session_state["cards_panel"].caption(
                "Tool calls will stream in here as the agent runs."
            )

    if audit_clicked and st.session_state.get("uploaded_path") is not None:
        with st.spinner("Goblin is hunting…"):
            _run_audit(Path(st.session_state["uploaded_path"]), lane_token)

    if st.session_state.get("error"):
        st.error(st.session_state["error"])

    # ------- Final report -------
    if st.session_state.get("final_report"):
        st.subheader("4. Audit report")
        _render_final_report(st.session_state["final_report"])


main()
