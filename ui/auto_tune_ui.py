"""GPU Goblin auto-tune UI — Streamlit frontend for scripts/auto_tune.py.

Run:
    streamlit run ui/auto_tune_ui.py

The UI:
  1. Form: pick model OR workload path, mode, steps, etc.
  2. Run: launches scripts/auto_tune.py as a subprocess with --events FILE
  3. Live progress: tails the events file, renders iteration cards as they
     arrive, updates the best-tokens/sec metric on every accepted change
  4. Final report: improvement vs baseline, accepted vs rejected
     experiments, waste-budget reduction chart, and a copyable diff.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Streamlit only puts ui/ on sys.path, so add the repo root for any helper
# imports that might land in shared modules later.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import altair as alt
import pandas as pd
import requests
import streamlit as st

REPO_ROOT = _REPO_ROOT
AUTO_TUNE_SCRIPT = REPO_ROOT / "scripts" / "auto_tune.py"

WASTE_BUCKETS = [
    "useful_gpu",
    "data_wait",
    "host_gap",
    "comm_excess",
    "memory_headroom",
    "precision_path",
    "kernel_shape",
]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GPU Goblin — Auto-Tune",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 GPU Goblin — Auto-Tune")
st.caption(
    "Iteratively find the fastest configuration for a fine-tuning workload "
    "on AMD MI300X. Pick a model, hit Run, watch tokens/sec climb."
)


# ---------------------------------------------------------------------------
# Form: inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Run configuration")

    # ---- Backend mode (Local subprocess vs remote GPU server) ----
    default_backend = os.environ.get("GOBLIN_AUTO_TUNE_URL", "")
    backend_mode = st.radio(
        "Backend",
        options=("Local subprocess", "Remote GPU server"),
        index=1 if default_backend else 0,
        help=(
            "Local: launches scripts/auto_tune.py here (this host needs an MI300X). "
            "Remote: POSTs to /auto-tune on a FastAPI server you've stood up on "
            "the GPU host. Use Remote when running the UI on HF Spaces."
        ),
    )
    backend_url = ""
    if backend_mode == "Remote GPU server":
        backend_url = st.text_input(
            "Backend URL",
            value=default_backend or "http://localhost:8000",
            help="Base URL of the FastAPI server (no trailing slash). "
            "/auto-tune is appended automatically.",
        )

    st.divider()
    workload_source = st.radio(
        "Workload source",
        options=("Model id", "Custom workload script"),
        index=0,
        help=(
            "Model id: auto-generates a baseline workload from the demo "
            "template (Qwen-style LoRA fine-tune). Custom: point at any "
            "Python training script."
        ),
    )

    model_id = ""
    workload_path = ""
    if workload_source == "Model id":
        model_id = st.text_input(
            "Model id",
            value="Qwen/Qwen2.5-7B-Instruct",
            help=(
                "Any HuggingFace causal-LM model id. For gated models "
                "(Llama, etc.) ensure HF_TOKEN is set in the environment."
            ),
        )
    else:
        workload_path = st.text_input(
            "Path to workload script (relative to repo root)",
            value="workloads/train_qwen_lora.py",
        )

    st.divider()
    st.subheader("Tuning strategy")

    mode = st.selectbox(
        "Mode",
        options=("hardcoded", "llm", "llm-explore"),
        index=0,
        help=(
            "hardcoded: priority-ordered playbook (no API key). "
            "llm: LLM picks one experiment per iteration (greedy). "
            "llm-explore: LLM proposes K candidates per iteration; best wins."
        ),
    )
    candidates_per_iteration = 3
    if mode == "llm-explore":
        candidates_per_iteration = st.slider(
            "Candidates per iteration (K)",
            min_value=2,
            max_value=6,
            value=3,
            help="Each iteration runs K benchmarks. Higher = broader search, more GPU time.",
        )

    steps = st.slider(
        "Steps per benchmark",
        min_value=10,
        max_value=100,
        value=20,
        step=5,
        help="More steps = lower variance but longer benchmarks.",
    )

    max_iterations = st.slider(
        "Max iterations",
        min_value=1,
        max_value=20,
        value=10,
        help="Cap on tuning iterations. Default 10 for llm modes, 5 for llm-explore.",
    )

    early_stop_after = st.slider(
        "Early stop after N non-improvements",
        min_value=1,
        max_value=10,
        value=3,
    )

    max_crashes = st.slider(
        "Max total crashes",
        min_value=1,
        max_value=10,
        value=4,
    )

    improvement_threshold = st.number_input(
        "Improvement threshold (%)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help="Min % gain to accept. 0.0 = any positive delta wins.",
    )

    st.divider()
    run_pressed = st.button("🚀 Run auto-tune", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _format_tps(v: float) -> str:
    return f"{v:,.0f}" if v else "—"


def _format_pct(v: float | None, suffix: str = "%") -> str:
    if v is None:
        return "—"
    return f"{v:+.2f}{suffix}"


def _waste_chart(baseline: dict, current: dict | None) -> alt.Chart:
    """Two-row stacked bar: baseline vs current waste_budget breakdown."""
    rows = []
    sources = [("baseline", baseline)]
    if current is not None and current is not baseline:
        sources.append(("current best", current))
    for label, m in sources:
        wb = (m or {}).get("waste_budget") or {}
        for bucket in WASTE_BUCKETS:
            v = float(wb.get(bucket, 0.0))
            if v <= 0:
                continue
            rows.append({"run": label, "bucket": bucket, "seconds": v})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("seconds:Q", title="seconds / step"),
            y=alt.Y("run:N", title=None, sort=["baseline", "current best"]),
            color=alt.Color(
                "bucket:N",
                scale=alt.Scale(scheme="tableau10"),
                sort=WASTE_BUCKETS,
            ),
            order=alt.Order("bucket:N", sort="ascending"),
            tooltip=["run", "bucket", alt.Tooltip("seconds:Q", format=".3f")],
        )
        .properties(height=120)
    )


def _render_metrics_row(baseline: dict | None, current: dict | None) -> None:
    """Top-of-page metrics: tokens/sec, mfu_pct, hbm, with deltas vs baseline."""
    cols = st.columns(4)
    if baseline is None:
        for c, label in zip(cols, ["tokens/sec", "mfu_pct", "hbm_peak (GB)", "iterations"]):
            c.metric(label, "—")
        return

    cur = current if current is not None else baseline
    base_tps = float(baseline.get("tokens_per_sec") or 0)
    cur_tps = float(cur.get("tokens_per_sec") or 0)
    base_mfu = float(baseline.get("mfu_pct") or 0)
    cur_mfu = float(cur.get("mfu_pct") or 0)
    base_hbm = float(baseline.get("hbm_peak_gb") or 0)
    cur_hbm = float(cur.get("hbm_peak_gb") or 0)

    cols[0].metric(
        "tokens/sec",
        _format_tps(cur_tps),
        delta=f"{cur_tps - base_tps:+,.0f}" if base_tps else None,
    )
    cols[1].metric(
        "MFU %",
        f"{cur_mfu:.2f}",
        delta=f"{cur_mfu - base_mfu:+.2f} pts" if base_mfu else None,
    )
    cols[2].metric(
        "HBM peak (GB)",
        f"{cur_hbm:.1f}",
        delta=f"{cur_hbm - base_hbm:+.1f}" if base_hbm else None,
    )


def _render_candidate_card(event: dict) -> None:
    """One candidate's outcome inside an iteration."""
    name = event.get("name", "?")
    rationale = event.get("rationale", "")
    outcome = event.get("outcome", "?")
    metrics = event.get("metrics") or {}
    delta = event.get("delta_vs_best")
    reason = event.get("reason", "")

    icon = {
        "evaluated": "📊",
        "skipped": "⏭️",
        "rejected": "❌",
        "crashed": "💥",
    }.get(outcome, "•")

    title = f"{icon}  **{name}**"
    if outcome == "evaluated" and delta is not None:
        title += f"  —  Δ {delta:+.2f}%"
    elif outcome != "evaluated":
        title += f"  —  *{outcome}*"

    with st.container(border=True):
        st.markdown(title)
        if rationale:
            st.caption(rationale)
        if metrics:
            sub = st.columns(4)
            sub[0].metric("tokens/sec", f"{metrics.get('tokens_per_sec', 0):,.0f}")
            sub[1].metric("MFU %", f"{metrics.get('mfu_pct', 0):.2f}")
            sub[2].metric("HBM peak GB", f"{metrics.get('hbm_peak_gb', 0):.1f}")
            sub[3].metric("Δ vs best", f"{delta:+.2f}%" if delta is not None else "—")
            wb = metrics.get("waste_budget") or {}
            non_zero = {k: v for k, v in wb.items() if k != "useful_gpu" and v > 0}
            if non_zero:
                wb_text = ", ".join(
                    f"`{k}={v:.3f}`"
                    for k, v in sorted(non_zero.items(), key=lambda kv: kv[1], reverse=True)
                )
                st.caption(f"Waste: useful_gpu=`{wb.get('useful_gpu', 0):.3f}`, {wb_text}")
        if reason and outcome != "evaluated":
            st.caption(f"Reason: {reason}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _build_command(events_file: Path) -> list[str]:
    cmd: list[str] = [sys.executable, "-u", str(AUTO_TUNE_SCRIPT)]
    if workload_source == "Model id":
        if not model_id.strip():
            raise ValueError("Model id is required.")
        cmd.extend(["--model", model_id.strip()])
    else:
        wp = workload_path.strip()
        if not wp:
            raise ValueError("Workload path is required.")
        full = (REPO_ROOT / wp).resolve() if not Path(wp).is_absolute() else Path(wp)
        if not full.exists():
            raise ValueError(f"Workload not found: {full}")
        cmd.append(str(full))
    cmd.extend([
        "--mode", mode,
        "--steps", str(steps),
        "--max-iterations", str(max_iterations),
        "--early-stop-after", str(early_stop_after),
        "--max-crashes", str(max_crashes),
        "--improvement-threshold", str(improvement_threshold),
        "--events", str(events_file),
    ])
    if mode == "llm-explore":
        cmd.extend(["--candidates-per-iteration", str(candidates_per_iteration)])
    return cmd


def _build_request_body() -> dict:
    """Same config as _build_command, but as a JSON body for POST /auto-tune."""
    body: dict[str, Any] = {
        "mode": mode,
        "steps": steps,
        "max_iterations": max_iterations,
        "early_stop_after": early_stop_after,
        "max_crashes": max_crashes,
        "improvement_threshold": float(improvement_threshold),
    }
    if mode == "llm-explore":
        body["candidates_per_iteration"] = candidates_per_iteration
    if workload_source == "Model id":
        if not model_id.strip():
            raise ValueError("Model id is required.")
        body["model"] = model_id.strip()
    else:
        if not workload_path.strip():
            raise ValueError("Workload path is required.")
        body["workload"] = workload_path.strip()
    return body


def _read_events(path: Path, seen: int) -> tuple[list[dict], int]:
    """Read events from byte position `seen` onward; return (new_events, new_seen)."""
    if not path.exists():
        return [], seen
    try:
        with path.open("r") as f:
            f.seek(seen)
            chunk = f.read()
            new_seen = f.tell()
    except OSError:
        return [], seen
    if not chunk:
        return [], new_seen
    out: list[dict] = []
    # The last line may be partial if the writer is mid-flush; drop it
    # and back the pointer up so we re-read it next tick.
    pieces = chunk.splitlines(keepends=True)
    if pieces and not pieces[-1].endswith("\n"):
        partial = pieces.pop()
        new_seen -= len(partial.encode("utf-8"))
    for line in pieces:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out, new_seen


# ---------------------------------------------------------------------------
# Initial state — empty page
# ---------------------------------------------------------------------------

if not run_pressed:
    st.info(
        "👈 Configure inputs in the sidebar and press **Run auto-tune**. "
        "The script will profile your workload, try MI300X-specific tuning "
        "changes, and report what worked."
    )
    _render_metrics_row(None, None)
    st.subheader("How it works")
    st.markdown(
        """
        1. **Baseline benchmark** — runs your workload as-is (or as generated
           from `--model`) on the GPU and measures tokens/sec, MFU, HBM peak,
           and a per-bucket waste budget.
        2. **Iterative tuning** — applies one MI300X-specific change at a
           time (e.g. `bf16`, `batch_size=16`, `TORCH_BLAS_PREFER_HIPBLASLT=1`)
           and re-benchmarks. Keeps changes that beat the current best.
        3. **Live progress** — every accepted/rejected/crashed candidate
           shows up here as it happens.
        4. **Final report** — overall improvement, which experiments won,
           how much wastage was recovered, and a diff against the baseline
           workload.
        """
    )
    st.stop()


# ---------------------------------------------------------------------------
# Run path: launch subprocess + tail events
# ---------------------------------------------------------------------------

st.subheader("Live run")
header = st.empty()

# Validate inputs early so the spinner doesn't show before a clear error
try:
    if backend_mode == "Local subprocess":
        events_file = Path(tempfile.NamedTemporaryFile(
            prefix="auto_tune_events_", suffix=".ndjson", delete=False
        ).name)
        cmd = _build_command(events_file)
        with header.container():
            st.code(" ".join(cmd), language="bash")
            st.caption(f"Events stream: `{events_file}`")
    else:
        if not backend_url.strip():
            raise ValueError("Backend URL is required for Remote GPU server mode.")
        body = _build_request_body()
        events_file = None
        cmd = None
        with header.container():
            st.code(
                f"POST {backend_url.rstrip('/')}/auto-tune\n"
                + json.dumps(body, indent=2),
                language="bash",
            )
            st.caption(
                "Events stream: SSE from remote server. "
                "All GPU work happens on that host."
            )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

baseline_metrics: dict | None = None
best_metrics: dict | None = None  # most recent accepted iteration's metrics
final_summary: dict | None = None
iter_idx_to_container: dict[int, Any] = {}

metrics_row = st.empty()
with metrics_row.container():
    _render_metrics_row(None, None)

progress_bar = st.progress(0, text="Starting auto_tune…")

iters_section = st.container()
iters_section.subheader("Iterations")

stdout_buffer: list[str] = []
expected_iters = max(1, max_iterations)
return_code: int | None = None


def _handle_event(event: dict) -> None:
    """Render one event into the live UI. Mutates module-level state for
    baseline/best_metrics/final_summary so the summary block below has
    the data it needs after the run."""
    global baseline_metrics, best_metrics, final_summary  # noqa: PLW0603
    etype = event.get("type")
    if etype == "started":
        st.session_state["started_event"] = event
    elif etype == "baseline":
        baseline_metrics = event["metrics"]
        best_metrics = baseline_metrics
        with metrics_row.container():
            _render_metrics_row(baseline_metrics, best_metrics)
        with iters_section:
            with st.container(border=True):
                st.markdown("**Baseline**")
                sub = st.columns(4)
                sub[0].metric("tokens/sec", f"{baseline_metrics.get('tokens_per_sec', 0):,.0f}")
                sub[1].metric("MFU %", f"{baseline_metrics.get('mfu_pct', 0):.2f}")
                sub[2].metric("HBM peak GB", f"{baseline_metrics.get('hbm_peak_gb', 0):.1f}")
                sub[3].metric("GPU util %", f"{baseline_metrics.get('gpu_util_pct', 0):.1f}")
                wb = baseline_metrics.get("waste_budget") or {}
                non_zero = {k: v for k, v in wb.items() if k != "useful_gpu" and v > 0}
                if non_zero:
                    wb_str = ", ".join(
                        f"`{k}={v:.3f}`"
                        for k, v in sorted(non_zero.items(), key=lambda kv: kv[1], reverse=True)
                    )
                    st.caption(f"Recoverable waste: {wb_str}")
    elif etype == "iter_start":
        i = event["iteration"]
        with iters_section:
            container = st.container(border=True)
            iter_idx_to_container[i] = container
            n_cand = len(event.get("candidates") or [])
            cand_summary = ", ".join(
                c.get("name", "?") for c in event.get("candidates") or []
            )
            container.markdown(
                f"**Iteration {i}**  ·  {n_cand} candidate{'s' if n_cand != 1 else ''}: {cand_summary}"
            )
        progress_bar.progress(
            min(0.99, (i - 1) / expected_iters),
            text=f"Iteration {i} of up to {expected_iters} — proposed: {cand_summary}",
        )
    elif etype == "candidate":
        i = event["iteration"]
        container = iter_idx_to_container.get(i)
        if container is not None:
            with container:
                _render_candidate_card(event)
        progress_bar.progress(
            min(0.99, (i - 1) / expected_iters + 0.05),
            text=f"Iter {i} · candidate {event.get('candidate_index')}/"
                 f"{event.get('n_candidates')}: {event.get('name')} "
                 f"({event.get('outcome')})",
        )
    elif etype == "merge_attempt":
        i = event["iteration"]
        container = iter_idx_to_container.get(i)
        if container is not None:
            with container:
                outcome = event.get("outcome", "?")
                names = ", ".join(event.get("candidate_names") or [])
                if outcome == "wins":
                    delta = event.get("delta_vs_best", 0)
                    container.success(
                        f"🔗 MERGE WINS — combined ({names}) hit Δ {delta:+.2f}% "
                        f"(beats individual best `{event.get('individual_best_name')}`)"
                    )
                elif outcome == "lost":
                    delta = event.get("delta_vs_best", 0)
                    container.info(
                        f"🔗 Merge tested ({names}) — Δ {delta:+.2f}%, didn't beat "
                        f"individual `{event.get('individual_best_name')}`"
                    )
                elif outcome == "crashed":
                    container.warning(f"💥 Merge crashed: {names}")
                else:
                    container.info(f"🔗 Merge skipped: {event.get('reason', '?')}")
    elif etype == "iter_done":
        i = event["iteration"]
        container = iter_idx_to_container.get(i)
        outcome = event.get("outcome")
        if container is not None:
            with container:
                if outcome == "accepted":
                    container.success(
                        f"✅ ACCEPTED — `{event.get('winner_name')}` "
                        f"(Δ {event.get('winner_delta', 0):+.2f}%)"
                    )
                else:
                    container.warning(
                        f"⏭️ ALL REJECTED — best was `{event.get('winner_name')}` "
                        f"(Δ {event.get('winner_delta', 0):+.2f}%, below threshold)"
                    )
        if outcome == "accepted" and event.get("best_metrics"):
            best_metrics = event["best_metrics"]
            with metrics_row.container():
                _render_metrics_row(baseline_metrics, best_metrics)
        progress_bar.progress(
            min(0.99, i / expected_iters),
            text=f"Iter {i} done — best so far: {event.get('best_tps', 0):,.0f} tok/s",
        )
    elif etype == "summary":
        final_summary = event
        progress_bar.progress(1.0, text="Auto-tune complete")
    elif etype == "error":
        st.error(event.get("message", "unknown error"))
    elif etype == "process_exit":
        rc = event.get("returncode", "?")
        st.error(
            f"Backend subprocess exited (code {rc}): "
            + event.get("message", "")
        )


# ---- Source the events from either local subprocess or remote SSE ----
if backend_mode == "Local subprocess":
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(REPO_ROOT),
        env={**os.environ},
    )

    seen_bytes = 0
    try:
        while True:
            # Pull stdout incrementally so the user sees the raw script log too
            while True:
                try:
                    line = proc.stdout.readline() if proc.stdout else ""
                except Exception:
                    line = ""
                if not line:
                    break
                stdout_buffer.append(line.rstrip())

            new_events, seen_bytes = _read_events(events_file, seen_bytes)
            for event in new_events:
                _handle_event(event)

            if proc.poll() is not None and not new_events:
                new_events, seen_bytes = _read_events(events_file, seen_bytes)
                if not new_events:
                    break
            time.sleep(0.4)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
    return_code = proc.returncode
else:
    # Remote SSE: stream from the FastAPI server
    url = backend_url.rstrip("/") + "/auto-tune"
    try:
        response = requests.post(
            url,
            json=body,
            stream=True,
            timeout=(10, None),  # connect timeout 10s, read timeout indefinite
            headers={"Accept": "text/event-stream"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Failed to reach backend at {url}: {exc}")
        st.stop()

    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            stdout_buffer.append(raw_line)
            # SSE framing: lines starting with `data: <json>`
            if raw_line.startswith("data:"):
                payload = raw_line[len("data:"):].strip()
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                _handle_event(event)
                if event.get("type") == "summary":
                    # Final event; the server may still send a process_exit
                    # but we can stop blocking the UI.
                    pass
            elif raw_line.startswith("event:") or raw_line.startswith(":"):
                # SSE event-name line or comment — ignored
                continue
    except requests.RequestException as exc:
        st.error(f"Stream interrupted: {exc}")
    return_code = 0 if final_summary is not None else 1

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Summary")

if final_summary is None:
    st.error(
        f"Auto-tune subprocess exited with code {return_code} but emitted "
        "no summary event. Check the raw stdout below for details."
    )
else:
    base_tps = float(final_summary.get("baseline_tps") or 0)
    best_tps = float(final_summary.get("best_tps") or 0)
    improvement_pct = float(final_summary.get("improvement_pct") or 0)
    base = final_summary.get("baseline_metrics") or {}
    best = final_summary.get("best_metrics") or {}

    summary_cols = st.columns(4)
    summary_cols[0].metric(
        "Baseline tokens/sec",
        f"{base_tps:,.0f}",
    )
    summary_cols[1].metric(
        "Best tokens/sec",
        f"{best_tps:,.0f}",
        delta=f"{best_tps - base_tps:+,.0f}",
    )
    summary_cols[2].metric(
        "Improvement",
        f"{improvement_pct:+.2f}%",
    )
    summary_cols[3].metric(
        "MFU baseline → best",
        f"{base.get('mfu_pct', 0):.1f} → {best.get('mfu_pct', 0):.1f} %",
    )

    chart = _waste_chart(base, best)
    if chart is not None:
        st.markdown("**Waste reduction (seconds/step, by bucket)**")
        st.altair_chart(chart, use_container_width=True)
        # Per-bucket reduction table
        diff_rows = []
        bwb = base.get("waste_budget") or {}
        cwb = best.get("waste_budget") or {}
        for bucket in WASTE_BUCKETS:
            bv = float(bwb.get(bucket, 0.0))
            cv = float(cwb.get(bucket, 0.0))
            if bv == 0 and cv == 0:
                continue
            diff_rows.append({
                "bucket": bucket,
                "baseline (s)": round(bv, 4),
                "best (s)": round(cv, 4),
                "Δ (s)": round(cv - bv, 4),
            })
        if diff_rows:
            st.dataframe(pd.DataFrame(diff_rows), use_container_width=True)

    accepted = final_summary.get("accepted") or []
    rejected = final_summary.get("rejected") or []

    col_a, col_r = st.columns(2)
    with col_a:
        st.markdown(f"**✅ Accepted ({len(accepted)})**")
        if accepted:
            st.dataframe(
                pd.DataFrame(accepted)[["name", "tps", "delta_pct"]].rename(
                    columns={"tps": "tokens/sec", "delta_pct": "Δ %"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("(no experiments accepted)")
    with col_r:
        st.markdown(f"**❌ Rejected ({len(rejected)})**")
        if rejected:
            st.dataframe(
                pd.DataFrame(rejected),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("(none)")

    env_vars = final_summary.get("best_env_vars") or {}
    if env_vars:
        st.markdown("**Required env vars for best config**")
        st.code("\n".join(f"export {k}={v}" for k, v in env_vars.items()), language="bash")

    best_path = final_summary.get("best_workload_path")
    base_path = final_summary.get("baseline_workload_path")
    if best_path and base_path:
        st.markdown("**Best workload script**")
        st.code(f"diff {base_path} {best_path}", language="bash")
        try:
            best_text = Path(best_path).read_text()
            st.download_button(
                "⬇️ Download best.py",
                data=best_text,
                file_name="best.py",
                mime="text/x-python",
            )
        except OSError:
            pass

# Raw stdout always at the bottom for debugging
with st.expander("Raw subprocess output"):
    st.code("\n".join(stdout_buffer), language="text")
