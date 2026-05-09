"""CLI driver — run `python -m agent <file>` to drive the audit without Streamlit.

Two output modes, picked automatically:
  * stdout is a TTY → pretty-printed, human-readable transcript (default
    for interactive debugging — fixes the "logs are clustered and messy"
    problem from running raw NDJSON through a terminal).
  * stdout is a pipe → newline-delimited JSON, one SSEEvent per line, so
    `python -m agent foo.py | jq` still works.

Force JSON regardless of TTY with `GOBLIN_AGENT_OUTPUT=json`.
Internal logging (`_LOG.warning(...)`) goes to stderr with a real
formatter — see `agent.logging_setup.configure`.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from agent.logging_setup import configure as _configure_logging
from agent.loop import run_audit
from agent.schemas import SSEEvent


def _is_pretty_mode() -> bool:
    forced = os.environ.get("GOBLIN_AGENT_OUTPUT", "").lower()
    if forced == "json":
        return False
    if forced == "pretty":
        return True
    return sys.stdout.isatty()


def _render_event_pretty(event: SSEEvent) -> str:
    """Human-friendly multi-line block per event.

    Each event gets a header line and an indented body so the agent's
    thoughts, tool calls, and tool results aren't mashed together.
    """
    header = _event_header(event)
    body = _event_body(event)
    if body:
        return f"{header}\n{_indent(body, '  ')}\n"
    return f"{header}\n"


def _event_header(event: SSEEvent) -> str:
    t = event.type
    if t == "thought":
        return "── thought ──"
    if t == "tool_call":
        return f"── tool_call: {event.data.get('name', '?')} ──"
    if t == "tool_result":
        ok = event.data.get("ok")
        marker = "ok" if ok else "FAIL"
        return f"── tool_result: {event.data.get('name', '?')} [{marker}] ──"
    if t == "final_report":
        return "── final_report ──"
    if t == "error":
        return "── error ──"
    return f"── {t} ──"


def _event_body(event: SSEEvent) -> str:
    data = event.data
    t = event.type
    if t == "thought":
        return str(data.get("text", "")).rstrip()
    if t == "tool_call":
        payload = data.get("input")
        return json.dumps(payload, indent=2, sort_keys=True, default=str)
    if t == "tool_result":
        # Surface warnings on their own lines so they don't get lost inside
        # a 1-KB JSON blob — this is the "I missed the amd-smi warning"
        # problem from the original report.
        result = data.get("result")
        warnings = _extract_warnings(result)
        result_block = json.dumps(result, indent=2, sort_keys=True, default=str)
        if warnings:
            warning_block = "\n".join(f"  ! {w}" for w in warnings)
            return f"warnings:\n{warning_block}\nresult:\n{result_block}"
        return result_block
    if t == "error":
        return str(data.get("message", ""))
    return json.dumps(data, indent=2, sort_keys=True, default=str)


def _extract_warnings(result: Any) -> list[str]:
    if isinstance(result, dict):
        warnings = result.get("warnings")
        if isinstance(warnings, list):
            return [str(w) for w in warnings]
    return []


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


async def _drive(file_path: str) -> int:
    rc = 0
    pretty = _is_pretty_mode()
    async for event in run_audit(file_path):
        if pretty:
            sys.stdout.write(_render_event_pretty(event) + "\n")
        else:
            sys.stdout.write(event.model_dump_json() + "\n")
        sys.stdout.flush()
        if event.type == "error":
            rc = 1
    return rc


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python -m agent <path-to-config-or-script>\n")
        return 2
    _configure_logging()
    return asyncio.run(_drive(sys.argv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
