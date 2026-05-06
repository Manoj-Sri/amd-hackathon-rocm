"""CLI driver — run `python -m agent <file>` to drive the audit without Streamlit.

Useful for local debugging. Prints each SSE event as one JSON line to stdout
so you can pipe it through `jq` or save to a transcript.
"""

from __future__ import annotations

import asyncio
import sys

from agent.loop import run_audit


async def _drive(file_path: str) -> int:
    rc = 0
    async for event in run_audit(file_path):
        # One event per line — newline-delimited JSON for easy `jq` use.
        sys.stdout.write(event.model_dump_json() + "\n")
        sys.stdout.flush()
        if event.type == "error":
            rc = 1
    return rc


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python -m agent <path-to-config-or-script>\n")
        return 2
    return asyncio.run(_drive(sys.argv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
