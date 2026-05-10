"""Structured logging setup for the agent + runner stack.

Without this, the `_LOG.warning(...)` calls scattered through
`agent/tools/benchmark.py`, `runner/protocol.py`, and `runner/profile_parser.py`
hit Python's `logging.lastResort` handler — which prints unformatted text
to stderr without timestamps or module names. That's the "messy logs"
symptom.

`configure(level=...)` is idempotent and safe to call from multiple entry
points (CLI driver, FastAPI server, tests). Honors the `GOBLIN_LOG_LEVEL`
env var so an operator can crank to DEBUG without code changes.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure(level: int | str | None = None) -> None:
    """Install a single, formatted stderr handler on the root logger.

    Idempotent: a second call replaces the formatter/level but does not
    pile additional handlers on top.
    """
    global _CONFIGURED

    resolved_level = _resolve_level(level)

    root = logging.getLogger()
    root.setLevel(resolved_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )

    if _CONFIGURED:
        for handler in root.handlers:
            handler.setLevel(resolved_level)
            handler.setFormatter(formatter)
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(resolved_level)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Keep noisy libraries from drowning the goblin's own warnings unless
    # the operator explicitly asked for DEBUG.
    if resolved_level > logging.DEBUG:
        for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def _resolve_level(level: int | str | None) -> int:
    if level is not None:
        if isinstance(level, int):
            return level
        return logging.getLevelName(level.upper())
    env = os.environ.get("GOBLIN_LOG_LEVEL")
    if env:
        return logging.getLevelName(env.upper())
    return logging.INFO
