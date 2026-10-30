"""JSONL logging setup for the trading system."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.config import resolve_project_path

_LOGS_DIR = resolve_project_path("logs")
_LOGS_DIR.mkdir(exist_ok=True)


def _get_jsonl_path() -> Path:
    """Return the JSONL log path for today (UTC)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _LOGS_DIR / f"decisions_{today}.jsonl"


# ── Console logger ────────────────────────────────────────────────────

_logger = logging.getLogger("trading_system")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)


def get_logger() -> logging.Logger:
    """Return the shared application logger."""
    return _logger


# ── JSONL decision logger ─────────────────────────────────────────────


def log_decision(data: dict) -> None:
    """Append a single JSON line to today's decision log.

    Each line includes an ISO-8601 ``timestamp`` key merged with *data*.
    """
    _LOGS_DIR.mkdir(exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **data}
    with open(_get_jsonl_path(), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
