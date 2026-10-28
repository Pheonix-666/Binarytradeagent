"""Clock utilities for consistent time handling."""

from datetime import datetime, timezone


def now_utc() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(timezone.utc)
