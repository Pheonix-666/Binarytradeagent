"""Clock utilities for consistent time handling and deterministic simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import time
from typing import Optional


class Clock(ABC):
    """Abstract clock interface for time queries and sleep."""

    @abstractmethod
    def now(self) -> datetime:
        """Return current datetime in UTC timezone."""
        ...

    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """Pause execution or advance fake time."""
        ...


class RealClock(Clock):
    """Real wall-clock system time."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class FakeClock(Clock):
    """Controllable fake clock for deterministic testing."""

    def __init__(
        self,
        start_time: Optional[datetime] = None,
        initial_time: Optional[datetime] = None,
    ):
        effective_time = initial_time if initial_time is not None else start_time
        if effective_time is None:
            self._current = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        elif effective_time.tzinfo is None:
            self._current = effective_time.replace(tzinfo=timezone.utc)
        else:
            self._current = effective_time.astimezone(timezone.utc)

    def now(self) -> datetime:
        return self._current

    def advance(
        self,
        delta: Optional[timedelta | float] = None,
        seconds: Optional[float] = None,
    ) -> datetime:
        if seconds is not None:
            delta = timedelta(seconds=seconds)
        elif isinstance(delta, (int, float)):
            delta = timedelta(seconds=delta)
        elif delta is None:
            delta = timedelta(0)
        self._current += delta
        return self._current

    def set_time(self, dt: datetime) -> None:
        if dt.tzinfo is None:
            self._current = dt.replace(tzinfo=timezone.utc)
        else:
            self._current = dt.astimezone(timezone.utc)

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.advance(seconds)


def now_utc() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(timezone.utc)
