"""Circuit breaker — halts trading on operational malfunctions.

PROJECT_SPEC.md Section 8 (Safety Layer):
  - Pause trading and alert if any happen:
      1. More than max_orders_per_window orders in safety.rate_limit_window_minutes.
      2. Price data older than stale_data_multiplier * check_interval.
      3. max_consecutive_exec_errors execution errors in a row.
      4. Balance read fails or changes in a way that doesn't match recorded trades.
  - Stays tripped across restarts until explicit manual reset command.
  - Serializable: to_dict() and from_dict().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Optional

from core.config import AppConfig


@dataclass
class BreakerState:
    """Serializable persistent state of the circuit breaker."""

    tripped: bool = False
    trip_reason: Optional[str] = None
    tripped_at: Optional[str] = None
    consecutive_exec_errors: int = 0
    recent_order_timestamps: list[str] = field(default_factory=list)
    last_trip_history: list[dict[str, Any]] = field(default_factory=list)


class CircuitBreaker:
    """Monitors system operation and trips on safety/rate/data faults."""

    def __init__(self, config: AppConfig, state: Optional[BreakerState] = None):
        self.config = config
        self.state = state or BreakerState()

    def is_tripped(self) -> bool:
        """Check if breaker is currently in tripped state."""
        return self.state.tripped

    @property
    def trip_reason(self) -> Optional[str]:
        return self.state.trip_reason

    def trip(self, reason: str, now: Optional[datetime] = None) -> None:
        """Trip circuit breaker with reason."""
        now_dt = now or datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        self.state.tripped = True
        self.state.trip_reason = reason
        self.state.tripped_at = now_iso
        self.state.last_trip_history.append({
            "timestamp": now_iso,
            "reason": reason,
        })

    def reset(self) -> None:
        """Manually reset the circuit breaker (explicit operator command only)."""
        self.state.tripped = False
        self.state.trip_reason = None
        self.state.tripped_at = None
        self.state.consecutive_exec_errors = 0

    def record_order(self, now: Optional[datetime] = None) -> bool:
        """Record an order placement and check rolling 10-minute rate limit.

        Returns:
            True if order allowed, False if limit exceeded and breaker tripped.
        """
        if self.state.tripped:
            return False

        now_dt = now or datetime.now(timezone.utc)
        window_mins = (
            getattr(self.config.safety, "rate_limit_window_minutes", 10)
            if hasattr(self.config, "safety")
            else 10
        )
        window_ago = now_dt - timedelta(minutes=window_mins)

        # Filter order timestamps within rate limit window
        recent = [
            ts for ts in self.state.recent_order_timestamps
            if datetime.fromisoformat(ts) >= window_ago
        ]
        recent.append(now_dt.isoformat())
        self.state.recent_order_timestamps = recent

        max_orders = self.config.safety.max_orders_per_window
        if len(recent) > max_orders:
            self.trip(
                f"Rate limit exceeded: {len(recent)} orders in {window_mins} minutes > maximum {max_orders}",
                now=now_dt,
            )
            return False

        return True

    def check_data_freshness(
        self,
        data_time: datetime,
        now: Optional[datetime] = None,
    ) -> bool:
        """Verify market data age against stale_data_multiplier * check_interval.

        Returns:
            True if fresh, False if stale and breaker tripped.
        """
        if self.state.tripped:
            return False

        now_dt = now or datetime.now(timezone.utc)
        interval_mins = self.config.trading.check_interval_minutes
        multiplier = self.config.safety.stale_data_multiplier
        max_age_secs = multiplier * interval_mins * 60.0

        if data_time.tzinfo is None:
            data_time = data_time.replace(tzinfo=timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)

        age_secs = (now_dt - data_time).total_seconds()
        if age_secs > max_age_secs:
            self.trip(
                f"Stale market data: age {age_secs:.1f}s > maximum {max_age_secs:.1f}s "
                f"({multiplier}x check interval {interval_mins}m)",
                now=now_dt,
            )
            return False

        return True

    def record_exec_success(self) -> None:
        """Reset consecutive execution error streak on successful execution."""
        self.state.consecutive_exec_errors = 0

    def record_exec_error(self, error_msg: str, now: Optional[datetime] = None) -> bool:
        """Record an execution error and trip if streak exceeds maximum.

        Returns:
            True if streak still within limits, False if threshold reached and breaker tripped.
        """
        now_dt = now or datetime.now(timezone.utc)
        self.state.consecutive_exec_errors += 1
        max_errors = self.config.safety.max_consecutive_exec_errors

        if self.state.consecutive_exec_errors >= max_errors:
            self.trip(
                f"Consecutive execution errors: {self.state.consecutive_exec_errors} >= {max_errors} (last error: {error_msg})",
                now=now_dt,
            )
            return False

        return True

    def check_balance_match(
        self,
        reported_balance: Optional[float],
        expected_balance: float,
        tolerance: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """Verify broker reported balance matches expected trade-reconciled balance.

        Returns:
            True if balance matches within tolerance, False if mismatch and breaker tripped.
        """
        now_dt = now or datetime.now(timezone.utc)
        tol = (
            tolerance
            if tolerance is not None
            else self.config.safety.balance_tolerance
            if hasattr(self.config, "safety")
            else 1.0
        )

        if reported_balance is None:
            self.trip("Balance read failure from broker/adapter", now=now_dt)
            return False

        if not math.isfinite(reported_balance) or not math.isfinite(expected_balance):
            self.trip(
                f"Balance mismatch: non-finite balance reported (reported={reported_balance}, expected={expected_balance})",
                now=now_dt,
            )
            return False

        diff = abs(reported_balance - expected_balance)
        if diff > tol:
            self.trip(
                f"Balance mismatch: reported ${reported_balance:,.2f} vs expected "
                f"${expected_balance:,.2f} (diff ${diff:,.2f} > tolerance ${tol:,.2f})",
                now=now_dt,
            )
            return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize circuit breaker state."""
        return asdict(self.state)

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: AppConfig) -> CircuitBreaker:
        """Restore circuit breaker from serialized dictionary."""
        state = BreakerState(**data)
        return cls(config=config, state=state)
