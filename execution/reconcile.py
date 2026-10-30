"""Startup reconciliation between database/state and broker platform.

PROJECT_SPEC.md Sections 8 & 9 (Execution / Safety):
  - System refuses to trade until startup reconcile completes.
  - Pulls open trades from broker adapter and matches against database/state.
  - Resolves PENDING write-ahead trades (found open -> OPEN, settled -> CLOSED, missing -> cancelled).
  - Unknown platform trades: count toward total exposure, generate urgent alert, trip breaker!
  - Verifies broker balance against trade accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from safety.breaker import CircuitBreaker


@dataclass
class ReconcileResult:
    """Summary of startup reconciliation."""

    success: bool
    reconciled_open_trades: list[Any] = field(default_factory=list)
    unknown_trades: list[Any] = field(default_factory=list)
    resolved_pending: list[dict[str, Any]] = field(default_factory=list)
    reported_balance: Optional[float] = None
    expected_balance: float = 0.0
    issues: list[str] = field(default_factory=list)


def reconcile_startup(
    adapter: Any,
    saved_state: dict[str, Any],
    breaker: CircuitBreaker,
    alerter: Optional[Any] = None,
    tolerance: float = 1.0,
    now: Optional[datetime] = None,
) -> ReconcileResult:
    """Perform pre-trading reconciliation on system startup.

    Parameters:
      adapter: Broker or exchange adapter providing get_open_trades() and get_balance().
      saved_state: Persistent state dictionary from StateManager.load().
      breaker: System CircuitBreaker instance.
      alerter: Optional Telegram/system alerting facility.
      tolerance: Allowable cash mismatch tolerance in dollars.
      now: Current UTC timestamp.

    Returns:
      ReconcileResult indicating whether startup reconciliation passed.
    """
    now_dt = now or datetime.now(timezone.utc)
    issues: list[str] = []
    unknown_trades: list[Any] = []
    resolved_pending: list[dict[str, Any]] = []

    # 1. Read open trades and balance from broker adapter
    try:
        broker_open = list(adapter.get_open_trades())
    except Exception as e:
        msg = f"Reconcile failed: error reading open trades from broker ({e})"
        issues.append(msg)
        breaker.trip(msg, now=now_dt)
        if alerter:
            alerter.send_alert(f"🚨 RECONCILE FAILURE: {msg}")
        return ReconcileResult(success=False, issues=issues)

    try:
        broker_balance = float(adapter.get_balance())
    except Exception as e:
        msg = f"Reconcile failed: error reading balance from broker ({e})"
        issues.append(msg)
        breaker.trip(msg, now=now_dt)
        if alerter:
            alerter.send_alert(f"🚨 RECONCILE FAILURE: {msg}")
        return ReconcileResult(success=False, issues=issues)

    saved_open = list(saved_state.get("open_trades", []))
    saved_pending = list(saved_state.get("pending_trades", []))
    expected_balance = float(saved_state.get("balance", broker_balance))

    # Helper to get trade ID
    def _tid(t: Any) -> str:
        if isinstance(t, dict):
            return str(t.get("id") or t.get("trade_id") or "")
        return str(getattr(t, "id", getattr(t, "trade_id", "")))

    broker_trades_by_id = {_tid(t): t for t in broker_open if _tid(t)}
    saved_open_ids = {_tid(t) for t in saved_open if _tid(t)}

    # 2. Resolve PENDING write-ahead trades
    for pt in saved_pending:
        p_id = _tid(pt)
        if p_id in broker_trades_by_id:
            # Confirmed open on broker -> upgrade to OPEN
            resolved_pending.append({
                "trade_id": p_id,
                "status": "OPEN",
                "resolution": "Confirmed open on broker platform",
            })
            saved_open.append(broker_trades_by_id[p_id])
            saved_open_ids.add(p_id)
        else:
            # Check if settled on broker or missing
            settled_trade = None
            if hasattr(adapter, "get_trade_history"):
                try:
                    for h in adapter.get_trade_history():
                        if _tid(h) == p_id:
                            settled_trade = h
                            break
                except Exception:
                    pass

            if settled_trade:
                resolved_pending.append({
                    "trade_id": p_id,
                    "status": "SETTLED",
                    "resolution": "Found in broker settlement history",
                })
            else:
                resolved_pending.append({
                    "trade_id": p_id,
                    "status": "CANCELLED",
                    "resolution": "Not found on platform; marked unplaced",
                })

    # 3. Detect unknown trades on broker (not initiated by bot)
    for b_trade in broker_open:
        b_id = _tid(b_trade)
        if b_id not in saved_open_ids:
            unknown_trades.append(b_trade)
            msg = (
                f"Unknown platform trade detected on broker: ID '{b_id}', "
                f"asset '{getattr(b_trade, 'asset', '')}', stake ${getattr(b_trade, 'stake', 0.0)}"
            )
            issues.append(msg)
            # Unknown trades count toward exposure, alert, and trip breaker!
            breaker.trip(f"Unknown open trade on platform: {msg}", now=now_dt)
            if alerter:
                alerter.send_alert(f"🚨 UNKNOWN TRADE ALERT: {msg}")

    # Reconciled open trades include recognized open trades PLUS unknown trades (for exposure safety)
    reconciled_open = list(saved_open) + list(unknown_trades)

    # 4. Verify balance accounting
    balance_matched = breaker.check_balance_match(
        reported_balance=broker_balance,
        expected_balance=expected_balance,
        tolerance=tolerance,
        now=now_dt,
    )
    if not balance_matched:
        diff = abs(broker_balance - expected_balance)
        msg = f"Balance mismatch at startup: broker=${broker_balance:,.2f} vs expected=${expected_balance:,.2f} (diff ${diff:,.2f})"
        issues.append(msg)
        if alerter:
            alerter.send_alert(f"🚨 BALANCE RECONCILIATION FAILURE: {msg}")

    success = (len(issues) == 0) and not breaker.is_tripped()

    return ReconcileResult(
        success=success,
        reconciled_open_trades=reconciled_open,
        unknown_trades=unknown_trades,
        resolved_pending=resolved_pending,
        reported_balance=broker_balance,
        expected_balance=expected_balance,
        issues=issues,
    )
