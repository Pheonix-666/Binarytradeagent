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
import logging
from typing import Any, Optional

from safety.breaker import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class ReconcileResult:
    """Summary of startup reconciliation."""

    success: bool
    reconciled_open_trades: list[Any] = field(default_factory=list)
    unknown_trades: list[Any] = field(default_factory=list)
    resolved_pending: list[dict[str, Any]] = field(default_factory=list)
    offline_settled_trades: list[Any] = field(default_factory=list)
    reported_balance: Optional[float] = None
    expected_balance: float = 0.0
    issues: list[str] = field(default_factory=list)


def reconcile_startup(
    adapter: Any,
    saved_state: dict[str, Any],
    breaker: CircuitBreaker,
    alerter: Optional[Any] = None,
    tolerance: Optional[float] = None,
    now: Optional[datetime] = None,
) -> ReconcileResult:
    """Perform pre-trading reconciliation on system startup.

    Parameters:
      adapter: Broker or exchange adapter providing get_open_trades() and get_balance().
      saved_state: Persistent state dictionary from StateManager.load().
      breaker: System CircuitBreaker instance.
      alerter: Optional Telegram/system alerting facility.
      tolerance: Allowable cash mismatch tolerance in dollars (defaults to safety.balance_tolerance).
      now: Current UTC timestamp.

    Returns:
      ReconcileResult indicating whether startup reconciliation passed.
    """
    now_dt = now or datetime.now(timezone.utc)
    issues: list[str] = []
    tol = (
        tolerance
        if tolerance is not None
        else getattr(breaker.config.safety, "balance_tolerance", 1.0)
        if hasattr(breaker, "config") and hasattr(breaker.config, "safety")
        else 1.0
    )
    unknown_trades: list[Any] = []
    resolved_pending: list[dict[str, Any]] = []
    offline_settled_trades: list[Any] = []

    # 1. Read open trades and balance from broker adapter
    try:
        broker_open = list(adapter.get_open_trades())
    except Exception as e:
        msg = f"Reconcile failed: error reading open trades from broker ({e})"
        issues.append(msg)
        breaker.trip(msg, now=now_dt)
        if alerter:
            try:
                alerter.send_alert(f"🚨 RECONCILE FAILURE: {msg}")
            except Exception:
                pass
        return ReconcileResult(success=False, issues=issues)

    try:
        broker_balance = float(adapter.get_balance())
    except Exception as e:
        msg = f"Reconcile failed: error reading balance from broker ({e})"
        issues.append(msg)
        breaker.trip(msg, now=now_dt)
        if alerter:
            try:
                alerter.send_alert(f"🚨 RECONCILE FAILURE: {msg}")
            except Exception:
                pass
        return ReconcileResult(success=False, issues=issues)

    # Read settlement history if supported
    history_readable = True
    broker_history: list[Any] = []
    if hasattr(adapter, "get_trade_history"):
        try:
            broker_history = list(adapter.get_trade_history())
        except Exception:
            history_readable = False
    else:
        history_readable = False

    saved_open = list(saved_state.get("open_trades", []))
    saved_pending = list(saved_state.get("pending_trades", []))
    base_expected_balance = float(saved_state.get("balance", broker_balance))

    def _tid(t: Any) -> str:
        if isinstance(t, dict):
            return str(t.get("id") or t.get("trade_id") or "")
        return str(getattr(t, "id", getattr(t, "trade_id", "")))

    broker_open_by_id = {_tid(t): t for t in broker_open if _tid(t)}
    broker_history_by_id = {_tid(t): t for t in broker_history if _tid(t)}
    saved_open_ids = {_tid(t) for t in saved_open if _tid(t)}

    # 2. Resolve PENDING write-ahead trades
    for pt in saved_pending:
        p_id = _tid(pt)
        if p_id in broker_open_by_id:
            resolved_pending.append({
                "trade_id": p_id,
                "status": "OPEN",
                "resolution": "Confirmed open on broker platform",
            })
            saved_open.append(broker_open_by_id[p_id])
            saved_open_ids.add(p_id)
        elif not history_readable:
            # Pending trades with unreadable history become UNKNOWN and trip breaker, never CANCELLED
            unknown_trades.append(pt)
            msg = f"Pending trade {p_id} could not be resolved: broker history unreadable"
            issues.append(msg)
            breaker.trip(f"Unreadable broker history for pending trade: {p_id}", now=now_dt)
            if alerter:
                try:
                    alerter.send_alert(f"🚨 UNKNOWN PENDING TRADE: {msg}")
                except Exception:
                    pass
            resolved_pending.append({
                "trade_id": p_id,
                "status": "UNKNOWN",
                "resolution": "Broker history unreadable; marked UNKNOWN",
            })
        elif p_id in broker_history_by_id:
            resolved_pending.append({
                "trade_id": p_id,
                "status": "SETTLED",
                "resolution": "Found in broker settlement history",
            })
            offline_settled_trades.append(broker_history_by_id[p_id])
        else:
            resolved_pending.append({
                "trade_id": p_id,
                "status": "CANCELLED",
                "resolution": "Not found on platform; marked unplaced",
            })

    # 3. Verify saved OPEN trades
    reconciled_open: list[Any] = []
    for ot in saved_open:
        o_id = _tid(ot)
        if o_id in broker_open_by_id:
            reconciled_open.append(broker_open_by_id[o_id])
        else:
            # Saved open trade is missing from broker open trades
            if not history_readable:
                msg = f"Saved open trade {o_id} missing from broker and history unreadable"
                issues.append(msg)
                breaker.trip(msg, now=now_dt)
                if alerter:
                    try:
                        alerter.send_alert(f"🚨 RECONCILE FAILURE: {msg}")
                    except Exception:
                        pass
            elif o_id in broker_history_by_id:
                # Settled offline — settle and remove from open
                h_trade = broker_history_by_id[o_id]
                offline_settled_trades.append(h_trade)
            else:
                msg = f"Saved open trade {o_id} missing from broker open trades and settlement history"
                issues.append(msg)
                breaker.trip(msg, now=now_dt)
                if alerter:
                    try:
                        alerter.send_alert(f"🚨 RECONCILE FAILURE: {msg}")
                    except Exception:
                        pass

    # 4. Detect unknown trades on broker (not initiated by bot)
    for b_trade in broker_open:
        b_id = _tid(b_trade)
        if b_id not in saved_open_ids:
            unknown_trades.append(b_trade)
            msg = (
                f"Unknown platform trade detected on broker: ID '{b_id}', "
                f"asset '{getattr(b_trade, 'asset', '')}', stake ${getattr(b_trade, 'stake', 0.0)}"
            )
            issues.append(msg)
            breaker.trip(f"Unknown open trade on platform: {msg}", now=now_dt)
            if alerter:
                try:
                    alerter.send_alert(f"🚨 UNKNOWN TRADE ALERT: {msg}")
                except Exception:
                    pass

    # Reconciled open trades include recognized open trades PLUS unknown trades (for exposure safety)
    final_reconciled_open = list(reconciled_open) + list(unknown_trades)

    # 5. Compute expected balance applying trades settled offline from broker history
    expected_balance = base_expected_balance
    tie_policy = getattr(breaker.config.trading, "tie_policy", "loss") if hasattr(breaker, "config") and hasattr(breaker.config, "trading") else "loss"
    for st in offline_settled_trades:
        res = getattr(st, "result", None) or (st.get("result") if isinstance(st, dict) else None)
        res_str = str(getattr(res, "value", res)).upper()
        raw_stake = getattr(st, "stake", getattr(st, "stake_amount", 0.0)) if not isinstance(st, dict) else st.get("stake", st.get("stake_amount", 0.0))
        stake = float(raw_stake or 0.0)
        payout = float(getattr(st, "payout", 0.85) if not isinstance(st, dict) else st.get("payout", 0.85))
        payout_recv = getattr(st, "payout_received", None) if not isinstance(st, dict) else st.get("payout_received")

        if payout_recv is not None:
            expected_balance += float(payout_recv)
        elif "WIN" in res_str:
            expected_balance += stake + (stake * payout)
        elif "TIE" in res_str:
            if tie_policy == "refund":
                expected_balance += stake
        elif "VOID" in res_str:
            expected_balance += stake

    # 6. Verify balance accounting
    balance_matched = breaker.check_balance_match(
        reported_balance=broker_balance,
        expected_balance=expected_balance,
        tolerance=tol,
        now=now_dt,
    )
    if not balance_matched:
        diff = abs(broker_balance - expected_balance)
        msg = f"Balance mismatch at startup: broker=${broker_balance:,.2f} vs expected=${expected_balance:,.2f} (diff ${diff:,.2f})"
        issues.append(msg)
        if alerter:
            try:
                alerter.send_alert(f"🚨 BALANCE RECONCILIATION FAILURE: {msg}")
            except Exception:
                pass

    success = (len(issues) == 0) and not breaker.is_tripped()

    return ReconcileResult(
        success=success,
        reconciled_open_trades=final_reconciled_open,
        unknown_trades=unknown_trades,
        resolved_pending=resolved_pending,
        offline_settled_trades=offline_settled_trades,
        reported_balance=broker_balance,
        expected_balance=expected_balance,
        issues=issues,
    )
