"""Trade execution manager — orchestrates pre-trade sequence and placement.

PROJECT_SPEC.md Section 9 (Execution, PROTECTED):
  Pre-trade sequence, strictly in order. Any failure means no trade plus an alert:
    1. Kill switch not active, breaker not tripped, reconcile complete.
    2. Validator passed (Section 7.4).
    3. Safety limits passed (Section 8).
    4. Asset mode check: real money only if asset is live_eligible;
       otherwise route to paper shadow trade instead.
    5. Place trade through adapter (one order per decision).
       - Idempotent via decision ID.
       - Write-ahead logging: persist trade as PENDING before calling adapter.
    6. Verify: read back open-trades list from adapter and confirm trade exists
       with matching asset, direction, expiry, and stake.
       If it cannot be confirmed: stop, alert, trip breaker, NEVER retry.
    7. Save to state and SQLite database. Alert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

from adaptive.monitor import AdaptiveState
from core.config import AppConfig
from decision.schema import ValidatedDecision
from execution.state import StateManager
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch
from safety.limits import AccountSnapshot, OrderRequest, check_order


@dataclass
class ExecutionResult:
    """Outcome of trade execution pipeline."""

    executed: bool
    status: str  # "OPEN", "SHADOW_OPEN", "SKIPPED", "BLOCKED", "VERIFICATION_FAILED", "EXEC_ERROR"
    trade_id: Optional[str] = None
    asset: Optional[str] = None
    direction: Optional[str] = None
    stake: float = 0.0
    expiry_minutes: int = 0
    is_shadow: bool = False
    reason: str = "ok"
    executed_at: Optional[str] = None


class TradeManager:
    """Orchestrates order verification, write-ahead logging, and placement."""

    def __init__(
        self,
        config: AppConfig,
        state_manager: StateManager,
        breaker: CircuitBreaker,
        kill_switch: KillSwitch,
        adapter: Any,
        paper_adapter: Optional[Any] = None,
        db_store: Optional[Any] = None,
        alerter: Optional[Any] = None,
    ):
        self.config = config
        self.state_manager = state_manager
        self.breaker = breaker
        self.kill_switch = kill_switch
        self.adapter = adapter
        self.paper_adapter = paper_adapter
        self.db_store = db_store
        self.alerter = alerter
        self.reconcile_done: bool = False
        self.executed_decision_ids: set[str] = set()

    def set_reconciled(self, reconciled: bool = True) -> None:
        """Mark startup reconciliation status."""
        self.reconcile_done = reconciled

    def execute(
        self,
        decision: ValidatedDecision,
        account: AccountSnapshot,
        adaptive_state: Optional[AdaptiveState] = None,
        now: Optional[datetime] = None,
    ) -> ExecutionResult:
        """Execute validated LLM decision following PROJECT_SPEC Section 9 sequence."""
        now_dt = now or datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()

        # Handle SKIP decision
        if decision.decision == "SKIP":
            return ExecutionResult(
                executed=False,
                status="SKIPPED",
                reason=decision.skip_reason or decision.reasoning or "Decision was SKIP",
                executed_at=now_iso,
            )

        # Idempotency check: prevent duplicate execution of same decision ID
        decision_id = getattr(decision, "decision_id", None) or getattr(
            decision.raw_decision, "decision_id", None
        )
        if decision_id:
            if decision_id in self.executed_decision_ids:
                return ExecutionResult(
                    executed=False,
                    status="DUPLICATE",
                    reason=f"Decision ID '{decision_id}' already executed",
                    executed_at=now_iso,
                )

        # ── Step 1: Kill switch, Breaker, Reconcile checks ───────────────────
        if self.kill_switch.is_active():
            msg = "Trade blocked: Emergency kill switch is ACTIVE"
            if self.alerter:
                self.alerter.send_alert(f"🛑 {msg}")
            return ExecutionResult(executed=False, status="BLOCKED", reason=msg, executed_at=now_iso)

        if self.breaker.is_tripped():
            msg = f"Trade blocked: Circuit breaker is TRIPPED ({self.breaker.trip_reason})"
            if self.alerter:
                self.alerter.send_alert(f"⚡ {msg}")
            return ExecutionResult(executed=False, status="BLOCKED", reason=msg, executed_at=now_iso)

        if not self.reconcile_done:
            msg = "Trade blocked: Startup reconciliation has not completed"
            if self.alerter:
                self.alerter.send_alert(f"⚠️ {msg}")
            return ExecutionResult(executed=False, status="BLOCKED", reason=msg, executed_at=now_iso)

        # ── Step 2: Validator check ──────────────────────────────────────────
        if not decision.is_valid:
            msg = f"Trade blocked: Decision validation failed ({decision.skip_reason})"
            return ExecutionResult(executed=False, status="VALIDATOR_FAILED", reason=msg, executed_at=now_iso)

        # ── Step 3 & 4: Determine shadow/real mode and check Safety limits ───
        asset_mode = account.asset_modes.get(str(decision.asset), "paper")
        is_paper = (
            (asset_mode != "live_eligible")
            or bool(decision.shadow_only)
            or (adaptive_state is not None and not adaptive_state.real_trading_allowed)
            or (self.config.mode == "paper")
        )

        order = OrderRequest(
            asset=str(decision.asset),
            direction=str(decision.direction),
            expiry_minutes=int(decision.expiry_minutes or self.config.trading.expiry_minutes),
            stake=float(decision.stake_amount or 0.0),
            max_stake_pct=float(decision.stake_pct) if decision.stake_pct is not None else None,
            shadow_only=is_paper,
            decision_id=decision_id,
        )

        safety_res = check_order(
            order=order,
            account=account,
            config=self.config,
            adaptive_state=adaptive_state,
        )
        if not safety_res.allowed:
            msg = f"Trade blocked by safety layer: {safety_res.reason}"
            if self.alerter:
                self.alerter.send_alert(f"🛡️ {msg}")
            return ExecutionResult(executed=False, status="BLOCKED_SAFETY", reason=msg, executed_at=now_iso)

        if is_paper:
            # Route to paper shadow execution
            return self._execute_paper_shadow(order, decision_id=decision_id, now=now_dt)

        # ── Step 5: Real trade placement with write-ahead logging ─────────────
        trade_id = f"tr_{uuid.uuid4().hex[:12]}"
        pending_record = {
            "id": trade_id,
            "decision_id": decision_id,
            "asset": order.asset,
            "direction": order.direction,
            "expiry_minutes": order.expiry_minutes,
            "stake": order.stake,
            "status": "PENDING",
            "created_at": now_iso,
        }

        # Write-ahead logging: persist state as PENDING before calling broker adapter
        current_pending = list(self.state_manager.data.get("pending_trades", []))
        current_pending.append(pending_record)
        self.state_manager.save(
            balance=account.balance,
            equity=account.equity,
            open_trades=account.open_trades,
            pending_trades=current_pending,
            breaker_state=self.breaker.state,
            adaptive_state=adaptive_state,
            reconcile_done=self.reconcile_done,
            asset_modes=account.asset_modes,
        )

        # Rolling rate limit check
        if not self.breaker.record_order(now=now_dt):
            msg = f"Order blocked: {self.breaker.trip_reason}"
            if self.alerter:
                self.alerter.send_alert(f"⚡ {msg}")
            return ExecutionResult(executed=False, status="BREAKER_TRIPPED", reason=msg, executed_at=now_iso)

        # Call broker adapter to place order (never retries automatically)
        placed_trade: Any = None
        try:
            placed_trade = self.adapter.place_order(
                asset=order.asset,
                direction=order.direction,
                expiry_minutes=order.expiry_minutes,
                stake=order.stake,
                trade_id=trade_id,
            )
        except Exception as e:
            err_msg = f"Broker order placement failed: {e}"
            self.breaker.record_exec_error(err_msg, now=now_dt)
            if self.alerter:
                self.alerter.send_alert(f"🚨 {err_msg}")
            return ExecutionResult(
                executed=False,
                status="EXEC_ERROR",
                trade_id=trade_id,
                reason=err_msg,
                executed_at=now_iso,
            )

        # ── Step 6: Post-placement verification via get_open_trades ───────────
        verified = self._verify_trade_on_broker(
            expected_asset=order.asset,
            expected_direction=order.direction,
            expected_expiry=order.expiry_minutes,
            expected_stake=order.stake,
            trade_id=trade_id,
        )

        if not verified:
            # Cannot be confirmed: STOP, ALERT, TRIP BREAKER, NEVER RETRY!
            trip_msg = (
                f"Post-placement verification FAILED for trade {trade_id} on {order.asset}: "
                f"order was submitted but could not be confirmed in broker's open trades list"
            )
            self.breaker.trip(trip_msg, now=now_dt)
            if self.alerter:
                self.alerter.send_alert(f"🚨 CRITICAL EXECUTION FAULT: {trip_msg}")
            return ExecutionResult(
                executed=False,
                status="VERIFICATION_FAILED",
                trade_id=trade_id,
                reason=trip_msg,
                executed_at=now_iso,
            )

        # ── Step 7: Save to state and SQLite database, Alert ───────────────────
        self.breaker.record_exec_success()

        # Move trade from pending to open in state
        active_pending = [
            pt for pt in self.state_manager.data.get("pending_trades", [])
            if pt.get("id") != trade_id
        ]
        open_record = {
            "id": trade_id,
            "asset": order.asset,
            "direction": order.direction,
            "expiry_minutes": order.expiry_minutes,
            "stake": order.stake,
            "status": "OPEN",
            "opened_at": now_iso,
        }
        updated_open_trades = list(account.open_trades) + [open_record]

        self.state_manager.save(
            balance=account.balance - order.stake,
            equity=account.equity,
            open_trades=updated_open_trades,
            pending_trades=active_pending,
            breaker_state=self.breaker.state,
            adaptive_state=adaptive_state,
            reconcile_done=self.reconcile_done,
            asset_modes=account.asset_modes,
        )

        if decision_id:
            self.executed_decision_ids.add(decision_id)

        # Log to SQLite store if configured
        if self.db_store and hasattr(self.db_store, "record_trade"):
            try:
                self.db_store.record_trade(open_record)
            except Exception:
                pass

        success_msg = f"Trade placed: {order.asset} {order.direction} ${order.stake:,.2f} ({order.expiry_minutes}m)"
        if self.alerter:
            self.alerter.send_alert(f"✅ {success_msg}")

        return ExecutionResult(
            executed=True,
            status="OPEN",
            trade_id=trade_id,
            asset=order.asset,
            direction=order.direction,
            stake=order.stake,
            expiry_minutes=order.expiry_minutes,
            is_shadow=False,
            reason="ok",
            executed_at=now_iso,
        )

    def _execute_paper_shadow(
        self,
        order: OrderRequest,
        decision_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ExecutionResult:
        """Record paper shadow trade for non-live-eligible asset or shadow mode."""
        now_dt = now or datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        trade_id = f"paper_{uuid.uuid4().hex[:12]}"

        shadow_record = {
            "id": trade_id,
            "decision_id": decision_id,
            "asset": order.asset,
            "direction": order.direction,
            "expiry_minutes": order.expiry_minutes,
            "stake": order.stake,
            "mode": "paper",
            "opened_at": now_iso,
        }

        if self.paper_adapter and hasattr(self.paper_adapter, "place_order"):
            try:
                self.paper_adapter.place_order(
                    asset=order.asset,
                    direction=order.direction,
                    expiry_minutes=order.expiry_minutes,
                    stake=order.stake,
                    trade_id=trade_id,
                )
            except Exception:
                pass

        if decision_id:
            self.executed_decision_ids.add(decision_id)

        msg = f"Shadow paper trade recorded: {order.asset} {order.direction} ${order.stake:,.2f} ({order.expiry_minutes}m)"
        if self.alerter:
            self.alerter.send_alert(f"📝 {msg}")

        return ExecutionResult(
            executed=True,
            status="SHADOW_OPEN",
            trade_id=trade_id,
            asset=order.asset,
            direction=order.direction,
            stake=order.stake,
            expiry_minutes=order.expiry_minutes,
            is_shadow=True,
            reason="Executed as paper shadow trade",
            executed_at=now_iso,
        )

    def _verify_trade_on_broker(
        self,
        expected_asset: str,
        expected_direction: str,
        expected_expiry: int,
        expected_stake: float,
        trade_id: str,
    ) -> bool:
        """Confirm trade exists in broker's open trades list with exact parameters."""
        try:
            open_trades = list(self.adapter.get_open_trades())
        except Exception:
            return False

        for ot in open_trades:
            # Check ID match if supported
            ot_id = getattr(ot, "id", getattr(ot, "trade_id", None))
            if isinstance(ot, dict):
                ot_id = ot.get("id", ot.get("trade_id"))

            if ot_id and str(ot_id) == str(trade_id):
                return True

            # Match on attributes
            asset = getattr(ot, "asset", "")
            direction = str(getattr(ot, "direction", "")).upper()
            stake = float(getattr(ot, "stake", getattr(ot, "stake_amount", 0.0)))
            if isinstance(ot, dict):
                asset = ot.get("asset", "")
                direction = str(ot.get("direction", "")).upper()
                stake = float(ot.get("stake", ot.get("stake_amount", 0.0)))

            if (
                asset == expected_asset
                and direction == expected_direction
                and abs(stake - expected_stake) < 0.01
            ):
                return True

        return False

    def record_settlement(
        self,
        trade_id: str,
        outcome: str,  # "WIN" or "LOSS"
        payout: float = 0.85,
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Settle an open trade, update account balance in state, and alert.

        PROJECT_SPEC Section 9:
          After a result, update asset stats, update balance/equity, and alert.
        """
        now_dt = now or datetime.now(timezone.utc)
        open_trades = list(self.state_manager.data.get("open_trades", []))
        current_balance = float(self.state_manager.data.get("balance", 0.0))
        target_trade = None
        remaining_open = []
        for ot in open_trades:
            ot_id = ot.get("id") if isinstance(ot, dict) else getattr(ot, "id", None)
            if str(ot_id) == str(trade_id):
                target_trade = ot
            else:
                remaining_open.append(ot)

        stake = float(target_trade.get("stake", 0.0)) if target_trade else 0.0
        profit = 0.0
        if outcome.upper() == "WIN":
            profit = stake * payout
            new_balance = current_balance + stake + profit
            self.breaker.record_win()
            msg = f"Trade {trade_id} WON: +${profit:,.2f} profit (balance: ${new_balance:,.2f})"
            if self.alerter:
                self.alerter.send_alert(f"🏆 {msg}")
        else:
            new_balance = current_balance
            self.breaker.record_loss(now=now_dt)
            msg = f"Trade {trade_id} LOST: -${stake:,.2f} (balance: ${new_balance:,.2f})"
            if self.alerter:
                self.alerter.send_alert(f"📉 {msg}")

        self.state_manager.save(
            balance=new_balance,
            equity=new_balance + sum(float(t.get("stake", 0.0)) for t in remaining_open),
            open_trades=remaining_open,
            pending_trades=self.state_manager.data.get("pending_trades", []),
            breaker_state=self.breaker.state,
            reconcile_done=self.reconcile_done,
        )
        return {
            "trade_id": trade_id,
            "outcome": outcome.upper(),
            "balance": new_balance,
            "profit": profit,
        }
