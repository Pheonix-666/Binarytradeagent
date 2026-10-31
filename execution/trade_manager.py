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
import logging
from typing import Any, Optional
import uuid

logger = logging.getLogger(__name__)

from adaptive.monitor import AdaptiveState
from core.config import AppConfig
from core.models import CanonicalTrade, TradeDirection, TradeReceipt
from data.store import TradeRecord, TradeRepository
from decision.schema import ValidatedDecision
from execution.state import StateManager
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch
from safety.limits import AccountSnapshot, OrderRequest, check_order


@dataclass
class ExecutionResult:
    """Outcome of trade execution pipeline."""

    executed: bool
    status: str  # "OPEN", "SHADOW_OPEN", "SKIPPED", "BLOCKED", "VERIFICATION_FAILED", "EXEC_ERROR", "DUPLICATE"
    trade_id: Optional[str] = None
    asset: Optional[str] = None
    direction: Optional[str] = None
    stake: float = 0.0
    expiry_minutes: int = 0
    is_shadow: bool = False
    reason: str = "ok"
    executed_at: Optional[str] = None
    receipt: Optional[TradeReceipt] = None


class TradeManager:
    """Orchestrates order verification, write-ahead logging, and placement."""

    # Safety-critical reasons that are never throttled (always sent immediately)
    _SAFETY_CRITICAL_PREFIXES = (
        "unknown direction",
        "Verification failed",
        "Verification FAILED",
        "CRITICAL",
        "Circuit breaker",
        "Emergency kill switch",
        "balance mismatch",
        "Balance mismatch",
        "Ambiguous adapter failure",
    )

    def __init__(
        self,
        config: AppConfig,
        state_manager: StateManager,
        breaker: CircuitBreaker,
        kill_switch: KillSwitch,
        adapter: Any,
        paper_adapter: Optional[Any] = None,
        repository: Optional[Any] = None,
        alerter: Optional[Any] = None,
        alert_throttle_minutes: int = 15,
        db_store: Optional[Any] = None,
    ):
        self.config = config
        self.state_manager = state_manager
        self.breaker = breaker
        self.kill_switch = kill_switch
        self.adapter = adapter
        self.paper_adapter = paper_adapter
        self.repository = repository if repository is not None else db_store
        self.alerter = alerter
        self.reconcile_done: bool = False
        self.alert_throttle_minutes = alert_throttle_minutes
        self.consecutive_alert_failures: int = 0
        self._alert_throttle: dict[tuple[str, str], tuple[datetime, int]] = {}

        # Restore executed decision IDs from state (pending and open trades)
        self.executed_decision_ids: set[str] = set()
        saved_open = list(self.state_manager.data.get("open_trades", []))
        saved_pending = list(self.state_manager.data.get("pending_trades", []))
        for t in saved_open + saved_pending:
            d_id = t.get("decision_id") if isinstance(t, dict) else getattr(t, "decision_id", None)
            if d_id:
                self.executed_decision_ids.add(str(d_id))

    def set_reconciled(self, reconciled: bool = True) -> None:
        """Mark startup reconciliation status."""
        self.reconcile_done = reconciled

    def _is_safety_critical(self, text: str) -> bool:
        """Return True if this alert must never be throttled."""
        return any(p in text for p in self._SAFETY_CRITICAL_PREFIXES)

    def _send_alert_safe(self, text: str) -> bool:
        """Safely send an alert, recording consecutive failures without throwing."""
        if self.alerter is None:
            return True
        try:
            ok = self.alerter.send_alert(text)
            if ok is False:
                self.consecutive_alert_failures += 1
            else:
                self.consecutive_alert_failures = 0
            return bool(ok)
        except Exception as exc:
            logger.error("Failed sending alert: %s", exc)
            self.consecutive_alert_failures += 1
            return False

    def _throttled_alert(
        self,
        text: str,
        asset: str = "",
        now: Optional[datetime] = None,
    ) -> None:
        """Send an alert, throttling non-critical 'Trade blocked' messages."""
        if self.alerter is None:
            return
        now_dt = now or datetime.now(timezone.utc)

        if self._is_safety_critical(text):
            self._send_alert_safe(text)
            return

        key = (text[:80], asset)
        entry = self._alert_throttle.get(key)
        if entry is not None:
            last_sent, suppressed = entry
            elapsed = (now_dt - last_sent).total_seconds() / 60.0
            if elapsed < self.alert_throttle_minutes:
                self._alert_throttle[key] = (last_sent, suppressed + 1)
                return
            if suppressed > 0:
                text = f"{text} [+{suppressed} suppressed in last {self.alert_throttle_minutes}m]"

        self._alert_throttle[key] = (now_dt, 0)
        self._send_alert_safe(text)

    def _count_matching_open_trades(
        self,
        expected_asset: str,
        expected_direction: str,
        expected_expiry: int,
        expected_stake: float,
    ) -> int:
        """Count currently open trades on adapter matching all attributes. Raises if snapshot cannot be read."""
        trades = list(self.adapter.get_open_trades())
        stake_tol = getattr(self.config.safety, "stake_verification_tolerance", 0.01) if hasattr(self.config, "safety") else 0.01
        count = 0
        for t in trades:
            ct = CanonicalTrade.from_any(t)
            if (
                ct.asset == expected_asset
                and ct.direction == expected_direction
                and ct.expiry_minutes == expected_expiry
                and abs(ct.stake - expected_stake) <= stake_tol
            ):
                count += 1
        return count

    def _verify_trade_on_broker(
        self,
        expected_asset: str,
        expected_direction: str,
        expected_expiry: int,
        expected_stake: float,
        trade_id: str,
        platform_trade_id: Optional[str] = None,
        before_count: int = 0,
    ) -> bool:
        """Confirm trade exists in broker open trades list via ID or exact count delta."""
        try:
            open_trades = list(self.adapter.get_open_trades())
        except Exception:
            return False

        ids_to_check = {str(trade_id)}
        if platform_trade_id:
            ids_to_check.add(str(platform_trade_id))

        for ot in open_trades:
            ct = CanonicalTrade.from_any(ot)
            if ct.id and ct.id in ids_to_check:
                return True

        # Fallback: if platform does not return/report matching ID,
        # open trades count must have increased by exactly +1
        try:
            after_count = self._count_matching_open_trades(
                expected_asset, expected_direction, expected_expiry, expected_stake
            )
        except Exception:
            return False
        return after_count == before_count + 1

    def place_trade(
        self,
        decision: ValidatedDecision,
        account: AccountSnapshot,
        adaptive_state: Optional[AdaptiveState] = None,
        now: Optional[datetime] = None,
    ) -> TradeReceipt:
        """Execute decision and return TradeReceipt including platform trade ID. Raises on failure."""
        res = self.execute(decision, account, adaptive_state=adaptive_state, now=now)
        if not res.executed or res.receipt is None:
            raise RuntimeError(f"Trade execution failed [{res.status}]: {res.reason}")
        return res.receipt

    def execute(
        self,
        decision: ValidatedDecision,
        account: AccountSnapshot,
        adaptive_state: Optional[AdaptiveState] = None,
        now: Optional[datetime] = None,
    ) -> ExecutionResult:
        """Execute validated decision strictly following PROJECT_SPEC Section 9 sequence."""
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

        # Record mode places NO orders at all
        if getattr(self.config, "mode", "paper") == "record":
            return ExecutionResult(
                executed=False,
                status="SKIPPED",
                reason="Record mode: order placement disabled",
                executed_at=now_iso,
            )

        decision_id = getattr(decision, "decision_id", None) or getattr(
            decision.raw_decision, "decision_id", None
        )

        # ── Step 1: Kill switch, Breaker, Reconcile checks ───────────────────
        if self.kill_switch.is_active():
            msg = "Trade blocked: Emergency kill switch is ACTIVE"
            self._send_alert_safe(f"🛑 {msg}")
            return ExecutionResult(executed=False, status="BLOCKED", reason=msg, executed_at=now_iso)

        if self.breaker.is_tripped():
            msg = f"Trade blocked: Circuit breaker is TRIPPED ({self.breaker.trip_reason})"
            self._send_alert_safe(f"⚡ {msg}")
            return ExecutionResult(executed=False, status="BLOCKED", reason=msg, executed_at=now_iso)

        if not self.reconcile_done:
            msg = "Trade blocked: Startup reconciliation has not completed"
            self._send_alert_safe(f"⚠️ {msg}")
            return ExecutionResult(executed=False, status="BLOCKED", reason=msg, executed_at=now_iso)

        # ── Step 2: Validator check ──────────────────────────────────────────
        if not decision.is_valid:
            msg = f"Trade blocked: Decision validation failed ({decision.skip_reason})"
            return ExecutionResult(executed=False, status="VALIDATOR_FAILED", reason=msg, executed_at=now_iso)

        # ── Mode check & Paper vs Real Routing ───────────────────────────────
        # Real orders only in "demo" or "live"; any other/unknown routes to paper
        mode_str = str(getattr(self.config, "mode", "paper")).lower()
        asset_mode = account.asset_modes.get(str(decision.asset), "paper")
        exec_cfg = getattr(self.config, "execution", None)
        demo_bypass = bool(
            exec_cfg
            and getattr(exec_cfg, "demo_bypasses_paper_gate", False)
            and mode_str == "demo"
        )
        is_real_candidate = (
            (mode_str in {"demo", "live"})
            and (asset_mode == "live_eligible" or demo_bypass)
            and not bool(decision.shadow_only)
        )

        # Missing adaptive config is treated as enabled (fail-closed)
        adaptive_cfg = getattr(self.config, "adaptive", None)
        adaptive_enabled = getattr(adaptive_cfg, "enabled", True) if adaptive_cfg is not None else True

        effective_adaptive_state = adaptive_state
        if not adaptive_enabled:
            # Explicitly disabled adaptive allows normal real orders via explicit NORMAL state
            effective_adaptive_state = adaptive_state or AdaptiveState()

        if is_real_candidate and effective_adaptive_state is not None and not effective_adaptive_state.real_trading_allowed:
            is_real_candidate = False

        # Alert failure threshold pause: pause real trading if alert failures >= max_alert_failures
        max_alert_fails = getattr(self.config.safety, "max_alert_failures", 5) if hasattr(self.config, "safety") else 5
        if is_real_candidate and self.consecutive_alert_failures >= max_alert_fails:
            logger.warning(
                "Alert delivery failures (%d >= %d); pausing real trading to paper shadow",
                self.consecutive_alert_failures,
                max_alert_fails,
            )
            is_real_candidate = False

        is_paper = not is_real_candidate

        # ── Step 3 & 4: Safety limits check ──────────────────────────────────
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
            adaptive_state=effective_adaptive_state,
        )
        if not safety_res.allowed:
            msg = f"Trade blocked by safety layer: {safety_res.reason}"
            if getattr(safety_res, "alert_required", False) and getattr(safety_res, "alert_text", ""):
                self._send_alert_safe(safety_res.alert_text)
            self._throttled_alert(f"🛡️ {msg}", asset=order.asset, now=now_dt)
            return ExecutionResult(executed=False, status="BLOCKED_SAFETY", reason=msg, executed_at=now_iso)

        if is_paper:
            return self._execute_paper_shadow(order, decision_id=decision_id, decision=decision, now=now_dt)

        # ── Step 5: Real trade placement with write-ahead logging ─────────────
        # Real orders REQUIRE a decision ID
        if not decision_id:
            msg = "Real order rejected: missing decision_id"
            return ExecutionResult(executed=False, status="BLOCKED", reason=msg, executed_at=now_iso)

        # Idempotency check: prevent duplicate execution of same decision ID
        if decision_id in self.executed_decision_ids:
            return ExecutionResult(
                executed=False,
                status="DUPLICATE",
                reason=f"Decision ID '{decision_id}' already executed",
                executed_at=now_iso,
            )
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
        self.state_manager.update(
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
            active_pending = [
                pt for pt in self.state_manager.data.get("pending_trades", [])
                if pt.get("id") != trade_id
            ]
            self.state_manager.update(
                pending_trades=active_pending,
                breaker_state=self.breaker.state,
            )
            self._send_alert_safe(f"⚡ {msg}")
            return ExecutionResult(executed=False, status="BREAKER_TRIPPED", reason=msg, executed_at=now_iso)

        # Snapshot open trades count matching order parameters before adapter call
        try:
            before_count = self._count_matching_open_trades(
                order.asset, order.direction, order.expiry_minutes, order.stake
            )
        except Exception as exc:
            err_msg = f"Failed to read open-trades snapshot before placement: {exc}"
            self.breaker.record_exec_error(err_msg, now=now_dt)
            active_pending = [
                pt for pt in self.state_manager.data.get("pending_trades", [])
                if pt.get("id") != trade_id
            ]
            self.state_manager.update(
                pending_trades=active_pending,
                breaker_state=self.breaker.state,
            )
            self._send_alert_safe(f"🚨 {err_msg}")
            return ExecutionResult(
                executed=False,
                status="BLOCKED",
                trade_id=trade_id,
                reason=err_msg,
                executed_at=now_iso,
            )

        # Call broker adapter to place trade (must raise if method is missing)
        placed_receipt: Any = None
        try:
            placed_receipt = self.adapter.place_trade(
                asset_id=order.asset,
                direction=TradeDirection(order.direction),
                expiry_min=order.expiry_minutes,
                stake_amount=order.stake,
                trade_id=trade_id,
            )
        except Exception as e:
            # Any ambiguous exception trips breaker immediately, keeps PENDING, resets reconcile_done
            err_msg = f"Ambiguous adapter failure: {e}"
            self.breaker.trip(err_msg, now=now_dt)
            self.reconcile_done = False
            self._send_alert_safe(f"🚨 CRITICAL EXECUTION FAULT: {err_msg}. PENDING trade {trade_id} retained; reconcile required.")
            return ExecutionResult(
                executed=False,
                status="EXEC_ERROR",
                trade_id=trade_id,
                reason=err_msg,
                executed_at=now_iso,
            )

        # ── Step 6: Post-placement verification via get_open_trades ───────────
        platform_id = getattr(placed_receipt, "platform_trade_id", None) or getattr(placed_receipt, "trade_id", None)
        verified = self._verify_trade_on_broker(
            expected_asset=order.asset,
            expected_direction=order.direction,
            expected_expiry=order.expiry_minutes,
            expected_stake=order.stake,
            trade_id=trade_id,
            platform_trade_id=str(platform_id) if platform_id else None,
            before_count=before_count,
        )

        if not verified:
            # Cannot be confirmed: STOP, ALERT, TRIP BREAKER, KEEP PENDING, NEVER RETRY!
            trip_msg = (
                f"Post-placement verification FAILED for trade {trade_id} on {order.asset}: "
                f"order was submitted but could not be confirmed in broker's open trades list"
            )
            self.breaker.trip(trip_msg, now=now_dt)
            self.reconcile_done = False
            self._send_alert_safe(f"🚨 CRITICAL EXECUTION FAULT: {trip_msg}")
            return ExecutionResult(
                executed=False,
                status="VERIFICATION_FAILED",
                trade_id=trade_id,
                reason=trip_msg,
                executed_at=now_iso,
            )

        # ── Step 7: Save to state and SQLite database, Alert ───────────────────
        self.breaker.record_exec_success()

        active_pending = [
            pt for pt in self.state_manager.data.get("pending_trades", [])
            if pt.get("id") != trade_id
        ]
        raw_payout = getattr(placed_receipt, "payout", None)
        if raw_payout is None or float(raw_payout) <= 0:
            err_msg = f"Real order {trade_id} receipt has missing or invalid payout: {raw_payout}"
            self.breaker.trip(err_msg, now=now_dt)
            self._send_alert_safe(f"🚨 CRITICAL: {err_msg}")
            return ExecutionResult(
                executed=False,
                status="EXEC_ERROR",
                trade_id=trade_id,
                reason=err_msg,
                executed_at=now_iso,
            )
        receipt_payout = float(raw_payout)

        open_record = {
            "id": trade_id,
            "platform_trade_id": str(platform_id) if platform_id else trade_id,
            "decision_id": decision_id,
            "asset": order.asset,
            "direction": order.direction,
            "expiry_minutes": order.expiry_minutes,
            "stake": order.stake,
            "payout": receipt_payout,
            "status": "OPEN",
            "opened_at": now_iso,
        }
        updated_open_trades = list(account.open_trades) + [open_record]

        self.state_manager.update(
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

        # Log real trade to TradeRepository; failed write raises alert and trips breaker
        if self.repository is not None:
            pattern_name = getattr(decision, "pattern_name", None)
            pattern_type = getattr(decision, "pattern_type", None)
            risk_score = getattr(decision, "risk_score", None)
            if risk_score is None and getattr(decision, "evidence", None):
                risk_score = getattr(decision.evidence, "risk_score", None)
            is_otc = getattr(decision, "is_otc", None)
            if is_otc is None:
                asset_str = str(order.asset)
                is_otc = "_otc" in asset_str.lower() or "otc" in asset_str.lower()

            trade_mode = str(getattr(self.config, "mode", "live")).lower()
            if trade_mode not in ("demo", "live"):
                trade_mode = "demo"

            try:
                record = TradeRecord(
                    id=trade_id,
                    opened_at=now_iso,
                    asset=order.asset,
                    is_otc=bool(is_otc),
                    direction=order.direction,
                    expiry_minutes=order.expiry_minutes,
                    stake=order.stake,
                    payout_at_entry=receipt_payout,
                    result="PENDING",
                    pnl=0.0,
                    mode=trade_mode,
                    pattern_name=pattern_name,
                    pattern_type=pattern_type,
                    risk_score=risk_score,
                    hour_utc=now_dt.hour,
                )
                self.repository.insert(record)
            except Exception as dbe:
                err_msg = f"Database write failure for real trade {trade_id}: {dbe}"
                logger.error(err_msg)
                self.breaker.trip(err_msg, now=now_dt)
                self._send_alert_safe(f"🚨 CRITICAL: {err_msg}")

        success_msg = f"Trade placed: {order.asset} {order.direction} ${order.stake:,.2f} ({order.expiry_minutes}m)"
        self._send_alert_safe(f"✅ {success_msg}")

        receipt = TradeReceipt(
            trade_id=trade_id,
            platform_trade_id=str(platform_id) if platform_id else trade_id,
            asset_id=order.asset,
            direction=TradeDirection(order.direction),
            expiry_minutes=order.expiry_minutes,
            stake_amount=order.stake,
            entry_price=float(getattr(placed_receipt, "entry_price", 0.0)),
            entry_time=now_dt,
            payout=receipt_payout,
            status="OPEN",
        )

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
            receipt=receipt,
        )

    def _execute_paper_shadow(
        self,
        order: OrderRequest,
        decision_id: Optional[str] = None,
        decision: Optional[ValidatedDecision] = None,
        now: Optional[datetime] = None,
    ) -> ExecutionResult:
        """Record paper shadow trade using paper_adapter.place_trade."""
        now_dt = now or datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        trade_id = f"paper_{uuid.uuid4().hex[:12]}"

        # Must raise if paper_adapter or place_trade is missing
        if self.paper_adapter is None:
            raise RuntimeError("paper_adapter is required for paper/shadow execution")

        receipt = self.paper_adapter.place_trade(
            asset_id=order.asset,
            direction=TradeDirection(order.direction),
            expiry_min=order.expiry_minutes,
            stake_amount=order.stake,
            trade_id=trade_id,
        )

        if decision_id:
            self.executed_decision_ids.add(decision_id)

        # Log shadow trade to TradeRepository if available
        if self.repository is not None:
            pattern_name = getattr(decision, "pattern_name", None)
            pattern_type = getattr(decision, "pattern_type", None)
            risk_score = getattr(decision, "risk_score", None)
            if risk_score is None and decision and getattr(decision, "evidence", None):
                risk_score = getattr(decision.evidence, "risk_score", None)
            is_otc = getattr(decision, "is_otc", None)
            if is_otc is None:
                asset_str = str(order.asset)
                is_otc = "_otc" in asset_str.lower() or "otc" in asset_str.lower()
            shadow_payout = float(getattr(receipt, "payout", None) or 0.85)

            try:
                record = TradeRecord(
                    id=trade_id,
                    opened_at=now_iso,
                    asset=order.asset,
                    is_otc=bool(is_otc),
                    direction=order.direction,
                    expiry_minutes=order.expiry_minutes,
                    stake=order.stake,
                    payout_at_entry=shadow_payout,
                    result="PENDING",
                    pnl=0.0,
                    mode="paper",
                    pattern_name=pattern_name,
                    pattern_type=pattern_type,
                    risk_score=risk_score,
                    hour_utc=now_dt.hour,
                )
                self.repository.insert(record)
            except Exception as dbe:
                logger.error("Database write failure for shadow trade %s: %s", trade_id, dbe)
                self._send_alert_safe(f"⚠️ Database write failure for shadow trade: {dbe}")

        msg = f"Shadow paper trade recorded: {order.asset} {order.direction} ${order.stake:,.2f} ({order.expiry_minutes}m)"
        self._send_alert_safe(f"📝 {msg}")

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
            receipt=receipt,
        )

    def record_settlement(
        self,
        trade_id: str,
        outcome: str,  # "WIN", "LOSS", "TIE", "VOID"
        payout: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Settle an open trade, update balance/equity in state, preserving adaptive/asset modes."""
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

        if target_trade is None:
            err_msg = f"Settlement failed: trade ID '{trade_id}' not found in open trades"
            self._send_alert_safe(f"🚨 {err_msg}")
            raise KeyError(err_msg)

        stake = float(target_trade.get("stake", 0.0))
        # Use payout stored in trade record at entry if available
        if "payout" in target_trade and target_trade["payout"] is not None:
            entry_payout = float(target_trade["payout"])
        elif payout is not None:
            entry_payout = float(payout)
        else:
            err_msg = f"Settlement failed for trade {trade_id}: no payout available in trade record or settlement call"
            self._send_alert_safe(f"🚨 {err_msg}")
            raise ValueError(err_msg)

        norm_outcome = str(outcome).upper().strip()
        if norm_outcome == "WIN":
            profit = stake * entry_payout
            new_balance = current_balance + stake + profit
            msg = f"Trade {trade_id} WON: +${profit:,.2f} profit (balance: ${new_balance:,.2f})"
            self._send_alert_safe(f"🏆 {msg}")
        elif norm_outcome == "LOSS":
            profit = -stake
            new_balance = current_balance  # stake was deducted at entry
            msg = f"Trade {trade_id} LOST: -${stake:,.2f} (balance: ${new_balance:,.2f})"
            self._send_alert_safe(f"📉 {msg}")
        elif norm_outcome == "TIE":
            tie_policy = getattr(self.config.trading, "tie_policy", "loss") if hasattr(self.config, "trading") else "loss"
            if tie_policy == "refund":
                profit = 0.0
                new_balance = current_balance + stake
                msg = f"Trade {trade_id} TIED (refunded): balance: ${new_balance:,.2f}"
            else:
                profit = -stake
                new_balance = current_balance
                msg = f"Trade {trade_id} TIED (treated as loss): balance: ${new_balance:,.2f}"
            self._send_alert_safe(f"🤝 {msg}")
        elif norm_outcome == "VOID":
            profit = 0.0
            new_balance = current_balance + stake
            msg = f"Trade {trade_id} VOIDED (refunded): balance: ${new_balance:,.2f}"
            self._send_alert_safe(f"🔄 {msg}")
        else:
            raise ValueError(f"Unknown settlement outcome: '{outcome}'. Expected WIN, LOSS, TIE, or VOID.")

        self.state_manager.update(
            balance=new_balance,
            equity=new_balance + sum(float(t.get("stake", 0.0)) for t in remaining_open),
            open_trades=remaining_open,
            pending_trades=self.state_manager.data.get("pending_trades", []),
            breaker_state=self.breaker.state,
            reconcile_done=self.reconcile_done,
        )

        if self.repository is not None:
            try:
                self.repository.update_result(
                    trade_id=str(trade_id),
                    closed_at=now_dt.isoformat(),
                    result=norm_outcome,
                    pnl=profit,
                )
            except Exception as dbe:
                err_msg = f"Database update failure on settlement for real trade {trade_id}: {dbe}"
                logger.error(err_msg)
                self.breaker.trip(err_msg, now=now_dt)
                self._send_alert_safe(f"🚨 CRITICAL: {err_msg}")

        return {
            "trade_id": trade_id,
            "outcome": norm_outcome,
            "balance": new_balance,
            "profit": profit,
        }

    def settle_shadow_trades(
        self,
        now: Optional[datetime] = None,
        prices_at_expiry: Optional[dict[str, float]] = None,
    ) -> list[Any]:
        """Settle open shadow trades on paper_adapter and update the TradeRepository."""
        if self.paper_adapter is None:
            return []
        now_dt = now or datetime.now(timezone.utc)
        settled = self.paper_adapter.settle_open_trades(
            current_time=now_dt,
            prices_at_expiry=prices_at_expiry,
        )
        if self.repository is not None:
            for trade in settled:
                tid = getattr(trade, "trade_id", getattr(trade, "id", None))
                res = getattr(trade, "result", None)
                res_str = res.value if hasattr(res, "value") else str(res)
                stake = float(getattr(trade, "stake_amount", getattr(trade, "stake", 0.0)))
                payout_rate = float(getattr(trade, "payout", 0.85) or 0.85)
                if res_str == "WIN":
                    pnl = stake * payout_rate
                elif res_str == "LOSS":
                    pnl = -stake
                elif res_str == "TIE":
                    tie_pol = getattr(self.paper_adapter, "tie_policy", "loss")
                    pnl = 0.0 if tie_pol == "refund" else -stake
                else:
                    pnl = 0.0
                closed_dt = getattr(trade, "exit_time", now_dt)
                try:
                    self.repository.update_result(
                        trade_id=str(tid),
                        closed_at=closed_dt,
                        result=res_str,
                        pnl=pnl,
                    )
                except Exception as exc:
                    logger.error("Failed to update repository for shadow trade %s: %s", tid, exc)
                    self._send_alert_safe(f"⚠️ Failed to update repository for shadow trade {tid}: {exc}")
        return settled
