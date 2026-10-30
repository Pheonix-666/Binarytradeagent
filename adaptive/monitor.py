"""Adaptive monitoring state machine for system risk regulation.

PROJECT_SPEC.md Adaptive Layer (Phase 8A):
  - State machine: NORMAL -> CAUTION -> DEFENSIVE -> PAUSED.
  - Pure function update(equity, closed_trades, now, prev_state) -> AdaptiveState.
  - Escalation: instantaneous upon breaching tier thresholds.
  - Recovery: one tier at a time with hysteresis recovery fraction.
  - PAUSED persists until requalification passes (shadow trades >= min_shadow_trades
    beating break-even), followed by a ramp-up period at reduced stake scale.
  - Serializable: to_dict() and from_dict().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Sequence

from core.config import AdaptiveConfig, AppConfig
from data.store import TradeRecord
from selection.stats import break_even
from adaptive.decay import evaluate_edge_decay

TierName = Literal["NORMAL", "CAUTION", "DEFENSIVE", "PAUSED"]


@dataclass
class AdaptiveState:
    """Serializable runtime state of the adaptive monitor."""

    tier: TierName = "NORMAL"
    reference_equity: float = 10000.0
    current_equity: float = 10000.0
    drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    real_trading_allowed: bool = True
    stake_scale: float = 1.0
    min_similarity_add: float = 0.0
    max_risk_cap: Optional[float] = None
    reversal_mode_override: Optional[str] = None
    in_rampup: bool = False
    rampup_trades_remaining: int = 0
    shadow_trades_count: int = 0
    shadow_wins: int = 0
    transition_reason: str = "initial"
    last_update_time: Optional[str] = None
    history: list[dict[str, Any]] = field(default_factory=list)
    edge_decay_detected: bool = False
    edge_decay_persistent: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdaptiveState:
        """Construct state from dictionary."""
        d = dict(data)
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in field_names}
        return cls(**filtered)


def _compute_consecutive_losses(closed_trades: Sequence[Any]) -> int:
    """Calculate current consecutive losses from the end of the closed trades sequence.

    Only evaluates settled trades (WIN, LOSS, REFUND/VOID).
    LOSS increments streak; WIN resets to 0; VOID/REFUND leaves streak unchanged.
    """
    streak = 0
    for trade in reversed(closed_trades):
        res = getattr(trade, "result", None)
        if isinstance(trade, dict):
            res = trade.get("result")

        if not res:
            continue
        res_str = str(res).upper()
        if res_str == "LOSS":
            streak += 1
        elif res_str == "WIN":
            break
        elif res_str in ("REFUND", "VOID", "TIE"):
            continue
        else:
            # Pending or unhandled result
            continue
    return streak


def update_monitor(
    equity: float,
    closed_trades: Sequence[Any],
    now: datetime,
    prev_state: Optional[AdaptiveState],
    config: AppConfig,
    shadow_trades: Optional[Sequence[Any]] = None,
    active_profile_trades: Optional[Sequence[Any]] = None,
) -> AdaptiveState:
    """Pure state machine step for adaptive risk monitoring.

    Parameters:
      equity: Current account equity (cash + open stakes).
      closed_trades: All closed real trades up to now.
      now: Current evaluation timestamp.
      prev_state: Previous AdaptiveState (or None on initial step).
      config: System application configuration containing adaptive parameters.
      shadow_trades: Optional sequence of shadow trades observed during PAUSED.

    Returns:
      New AdaptiveState with updated tier, sizing, and threshold parameters.
    """
    ad_cfg: AdaptiveConfig = config.adaptive
    now_iso = now.isoformat() if hasattr(now, "isoformat") else str(now)

    if prev_state is None:
        prev_state = AdaptiveState(
            tier="NORMAL",
            reference_equity=equity,
            current_equity=equity,
            drawdown_pct=0.0,
            consecutive_losses=0,
            real_trading_allowed=True,
            stake_scale=1.0,
            history=[
                {
                    "timestamp": now_iso,
                    "from_tier": "INITIAL",
                    "to_tier": "NORMAL",
                    "reason": "Monitor initialized",
                    "drawdown_pct": 0.0,
                    "consecutive_losses": 0,
                }
            ],
        )

    # 1. Update reference equity based on reference mode
    ref_mode = ad_cfg.reference
    prev_ref = prev_state.reference_equity
    if ref_mode == "peak":
        ref_eq = max(prev_ref, equity)
    elif ref_mode == "session_start":
        # Retains session_start until reset externally
        ref_eq = prev_ref if prev_ref > 0.0 else equity
    elif ref_mode == "checkpoint":
        ref_eq = prev_ref if prev_ref > 0.0 else equity
    else:
        ref_eq = max(prev_ref, equity)

    # 2. Drawdown & Streak Calculation
    if ref_eq > 0.0 and equity < ref_eq:
        drawdown_pct = ((ref_eq - equity) / ref_eq) * 100.0
    else:
        drawdown_pct = 0.0

    consec_losses = _compute_consecutive_losses(closed_trades)

    # Carry forward shadow trade counts and ramp-up state
    shadow_count = prev_state.shadow_trades_count
    shadow_wins = prev_state.shadow_wins
    in_rampup = prev_state.in_rampup
    rampup_remaining = prev_state.rampup_trades_remaining

    # Decrement ramp-up if in ramp-up and new trades closed
    if in_rampup:
        # Check if new real trades closed since last update
        if len(closed_trades) > len(getattr(prev_state, "_trade_count", [])):
            trades_added = len(closed_trades) - len(getattr(prev_state, "_trade_count", []))
            rampup_remaining = max(0, rampup_remaining - trades_added)
            if rampup_remaining == 0:
                in_rampup = False

    # Process new shadow trades if currently PAUSED
    if prev_state.tier == "PAUSED" and shadow_trades is not None:
        # Update shadow metrics from supplied shadow trades
        shadow_count = len(shadow_trades)
        shadow_wins = sum(
            1 for t in shadow_trades
            if str(getattr(t, "result", getattr(t, "get", lambda k: "")("result"))).upper() == "WIN"
        )

    # 3. Tier Evaluation Logic
    tiers = ad_cfg.tiers
    hyst_frac = ad_cfg.hysteresis_recovery_fraction
    curr_tier: TierName = prev_state.tier
    new_tier: TierName = curr_tier
    reason: str = f"Maintained {curr_tier}"

    # Escalation thresholds
    pause_dd = tiers.paused.drawdown_pct
    defensive_dd = tiers.defensive.drawdown_pct
    caution_dd = tiers.caution.drawdown_pct
    caution_losses = tiers.caution.consecutive_losses

    # Check for direct escalation (fast reactions: drawdown & consecutive losses)
    if drawdown_pct >= (pause_dd - 1e-9):
        if curr_tier != "PAUSED":
            new_tier = "PAUSED"
            reason = f"Escalated to PAUSED: drawdown {drawdown_pct:.2f}% >= {pause_dd}%"
    elif drawdown_pct >= (defensive_dd - 1e-9):
        if curr_tier not in ("PAUSED", "DEFENSIVE"):
            new_tier = "DEFENSIVE"
            reason = f"Escalated to DEFENSIVE: drawdown {drawdown_pct:.2f}% >= {defensive_dd}%"
    elif (drawdown_pct >= (caution_dd - 1e-9)) or (consec_losses >= caution_losses):
        if curr_tier == "NORMAL":
            new_tier = "CAUTION"
            trigger_detail = []
            if drawdown_pct >= (caution_dd - 1e-9):
                trigger_detail.append(f"drawdown {drawdown_pct:.2f}% >= {caution_dd}%")
            if consec_losses >= caution_losses:
                trigger_detail.append(f"consecutive losses {consec_losses} >= {caution_losses}")
            reason = f"Escalated to CAUTION: {', '.join(trigger_detail)}"

    # 3b. Edge-Decay Escalation
    decay_trades = (
        active_profile_trades
        if active_profile_trades is not None
        else (list(closed_trades) + list(shadow_trades or []))
    )
    payout_default = getattr(config.backtest, "default_payout", 0.85)
    decay_res = evaluate_edge_decay(decay_trades, config=config, default_payout=payout_default)

    tier_order = {"NORMAL": 0, "CAUTION": 1, "DEFENSIVE": 2, "PAUSED": 3}
    if decay_res.recommended_tier:
        min_decay_tier = decay_res.recommended_tier
        if tier_order.get(min_decay_tier, 0) > tier_order.get(new_tier, 0):
            new_tier = min_decay_tier
            reason = f"Escalated to {new_tier}: {decay_res.reason}"

    # Check for recovery (slow reactions, one tier down at a time with hysteresis)
    if new_tier == curr_tier:
        if curr_tier == "PAUSED":
            # PAUSED persists until requalification passes
            req_cfg = ad_cfg.requalification
            payout = getattr(config.backtest, "default_payout", 0.85)
            be = break_even(payout)
            req_target = be + req_cfg.win_rate_margin

            if shadow_count >= req_cfg.min_shadow_trades:
                win_rate = shadow_wins / shadow_count if shadow_count > 0 else 0.0
                if win_rate >= req_target:
                    # Requalification passed! Move down one tier to DEFENSIVE with ramp-up
                    new_tier = "DEFENSIVE"
                    in_rampup = True
                    rampup_remaining = req_cfg.rampup_trades
                    reason = (
                        f"Requalification passed ({shadow_count} shadow trades, win rate {win_rate:.3f} >= {req_target:.3f}). "
                        f"Recovered to DEFENSIVE with {rampup_remaining} ramp-up trades"
                    )
        elif curr_tier == "DEFENSIVE":
            # Defensive recovery: drawdown must fall below defensive threshold * (1 - hysteresis)
            # AND edge decay is not demanding DEFENSIVE
            can_recover_defensive = (decay_res.recommended_tier not in ("DEFENSIVE", "PAUSED"))
            recovery_threshold = defensive_dd * (1.0 - hyst_frac)
            if can_recover_defensive and (drawdown_pct <= (recovery_threshold + 1e-9)):
                new_tier = "CAUTION"
                reason = (
                    f"Recovered to CAUTION: drawdown {drawdown_pct:.2f}% <= recovery threshold {recovery_threshold:.2f}%"
                )
        elif curr_tier == "CAUTION":
            # Caution recovery: drawdown must fall below caution threshold * (1 - hysteresis)
            # AND consecutive losses streak broken (< caution_losses)
            # AND edge decay is not active (recommended_tier is None)
            can_recover_caution = (decay_res.recommended_tier is None)
            recovery_threshold = caution_dd * (1.0 - hyst_frac)
            if can_recover_caution and (drawdown_pct <= (recovery_threshold + 1e-9)) and (consec_losses < caution_losses):
                new_tier = "NORMAL"
                reason = (
                    f"Recovered to NORMAL: drawdown {drawdown_pct:.2f}% <= {recovery_threshold:.2f}%, "
                    f"losses {consec_losses} < {caution_losses}, and edge intact"
                )

    # 4. Sizing and Threshold parameters by Tier
    real_trading = True
    stake_scale = 1.0
    sim_add = 0.0
    risk_cap = None
    rev_override = None

    if new_tier == "PAUSED":
        real_trading = False
        stake_scale = 0.0
        sim_add = tiers.defensive.min_similarity_add
        risk_cap = tiers.defensive.max_risk_cap
        rev_override = "off"
    elif new_tier == "DEFENSIVE":
        real_trading = True
        stake_scale = tiers.defensive.stake_scale
        sim_add = tiers.defensive.min_similarity_add
        risk_cap = tiers.defensive.max_risk_cap
        rev_override = tiers.defensive.reversal_mode
    elif new_tier == "CAUTION":
        real_trading = True
        stake_scale = tiers.caution.stake_scale
        sim_add = tiers.caution.min_similarity_add
        risk_cap = tiers.caution.max_risk_cap
        rev_override = None
    else:  # NORMAL
        real_trading = True
        stake_scale = 1.0
        sim_add = 0.0
        risk_cap = None
        rev_override = None

    # Apply ramp-up stake scale if in ramp-up
    if in_rampup and real_trading:
        rampup_scale = ad_cfg.requalification.rampup_stake_scale
        stake_scale = min(stake_scale, rampup_scale)

    history = list(prev_state.history)
    if new_tier != curr_tier:
        history.append({
            "timestamp": now_iso,
            "from_tier": curr_tier,
            "to_tier": new_tier,
            "reason": reason,
            "drawdown_pct": round(drawdown_pct, 2),
            "consecutive_losses": consec_losses,
        })

    next_state = AdaptiveState(
        tier=new_tier,
        reference_equity=round(ref_eq, 2),
        current_equity=round(equity, 2),
        drawdown_pct=round(drawdown_pct, 2),
        consecutive_losses=consec_losses,
        real_trading_allowed=real_trading,
        stake_scale=stake_scale,
        min_similarity_add=sim_add,
        max_risk_cap=risk_cap,
        reversal_mode_override=rev_override,
        in_rampup=in_rampup,
        rampup_trades_remaining=rampup_remaining,
        shadow_trades_count=shadow_count,
        shadow_wins=shadow_wins,
        transition_reason=reason,
        last_update_time=now_iso,
        history=history,
        edge_decay_detected=decay_res.decay_detected,
        edge_decay_persistent=decay_res.persistent,
    )
    # Store trade count for internal reference
    setattr(next_state, "_trade_count", list(closed_trades))
    return next_state


class AdaptiveController:
    """Controller wrapping stateful tracking for backtest and live execution."""

    def __init__(self, config: AppConfig, initial_state: Optional[AdaptiveState] = None):
        self.config = config
        self.state = initial_state or AdaptiveState(
            reference_equity=getattr(config.backtest, "starting_balance", 10000.0),
            current_equity=getattr(config.backtest, "starting_balance", 10000.0),
        )
        self.shadow_trades: list[Any] = []

    def set_checkpoint(self, equity: float) -> None:
        """Manually update reference checkpoint equity."""
        self.state.reference_equity = equity

    def record_shadow_trade(self, trade: Any) -> None:
        """Record shadow trade observed while paused."""
        self.shadow_trades.append(trade)

    def step(
        self,
        equity: float,
        closed_trades: Sequence[Any],
        now: datetime,
        active_profile_trades: Optional[Sequence[Any]] = None,
    ) -> AdaptiveState:
        """Advance controller state with latest account equity and closed trades."""
        self.state = update_monitor(
            equity=equity,
            closed_trades=closed_trades,
            now=now,
            prev_state=self.state,
            config=self.config,
            shadow_trades=self.shadow_trades,
            active_profile_trades=active_profile_trades,
        )
        return self.state

    def to_dict(self) -> dict[str, Any]:
        """Serialize controller state."""
        return {
            "state": self.state.to_dict(),
            "shadow_trades_count": len(self.shadow_trades),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: AppConfig) -> AdaptiveController:
        """Restore controller from serialized state."""
        state_data = data.get("state", {})
        state = AdaptiveState.from_dict(state_data)
        ctrl = cls(config=config, initial_state=state)
        return ctrl
