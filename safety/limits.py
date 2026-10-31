"""Safety limits — pure validation engine for order execution.

PROJECT_SPEC.md Section 8 (Safety Layer):
  - Stake limits: each stake between min_pct and max_pct of equity,
    tier-scaled max, capped at platform max_amount, floored to round_to.
  - Total open stakes at most total_cap_pct.
  - No opposite trades on the same asset at once.
  - Real trading halted if adaptive state real_trading_allowed is False.
  - Asset mode check: real money only for live_eligible assets.
  - Expiry minutes must belong to allowed_expiries.
  - Daily loss limit enforcement.
  - Mode guard: live mode requires LIVE_CONFIRM=YES.
  - Zero hardcoded limits, pure function, no disk or network I/O.
  - adaptive_state is a required parameter (no default). Real orders
    must always supply an explicit state; None blocks the order.
    Shadow/paper orders may pass None (treated as NORMAL).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import os
from typing import Any, Optional, Sequence

from adaptive.monitor import AdaptiveState
from core.config import AppConfig
from core.models import to_canonical_trade

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderRequest:
    """Standard order payload submitted for pre-execution safety verification."""

    asset: str
    direction: str  # "CALL" or "PUT"
    expiry_minutes: int
    stake: float
    max_stake_pct: Optional[float] = None
    shadow_only: bool = False
    decision_id: Optional[str] = None


@dataclass
class AccountSnapshot:
    """Account financial and portfolio state at order evaluation time."""

    balance: float
    equity: float
    open_trades: list[Any] = field(default_factory=list)
    asset_modes: dict[str, str] = field(default_factory=dict)
    daily_starting_balance: Optional[float] = None
    today_loss: float = 0.0


@dataclass(frozen=True)
class OrderCheckResult:
    """Outcome of order safety verification."""

    allowed: bool
    reason: str = "ok"
    alert_required: bool = False
    alert_text: str = ""

    def __bool__(self) -> bool:
        return self.allowed


def Allow() -> OrderCheckResult:
    """Convenience factory for allowed check."""
    return OrderCheckResult(allowed=True, reason="ok")


def Block(
    reason: str,
    alert_required: bool = False,
    alert_text: str = "",
) -> OrderCheckResult:
    """Convenience factory for blocked check."""
    return OrderCheckResult(
        allowed=False,
        reason=reason,
        alert_required=alert_required,
        alert_text=alert_text,
    )


def check_order(
    order: Any,
    account: Any,
    config: AppConfig,
    adaptive_state: Optional[AdaptiveState],
) -> OrderCheckResult:
    """Pure validation check for an order against all safety rules.

    Parameters:
      order: OrderRequest or object with asset, direction, expiry_minutes, stake.
      account: AccountSnapshot or object with balance, equity, open_trades, asset_modes.
      config: System AppConfig with safety, stake, and trading configurations.
      adaptive_state: REQUIRED — no default; callers must always pass explicitly.
          For REAL orders: supply the current AdaptiveState.  Passing None for a
          real order immediately returns Block("adaptive state missing").
          For SHADOW/PAPER orders: None is accepted and treated as NORMAL
          (stake_scale=1.0, real_trading_allowed=True).
          When adaptive is DISABLED in config: pass AdaptiveState() which uses
          the safe NORMAL defaults (tier="NORMAL", real_trading_allowed=True,
          stake_scale=1.0).

    Returns:
      OrderCheckResult: Allow() or Block(reason, alert_required, alert_text).
    """
    asset = getattr(order, "asset", "")
    direction = str(getattr(order, "direction", "")).upper()
    if direction not in ("CALL", "PUT"):
        return Block(f"Invalid direction '{direction}'; must be 'CALL' or 'PUT'")
    expiry_minutes = int(getattr(order, "expiry_minutes", 0))
    stake = float(getattr(order, "stake", getattr(order, "stake_amount", 0.0)))
    cand_max_pct = getattr(order, "max_stake_pct", None)
    shadow_only = bool(getattr(order, "shadow_only", False))

    equity = float(getattr(account, "equity", getattr(account, "balance", 0.0)))
    if equity <= 0.0:
        equity = float(getattr(account, "balance", 0.0))
    if equity <= 0.0:
        return Block("Account equity/balance is zero or negative")

    open_trades = list(getattr(account, "open_trades", []))
    asset_modes = getattr(account, "asset_modes", {})

    # 1. Mode Guard
    if config.mode == "live" and not shadow_only:
        env_var = config.live_requires_env
        if os.environ.get(env_var) != "YES":
            return Block(f"Mode is 'live' but {env_var} is not 'YES'")

    # 2. Real Trading Allowed by Adaptive Regulator
    if not shadow_only:
        # Real orders require an explicit adaptive state — None is a programming error.
        if adaptive_state is None:
            return Block(
                "adaptive state missing: real orders require an explicit AdaptiveState "
                "(pass AdaptiveState() when adaptive is disabled)",
                alert_required=True,
                alert_text="🚨 check_order called for a real order without adaptive state",
            )
        if not adaptive_state.real_trading_allowed or adaptive_state.tier == "PAUSED":
            return Block(
                f"Real trading currently halted by adaptive regulator (tier={adaptive_state.tier})"
            )

    # 3. Asset Mode Check
    if not shadow_only:
        mode = asset_modes.get(asset, "paper")
        exec_cfg = getattr(config, "execution", None)
        demo_bypass = bool(
            exec_cfg
            and getattr(exec_cfg, "demo_bypasses_paper_gate", False)
            and getattr(config, "mode", "paper") == "demo"
        )
        if mode != "live_eligible" and not demo_bypass:
            return Block(
                f"Asset '{asset}' is in mode '{mode}'; real trading requires 'live_eligible'"
            )

    # 4. Expiry Check
    if expiry_minutes not in config.trading.allowed_expiries:
        return Block(
            f"Expiry {expiry_minutes}m not in allowed_expiries {config.trading.allowed_expiries}"
        )

    # 5. Open Trades Validation: Opposite Trades & Missing/Invalid Stake Fail Closed
    current_open_stakes = 0.0
    for ot in open_trades:
        c_ot = to_canonical_trade(ot)
        # Missing or unparseable stake check
        if math.isnan(c_ot.stake) or c_ot.stake < 0:
            if not shadow_only:
                return Block(
                    f"open trade with missing or invalid stake on {c_ot.asset or 'unknown'}",
                    alert_required=True,
                    alert_text=f"🚨 Open trade with missing or invalid stake on {c_ot.asset or 'unknown'}",
                )
            stk_to_add = 0.0
        else:
            stk_to_add = c_ot.stake
        current_open_stakes += stk_to_add

        if c_ot.asset == asset:
            if c_ot.direction in ("CALL", "PUT"):
                if c_ot.direction != direction:
                    return Block(
                        f"Opposite direction trade already open on {asset} ({c_ot.direction} vs {direction})"
                    )
            else:
                # Open trade on same asset has missing or unparseable direction -> fail closed for real orders
                if not shadow_only:
                    return Block(
                        f"open trade with unknown direction on {asset}",
                        alert_required=True,
                        alert_text=f"🚨 Open trade with unknown direction on {asset}",
                    )

    # 6. Absolute Stake Limits
    min_amount = config.stake.min_amount
    if stake < min_amount - 1e-6:
        return Block(f"Stake {stake} < minimum allowed amount {min_amount}")

    max_amount = config.stake.max_amount
    if max_amount is not None and stake > max_amount + 1e-6:
        return Block(f"Stake {stake} > maximum allowed amount {max_amount}")

    # 7. Rounding Check
    round_to = config.stake.round_to
    round_tol = getattr(config.safety, "rounding_tolerance", 0.001) if hasattr(config, "safety") else 0.001
    if round_to > 0:
        floored = math.floor(stake / round_to + 1e-9) * round_to
        if abs(stake - floored) > round_tol:
            return Block(
                f"Stake {stake} is not floored to round_to {round_to} (expected {floored})"
            )

    # 8. Stake % of Equity Limits
    min_pct = config.stake.min_pct
    max_pct = config.stake.max_pct

    min_stake_amt = math.floor((equity * (min_pct / 100.0)) / round_to) * round_to
    if min_stake_amt < min_amount:
        min_stake_amt = min_amount

    if stake < min_stake_amt - 1e-6:
        return Block(
            f"Stake {stake} < min_pct {min_pct}% of equity {equity} ({min_stake_amt})"
        )

    stake_scale = adaptive_state.stake_scale if adaptive_state is not None else 1.0
    effective_max_pct = max_pct
    if cand_max_pct is not None:
        effective_max_pct = min(effective_max_pct, float(cand_max_pct))
    tier_scaled_max_pct = effective_max_pct * stake_scale

    max_allowed_stake = math.floor((equity * (tier_scaled_max_pct / 100.0)) / round_to) * round_to
    if max_amount is not None:
        max_allowed_stake = min(max_allowed_stake, max_amount)

    # If the tier-scaled maximum is below the minimum stake, BLOCK (do not raise ceiling)
    if max_allowed_stake < min_stake_amt:
        return Block(
            f"Tier-scaled maximum allowed {max_allowed_stake} ({tier_scaled_max_pct:.2f}% of equity {equity}) "
            f"is below minimum stake {min_stake_amt}"
        )

    if stake > max_allowed_stake + 1e-6:
        return Block(
            f"Stake {stake} exceeds tier-scaled maximum allowed {max_allowed_stake} "
            f"({tier_scaled_max_pct:.2f}% of equity {equity})"
        )

    # 9. Total Open Exposure Cap
    total_cap_pct = config.stake.total_cap_pct
    new_total_stakes = current_open_stakes + stake
    exposure_pct = (new_total_stakes / equity) * 100.0
    if exposure_pct > total_cap_pct + 1e-6:
        return Block(
            f"Total open exposure {exposure_pct:.2f}% (${new_total_stakes:.2f}) "
            f"exceeds total_cap_pct {total_cap_pct}% (${equity * total_cap_pct / 100.0:.2f})"
        )

    # 10. Daily Loss Limit
    daily_limit_pct = config.stake.daily_loss_limit_pct
    if daily_limit_pct is not None:
        daily_ref = getattr(account, "daily_starting_balance", None)
        if daily_ref is None or daily_ref <= 0.0:
            daily_ref = equity
        max_daily_loss = daily_ref * (daily_limit_pct / 100.0)
        today_loss = float(getattr(account, "today_loss", 0.0))
        if today_loss >= max_daily_loss - 1e-6:
            return Block(
                f"Daily loss limit reached: today's loss ${today_loss:.2f} >= "
                f"maximum ${max_daily_loss:.2f} ({daily_limit_pct}% of ${daily_ref:.2f})"
            )

    return Allow()
