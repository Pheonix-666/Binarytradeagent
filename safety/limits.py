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
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Any, Optional, Sequence

from adaptive.monitor import AdaptiveState
from core.config import AppConfig


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

    def __bool__(self) -> bool:
        return self.allowed


def Allow() -> OrderCheckResult:
    """Convenience factory for allowed check."""
    return OrderCheckResult(allowed=True, reason="ok")


def Block(reason: str) -> OrderCheckResult:
    """Convenience factory for blocked check."""
    return OrderCheckResult(allowed=False, reason=reason)


def check_order(
    order: Any,
    account: Any,
    config: AppConfig,
    adaptive_state: Optional[AdaptiveState] = None,
) -> OrderCheckResult:
    """Pure validation check for an order against all safety rules.

    Parameters:
      order: OrderRequest or object with asset, direction, expiry_minutes, stake.
      account: AccountSnapshot or object with balance, equity, open_trades, asset_modes.
      config: System AppConfig with safety, stake, and trading configurations.
      adaptive_state: Runtime AdaptiveState from adaptive regulator (if active).

    Returns:
      OrderCheckResult: Allow() or Block(reason).
    """
    asset = getattr(order, "asset", "")
    direction = str(getattr(order, "direction", "")).upper()
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
        if adaptive_state is not None:
            if not adaptive_state.real_trading_allowed or adaptive_state.tier == "PAUSED":
                return Block(
                    f"Real trading currently halted by adaptive regulator (tier={adaptive_state.tier})"
                )

    # 3. Asset Mode Check
    if not shadow_only:
        mode = asset_modes.get(asset, "paper")
        if mode != "live_eligible":
            return Block(
                f"Asset '{asset}' is in mode '{mode}'; real trading requires 'live_eligible'"
            )

    # 4. Expiry Check
    if expiry_minutes not in config.trading.allowed_expiries:
        return Block(
            f"Expiry {expiry_minutes}m not in allowed_expiries {config.trading.allowed_expiries}"
        )

    # 5. No Opposite Trades on Same Asset
    for ot in open_trades:
        ot_asset = getattr(ot, "asset", "")
        if isinstance(ot, dict):
            ot_asset = ot.get("asset", "")
        if ot_asset == asset:
            ot_dir = str(getattr(ot, "direction", getattr(ot, "get", lambda k: "")("direction"))).upper()
            if ot_dir and ot_dir != direction:
                return Block(
                    f"Opposite direction trade already open on {asset} ({ot_dir} vs {direction})"
                )

    # 5b. Per-Asset Exposure and Open Trades Limit
    max_open_per_asset = getattr(config.safety, "max_open_trades_per_asset", None) or getattr(config.trading, "max_trades_per_asset", None)
    max_asset_exp_pct = getattr(config.stake, "max_asset_exposure_pct", None)
    asset_stakes = 0.0
    asset_count = 0
    for ot in open_trades:
        ot_asset = ot.get("asset", "") if isinstance(ot, dict) else getattr(ot, "asset", "")
        if ot_asset == asset:
            asset_count += 1
            stk = ot.get("stake", ot.get("stake_amount", 0.0)) if isinstance(ot, dict) else getattr(ot, "stake", getattr(ot, "stake_amount", 0.0))
            asset_stakes += float(stk)

    if max_open_per_asset is not None and (asset_count + 1) > max_open_per_asset:
        return Block(f"Per-asset trade limit reached for {asset}: {asset_count + 1} > maximum {max_open_per_asset}")

    if max_asset_exp_pct is not None:
        asset_exp = ((asset_stakes + stake) / equity) * 100.0
        if asset_exp > max_asset_exp_pct + 1e-6:
            return Block(
                f"Per-asset exposure {asset_exp:.2f}% on {asset} exceeds limit {max_asset_exp_pct}%"
            )

    # 5c. Max Daily Trades Limit
    max_daily_trades = getattr(config.safety, "max_daily_trades", None) or getattr(config.trading, "max_daily_trades", None)
    today_trades = getattr(account, "today_trades", None)
    if max_daily_trades is not None and today_trades is not None and today_trades >= max_daily_trades:
        return Block(
            f"Daily trade count limit reached: {today_trades} >= maximum {max_daily_trades}"
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
    if round_to > 0:
        floored = math.floor(stake / round_to + 1e-9) * round_to
        if abs(stake - floored) > 0.001:
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

    allowed_ceiling = max(max_allowed_stake, min_stake_amt)
    if stake > allowed_ceiling + 1e-6:
        return Block(
            f"Stake {stake} exceeds tier-scaled maximum allowed {allowed_ceiling} "
            f"({tier_scaled_max_pct:.2f}% of equity {equity})"
        )

    # 9. Total Open Exposure Cap
    total_cap_pct = config.stake.total_cap_pct
    current_open_stakes = 0.0
    for ot in open_trades:
        stk = getattr(ot, "stake", getattr(ot, "stake_amount", 0.0))
        if isinstance(ot, dict):
            stk = ot.get("stake", ot.get("stake_amount", 0.0))
        current_open_stakes += float(stk)

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
