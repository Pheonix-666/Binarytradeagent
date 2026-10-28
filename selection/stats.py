"""Stats engine — performance metrics, Wilson confidence intervals, and paper gate logic.

Sections 6.4, 6.5, & 10:
  - Pure functions over trade lists.
  - Strict as_of timestamp filtering to eliminate look-ahead bias in backtests.
  - Separation by trading mode (paper and live never mixed).
  - Break-even rate: 1 / (1 + payout).
  - EV per trade: win_rate * payout - (1 - win_rate).
  - Realized return: total pnl / total stake.
  - Wilson confidence intervals (label "unproven" if CI includes break-even).
  - Paper gate: determines "paper" vs "live_eligible" status.
  - Pattern auto-disable when rolling win rate falls below break-even.
  - Session risk by hour of day (feeds Phase 4 RiskInput).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from core.config import AppConfig
from data.store import TradeRecord


@dataclass(frozen=True)
class AssetStats:
    """Performance statistics for a single asset."""

    asset: str
    mode: str
    n: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    mean_payout: float
    break_even: float
    ev: float
    realized_return: float
    wilson_ci: tuple[float, float]  # (lower, upper)
    edge_label: str  # "positive_edge", "negative_edge", "unproven"
    rolling_win_rate: float
    rolling_n: int


@dataclass(frozen=True)
class PatternStats:
    """Performance statistics for a chart pattern."""

    pattern_name: str
    pattern_type: str  # "continuation" or "reversal"
    mode: str
    n: int
    wins: int
    win_rate: float
    break_even: float
    rolling_win_rate: float
    is_disabled: bool


def break_even(payout: float) -> float:
    """Compute break-even win rate: 1 / (1 + payout)."""
    if payout <= 0.0 or math.isnan(payout):
        return 0.5
    return float(1.0 / (1.0 + payout))


def ev(win_rate: float, payout: float) -> float:
    """Compute Expected Value per trade: win_rate * payout - (1 - win_rate)."""
    return float(win_rate * payout - (1.0 - win_rate))


def shrunk_win_rate(
    wins: int,
    n: int,
    break_even_rate: float,
    k: float = 50.0,
) -> float:
    """Compute shrunk win rate pulled towards break-even rate.

    Formula:
      wr_adj = (wins + k * break_even_rate) / (n + k)
    """
    if n + k <= 0:
        return float(break_even_rate)
    return float((wins + k * break_even_rate) / (n + k))


def realized_return(trades: Sequence[TradeRecord]) -> float:
    """Compute realized return: total pnl / total stake."""
    total_stake = sum(t.stake for t in trades)
    if total_stake <= 0.0:
        return 0.0
    total_pnl = sum(t.pnl for t in trades)
    return float(total_pnl / total_stake)


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute 95% Wilson score confidence interval for a binomial proportion.

    Formula:
      center = (p + z^2 / (2n)) / (1 + z^2 / n)
      spread = z * sqrt(p*(1-p)/n + z^2 / (4n^2)) / (1 + z^2 / n)
    """
    if n <= 0:
        return (0.0, 1.0)

    p = float(wins) / float(n)
    z2 = z * z
    denom = 1.0 + (z2 / float(n))
    center = (p + (z2 / (2.0 * float(n)))) / denom
    spread = (z * math.sqrt((p * (1.0 - p) / float(n)) + (z2 / (4.0 * float(n) * float(n))))) / denom

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return (round(lower, 4), round(upper, 4))


def filter_trades_as_of(
    trades: Sequence[TradeRecord],
    as_of: Optional[Union[str, datetime]] = None,
) -> list[TradeRecord]:
    """Filter trades to only those closed at or before as_of (prevents look-ahead)."""
    if as_of is None:
        return [t for t in trades if t.closed_at is not None]

    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            as_of_dt = as_of.replace(tzinfo=timezone.utc)
        else:
            as_of_dt = as_of.astimezone(timezone.utc)
        as_of_str = as_of_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        as_of_str = str(as_of)

    filtered = []
    for t in trades:
        if t.closed_at is not None and t.closed_at <= as_of_str:
            filtered.append(t)
    return filtered


def compute_asset_stats(
    trades: Sequence[TradeRecord],
    asset: str,
    mode: Union[str, Sequence[str]] = "paper",
    as_of: Optional[Union[str, datetime]] = None,
    rolling_n: int = 100,
) -> AssetStats:
    """Compute performance stats for an asset using historical trades."""
    if isinstance(mode, str):
        allowed_modes = {mode}
        mode_label = mode
    else:
        allowed_modes = set(mode)
        mode_label = "+".join(sorted(allowed_modes))

    asset_trades = [t for t in trades if t.asset == asset and t.mode in allowed_modes]
    settled = filter_trades_as_of(asset_trades, as_of)

    n = len(settled)
    if n == 0:
        return AssetStats(
            asset=asset,
            mode=mode_label,
            n=0,
            wins=0,
            losses=0,
            ties=0,
            win_rate=0.0,
            mean_payout=0.85,
            break_even=break_even(0.85),
            ev=ev(0.0, 0.85),
            realized_return=0.0,
            wilson_ci=(0.0, 1.0),
            edge_label="unproven",
            rolling_win_rate=0.0,
            rolling_n=rolling_n,
        )

    wins = sum(1 for t in settled if t.result == "WIN")
    losses = sum(1 for t in settled if t.result == "LOSS")
    ties = sum(1 for t in settled if t.result == "TIE")
    win_rate = float(wins) / float(n)

    mean_payout = float(np.mean([t.payout_at_entry for t in settled]))
    be = break_even(mean_payout)
    ev_val = ev(win_rate, mean_payout)
    ret = realized_return(settled)
    ci = wilson_interval(wins, n)

    # Label "unproven" if confidence interval includes break-even
    if ci[0] <= be <= ci[1]:
        edge_label = "unproven"
    elif ci[0] > be:
        edge_label = "positive_edge"
    else:
        edge_label = "negative_edge"

    # Rolling win rate over the last rolling_n trades
    rolling_window = settled[-rolling_n:] if len(settled) > rolling_n else settled
    rolling_wins = sum(1 for t in rolling_window if t.result == "WIN")
    rolling_win_rate = float(rolling_wins) / float(len(rolling_window))

    return AssetStats(
        asset=asset,
        mode=mode_label,
        n=n,
        wins=wins,
        losses=losses,
        ties=ties,
        win_rate=round(win_rate, 4),
        mean_payout=round(mean_payout, 4),
        break_even=round(be, 4),
        ev=round(ev_val, 4),
        realized_return=round(ret, 4),
        wilson_ci=ci,
        edge_label=edge_label,
        rolling_win_rate=round(rolling_win_rate, 4),
        rolling_n=rolling_n,
    )


def hour_risk(
    trades: Sequence[TradeRecord],
    hour: int,
    session_min_samples: int = 20,
    mode: Union[str, Sequence[str]] = "paper",
    as_of: Optional[Union[str, datetime]] = None,
) -> Optional[float]:
    """Compute 0-100 session risk for a given hour bucket (UTC 0-23).

    Returns None if fewer than session_min_samples trades exist for that hour.
    Higher score = riskier (lower win rate).
    """
    if isinstance(mode, str):
        allowed_modes = {mode}
    else:
        allowed_modes = set(mode)

    mode_trades = [t for t in trades if t.mode in allowed_modes and t.hour_utc == hour]
    settled = filter_trades_as_of(mode_trades, as_of)

    if len(settled) < session_min_samples:
        return None

    wins = sum(1 for t in settled if t.result == "WIN")
    win_rate = float(wins) / float(len(settled))
    mean_payout = float(np.mean([t.payout_at_entry for t in settled]))
    be = break_even(mean_payout)

    # Risk score formula from Section 6.5
    val = (be + 0.15 - win_rate) / 0.20
    score = 100.0 * float(np.clip(val, 0.01, 1.0))
    return float(np.clip(score, 1.0, 100.0))


def compute_pattern_stats(
    trades: Sequence[TradeRecord],
    pattern_name: str,
    pattern_type: str = "continuation",
    mode: str = "paper",
    as_of: Optional[Union[str, datetime]] = None,
    rolling_n: int = 100,
) -> PatternStats:
    """Compute win rate and auto-disable status for a chart pattern.

    Rule: A reversal pattern whose rolling win rate is below break-even is auto-disabled.
    """
    pattern_trades = [
        t for t in trades if t.pattern_name == pattern_name and t.mode == mode
    ]
    settled = filter_trades_as_of(pattern_trades, as_of)
    n = len(settled)

    if n == 0:
        return PatternStats(
            pattern_name=pattern_name,
            pattern_type=pattern_type,
            mode=mode,
            n=0,
            wins=0,
            win_rate=0.0,
            break_even=break_even(0.85),
            rolling_win_rate=0.0,
            is_disabled=False,
        )

    wins = sum(1 for t in settled if t.result == "WIN")
    win_rate = float(wins) / float(n)
    mean_payout = float(np.mean([t.payout_at_entry for t in settled]))
    be = break_even(mean_payout)

    rolling_window = settled[-rolling_n:] if len(settled) > rolling_n else settled
    rolling_wins = sum(1 for t in rolling_window if t.result == "WIN")
    rolling_win_rate = float(rolling_wins) / float(len(rolling_window))

    # Auto-disable rule: reversal pattern below break-even
    is_disabled = False
    if pattern_type.lower() == "reversal" and len(rolling_window) >= 10:
        if rolling_win_rate < be:
            is_disabled = True

    return PatternStats(
        pattern_name=pattern_name,
        pattern_type=pattern_type,
        mode=mode,
        n=n,
        wins=wins,
        win_rate=round(win_rate, 4),
        break_even=round(be, 4),
        rolling_win_rate=round(rolling_win_rate, 4),
        is_disabled=is_disabled,
    )


def asset_mode(
    stats: AssetStats,
    config: AppConfig,
) -> tuple[str, str]:
    """Evaluate whether an asset qualifies for live trading (Section 10 Paper Gate).

    Returns: ("paper" | "live_eligible", reason_description)

    Standard requirements for live_eligible:
      1. paper_trades >= paper_gate.min_paper_trades (default 200)
      2. win_rate > break_even + min_edge_margin (default +0.02)
      3. rolling_win_rate >= break_even (drops back to paper if below)

    Optional stricter requirement:
      If paper_gate.require_ci_lower_bound is True, also requires:
      4. wilson_ci[0] > break_even (Wilson interval lower bound > break_even)
    """
    if stats.mode != "paper":
        return stats.mode, f"Asset is already in {stats.mode} mode"

    min_trades = config.paper_gate.min_paper_trades
    min_margin = config.paper_gate.min_edge_margin
    required_win_rate = stats.break_even + min_margin

    ci_lower = stats.wilson_ci[0]
    ci_lower_met = ci_lower > stats.break_even
    ci_info = f"[CI lower bound: {ci_lower:.4f} > BE {stats.break_even:.4f}: {ci_lower_met}]"

    standard_failures = []
    if stats.n < min_trades:
        standard_failures.append(f"Insufficient paper trades: {stats.n} < {min_trades}")

    if stats.win_rate <= required_win_rate:
        standard_failures.append(
            f"Win rate ({stats.win_rate:.4f}) does not exceed break-even + margin "
            f"({stats.break_even:.4f} + {min_margin:.4f} = {required_win_rate:.4f})"
        )

    if stats.rolling_win_rate < stats.break_even:
        standard_failures.append(
            f"Rolling win rate ({stats.rolling_win_rate:.4f}) dropped below "
            f"break-even ({stats.break_even:.4f})"
        )

    if standard_failures:
        reason = f"{'; '.join(standard_failures)}. {ci_info}"
        return "paper", reason

    # Standard requirements passed; check optional stricter CI-lower-bound requirement
    if config.paper_gate.require_ci_lower_bound and not ci_lower_met:
        reason = (
            f"Standard gate passed, but CI lower bound condition failed: "
            f"CI lower {ci_lower:.4f} <= break-even {stats.break_even:.4f}. {ci_info}"
        )
        return "paper", reason

    reason = (
        f"Passed paper gate: {stats.n} trades, win rate {stats.win_rate:.4f} > "
        f"{required_win_rate:.4f}, rolling win rate {stats.rolling_win_rate:.4f} >= {stats.break_even:.4f}. {ci_info}"
    )
    return "live_eligible", reason
