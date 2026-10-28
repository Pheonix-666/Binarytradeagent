"""Support and resistance levels — swing detection and proximity calculation.

Strict Rule:
    No look-ahead. A swing at candle i with right window R is only identified
    using data up to i + R. When evaluating at candle N, only i <= N - R are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd

from core.models import TradeDirection


@dataclass
class Level:
    """Identified price level (support or resistance)."""

    price: float
    level_type: Literal["support", "resistance"]
    touches: int = 1
    candle_index: int = 0


def find_swing_levels(
    df: pd.DataFrame,
    left_bars: int = 5,
    right_bars: int = 2,
    cluster_tolerance_atr: float = 0.5,
    atr: Optional[float] = None,
) -> list[Level]:
    """Identify confirmed swing high (resistance) and swing low (support) levels.

    Strictly no look-ahead: at index N, only swings at i <= N - right_bars
    are confirmed.
    """
    n = len(df)
    min_required = left_bars + right_bars + 1
    if n < min_required:
        return []

    highs = df["high"].values
    lows = df["low"].values

    raw_levels: list[Level] = []

    # Swing high: highs[i] > highs[i-left .. i-1] and highs[i] >= highs[i+1 .. i+right]
    # Swing low: lows[i] < lows[i-left .. i-1] and lows[i] <= lows[i+1 .. i+right]
    for i in range(left_bars, n - right_bars):
        # Check swing high
        is_swing_high = True
        for l in range(1, left_bars + 1):
            if highs[i] <= highs[i - l]:
                is_swing_high = False
                break
        if is_swing_high:
            for r in range(1, right_bars + 1):
                if highs[i] < highs[i + r]:
                    is_swing_high = False
                    break
        if is_swing_high:
            raw_levels.append(
                Level(
                    price=float(highs[i]),
                    level_type="resistance",
                    touches=1,
                    candle_index=i,
                )
            )

        # Check swing low
        is_swing_low = True
        for l in range(1, left_bars + 1):
            if lows[i] >= lows[i - l]:
                is_swing_low = False
                break
        if is_swing_low:
            for r in range(1, right_bars + 1):
                if lows[i] > lows[i + r]:
                    is_swing_low = False
                    break
        if is_swing_low:
            raw_levels.append(
                Level(
                    price=float(lows[i]),
                    level_type="support",
                    touches=1,
                    candle_index=i,
                )
            )

    if not raw_levels:
        return []

    # Cluster nearby levels if atr is provided
    if atr is not None and atr > 0:
        tol = cluster_tolerance_atr * atr
        return _cluster_levels(raw_levels, tol)

    return raw_levels


def _cluster_levels(levels: list[Level], tolerance: float) -> list[Level]:
    """Group levels of the same type within tolerance into clustered levels."""
    clustered: list[Level] = []
    # Separate by type
    for ltype in ["support", "resistance"]:
        subset = [lvl for lvl in levels if lvl.level_type == ltype]
        if not subset:
            continue
        subset.sort(key=lambda x: x.price)

        current_cluster = [subset[0]]
        for lvl in subset[1:]:
            if lvl.price - current_cluster[-1].price <= tolerance:
                current_cluster.append(lvl)
            else:
                avg_price = sum(x.price for x in current_cluster) / len(current_cluster)
                clustered.append(
                    Level(
                        price=avg_price,
                        level_type=ltype,  # type: ignore[arg-type]
                        touches=len(current_cluster),
                        candle_index=current_cluster[-1].candle_index,
                    )
                )
                current_cluster = [lvl]

        if current_cluster:
            avg_price = sum(x.price for x in current_cluster) / len(current_cluster)
            clustered.append(
                Level(
                    price=avg_price,
                    level_type=ltype,  # type: ignore[arg-type]
                    touches=len(current_cluster),
                    candle_index=current_cluster[-1].candle_index,
                )
            )

    return clustered


def get_nearest_level(
    current_price: float,
    levels: list[Level],
    target_type: Optional[Literal["support", "resistance"]] = None,
    atr: float = 1.0,
    at_level_atr_threshold: float = 0.5,
) -> dict:
    """Find the nearest level of target_type (or nearest of any type if None).

    Returns dict:
      {
        'type': 'support' | 'resistance',
        'price': float,
        'distance_atr': float,
        'is_at_level': bool,
        'touches': int,
      }
    """
    candidates = levels
    if target_type is not None:
        candidates = [lvl for lvl in levels if lvl.level_type == target_type]

    if not candidates:
        return {
            "type": target_type or "none",
            "price": current_price,
            "distance_atr": 999.0,
            "is_at_level": False,
            "touches": 0,
        }

    # Find candidate with minimum distance in price
    atr_safe = atr if atr > 0 else 1.0
    best_level = min(candidates, key=lambda lvl: abs(current_price - lvl.price))
    dist_price = abs(current_price - best_level.price)
    dist_atr = dist_price / atr_safe

    return {
        "type": best_level.level_type,
        "price": best_level.price,
        "distance_atr": round(dist_atr, 2),
        "is_at_level": dist_atr <= at_level_atr_threshold,
        "touches": best_level.touches,
    }


def get_nearest_opposing_level(
    current_price: float,
    trade_direction: TradeDirection,
    levels: list[Level],
    atr: float = 1.0,
) -> dict:
    """Find nearest opposing level in the direction of the trade.

    For CALL: opposing level is resistance above current_price.
    For PUT: opposing level is support below current_price.
    """
    atr_safe = atr if atr > 0 else 1.0
    if trade_direction == TradeDirection.CALL:
        opposing = [lvl for lvl in levels if lvl.level_type == "resistance" and lvl.price >= current_price]
        opposing_type = "resistance"
    else:
        opposing = [lvl for lvl in levels if lvl.level_type == "support" and lvl.price <= current_price]
        opposing_type = "support"

    if not opposing:
        return {
            "type": opposing_type,
            "price": current_price,
            "distance_atr": 999.0,
            "is_at_level": False,
            "touches": 0,
        }

    best = min(opposing, key=lambda lvl: abs(current_price - lvl.price))
    dist_atr = abs(current_price - best.price) / atr_safe
    return {
        "type": best.level_type,
        "price": best.price,
        "distance_atr": round(dist_atr, 2),
        "is_at_level": dist_atr <= 0.5,
        "touches": best.touches,
    }
