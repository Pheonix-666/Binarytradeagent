"""Asset selector — top-asset ranking by Tier 1 (EV) & Tier 2 (Payout), and correlation mapping.

Sections 6.4 & 10:
  - Runs every asset_reselect_hours.
  - Excludes closed assets (is_open == False).
  - Tier 1: Assets with >= min_trades_for_stats trades, ranked by EV.
  - Tier 2: Assets without enough stats, ranked by live payout.
  - Slot allocation: up to max_measured_slots from Tier 1, remainder from Tier 2.
    If Tier 1 has fewer candidates, Tier 2 fills more up to top_assets.
  - Deterministic tie-breaks documented in sorting keys.
  - Real-market return correlation map (OTC assets strictly excluded).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from core.config import AppConfig
from core.models import AssetQuote
from selection.stats import AssetStats, break_even, ev


@dataclass(frozen=True)
class SelectedAsset:
    """A selected asset ranked for trading consideration."""

    asset_id: str
    tier: int  # 1 (measured) or 2 (new/untested)
    rank: int  # 1-based overall rank
    reason: str
    payout: float
    ev: Optional[float]
    win_rate: Optional[float]
    trades_count: int
    is_otc: bool


def select_top_assets(
    quotes: Mapping[str, AssetQuote],
    stats_by_asset: Mapping[str, AssetStats],
    config: AppConfig,
) -> list[SelectedAsset]:
    """Select and rank top assets according to two-tier allocation rules.

    Tier 1: >= min_trades_for_stats trades, ranked by EV descending.
            Deterministic tie-breaks: (-ev, -win_rate, -payout, asset_id).
    Tier 2: < min_trades_for_stats trades (or no stats), ranked by payout descending.
            Deterministic tie-breaks: (-payout, -trades_count, asset_id).

    Slot allocation:
      - Takes up to config.trading.max_measured_slots from Tier 1.
      - Fills remaining slots up to config.trading.top_assets from Tier 2.
      - If Tier 1 has fewer than max_measured_slots, Tier 2 fills the deficit.
      - Excludes any asset with is_open == False.
    """
    min_trades = config.trading.min_trades_for_stats
    top_limit = config.trading.top_assets
    max_t1_slots = config.trading.max_measured_slots

    tier1_candidates: list[dict] = []
    tier2_candidates: list[dict] = []

    for asset_id, quote in quotes.items():
        # Exclude closed assets
        if not quote.is_open:
            continue

        stats = stats_by_asset.get(asset_id)
        n_trades = stats.n if stats else 0
        payout = quote.payout

        if stats and n_trades >= min_trades:
            tier1_candidates.append(
                {
                    "asset_id": asset_id,
                    "tier": 1,
                    "ev": stats.ev,
                    "win_rate": stats.win_rate,
                    "payout": payout,
                    "trades_count": n_trades,
                    "is_otc": quote.is_otc,
                    "reason": f"Tier 1: EV={stats.ev:+.4f}, WinRate={stats.win_rate:.1%}, n={n_trades}",
                }
            )
        else:
            calc_be = break_even(payout)
            tier2_candidates.append(
                {
                    "asset_id": asset_id,
                    "tier": 2,
                    "ev": stats.ev if stats else None,
                    "win_rate": stats.win_rate if stats else None,
                    "payout": payout,
                    "trades_count": n_trades,
                    "is_otc": quote.is_otc,
                    "reason": f"Tier 2: Payout={payout:.1%}, BE={calc_be:.1%}, n={n_trades}",
                }
            )

    # Deterministic sorting
    # Tier 1: highest EV first, then highest win rate, highest payout, alphabetical asset_id
    tier1_candidates.sort(
        key=lambda x: (
            -float(x["ev"] if x["ev"] is not None else -999.0),
            -float(x["win_rate"] if x["win_rate"] is not None else 0.0),
            -float(x["payout"]),
            str(x["asset_id"]),
        )
    )

    # Tier 2: highest payout first, then highest trade count, alphabetical asset_id
    tier2_candidates.sort(
        key=lambda x: (
            -float(x["payout"]),
            -int(x["trades_count"]),
            str(x["asset_id"]),
        )
    )

    # Slot allocation
    selected_dicts: list[dict] = []

    # 1. Fill from Tier 1 up to max_measured_slots
    t1_take = min(len(tier1_candidates), max_t1_slots)
    selected_dicts.extend(tier1_candidates[:t1_take])

    # 2. Fill remaining slots up to top_assets from Tier 2
    remaining_needed = top_limit - len(selected_dicts)
    if remaining_needed > 0:
        t2_take = min(len(tier2_candidates), remaining_needed)
        selected_dicts.extend(tier2_candidates[:t2_take])

    # Final conversion to SelectedAsset dataclass with 1-based ranks
    results = [
        SelectedAsset(
            asset_id=d["asset_id"],
            tier=d["tier"],
            rank=i + 1,
            reason=d["reason"],
            payout=d["payout"],
            ev=d["ev"],
            win_rate=d["win_rate"],
            trades_count=d["trades_count"],
            is_otc=d["is_otc"],
        )
        for i, d in enumerate(selected_dicts)
    ]

    return results


def compute_correlation_map(
    candles_by_asset: Mapping[str, pd.DataFrame],
    quotes: Mapping[str, AssetQuote],
    lookback: int = 50,
) -> pd.DataFrame:
    """Compute Pearson correlation matrix of percentage returns across REAL-MARKET assets only.

    Rule from Section 6.4:
      OTC assets are strictly excluded from the correlation map.
    """
    return_series: dict[str, pd.Series] = {}

    for asset_id, df in candles_by_asset.items():
        quote = quotes.get(asset_id)
        # Exclude OTC assets
        if quote is not None and quote.is_otc:
            continue

        if df.empty or len(df) < 5:
            continue

        tail_df = df.tail(lookback)
        closes = pd.Series(tail_df["close"].values)
        pct_returns = closes.pct_change().dropna()

        # Guard against zero variance / constant price series
        if pct_returns.std() > 1e-8:
            return_series[asset_id] = pct_returns

    if not return_series:
        return pd.DataFrame()

    returns_df = pd.DataFrame(return_series)
    return returns_df.corr(method="pearson").round(4)


def related_assets(
    asset: str,
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
) -> list[str]:
    """Return all real-market assets having return correlation >= threshold with asset.

    Excludes self, sorted by correlation descending.
    """
    if corr_matrix.empty or asset not in corr_matrix.columns:
        return []

    asset_col = corr_matrix[asset]
    related: list[tuple[str, float]] = []

    for other_asset, val in asset_col.items():
        if other_asset == asset:
            continue
        if not np.isnan(val) and val >= threshold:
            related.append((str(other_asset), float(val)))

    related.sort(key=lambda x: -x[1])
    return [item[0] for item in related]
