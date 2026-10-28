"""Selection package — asset selector, ranking, stats, and correlation."""

from selection.asset_selector import (
    SelectedAsset,
    compute_correlation_map,
    related_assets,
    select_top_assets,
)
from selection.stats import (
    AssetStats,
    PatternStats,
    asset_mode,
    break_even,
    compute_asset_stats,
    compute_pattern_stats,
    ev,
    filter_trades_as_of,
    hour_risk,
    realized_return,
    wilson_interval,
)

__all__ = [
    "AssetStats",
    "PatternStats",
    "SelectedAsset",
    "break_even",
    "ev",
    "realized_return",
    "wilson_interval",
    "filter_trades_as_of",
    "compute_asset_stats",
    "compute_pattern_stats",
    "hour_risk",
    "asset_mode",
    "select_top_assets",
    "compute_correlation_map",
    "related_assets",
]
