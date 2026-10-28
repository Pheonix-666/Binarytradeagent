"""Unit tests for Phase 5: Asset selector, statistics, paper gate, and SQLite trade repository.

Requirements:
  - EV and break-even with known numbers (win_rate=0.6, payout=0.85 gives EV=0.11; payout=0.85 gives BE ≈ 0.5405)
  - Wilson interval known values (50/100 ≈ [0.404, 0.596])
  - Gate boundary (199 vs 200 trades, with and without margin)
  - Demotion on rolling win rate below break-even
  - Tiers and slot filling (max_measured_slots from Tier 1, deficit filled by Tier 2)
  - Closed assets excluded
  - OTC excluded from correlation map
  - hour_risk min samples requirement
  - as_of excludes later trades (no look-ahead)
  - Paper and live stats separated
  - SQLite TradeRepository CRUD operations
"""

from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from core.config import AppConfig, load_config
from core.models import AssetQuote
from data.store import DataStore, TradeRecord, TradeRepository
from selection.asset_selector import (
    compute_correlation_map,
    related_assets,
    select_top_assets,
)
from selection.stats import (
    AssetStats,
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


@pytest.fixture
def test_config() -> AppConfig:
    return load_config("config.yaml")


# ── Math and Statistical Formulas Tests ────────────────────────────────


def test_ev_and_break_even_known_numbers():
    """Verify EV and break-even formulas with exact known specification numbers."""
    payout = 0.85
    be = break_even(payout)
    # 1 / 1.85 ≈ 0.54054
    assert abs(be - 0.5405) < 0.001

    win_rate = 0.60
    ev_val = ev(win_rate, payout)
    # 0.6 * 0.85 - 0.4 = 0.51 - 0.4 = 0.11
    assert abs(ev_val - 0.11) < 0.0001


def test_wilson_interval_known_numbers():
    """Verify Wilson confidence interval: 50 wins out of 100 trials is approx [0.404, 0.596]."""
    lower, upper = wilson_interval(50, 100, z=1.96)
    assert abs(lower - 0.404) < 0.005
    assert abs(upper - 0.596) < 0.005

    # Zero trades returns (0.0, 1.0)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_realized_return():
    """Verify realized return: total pnl / total stake."""
    trades = [
        TradeRecord(id="1", opened_at="2026-10-29T10:00:00Z", asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=10.0, payout_at_entry=0.85, result="WIN", pnl=8.5, mode="paper"),
        TradeRecord(id="2", opened_at="2026-10-29T10:05:00Z", asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=10.0, payout_at_entry=0.85, result="LOSS", pnl=-10.0, mode="paper"),
    ]
    # Total stake = 20, Total pnl = -1.5 -> Return = -1.5 / 20 = -0.075 (-7.5%)
    assert realized_return(trades) == -0.075


# ── Look-Ahead & Filtering Tests ───────────────────────────────────────


def test_as_of_excludes_later_trades():
    """Trades closed after as_of timestamp must be strictly excluded from stats."""
    t1 = TradeRecord(id="1", opened_at="2026-10-29T10:00:00Z", closed_at="2026-10-29T10:30:00Z", asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=10.0, payout_at_entry=0.85, result="WIN", pnl=8.5, mode="paper")
    t2 = TradeRecord(id="2", opened_at="2026-10-29T11:00:00Z", closed_at="2026-10-29T11:30:00Z", asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=10.0, payout_at_entry=0.85, result="WIN", pnl=8.5, mode="paper")
    t3 = TradeRecord(id="3", opened_at="2026-10-29T12:00:00Z", closed_at="2026-10-29T12:30:00Z", asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=10.0, payout_at_entry=0.85, result="LOSS", pnl=-10.0, mode="paper")

    trades = [t1, t2, t3]

    # Evaluated at 11:30 -> includes t1 and t2, excludes t3
    as_of = datetime(2026, 10, 29, 11, 30, 0, tzinfo=timezone.utc)
    filtered = filter_trades_as_of(trades, as_of)
    assert len(filtered) == 2
    assert t3 not in filtered

    stats = compute_asset_stats(trades, asset="EURUSD", mode="paper", as_of=as_of)
    assert stats.n == 2
    assert stats.win_rate == 1.0


def test_paper_and_live_stats_never_mixed():
    """Paper trades and live trades for the same asset are tracked independently."""
    t_paper = TradeRecord(id="p1", opened_at="2026-10-29T10:00:00Z", closed_at="2026-10-29T10:30:00Z", asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=10.0, payout_at_entry=0.85, result="WIN", pnl=8.5, mode="paper")
    t_live = TradeRecord(id="l1", opened_at="2026-10-29T10:00:00Z", closed_at="2026-10-29T10:30:00Z", asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=50.0, payout_at_entry=0.85, result="LOSS", pnl=-50.0, mode="live")

    trades = [t_paper, t_live]

    paper_stats = compute_asset_stats(trades, asset="EURUSD", mode="paper")
    live_stats = compute_asset_stats(trades, asset="EURUSD", mode="live")

    assert paper_stats.n == 1
    assert paper_stats.win_rate == 1.0

    assert live_stats.n == 1
    assert live_stats.win_rate == 0.0


# ── Paper Gate & Demotion Tests ────────────────────────────────────────


def test_paper_gate_boundary_trade_count(test_config):
    """199 paper trades fails gate; 200 paper trades with sufficient edge passes."""
    be = break_even(0.85)  # ≈ 0.5405
    win_rate = be + 0.03   # 0.5705 > be + 0.02 margin

    # 1. 199 trades -> fails on trade count
    stats_199 = AssetStats(
        asset="EURUSD", mode="paper", n=199, wins=int(199 * win_rate), losses=199 - int(199 * win_rate),
        ties=0, win_rate=win_rate, mean_payout=0.85, break_even=be, ev=ev(win_rate, 0.85),
        realized_return=0.1, wilson_ci=(0.50, 0.64), edge_label="unproven",
        rolling_win_rate=win_rate, rolling_n=100,
    )
    mode, reason = asset_mode(stats_199, test_config)
    assert mode == "paper"
    assert "199 < 200" in reason

    # 2. 200 trades -> passes
    stats_200 = AssetStats(
        asset="EURUSD", mode="paper", n=200, wins=int(200 * win_rate), losses=200 - int(200 * win_rate),
        ties=0, win_rate=win_rate, mean_payout=0.85, break_even=be, ev=ev(win_rate, 0.85),
        realized_return=0.1, wilson_ci=(0.50, 0.64), edge_label="unproven",
        rolling_win_rate=win_rate, rolling_n=100,
    )
    mode, reason = asset_mode(stats_200, test_config)
    assert mode == "live_eligible"


def test_paper_gate_edge_margin_boundary(test_config):
    """Win rate must strictly exceed break-even + min_edge_margin (0.02)."""
    be = break_even(0.85)  # ≈ 0.54054

    # Win rate exactly at be + 0.015 (< 0.02 margin) -> fails
    wr_insufficient = be + 0.015
    stats_low = AssetStats(
        asset="EURUSD", mode="paper", n=250, wins=int(250 * wr_insufficient), losses=250 - int(250 * wr_insufficient),
        ties=0, win_rate=wr_insufficient, mean_payout=0.85, break_even=be, ev=ev(wr_insufficient, 0.85),
        realized_return=0.01, wilson_ci=(0.49, 0.61), edge_label="unproven",
        rolling_win_rate=wr_insufficient, rolling_n=100,
    )
    mode, reason = asset_mode(stats_low, test_config)
    assert mode == "paper"
    assert "does not exceed break-even + margin" in reason


def test_paper_gate_rolling_win_rate_demotion(test_config):
    """An asset with high overall win rate falls back to 'paper' if rolling win rate drops below break-even."""
    be = break_even(0.85)  # ≈ 0.54054
    stats = AssetStats(
        asset="EURUSD", mode="paper", n=300, wins=180, losses=120, ties=0,
        win_rate=0.60,  # 60% overall
        mean_payout=0.85, break_even=be, ev=ev(0.60, 0.85),
        realized_return=0.11, wilson_ci=(0.54, 0.65), edge_label="positive_edge",
        rolling_win_rate=0.51,  # 51% < 54.05% break-even
        rolling_n=100,
    )
    mode, reason = asset_mode(stats, test_config)
    assert mode == "paper"
    assert "Rolling win rate (0.5100) dropped below break-even" in reason


def test_pattern_auto_disable_on_rolling_win_rate():
    """A reversal pattern is auto-disabled when its rolling win rate falls below break-even."""
    # 15 trades for reversal pattern with low rolling win rate
    trades = [
        TradeRecord(
            id=f"t_{i}", opened_at="2026-10-29T10:00:00Z", closed_at="2026-10-29T10:30:00Z",
            asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=10.0,
            payout_at_entry=0.85, result="WIN" if i < 6 else "LOSS", pnl=8.5 if i < 6 else -10.0,
            mode="paper", pattern_name="double_bottom", pattern_type="reversal",
        )
        for i in range(15)
    ]
    # 6 wins out of 15 = 40.0% win rate (< 54.05% break-even)
    p_stats = compute_pattern_stats(trades, pattern_name="double_bottom", pattern_type="reversal", mode="paper")
    assert p_stats.is_disabled is True


# ── Session Hour Risk Tests ────────────────────────────────────────────


def test_hour_risk_min_samples():
    """hour_risk returns None if fewer than session_min_samples trades exist."""
    trades = [
        TradeRecord(
            id=f"t_{i}", opened_at="2026-10-29T14:00:00Z", closed_at="2026-10-29T14:30:00Z",
            asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=10.0,
            payout_at_entry=0.85, result="WIN", pnl=8.5, mode="paper", hour_utc=14,
        )
        for i in range(10)  # only 10 trades, min_samples=20
    ]
    assert hour_risk(trades, hour=14, session_min_samples=20, mode="paper") is None

    # Add 15 more trades (total 25 >= 20)
    for i in range(10, 25):
        trades.append(
            TradeRecord(
                id=f"t_{i}", opened_at="2026-10-29T14:00:00Z", closed_at="2026-10-29T14:30:00Z",
                asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=10.0,
                payout_at_entry=0.85, result="WIN" if i % 2 == 0 else "LOSS", pnl=8.5 if i % 2 == 0 else -10.0,
                mode="paper", hour_utc=14,
            )
        )
    risk_val = hour_risk(trades, hour=14, session_min_samples=20, mode="paper")
    assert risk_val is not None
    assert 1.0 <= risk_val <= 100.0


# ── Asset Selector & Ranking Tests ─────────────────────────────────────


def test_select_top_assets_tiers_and_slot_filling(test_config):
    """Tier 1 takes up to max_measured_slots; Tier 2 fills remainder up to top_assets."""
    quotes = {
        # Tier 1 candidates (measured)
        "EURUSD": AssetQuote(asset_id="EURUSD", name="EUR/USD", is_otc=False, payout=0.85, is_open=True),
        "GBPUSD": AssetQuote(asset_id="GBPUSD", name="GBP/USD", is_otc=False, payout=0.82, is_open=True),
        "USDJPY": AssetQuote(asset_id="USDJPY", name="USD/JPY", is_otc=False, payout=0.88, is_open=True),
        "AUDUSD": AssetQuote(asset_id="AUDUSD", name="AUD/USD", is_otc=False, payout=0.80, is_open=True),
        # Tier 2 candidates (new / untested)
        "NZDUSD": AssetQuote(asset_id="NZDUSD", name="NZD/USD", is_otc=False, payout=0.92, is_open=True),
        "USDCAD": AssetQuote(asset_id="USDCAD", name="USD/CAD", is_otc=False, payout=0.90, is_open=True),
        "USDCHF": AssetQuote(asset_id="USDCHF", name="USD/CHF", is_otc=False, payout=0.84, is_open=True),
    }

    # Stats: EURUSD, GBPUSD, USDJPY, AUDUSD have >= 30 trades
    stats_map = {
        "EURUSD": AssetStats(asset="EURUSD", mode="paper", n=50, wins=32, losses=18, ties=0, win_rate=0.64, mean_payout=0.85, break_even=0.54, ev=0.18, realized_return=0.15, wilson_ci=(0.50, 0.76), edge_label="unproven", rolling_win_rate=0.64, rolling_n=100),
        "GBPUSD": AssetStats(asset="GBPUSD", mode="paper", n=40, wins=26, losses=14, ties=0, win_rate=0.65, mean_payout=0.82, break_even=0.55, ev=0.15, realized_return=0.12, wilson_ci=(0.50, 0.77), edge_label="unproven", rolling_win_rate=0.65, rolling_n=100),
        "USDJPY": AssetStats(asset="USDJPY", mode="paper", n=35, wins=20, losses=15, ties=0, win_rate=0.57, mean_payout=0.88, break_even=0.53, ev=0.10, realized_return=0.08, wilson_ci=(0.41, 0.72), edge_label="unproven", rolling_win_rate=0.57, rolling_n=100),
        "AUDUSD": AssetStats(asset="AUDUSD", mode="paper", n=30, wins=16, losses=14, ties=0, win_rate=0.53, mean_payout=0.80, break_even=0.55, ev=-0.05, realized_return=-0.04, wilson_ci=(0.36, 0.70), edge_label="unproven", rolling_win_rate=0.53, rolling_n=100),
    }

    # config.trading.top_assets = 5, config.trading.max_measured_slots = 3
    selected = select_top_assets(quotes, stats_map, test_config)

    assert len(selected) == 5

    # Top 3 must be Tier 1 ranked by EV: EURUSD (0.18), GBPUSD (0.15), USDJPY (0.10)
    assert selected[0].asset_id == "EURUSD" and selected[0].tier == 1
    assert selected[1].asset_id == "GBPUSD" and selected[1].tier == 1
    assert selected[2].asset_id == "USDJPY" and selected[2].tier == 1

    # Remaining 2 must be Tier 2 ranked by payout: NZDUSD (0.92), USDCAD (0.90)
    assert selected[3].asset_id == "NZDUSD" and selected[3].tier == 2
    assert selected[4].asset_id == "USDCAD" and selected[4].tier == 2


def test_select_top_assets_tier1_deficit_filled_by_tier2(test_config):
    """When Tier 1 has fewer candidates than max_measured_slots, Tier 2 fills the deficit."""
    quotes = {
        "EURUSD": AssetQuote(asset_id="EURUSD", name="EUR/USD", is_otc=False, payout=0.85, is_open=True),
        "GBPUSD": AssetQuote(asset_id="GBPUSD", name="GBP/USD", is_otc=False, payout=0.92, is_open=True),
        "USDJPY": AssetQuote(asset_id="USDJPY", name="USD/JPY", is_otc=False, payout=0.90, is_open=True),
        "AUDUSD": AssetQuote(asset_id="AUDUSD", name="AUD/USD", is_otc=False, payout=0.88, is_open=True),
        "NZDUSD": AssetQuote(asset_id="NZDUSD", name="NZD/USD", is_otc=False, payout=0.86, is_open=True),
    }

    # Only 1 asset has >= 30 trades (EURUSD)
    stats_map = {
        "EURUSD": AssetStats(asset="EURUSD", mode="paper", n=50, wins=30, losses=20, ties=0, win_rate=0.60, mean_payout=0.85, break_even=0.54, ev=0.11, realized_return=0.10, wilson_ci=(0.46, 0.72), edge_label="unproven", rolling_win_rate=0.60, rolling_n=100),
    }

    selected = select_top_assets(quotes, stats_map, test_config)
    assert len(selected) == 5

    # 1 from Tier 1, 4 from Tier 2
    assert selected[0].tier == 1 and selected[0].asset_id == "EURUSD"
    assert all(s.tier == 2 for s in selected[1:])


def test_select_top_assets_excludes_closed_assets(test_config):
    """Closed assets (is_open == False) must be strictly excluded."""
    quotes = {
        "EURUSD": AssetQuote(asset_id="EURUSD", name="EUR/USD", is_otc=False, payout=0.85, is_open=True),
        "GBPUSD": AssetQuote(asset_id="GBPUSD", name="GBP/USD", is_otc=False, payout=0.95, is_open=False),  # CLOSED
    }
    selected = select_top_assets(quotes, {}, test_config)
    asset_ids = [s.asset_id for s in selected]
    assert "EURUSD" in asset_ids
    assert "GBPUSD" not in asset_ids


# ── Correlation Map Tests ──────────────────────────────────────────────


def test_correlation_map_excludes_otc_assets():
    """Real-market return correlation map strictly excludes OTC assets."""
    np.random.seed(42)
    candles = {
        "EURUSD": pd.DataFrame({"close": np.linspace(1.08, 1.09, 50) + np.random.normal(0, 0.0001, 50)}),
        "GBPUSD": pd.DataFrame({"close": np.linspace(1.28, 1.29, 50) + np.random.normal(0, 0.0001, 50)}),
        "EURUSD_OTC": pd.DataFrame({"close": np.linspace(1.08, 1.09, 50) + np.random.normal(0, 0.0001, 50)}),
    }
    quotes = {
        "EURUSD": AssetQuote(asset_id="EURUSD", name="EUR/USD", is_otc=False, payout=0.85, is_open=True),
        "GBPUSD": AssetQuote(asset_id="GBPUSD", name="GBP/USD", is_otc=False, payout=0.85, is_open=True),
        "EURUSD_OTC": AssetQuote(asset_id="EURUSD_OTC", name="EUR/USD OTC", is_otc=True, payout=0.92, is_open=True),
    }

    corr_df = compute_correlation_map(candles, quotes, lookback=50)

    # OTC asset must NOT appear in correlation matrix
    assert "EURUSD_OTC" not in corr_df.columns
    assert "EURUSD" in corr_df.columns
    assert "GBPUSD" in corr_df.columns

    rel = related_assets("EURUSD", corr_df, threshold=0.5)
    assert isinstance(rel, list)


# ── SQLite TradeRepository Tests ───────────────────────────────────────


def test_trade_repository_crud(tmp_path):
    """Test insert, update_result, get_trades, and get_trade_by_id in TradeRepository."""
    db_file = tmp_path / "test_trades.sqlite"
    store = DataStore(db_path=db_file)
    repo = TradeRepository(store)

    trade = TradeRecord(
        id="trade_123",
        opened_at="2026-10-29T10:00:00Z",
        asset="EURUSD",
        is_otc=False,
        direction="CALL",
        expiry_minutes=30,
        stake=25.0,
        payout_at_entry=0.85,
        result="PENDING",
        pnl=0.0,
        mode="paper",
        pattern_name="double_bottom",
        pattern_type="reversal",
        risk_score=35,
    )

    # 1. Insert
    repo.insert(trade)

    # 2. Get by ID
    fetched = repo.get_trade_by_id("trade_123")
    assert fetched is not None
    assert fetched.asset == "EURUSD"
    assert fetched.result == "PENDING"
    assert fetched.hour_utc == 10

    # 3. Update Result
    updated = repo.update_result("trade_123", closed_at="2026-10-29T10:30:00Z", result="WIN", pnl=21.25)
    assert updated is True

    # 4. Verify updated state
    fetched_after = repo.get_trade_by_id("trade_123")
    assert fetched_after.result == "WIN"
    assert fetched_after.pnl == 21.25
    assert fetched_after.closed_at == "2026-10-29T10:30:00Z"

    # 5. Query with filter
    all_paper = repo.get_trades(asset="EURUSD", mode="paper")
    assert len(all_paper) == 1
    assert all_paper[0].id == "trade_123"

    live_trades = repo.get_trades(asset="EURUSD", mode="live")
    assert len(live_trades) == 0


# ── Shrinkage, Modes & Stricter Paper Gate Tests ───────────────────────


def test_shrunk_win_rate_known_numbers():
    """Small n is pulled strongly toward break-even; large n converges to raw win rate."""
    from selection.stats import shrunk_win_rate

    be = 0.54054054  # break-even for 85% payout
    k = 50.0

    # 1. Small n = 5, all 5 wins (raw win rate = 1.0)
    # wr_adj = (5 + 50 * 0.54054054) / (5 + 50) = 32.027027 / 55 ≈ 0.582309
    adj_small = shrunk_win_rate(wins=5, n=5, break_even_rate=be, k=k)
    assert abs(adj_small - 0.5823) < 0.001
    assert adj_small < 0.60  # Pulled strongly from 100% down towards ~54%

    # 2. Large n = 5000, 3000 wins (raw win rate = 0.6000)
    # wr_adj = (3000 + 50 * 0.54054054) / (5000 + 50) = 3027.027 / 5050 ≈ 0.599411
    adj_large = shrunk_win_rate(wins=3000, n=5000, break_even_rate=be, k=k)
    assert abs(adj_large - 0.5994) < 0.001
    assert abs(adj_large - 0.6000) < 0.001  # Converged to raw win rate within 0.1%


def test_select_top_assets_ranks_by_shrunk_ev(test_config):
    """Tier 1 ranking uses shrunk EV, preventing low-n hot streaks from outranking established assets."""
    quotes = {
        "ASSET_HOT": AssetQuote(asset_id="ASSET_HOT", name="Hot", is_otc=False, payout=0.85, is_open=True),
        "ASSET_SOLID": AssetQuote(asset_id="ASSET_SOLID", name="Solid", is_otc=False, payout=0.85, is_open=True),
    }

    # ASSET_HOT: n=30, wins=25 (83.3% win rate, raw EV = 0.8333 * 0.85 - 0.1667 = +0.5416)
    # Shrunk wr = (25 + 50*0.54054) / 80 ≈ 0.6503 -> shrunk EV ≈ 0.2031
    # ASSET_SOLID: n=200, wins=140 (70.0% win rate, raw EV = 0.70 * 0.85 - 0.30 = +0.2950)
    # Shrunk wr = (140 + 50*0.54054) / 250 ≈ 0.6681 -> shrunk EV ≈ 0.2359
    stats_map = {
        "ASSET_HOT": AssetStats(
            asset="ASSET_HOT", mode="paper", n=30, wins=25, losses=5, ties=0,
            win_rate=0.8333, mean_payout=0.85, break_even=0.5405, ev=0.5416,
            realized_return=0.50, wilson_ci=(0.66, 0.93), edge_label="positive_edge",
            rolling_win_rate=0.8333, rolling_n=100,
        ),
        "ASSET_SOLID": AssetStats(
            asset="ASSET_SOLID", mode="paper", n=200, wins=140, losses=60, ties=0,
            win_rate=0.7000, mean_payout=0.85, break_even=0.5405, ev=0.2950,
            realized_return=0.25, wilson_ci=(0.63, 0.76), edge_label="positive_edge",
            rolling_win_rate=0.7000, rolling_n=100,
        ),
    }

    selected = select_top_assets(quotes, stats_map, test_config)
    # ASSET_SOLID has higher shrunk EV (0.2359 > 0.2031), so it ranks rank 1!
    assert selected[0].asset_id == "ASSET_SOLID"
    assert selected[1].asset_id == "ASSET_HOT"

    # But raw EV is preserved in output for display
    assert selected[0].ev == 0.2950
    assert selected[1].ev == 0.5416


def test_paper_gate_require_ci_lower_bound_flag(test_config):
    """Optional stricter paper gate requires Wilson CI lower bound > break_even."""
    be = break_even(0.85)  # ≈ 0.54054
    # 116 wins out of 200 = 58% win rate (> 54.05% + 2% margin = 56.05%)
    # Wilson CI for 116/200 has lower bound ≈ 0.5113 <= 0.54054
    lower, upper = wilson_interval(116, 200)
    assert lower < be  # CI lower bound is below break-even

    stats = AssetStats(
        asset="EURUSD", mode="paper", n=200, wins=116, losses=84, ties=0,
        win_rate=0.58, mean_payout=0.85, break_even=be, ev=ev(0.58, 0.85),
        realized_return=0.07, wilson_ci=(lower, upper), edge_label="unproven",
        rolling_win_rate=0.58, rolling_n=100,
    )

    # 1. Standard gate (require_ci_lower_bound = False)
    test_config.paper_gate.require_ci_lower_bound = False
    mode_std, reason_std = asset_mode(stats, test_config)
    assert mode_std == "live_eligible"
    assert "Passed paper gate" in reason_std
    assert f"[CI lower bound: {lower:.4f} > BE {be:.4f}: False]" in reason_std

    # 2. Stricter gate (require_ci_lower_bound = True)
    test_config.paper_gate.require_ci_lower_bound = True
    mode_strict, reason_strict = asset_mode(stats, test_config)
    assert mode_strict == "paper"
    assert "CI lower bound condition failed" in reason_strict
    assert f"[CI lower bound: {lower:.4f} > BE {be:.4f}: False]" in reason_strict


def test_stats_modes_separation(test_config):
    """Trade modes separation: selection and risk use configured modes; paper_gate uses paper only; backtest is never mixed."""
    trades = [
        # Paper trade (WIN)
        TradeRecord(id="p1", opened_at="2026-10-29T10:00:00Z", closed_at="2026-10-29T10:30:00Z", asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=10.0, payout_at_entry=0.85, result="WIN", pnl=8.5, mode="paper"),
        # Demo trade (WIN)
        TradeRecord(id="d1", opened_at="2026-10-29T10:05:00Z", closed_at="2026-10-29T10:35:00Z", asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=10.0, payout_at_entry=0.85, result="WIN", pnl=8.5, mode="demo"),
        # Live trade (LOSS)
        TradeRecord(id="l1", opened_at="2026-10-29T10:10:00Z", closed_at="2026-10-29T10:40:00Z", asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=10.0, payout_at_entry=0.85, result="LOSS", pnl=-10.0, mode="live"),
        # Backtest trade (WIN) - must NEVER be mixed with live/demo/paper!
        TradeRecord(id="b1", opened_at="2026-10-29T10:15:00Z", closed_at="2026-10-29T10:45:00Z", asset="EURUSD", is_otc=False, direction="CALL", expiry_minutes=30, stake=10.0, payout_at_entry=0.85, result="WIN", pnl=8.5, mode="backtest"),
    ]

    # 1. Ranking modes from config: ["paper", "demo", "live"]
    ranking_modes = test_config.selection.ranking_modes
    assert "backtest" not in ranking_modes
    ranking_stats = compute_asset_stats(trades, asset="EURUSD", mode=ranking_modes)
    assert ranking_stats.n == 3  # p1, d1, l1 (excludes b1)
    assert ranking_stats.wins == 2
    assert ranking_stats.losses == 1

    # 2. Risk modes from config: ["paper", "demo", "live"]
    risk_modes = test_config.risk.stats_modes
    assert "backtest" not in risk_modes
    risk_stats = compute_asset_stats(trades, asset="EURUSD", mode=risk_modes)
    assert risk_stats.n == 3

    # 3. Paper gate strictly uses "paper" mode
    paper_stats = compute_asset_stats(trades, asset="EURUSD", mode="paper")
    assert paper_stats.n == 1
    assert paper_stats.wins == 1
    assert paper_stats.mode == "paper"


