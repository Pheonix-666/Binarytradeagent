"""Unit tests for Phase 4: Risk engine and stake calculator.

Requirements:
  - risk 1 -> 20%, risk 100 -> 2%, risk 50 -> ~11.09%
  - property test: stake never increases as risk increases (over 1-100)
  - each sub-score at its boundaries
  - every missing-input default (pattern=100, untested_asset=100, session=50, no_sr=10)
  - weight changes alter the result
  - min_amount SKIP: returns None if below min_amount (no rounding up)
  - stake rounding to increment
"""

from __future__ import annotations

import pytest

from core.config import RiskConfig, RiskDefaultsConfig, RiskWeightsConfig, StakeConfig
from risk.engine import (
    RiskInput,
    RiskReport,
    compute_pattern_subscore,
    compute_risk_score,
    compute_session_subscore,
    compute_sr_distance_subscore,
    compute_volatility_subscore,
    compute_win_rate_subscore,
)
from risk.stake import compute_equity, max_stake_pct, stake_amount


@pytest.fixture
def default_risk_config() -> RiskConfig:
    return RiskConfig(
        weights=RiskWeightsConfig(
            volatility=0.2,
            pattern=0.2,
            win_rate=0.2,
            session=0.2,
            sr_distance=0.2,
        ),
        volatility_lookback_days=30,
        atr_period=14,
        session_min_samples=20,
        defaults=RiskDefaultsConfig(
            pattern_missing=100.0,
            untested_asset=100.0,
            session_missing=50.0,
            no_sr_level=10.0,
            volatility_missing=50.0,
        ),
        sr_distance_min_atr=0.5,
        sr_distance_max_atr=3.0,
        sr_distance_near_score=90.0,
        sr_distance_far_score=10.0,
    )


# ── Stake Formula Tests ────────────────────────────────────────────────


def test_max_stake_pct_known_values():
    """risk 1 -> 20%, risk 100 -> 2%, risk 50 -> ~11.09%."""
    assert max_stake_pct(1.0, min_pct=2.0, max_pct=20.0) == 20.0
    assert max_stake_pct(100.0, min_pct=2.0, max_pct=20.0) == 2.0

    # risk 50: 20 - (49/99)*18 = 20 - 8.9090909... = 11.090909...
    mid = max_stake_pct(50.0, min_pct=2.0, max_pct=20.0)
    assert abs(mid - 11.0909) < 0.001


def test_max_stake_pct_property_monotonic_decrease():
    """Property test: stake percentage never increases as risk score increases from 1 to 100."""
    previous = max_stake_pct(1.0, min_pct=2.0, max_pct=20.0)
    for r in range(2, 101):
        current = max_stake_pct(float(r), min_pct=2.0, max_pct=20.0)
        assert current <= previous, f"Stake increased at risk {r}: {current} > {previous}"
        previous = current


def test_max_stake_pct_clamped_at_extremes():
    """Out-of-range risk inputs must be clamped safely to [min_pct, max_pct]."""
    assert max_stake_pct(0.0, min_pct=2.0, max_pct=20.0) == 20.0
    assert max_stake_pct(-10.0, min_pct=2.0, max_pct=20.0) == 20.0
    assert max_stake_pct(150.0, min_pct=2.0, max_pct=20.0) == 2.0


def test_stake_amount_rounding_and_skip():
    """Stake calculation rounds to increment and skips when below min_amount."""
    equity = 1000.0  # $1,000 equity

    # 5.4% of 1000 = $54 -> rounds to 54.0 with round_to=1.0
    assert stake_amount(equity, pct=5.4, min_amount=1.0, round_to=1.0) == 54.0

    # 5.6% of 1000 = $56 -> rounds to 56.0
    assert stake_amount(equity, pct=5.6, min_amount=1.0, round_to=1.0) == 56.0

    # Fractional round_to=0.5: 5.24% of 1000 = $52.4 -> rounds to 52.5
    assert stake_amount(equity, pct=5.26, min_amount=1.0, round_to=0.5) == 52.5

    # Skip when below min_amount:
    # 2% of $40 equity = $0.80. min_amount=1.0. Must return None (SKIP), NOT round up to 1.0!
    assert stake_amount(equity=40.0, pct=2.0, min_amount=1.0, round_to=1.0) is None

    # Zero or negative inputs return None
    assert stake_amount(equity=0.0, pct=5.0) is None
    assert stake_amount(equity=100.0, pct=0.0) is None


def test_compute_equity():
    """Equity = cash balance + open stakes."""
    assert compute_equity(cash_balance=500.0, open_stakes=50.0) == 550.0
    assert compute_equity(cash_balance=500.0, open_stakes=0.0) == 500.0
    assert compute_equity(cash_balance=-10.0, open_stakes=0.0) == 0.0


# ── Sub-Score Boundary Tests ───────────────────────────────────────────


def test_volatility_subscore_boundaries(default_risk_config):
    """Volatility subscore clamped to 0-100 and defaults when missing."""
    assert compute_volatility_subscore(0.0, default_risk_config) == 0.0
    assert compute_volatility_subscore(100.0, default_risk_config) == 100.0
    assert compute_volatility_subscore(50.0, default_risk_config) == 50.0
    assert compute_volatility_subscore(-10.0, default_risk_config) == 0.0
    assert compute_volatility_subscore(120.0, default_risk_config) == 100.0
    assert compute_volatility_subscore(None, default_risk_config) == 50.0


def test_pattern_subscore_boundaries(default_risk_config):
    """Pattern subscore is 100 - similarity; defaults to 100 if missing."""
    assert compute_pattern_subscore(100.0, default_risk_config) == 0.0
    assert compute_pattern_subscore(0.0, default_risk_config) == 100.0
    assert compute_pattern_subscore(75.0, default_risk_config) == 25.0
    assert compute_pattern_subscore(None, default_risk_config) == 100.0


def test_win_rate_subscore_boundaries(default_risk_config):
    """Win rate subscore: 100 * clip((break_even + 0.15 - win_rate) / 0.20, 0.01, 1)."""
    payout = 0.85
    be = 1.0 / (1.0 + payout)  # ≈ 0.54054
    min_trades = 30

    # Untested asset (trades < min_trades) -> default 100
    assert compute_win_rate_subscore(0.70, asset_trades=10, payout=payout, min_trades_for_stats=min_trades, risk_cfg=default_risk_config) == 100.0
    assert compute_win_rate_subscore(None, asset_trades=50, payout=payout, min_trades_for_stats=min_trades, risk_cfg=default_risk_config) == 100.0

    # Very high win rate (be + 0.15 or higher) -> clips to 1.0 (min risk)
    high_wr = be + 0.16
    score_high = compute_win_rate_subscore(high_wr, asset_trades=50, payout=payout, min_trades_for_stats=min_trades, risk_cfg=default_risk_config)
    assert score_high == 1.0

    # Very low win rate (be - 0.05 or lower) -> clips to 100.0 (max risk)
    low_wr = be - 0.10
    score_low = compute_win_rate_subscore(low_wr, asset_trades=50, payout=payout, min_trades_for_stats=min_trades, risk_cfg=default_risk_config)
    assert score_low == 100.0

    # Exact midpoint: win_rate == be + 0.05 -> (0.15 - 0.05)/0.20 = 0.50 -> 50.0
    mid_wr = be + 0.05
    score_mid = compute_win_rate_subscore(mid_wr, asset_trades=50, payout=payout, min_trades_for_stats=min_trades, risk_cfg=default_risk_config)
    assert abs(score_mid - 50.0) < 0.01


def test_session_subscore_boundaries(default_risk_config):
    """Session risk subscore clamped and defaults to 50 when missing."""
    assert compute_session_subscore(0.0, default_risk_config) == 0.0
    assert compute_session_subscore(100.0, default_risk_config) == 100.0
    assert compute_session_subscore(None, default_risk_config) == 50.0


def test_sr_distance_subscore_boundaries(default_risk_config):
    """S/R distance subscore: <=0.5 ATR -> 90; >=3.0 ATR -> 10; linear between; None -> 10."""
    # Under 0.5 ATR = 90
    assert compute_sr_distance_subscore(0.5, default_risk_config) == 90.0
    assert compute_sr_distance_subscore(0.2, default_risk_config) == 90.0

    # 3.0 ATR or more = 10
    assert compute_sr_distance_subscore(3.0, default_risk_config) == 10.0
    assert compute_sr_distance_subscore(5.0, default_risk_config) == 10.0

    # Midpoint: 1.75 ATR -> (90 + 10) / 2 = 50.0
    assert compute_sr_distance_subscore(1.75, default_risk_config) == 50.0

    # Missing level -> default 10.0
    assert compute_sr_distance_subscore(None, default_risk_config) == 10.0


# ── Composite Risk Engine & Default Handling Tests ─────────────────────


def test_risk_engine_every_missing_input_default(default_risk_config):
    """When all optional inputs are None, the risk engine uses config defaults without raising."""
    inp = RiskInput(
        volatility_percentile=None,     # default 50
        best_pattern_similarity=None,   # default 100
        asset_win_rate=None,            # default 100
        asset_trades=0,                 # < 30 -> untested default 100
        payout=0.85,
        hour_of_day_risk=None,          # default 50
        sr_distance_atr=None,           # default 10
    )

    report = compute_risk_score(inp, default_risk_config, min_trades_for_stats=30)

    assert isinstance(report, RiskReport)
    assert report.volatility_score == 50.0
    assert report.pattern_score == 100.0
    assert report.win_rate_score == 100.0
    assert report.session_score == 50.0
    assert report.sr_distance_score == 10.0

    # Expected weighted average: 0.2*(50 + 100 + 100 + 50 + 10) = 0.2 * 310 = 62.0
    assert report.score == 62
    assert report.raw_weighted_score == 62.0


def test_risk_engine_weight_changes_alter_result(default_risk_config):
    """Changing sub-score weights directly shifts the final risk score."""
    inp = RiskInput(
        volatility_percentile=10.0,
        best_pattern_similarity=95.0,  # pattern risk = 5
        asset_win_rate=0.75,
        asset_trades=50,
        payout=0.85,
        hour_of_day_risk=90.0,         # high session risk = 90
        sr_distance_atr=3.5,           # low S/R risk = 10
    )

    # 1. Base weights (equal 0.2 each)
    report_base = compute_risk_score(inp, default_risk_config)

    # 2. Config heavily weighted towards session risk (90.0)
    heavy_session_cfg = RiskConfig(
        weights=RiskWeightsConfig(
            volatility=0.05,
            pattern=0.05,
            win_rate=0.05,
            session=0.80,
            sr_distance=0.05,
        ),
        volatility_lookback_days=30,
        atr_period=14,
        session_min_samples=20,
        defaults=default_risk_config.defaults,
    )
    report_heavy_session = compute_risk_score(inp, heavy_session_cfg)

    # 3. Config heavily weighted towards pattern risk (5.0)
    heavy_pattern_cfg = RiskConfig(
        weights=RiskWeightsConfig(
            volatility=0.05,
            pattern=0.80,
            win_rate=0.05,
            session=0.05,
            sr_distance=0.05,
        ),
        volatility_lookback_days=30,
        atr_period=14,
        session_min_samples=20,
        defaults=default_risk_config.defaults,
    )
    report_heavy_pattern = compute_risk_score(inp, heavy_pattern_cfg)

    assert report_heavy_session.score > report_base.score
    assert report_heavy_pattern.score < report_base.score
    assert report_heavy_session.score >= 70
    assert report_heavy_pattern.score <= 20
