"""Tests for configuration validation."""

import os
from pathlib import Path
import pytest
from pydantic import ValidationError

from core.config import (
    AppConfig,
    load_config,
    TradingConfig,
    StakeConfig,
    TrendConfig,
    RiskWeightsConfig,
)


def test_load_valid_config():
    """Default config.yaml should load without error and match expected defaults."""
    cfg = load_config("config.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.mode == "paper"
    assert cfg.live_requires_env == "LIVE_CONFIRM"
    assert cfg.data.daily_boundary_tz == "UTC"
    assert cfg.trading.expiry_minutes == 30
    assert cfg.trading.allowed_expiries == [15, 30, 60]
    assert cfg.stake.min_pct == 2
    assert cfg.stake.max_pct == 20
    assert cfg.stake.total_cap_pct == 35
    assert cfg.stake.daily_loss_limit_pct is None
    assert cfg.gate.strict_trend_gate is True
    assert cfg.gate.reversal_mode == "off"
    assert cfg.gate.trend.daily.ema_fast == 20
    assert cfg.gate.trend.daily.ema_slow == 50
    assert cfg.gate.trend.daily.warmup_multiplier == 3
    assert cfg.gate.trend.h1.ema_fast == 20
    assert cfg.gate.trend.h1.ema_slow == 50
    assert cfg.gate.trend.m15.ema_fast == 20
    assert cfg.gate.trend.m15.ema_slow == 50
    assert cfg.levels.swing_left_bars == 5
    assert cfg.levels.swing_right_bars == 2
    assert cfg.levels.cluster_tolerance_atr == 0.5
    assert cfg.levels.at_level_tolerance_atr == 0.5
    assert cfg.patterns.min_similarity == 60
    assert cfg.risk.volatility_lookback_days == 30
    assert cfg.paper_gate.min_paper_trades == 200
    assert cfg.safety.max_orders_per_10min == 6
    assert cfg.safety.kill_file == "KILL"
    assert cfg.llm.provider == "anthropic"
    assert cfg.alerts.telegram_enabled is True
    assert cfg.backtest.default_payout == 0.85


def test_min_pct_greater_than_max_pct_rejected(temp_config_file):
    """min_pct > max_pct must be rejected."""
    path = temp_config_file(**{"stake.min_pct": 25, "stake.max_pct": 20})
    with pytest.raises(ValidationError, match="min_pct=25.0 must be <= max_pct=20.0"):
        load_config(path)


def test_max_pct_greater_than_total_cap_rejected(temp_config_file):
    """max_pct > total_cap_pct must be rejected."""
    path = temp_config_file(**{"stake.max_pct": 40, "stake.total_cap_pct": 35})
    with pytest.raises(ValidationError, match="max_pct=40.0 must be <= total_cap_pct=35.0"):
        load_config(path)


def test_expiry_not_in_allowed_rejected(temp_config_file):
    """expiry_minutes not in allowed_expiries must be rejected."""
    path = temp_config_file(**{"trading.expiry_minutes": 45})
    with pytest.raises(ValidationError, match="expiry_minutes=45 not in allowed_expiries"):
        load_config(path)


def test_allowed_expiries_invalid_rejected(temp_config_file):
    """allowed_expiries with values other than 15, 30, 60 must be rejected."""
    path = temp_config_file(**{"trading.allowed_expiries": [10, 30, 60], "trading.expiry_minutes": 30})
    with pytest.raises(ValidationError, match="must be subset of"):
        load_config(path)


def test_max_measured_slots_exceeds_top_assets_rejected(temp_config_file):
    """max_measured_slots > top_assets must be rejected."""
    path = temp_config_file(**{"trading.max_measured_slots": 6, "trading.top_assets": 5})
    with pytest.raises(ValidationError, match="max_measured_slots=6 must be <= top_assets=5"):
        load_config(path)


def test_ema_fast_gte_slow_rejected(temp_config_file):
    """ema_fast >= ema_slow must be rejected for any timeframe (daily, h1, m15)."""
    # Daily
    path_daily = temp_config_file(**{"gate.trend.daily.ema_fast": 50, "gate.trend.daily.ema_slow": 20})
    with pytest.raises(ValidationError, match="ema_fast=50 must be < ema_slow=20"):
        load_config(path_daily)

    # H1
    path_h1 = temp_config_file(**{"gate.trend.h1.ema_fast": 60, "gate.trend.h1.ema_slow": 30})
    with pytest.raises(ValidationError, match="ema_fast=60 must be < ema_slow=30"):
        load_config(path_h1)

    # M15
    path_m15 = temp_config_file(**{"gate.trend.m15.ema_fast": 25, "gate.trend.m15.ema_slow": 25})
    with pytest.raises(ValidationError, match="ema_fast=25 must be < ema_slow=25"):
        load_config(path_m15)


def test_risk_weights_dont_sum_to_one_rejected(temp_config_file):
    """Risk weights must sum to 1.0."""
    path = temp_config_file(**{"risk.weights.volatility": 0.5})
    with pytest.raises(ValidationError, match="Risk weights must sum to 1.0"):
        load_config(path)


def test_live_mode_without_env_rejected(temp_config_file, monkeypatch):
    """Live mode without LIVE_CONFIRM=YES must raise ValueError."""
    monkeypatch.delenv("LIVE_CONFIRM", raising=False)
    path = temp_config_file(**{"mode": "live"})
    with pytest.raises(ValueError, match="mode is 'live' but env var LIVE_CONFIRM is not 'YES'"):
        load_config(path)


def test_live_mode_with_wrong_env_rejected(temp_config_file, monkeypatch):
    """Live mode with LIVE_CONFIRM != YES must raise ValueError."""
    monkeypatch.setenv("LIVE_CONFIRM", "NO")
    path = temp_config_file(**{"mode": "live"})
    with pytest.raises(ValueError, match="mode is 'live' but env var LIVE_CONFIRM is not 'YES'"):
        load_config(path)


def test_live_mode_with_env_accepted(temp_config_file, monkeypatch):
    """Live mode with LIVE_CONFIRM=YES must load successfully."""
    monkeypatch.setenv("LIVE_CONFIRM", "YES")
    path = temp_config_file(**{"mode": "live"})
    cfg = load_config(path)
    assert cfg.mode == "live"


def test_invalid_mode_rejected(temp_config_file):
    """Invalid mode string must be rejected."""
    path = temp_config_file(**{"mode": "real_money"})
    with pytest.raises(ValidationError):
        load_config(path)


def test_invalid_reversal_mode_rejected(temp_config_file):
    """Invalid reversal_mode string must be rejected."""
    path = temp_config_file(**{"gate.reversal_mode": "always_on"})
    with pytest.raises(ValidationError):
        load_config(path)


def test_negative_values_rejected(temp_config_file):
    """Negative values in stake percentages must be rejected."""
    path = temp_config_file(**{"stake.min_pct": -2})
    with pytest.raises(ValidationError):
        load_config(path)


def test_daily_loss_limit_null_accepted(temp_config_file):
    """daily_loss_limit_pct can be null."""
    path = temp_config_file(**{"stake.daily_loss_limit_pct": None})
    cfg = load_config(path)
    assert cfg.stake.daily_loss_limit_pct is None


def test_daily_loss_limit_valid_accepted(temp_config_file):
    """daily_loss_limit_pct with valid positive value loads OK."""
    path = temp_config_file(**{"stake.daily_loss_limit_pct": 10.0})
    cfg = load_config(path)
    assert cfg.stake.daily_loss_limit_pct == 10.0


def test_daily_loss_limit_invalid_rejected(temp_config_file):
    """daily_loss_limit_pct <= 0 or > 100 must be rejected."""
    path = temp_config_file(**{"stake.daily_loss_limit_pct": -5.0})
    with pytest.raises(ValidationError, match="must be >0 and <=100"):
        load_config(path)

    path2 = temp_config_file(**{"stake.daily_loss_limit_pct": 105.0})
    with pytest.raises(ValidationError, match="must be >0 and <=100"):
        load_config(path2)


def test_window_lengths_non_positive_rejected(temp_config_file):
    """window_lengths with non-positive values must be rejected."""
    path = temp_config_file(**{"patterns.window_lengths": [30, -10, 60]})
    with pytest.raises(ValidationError, match="window_lengths contains non-positive value"):
        load_config(path)

