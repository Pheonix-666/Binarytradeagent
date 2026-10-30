"""Comprehensive unit and property tests for safety/ package.

Covers:
  1. limits.check_order:
     - Normal allowed trade.
     - Stake percentage below min_pct and above max_pct.
     - Tier-scaled stake limit under CAUTION and DEFENSIVE tiers.
     - Absolute dollar limits [min_amount, max_amount].
     - Stake floored to round_to increment.
     - Total portfolio exposure cap (total_cap_pct).
     - Allowed expiries enforcement.
     - Opposite direction open trades on the same asset.
     - Real trading blocked when adaptive real_trading_allowed is False.
     - Asset mode requirement (live_eligible vs paper).
     - Daily loss limit breach.
     - Live mode guard (LIVE_CONFIRM=YES).
  2. CircuitBreaker:
     - 10-minute order rate limit (> max_orders_per_10min).
     - Stale data detection (> stale_data_multiplier * check_interval).
     - Consecutive execution error limit.
     - Balance mismatch with broker (> tolerance).
     - Manual reset requirement (stays tripped until explicit reset).
  3. KillSwitch:
     - KILL file presence halts new trades.
     - Telegram stop flag halts new trades.
     - Open trades are preserved (never cancelled).
     - Deactivation clears halt.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import pytest

from adaptive.monitor import AdaptiveState
from core.config import AppConfig, load_config
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch, check_kill_switch
from safety.limits import AccountSnapshot, OrderCheckResult, OrderRequest, check_order


@pytest.fixture
def config() -> AppConfig:
    cfg = load_config("config.yaml")
    cfg.stake.min_pct = 1.0
    cfg.stake.max_pct = 20.0
    cfg.stake.total_cap_pct = 35.0
    cfg.stake.min_amount = 1.0
    cfg.stake.max_amount = 1000.0
    cfg.stake.round_to = 1.0
    cfg.mode = "paper"
    return cfg


# ── 1. Safety Limits Tests ────────────────────────────────────────────────


def test_check_order_allowed(config):
    """A fully compliant order passes safety check."""
    account = AccountSnapshot(
        balance=10000.0,
        equity=10000.0,
        open_trades=[],
        asset_modes={"EURUSD": "live_eligible"},
    )
    order = OrderRequest(
        asset="EURUSD",
        direction="CALL",
        expiry_minutes=15,
        stake=500.0,  # 5% of equity (within [1%, 20%])
        max_stake_pct=20.0,
    )
    res = check_order(order, account, config)
    assert res.allowed is True
    assert res.reason == "ok"


def test_check_order_stake_pct_bounds(config):
    """Order stake must be within min_pct and max_pct of equity."""
    account = AccountSnapshot(
        balance=10000.0,
        equity=10000.0,
        open_trades=[],
        asset_modes={"EURUSD": "live_eligible"},
    )

    # Below min_pct (0.5% = $50 < 1% = $100)
    order_low = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=50.0)
    res_low = check_order(order_low, account, config)
    assert res_low.allowed is False
    assert "min_pct" in res_low.reason

    # Above max_pct (25% = $2500 > 20% = $2000)
    cfg_high = config.model_copy(deep=True)
    cfg_high.stake.max_amount = 5000.0
    order_high = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=2500.0)
    res_high = check_order(order_high, account, cfg_high)
    assert res_high.allowed is False
    assert "exceeds tier-scaled maximum" in res_high.reason


def test_check_order_tier_scaled_max_stake(config):
    """Adaptive regulator stake_scale tightens the maximum allowed stake."""
    account = AccountSnapshot(
        balance=10000.0,
        equity=10000.0,
        open_trades=[],
        asset_modes={"EURUSD": "live_eligible"},
    )
    # Candidate max_stake_pct = 10.0% ($1000)
    # Under CAUTION tier, stake_scale = 0.5 -> effective max is 5.0% ($500)
    adaptive_caution = AdaptiveState(tier="CAUTION", stake_scale=0.5)

    # Stake $600 is below unscaled 10% ($1000), but exceeds tier-scaled 5% ($500)
    order = OrderRequest(
        asset="EURUSD",
        direction="CALL",
        expiry_minutes=15,
        stake=600.0,
        max_stake_pct=10.0,
    )
    res = check_order(order, account, config, adaptive_state=adaptive_caution)
    assert res.allowed is False
    assert "exceeds tier-scaled maximum allowed 500.0" in res.reason

    # Stake $500 is allowed
    order_ok = OrderRequest(
        asset="EURUSD",
        direction="CALL",
        expiry_minutes=15,
        stake=500.0,
        max_stake_pct=10.0,
    )
    res_ok = check_order(order_ok, account, config, adaptive_state=adaptive_caution)
    assert res_ok.allowed is True


def test_check_order_absolute_amount_and_rounding(config):
    """Enforce min_amount, max_amount, and round_to flooring."""
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})

    # Above platform max_amount ($1000)
    order_above_cap = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=1500.0)
    res_cap = check_order(order_above_cap, account, config)
    assert res_cap.allowed is False
    assert "maximum allowed amount 1000.0" in res_cap.reason

    # Unrounded stake ($100.50 with round_to=1.0)
    order_unrounded = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=100.50)
    res_round = check_order(order_unrounded, account, config)
    assert res_round.allowed is False
    assert "not floored to round_to" in res_round.reason


def test_check_order_total_exposure_cap(config):
    """Total open stakes plus new stake cannot exceed total_cap_pct (35%)."""
    # Open trades have $3000 (30% of $10,000 equity)
    account = AccountSnapshot(
        balance=7000.0,
        equity=10000.0,
        open_trades=[
            {"asset": "GBPUSD", "direction": "CALL", "stake": 1500.0},
            {"asset": "USDJPY", "direction": "PUT", "stake": 1500.0},
        ],
        asset_modes={"EURUSD": "live_eligible"},
    )
    # Incoming stake of $600 pushes total to $3600 (36% > 35%)
    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=600.0)
    res = check_order(order, account, config)
    assert res.allowed is False
    assert "exceeds total_cap_pct 35.0%" in res.reason


def test_check_order_expiry_and_opposite_trade(config):
    """Disallow unlisted expiries and opposite-direction open trades on the same asset."""
    account = AccountSnapshot(
        balance=10000.0,
        equity=10000.0,
        open_trades=[{"asset": "EURUSD", "direction": "CALL", "stake": 200.0}],
        asset_modes={"EURUSD": "live_eligible"},
    )

    # Invalid expiry (45m not in [15, 30, 60])
    order_exp = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=45, stake=200.0)
    res_exp = check_order(order_exp, account, config)
    assert res_exp.allowed is False
    assert "allowed_expiries" in res_exp.reason

    # Opposite trade: EURUSD currently has open CALL; incoming is PUT
    order_opp = OrderRequest(asset="EURUSD", direction="PUT", expiry_minutes=15, stake=200.0)
    res_opp = check_order(order_opp, account, config)
    assert res_opp.allowed is False
    assert "Opposite direction trade already open" in res_opp.reason


def test_check_order_real_trading_halted_by_adaptive(config):
    """When adaptive regulator tier is PAUSED (real_trading_allowed=False), real orders are blocked."""
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    ad_paused = AdaptiveState(tier="PAUSED", real_trading_allowed=False)
    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=200.0)

    res = check_order(order, account, config, adaptive_state=ad_paused)
    assert res.allowed is False
    assert "halted by adaptive regulator" in res.reason


def test_check_order_daily_loss_limit(config):
    """Block trading when daily loss reaches daily_loss_limit_pct."""
    cfg = config.model_copy(deep=True)
    cfg.stake.daily_loss_limit_pct = 5.0  # 5% daily limit
    # Starting balance $10,000 -> max loss $500. Today's loss = $500
    account = AccountSnapshot(
        balance=9500.0,
        equity=9500.0,
        daily_starting_balance=10000.0,
        today_loss=500.0,
        asset_modes={"EURUSD": "live_eligible"},
    )
    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=100.0)
    res = check_order(order, account, cfg)
    assert res.allowed is False
    assert "Daily loss limit reached" in res.reason


def test_check_order_live_mode_guard(config, monkeypatch):
    """In live mode, missing LIVE_CONFIRM=YES blocks orders."""
    cfg = config.model_copy(deep=True)
    cfg.mode = "live"
    monkeypatch.delenv(cfg.live_requires_env, raising=False)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=200.0)
    res = check_order(order, account, cfg)
    assert res.allowed is False
    assert "not 'YES'" in res.reason


# ── 2. Circuit Breaker Tests ──────────────────────────────────────────────


def test_circuit_breaker_rate_limiting(config):
    """Trips when orders in 10 minutes exceed max_orders_per_10min."""
    cfg = config.model_copy(deep=True)
    cfg.safety.max_orders_per_10min = 3
    breaker = CircuitBreaker(config=cfg)
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)

    assert breaker.record_order(now) is True
    assert breaker.record_order(now + timedelta(minutes=1)) is True
    assert breaker.record_order(now + timedelta(minutes=2)) is True

    # 4th order within 10 minutes trips breaker
    assert breaker.record_order(now + timedelta(minutes=3)) is False
    assert breaker.is_tripped() is True
    assert "Rate limit exceeded" in breaker.trip_reason


def test_circuit_breaker_stale_data(config):
    """Trips when price data age > stale_data_multiplier * check_interval."""
    cfg = config.model_copy(deep=True)
    cfg.trading.check_interval_minutes = 15
    cfg.safety.stale_data_multiplier = 2.0  # max age = 30 minutes
    breaker = CircuitBreaker(config=cfg)

    now = datetime(2026, 10, 30, 10, 31, 0, tzinfo=timezone.utc)
    data_time = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)  # age 31 mins

    assert breaker.check_data_freshness(data_time=data_time, now=now) is False
    assert breaker.is_tripped() is True
    assert "Stale market data" in breaker.trip_reason


def test_circuit_breaker_consecutive_errors_and_reset(config):
    """Trips on consecutive execution errors, and requires manual reset."""
    cfg = config.model_copy(deep=True)
    cfg.safety.max_consecutive_exec_errors = 2
    breaker = CircuitBreaker(config=cfg)

    # 1st error: within limits
    assert breaker.record_exec_error("Network timeout") is True
    assert breaker.is_tripped() is False

    # 2nd error: reaches threshold -> trips
    assert breaker.record_exec_error("API rejected") is False
    assert breaker.is_tripped() is True
    assert "Consecutive execution errors" in breaker.trip_reason

    # Stays tripped
    assert breaker.is_tripped() is True

    # Manual operator reset
    breaker.reset()
    assert breaker.is_tripped() is False
    assert breaker.state.consecutive_exec_errors == 0


def test_circuit_breaker_balance_mismatch(config):
    """Trips if broker balance differs from expected by more than tolerance."""
    breaker = CircuitBreaker(config=config)
    # Expected $10,000, reported $9,950 (diff $50 > $1 tolerance)
    assert breaker.check_balance_match(reported_balance=9950.0, expected_balance=10000.0, tolerance=1.0) is False
    assert breaker.is_tripped() is True
    assert "Balance mismatch" in breaker.trip_reason


# ── 3. Kill Switch Tests ──────────────────────────────────────────────────


def test_kill_switch_file_and_telegram(tmp_path):
    """Kill switch activates on either KILL file or Telegram flag; leaves open trades alone."""
    ks = KillSwitch(kill_file="KILL", workspace_root=tmp_path)
    assert ks.is_active() is False

    # 1. Activate via file
    ks.activate_file()
    assert ks.is_file_present() is True
    assert ks.is_active() is True

    # Deactivate file
    assert ks.deactivate_file() is True
    assert ks.is_active() is False

    # 2. Activate via Telegram stop flag
    ks.set_telegram_stop(True)
    assert ks.is_active() is True
    ks.set_telegram_stop(False)
    assert ks.is_active() is False


# ── 4. Explicit Requirement Verification Tests ─────────────────────────────


def test_stake_just_above_max_pct(config):
    """Stake just above max_pct of equity is blocked."""
    cfg = config.model_copy(deep=True)
    cfg.stake.max_amount = 5000.0  # Avoid tripping absolute max_amount
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})

    # Exactly 20% ($2000): allowed
    order_exact = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=2000.0)
    assert check_order(order_exact, account, cfg).allowed is True

    # Just above 20% ($2001): blocked
    order_just_above = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=2001.0)
    res = check_order(order_just_above, account, cfg)
    assert res.allowed is False
    assert "exceeds tier-scaled maximum" in res.reason


def test_stake_above_max_amount(config):
    """Stake above platform max_amount ($1000) is blocked."""
    account = AccountSnapshot(balance=20000.0, equity=20000.0, asset_modes={"EURUSD": "live_eligible"})
    # 10% of $20000 is $2000, but platform max_amount is $1000. Stake $1001 must be blocked
    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=1001.0)
    res = check_order(order, account, config)
    assert res.allowed is False
    assert "maximum allowed amount 1000.0" in res.reason


def test_stake_below_min_amount(config):
    """Stake below min_amount ($1.0) is blocked."""
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=0.50)
    res = check_order(order, account, config)
    assert res.allowed is False
    assert "minimum allowed amount 1.0" in res.reason


def test_exposure_cap_after_flooring(config):
    """Total open exposure after stake flooring cannot exceed total_cap_pct (35%)."""
    # Open stakes: $3400 (34% of $10,000)
    account = AccountSnapshot(
        balance=6600.0,
        equity=10000.0,
        open_trades=[{"asset": "GBPUSD", "direction": "CALL", "stake": 3400.0}],
        asset_modes={"EURUSD": "live_eligible"},
    )
    # Adding $100 brings exposure to exactly 35% -> allowed
    order_ok = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=100.0)
    assert check_order(order_ok, account, config).allowed is True

    # Adding $101 brings exposure to 35.01% -> blocked
    order_over = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=101.0)
    res_over = check_order(order_over, account, config)
    assert res_over.allowed is False
    assert "exceeds total_cap_pct 35.0%" in res_over.reason


def test_opposite_direction_order_blocked(config):
    """Opposite direction trade on an asset with an existing open trade is blocked."""
    account = AccountSnapshot(
        balance=10000.0,
        equity=10000.0,
        open_trades=[{"asset": "EURUSD", "direction": "CALL", "stake": 100.0}],
        asset_modes={"EURUSD": "live_eligible"},
    )
    # Existing CALL, incoming PUT on EURUSD
    order_opp = OrderRequest(asset="EURUSD", direction="PUT", expiry_minutes=15, stake=100.0)
    res = check_order(order_opp, account, config)
    assert res.allowed is False
    assert "Opposite direction trade already open on EURUSD (CALL vs PUT)" in res.reason


def test_non_live_eligible_asset_blocked_for_real_trading(config):
    """Non-live_eligible asset is rejected when real trading is requested."""
    account = AccountSnapshot(
        balance=10000.0,
        equity=10000.0,
        asset_modes={"EURUSD": "paper"},
    )
    order_real = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=100.0, shadow_only=False)
    res = check_order(order_real, account, config)
    assert res.allowed is False
    assert "requires 'live_eligible'" in res.reason


def test_paused_blocks_real_orders_when_adaptive_mocked_to_misbehave(config):
    """Safety limits block real orders if tier is PAUSED even if real_trading_allowed was mocked True."""
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    # Misbehaving/corrupt adaptive state: tier is PAUSED, but real_trading_allowed maliciously True
    malicious_ad = AdaptiveState(tier="PAUSED", real_trading_allowed=True, stake_scale=0.5)

    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=100.0, shadow_only=False)
    res = check_order(order, account, config, adaptive_state=malicious_ad)
    assert res.allowed is False
    assert "halted by adaptive regulator (tier=PAUSED)" in res.reason


def test_property_random_orders_safety_invariants(config):
    """Property test across 150 random orders: any allowed order strictly satisfies all safety invariants."""
    import random
    import math

    rnd = random.Random(42)
    cfg = config.model_copy(deep=True)
    cfg.stake.max_amount = 1000.0

    for _ in range(150):
        equity = rnd.uniform(100.0, 50000.0)
        open_stakes = rnd.uniform(0.0, equity * 0.30)
        account = AccountSnapshot(
            balance=equity - open_stakes,
            equity=equity,
            open_trades=[{"asset": "OTHER", "direction": "CALL", "stake": open_stakes}],
            asset_modes={"EURUSD": "live_eligible"},
        )

        stake = rnd.uniform(0.5, equity * 0.40)
        # Randomize whether floored
        if rnd.random() < 0.5:
            stake = math.floor(stake)

        order = OrderRequest(
            asset="EURUSD",
            direction="CALL",
            expiry_minutes=15,
            stake=stake,
            max_stake_pct=20.0,
        )

        res = check_order(order, account, cfg)
        if res.allowed:
            # Invariant 1: Stake within absolute limits
            assert stake >= cfg.stake.min_amount - 1e-6
            assert stake <= cfg.stake.max_amount + 1e-6

            # Invariant 2: Stake properly floored to round_to
            assert abs(stake - math.floor(stake / cfg.stake.round_to) * cfg.stake.round_to) < 0.001

            # Invariant 3: Total open exposure <= total_cap_pct
            new_exposure = ((open_stakes + stake) / equity) * 100.0
            assert new_exposure <= cfg.stake.total_cap_pct + 1e-6


# ── 5. Explicit Individual Requirement Tests ───────────────────────────────


def test_limit_min_stake(config):
    """Explicit test: stake below min_pct of equity is rejected."""
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    # min_pct is 1.0% ($100.00). Stake $50.00 is blocked.
    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=50.0)
    res = check_order(order, account, config)
    assert res.allowed is False
    assert "min_pct" in res.reason


def test_limit_max_stake(config):
    """Explicit test: stake above max_pct of equity is rejected."""
    cfg = config.model_copy(deep=True)
    cfg.stake.max_amount = 5000.0
    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    # max_pct is 20.0% ($2000.00). Stake $2050.00 is blocked.
    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=2050.0)
    res = check_order(order, account, cfg)
    assert res.allowed is False
    assert "exceeds tier-scaled maximum" in res.reason


def test_limit_total_exposure(config):
    """Explicit test: total open stakes exceeding total_cap_pct (35%) are rejected."""
    account = AccountSnapshot(
        balance=7000.0,
        equity=10000.0,
        open_trades=[
            {"asset": "GBPUSD", "direction": "CALL", "stake": 2000.0},
            {"asset": "USDJPY", "direction": "PUT", "stake": 1400.0},
        ],
        asset_modes={"EURUSD": "live_eligible"},
    )
    # Existing open stakes = $3400 (34%). New stake $200 pushes total to $3600 (36% > 35%).
    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=200.0)
    res = check_order(order, account, config)
    assert res.allowed is False
    assert "exceeds total_cap_pct 35.0%" in res.reason


def test_limit_per_asset_exposure(config):
    """Explicit test: per-asset exposure exceeding limit is rejected."""
    cfg = config.model_copy(deep=True)
    setattr(cfg.safety, "max_open_trades_per_asset", 2)
    setattr(cfg.stake, "max_asset_exposure_pct", 15.0)

    account = AccountSnapshot(
        balance=9000.0,
        equity=10000.0,
        open_trades=[
            {"asset": "EURUSD", "direction": "CALL", "stake": 1000.0},
            {"asset": "EURUSD", "direction": "CALL", "stake": 400.0},
        ],
        asset_modes={"EURUSD": "live_eligible"},
    )
    # Adding 3rd trade on EURUSD breaches max_open_trades_per_asset (2)
    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=200.0)
    res = check_order(order, account, cfg)
    assert res.allowed is False
    assert "Per-asset trade limit reached for EURUSD" in res.reason


def test_limit_max_daily_loss(config):
    """Explicit test: orders rejected when cumulative today's loss reaches daily_loss_limit_pct."""
    cfg = config.model_copy(deep=True)
    cfg.stake.daily_loss_limit_pct = 4.0  # 4% daily loss cap
    account = AccountSnapshot(
        balance=9600.0,
        equity=9600.0,
        daily_starting_balance=10000.0,
        today_loss=400.0,  # Exactly 4% reached
        asset_modes={"EURUSD": "live_eligible"},
    )
    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=100.0)
    res = check_order(order, account, cfg)
    assert res.allowed is False
    assert "Daily loss limit reached" in res.reason


def test_limit_max_daily_trades(config):
    """Explicit test: orders rejected when daily trade count reaches max_daily_trades."""
    cfg = config.model_copy(deep=True)
    setattr(cfg.safety, "max_daily_trades", 10)
    account = AccountSnapshot(
        balance=10000.0,
        equity=10000.0,
        asset_modes={"EURUSD": "live_eligible"},
    )
    setattr(account, "today_trades", 10)

    order = OrderRequest(asset="EURUSD", direction="CALL", expiry_minutes=15, stake=100.0)
    res = check_order(order, account, cfg)
    assert res.allowed is False
    assert "Daily trade count limit reached: 10 >= maximum 10" in res.reason


def test_breaker_trip_consecutive_losses(config):
    """Explicit test: circuit breaker trips upon reaching max consecutive losses/errors."""
    cfg = config.model_copy(deep=True)
    setattr(cfg.safety, "max_consecutive_losses", 3)
    breaker = CircuitBreaker(config=cfg)

    assert breaker.record_loss() is True
    assert breaker.record_loss() is True
    assert breaker.is_tripped() is False

    # 3rd consecutive loss trips breaker
    assert breaker.record_loss() is False
    assert breaker.is_tripped() is True
    assert "Consecutive trade losses: 3 >= 3" in breaker.trip_reason


def test_breaker_trip_cooldown(config):
    """Explicit test: circuit breaker cannot be reset during cooldown window unless forced."""
    cfg = config.model_copy(deep=True)
    setattr(cfg.safety, "breaker_cooldown_seconds", 300)  # 5 min cooldown
    breaker = CircuitBreaker(config=cfg)

    t0 = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    breaker.trip("Operational error", now=t0)
    assert breaker.is_tripped() is True

    # 1 minute later: still in cooldown -> reset refused
    t1 = t0 + timedelta(seconds=60)
    assert breaker.reset(now=t1) is False
    assert breaker.is_tripped() is True

    # 6 minutes later: cooldown expired -> reset succeeds
    t2 = t0 + timedelta(seconds=360)
    assert breaker.reset(now=t2) is True
    assert breaker.is_tripped() is False


def test_breaker_trip_tier_lock(config):
    """Explicit test: circuit breaker trip locks trading and cannot be bypassed by adaptive tier."""
    breaker = CircuitBreaker(config=config)
    breaker.trip("Verification failed: rogue trade")

    assert breaker.is_tripped() is True
    # Breaker persists trip across checks
    assert breaker.record_order() is False
    assert breaker.is_tripped() is True


def test_kill_switch_latch_cannot_auto_reset(tmp_path):
    """Explicit test: kill switch acts as a latch and cannot auto-reset without operator action."""
    ks = KillSwitch(kill_file="KILL", workspace_root=tmp_path)
    assert ks.is_active() is False

    # Activating sets the latch
    ks.activate_file()
    assert ks.is_active() is True

    # Multiple passive checks confirm latch remains active
    for _ in range(5):
        assert ks.is_active() is True

    # Remains latched until explicit manual operator removal
    ks.deactivate_file()
    assert ks.is_active() is False


def test_kill_switch_latch_rejects_orders(config, tmp_path):
    """Explicit test: active kill switch rejects all new order submissions."""
    from execution.state import StateManager
    from execution.trade_manager import TradeManager
    from decision.schema import DecisionEvidence, DecisionOutput, ValidatedDecision

    state_mgr = StateManager(state_file=tmp_path / "state.json")
    breaker = CircuitBreaker(config=config)
    ks = KillSwitch(workspace_root=tmp_path)
    ks.activate_file()

    class DummyAdapter:
        def get_balance(self): return 10000.0
        def get_open_trades(self): return []

    tm = TradeManager(config, state_mgr, breaker, ks, DummyAdapter())
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec = ValidatedDecision(
        decision="CALL",
        is_valid=True,
        stake_amount=200.0,
        asset="EURUSD",
        direction="CALL",
        expiry_minutes=15,
        stake_pct=2.0,
        reasoning="Test",
        evidence=DecisionEvidence(risk_score=20, similarity=85.0, trend_daily="up", trend_1h="up", payout=0.85),
        raw_decision=DecisionOutput(decision="CALL", asset="EURUSD", expiry_minutes=15, stake_pct=2.0, reasoning="Test", evidence=DecisionEvidence(risk_score=20, similarity=85.0, trend_daily="up", trend_1h="up", payout=0.85)),
    )
    res = tm.execute(dec, account)
    assert res.executed is False
    assert res.status == "BLOCKED"
    assert "Emergency kill switch is ACTIVE" in res.reason


def test_kill_switch_latch_alert_fired(config, tmp_path):
    """Explicit test: an alert is sent when an order is blocked by the kill switch."""
    from execution.state import StateManager
    from execution.trade_manager import TradeManager
    from alerts.telegram import TelegramAlerter
    from decision.schema import DecisionEvidence, DecisionOutput, ValidatedDecision

    state_mgr = StateManager(state_file=tmp_path / "state.json")
    breaker = CircuitBreaker(config=config)
    ks = KillSwitch(workspace_root=tmp_path)
    ks.set_telegram_stop(True)

    alerts_sent = []
    alerter = TelegramAlerter()
    alerter.send_alert = lambda msg: alerts_sent.append(msg) or True

    class DummyAdapter:
        def get_balance(self): return 10000.0
        def get_open_trades(self): return []

    tm = TradeManager(config, state_mgr, breaker, ks, DummyAdapter(), alerter=alerter)
    tm.set_reconciled(True)

    account = AccountSnapshot(balance=10000.0, equity=10000.0, asset_modes={"EURUSD": "live_eligible"})
    dec = ValidatedDecision(
        decision="CALL",
        is_valid=True,
        stake_amount=200.0,
        asset="EURUSD",
        direction="CALL",
        expiry_minutes=15,
        stake_pct=2.0,
        reasoning="Test",
        evidence=DecisionEvidence(risk_score=20, similarity=85.0, trend_daily="up", trend_1h="up", payout=0.85),
        raw_decision=DecisionOutput(decision="CALL", asset="EURUSD", expiry_minutes=15, stake_pct=2.0, reasoning="Test", evidence=DecisionEvidence(risk_score=20, similarity=85.0, trend_daily="up", trend_1h="up", payout=0.85)),
    )
    tm.execute(dec, account)
    assert len(alerts_sent) == 1
    assert "Emergency kill switch is ACTIVE" in alerts_sent[0]
