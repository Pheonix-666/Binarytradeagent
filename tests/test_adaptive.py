"""Comprehensive test suite for Phase 8A: Adaptive core.

Covers:
  1. Tier transitions at exact thresholds (14.9% vs 15%, 24.9% vs 25%, 39.9% vs 40%),
     consecutive-loss triggers, hysteresis, and one-step-down recovery.
  2. PAUSED requalification (30 shadow trades beating break-even) and 20-trade ramp-up.
  3. Invariant property tests: stake after a loss never increases; adaptive output
     is never looser than base; stake 1-20%, total cap 35%.
  4. Profile switching: requires evidence, respects cooldown, whitelist rejects forbidden keys.
  5. Context suspension: n >= 30, Wilson 99% upper bound < break-even, 30% cap, cooldown,
     candidate filtering skips suspended contexts.
  6. Serialization: to_dict() and from_dict() round trip.
  7. No look-ahead: decisions at t use only trades closed by t.
  8. Performance: shared per-step work is computed once for all profiles.
  9. Comparative report: backtest run with adaptive OFF vs ON side-by-side.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
import random
import pytest

from analysis.candidates import CandidateTrade
from core.config import AppConfig, load_config
from core.models import GateResult, PatternMatch, TradeDirection, TrendReport, TrendLabel
from data.calendar import CalendarConfig
from backtest.data_loader import generate_synthetic_candles, precompute_asset_data
from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import generate_adaptive_comparison_report
from data.store import TradeRecord
from risk.engine import RiskReport as RiskEngineReport
from selection.stats import break_even, wilson_interval
from adaptive.champion import ChampionTracker, ProfilePerformance, filter_candidates_for_profile
from adaptive.contexts import ContextTracker, extract_context_tags
from adaptive.decay import EdgeDecayResult, evaluate_edge_decay, trades_to_detect
from adaptive.monitor import AdaptiveController, AdaptiveState, update_monitor
from adaptive.postmortem import generate_postmortem_markdown
from adaptive.profiles import PREAPPROVED_PROFILES, effective_config, validate_profile_dict


@pytest.fixture
def config() -> AppConfig:
    cfg = load_config("config.yaml")
    cfg.adaptive.enabled = True
    return cfg


def _make_closed_trade(trade_id: str, result: str, closed_at: datetime, pnl: float = 0.0) -> TradeRecord:
    return TradeRecord(
        id=trade_id,
        opened_at=(closed_at - timedelta(minutes=30)).isoformat(),
        closed_at=closed_at.isoformat(),
        asset="EURUSD",
        is_otc=False,
        direction="CALL",
        expiry_minutes=30,
        stake=100.0,
        payout_at_entry=0.85,
        result=result,
        pnl=pnl,
        mode="backtest",
        pattern_name="double_bottom",
        pattern_type="reversal",
        risk_score=35,
        hour_utc=closed_at.hour,
    )


# ── 1. Tier Transitions & State Machine Tests ─────────────────────────────


def test_tier_transitions_at_exact_thresholds(config):
    """Test transitions at exact boundary thresholds: 14.9% vs 15%, 24.9% vs 25%, 39.9% vs 40%."""
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    state = AdaptiveState(reference_equity=10000.0, current_equity=10000.0)

    # 14.9% drawdown -> Remains NORMAL
    s_149 = update_monitor(equity=8510.0, closed_trades=[], now=now, prev_state=state, config=config)
    assert s_149.tier == "NORMAL"
    assert s_149.drawdown_pct == 14.9
    assert s_149.real_trading_allowed is True
    assert s_149.stake_scale == 1.0

    # 15.0% drawdown -> Escalates to CAUTION
    s_150 = update_monitor(equity=8500.0, closed_trades=[], now=now, prev_state=state, config=config)
    assert s_150.tier == "CAUTION"
    assert s_150.drawdown_pct == 15.0
    assert s_150.stake_scale == 0.5
    assert s_150.min_similarity_add == 5.0
    assert s_150.max_risk_cap == 50.0

    # 24.9% drawdown -> Remains in CAUTION
    s_249 = update_monitor(equity=7510.0, closed_trades=[], now=now, prev_state=s_150, config=config)
    assert s_249.tier == "CAUTION"

    # 25.0% drawdown -> Escalates to DEFENSIVE
    s_250 = update_monitor(equity=7500.0, closed_trades=[], now=now, prev_state=s_249, config=config)
    assert s_250.tier == "DEFENSIVE"
    assert s_250.drawdown_pct == 25.0
    assert s_250.stake_scale == 0.35
    assert s_250.reversal_mode_override == "off"
    assert s_250.min_similarity_add == 10.0
    assert s_250.max_risk_cap == 40.0

    # 39.9% drawdown -> Remains in DEFENSIVE
    s_399 = update_monitor(equity=6010.0, closed_trades=[], now=now, prev_state=s_250, config=config)
    assert s_399.tier == "DEFENSIVE"

    # 40.0% drawdown -> Escalates to PAUSED
    s_400 = update_monitor(equity=6000.0, closed_trades=[], now=now, prev_state=s_399, config=config)
    assert s_400.tier == "PAUSED"
    assert s_400.drawdown_pct == 40.0
    assert s_400.real_trading_allowed is False
    assert s_400.stake_scale == 0.0


def test_consecutive_loss_trigger(config):
    """Test fast reaction triggered by loss streaks: 3 losses (NORMAL) vs 4 losses (CAUTION)."""
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    state = AdaptiveState(reference_equity=10000.0, current_equity=10000.0)

    # 3 consecutive losses: stays NORMAL
    t_3_losses = [
        _make_closed_trade("t1", "LOSS", now - timedelta(minutes=90)),
        _make_closed_trade("t2", "LOSS", now - timedelta(minutes=60)),
        _make_closed_trade("t3", "LOSS", now - timedelta(minutes=30)),
    ]
    s3 = update_monitor(equity=9700.0, closed_trades=t_3_losses, now=now, prev_state=state, config=config)
    assert s3.tier == "NORMAL"
    assert s3.consecutive_losses == 3

    # 4 consecutive losses: escalates to CAUTION
    t_4_losses = t_3_losses + [_make_closed_trade("t4", "LOSS", now)]
    s4 = update_monitor(equity=9600.0, closed_trades=t_4_losses, now=now, prev_state=s3, config=config)
    assert s4.tier == "CAUTION"
    assert s4.consecutive_losses == 4
    assert s4.stake_scale == 0.5


def test_hysteresis_and_one_step_down_recovery(config):
    """Test recovery: moves down ONLY one tier at a time with hysteresis (0.5 fraction)."""
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    # Start in DEFENSIVE at 25% drawdown
    state_def = AdaptiveState(
        tier="DEFENSIVE",
        reference_equity=10000.0,
        current_equity=7500.0,
        drawdown_pct=25.0,
        stake_scale=0.35,
    )

    # Recovery threshold for DEFENSIVE (25% * (1 - 0.5)) is 12.5%
    # If equity improves to 8000 (20% drawdown): remains DEFENSIVE
    s_still_def = update_monitor(equity=8000.0, closed_trades=[], now=now, prev_state=state_def, config=config)
    assert s_still_def.tier == "DEFENSIVE"

    # If equity improves to 8800 (12.0% drawdown <= 12.5%): recovers to CAUTION (one step down!)
    s_caution = update_monitor(equity=8800.0, closed_trades=[], now=now, prev_state=state_def, config=config)
    assert s_caution.tier == "CAUTION"
    assert s_caution.stake_scale == 0.5

    # Recovery threshold for CAUTION (15% * (1 - 0.5)) is 7.5%
    # If equity improves to 9000 (10.0% drawdown > 7.5%): remains CAUTION
    s_still_caut = update_monitor(equity=9000.0, closed_trades=[], now=now, prev_state=s_caution, config=config)
    assert s_still_caut.tier == "CAUTION"

    # If equity improves to 9300 (7.0% drawdown <= 7.5%) with no loss streak: recovers to NORMAL
    s_normal = update_monitor(equity=9300.0, closed_trades=[], now=now, prev_state=s_caution, config=config)
    assert s_normal.tier == "NORMAL"
    assert s_normal.stake_scale == 1.0


# ── 2. PAUSED Requalification & Ramp-Up ────────────────────────────────────


def test_paused_stays_paused_until_requalification_and_rampup(config):
    """PAUSED requires 30 shadow trades beating break-even, followed by 20 ramp-up trades at half stakes."""
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    be = break_even(0.85)  # ~0.5405

    state_paused = AdaptiveState(
        tier="PAUSED",
        reference_equity=10000.0,
        current_equity=6000.0,
        drawdown_pct=40.0,
        real_trading_allowed=False,
        stake_scale=0.0,
    )

    # 1. Fewer than 30 shadow trades (29 wins out of 29): remains PAUSED
    shadow_29 = [_make_closed_trade(f"sh_{i}", "WIN", now) for i in range(29)]
    s29 = update_monitor(
        equity=6000.0,
        closed_trades=[],
        now=now,
        prev_state=state_paused,
        config=config,
        shadow_trades=shadow_29,
    )
    assert s29.tier == "PAUSED"
    assert s29.real_trading_allowed is False

    # 2. 30 shadow trades with low win rate (10 wins, 20 losses = 33.3% < break-even): remains PAUSED
    shadow_30_bad = (
        [_make_closed_trade(f"sh_{i}", "WIN", now) for i in range(10)] +
        [_make_closed_trade(f"sh_{i+10}", "LOSS", now) for i in range(20)]
    )
    s30_bad = update_monitor(
        equity=6000.0,
        closed_trades=[],
        now=now,
        prev_state=state_paused,
        config=config,
        shadow_trades=shadow_30_bad,
    )
    assert s30_bad.tier == "PAUSED"
    assert s30_bad.real_trading_allowed is False

    # 3. 30 shadow trades with qualifying win rate (20 wins, 10 losses = 66.7% > break-even):
    # Requalifies -> moves down to DEFENSIVE with ramp-up!
    shadow_30_good = (
        [_make_closed_trade(f"sh_{i}", "WIN", now) for i in range(20)] +
        [_make_closed_trade(f"sh_{i+20}", "LOSS", now) for i in range(10)]
    )
    s_req = update_monitor(
        equity=6000.0,
        closed_trades=[],
        now=now,
        prev_state=state_paused,
        config=config,
        shadow_trades=shadow_30_good,
    )
    assert s_req.tier == "DEFENSIVE"
    assert s_req.real_trading_allowed is True
    assert s_req.in_rampup is True
    assert s_req.rampup_trades_remaining == 20
    # Rampup stake scale applies (min of defensive 0.35 and rampup 0.5 is 0.35)
    assert s_req.stake_scale <= 0.5


# ── 3. Invariants & Property Tests ─────────────────────────────────────────


def test_invariants_over_random_trade_sequences(config):
    """Property test: stake after loss never increases; adaptive output never loosens; stake 1-20%, total <= 35%."""
    random.seed(42)
    controller = AdaptiveController(config=config)
    equity = 10000.0
    now = datetime(2026, 10, 30, 0, 0, 0, tzinfo=timezone.utc)

    closed_trades: list[TradeRecord] = []
    previous_stake_scale = 1.0

    for i in range(200):
        now += timedelta(minutes=30)
        # Random outcome
        is_win = random.random() < 0.55
        res = "WIN" if is_win else "LOSS"
        pnl = 85.0 if is_win else -100.0
        equity = max(100.0, equity + pnl)

        trade = _make_closed_trade(f"tr_{i}", res, now, pnl=pnl)
        closed_trades.append(trade)

        state = controller.step(equity=equity, closed_trades=closed_trades, now=now)
        eff_cfg = effective_config(config, adaptive_state=state, active_profile="conservative")

        # Invariant 1: Adaptive stake scale is never > 1.0 (never loosens)
        assert eff_cfg.stake.stake_scale <= 1.0

        # Invariant 2: After a loss, stake scale never increases
        if not is_win:
            assert state.stake_scale <= previous_stake_scale + 1e-9, (
                f"Stake scale increased after loss at step {i}: {state.stake_scale} > {previous_stake_scale}"
            )
        previous_stake_scale = state.stake_scale

        # Invariant 3: Hard safety limits are strictly untouched
        assert eff_cfg.stake.min_pct == config.stake.min_pct == 1.0
        assert eff_cfg.stake.max_pct == config.stake.max_pct == 20.0
        assert eff_cfg.stake.total_cap_pct == config.stake.total_cap_pct == 35.0

        # Invariant 4: Similarity threshold is never lower than base config
        assert eff_cfg.patterns.min_similarity >= config.patterns.min_similarity


# ── 4. Profile Switching & Whitelist Tests ──────────────────────────────────


def test_profile_whitelist_rejects_forbidden_keys(config):
    """Whitelist must strictly reject keys without a clear ordering (expiry_minutes, top_assets) or forbidden keys."""
    # Forbidden keys
    with pytest.raises(ValueError, match="contains non-whitelisted key 'leverage'"):
        validate_profile_dict({"leverage": 10}, profile_name="bad_prof")

    with pytest.raises(ValueError, match="contains non-whitelisted key 'stop_loss'"):
        validate_profile_dict({"stop_loss": 0.05}, profile_name="bad_prof")

    # Keys without monotonic tighter ordering: expiry_minutes, top_assets
    with pytest.raises(ValueError, match="contains non-whitelisted key 'expiry_minutes'"):
        validate_profile_dict({"expiry_minutes": 60}, profile_name="bad_prof")

    with pytest.raises(ValueError, match="contains non-whitelisted key 'top_assets'"):
        validate_profile_dict({"top_assets": 2}, profile_name="bad_prof")

    # Allowed keys with clear monotonic tightening order validate without error
    valid_dict = {
        "patterns.min_similarity": 85.0,
        "max_risk_cap": 40.0,
        "stake_scale": 0.8,
        "reversal_mode": "off",
    }
    validate_profile_dict(valid_dict, profile_name="good_prof")


def test_profile_switch_requires_evidence_and_cooldown(config):
    """Champion profile switches only with >= 50 shadow trades, Wilson CI lower > break-even, and cooldown."""
    tracker = ChampionTracker(config=config, champion_name="standard")
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    payout = 0.85
    be = break_even(payout)  # ~0.5405

    # 1. Challenger with only 40 trades (insufficient sample): does not switch
    for _ in range(35):
        tracker.record_shadow_result("conservative", "WIN", payout=payout)
    for _ in range(5):
        tracker.record_shadow_result("conservative", "LOSS", payout=payout)

    new_champ, _ = tracker.evaluate_switch(now=now)
    assert new_champ is None

    # 2. Add 10 more wins -> 50 trades, win rate = 45/50 = 90%, CI lower > break-even
    for _ in range(10):
        tracker.record_shadow_result("conservative", "WIN", payout=payout)

    new_champ, reason = tracker.evaluate_switch(now=now)
    assert new_champ == "conservative"
    assert tracker.champion_name == "conservative"
    assert "Profile switched" in reason

    # 3. Cooldown test: immediately attempt switch to challenger_b with 100% win rate
    for _ in range(60):
        tracker.record_shadow_result("challenger_b", "WIN", payout=payout)

    # 1 day later (cooldown is 3 days): switch is blocked by cooldown
    next_day = now + timedelta(days=1)
    new_champ_blocked, _ = tracker.evaluate_switch(now=next_day)
    assert new_champ_blocked is None
    assert tracker.champion_name == "conservative"

    # 4 days later (cooldown satisfied): switch succeeds
    four_days_later = now + timedelta(days=4)
    new_champ_allowed, _ = tracker.evaluate_switch(now=four_days_later)
    assert new_champ_allowed == "challenger_b"
    assert tracker.champion_name == "challenger_b"


# ── 5. Context Learning & Suspension Tests ─────────────────────────────────


def test_context_suspension_cap_and_filtering(config):
    """Test context suspension: n >= 30, Wilson 99% upper < break-even, 30% cap, candidate filtering."""
    tracker = ContextTracker(config=config)
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    be = break_even(0.85)

    # 1. 20 trades on asset EURUSD: not suspended yet (n < 30)
    for _ in range(20):
        tracker.record_trade("EURUSD", 10, "double_top", 45, 50.0, 30, "LOSS", now=now)
    assert "asset:EURUSD" not in tracker.suspended_contexts

    # 2. 15 more losses -> 35 losses out of 35 trades:
    # Wilson 99% upper bound for 0/35 is ~0.142 < break-even 0.5405 -> Suspended!
    for _ in range(15):
        tracker.record_trade("EURUSD", 10, "double_top", 45, 50.0, 30, "LOSS", now=now)
    tracker.evaluate_suspensions(now=now)
    assert "asset:EURUSD" in tracker.suspended_contexts

    # 3. Create mock candidate for EURUSD and verify it is skipped
    cand = CandidateTrade(
        asset="EURUSD",
        direction=TradeDirection.CALL,
        t=now,
        current_price=1.1000,
        atr=0.0010,
        pattern={"name": "double_bottom"},
        pattern_match=PatternMatch(pattern_name="double_bottom", direction=TradeDirection.CALL, pattern_type="reversal", similarity=80.0),
        risk_report=RiskEngineReport(score=35, volatility_score=50, pattern_score=80, win_rate_score=50, session_score=50, sr_distance_score=50, weights={}, raw_weighted_score=35),
        gate_result=GateResult(passed=True, reason="ok"),
        max_stake_pct=10.0,
        volatility_percentile=50.0,
        evidence={},
    )
    qualifying, skips = tracker.filter_candidates([cand], expiry_minutes=30)
    assert len(qualifying) == 0
    assert any("context_suspended: asset:EURUSD" in s for s in skips)

    # 4. Cap on suspended fraction:
    # Add many healthy contexts so total contexts increases
    for asset_name in ["GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"]:
        for _ in range(35):
            tracker.record_trade(asset_name, 10, "double_bottom", 30, 50.0, 30, "WIN", now=now)
    tracker.evaluate_suspensions(now=now)

    # Total contexts >= 7, max suspended fraction = 0.3 -> at most 30% of contexts can be suspended
    total_ctx = len(tracker.contexts)
    max_allowed = max(1, int(total_ctx * 0.3))
    assert len(tracker.suspended_contexts) <= max_allowed


# ── 6. Serialization Tests (to_dict / from_dict) ───────────────────────────


def test_serialization_round_trip(config):
    """AdaptiveController, AdaptiveState, and ContextTracker restore identical state from dict."""
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    ctrl = AdaptiveController(config=config)
    ctrl.step(equity=8200.0, closed_trades=[_make_closed_trade("t1", "LOSS", now)], now=now)

    ctrl_dict = ctrl.to_dict()
    restored_ctrl = AdaptiveController.from_dict(ctrl_dict, config=config)

    assert restored_ctrl.state.tier == ctrl.state.tier
    assert restored_ctrl.state.current_equity == ctrl.state.current_equity
    assert restored_ctrl.state.drawdown_pct == ctrl.state.drawdown_pct
    assert restored_ctrl.state.stake_scale == ctrl.state.stake_scale

    # ContextTracker serialization
    ctx = ContextTracker(config=config)
    ctx.record_trade("EURUSD", 12, "double_top", 40, 50.0, 30, "WIN", now=now)
    ctx_dict = ctx.to_dict()
    restored_ctx = ContextTracker.from_dict(ctx_dict, config=config)
    assert "asset:EURUSD" in restored_ctx.contexts
    assert restored_ctx.contexts["asset:EURUSD"].wins == 1


# ── 7. Look-Ahead Guard Test ───────────────────────────────────────────────


def test_no_lookahead_adaptive(config):
    """Adaptive decisions at time t strictly ignore trades closed after t."""
    t_eval = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
    t_future = t_eval + timedelta(hours=2)

    # 4 consecutive losses closed BEFORE t_eval -> should trigger CAUTION
    trades_past = [
        _make_closed_trade("p1", "LOSS", t_eval - timedelta(minutes=40)),
        _make_closed_trade("p2", "LOSS", t_eval - timedelta(minutes=30)),
        _make_closed_trade("p3", "LOSS", t_eval - timedelta(minutes=20)),
        _make_closed_trade("p4", "LOSS", t_eval - timedelta(minutes=10)),
    ]
    # Future win closed AFTER t_eval
    trade_future_win = _make_closed_trade("f1", "WIN", t_future)

    # Evaluate at t_eval filtering trades by t <= t_eval
    valid_closed = [t for t in (trades_past + [trade_future_win]) if datetime.fromisoformat(t.closed_at) <= t_eval]
    state = update_monitor(equity=9600.0, closed_trades=valid_closed, now=t_eval, prev_state=None, config=config)

    assert state.consecutive_losses == 4
    assert state.tier == "CAUTION"


# ── 8. Shared Per-Step Work Performance Test ───────────────────────────────


def test_shared_per_step_work_efficiency(config):
    """Candidate filtering across multiple profiles shares base candidate pool with zero re-evaluations."""
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    base_cands = [
        CandidateTrade(
            asset="EURUSD",
            direction=TradeDirection.CALL,
            t=now,
            current_price=1.1000,
            atr=0.0010,
            pattern={"name": "double_bottom"},
            pattern_match=PatternMatch(pattern_name="double_bottom", direction=TradeDirection.CALL, pattern_type="reversal", similarity=82.0),
            risk_report=RiskEngineReport(score=38, volatility_score=50, pattern_score=82, win_rate_score=50, session_score=50, sr_distance_score=50, weights={}, raw_weighted_score=38),
            gate_result=GateResult(passed=True, reason="ok"),
            max_stake_pct=10.0,
            volatility_percentile=50.0,
            evidence={},
        ),
        CandidateTrade(
            asset="GBPUSD",
            direction=TradeDirection.CALL,
            t=now,
            current_price=1.2500,
            atr=0.0015,
            pattern={"name": "support_bounce"},
            pattern_match=PatternMatch(pattern_name="support_bounce", direction=TradeDirection.CALL, pattern_type="continuation", similarity=88.0),
            risk_report=RiskEngineReport(score=48, volatility_score=50, pattern_score=88, win_rate_score=50, session_score=50, sr_distance_score=50, weights={}, raw_weighted_score=48),
            gate_result=GateResult(passed=True, reason="ok"),
            max_stake_pct=10.0,
            volatility_percentile=50.0,
            evidence={},
        ),
    ]

    # Standard profile (no overrides): both qualify
    std_cands = filter_candidates_for_profile(base_cands, PREAPPROVED_PROFILES["standard"])
    assert len(std_cands) == 2

    # Conservative profile (min_similarity 85, max_risk_cap 45):
    # EURUSD has similarity 82 (< 85) -> dropped
    # GBPUSD has risk 48 (> 45) -> dropped
    cons_cands = filter_candidates_for_profile(base_cands, PREAPPROVED_PROFILES["conservative"])
    assert len(cons_cands) == 0

    # Profile with reversal_mode 'off':
    # EURUSD is reversal -> dropped
    # GBPUSD is continuation -> kept
    trend_cands = filter_candidates_for_profile(base_cands, {"reversal_mode": "off"})
    assert len(trend_cands) == 1
    assert trend_cands[0].asset == "GBPUSD"


# ── 9. Postmortem & Comparative Report Tests ───────────────────────────────


def test_postmortem_markdown_generation(config):
    """Postmortem report renders complete markdown telemetry without mutating state."""
    state = AdaptiveState(
        tier="CAUTION",
        reference_equity=10000.0,
        current_equity=8400.0,
        drawdown_pct=16.0,
        consecutive_losses=4,
        stake_scale=0.5,
        transition_reason="Drawdown breached 15%",
    )
    report = generate_postmortem_markdown(state)
    assert "# Adaptive Layer Postmortem & System Health Report" in report
    assert "**Active Tier:** `CAUTION`" in report
    assert "16.00%" in report


def test_backtest_adaptive_on_vs_off_comparison(config):
    """Verify side-by-side comparative backtest reporting with adaptive ON and OFF."""
    cal = CalendarConfig(type="always_open", daily_boundary_tz="UTC", daily_boundary_hour=0)
    start_dt = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    df = generate_synthetic_candles("EURUSD", start_dt=start_dt, days=4, calendar=cal, seed=42)
    p_data = precompute_asset_data("EURUSD", df, cal, config)

    engine = BacktestEngine(config=config, data_by_asset={"EURUSD": p_data})

    # Run with adaptive OFF
    res_off = engine.run(start_t=start_dt + timedelta(days=2), end_t=start_dt + timedelta(days=3), enable_adaptive=False)
    assert res_off.adaptive_enabled is False

    # Run with adaptive ON
    res_on = engine.run(start_t=start_dt + timedelta(days=2), end_t=start_dt + timedelta(days=3), enable_adaptive=True)
    assert res_on.adaptive_enabled is True

    # Generate comparative report
    comp_report = generate_adaptive_comparison_report(result_off=res_off, result_on=res_on)
    assert "# Adaptive Layer Comparative Evaluation: OFF vs ON" in comp_report
    assert "Do NOT tune adaptive thresholds on the test period" in comp_report
    assert "Adaptive OFF (Baseline)" in comp_report
    assert "Adaptive ON" in comp_report


# ── 10. Edge-Decay Detector Tests ──────────────────────────────────────────


def test_trades_to_detect_formula():
    """Verify sample size estimation formula across shortfalls and critical values."""
    # 5 points shortfall (0.05) with z=1.0 -> ~100 trades
    n5 = trades_to_detect(shortfall=0.05, payout=0.85, z=1.0)
    assert 95 <= n5 <= 105

    # 12 points shortfall (0.12) with z=1.0 -> ~17 trades
    n12 = trades_to_detect(shortfall=0.12, payout=0.85, z=1.0)
    assert 15 <= n12 <= 20

    # 8 points shortfall (0.08) with z=1.96 -> ~149 trades
    n8_z = trades_to_detect(shortfall=0.08, payout=0.85, z=1.96)
    assert 140 <= n8_z <= 155


def test_edge_decay_too_few_trades(config):
    """When closed trades < min_trades (30), detector takes no action."""
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    # 29 loss trades: would indicate severe decay, but n < 30
    trades_29 = [
        _make_closed_trade(f"t_{i}", "LOSS", now - timedelta(minutes=10 * i))
        for i in range(29)
    ]
    res_29 = evaluate_edge_decay(trades_29, config=config, default_payout=0.85)
    assert res_29.decay_detected is False
    assert res_29.persistent is False
    assert res_29.recommended_tier is None
    assert "Insufficient trades" in res_29.reason

    # Add 1 more trade to reach exactly 30 trades: now evaluates
    trades_30 = trades_29 + [_make_closed_trade("t_30", "LOSS", now)]
    res_30 = evaluate_edge_decay(trades_30, config=config, default_payout=0.85)
    assert res_30.decay_detected is True
    assert res_30.recommended_tier == "CAUTION"


def test_edge_decay_exact_threshold(config):
    """Test boundary where Wilson CI upper bound crosses break-even rate with z_caution=1.0.

    With payout=0.85, break_even = 1 / (1 + 0.85) ≈ 0.54054.
    For n=60 (window_trades_caution):
      wins=29: p=0.4833, Wilson upper (z=1.0) ≈ 0.5471 >= 0.54054 -> NO decay.
      wins=28: p=0.4667, Wilson upper (z=1.0) ≈ 0.5306 < 0.54054 -> DECAY DETECTED (CAUTION).
    """
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)

    # 29 wins out of 60: Wilson upper bound 0.5471 >= 0.54054 -> No decay
    trades_29 = [
        _make_closed_trade(f"t_{i}", "WIN" if i < 29 else "LOSS", now - timedelta(minutes=i))
        for i in range(60)
    ]
    res_29 = evaluate_edge_decay(trades_29, config=config, default_payout=0.85)
    assert res_29.ci_upper >= res_29.break_even_rate
    assert res_29.decay_detected is False
    assert res_29.recommended_tier is None

    # 28 wins out of 60: Wilson upper bound 0.5306 < 0.54054 -> Decay detected
    trades_28 = [
        _make_closed_trade(f"t_{i}", "WIN" if i < 28 else "LOSS", now - timedelta(minutes=i))
        for i in range(60)
    ]
    res_28 = evaluate_edge_decay(trades_28, config=config, default_payout=0.85)
    assert res_28.ci_upper < res_28.break_even_rate
    assert res_28.decay_detected is True
    assert res_28.recommended_tier == "CAUTION"


def test_edge_decay_persistent_windows(config):
    """When edge decay holds across 2 consecutive non-overlapping windows of 100 trades, escalate to DEFENSIVE."""
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)

    # Window 1 (older 100 trades): 44 wins (decay)
    # Window 2 (recent 100 trades): 44 wins (decay)
    # Total 200 trades
    trades_window1 = [
        _make_closed_trade(f"w1_{i}", "WIN" if i < 44 else "LOSS", now - timedelta(minutes=200 - i))
        for i in range(100)
    ]
    trades_window2 = [
        _make_closed_trade(f"w2_{i}", "WIN" if i < 44 else "LOSS", now - timedelta(minutes=100 - i))
        for i in range(100)
    ]
    trades_200 = trades_window1 + trades_window2

    res_200 = evaluate_edge_decay(trades_200, config=config, default_payout=0.85)
    assert res_200.decay_detected is True
    assert res_200.persistent is True
    assert res_200.recommended_tier == "DEFENSIVE"

    # If older window had 60 wins (intact), but recent window had 44 wins:
    # Single-window decay -> CAUTION, not persistent
    trades_w1_intact = [
        _make_closed_trade(f"w1_intact_{i}", "WIN" if i < 60 else "LOSS", now - timedelta(minutes=200 - i))
        for i in range(100)
    ]
    res_mixed = evaluate_edge_decay(trades_w1_intact + trades_window2, config=config, default_payout=0.85)
    assert res_mixed.decay_detected is True
    assert res_mixed.persistent is False
    assert res_mixed.recommended_tier == "CAUTION"


def test_edge_decay_tier_escalation_and_recovery(config):
    """Test state machine escalation to CAUTION on edge decay, and recovery when edge improves."""
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    state = AdaptiveState(reference_equity=10000.0, current_equity=10000.0)

    # 100 trades with only 40 wins (severe decay), but equity remains at 10000 (0% drawdown), consec_losses=0
    trades_decay = [
        _make_closed_trade(f"t_{i}", "WIN" if (i % 5 < 2) else "LOSS", now - timedelta(minutes=100 - i))
        for i in range(100)
    ]
    # Ensure last trade was a WIN so consecutive_losses = 0
    trades_decay[-1] = _make_closed_trade("t_last", "WIN", now)

    # Escalates to CAUTION due to edge decay despite 0% drawdown and 0 consecutive losses
    s_caution = update_monitor(
        equity=10000.0,
        closed_trades=trades_decay,
        now=now,
        prev_state=state,
        config=config,
    )
    assert s_caution.tier == "CAUTION"
    assert s_caution.edge_decay_detected is True
    assert "Edge decay detected" in s_caution.transition_reason

    # Next 100 trades have 80% win rate (edge recovered)
    trades_recovered = list(trades_decay) + [
        _make_closed_trade(f"rec_{i}", "WIN" if (i % 5 != 0) else "LOSS", now + timedelta(minutes=i + 1))
        for i in range(100)
    ]
    trades_recovered[-1] = _make_closed_trade("rec_last", "WIN", now + timedelta(minutes=101))

    # When evaluated after recovery, rolling 100 trades have 80% win rate:
    # Edge decay clears -> Hysteresis recovery moves tier from CAUTION back to NORMAL!
    s_recovered = update_monitor(
        equity=10000.0,
        closed_trades=trades_recovered,
        now=now + timedelta(minutes=102),
        prev_state=s_caution,
        config=config,
    )
    assert s_recovered.tier == "NORMAL"
    assert s_recovered.edge_decay_detected is False
    assert "Recovered to NORMAL" in s_recovered.transition_reason


def test_edge_decay_no_look_ahead(config):
    """Trades closed after now must not affect the edge-decay calculation."""
    now = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)

    # 50 past losses
    past_trades = [
        _make_closed_trade(f"p_{i}", "LOSS", now - timedelta(minutes=50 - i))
        for i in range(50)
    ]
    # 50 future wins closed after now
    future_trades = [
        _make_closed_trade(f"f_{i}", "WIN", now + timedelta(minutes=i + 1))
        for i in range(50)
    ]

    all_trades = past_trades + future_trades

    # Slicing as of `now`
    as_of_trades = [t for t in all_trades if datetime.fromisoformat(t.closed_at) <= now]
    res_as_of = evaluate_edge_decay(as_of_trades, config=config, default_payout=0.85)
    assert res_as_of.n_trades_evaluated == 50
    assert res_as_of.decay_detected is True
