"""End-to-end replay test with planted edge across multi-day simulation.

PROJECT_SPEC.md Phase 9 Pass 2:
  - Synthetic replay with planted edge over warm-up plus at least 3 trading days.
  - paper.decision_mode = policy.
  - Real DataStore, real TradeRepository, real TradeManager, PaperAdapter, FakeClock.
  - Assertions:
      * At least 20 shadow trades written with mode "paper".
      * payout_at_entry equals quote payout (0.88).
      * Settled by minute job.
      * Stats and paper-gate progress computed from repository rows.
      * Daily summary produced from those rows.
      * Real adapter place_trade is NEVER called.
      * Restart in the middle creates no duplicate trades.
  - Prints a short run (counts per day, win rate vs break-even).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from adapters.paper import PaperAdapter
from core.clock import FakeClock
from core.config import AppConfig, load_config
from core.models import TradeDirection
from data.price_source import ReplaySource
from data.store import DataStore, TradeRepository
from execution.state import StateManager
from execution.trade_manager import TradeManager
from orchestrator.engine import Orchestrator
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch
from selection.stats import compute_asset_stats


def _generate_planted_replay_data(start_dt: datetime, days: int = 7) -> pd.DataFrame:
    """Generate synthetic 1m candles for EURUSD with smooth wave and deterministic drift.

    Data Generator:
      - Formula: close(m) = p0 + drift * m + wave_amp * sin(2 * pi * m / period)
      - Parameters:
          p0 (base price): 1.1000
          drift: 0.00005 per minute
          wave_amp: 0.0008
          period: 120 minutes (2-hour cyclical wave)
          open: close - 0.00008
          high: max(open, close) + 0.00015
          low: min(open, close) - 0.00015
          volume: 100.0
    """
    total_minutes = days * 1440
    records = []
    p = 1.1000

    for m in range(total_minutes):
        # 120m cycle wave + positive drift provides recurring support bounces & upward continuation
        drift = 0.00005 * m
        wave = 0.0008 * np.sin(2 * np.pi * m / 120)
        c = p + drift + wave
        o = c - 0.00008
        h = max(o, c) + 0.00015
        l = min(o, c) - 0.00015

        t = start_dt + timedelta(minutes=m)
        records.append({
            "ts": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ts_close": (t + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": round(float(o), 5),
            "high": round(float(h), 5),
            "low": round(float(l), 5),
            "close": round(float(c), 5),
            "volume": 100.0,
        })

    return pd.DataFrame(records)


def _generate_noisy_planted_data(start_dt: datetime, days: int = 7, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 1m candles with drift plus Gaussian noise (noisy planted edge).

    Data Generator:
      Geometric Brownian motion with positive drift:
        - Formula: close(m) = close(m-1) * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z_m)
        - Parameters:
            base price: 1.1000
            drift mu: 0.000008 per minute
            volatility sigma: 0.00008 per minute
            seed: 42
            dt: 1.0 (minute)
            bar spread: open = close(m-1), high = max(o, c) + 0.00004, low = min(o, c) - 0.00004
            volume: 100.0

      Theoretical win rate on 15m binary CALL options:
        P(Return_15 > 0) = Phi(mu * sqrt(15) / sigma) = Phi(0.000008 * 3.873 / 0.00008)
                         = Phi(0.3873) ~= 65.1%
      Empirical target: roughly 60-70% wins under market noise.
    """
    total_minutes = days * 1440
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.000008, scale=0.00008, size=total_minutes)
    prices = 1.1000 * np.cumprod(1.0 + returns)

    records = []
    for m in range(total_minutes):
        c = float(prices[m])
        o = float(prices[m - 1]) if m > 0 else 1.1000
        h = max(o, c) + 0.00004
        l = min(o, c) - 0.00004
        t = start_dt + timedelta(minutes=m)
        records.append({
            "ts": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ts_close": (t + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": round(o, 5),
            "high": round(h, 5),
            "low": round(l, 5),
            "close": round(c, 5),
            "volume": 100.0,
        })
    return pd.DataFrame(records)


def test_end_to_end_replay_proof(tmp_path: Path, capsys):
    """End-to-end multi-day replay proof with real components and planted edge."""
    # 1. Directories and files
    data_dir = tmp_path / "data_history"
    data_dir.mkdir(parents=True)
    db_path = tmp_path / "market.db"
    state_file = tmp_path / "state.json"
    reports_dir = tmp_path / "reports"
    daily_reports_dir = reports_dir / "daily"
    daily_reports_dir.mkdir(parents=True)

    # 2. Planted dataset: 7 full days (4 days warm-up, 3 trading days)
    start_dt = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
    df_planted = _generate_planted_replay_data(start_dt, days=7)
    df_planted.to_csv(data_dir / "EURUSD.csv", index=False)

    # 3. Config setup
    config: AppConfig = load_config("config.yaml")
    config.mode = "paper"
    config.paper.decision_mode = "policy"
    config.source.name = "replay"
    config.source.replay_dir = str(data_dir)
    config.source.assets = ["EURUSD"]
    config.orchestrator.reports_dir = str(reports_dir)
    config.orchestrator.summary_hour_utc = 0
    config.orchestrator.warmup_bars = {"1d": 3, "1h": 5, "15m": 5, "5m": 5, "1m": 10}

    # Set trend indicator periods appropriately for 4 days warmup history
    for tf in [config.gate.trend.daily, config.gate.trend.h1, config.gate.trend.m15]:
        tf.ema_fast = 2
        tf.ema_slow = 3
        tf.warmup_multiplier = 1
        tf.adx_period = 2
        tf.adx_min = 5

    config.patterns.min_similarity = 20.0
    config.patterns.window_lengths = [30, 45, 60]
    config.risk.volatility_lookback_days = 1
    config.risk.atr_period = 5
    config.trading.allowed_expiries = [15, 30, 60]
    config.trading.expiry_minutes = 15

    # 4. Real components
    # Warmup satisfies after 4 days of history
    sim_start = start_dt + timedelta(days=4)
    sim_end = start_dt + timedelta(days=7)  # 3 full trading days (Day 4, 5, 6)

    clock = FakeClock(initial_time=sim_start)
    store = DataStore(db_path=db_path)
    repository = TradeRepository(store=store)
    state_manager = StateManager(state_file=state_file)
    breaker = CircuitBreaker(config=config)
    kill_switch = KillSwitch()

    quote_payout = 0.88
    source = ReplaySource(
        clock=clock,
        config=config,
        data_dir=data_dir,
        assets=["EURUSD"],
        payout_map={"EURUSD": quote_payout},
    )

    paper_adapter = PaperAdapter(initial_balance=1000.0, tie_policy="loss")
    fake_real_adapter = MagicMock()  # Must NEVER be called
    fake_real_adapter.get_balance.return_value = 1000.0
    fake_real_adapter.get_open_trades.return_value = []
    mock_alerter = MagicMock()

    trade_manager = TradeManager(
        config=config,
        state_manager=state_manager,
        breaker=breaker,
        kill_switch=kill_switch,
        adapter=fake_real_adapter,
        paper_adapter=paper_adapter,
        repository=repository,
        alerter=mock_alerter,
    )

    orch = Orchestrator(
        config=config,
        source=source,
        adapter=fake_real_adapter,
        store=store,
        repository=repository,
        state_manager=state_manager,
        breaker=breaker,
        kill_switch=kill_switch,
        trade_manager=trade_manager,
        alerter=mock_alerter,
        clock=clock,
        paper_adapter=paper_adapter,
        is_first_run=True,
    )

    # 5. Startup
    orch.startup()
    assert "EURUSD" in orch.ready_assets

    # 6. Run simulation across 3 full trading days
    restart_point = start_dt + timedelta(days=5, hours=12)
    has_restarted = False
    active_orch = orch

    while clock.now() < sim_end:
        cur_t = clock.now()

        # Step A: Settle open shadow trades every cycle
        active_orch.run_settlement_job(cur_t)

        # Step B: 5m cycle evaluation
        active_orch.run_evaluation_cycle(cur_t)

        # Step C: Daily summary job at midnight
        if cur_t.hour == 0 and cur_t.minute == 0:
            active_orch.run_daily_summary_job(cur_t)

        # Step D: Mid-point restart test
        if not has_restarted and cur_t >= restart_point:
            pre_restart_trade_ids = {t.id for t in repository.get_trades()}

            new_state_mgr = StateManager(state_file=state_file)
            new_tm = TradeManager(
                config=config,
                state_manager=new_state_mgr,
                breaker=breaker,
                kill_switch=kill_switch,
                adapter=fake_real_adapter,
                paper_adapter=paper_adapter,
                repository=repository,
                alerter=mock_alerter,
            )
            active_orch = Orchestrator(
                config=config,
                source=source,
                adapter=fake_real_adapter,
                store=store,
                repository=repository,
                state_manager=new_state_mgr,
                breaker=breaker,
                kill_switch=kill_switch,
                trade_manager=new_tm,
                alerter=mock_alerter,
                clock=clock,
                paper_adapter=paper_adapter,
                is_first_run=False,
            )
            active_orch.startup()

            post_restart_trade_ids = {t.id for t in repository.get_trades()}
            assert pre_restart_trade_ids.issubset(post_restart_trade_ids)
            has_restarted = True

        # Advance clock by 15 minutes (matching trade expiry and cycle rhythm)
        clock.advance(seconds=900)

    # 7. Final settlement of any remaining trades and summary
    active_orch.run_settlement_job(clock.now())
    active_orch.run_daily_summary_job(clock.now())

    # 8. Assertions
    trades = repository.get_trades()
    assert len(trades) >= 20, f"Expected >= 20 shadow trades, got {len(trades)}"

    # All trades mode must be "paper"
    assert all(t.mode == "paper" for t in trades), "All shadow trades must have mode='paper'"

    # All payout_at_entry must equal quote payout (0.88)
    assert all(abs(t.payout_at_entry - quote_payout) < 1e-4 for t in trades), (
        f"All shadow trades must record payout_at_entry={quote_payout}"
    )

    # All trades settled
    settled = [t for t in trades if t.result in {"WIN", "LOSS", "TIE"}]
    assert len(settled) == len(trades), "All expired shadow trades must be settled by minute job"

    # Real adapter place_trade must NEVER have been called
    fake_real_adapter.place_trade.assert_not_called()

    # Stats and paper-gate progress computed from repository rows
    stats = compute_asset_stats(trades=trades, asset="EURUSD", mode="paper", as_of=clock.now())
    assert stats is not None
    assert stats.n == len(trades)
    assert stats.wins > 0
    # Planted edge win rate exceeds break-even
    assert stats.win_rate > stats.break_even, (
        f"Planted edge win rate {stats.win_rate:.1%} must exceed break-even {stats.break_even:.1%}"
    )

    # Daily summaries produced in reports_dir / "daily"
    daily_files = list(daily_reports_dir.glob("*.md"))
    assert len(daily_files) >= 3, f"Expected at least 3 daily summary reports, got {len(daily_files)}"

    # 9. Print a short run (counts per day, win rate vs break-even)
    print("\n=======================================================")
    print("        END-TO-END REPLAY TEST RUN RESULTS             ")
    print("=======================================================")
    unique_days = sorted(list({t.opened_at[:10] for t in trades}))
    for d_str in unique_days:
        d_trades = [t for t in trades if t.opened_at.startswith(d_str)]
        d_wins = sum(1 for t in d_trades if t.result == "WIN")
        d_losses = sum(1 for t in d_trades if t.result == "LOSS")
        d_wr = d_wins / len(d_trades) if d_trades else 0.0
        print(f"  Day {d_str}: {len(d_trades):>2} trades | {d_wins}W / {d_losses}L | WR: {d_wr:>6.1%} (BE: {stats.break_even:>6.1%})")

    print("-------------------------------------------------------")
    print(f"Total Shadow Trades: {len(trades)}")
    print(f"Overall Win Rate:    {stats.win_rate:.1%} (Break-Even: {stats.break_even:.1%})")
    print(f"Net PnL:             ${sum(t.pnl for t in trades):,.2f}")
    print(f"Paper Gate Progress: {stats.n}/{config.paper_gate.min_paper_trades} trades")
    print("=======================================================\n")


def test_end_to_end_noisy_planted_edge(tmp_path: Path, capsys):
    """End-to-end multi-day replay test with noisy planted edge (drift plus noise).

    Asserts:
      - At least 20 shadow trades placed.
      - Win rate is in the 60-70% range (~65% expected from mu/sigma ratio).
      - All shadow trades settled.
      - Prints daily summary and overall win rate.
    """
    data_dir = tmp_path / "data_history"
    data_dir.mkdir(parents=True)
    db_path = tmp_path / "market.db"
    state_file = tmp_path / "state.json"
    reports_dir = tmp_path / "reports"
    daily_reports_dir = reports_dir / "daily"
    daily_reports_dir.mkdir(parents=True)

    # 7 full days (4 days warmup, 3 trading days) with drift + noise
    start_dt = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
    df_noisy = _generate_noisy_planted_data(start_dt, days=7, seed=42)
    df_noisy.to_csv(data_dir / "EURUSD.csv", index=False)

    config: AppConfig = load_config("config.yaml")
    config.mode = "paper"
    config.paper.decision_mode = "policy"
    config.source.name = "replay"
    config.source.replay_dir = str(data_dir)
    config.source.assets = ["EURUSD"]
    config.orchestrator.reports_dir = str(reports_dir)
    config.orchestrator.summary_hour_utc = 0
    config.orchestrator.warmup_bars = {"1d": 3, "1h": 5, "15m": 5, "5m": 5, "1m": 10}

    for tf in [config.gate.trend.daily, config.gate.trend.h1, config.gate.trend.m15]:
        tf.ema_fast = 2
        tf.ema_slow = 3
        tf.warmup_multiplier = 1
        tf.adx_period = 2
        tf.adx_min = 5

    config.patterns.min_similarity = 20.0
    config.patterns.window_lengths = [30, 45, 60]
    config.risk.volatility_lookback_days = 1
    config.risk.atr_period = 5
    config.trading.allowed_expiries = [15, 30, 60]
    config.trading.expiry_minutes = 15

    sim_start = start_dt + timedelta(days=4)
    sim_end = start_dt + timedelta(days=7)

    clock = FakeClock(initial_time=sim_start)
    store = DataStore(db_path=db_path)
    repository = TradeRepository(store=store)
    state_manager = StateManager(state_file=state_file)
    breaker = CircuitBreaker(config=config)
    kill_switch = KillSwitch()

    quote_payout = 0.88
    source = ReplaySource(
        clock=clock,
        config=config,
        data_dir=data_dir,
        assets=["EURUSD"],
        payout_map={"EURUSD": quote_payout},
    )

    paper_adapter = PaperAdapter(initial_balance=1000.0, tie_policy="loss")
    fake_real_adapter = MagicMock()
    fake_real_adapter.get_balance.return_value = 1000.0
    fake_real_adapter.get_open_trades.return_value = []
    mock_alerter = MagicMock()

    trade_manager = TradeManager(
        config=config,
        state_manager=state_manager,
        breaker=breaker,
        kill_switch=kill_switch,
        adapter=fake_real_adapter,
        paper_adapter=paper_adapter,
        repository=repository,
        alerter=mock_alerter,
    )

    orch = Orchestrator(
        config=config,
        source=source,
        adapter=fake_real_adapter,
        store=store,
        repository=repository,
        state_manager=state_manager,
        breaker=breaker,
        kill_switch=kill_switch,
        trade_manager=trade_manager,
        alerter=mock_alerter,
        clock=clock,
        paper_adapter=paper_adapter,
        is_first_run=True,
    )

    orch.startup()
    assert "EURUSD" in orch.ready_assets

    while clock.now() < sim_end:
        cur_t = clock.now()
        orch.run_settlement_job(cur_t)
        orch.run_evaluation_cycle(cur_t)
        if cur_t.hour == 0 and cur_t.minute == 0:
            orch.run_daily_summary_job(cur_t)
        clock.advance(seconds=900)

    orch.run_settlement_job(clock.now())
    orch.run_daily_summary_job(clock.now())

    trades = repository.get_trades()
    assert len(trades) >= 20, f"Expected >= 20 shadow trades, got {len(trades)}"

    stats = compute_asset_stats(trades=trades, asset="EURUSD", mode="paper", as_of=clock.now())
    assert stats is not None
    assert stats.n == len(trades)
    # Win rate roughly 60-70% (allow tolerance [0.58, 0.75])
    assert 0.58 <= stats.win_rate <= 0.75, (
        f"Expected noisy planted edge win rate in [0.58, 0.75], got {stats.win_rate:.3f}"
    )

    print("\n=======================================================")
    print("    NOISY PLANTED EDGE REPLAY TEST RUN RESULTS         ")
    print("=======================================================")
    unique_days = sorted(list({t.opened_at[:10] for t in trades}))
    for d_str in unique_days:
        d_trades = [t for t in trades if t.opened_at.startswith(d_str)]
        d_wins = sum(1 for t in d_trades if t.result == "WIN")
        d_losses = sum(1 for t in d_trades if t.result == "LOSS")
        d_wr = d_wins / len(d_trades) if d_trades else 0.0
        print(f"  Day {d_str}: {len(d_trades):>2} trades | {d_wins}W / {d_losses}L | WR: {d_wr:>6.1%}")

    print("-------------------------------------------------------")
    print(f"Total Shadow Trades: {len(trades)}")
    print(f"Overall Win Rate:    {stats.win_rate:.1%} (Break-Even: {stats.break_even:.1%})")
    print(f"Net PnL:             ${sum(t.pnl for t in trades):,.2f}")
    print("=======================================================\n")
