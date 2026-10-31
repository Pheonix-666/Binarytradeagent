"""End-to-end replay test on zero-drift random walk (no-edge control).

PROJECT_SPEC.md Phase 9 Pass 3:
  - Same orchestrator path as test_end_to_end_replay (real store, repository,
    TradeManager, PaperAdapter, FakeClock, paper.decision_mode=policy).
  - Zero-drift random walk replay data with realistic noise (no planted edge, no smooth waves).
  - Runs at least 3 seeds and enough days to get at least 150 trades per seed.
    NOTE: Pattern similarity threshold `config.patterns.min_similarity` is relaxed to 10.0
    (from default 60.0) specifically for this test to generate sufficient trades
    on pure zero-drift noise without structural patterns.
  - Asserts, per seed and pooled, that the win rate Wilson 95% interval contains 0.5
    (random outcomes), proving no look-ahead bias in orchestrator, paper entry, or replay source.
  - Prints win rate, trades, and Wilson CI per seed and pooled.
"""

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from adapters.paper import PaperAdapter
from core.clock import FakeClock
from core.config import AppConfig, load_config
from data.price_source import ReplaySource
from data.store import DataStore, TradeRepository
from execution.state import StateManager
from execution.trade_manager import TradeManager
from orchestrator.engine import Orchestrator
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch


def _generate_zero_drift_data(start_dt: datetime, days: int = 10, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 1m candles via zero-drift random walk with realistic noise.

    Data Generator:
      - Formula: close(m) = close(m-1) * (1.0 + r_m), where r_m ~ Normal(0, 0.00010)
      - Zero drift: mu = 0.0 (no directionality or edge)
      - Volatility: sigma = 0.00010 (~0.01% per minute, realistic currency noise)
      - Spread: open = close(m-1), high = max(o, c) + 0.00004, low = min(o, c) - 0.00004
      - Volume: 100.0
    """
    total_minutes = days * 1440
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0, scale=0.00010, size=total_minutes)
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


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute two-sided Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054  # 95% two-sided z
    p_hat = k / n
    denominator = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt((p_hat * (1.0 - p_hat)) / n + (z ** 2) / (4 * (n ** 2)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def test_end_to_end_no_edge_control(tmp_path: Path, capsys):
    """Run zero-drift random walk across 3 seeds and assert Wilson 95% CI contains 0.5."""
    seeds = [42, 123, 999]
    total_trades = 0
    total_wins = 0
    total_losses = 0

    print("\n" + "=" * 86)
    print("                NO-EDGE CONTROL END-TO-END REPLAY RUN RESULTS")
    print("=" * 86)
    print(f"{'Seed':<8} | {'Trades':<8} | {'Wins':<6} | {'Losses':<8} | {'Win Rate':<10} | {'Wilson 95% CI':<18} | {'Contains 0.5'}")
    print("-" * 86)

    for seed in seeds:
        data_dir = tmp_path / f"data_{seed}"
        data_dir.mkdir(parents=True)
        db_path = tmp_path / f"market_{seed}.db"
        state_file = tmp_path / f"state_{seed}.json"
        reports_dir = tmp_path / f"reports_{seed}"
        (reports_dir / "daily").mkdir(parents=True)

        start_dt = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
        df_no_edge = _generate_zero_drift_data(start_dt, days=10, seed=seed)
        df_no_edge.to_csv(data_dir / "EURUSD.csv", index=False)

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

        # Relax pattern similarity threshold from 60.0 to 10.0 to capture trades on pure noise
        config.patterns.min_similarity = 10.0
        config.patterns.window_lengths = [30, 45, 60]
        config.risk.volatility_lookback_days = 1
        config.risk.atr_period = 5
        config.trading.allowed_expiries = [15, 30, 60]
        config.trading.expiry_minutes = 15

        sim_start = start_dt + timedelta(days=3)
        sim_end = start_dt + timedelta(days=10)

        clock = FakeClock(initial_time=sim_start)
        store = DataStore(db_path=db_path)
        repository = TradeRepository(store=store)
        state_manager = StateManager(state_file=state_file)
        breaker = CircuitBreaker(config=config)
        kill_switch = KillSwitch()

        source = ReplaySource(
            clock=clock,
            config=config,
            data_dir=data_dir,
            assets=["EURUSD"],
            payout_map={"EURUSD": 0.88},
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

        while clock.now() < sim_end:
            cur_t = clock.now()
            orch.run_settlement_job(cur_t)
            orch.run_evaluation_cycle(cur_t)
            if cur_t.hour == 0 and cur_t.minute == 0:
                orch.run_daily_summary_job(cur_t)
            clock.advance(seconds=900)

        orch.run_settlement_job(clock.now())
        trades = repository.get_trades()

        assert len(trades) >= 150, f"Expected >= 150 shadow trades for seed {seed}, got {len(trades)}"
        wins = sum(1 for t in trades if t.result == "WIN")
        losses = sum(1 for t in trades if t.result == "LOSS")
        wr = wins / len(trades) if trades else 0.0
        ci_l, ci_h = wilson_score_interval(wins, len(trades))
        contains_half = ci_l <= 0.5 <= ci_h

        assert contains_half, (
            f"Seed {seed} Wilson 95% CI [{ci_l:.3f}, {ci_h:.3f}] does not contain 0.5! "
            f"WR={wr:.3f} indicates look-ahead bias."
        )

        total_trades += len(trades)
        total_wins += wins
        total_losses += losses

        ci_str = f"[{ci_l:.3f}, {ci_h:.3f}]"
        print(f"{seed:<8} | {len(trades):<8} | {wins:<6} | {losses:<8} | {wr:<10.1%} | {ci_str:<18} | {'YES' if contains_half else 'NO'}")

    pooled_wr = total_wins / total_trades if total_trades else 0.0
    p_low, p_high = wilson_score_interval(total_wins, total_trades)
    pooled_contains_half = p_low <= 0.5 <= p_high

    print("-" * 86)
    p_ci_str = f"[{p_low:.3f}, {p_high:.3f}]"
    print(f"{'POOLED':<8} | {total_trades:<8} | {total_wins:<6} | {total_losses:<8} | {pooled_wr:<10.1%} | {p_ci_str:<18} | {'YES' if pooled_contains_half else 'NO'}")
    print("=" * 86 + "\n")

    assert pooled_contains_half, (
        f"Pooled Wilson 95% CI [{p_low:.3f}, {p_high:.3f}] does not contain 0.5! "
        f"Pooled WR={pooled_wr:.3f} indicates look-ahead bias."
    )
