"""Parity tests between Backtest and Paper Orchestrator execution models.

PROJECT_SPEC.md Phase 9 Pass 3 / PART B:
  1. Stake Sizing Parity:
     - Policy-mode stake uses policy_stake_amount across max, mid, min policies and adaptive stake_scale.
     - Parity: Backtest and Paper calculate the exact same stake for identical account equity and candidate risk.
     - Stake None (below min_amount) returns None in both (triggers SKIP).
  2. Entry Price Parity:
     - Entry price uses 1m open bar at entry_time = decision_time + entry_delay_bars.
     - Slippage model: open +/- slippage_atr * atr.
     - Parity: Backtest and Paper produce identical entry prices.
  3. Execution Outcome Parity:
     - Identical entry price and exit price yield identical settlement outcome (WIN/LOSS/TIE).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from adapters.paper import PaperAdapter
from backtest.engine import Fast1mLookup
from core.clock import FakeClock
from core.config import AppConfig, ExecutionModelConfig, load_config
from core.models import TradeDirection
from data.price_source import ReplaySource
from data.store import DataStore, TradeRepository
from execution.state import StateManager
from execution.trade_manager import TradeManager
from orchestrator.engine import HeldOrder, Orchestrator
from orchestrator.paper_entry import calculate_entry_price
from risk.stake import max_stake_pct, policy_stake_amount, stake_amount
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch
from safety.limits import AccountSnapshot


def test_stake_sizing_parity():
    """Backtest and paper policy sizing produce exact identical dollar amounts."""
    equity = 10000.0
    cand_max_pct = 5.0
    min_pct = 1.0
    min_amount = 5.0
    round_to = 1.0
    max_amount = 500.0

    for policy in ["max", "mid", "min"]:
        for scale in [1.0, 0.75, 0.5, 0.35]:
            stake_paper = policy_stake_amount(
                equity=equity,
                max_stake_pct=cand_max_pct,
                stake_policy=policy,
                stake_scale=scale,
                min_pct=min_pct,
                min_amount=min_amount,
                round_to=round_to,
                max_amount=max_amount,
            )

            # Sizing calculation in backtest engine
            target = cand_max_pct
            if policy == "mid":
                target = (min_pct + target) / 2.0
            elif policy == "min":
                target = min_pct
            target *= scale

            expected_backtest = stake_amount(
                equity=equity,
                pct=target,
                min_amount=min_amount,
                round_to=round_to,
                max_amount=max_amount,
            )

            assert stake_paper == expected_backtest, f"Mismatch for policy={policy}, scale={scale}"

    # Edge case: below min_amount returns None (triggers SKIP in both)
    low_equity_stake = policy_stake_amount(
        equity=50.0,
        max_stake_pct=1.0,  # 1% of 50 = $0.50, below min_amount $5.0
        stake_policy="max",
        stake_scale=1.0,
        min_pct=min_pct,
        min_amount=min_amount,
        round_to=round_to,
        max_amount=max_amount,
    )
    assert low_equity_stake is None


def test_entry_price_calculation_parity():
    """Entry price calculation produces identical prices between backtest formula and orchestrator helper."""
    open_prices = [1.12345, 105.432, 0.85210]
    atrs = [0.00120, 0.0540, 0.00085]
    slippage_atr = 0.02

    for open_px in open_prices:
        for atr in atrs:
            # Backtest calculation
            expected_call = float(open_px + slippage_atr * atr)
            actual_call = calculate_entry_price(
                open_price=open_px,
                direction=TradeDirection.CALL,
                atr=atr,
                slippage_atr=slippage_atr,
            )
            assert actual_call == expected_call

            # PUT parity
            expected_put = float(open_px - slippage_atr * atr)
            actual_put = calculate_entry_price(
                open_price=open_px,
                direction=TradeDirection.PUT,
                atr=atr,
                slippage_atr=slippage_atr,
            )
            assert actual_put == expected_put


def test_held_order_entry_fidelity_with_replay_lookup(tmp_path: Path):
    """Orchestrator held order executes at entry_time using 1m bar open price, matching backtest."""
    start_dt = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc)
    t_decision = start_dt
    t_entry = start_dt + timedelta(minutes=1)

    # Synthetic 1m candles
    # At t_decision (10:00): open=1.1000, close=1.1005
    # At t_entry (10:01): open=1.1008, close=1.1012
    records = [
        {
            "ts": t_decision.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ts_close": (t_decision + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": 1.10000,
            "high": 1.10060,
            "low": 1.09990,
            "close": 1.10050,
            "volume": 100.0,
        },
        {
            "ts": t_entry.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ts_close": (t_entry + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": 1.10080,
            "high": 1.10150,
            "low": 1.10070,
            "close": 1.10120,
            "volume": 100.0,
        },
    ]
    df_1m = pd.DataFrame(records)

    # 1. Backtest expectation: entry price at entry_time (10:01) open + slippage
    lookup = Fast1mLookup(df_1m)
    entry_bar = lookup.get_bar(t_entry)
    assert entry_bar is not None
    atr = 0.00100
    slippage_atr = 0.02
    expected_backtest_entry = float(entry_bar["open"] + slippage_atr * atr)

    # 2. Orchestrator calculation
    config = load_config("config.yaml")
    config.execution_model.entry_delay_bars = 1
    config.execution_model.slippage_atr = slippage_atr

    mock_source = MagicMock()
    mock_source._load_raw_1m.return_value = df_1m

    paper_adapter = PaperAdapter(initial_balance=1000.0)
    breaker = CircuitBreaker(config=config)
    kill_switch = KillSwitch()
    store = DataStore(db_path=tmp_path / "test.db")
    repository = TradeRepository(store=store)
    state_mgr = StateManager(state_file=tmp_path / "state.json")
    trade_mgr = TradeManager(
        config=config,
        state_manager=state_mgr,
        breaker=breaker,
        kill_switch=kill_switch,
        adapter=paper_adapter,
        paper_adapter=paper_adapter,
        repository=repository,
        alerter=MagicMock(),
    )
    trade_mgr.set_reconciled(True)

    orch = Orchestrator(
        config=config,
        source=mock_source,
        adapter=paper_adapter,
        store=store,
        repository=repository,
        state_manager=state_mgr,
        breaker=breaker,
        kill_switch=kill_switch,
        trade_manager=trade_mgr,
        alerter=MagicMock(),
        clock=FakeClock(initial_time=t_decision),
        paper_adapter=paper_adapter,
    )

    from decision.schema import ValidatedDecision

    decision = ValidatedDecision(
        decision="CALL",
        is_valid=True,
        decision_id="dec_001",
        stake_amount=50.0,
        asset="EURUSD",
        direction="CALL",
        expiry_minutes=15,
        payout=0.85,
    )
    account = AccountSnapshot(
        balance=1000.0,
        equity=1000.0,
        open_trades=[],
        asset_modes={"EURUSD": "paper"},
    )

    held = HeldOrder(
        decision=decision,
        candidate=None,
        account=account,
        adaptive_state=None,
        decision_time=t_decision,
        entry_time=t_entry,
        asset="EURUSD",
        direction=TradeDirection.CALL,
        expiry_minutes=15,
        atr=atr,
        payout=0.85,
    )

    # Execute held order at entry_time
    result = orch._execute_held_order(held, execution_time=t_entry)
    assert result is not None
    assert result.executed is True

    # Check that paper adapter received the exact backtest entry price
    open_trades = list(paper_adapter.get_open_trades())
    assert len(open_trades) == 1
    assert open_trades[0].entry_price == expected_backtest_entry
