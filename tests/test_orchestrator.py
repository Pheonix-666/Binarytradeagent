"""Comprehensive tests for Orchestrator, Jobs, Startup, and Telegram Worker.

PROJECT_SPEC.md Phase 9:
  - Startup fail closed: demo/live rejected with 'not enabled until Phase 10'. Missing repo rejected.
  - Aligned 5m cycles, candidate evaluation, JSONL logging, skip reasons.
  - 1m settlement job via PaperAdapter with TradeRepository update.
  - Heartbeat, asset reselection, and daily summary markdown generation.
  - Telegram worker, operator slash commands, ticket confirmation and timeout.
  - Fault injection: stale data, source errors.
  - Multi-cycle synthetic replay run.
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.parse
import pandas as pd
import pytest

from adapters.paper import PaperAdapter
from adaptive.monitor import AdaptiveState
from alerts.telegram import TelegramAlerter
from analysis.candidates import CandidateEvaluationResult, CandidateTrade
from core.clock import FakeClock
from core.config import AppConfig, OrchestratorConfig, PaperConfig, SourceConfig, load_config
from core.models import TradeDirection
from decision.schema import ValidatedDecision
from execution.trade_manager import ExecutionResult
from data.price_source import ReplaySource
from data.store import DataStore, TradeRecord, TradeRepository
from execution.state import StateManager
from execution.trade_manager import TradeManager
from orchestrator.engine import Orchestrator
from orchestrator.fresh_inputs import build_account_snapshot
from orchestrator.paper_entry import PaperEntryModel
from orchestrator.reports import generate_daily_summary_markdown
from orchestrator.telegram_worker import ConfirmationTicket, TelegramWorker
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch


@pytest.fixture
def mock_alerter() -> TelegramAlerter:
    alerter = TelegramAlerter(token="dummy_token", chat_id="12345")
    alerter.send_alert = MagicMock(return_value=True)  # type: ignore
    return alerter


@pytest.fixture
def test_env(tmp_path: Path, mock_alerter: TelegramAlerter):
    db_path = tmp_path / "test_market.db"
    store = DataStore(db_path=db_path)
    repository = TradeRepository(store=store)

    state_path = tmp_path / "state.json"
    state_manager = StateManager(state_file=state_path)

    # Initial state
    state_manager.data = {
        "balance": 1000.0,
        "equity": 1000.0,
        "open_trades": [],
        "pending_trades": [],
        "adaptive_state": {"tier": "NORMAL"},
        "telegram_offset": 0,
    }

    config = load_config("config.yaml")
    config.mode = "paper"
    config.source.assets = ["EURUSD"]
    config.orchestrator.reports_dir = str(tmp_path / "reports")

    breaker = CircuitBreaker(config=config)
    kill_switch = KillSwitch(workspace_root=tmp_path)

    paper_adapter = PaperAdapter(initial_balance=1000.0)

    trade_manager = TradeManager(
        config=config,
        state_manager=state_manager,
        breaker=breaker,
        kill_switch=kill_switch,
        adapter=paper_adapter,
        paper_adapter=paper_adapter,
        repository=repository,
        alerter=mock_alerter,
    )

    # Synthetic 1m candles CSV
    history_dir = tmp_path / "data_history"
    history_dir.mkdir(parents=True)
    base_t = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    rows = []
    px = 1.1000
    for i in range(180):
        t_open = base_t + timedelta(minutes=i)
        t_close = t_open + timedelta(minutes=1)
        o = px + i * 0.0001
        rows.append({
            "ts": t_open.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ts_close": t_close.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": o,
            "high": o + 0.0005,
            "low": o - 0.0005,
            "close": o + 0.0001,
            "volume": 100.0,
        })
    import pandas as pd
    pd.DataFrame(rows).to_csv(history_dir / "EURUSD.csv", index=False)

    clock = FakeClock(initial_time=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc))
    source = ReplaySource(clock=clock, config=config, data_dir=history_dir, assets=["EURUSD"])

    return {
        "config": config,
        "store": store,
        "repository": repository,
        "state_manager": state_manager,
        "breaker": breaker,
        "kill_switch": kill_switch,
        "paper_adapter": paper_adapter,
        "trade_manager": trade_manager,
        "mock_alerter": mock_alerter,
        "source": source,
        "clock": clock,
        "tmp_path": tmp_path,
    }


def test_startup_rejects_demo_and_live(test_env):
    """Refuse to start if mode is demo or live with exact message 'not enabled until Phase 10'."""
    env = test_env
    env["config"].mode = "demo"
    orch = Orchestrator(
        config=env["config"],
        source=env["source"],
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=env["mock_alerter"],
        clock=env["clock"],
    )
    with pytest.raises(ValueError, match="not enabled until Phase 10"):
        orch.validate_startup()

    env["config"].mode = "live"
    with pytest.raises(ValueError, match="not enabled until Phase 10"):
        orch.validate_startup()


def test_startup_rejects_missing_trade_repository(test_env):
    """Refuse to start if no TradeRepository is supplied."""
    env = test_env
    orch = Orchestrator(
        config=env["config"],
        source=env["source"],
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=None,  # type: ignore
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=env["mock_alerter"],
        clock=env["clock"],
    )
    with pytest.raises(ValueError, match="TradeRepository is required"):
        orch.validate_startup()


def test_startup_success_in_record_and_paper(test_env):
    """Startup succeeds in record and paper mode, executes reconcile, sets reconciled, and sends started alert."""
    env = test_env
    env["config"].mode = "paper"
    orch = Orchestrator(
        config=env["config"],
        source=env["source"],
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=env["mock_alerter"],
        clock=env["clock"],
        is_first_run=True,
    )
    orch.startup()
    assert env["trade_manager"].reconcile_done is True
    assert env["mock_alerter"].send_alert.called
    alert_text = env["mock_alerter"].send_alert.call_args_list[-1][0][0]
    assert "Trading system started in *paper* mode" in alert_text


def test_paper_entry_model():
    """PaperEntryModel adds slippage against the trade and calculates correct timestamps."""
    model = PaperEntryModel(entry_delay_bars=1, slippage_atr=0.1)
    t = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)

    # CALL: entry price should be base + slippage
    call_res = model.calculate_entry(
        decision_time=t,
        base_price=100.0,
        direction=TradeDirection.CALL,
        expiry_minutes=5,
        atr=2.0,
    )
    assert call_res.entry_time == datetime(2026, 10, 1, 12, 1, tzinfo=timezone.utc)
    assert call_res.exit_time == datetime(2026, 10, 1, 12, 6, tzinfo=timezone.utc)
    assert call_res.slippage == 0.2
    assert call_res.entry_price == 100.2

    # PUT: entry price should be base - slippage
    put_res = model.calculate_entry(
        decision_time=t,
        base_price=100.0,
        direction=TradeDirection.PUT,
        expiry_minutes=5,
        atr=2.0,
    )
    assert put_res.entry_price == 99.8


def test_build_account_snapshot_fresh(test_env):
    """AccountSnapshot is constructed fresh from adapter, repo, and config."""
    env = test_env
    now = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    snap = build_account_snapshot(
        adapter=env["paper_adapter"],
        repository=env["repository"],
        config=env["config"],
        now=now,
        assets=["EURUSD"],
    )
    assert snap.balance == 1000.0
    assert snap.equity == 1000.0
    assert snap.asset_modes == {"EURUSD": "paper"}
    assert snap.daily_starting_balance == 1000.0
    assert snap.today_loss == 0.0


def test_telegram_worker_ticket_confirm_and_timeout(mock_alerter, test_env):
    """TelegramWorker registers trade ticket, processes /approve and /reject, and handles timeouts."""
    env = test_env
    clock = FakeClock(initial_time=datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc))

    worker = TelegramWorker(
        alerter=mock_alerter,
        state_manager=env["state_manager"],
        config=env["config"],
        clock=clock,
    )

    approved = []
    rejected = []

    # Register ticket with 10s timeout
    ticket = worker.register_ticket(
        ticket_id="TKT-001",
        decision=MagicMock(asset="EURUSD", direction="CALL", stake=10.0, expiry_minutes=5),
        account=MagicMock(),
        on_approve=lambda: approved.append("TKT-001"),
        on_reject=lambda: rejected.append("TKT-001"),
        timeout_seconds=10,
    )

    assert ticket.status == "PENDING"
    assert mock_alerter.send_alert.called

    # 1. Approve ticket
    res = worker._cmd_approve(["TKT-001"])
    assert "approved and dispatched" in res
    assert ticket.status == "APPROVED"
    assert approved == ["TKT-001"]

    # 2. Reject another ticket
    ticket2 = worker.register_ticket(
        ticket_id="TKT-002",
        decision=MagicMock(asset="EURUSD", direction="PUT", stake=10.0, expiry_minutes=5),
        account=MagicMock(),
        on_reject=lambda: rejected.append("TKT-002"),
        timeout_seconds=10,
    )
    res_rej = worker._cmd_reject(["TKT-002"])
    assert "rejected" in res_rej
    assert ticket2.status == "REJECTED"
    assert rejected == ["TKT-002"]

    # 3. Timeout handling
    ticket3 = worker.register_ticket(
        ticket_id="TKT-003",
        decision=MagicMock(asset="EURUSD", direction="CALL", stake=10.0, expiry_minutes=5),
        account=MagicMock(),
        timeout_seconds=5,
    )
    clock.advance(seconds=6)
    worker.check_expired_tickets()
    assert ticket3.status == "EXPIRED"

    # Attempting to approve an expired ticket fails
    res_exp = worker._cmd_approve(["TKT-003"])
    assert "already EXPIRED" in res_exp or "expired" in res_exp


def test_stale_data_skips_cycle_and_alerts(test_env):
    """When latest market data is older than max_data_staleness_minutes, cycle is skipped."""
    env = test_env
    # CSV data ends at 12:00 UTC (9:00 + 180 min); set clock to 14:00 UTC (2 hours stale!)
    env["clock"].set_time(datetime(2026, 10, 1, 14, 0, tzinfo=timezone.utc))

    orch = Orchestrator(
        config=env["config"],
        source=env["source"],
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=env["mock_alerter"],
        clock=env["clock"],
    )

    results = orch.run_evaluation_cycle(env["clock"].now())
    # No trades executed due to stale data
    assert len(results) == 0

    # JSONL log has skip record
    assert orch.cycle_log_path.exists()
    content = orch.cycle_log_path.read_text(encoding="utf-8")
    assert "Data freshness check failed" in content


def test_settlement_job_updates_repository(test_env):
    """1m settlement job settles open trades through PaperAdapter and updates TradeRepository."""
    env = test_env
    cur_t = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
    env["clock"].set_time(cur_t)

    # Place a paper trade in TradeManager
    rec = TradeRecord(
        id="trade_123",
        opened_at=cur_t.isoformat(),
        asset="EURUSD",
        is_otc=False,
        direction=TradeDirection.CALL,
        expiry_minutes=5,
        stake=20.0,
        payout_at_entry=0.85,
        result="PENDING",
        pnl=0.0,
        mode="paper",
        pattern_name="test_pattern",
        pattern_type="continuation",
        risk_score=15,
        hour_utc=10,
    )
    env["repository"].insert(rec)

    # Put matching trade in PaperAdapter
    env["paper_adapter"].set_simulated_time(cur_t)
    env["paper_adapter"].set_price("EURUSD", 1.1000)
    receipt = env["paper_adapter"].place_trade(
        asset_id="EURUSD",
        direction="CALL",
        expiry_min=5,
        stake_amount=20.0,
        payout=0.85,
        trade_id="trade_123",
    )

    # Advance clock past expiry (10:06 UTC) and set winning exit price (1.1050 > 1.1000)
    cur_t += timedelta(minutes=6)
    env["clock"].set_time(cur_t)
    env["paper_adapter"].set_price("EURUSD", 1.1050)

    orch = Orchestrator(
        config=env["config"],
        source=env["source"],
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=env["mock_alerter"],
        clock=env["clock"],
        paper_adapter=env["paper_adapter"],
    )

    settled = orch.run_settlement_job(cur_t)
    assert len(settled) == 1
    assert settled[0].trade_id == "trade_123"

    # TradeRepository updated to WIN
    updated = env["repository"].get_trade("trade_123")
    assert updated is not None
    assert updated.result == "WIN"
    assert updated.pnl == 17.0  # 20.0 * 0.85


def test_daily_summary_markdown_generation(test_env):
    """Daily summary job generates reports/daily/YYYY-MM-DD.md and sends Telegram summary."""
    env = test_env
    cur_t = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)  # Hour 0
    env["clock"].set_time(cur_t)

    # Add a settled trade to repository
    env["repository"].insert(
        TradeRecord(
            id="t1",
            opened_at="2026-10-01T05:00:00Z",
            closed_at="2026-10-01T05:05:00Z",
            asset="EURUSD",
            is_otc=False,
            direction="CALL",
            expiry_minutes=5,
            stake=10.0,
            payout_at_entry=0.85,
            result="WIN",
            pnl=8.5,
            mode="paper",
            hour_utc=5,
        )
    )

    orch = Orchestrator(
        config=env["config"],
        source=env["source"],
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=env["mock_alerter"],
        clock=env["clock"],
    )

    report_path = orch.run_daily_summary_job(cur_t)
    assert report_path is not None
    assert Path(report_path).exists()

    content = Path(report_path).read_text(encoding="utf-8")
    assert "# Daily Performance Report — 2026-10-01" in content
    assert "EURUSD" in content
    assert "$8.50" in content
    assert env["mock_alerter"].send_alert.called


def test_overrun_skips_missed_cycles(test_env):
    """Cycle overrun skips missed cycles without queuing past intervals."""
    env = test_env
    # Start at 10:00:00
    cur_t = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
    env["clock"].set_time(cur_t)

    orch = Orchestrator(
        config=env["config"],
        source=env["source"],
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=env["mock_alerter"],
        clock=env["clock"],
    )

    # Next 5m aligned time from 10:00:00 with delay
    delay = env["config"].orchestrator.candle_delay_seconds
    next_t = orch.compute_next_aligned_5m(cur_t)
    assert next_t == datetime(2026, 10, 1, 10, 5, delay, tzinfo=timezone.utc)

    # Simulate heavy overrun: clock advances by 12 minutes to 10:12:00
    env["clock"].advance(seconds=12 * 60)
    overrun_next = orch.compute_next_aligned_5m(env["clock"].now())
    # Should skip 10:05 and 10:10, next is 10:15 + delay
    assert overrun_next == datetime(2026, 10, 1, 10, 15, delay, tzinfo=timezone.utc)


def test_multi_cycle_synthetic_replay_run(test_env):
    """Run orchestrator across consecutive 5m cycles: candidate evaluation, execution, settlement."""
    env = test_env
    cur_t = datetime(2026, 10, 1, 10, 15, tzinfo=timezone.utc)
    env["clock"].set_time(cur_t)

    orch = Orchestrator(
        config=env["config"],
        source=env["source"],
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=env["mock_alerter"],
        clock=env["clock"],
        paper_adapter=env["paper_adapter"],
        is_first_run=True,
    )
    orch.startup()

    # Step through 3 evaluation cycles at 10:15, 10:20, 10:25
    for _ in range(3):
        orch.step_single_cycle(env["clock"].now())
        env["clock"].advance(seconds=300)

    # Verify JSONL log has records
    assert orch.cycle_log_path.exists()
    lines = orch.cycle_log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 3
    rec0 = json.loads(lines[0])
    assert "timestamp" in rec0
    assert "asset" in rec0
    assert "action" in rec0


def test_causal_invariance_perturbation(test_env, tmp_path: Path):
    """Perturbing market data occurring after t must not change candidates or decisions at t (zero look-ahead)."""
    env = test_env
    t_eval = datetime(2026, 10, 1, 10, 15, tzinfo=timezone.utc)
    env["clock"].set_time(t_eval)

    orch = Orchestrator(
        config=env["config"],
        source=env["source"],
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=env["mock_alerter"],
        clock=env["clock"],
        paper_adapter=env["paper_adapter"],
        is_first_run=True,
    )
    orch.startup()

    # Step 1: Run evaluation at t_eval before perturbation
    res_before = orch.run_evaluation_cycle(t_eval)

    # Step 2: Perturb future candles (ts > t_eval) in the underlying raw CSV
    csv_file = tmp_path / "data_history" / "EURUSD.csv"
    import pandas as pd
    df_raw = pd.read_csv(csv_file)
    df_raw["ts_dt"] = pd.to_datetime(df_raw["ts"], utc=True)

    # Invert or drastically spike future prices after t_eval
    mask_future = df_raw["ts_dt"] > t_eval
    df_raw.loc[mask_future, "open"] *= 10.0
    df_raw.loc[mask_future, "high"] *= 10.0
    df_raw.loc[mask_future, "low"] *= 10.0
    df_raw.loc[mask_future, "close"] *= 10.0
    df_raw.drop(columns=["ts_dt"]).to_csv(csv_file, index=False)

    # Clear cache in source and store so new file is re-read
    env["source"]._raw_cache.clear()
    with env["store"]._get_connection() as conn:
        conn.execute("DELETE FROM candles")
        conn.commit()

    # Step 3: Run evaluation at t_eval after perturbation
    res_after = orch.run_evaluation_cycle(t_eval)

    # Assert invariant: decisions and results at t_eval are 100% identical
    assert len(res_before) == len(res_after)
    for b, a in zip(res_before, res_after):
        assert b.executed == a.executed
        assert b.status == a.status
        assert b.reason == a.reason


def test_timestamp_normalization_and_stats_filtering(tmp_path: Path):
    """normalize_trade_timestamp normalizes all formats to ISO-8601 UTC; real string & shadow datetime sort consistently."""
    db_path = tmp_path / "ts_test.db"
    store = DataStore(db_path=db_path)
    repo = TradeRepository(store=store)

    # 1. Real settlement with string format (non-Z or space)
    repo.insert(
        TradeRecord(
            id="trade_real",
            opened_at="2026-10-01 10:00:00",
            closed_at="2026-10-01T10:05:00+00:00",
            asset="EURUSD",
            is_otc=False,
            direction="CALL",
            expiry_minutes=5,
            stake=10.0,
            payout_at_entry=0.85,
            result="WIN",
            pnl=8.5,
            mode="demo",
            hour_utc=10,
        )
    )

    # 2. Shadow settlement with datetime object
    dt_open = datetime(2026, 10, 1, 10, 10, tzinfo=timezone.utc)
    dt_close = datetime(2026, 10, 1, 10, 15, tzinfo=timezone.utc)
    repo.insert(
        TradeRecord(
            id="trade_shadow",
            opened_at=dt_open.isoformat(),
            closed_at=dt_close.isoformat(),
            asset="EURUSD",
            is_otc=False,
            direction="PUT",
            expiry_minutes=5,
            stake=10.0,
            payout_at_entry=0.85,
            result="LOSS",
            pnl=-10.0,
            mode="paper",
            hour_utc=10,
        )
    )

    # Verify both are normalized in SQLite to standard ISO-8601 UTC '%Y-%m-%dT%H:%M:%SZ'
    t_real = repo.get_trade_by_id("trade_real")
    t_shadow = repo.get_trade_by_id("trade_shadow")
    assert t_real.opened_at == "2026-10-01T10:00:00Z"
    assert t_real.closed_at == "2026-10-01T10:05:00Z"
    assert t_shadow.opened_at == "2026-10-01T10:10:00Z"
    assert t_shadow.closed_at == "2026-10-01T10:15:00Z"

    # Query trades using as_of datetime in stats
    from selection.stats import compute_asset_stats
    all_trades = repo.get_trades()

    # as_of at 10:07 UTC -> includes trade_real, excludes trade_shadow
    st_early = compute_asset_stats(trades=all_trades, asset="EURUSD", mode="demo", as_of=datetime(2026, 10, 1, 10, 7, tzinfo=timezone.utc))
    assert st_early is not None
    assert st_early.n == 1
    assert st_early.wins == 1

    # as_of at 10:20 UTC for paper -> includes trade_shadow
    st_late = compute_asset_stats(trades=all_trades, asset="EURUSD", mode="paper", as_of=datetime(2026, 10, 1, 10, 20, tzinfo=timezone.utc))
    assert st_late is not None
    assert st_late.n == 1
    assert st_late.losses == 1


def test_candidate_and_validated_decision_carry_metadata(test_env):
    """CandidateTrade and ValidatedDecision carry pattern_name, pattern_type, risk_score, payout and is_otc."""
    cand = CandidateTrade(
        asset="BTCUSD_otc",
        direction=TradeDirection.CALL,
        t=datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc),
        current_price=50000.0,
        atr=100.0,
        pattern={"name": "BullishEngulfing", "type": "reversal"},
        pattern_match=MagicMock(),
        risk_report=MagicMock(score=15),
        gate_result=MagicMock(),
        max_stake_pct=0.02,
        volatility_percentile=50.0,
        payout=0.92,
        is_otc=True,
    )
    assert cand.pattern_name == "BullishEngulfing"
    assert cand.pattern_type == "reversal"
    assert cand.risk_score == 15
    assert cand.payout == 0.92
    assert cand.is_otc is True

    decision = ValidatedDecision(
        decision="CALL",
        is_valid=True,
        asset="BTCUSD_otc",
        direction="CALL",
        expiry_minutes=5,
        stake_amount=10.0,
        pattern_name=cand.pattern_name,
        pattern_type=cand.pattern_type,
        risk_score=cand.risk_score,
        payout=cand.payout,
        is_otc=cand.is_otc,
    )
    assert decision.payout == 0.92
    assert decision.is_otc is True
    assert decision.pattern_name == "BullishEngulfing"


def test_no_payout_quote_skips_candidate(test_env):
    """If the source has no payout (or None) for an asset, skip that candidate and log 'no payout'."""
    env = test_env
    cur_t = datetime(2026, 10, 1, 10, 15, tzinfo=timezone.utc)
    env["clock"].set_time(cur_t)

    # Set payout to None for EURUSD
    env["source"].payout_map = {"EURUSD": 0.0}

    orch = Orchestrator(
        config=env["config"],
        source=env["source"],
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=env["mock_alerter"],
        clock=env["clock"],
        paper_adapter=env["paper_adapter"],
        is_first_run=True,
    )
    orch.startup()

    results = orch.run_evaluation_cycle(cur_t)
    assert len(results) == 0

    assert orch.cycle_log_path.exists()
    content = orch.cycle_log_path.read_text(encoding="utf-8")
    assert '"reason": "no payout"' in content


def test_shadow_trade_without_payout_raises_and_equals_quote_payout(test_env):
    """A shadow trade without payout raises, and a valid shadow trade records payout_at_entry equal to quote payout."""
    env = test_env
    tm = env["trade_manager"]
    tm.set_reconciled(True)
    account = build_account_snapshot(
        adapter=env["paper_adapter"],
        repository=env["repository"],
        config=env["config"],
        now=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
        assets=["EURUSD"],
    )

    # 1. Decision with payout=None should raise when executing shadow trade
    dec_no_payout = ValidatedDecision(
        decision="CALL",
        is_valid=True,
        decision_id="dec_no_pay",
        asset="EURUSD",
        direction="CALL",
        expiry_minutes=15,
        stake_amount=10.0,
        payout=None,
    )
    with pytest.raises(ValueError, match="payout must be positive"):
        tm.execute(dec_no_payout, account=account)

    # 2. Decision with quote payout (0.88) records payout_at_entry == 0.88
    dec_with_payout = ValidatedDecision(
        decision="CALL",
        is_valid=True,
        decision_id="dec_with_pay",
        asset="EURUSD",
        direction="CALL",
        expiry_minutes=15,
        stake_amount=10.0,
        payout=0.88,
    )
    res = tm.execute(dec_with_payout, account=account)
    assert res.executed is True
    rec = env["repository"].get_trade_by_id(res.trade_id)
    assert rec is not None
    assert rec.payout_at_entry == 0.88


def test_warmup_insufficient_history_stays_in_record_only(test_env, mock_alerter, tmp_path):
    """Assets with insufficient history stay in record-only until ready and log warnings."""
    env = test_env
    clock = FakeClock(initial_time=datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc))

    mock_source = MagicMock()
    mock_source.now.return_value = clock.now()
    mock_source.get_assets.return_value = [
        MagicMock(asset_id="EURUSD", name="EURUSD", is_otc=False, payout=0.85, is_open=True)
    ]
    # Return only 5 bars (insufficient for 150 bars warmup)
    short_df = pd.DataFrame({
        "ts": [clock.now() - timedelta(minutes=i) for i in range(5, 0, -1)],
        "ts_close": [clock.now() - timedelta(minutes=i-1) for i in range(5, 0, -1)],
        "open": [1.10] * 5,
        "high": [1.11] * 5,
        "low": [1.09] * 5,
        "close": [1.105] * 5,
        "volume": [100.0] * 5,
    })
    mock_source.get_candles.return_value = short_df

    orch = Orchestrator(
        config=env["config"],
        source=mock_source,
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=mock_alerter,
        clock=clock,
        paper_adapter=env["paper_adapter"],
    )
    orch.startup()
    assert "EURUSD" in orch.record_only_assets
    assert "EURUSD" not in orch.ready_assets

    # In evaluation cycle, asset stays in record-only
    results = orch.run_evaluation_cycle(clock.now())
    assert len(results) == 0
    cycle_content = orch.cycle_log_path.read_text(encoding="utf-8")
    assert "insufficient history (record-only)" in cycle_content


def test_telegram_polling_updates_older_dropped(test_env, mock_alerter):
    """Updates older than startup are dropped and offset advanced."""
    env = test_env
    worker = TelegramWorker(
        alerter=mock_alerter,
        state_manager=env["state_manager"],
        config=env["config"],
    )

    with patch.object(worker, "_fetch_updates") as mock_fetch:
        mock_fetch.return_value = [
            {"update_id": 100, "message": {"text": "/status"}},
            {"update_id": 105, "message": {"text": "/stop"}},
        ]
        worker.drop_pending_updates()
        assert worker.offset == 106
        assert env["state_manager"].data.get("telegram_offset") == 106


def test_telegram_polling_offset_persisted(test_env, mock_alerter):
    """Offset is persisted in state and restored on worker restart."""
    env = test_env
    worker = TelegramWorker(
        alerter=mock_alerter,
        state_manager=env["state_manager"],
        config=env["config"],
    )
    worker.offset = 250
    worker._save_offset()
    assert env["state_manager"].data["telegram_offset"] == 250

    worker2 = TelegramWorker(
        alerter=mock_alerter,
        state_manager=env["state_manager"],
        config=env["config"],
    )
    assert worker2.offset == 250


def test_telegram_polling_wrong_chat_id_ignored(test_env, mock_alerter):
    """Commands from unauthorized chat IDs are rejected."""
    env = test_env
    mock_alerter._chat_id = "12345"
    worker = TelegramWorker(
        alerter=mock_alerter,
        state_manager=env["state_manager"],
        config=env["config"],
    )
    res = worker.process_message({"chat": {"id": "99999"}, "text": "/stop"})
    assert res is None


def test_telegram_polling_stop_halts_new_trades(test_env, mock_alerter):
    """Operator /stop activates kill switch and halts new trades."""
    env = test_env
    mock_alerter._chat_id = "12345"
    worker = TelegramWorker(
        alerter=mock_alerter,
        state_manager=env["state_manager"],
        config=env["config"],
    )
    # Register operator commands
    from alerts.telegram import create_operator_command_handlers
    handlers = create_operator_command_handlers(
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        config=env["config"],
        trade_manager=env["trade_manager"],
    )
    worker.handlers.update(handlers)

    res = worker.process_message({"chat": {"id": "12345"}, "text": "/stop"})
    assert "STOPPED" in str(res)
    assert env["kill_switch"].is_active() is True

    # Order execution blocked by kill switch
    env["trade_manager"].set_reconciled(True)
    account = build_account_snapshot(
        adapter=env["paper_adapter"],
        repository=env["repository"],
        config=env["config"],
        now=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
        assets=["EURUSD"],
    )
    dec = ValidatedDecision(
        decision="CALL",
        is_valid=True,
        decision_id="dec_blocked",
        asset="EURUSD",
        direction="CALL",
        expiry_minutes=15,
        stake_amount=10.0,
        payout=0.85,
    )
    exec_res = env["trade_manager"].execute(dec, account=account)
    assert exec_res.executed is False
    assert "kill switch" in exec_res.reason.lower()


def test_fault_price_source_down_survives_and_alerts(test_env, mock_alerter):
    """Price source outages survive and trigger alerts after consecutive threshold."""
    env = test_env
    clock = FakeClock(initial_time=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc))

    bad_source = MagicMock()
    bad_source.now.return_value = clock.now()
    bad_source.get_assets.side_effect = RuntimeError("REST API 500 Connection Refused")

    orch = Orchestrator(
        config=env["config"],
        source=bad_source,
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=mock_alerter,
        clock=clock,
        paper_adapter=env["paper_adapter"],
    )

    # 3 consecutive failures
    for _ in range(3):
        res = orch.run_evaluation_cycle(clock.now())
        assert len(res) == 0

    assert orch.consecutive_source_failures == 3
    # An alert should have been sent
    assert any("Price source down" in str(c[0][0]) for c in mock_alerter.send_alert.call_args_list)


def test_fault_llm_down_skips_and_continues(test_env, mock_alerter):
    """LLM decision exceptions log SKIP and allow loop to continue."""
    env = test_env
    clock = FakeClock(initial_time=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc))

    mock_llm = MagicMock()
    mock_llm.decide.side_effect = RuntimeError("LLM 503 Overloaded")

    # Seed store with candles
    t = clock.now()
    df_1m = pd.DataFrame({
        "ts": [t - timedelta(minutes=i) for i in range(100, 0, -1)],
        "ts_close": [t - timedelta(minutes=i-1) for i in range(100, 0, -1)],
        "open": [1.10] * 100,
        "high": [1.11] * 100,
        "low": [1.09] * 100,
        "close": [1.105] * 100,
        "volume": [100.0] * 100,
    })
    env["store"].save_candles("EURUSD", "1m", df_1m)

    cfg = env["config"].model_copy(deep=True)
    cfg.paper.decision_mode = "llm"

    mock_source = MagicMock()
    mock_source.now.return_value = t
    mock_source.get_assets.return_value = [
        MagicMock(asset_id="EURUSD", name="EURUSD", is_otc=False, payout=0.85, is_open=True)
    ]
    mock_source.get_candles.return_value = df_1m

    orch = Orchestrator(
        config=cfg,
        source=mock_source,
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=mock_alerter,
        clock=clock,
        paper_adapter=env["paper_adapter"],
        decision_service=mock_llm,
    )
    orch.ready_assets.add("EURUSD")

    # Mock build_candidates to return 1 candidate
    with patch("orchestrator.engine.build_candidates") as mock_bc:
        mock_cand = MagicMock()
        mock_bc.return_value = CandidateEvaluationResult(asset="EURUSD", t=t, candidates=[mock_cand])
        results = orch.run_evaluation_cycle(t)
        assert len(results) == 0

    log_txt = orch.cycle_log_path.read_text(encoding="utf-8")
    assert "LLM error" in log_txt


def test_fault_repository_write_failure_trips_breaker(test_env, mock_alerter):
    """Database write failure on real trade trips circuit breaker and raises alert."""
    env = test_env
    tm = env["trade_manager"]
    tm.set_reconciled(True)
    account = build_account_snapshot(
        adapter=env["paper_adapter"],
        repository=env["repository"],
        config=env["config"],
        now=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
        assets=["EURUSD"],
    )

    # Mock real adapter and real order execution
    fake_real_adapter = MagicMock()
    fake_receipt = MagicMock()
    fake_receipt.payout = 0.85
    fake_receipt.platform_trade_id = "plat_123"
    fake_real_adapter.place_trade.return_value = fake_receipt
    fake_real_adapter.get_open_trades.return_value = [
        {"id": "plat_123", "asset": "EURUSD", "direction": "CALL", "expiry_minutes": 15, "stake": 10.0}
    ]
    tm.adapter = fake_real_adapter
    tm.config = env["config"].model_copy(update={"mode": "live"}, deep=True)

    # Make repository insert raise
    env["repository"].insert = MagicMock(side_effect=RuntimeError("disk I/O error"))

    dec = ValidatedDecision(
        decision="CALL",
        is_valid=True,
        decision_id="dec_real",
        asset="EURUSD",
        direction="CALL",
        expiry_minutes=15,
        stake_amount=10.0,
        payout=0.85,
        shadow_only=False,
    )
    account.asset_modes["EURUSD"] = "live_eligible"
    adaptive_state = AdaptiveState(tier="NORMAL", real_trading_allowed=True, stake_scale=1.0)

    import os
    with patch.dict(os.environ, {"LIVE_CONFIRM": "YES"}):
        res = tm.execute(dec, account=account, adaptive_state=adaptive_state)
    assert env["breaker"].is_tripped() is True
    assert any("CRITICAL: Database write failure" in str(c[0][0]) for c in env["mock_alerter"].send_alert.call_args_list)


def test_record_mode_collects_candles_and_no_trades(test_env, mock_alerter):
    """Record mode collects candles and payouts, making no decisions or trades."""
    env = test_env
    clock = FakeClock(initial_time=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc))

    cfg = env["config"].model_copy(deep=True)
    cfg.mode = "record"

    mock_source = MagicMock()
    mock_source.now.return_value = clock.now()
    mock_source.get_assets.return_value = [
        MagicMock(asset_id="EURUSD", name="EURUSD", is_otc=False, payout=0.85, is_open=True)
    ]
    df_1m = pd.DataFrame({
        "ts": [clock.now() - timedelta(minutes=i) for i in range(50, 0, -1)],
        "ts_close": [clock.now() - timedelta(minutes=i-1) for i in range(50, 0, -1)],
        "open": [1.10] * 50,
        "high": [1.11] * 50,
        "low": [1.09] * 50,
        "close": [1.105] * 50,
        "volume": [100.0] * 50,
    })
    mock_source.get_candles.return_value = df_1m

    orch = Orchestrator(
        config=cfg,
        source=mock_source,
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=mock_alerter,
        clock=clock,
        paper_adapter=env["paper_adapter"],
    )

    results = orch.run_evaluation_cycle(clock.now())
    assert len(results) == 0
    assert len(env["repository"].get_trades()) == 0

    cycle_content = orch.cycle_log_path.read_text(encoding="utf-8")
    assert '"action": "RECORD"' in cycle_content


def test_heartbeat_message_content(test_env, mock_alerter):
    """Heartbeat message formatting and content."""
    env = test_env
    clock = FakeClock(initial_time=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc))
    orch = Orchestrator(
        config=env["config"],
        source=MagicMock(),
        adapter=env["paper_adapter"],
        store=env["store"],
        repository=env["repository"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        kill_switch=env["kill_switch"],
        trade_manager=env["trade_manager"],
        alerter=mock_alerter,
        clock=clock,
        paper_adapter=env["paper_adapter"],
    )

    orch.run_heartbeat_job(clock.now())
    assert mock_alerter.send_alert.called
    msg = mock_alerter.send_alert.call_args[0][0]
    assert "💓 *Heartbeat*: OK" in msg
    assert "Mode: paper" in msg
    assert "Tier: NORMAL | Breaker: OK" in msg
    assert "Balance: $1,000.00" in msg


def test_cli_main_demo_live_rejection(capsys):
    """CLI run rejects demo and live modes with exit code 1 and required message."""
    import main as main_cli

    res_demo = main_cli.main(["run", "--mode", "demo"])
    assert res_demo == 1
    captured = capsys.readouterr()
    assert "not enabled until Phase 10" in captured.err

    res_live = main_cli.main(["run", "--mode", "live"])
    assert res_live == 1
    captured2 = capsys.readouterr()
    assert "not enabled until Phase 10" in captured2.err


def test_cli_main_status_and_summary(capsys):
    """CLI status and summary commands run and output formatted sections."""
    import main as main_cli

    res_status = main_cli.main(["status"])
    assert res_status == 0
    cap_status = capsys.readouterr()
    assert "TRADING SYSTEM STATUS" in cap_status.out

    res_summary = main_cli.main(["summary", "--date", "2026-10-31"])
    assert res_summary == 0
    cap_summary = capsys.readouterr()
    assert "Daily Performance Report" in cap_summary.out


def test_telegram_payload_preserves_unicode():
    """Telegram outgoing alert preserves unicode and emojis in HTTP payload."""
    alerter = TelegramAlerter(token="test_bot_token", chat_id="987654")
    sample_text = "🚀 Alert: Win +$8.50 📈 [Wilson CI: 50.0% – 75.0%]"

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        alerter.send_alert(sample_text)

        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        payload_data = req.data.decode("utf-8")
        parsed = urllib.parse.parse_qs(payload_data)
        assert parsed["text"][0] == sample_text


def test_daily_summary_known_numbers(test_env, tmp_path):
    """Daily summary computes exact expected metrics for known trades and skip reasons."""
    env = test_env
    repo = env["repository"]
    date_str = "2026-10-31"

    # Seed 10 wins ($10 stake, 0.85 payout, +$8.50 pnl), 5 losses (-$10 pnl), 1 tie (-$10 pnl)
    # Total settled: 16 trades
    # Wins: 10, Losses: 5, Ties: 1
    # Win rate: 10/16 = 62.5%
    # Total PnL: 10 * 8.50 - 6 * 10 = +$25.00
    # Total Stake: 16 * $10 = $160.00
    # Realized Return: 25 / 160 = +15.625%
    for i in range(10):
        repo.insert(TradeRecord(
            id=f"win_{i}",
            opened_at=f"{date_str}T10:00:00Z",
            closed_at=f"{date_str}T10:15:00Z",
            asset="EURUSD",
            is_otc=False,
            direction="CALL",
            expiry_minutes=15,
            stake=10.0,
            payout_at_entry=0.85,
            result="WIN",
            pnl=8.50,
            mode="paper",
        ))
    for i in range(5):
        repo.insert(TradeRecord(
            id=f"loss_{i}",
            opened_at=f"{date_str}T11:00:00Z",
            closed_at=f"{date_str}T11:15:00Z",
            asset="EURUSD",
            is_otc=False,
            direction="PUT",
            expiry_minutes=15,
            stake=10.0,
            payout_at_entry=0.85,
            result="LOSS",
            pnl=-10.0,
            mode="paper",
        ))
    repo.insert(TradeRecord(
        id="tie_0",
        opened_at=f"{date_str}T12:00:00Z",
        closed_at=f"{date_str}T12:15:00Z",
        asset="EURUSD",
        is_otc=False,
        direction="CALL",
        expiry_minutes=15,
        stake=10.0,
        payout_at_entry=0.85,
        result="TIE",
        pnl=-10.0,
        mode="paper",
    ))

    skips = {"no payout": 4, "Market closed": 2}
    md, tg = generate_daily_summary_markdown(
        date_str=date_str,
        repository=repo,
        config=env["config"],
        state_manager=env["state_manager"],
        breaker=env["breaker"],
        assets=["EURUSD"],
        skip_counts=skips,
    )

    # Check metrics
    assert "Total Settled Trades**: 16" in md
    assert "Wins / Losses / Ties**: 10 / 5 / 1" in md
    assert "62.5%" in md
    assert "$25.00" in md
    assert "+15.62%" in md
    assert "Break-Even Rate**: 54.1%" in md
    assert "no payout | 4" in md
    assert "Market closed | 2" in md


def test_sanity_alarm_in_paper_mode_on_replay_source(test_env):
    """Sanity alarm triggers when source=replay and win_rate >= 70% with >= 100 trades, never for live public."""
    from orchestrator.reports import generate_daily_summary_markdown

    repo = test_env["repository"]
    date_str = "2026-10-15"

    # Insert 100 trades: 75 wins, 25 losses (75% win rate)
    for i in range(75):
        repo.insert(TradeRecord(
            id=f"win_alarm_{i}",
            opened_at=f"{date_str}T08:00:00Z",
            closed_at=f"{date_str}T08:15:00Z",
            asset="EURUSD",
            is_otc=False,
            direction="CALL",
            expiry_minutes=15,
            stake=10.0,
            payout_at_entry=0.88,
            result="WIN",
            pnl=8.8,
            mode="paper",
        ))
    for i in range(25):
        repo.insert(TradeRecord(
            id=f"loss_alarm_{i}",
            opened_at=f"{date_str}T09:00:00Z",
            closed_at=f"{date_str}T09:15:00Z",
            asset="EURUSD",
            is_otc=False,
            direction="CALL",
            expiry_minutes=15,
            stake=10.0,
            payout_at_entry=0.88,
            result="LOSS",
            pnl=-10.0,
            mode="paper",
        ))

    # Case 1: Replay source -> ALARM triggers
    test_env["config"].source.name = "replay"
    md_replay, tg_replay = generate_daily_summary_markdown(
        date_str=date_str,
        repository=repo,
        config=test_env["config"],
        state_manager=test_env["state_manager"],
        breaker=test_env["breaker"],
        assets=["EURUSD"],
    )
    assert "SANITY ALARM: Observed test set win rate >= 70% with >= 100 trades!" in md_replay
    assert "> [!WARNING]" in md_replay
    assert "⚠️ SANITY ALARM" in tg_replay

    # Case 2: Live public source -> ALARM NEVER triggers
    test_env["config"].source.name = "crypto_public"
    md_live, tg_live = generate_daily_summary_markdown(
        date_str=date_str,
        repository=repo,
        config=test_env["config"],
        state_manager=test_env["state_manager"],
        breaker=test_env["breaker"],
        assets=["EURUSD"],
    )
    assert "SANITY ALARM" not in md_live
    assert "SANITY ALARM" not in tg_live

