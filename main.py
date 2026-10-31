"""Automated Fixed-Time Trading System — Main Entry Point.

PROJECT_SPEC.md Phase 9:
  - CLI:
      run: python main.py run --mode record|paper --source replay|crypto_public
      status: python main.py status
      summary: python main.py summary --date YYYY-MM-DD
  - Signal handlers for graceful SIGINT and SIGTERM.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import signal
import sys
from typing import Optional

from adapters.paper import PaperAdapter
from alerts.telegram import TelegramAlerter, create_operator_command_handlers
from core.clock import RealClock
from core.config import AppConfig, load_config
from core.logging_setup import get_logger
from data.price_source import CryptoPublicSource, PriceSource, ReplaySource
from data.store import DataStore, TradeRepository
from execution.state import StateManager
from execution.trade_manager import TradeManager
from orchestrator.engine import Orchestrator
from orchestrator.reports import generate_daily_summary_markdown
from orchestrator.telegram_worker import TelegramWorker
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch
from selection.stats import asset_mode, compute_asset_stats

logger = logging.getLogger("trading_system.main")


def build_price_source(config: AppConfig, source_name: str) -> PriceSource:
    clock = RealClock()
    name = source_name.lower()
    if name == "replay":
        return ReplaySource(clock=clock, config=config)
    elif name == "crypto_public":
        return CryptoPublicSource(clock=clock, config=config)
    else:
        raise ValueError(f"Unsupported source '{source_name}'. Permitted in Phase 9: 'replay', 'crypto_public'.")


def handle_run(args: argparse.Namespace) -> int:
    config: AppConfig = load_config(args.config)

    # CLI parameter overrides
    if args.mode:
        config.mode = args.mode
    if args.source:
        config.source.name = args.source

    # Fail closed check on mode
    if config.mode in {"demo", "live"}:
        sys.stderr.write(f"Startup refused: mode '{config.mode}' is not enabled until Phase 10\n")
        return 1

    if config.mode not in {"record", "paper"}:
        sys.stderr.write(f"Startup refused: unsupported mode '{config.mode}'. Only 'record' and 'paper' allowed.\n")
        return 1

    print(f"Initializing Trading System in {config.mode} mode (source: {config.source.name})...")

    clock = RealClock()
    db_path = getattr(config.data, "db_path", "data/market.db") if hasattr(config, "data") else "data/market.db"
    store = DataStore(db_path=db_path)
    repository = TradeRepository(store=store)
    source = build_price_source(config, config.source.name)

    state_path = getattr(config.execution, "state_file_path", "state/trading_state.json")
    state_manager = StateManager(state_file=state_path)
    breaker = CircuitBreaker(config=config)
    kill_switch = KillSwitch()

    alerter = TelegramAlerter()
    breaker.alerter = alerter

    paper_adapter = PaperAdapter(
        initial_balance=float(getattr(config.trading, "initial_balance", 1000.0)),
        tie_policy=getattr(config.trading, "tie_policy", "loss"),
    )

    trade_manager = TradeManager(
        config=config,
        state_manager=state_manager,
        breaker=breaker,
        kill_switch=kill_switch,
        adapter=paper_adapter,
        paper_adapter=paper_adapter,
        repository=repository,
        alerter=alerter,
    )

    telegram_worker: Optional[TelegramWorker] = None
    if alerter.is_configured:
        handlers = create_operator_command_handlers(
            state_manager=state_manager,
            breaker=breaker,
            kill_switch=kill_switch,
            config=config,
            trade_manager=trade_manager,
        )
        telegram_worker = TelegramWorker(
            alerter=alerter,
            state_manager=state_manager,
            config=config,
            clock=clock,
            command_handlers=handlers,
        )

    orchestrator = Orchestrator(
        config=config,
        source=source,
        adapter=paper_adapter,
        store=store,
        repository=repository,
        state_manager=state_manager,
        breaker=breaker,
        kill_switch=kill_switch,
        trade_manager=trade_manager,
        alerter=alerter,
        clock=clock,
        paper_adapter=paper_adapter,
        telegram_worker=telegram_worker,
        is_first_run=bool(args.first_run),
    )

    # Signal handlers
    def _sig_handler(sig: int, frame: Any) -> None:
        print(f"\nReceived signal {sig}. Initiating graceful shutdown...")
        orchestrator.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        orchestrator.run_loop()
    except KeyboardInterrupt:
        orchestrator.stop()
    except Exception as exc:
        logger.error("Fatal exception in orchestrator: %s", exc, exc_info=True)
        return 1

    print("Trading System shutdown completed cleanly.")
    return 0


def handle_status(args: argparse.Namespace) -> int:
    config: AppConfig = load_config(args.config)
    state_path = getattr(config.execution, "state_file_path", "state/trading_state.json")
    state_manager = StateManager(state_file=state_path)
    state_manager.load()
    data = state_manager.data

    breaker = CircuitBreaker(config=config)
    kill_switch = KillSwitch()

    db_path = getattr(config.data, "db_path", "data/market.db") if hasattr(config, "data") else "data/market.db"
    store = DataStore(db_path=db_path)
    repo = TradeRepository(store=store)

    bal = data.get("balance", 0.0)
    open_trades = data.get("open_trades", [])
    tier = data.get("adaptive_state", {}).get("tier", "NORMAL")
    brk_status = "TRIPPED" if breaker.is_tripped() else "OK"
    ks_status = "ACTIVE" if kill_switch.is_active() else "OFF"
    last_reselect = data.get("last_asset_reselection_time", "N/A")

    print("==================================================")
    print("               TRADING SYSTEM STATUS              ")
    print("==================================================")
    print(f"Mode:                 {config.mode}")
    print(f"Balance:              ${bal:,.2f}")
    print(f"Open Paper Trades:    {len(open_trades)}")
    print(f"Adaptive Tier:        {tier}")
    print(f"Circuit Breaker:      {brk_status}")
    print(f"Kill Switch:          {ks_status}")
    print(f"Last Asset Reselect:  {last_reselect}")
    print("--------------------------------------------------")
    print("Paper Gate Progress:")

    all_trades = repo.get_trades()
    now_utc = datetime.now(timezone.utc)
    for asset in config.source.assets:
        st = compute_asset_stats(trades=all_trades, asset=asset, mode="paper", as_of=now_utc)
        if st:
            mode_str, reason = asset_mode(st, config)
            print(f"  • {asset:<10}: {st.n:>3}/{config.paper_gate.min_paper_trades} trades | WR: {st.win_rate:>6.1%} (BE {st.break_even:>6.1%}) -> [{mode_str.upper()}]")
        else:
            print(f"  • {asset:<10}:   0/{config.paper_gate.min_paper_trades} trades -> [PAPER]")
    print("==================================================")
    return 0


def handle_summary(args: argparse.Namespace) -> int:
    config: AppConfig = load_config(args.config)
    db_path = getattr(config.data, "db_path", "data/market.db") if hasattr(config, "data") else "data/market.db"
    store = DataStore(db_path=db_path)
    repo = TradeRepository(store=store)

    state_path = getattr(config.execution, "state_file_path", "state/trading_state.json")
    state_manager = StateManager(state_file=state_path)
    state_manager.load()
    breaker = CircuitBreaker(config=config)

    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    md_content, tg_msg = generate_daily_summary_markdown(
        date_str=date_str,
        repository=repo,
        config=config,
        state_manager=state_manager,
        breaker=breaker,
    )

    base_reports_dir = Path(getattr(config.orchestrator, "reports_dir", "reports/daily"))
    reports_dir = base_reports_dir if base_reports_dir.name == "daily" else base_reports_dir / "daily"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"{date_str}.md"
    report_file.write_text(md_content, encoding="utf-8")

    print(md_content)
    print(f"\nReport saved to: {report_file}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Automated Fixed-Time Trading System")
    subparsers = parser.add_subparsers(dest="command", help="System command to execute")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Start the trading system orchestrator")
    run_parser.add_argument("--mode", choices=["record", "paper", "demo", "live"], help="Execution mode")
    run_parser.add_argument("--source", choices=["replay", "crypto_public", "binomo_readonly"], help="Price source")
    run_parser.add_argument("--first-run", action="store_true", help="Explicit first run flag initializing state")
    run_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    # Command: status
    status_parser = subparsers.add_parser("status", help="Display current system status and paper gate progress")
    status_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    # Command: summary
    summary_parser = subparsers.add_parser("summary", help="Generate daily performance summary")
    summary_parser.add_argument("--date", help="Date in YYYY-MM-DD format (defaults to today)")
    summary_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    args = parser.parse_args(argv)

    if args.command == "run":
        return handle_run(args)
    elif args.command == "status":
        return handle_status(args)
    elif args.command == "summary":
        return handle_summary(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
