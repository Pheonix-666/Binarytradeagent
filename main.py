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
from core.env import check_environment, load_project_env
from core.logging_setup import get_logger

# Load .env at module startup with override=False (real environment variables win)
load_project_env(override=False)
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


def build_price_source(config: AppConfig, source_name: str, session: Optional[Any] = None) -> PriceSource:
    clock = RealClock()
    name = source_name.lower()
    if name == "replay":
        return ReplaySource(clock=clock, config=config)
    elif name == "crypto_public":
        return CryptoPublicSource(clock=clock, config=config)
    elif name == "binomo_readonly":
        from adapters.binomo.session import BinomoSession
        from adapters.binomo.reader import BinomoReadOnlySource
        s = session or BinomoSession(config=config)
        if session is None:
            s.start(headed=True)
            s.ensure_logged_in(timeout=300)
        return BinomoReadOnlySource(session=s, config=config)
    else:
        raise ValueError(f"Unsupported source '{source_name}'. Permitted: 'replay', 'crypto_public', 'binomo_readonly'.")


def handle_run(args: argparse.Namespace) -> int:
    config: AppConfig = load_config(args.config)

    # CLI parameter overrides
    if args.mode:
        config.mode = args.mode
    if args.source:
        config.source.name = args.source

    # Fail closed check on mode: live mode stays rejected in Phase 10B
    if config.mode == "live":
        sys.stderr.write("Startup refused: mode 'live' is rejected in Phase 10B (demo only).\n")
        return 1

    if config.mode not in {"record", "paper", "demo"}:
        sys.stderr.write(f"Startup refused: unsupported mode '{config.mode}'. Only 'record', 'paper', and 'demo' allowed.\n")
        return 1

    print(f"Initializing Trading System in {config.mode} mode (source: {config.source.name})...")

    clock = RealClock()
    db_path = getattr(config.data, "db_path", "data/market.db") if hasattr(config, "data") else "data/market.db"
    store = DataStore(db_path=db_path)
    repository = TradeRepository(store=store)

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

    active_adapter: PlatformAdapter = paper_adapter
    binomo_session: Optional[Any] = None

    if config.mode == "demo":
        from adapters.binomo.session import BinomoSession
        from adapters.binomo.reader import BinomoReadOnlySource
        from adapters.binomo.trader import BinomoAdapter, SelectorResolutionError

        # 1. Startup requirement: Telegram configured (warn if not)
        if not alerter.is_configured:
            warn_msg = "WARNING: Telegram alerts are unset; operator notifications and approval tickets will be disabled."
            logger.warning(warn_msg)
            print(warn_msg)

        # 2. Startup requirement: execution.demo_bypasses_paper_gate: true
        # Document why:
        # Paper gate requires 200 paper trades per asset before promoting to live_eligible.
        # In demo mode, we are trading simulated funds on the broker platform to test the live-like
        # adapter wiring, UI automation, and execution pipelines. Requiring 200 paper trades on each asset
        # would block demo testing unless bypassed.
        if not getattr(config.execution, "demo_bypasses_paper_gate", False):
            logger.info("Setting execution.demo_bypasses_paper_gate = True for demo mode.")
            config.execution.demo_bypasses_paper_gate = True

        # 3. Startup requirement: logged-in session with demo detected
        print("Connecting to Binomo headed session for demo execution...")
        binomo_session = BinomoSession(config=config)
        binomo_session.start(headed=True)
        try:
            binomo_session.ensure_logged_in(timeout=300.0)
        except Exception as exc:
            sys.stderr.write(f"Startup refused: Binomo session login verification failed: {exc}\n")
            binomo_session.close()
            return 1

        if not binomo_session.is_demo():
            err_msg = "Startup refused: Active account on Binomo is NOT a demo account (real account detected)!"
            sys.stderr.write(f"{err_msg}\n")
            logger.critical(err_msg)
            binomo_session.close()
            return 1

        # Use shared session for price source
        source = BinomoReadOnlySource(session=binomo_session, config=config, alerter=alerter)

        # Build BinomoAdapter
        binomo_adapter = BinomoAdapter(
            session=binomo_session,
            ws_listener=binomo_session.ws_listener,
            reader=source,
            config=config,
            breaker=breaker,
            alerter=alerter,
        )

        # 4. Startup requirement: binomo.ui validated
        try:
            binomo_adapter.validate_ui_selectors(binomo_session.page)
        except SelectorResolutionError as exc:
            sys.stderr.write(f"Startup refused: {exc}\n")
            binomo_session.close()
            return 1

        # 5. Startup requirement: platform stake limits loaded
        min_stk, max_stk = binomo_adapter.load_platform_limits()
        print(f"Loaded platform currency stake limits: min=${min_stk:.2f}, max=${max_stk:.2f}")

        active_adapter = binomo_adapter
    else:
        source = build_price_source(config, config.source.name)

    trade_manager = TradeManager(
        config=config,
        state_manager=state_manager,
        breaker=breaker,
        kill_switch=kill_switch,
        adapter=active_adapter,
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
        adapter=active_adapter,
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
    finally:
        if binomo_session:
            try:
                binomo_session.close()
            except Exception:
                pass

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
    tier = (data.get("adaptive_state") or {}).get("tier", "NORMAL")
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


def handle_binomo_calibrate(args: argparse.Namespace) -> int:
    config: AppConfig = load_config(args.config)
    from adapters.binomo.session import BinomoSession
    from adapters.binomo.reader import BinomoReadOnlySource
    from adapters.binomo.calibrate import run_calibration

    print("Opening Binomo session for calibration...")
    session = BinomoSession(config=config)
    try:
        session.start(headed=True)
        print("Waiting for login...")
        session.ensure_logged_in(timeout=300)
        reader = BinomoReadOnlySource(session=session, config=config)
        duration = getattr(args, "duration", 300)
        interval = getattr(args, "interval", 20)
        run_calibration(
            session=session,
            reader=reader,
            duration_seconds=duration,
            interval_seconds=interval,
        )
    finally:
        session.close()
    return 0


def handle_check_env(args: argparse.Namespace) -> int:
    config: AppConfig = load_config(args.config)
    if getattr(args, "mode", None):
        config.mode = args.mode
    exit_code, lines = check_environment(config)
    print("\n".join(lines))
    return exit_code


def handle_binomo_validate_ui(args: argparse.Namespace) -> int:
    """Validate Binomo UI selectors against active page without clicking, typing or filling."""
    from adapters.binomo.session import BinomoSession
    from adapters.binomo.trader import check_selector_spec, resolve_locator

    config: AppConfig = load_config(args.config)
    session = BinomoSession(config=config)
    has_error = False

    try:
        page = session.start(headed=True)
        print("Waiting for login... Please log in manually in the browser window.")
        session.ensure_logged_in(timeout=300.0)

        if not session.is_demo():
            sys.stderr.write("ERROR: Demo account required for UI validation, but live account detected.\n")
            return 1

        ui_cfg = getattr(config.binomo, "ui", None)
        if not ui_cfg:
            sys.stderr.write("ERROR: Missing binomo.ui configuration.\n")
            return 1

        print("Validating Binomo UI controls...")

        controls = [
            ("amount_input", getattr(ui_cfg, "amount_input", None), {}),
            ("expiry_selector", getattr(ui_cfg, "expiry_selector", None), {}),
            ("call_button", getattr(ui_cfg, "call_button", None), {}),
            ("put_button", getattr(ui_cfg, "put_button", None), {}),
            ("asset_selector", getattr(ui_cfg, "asset_selector", None), {}),
            ("demo_indicator", getattr(ui_cfg, "demo_indicator", None), {}),
            ("active_asset_label", getattr(ui_cfg, "active_asset_label", None), {}),
            ("active_expiry_label", getattr(ui_cfg, "active_expiry_label", None), {}),
        ]

        val_samples = getattr(ui_cfg, "validation_samples", {}) or {}
        sample_exp = val_samples.get("expiry_option") or val_samples.get("expiry_label") or "1 min"
        sample_asset = val_samples.get("asset_option") or val_samples.get("asset_name") or "EUR/USD"

        templates = [
            ("expiry_option", getattr(ui_cfg, "expiry_option", None), {"label": sample_exp}),
            ("asset_option", getattr(ui_cfg, "asset_option", None), {"name": sample_asset}),
        ]

        active_labels = {"active_asset_label", "active_expiry_label", "demo_indicator"}

        for name, spec, kwargs in controls + templates:
            err = check_selector_spec(name, spec)
            if err:
                print(f"{name}: NOT FOUND ({err})")
                has_error = True
                continue

            try:
                loc = resolve_locator(page, spec, **kwargs)
                cnt = loc.count() if hasattr(loc, "count") else 0
                if cnt > 0:
                    print(f"{name}: resolved (count {cnt})")
                    if name in active_labels:
                        txt = ""
                        try:
                            first_loc = loc.first if hasattr(loc, "first") else loc
                            if hasattr(first_loc, "inner_text"):
                                txt = first_loc.inner_text()
                            elif hasattr(first_loc, "text_content"):
                                txt = first_loc.text_content()
                        except Exception:
                            pass
                        print(f"  visible text: '{txt.strip()}'")
                else:
                    print(f"{name}: NOT FOUND")
                    has_error = True
            except Exception as exc:
                print(f"{name}: NOT FOUND ({exc})")
                has_error = True

        call_labels = getattr(ui_cfg, "call_button_labels", []) or []
        if not call_labels or any(lbl == "TODO_CALIBRATE" for lbl in call_labels):
            print("call_button_labels: NOT FOUND (uncalibrated or empty)")
            has_error = True
        else:
            print(f"call_button_labels: resolved ({call_labels})")

        put_labels = getattr(ui_cfg, "put_button_labels", []) or []
        if not put_labels or any(lbl == "TODO_CALIBRATE" for lbl in put_labels):
            print("put_button_labels: NOT FOUND (uncalibrated or empty)")
            has_error = True
        else:
            print(f"put_button_labels: resolved ({put_labels})")

    except Exception as exc:
        sys.stderr.write(f"Validation error: {exc}\n")
        has_error = True
    finally:
        session.close()

    return 1 if has_error else 0


def main(argv: Optional[list[str]] = None) -> int:
    # Check if custom --env-file is supplied
    custom_env: Optional[str] = None
    args_list = argv if argv is not None else sys.argv[1:]
    for idx, a in enumerate(args_list):
        if a == "--env-file" and idx + 1 < len(args_list):
            custom_env = args_list[idx + 1]
            break
        elif a.startswith("--env-file="):
            custom_env = a.split("=", 1)[1]
            break

    load_project_env(env_path=custom_env, override=False)
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

    # Command: binomo-calibrate
    calibrate_parser = subparsers.add_parser("binomo-calibrate", help="Run passive trading panel calibration")
    calibrate_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    calibrate_parser.add_argument("--duration", type=int, default=300, help="Calibration duration in seconds (default: 300)")
    calibrate_parser.add_argument("--interval", type=int, default=20, help="Snapshot interval in seconds (default: 20)")

    # Command: check-env
    check_env_parser = subparsers.add_parser("check-env", help="Check expected environment variables and active system configuration")
    check_env_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    check_env_parser.add_argument("--mode", choices=["record", "paper", "demo", "live"], help="Execution mode override")
    check_env_parser.add_argument("--env-file", default=None, help="Path to custom .env file")

    # Command: binomo-validate-ui
    validate_ui_parser = subparsers.add_parser("binomo-validate-ui", help="Validate Binomo UI selectors without interacting")
    validate_ui_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    args = parser.parse_args(argv)

    if args.command == "run":
        return handle_run(args)
    elif args.command == "status":
        return handle_status(args)
    elif args.command == "summary":
        return handle_summary(args)
    elif args.command == "binomo-calibrate":
        return handle_binomo_calibrate(args)
    elif args.command == "check-env":
        return handle_check_env(args)
    elif args.command == "binomo-validate-ui":
        return handle_binomo_validate_ui(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
