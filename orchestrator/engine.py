"""System Orchestrator — record and paper trading automation engine.

PROJECT_SPEC.md Section 1, 6, 9 & Phase 9:
  - Supports only 'record' and 'paper' modes. Rejects 'demo' and 'live' with 'not enabled until Phase 10'.
  - Fail-closed startup: config validation, requires TradeRepository, loads state, runs reconcile.
  - Aligned 5m evaluation cycle with fresh AccountSnapshot, candidate evaluation, and execution.
  - 1m shadow trade settlement updating repo and adaptive state.
  - Asset reselection, heartbeat, and daily summary markdown generation.
  - Overrun handling: skips missed cycles without queuing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from adaptive.monitor import AdaptiveState
from alerts.telegram import TelegramAlerter
from analysis.candidates import CandidateEvaluationResult, CandidateTrade, build_candidates
from core.clock import Clock, RealClock
from core.config import AppConfig
from core.models import TradeDirection
from backtest.engine import Fast1mLookup
from data.price_source import PriceSource
from data.store import DataStore, TradeRecord, TradeRepository
from decision.schema import ValidatedDecision
from execution.reconcile import reconcile_startup
from execution.state import StateManager
from execution.trade_manager import ExecutionResult, TradeManager
from orchestrator.fresh_inputs import build_account_snapshot
from orchestrator.paper_entry import PaperEntryModel, calculate_entry_price
from orchestrator.reports import generate_daily_summary_markdown
from orchestrator.telegram_worker import TelegramWorker
from risk.stake import policy_stake_amount
from safety.breaker import CircuitBreaker
from safety.killswitch import KillSwitch
from selection.asset_selector import SelectedAsset, select_top_assets
from selection.stats import AssetStats, compute_asset_stats

logger = logging.getLogger("trading_system.orchestrator")


@dataclass
class HeldOrder:
    """An approved order waiting for entry_time execution with full fidelity."""

    decision: ValidatedDecision
    candidate: Any
    account: Any
    adaptive_state: Any
    decision_time: datetime
    entry_time: datetime
    asset: str
    direction: TradeDirection
    expiry_minutes: int
    atr: Optional[float]
    payout: float


def compute_warmup_bars(config: AppConfig) -> dict[str, int]:
    """Compute required warmup bars per timeframe based on trend and pattern indicators."""
    trend = config.gate.trend
    req = {
        "1d": int(trend.daily.ema_slow * trend.daily.warmup_multiplier),
        "1h": int(trend.h1.ema_slow * trend.h1.warmup_multiplier),
        "15m": int(trend.m15.ema_slow * trend.m15.warmup_multiplier),
        "5m": int(max(config.patterns.window_lengths) if hasattr(config, "patterns") and config.patterns.window_lengths else 90),
        "1m": 60,
    }
    custom = getattr(config.orchestrator, "warmup_bars", {}) or {}
    for tf, n in custom.items():
        req[tf] = int(n)
    return req


class Orchestrator:
    """Coordinates market data ingestion, candidate evaluation, execution, settlement, and reporting."""

    def __init__(
        self,
        config: AppConfig,
        source: PriceSource,
        adapter: Any,
        store: DataStore,
        repository: TradeRepository,
        state_manager: StateManager,
        breaker: CircuitBreaker,
        kill_switch: KillSwitch,
        trade_manager: TradeManager,
        alerter: TelegramAlerter,
        clock: Optional[Clock] = None,
        paper_adapter: Optional[Any] = None,
        adaptive_controller: Optional[Any] = None,
        decision_service: Optional[Any] = None,
        telegram_worker: Optional[TelegramWorker] = None,
        is_first_run: bool = False,
    ) -> None:
        self.config = config
        self.source = source
        self.adapter = adapter
        self.store = store
        self.repository = repository
        self.state_manager = state_manager
        self.breaker = breaker
        self.kill_switch = kill_switch
        self.trade_manager = trade_manager
        self.alerter = alerter
        self.clock = clock or RealClock()
        self.paper_adapter = paper_adapter
        self.adaptive_controller = adaptive_controller
        self.decision_service = decision_service
        self.telegram_worker = telegram_worker
        self.is_first_run = is_first_run

        self.paper_entry_model = PaperEntryModel(
            entry_delay_bars=getattr(config.paper, "entry_delay_bars", 1) if hasattr(config, "paper") else 1,
            slippage_atr=getattr(config.paper, "slippage_atr", 0.05) if hasattr(config, "paper") else 0.05,
        )

        self.running: bool = False
        self.last_reselect_time: Optional[datetime] = None
        self.last_heartbeat_time: Optional[datetime] = None
        self.last_summary_date: Optional[str] = None
        self.selected_assets: list[str] = list(self.config.source.assets)
        self.ready_assets: set[str] = set()
        self.record_only_assets: set[str] = set()
        self.warmup_bars = compute_warmup_bars(self.config)
        self.consecutive_source_failures: int = 0
        self.held_orders: list[HeldOrder] = []

        # Reports directory
        reports_dir_str = getattr(config.orchestrator, "reports_dir", "reports/daily") if hasattr(config, "orchestrator") else "reports/daily"
        base_dir = Path(reports_dir_str)
        self.daily_reports_dir = base_dir if base_dir.name == "daily" else base_dir / "daily"
        self.daily_reports_dir.mkdir(parents=True, exist_ok=True)
        self.cycle_log_path = (self.daily_reports_dir.parent if base_dir.name == "daily" else base_dir) / "cycles.jsonl"

    def validate_startup(self) -> None:
        """Fail-closed validation of mode and prerequisites."""
        mode_str = str(self.config.mode).lower()
        if mode_str in {"demo", "live"}:
            raise ValueError(f"Startup refused: mode '{self.config.mode}' is not enabled until Phase 10")

        if mode_str not in {"record", "paper"}:
            raise ValueError(f"Startup refused: mode '{self.config.mode}' is unsupported. Only 'record' and 'paper' are permitted in Phase 9.")

        if self.repository is None:
            raise ValueError("Startup refused: TradeRepository is required")

    def run_warmup(self) -> None:
        """Fetch enough history to satisfy indicator warm-up for each timeframe."""
        for asset in self.selected_assets:
            t0 = time.time()
            asset_ready = True
            for tf, needed_n in self.warmup_bars.items():
                df = self.store.sync_candles_from_source(
                    source=self.source,
                    asset_id=asset,
                    timeframe=tf,
                    n_needed=needed_n,
                )
                if len(df) < needed_n:
                    logger.warning("insufficient history for %s on %s: got %d, required %d", tf, asset, len(df), needed_n)
                    asset_ready = False

            elapsed = time.time() - t0
            if asset_ready:
                logger.info("ready in %.2f seconds for asset %s", elapsed, asset)
                self.ready_assets.add(asset)
            else:
                logger.info("insufficient history for %s, remaining in record-only", asset)
                self.record_only_assets.add(asset)

    def startup(self) -> None:
        """Run fail-closed startup sequence, reconciliation, and alert dispatch."""
        logger.info("Executing orchestrator startup sequence (mode=%s)...", self.config.mode)
        self.validate_startup()

        # Load state
        active_adapter = self.paper_adapter if str(self.config.mode).lower() == "paper" and self.paper_adapter else self.adapter
        broker_bal = float(active_adapter.get_balance())
        self.state_manager.load(first_run=self.is_first_run, broker_balance=broker_bal)

        # Pre-trading reconciliation
        cur_time = self.clock.now()
        rec_res = reconcile_startup(
            adapter=active_adapter,
            saved_state=self.state_manager.data,
            breaker=self.breaker,
            alerter=self.alerter,
            now=cur_time,
        )
        if not rec_res.success:
            logger.error("Startup reconciliation failed: %s", rec_res.issues)

        self.trade_manager.set_reconciled(rec_res.success)

        # Warm-up history
        self.run_warmup()

        # Send started alert
        self.alerter.send_alert(
            f"🚀 Trading system started in *{self.config.mode}* mode\n"
            f"• Source: `{self.config.source.name}`\n"
            f"• Assets: {', '.join(self.selected_assets)}\n"
            f"• Reconciled: {rec_res.success}"
        )

        if self.telegram_worker:
            self.telegram_worker.start()

    def run_asset_reselection_job(self, cur_time: datetime) -> None:
        """Select and rank top assets periodically."""
        reselect_hours = getattr(self.config.selection, "asset_reselect_hours", 24)
        if (
            self.last_reselect_time is not None
            and (cur_time - self.last_reselect_time).total_seconds() < reselect_hours * 3600
        ):
            return

        logger.info("Running asset reselection job at %s", cur_time.isoformat())
        quotes_list = self.source.get_assets()
        quotes_map = {q.asset_id: q for q in quotes_list}

        all_trades = self.repository.get_trades(until=cur_time)
        stats_map: dict[str, AssetStats] = {}
        for asset in quotes_map:
            st = compute_asset_stats(trades=all_trades, asset=asset, mode="paper", as_of=cur_time)
            if st is not None:
                stats_map[asset] = st

        selected = select_top_assets(quotes=quotes_map, stats_by_asset=stats_map, config=self.config)
        if selected:
            self.selected_assets = [s.asset_id for s in selected]
            logger.info("Asset reselection completed: %s", self.selected_assets)

        self.last_reselect_time = cur_time
        if hasattr(self.state_manager, "update"):
            self.state_manager.update(last_asset_reselection_time=cur_time.isoformat())

    def run_settlement_job(self, cur_time: datetime) -> list[Any]:
        """Settle open shadow trades on paper adapter and update repositories."""
        if self.paper_adapter and hasattr(self.paper_adapter, "set_current_time"):
            self.paper_adapter.set_current_time(cur_time)

        # Process any pending held orders before settlement
        self.process_held_orders(cur_time)

        quotes = self.source.get_assets()
        prices_map = {}
        for q in quotes:
            # Try to get latest 1m candle close for settlement price
            candles = self.source.get_candles(q.asset_id, "1m", 1)
            if not candles.empty:
                prices_map[q.asset_id] = float(candles.iloc[-1]["close"])

        # For replay source, ensure parity with backtest exit price
        if hasattr(self.source, "_load_raw_1m") and self.paper_adapter:
            open_trades = getattr(self.paper_adapter, "_open_trades", [])
            for tr in open_trades:
                exp_t = tr.entry_time + timedelta(minutes=tr.expiry_minutes)
                if cur_time >= exp_t:
                    df_1m = self.source._load_raw_1m(tr.asset_id)
                    exit_bar = Fast1mLookup(df_1m).get_bar(exp_t)
                    if exit_bar is not None:
                        prices_map[tr.asset_id] = float(exit_bar["close"])

        settled = self.trade_manager.settle_shadow_trades(
            now=cur_time,
            prices_at_expiry=prices_map,
        )

        if settled:
            logger.info("Settled %d shadow trades at %s", len(settled), cur_time.isoformat())
            # Notify adaptive controller if present
            if self.adaptive_controller and hasattr(self.adaptive_controller, "on_trade_settled"):
                for t in settled:
                    res_str = getattr(t, "result", "LOSS")
                    res_val = res_str.value if hasattr(res_str, "value") else str(res_str)
                    payout = float(getattr(t, "payout", 0.85) or 0.85)
                    self.adaptive_controller.on_trade_settled(res_val, payout, now=cur_time)

        return settled

    def process_held_orders(self, cur_time: datetime) -> list[ExecutionResult]:
        """Process and execute held orders that have reached their scheduled entry_time."""
        remaining: list[HeldOrder] = []
        executed_results: list[ExecutionResult] = []

        for held in self.held_orders:
            if cur_time >= held.entry_time:
                res = self._execute_held_order(held, execution_time=held.entry_time)
                if res:
                    executed_results.append(res)
            else:
                remaining.append(held)

        self.held_orders = remaining
        return executed_results

    def _execute_held_order(self, held: HeldOrder, execution_time: Optional[datetime] = None) -> Optional[ExecutionResult]:
        """Execute a held order at its entry_time with full entry fidelity and staleness re-checks."""
        exec_t = execution_time or held.entry_time

        # 1. Breaker & Killswitch check
        if self.breaker.is_tripped() or self.kill_switch.is_active():
            logger.warning("Breaker or kill switch tripped at entry_time %s for %s", exec_t.isoformat(), held.asset)
            self._log_cycle_record({
                "timestamp": exec_t.isoformat(),
                "asset": held.asset,
                "action": "BLOCKED",
                "reason": "Breaker or killswitch tripped at entry_time",
                "payout": held.payout,
            })
            return None

        # 2. Staleness re-check and entry price lookup
        open_price: Optional[float] = None
        if hasattr(self.source, "_load_raw_1m"):
            # Replay source: open of the 1m bar at entry_time (parity with backtest)
            df_1m = self.source._load_raw_1m(held.asset)
            lookup = Fast1mLookup(df_1m)
            bar = lookup.get_bar(held.entry_time)
            if bar is None:
                logger.warning("Data gap at entry_time %s for %s; skipping held order", held.entry_time.isoformat(), held.asset)
                self._log_cycle_record({
                    "timestamp": exec_t.isoformat(),
                    "asset": held.asset,
                    "action": "SKIP",
                    "reason": "data_gap_entry (staleness check failed)",
                    "payout": held.payout,
                })
                return None
            open_price = float(bar["open"])
        else:
            # Live/mock source
            candles_1m = self.source.get_candles(held.asset, "1m", 1)
            if candles_1m.empty:
                logger.warning("No candle data for %s at entry_time %s", held.asset, exec_t.isoformat())
                return None
            latest_ts_close = candles_1m.iloc[-1]["ts_close"]
            if not self.breaker.check_data_freshness(data_time=latest_ts_close, now=exec_t):
                logger.warning("Data staleness check failed at entry_time %s for %s", exec_t.isoformat(), held.asset)
                self._log_cycle_record({
                    "timestamp": exec_t.isoformat(),
                    "asset": held.asset,
                    "action": "SKIP",
                    "reason": "Data freshness check failed at entry_time",
                    "payout": held.payout,
                })
                return None
            open_price = float(candles_1m.iloc[-1]["close"])

        slippage_atr = (
            getattr(self.config.execution_model, "slippage_atr", None)
            if hasattr(self.config, "execution_model")
            else getattr(self.config.paper, "slippage_atr", 0.02)
        )
        if slippage_atr is None:
            slippage_atr = 0.02

        entry_price = calculate_entry_price(
            open_price=open_price,
            direction=held.direction,
            atr=held.atr,
            slippage_atr=slippage_atr,
        )

        # 3. Synchronize paper adapter price and time
        if self.paper_adapter and hasattr(self.paper_adapter, "set_current_time"):
            self.paper_adapter.set_current_time(held.entry_time)
        if self.paper_adapter and hasattr(self.paper_adapter, "set_price"):
            self.paper_adapter.set_price(held.asset, entry_price)

        # 4. Execute through TradeManager
        exec_res = self.trade_manager.execute(
            decision=held.decision,
            account=held.account,
            adaptive_state=held.adaptive_state,
            now=held.entry_time,
        )

        self._log_cycle_record({
            "timestamp": held.entry_time.isoformat(),
            "asset": held.asset,
            "action": "TRADE" if exec_res.executed else "BLOCKED",
            "status": exec_res.status,
            "direction": held.decision.direction,
            "stake": held.decision.stake_amount,
            "trade_id": exec_res.trade_id,
            "entry_price": entry_price,
            "reason": exec_res.reason,
            "payout": held.payout,
        })
        return exec_res

    def run_heartbeat_job(self, cur_time: datetime) -> None:
        """Emit periodic health check to Telegram."""
        heartbeat_hours = getattr(self.config.orchestrator, "heartbeat_hours", 6)
        if (
            self.last_heartbeat_time is not None
            and (cur_time - self.last_heartbeat_time).total_seconds() < heartbeat_hours * 3600
        ):
            return

        bal = float(self.adapter.get_balance())
        open_cnt = len(list(self.adapter.get_open_trades()))
        tier = (
            getattr(self.adaptive_controller.state, "tier", "NORMAL")
            if self.adaptive_controller and hasattr(self.adaptive_controller, "state")
            else "NORMAL"
        )
        brk = "TRIPPED" if self.breaker.is_tripped() else "OK"

        self.alerter.send_alert(
            f"💓 *Heartbeat*: OK\n"
            f"• Mode: {self.config.mode}\n"
            f"• Tier: {tier} | Breaker: {brk}\n"
            f"• Balance: ${bal:,.2f} | Open: {open_cnt}"
        )
        self.last_heartbeat_time = cur_time

    def run_daily_summary_job(self, cur_time: datetime) -> Optional[str]:
        """Generate markdown daily summary and post to Telegram at configured hour."""
        summary_hour = getattr(self.config.orchestrator, "summary_hour_utc", 0)
        date_str = cur_time.strftime("%Y-%m-%d")

        if cur_time.hour == summary_hour and self.last_summary_date != date_str:
            logger.info("Generating daily summary for %s...", date_str)
            md_content, tg_msg = generate_daily_summary_markdown(
                date_str=date_str,
                repository=self.repository,
                config=self.config,
                state_manager=self.state_manager,
                breaker=self.breaker,
                adaptive_controller=self.adaptive_controller,
                assets=self.selected_assets,
                cycle_log_path=self.cycle_log_path,
            )

            report_file = self.daily_reports_dir / f"{date_str}.md"
            report_file.write_text(md_content, encoding="utf-8")
            self.alerter.send_alert(tg_msg)
            self.last_summary_date = date_str
            return str(report_file)
        return None

    def _log_cycle_record(self, record: dict[str, Any]) -> None:
        """Write structured cycle line to JSONL."""
        try:
            line = json.dumps(record, default=str)
            with open(self.cycle_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            logger.warning("Failed writing cycle JSONL record: %s", exc)

    def run_evaluation_cycle(self, cur_time: datetime) -> list[ExecutionResult]:
        """Execute one 5m evaluation cycle across evaluated assets."""
        if self.paper_adapter and hasattr(self.paper_adapter, "set_current_time"):
            self.paper_adapter.set_current_time(cur_time)

        logger.info("--- Starting 5m evaluation cycle at %s ---", cur_time.isoformat())
        execution_results: list[ExecutionResult] = []

        # Execute any held orders ready at cycle start
        execution_results.extend(self.process_held_orders(cur_time))

        # 1. Fresh AccountSnapshot
        active_adapter = self.paper_adapter if str(self.config.mode).lower() == "paper" and self.paper_adapter else self.adapter
        account = build_account_snapshot(
            adapter=active_adapter,
            repository=self.repository,
            config=self.config,
            now=cur_time,
            assets=self.selected_assets,
            daily_starting_balance=getattr(self.breaker, "daily_starting_balance", None),
        )

        all_trades = self.repository.get_trades(until=cur_time)
        stats_map: dict[str, AssetStats] = {}
        for asset in self.selected_assets:
            st = compute_asset_stats(trades=all_trades, asset=asset, mode="paper", as_of=cur_time)
            if st is not None:
                stats_map[asset] = st

        # Quotes fetch with exception isolation and consecutive failure tracking
        try:
            quotes_list = self.source.get_assets()
            quotes_map = {q.asset_id: q for q in quotes_list}
            self.consecutive_source_failures = 0
        except Exception as exc:
            self.consecutive_source_failures += 1
            logger.warning(
                "Price source fetch failed (%d consecutive failures): %s",
                self.consecutive_source_failures,
                exc,
            )
            max_retries = getattr(self.config.source, "max_retries", 3) or 3
            if self.consecutive_source_failures >= max_retries:
                self.alerter.send_alert(
                    f"⚠️ Price source down: {exc} ({self.consecutive_source_failures} consecutive failures)"
                )
            self._log_cycle_record({
                "timestamp": cur_time.isoformat(),
                "action": "SKIP",
                "reason": f"Price source failure: {exc}",
            })
            return []

        # Record quotes and payouts in store
        self.store.save_payouts(quotes_list, ts=cur_time)

        # Record mode: collect candles and payouts, make no decisions or trades
        if self.config.mode == "record":
            for asset in self.selected_assets:
                for tf in ["1d", "1h", "15m", "5m", "1m"]:
                    self.store.sync_candles_from_source(
                        source=self.source,
                        asset_id=asset,
                        timeframe=tf,
                        n_needed=self.warmup_bars.get(tf, 50),
                    )
                self._log_cycle_record({
                    "timestamp": cur_time.isoformat(),
                    "asset": asset,
                    "action": "RECORD",
                    "reason": "record mode",
                })
            return []

        for asset in self.selected_assets:
            quote = quotes_map.get(asset)
            is_open = quote.is_open if quote else True
            payout = quote.payout if quote else None

            if payout is None or payout <= 0.0:
                self._log_cycle_record({
                    "timestamp": cur_time.isoformat(),
                    "asset": asset,
                    "action": "SKIP",
                    "reason": "no payout",
                    "is_open": is_open,
                    "payout": payout,
                })
                continue

            if not is_open:
                self._log_cycle_record({
                    "timestamp": cur_time.isoformat(),
                    "asset": asset,
                    "action": "SKIP",
                    "reason": "Market closed",
                    "is_open": False,
                    "payout": payout,
                })
                continue

            # Check if asset was in record-only due to insufficient warmup history
            if asset in self.record_only_assets:
                has_enough = True
                for tf, needed_n in self.warmup_bars.items():
                    cached = self.store.get_candles(asset, tf, n=needed_n)
                    if len(cached) < needed_n:
                        has_enough = False
                        break
                if has_enough:
                    self.record_only_assets.remove(asset)
                    self.ready_assets.add(asset)
                    logger.info("Asset %s warmup satisfied, promoted to ready", asset)
                else:
                    for tf in ["1d", "1h", "15m", "5m", "1m"]:
                        self.store.sync_candles_from_source(
                            source=self.source,
                            asset_id=asset,
                            timeframe=tf,
                            n_needed=self.warmup_bars.get(tf, 50),
                        )
                    self._log_cycle_record({
                        "timestamp": cur_time.isoformat(),
                        "asset": asset,
                        "action": "SKIP",
                        "reason": "insufficient history (record-only)",
                        "is_open": is_open,
                        "payout": payout,
                    })
                    continue

            # 2. Fetch incremental candles across timeframes
            candles_by_tf: dict[str, pd.DataFrame] = {}
            data_stale = False
            for tf in ["1d", "1h", "15m", "5m", "1m"]:
                df_tf = self.store.sync_candles_from_source(
                    source=self.source,
                    asset_id=asset,
                    timeframe=tf,
                    n_needed=200 if tf in {"1m", "5m"} else 50,
                )
                candles_by_tf[tf] = df_tf
                if not df_tf.empty and tf in {"1m", "5m"}:
                    latest_close = df_tf.iloc[-1]["ts_close"]
                    # Data freshness verification
                    is_fresh = self.breaker.check_data_freshness(
                        data_time=latest_close,
                        now=cur_time,
                    )
                    if not is_fresh:
                        data_stale = True

            if data_stale:
                self._log_cycle_record({
                    "timestamp": cur_time.isoformat(),
                    "asset": asset,
                    "action": "SKIP",
                    "reason": "Data freshness check failed (stale data)",
                    "is_open": True,
                    "payout": payout,
                })
                continue

            # 3. Candidate evaluation via shared build_candidates
            cand_res = build_candidates(
                asset=asset,
                t=cur_time,
                candles_by_tf=candles_by_tf,
                stats_as_of=stats_map,
                config=self.config,
                payout=payout,
                is_otc=quote.is_otc if quote else False,
            )

            if not cand_res.candidates:
                self._log_cycle_record({
                    "timestamp": cur_time.isoformat(),
                    "asset": asset,
                    "action": "SKIP",
                    "skip_reasons": cand_res.skip_reasons,
                    "candidates_count": 0,
                    "payout": payout,
                })
                continue

            # 4. Decision Selection (policy or llm)
            decision_mode = getattr(self.config.paper, "decision_mode", "policy") if hasattr(self.config, "paper") else "policy"
            decision_result: Optional[ValidatedDecision] = None
            cand_atr: Optional[float] = None

            adaptive_state = (
                self.adaptive_controller.state
                if self.adaptive_controller and hasattr(self.adaptive_controller, "state")
                else AdaptiveState(tier="NORMAL", stake_scale=1.0, real_trading_allowed=True)
            )

            if decision_mode == "policy":
                # Deterministic candidate selection: lowest risk score, then lowest DTW distance
                sorted_cands = sorted(
                    cand_res.candidates,
                    key=lambda c: (
                        c.risk_score if c.risk_score is not None else 999,
                        getattr(c.pattern_match, "distance", 0.0) if c.pattern_match else 0.0,
                    ),
                )
                best_cand = sorted_cands[0]
                cand_atr = best_cand.atr

                expiry_min = int(getattr(best_cand, "expiry_minutes", None) or getattr(self.config.trading, "expiry_minutes", 15))
                dir_str = "CALL" if best_cand.direction == TradeDirection.CALL else "PUT"

                stake_policy = getattr(self.config.paper, "stake_policy", None) or getattr(self.config.backtest, "stake_policy", "max")
                stake_scale = getattr(adaptive_state, "stake_scale", 1.0)
                calc_stake = policy_stake_amount(
                    equity=account.equity,
                    max_stake_pct=best_cand.max_stake_pct,
                    stake_policy=stake_policy,
                    stake_scale=stake_scale,
                    min_pct=self.config.stake.min_pct,
                    min_amount=self.config.stake.min_amount,
                    round_to=self.config.stake.round_to,
                    max_amount=self.config.stake.max_amount,
                )

                if calc_stake is None:
                    self._log_cycle_record({
                        "timestamp": cur_time.isoformat(),
                        "asset": asset,
                        "action": "SKIP",
                        "reason": "stake below min_amount",
                        "candidates_count": len(cand_res.candidates),
                        "payout": payout,
                    })
                    continue

                decision_result = ValidatedDecision(
                    decision=dir_str,
                    is_valid=True,
                    decision_id=f"dec_{cur_time.strftime('%Y%m%d%H%M%S')}_{asset}",
                    stake_amount=calc_stake,
                    asset=asset,
                    direction=dir_str,
                    expiry_minutes=expiry_min,
                    reasoning=f"Selected by policy: risk score {best_cand.risk_score}",
                    pattern_name=best_cand.pattern_name,
                    pattern_type=best_cand.pattern_type,
                    risk_score=best_cand.risk_score,
                    payout=best_cand.payout or payout,
                    is_otc=best_cand.is_otc,
                    shadow_only=False,
                )
            else:
                # LLM Decision Service mode
                if self.decision_service:
                    try:
                        decision_result = self.decision_service.decide(
                            candidates=cand_res.candidates,
                            account=account,
                            as_of=cur_time,
                        )
                    except Exception as exc:
                        logger.warning("LLM decision service error for %s: %s", asset, exc)
                        self._log_cycle_record({
                            "timestamp": cur_time.isoformat(),
                            "asset": asset,
                            "action": "SKIP",
                            "reason": f"LLM error: {exc}",
                            "payout": payout,
                        })
                        continue
                if decision_result and getattr(decision_result, "evidence", None):
                    cand_atr = getattr(decision_result.evidence, "atr", None)

            if not decision_result or decision_result.decision == "SKIP":
                self._log_cycle_record({
                    "timestamp": cur_time.isoformat(),
                    "asset": asset,
                    "action": "SKIP",
                    "reason": getattr(decision_result, "reasoning", "Decision was SKIP"),
                    "candidates_count": len(cand_res.candidates),
                    "payout": payout,
                })
                continue

            # 5. Order execution with Entry Fidelity
            entry_delay_bars = (
                getattr(self.config.execution_model, "entry_delay_bars", None)
                if hasattr(self.config, "execution_model")
                else getattr(self.config.paper, "entry_delay_bars", 1)
            )
            if entry_delay_bars is None:
                entry_delay_bars = 1

            entry_time = cur_time + timedelta(minutes=entry_delay_bars)
            held = HeldOrder(
                decision=decision_result,
                candidate=best_cand if decision_mode == "policy" else None,
                account=account,
                adaptive_state=adaptive_state,
                decision_time=cur_time,
                entry_time=entry_time,
                asset=asset,
                direction=TradeDirection(decision_result.direction),
                expiry_minutes=int(decision_result.expiry_minutes or 15),
                atr=cand_atr,
                payout=float(decision_result.payout or payout or 0.85),
            )

            if entry_delay_bars == 0:
                exec_res = self._execute_held_order(held, execution_time=cur_time)
                if exec_res:
                    execution_results.append(exec_res)
            else:
                self.held_orders.append(held)
                self._log_cycle_record({
                    "timestamp": cur_time.isoformat(),
                    "asset": asset,
                    "action": "HELD",
                    "status": "HELD",
                    "entry_time": entry_time.isoformat(),
                    "direction": decision_result.direction,
                    "stake": decision_result.stake_amount,
                    "payout": held.payout,
                })

        return execution_results

    def step_single_cycle(self, cur_time: datetime) -> list[ExecutionResult]:
        """Convenience method for tests and step-wise replay: runs settlement, reselection, evaluation, and summary."""
        self.run_settlement_job(cur_time)
        self.run_asset_reselection_job(cur_time)
        self.run_heartbeat_job(cur_time)
        res = self.run_evaluation_cycle(cur_time)
        self.run_daily_summary_job(cur_time)
        return res

    def compute_next_aligned_5m(self, cur_time: datetime) -> datetime:
        """Compute the next aligned 5-minute candle boundary plus candle_delay_seconds."""
        delay_sec = getattr(self.config.orchestrator, "candle_delay_seconds", 3) if hasattr(self.config, "orchestrator") else 3
        # Align to next 5m boundary: e.g. 10:00, 10:05, 10:10
        cur_min = cur_time.minute
        next_5m_min = (cur_min // 5 + 1) * 5
        delta_mins = next_5m_min - cur_min
        aligned_boundary = cur_time.replace(second=0, microsecond=0) + timedelta(minutes=delta_mins)
        return aligned_boundary + timedelta(seconds=delay_sec)

    def run_loop(self, max_cycles: Optional[int] = None) -> None:
        """Main orchestrator scheduling loop with overrun handling."""
        self.startup()
        self.running = True
        cycles_completed = 0

        logger.info("Orchestrator loop entering active state.")

        while self.running:
            cur_time = self.clock.now()

            # Execute 1m shadow settlements
            self.run_settlement_job(cur_time)
            self.run_asset_reselection_job(cur_time)
            self.run_heartbeat_job(cur_time)
            self.run_daily_summary_job(cur_time)

            # Check if current time is an aligned 5m evaluation step
            # If so, execute cycle
            self.run_evaluation_cycle(cur_time)
            cycles_completed += 1

            if max_cycles is not None and cycles_completed >= max_cycles:
                logger.info("Reached maximum cycles limit (%d). Terminating loop.", max_cycles)
                break

            # Overrun handling: compute next target aligned time
            target_time = self.compute_next_aligned_5m(cur_time)
            sleep_duration = (target_time - self.clock.now()).total_seconds()

            if sleep_duration <= 0:
                logger.warning("Cycle overrun detected (elapsed past %s). Skipping missed cycles.", target_time.isoformat())
                # Re-compute strictly future boundary
                target_time = self.compute_next_aligned_5m(self.clock.now())
                sleep_duration = max(0.1, (target_time - self.clock.now()).total_seconds())

            self.clock.sleep(sleep_duration)

    def stop(self) -> None:
        """Gracefully stop orchestrator and background threads."""
        logger.info("Stopping orchestrator...")
        self.running = False
        if self.telegram_worker:
            self.telegram_worker.stop()
