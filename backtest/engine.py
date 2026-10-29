"""Deterministic backtesting engine with single code-path candidate evaluation.

PROJECT_SPEC.md Section 11 & Phase 6:
  - Steps through time at trading.check_interval_minutes.
  - Slices candles causally: ts <= t (zero look-ahead).
  - Evaluates candidates through shared build_candidates().
  - Applies deterministic LLM decision stand-in:
      * decision_policy: 'lowest_risk' (default) or 'highest_similarity'
      * stake_policy: 'max' (default), 'mid', or 'min'
  - Sizing & Portfolio rules:
      * Floored stake amount via stake_amount()
      * 35% total exposure cap on open stakes
      * Opposite-direction block per asset
      * Same-direction stacking allowed
  - Fills:
      * Entry at next 1m bar OPEN + slippage_atr * ATR against trade
      * Exit at entry_time + expiry 1m bar CLOSE
      * Tie policy: refund or loss (default loss)
  - Records trades to in-memory SQLite TradeRepository with mode 'backtest'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from analysis.candidates import CandidateTrade, build_candidates
from core.config import AppConfig
from core.logging_setup import get_logger
from core.models import TradeDirection
from data.store import DataStore, TradeRecord, TradeRepository
from decision.llm_client import FakeLLMClient, LLMClient, get_llm_client
from decision.schema import SystemState
from decision.service import DecisionService
from patterns.library import PatternLibrary, get_pattern_library
from risk.stake import compute_equity, stake_amount
from selection.stats import compute_asset_stats, hour_risk

logger = get_logger()


@dataclass
class OpenTrade:
    """An active position in the backtest engine awaiting expiry."""

    trade_id: str
    asset: str
    direction: TradeDirection
    stake: float
    entry_time: datetime
    entry_price: float
    expiry_minutes: int
    exit_time: datetime
    payout: float
    pattern_name: str
    pattern_type: str
    risk_score: int
    evidence: dict[str, Any] = field(default_factory=dict)


class Fast1mLookup:
    """High-performance O(log N) 1m bar lookup using sorted DatetimeIndex searchsorted."""

    def __init__(self, df_1m: pd.DataFrame):
        if df_1m.empty or "ts" not in df_1m.columns:
            self.ts_idx = pd.DatetimeIndex([])
            self.open_prices = np.array([], dtype=np.float64)
            self.close_prices = np.array([], dtype=np.float64)
        else:
            ts_utc = pd.to_datetime(df_1m["ts"], utc=True)
            self.ts_idx = pd.DatetimeIndex(ts_utc)
            self.open_prices = df_1m["open"].values.astype(np.float64)
            self.close_prices = df_1m["close"].values.astype(np.float64)

    def get_bar(self, t: datetime) -> Optional[dict[str, float]]:
        if len(self.ts_idx) == 0:
            return None
        if t.tzinfo is None:
            t_utc = t.replace(tzinfo=timezone.utc)
        else:
            t_utc = t.astimezone(timezone.utc)
        pos = self.ts_idx.searchsorted(t_utc)
        if pos < len(self.ts_idx) and self.ts_idx[pos] == t_utc:
            return {"open": float(self.open_prices[pos]), "close": float(self.close_prices[pos])}
        return None


def settle_position(
    ot: OpenTrade,
    exit_bar: Optional[dict[str, float]],
    tie_policy: str,
) -> tuple[str, float, float]:
    """Settle an open position causally.

    Returns:
        (result, pnl, cash_refund)
        where cash_refund is the total cash added back to current balance:
        - WIN: stake + (stake * payout)
        - LOSS: 0.0 (stake was already deducted at entry)
        - TIE (refund): stake
        - TIE (loss): 0.0
        - VOID (missing exit bar): stake (full refund)
    """
    if exit_bar is None:
        return "VOID", 0.0, ot.stake

    exit_price = exit_bar["close"]
    if ot.direction == TradeDirection.CALL:
        if exit_price > ot.entry_price:
            result = "WIN"
        elif exit_price < ot.entry_price:
            result = "LOSS"
        else:
            result = "TIE"
    else:  # PUT
        if exit_price < ot.entry_price:
            result = "WIN"
        elif exit_price > ot.entry_price:
            result = "LOSS"
        else:
            result = "TIE"

    if result == "TIE":
        if tie_policy == "refund":
            pnl = 0.0
            cash_refund = ot.stake
        else:  # loss
            result = "LOSS"
            pnl = -ot.stake
            cash_refund = 0.0
    elif result == "WIN":
        pnl = ot.stake * ot.payout
        cash_refund = ot.stake + pnl
    else:  # LOSS
        pnl = -ot.stake
        cash_refund = 0.0

    return result, pnl, cash_refund


@dataclass
class BacktestResult:
    """Summary and trade log resulting from a completed backtest run."""

    trades: list[TradeRecord]
    equity_curve: list[tuple[datetime, float]]
    starting_balance: float
    final_equity: float
    total_pnl: float
    max_drawdown: float
    skip_counts: dict[str, int]
    total_steps: int
    variant_name: str = ""


class BacktestEngine:
    """Simulates trading execution over historical or synthetic market data."""

    def __init__(
        self,
        config: AppConfig,
        data_by_asset: Mapping[str, Any],  # PrecomputedAssetData
        pattern_library: Optional[PatternLibrary] = None,
        store: Optional[DataStore] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.config = config
        self.data_by_asset = data_by_asset
        templates_dir = getattr(config.patterns, "templates_dir", "patterns/templates")
        self.pattern_library = pattern_library or get_pattern_library(templates_dir)

        # In-memory trade repository for backtest trades
        self.store = store or DataStore(db_path=":memory:")
        self.repo = TradeRepository(self.store)

        # Optional LLM client and decision service for backtest.decision_policy == 'llm'
        self.llm_client = llm_client
        self.decision_service = DecisionService()

    def run(
        self,
        start_t: datetime,
        end_t: datetime,
        expiry_minutes: int = 30,
        reversal_mode: Optional[str] = None,
        variant_name: str = "default",
    ) -> BacktestResult:
        """Run the simulation from start_t to end_t."""
        cfg = self.config.model_copy(deep=True)
        if reversal_mode is not None:
            cfg.gate.reversal_mode = reversal_mode  # type: ignore

        check_interval = timedelta(minutes=cfg.trading.check_interval_minutes)
        cash = cfg.backtest.starting_balance
        open_trades: list[OpenTrade] = []
        settled_trades: list[TradeRecord] = []
        equity_curve: list[tuple[datetime, float]] = []

        skip_counts: dict[str, int] = {
            "gate": 0,
            "no_pattern": 0,
            "insufficient_data": 0,
            "stake_below_min": 0,
            "exposure_cap": 0,
            "opposite_direction": 0,
            "insufficient_cash": 0,
            "data_gap_entry": 0,
            "data_gap_exit": 0,
        }
        total_steps = 0
        peak_equity = cash
        max_drawdown = 0.0

        trade_counter = 0

        if cfg.backtest.decision_policy == "llm":
            print(
                f"WARNING: backtest.decision_policy='llm' is active! Invoking LLM decision service "
                f"(capped at {cfg.backtest.max_llm_steps} steps)."
            )
        llm_steps_called = 0

        # Fast O(log N) lookup of 1m bars using numpy arrays
        fast_lookup_by_asset = {
            asset: Fast1mLookup(p_data.df_1m)
            for asset, p_data in self.data_by_asset.items()
        }

        cur_t = start_t
        while cur_t <= end_t:
            total_steps += 1

            # 1. Settle any open trades whose exit_time <= cur_t
            remaining_open: list[OpenTrade] = []
            for ot in open_trades:
                if ot.exit_time <= cur_t:
                    lookup = fast_lookup_by_asset.get(ot.asset)
                    exit_bar = lookup.get_bar(ot.exit_time) if lookup else None
                    result, pnl, cash_refund = settle_position(ot, exit_bar, cfg.backtest.tie_policy)
                    if result == "VOID":
                        skip_counts["data_gap_exit"] += 1
                    cash += cash_refund

                    # Update database record
                    self.repo.update_result(
                        trade_id=ot.trade_id,
                        closed_at=ot.exit_time.isoformat(),
                        result=result,
                        pnl=pnl,
                    )
                    updated_record = self.repo.get_trade_by_id(ot.trade_id)
                    if updated_record:
                        settled_trades.append(updated_record)
                else:
                    remaining_open.append(ot)

            open_trades = remaining_open

            # 2. Compute current equity
            current_open_stakes = sum(ot.stake for ot in open_trades)
            equity = compute_equity(cash, current_open_stakes)
            equity_curve.append((cur_t, equity))

            if equity > peak_equity:
                peak_equity = equity
            if peak_equity > 0:
                dd = (peak_equity - equity) / peak_equity
                if dd > max_drawdown:
                    max_drawdown = dd

            # 3. Query closed backtest trades as of cur_t for statistical inputs
            as_of_closed = [t for t in settled_trades if t.closed_at and datetime.fromisoformat(t.closed_at) <= cur_t]
            stats_as_of = {}
            for asset in self.data_by_asset.keys():
                stats_as_of[asset] = compute_asset_stats(
                    as_of_closed,
                    asset=asset,
                    mode="backtest",
                    as_of=cur_t,
                )

            current_hour_risk = hour_risk(
                as_of_closed,
                hour=cur_t.hour,
                session_min_samples=cfg.risk.session_min_samples,
                mode="backtest",
                as_of=cur_t,
            )

            # 4. Evaluate candidates for each asset
            step_candidates: list[CandidateTrade] = []
            candidates_by_asset: dict[str, list[CandidateTrade]] = {}

            for asset, p_data in self.data_by_asset.items():
                sliced_candles = p_data.slice_as_of(cur_t)
                payout = getattr(cfg.backtest, "default_payout", 0.85)

                eval_result = build_candidates(
                    asset=asset,
                    t=cur_t,
                    candles_by_tf=sliced_candles,
                    stats_as_of=stats_as_of,
                    config=cfg,
                    hour_risk_val=current_hour_risk,
                    payout=payout,
                    pattern_library=self.pattern_library,
                    calendar=p_data.calendar,
                )

                # Record skip reasons
                for r in eval_result.skip_reasons:
                    if "insufficient data" in r:
                        skip_counts["insufficient_data"] += 1
                    elif "no pattern" in r:
                        skip_counts["no_pattern"] += 1
                    elif "gate" in r:
                        skip_counts["gate"] += 1

                if eval_result.candidates:
                    step_candidates.extend(eval_result.candidates)
                    candidates_by_asset[asset] = eval_result.candidates

            # Select and commit trades according to decision_policy
            if cfg.backtest.decision_policy == "llm":
                if step_candidates and llm_steps_called < cfg.backtest.max_llm_steps and self.llm_client is not None:
                    llm_steps_called += 1
                    curr_prices = {}
                    for a, p_d in self.data_by_asset.items():
                        m1 = p_d.slice_as_of(cur_t).get("1m")
                        if m1 is not None and not m1.empty:
                            curr_prices[a] = float(m1["close"].iloc[-1])

                    state = SystemState(
                        kill_switch_active=False,
                        breaker_tripped=False,
                        reconcile_done=True,
                        current_prices=curr_prices,
                        open_trades=open_trades,
                        asset_modes={a: "live_eligible" for a in self.data_by_asset},
                    )
                    decision = self.decision_service.decide(
                        candidates=step_candidates,
                        state=state,
                        config=cfg,
                        client=self.llm_client,
                        now=cur_t,
                    )
                    if decision.is_valid and decision.decision in ("CALL", "PUT"):
                        cand = next(
                            (
                                c for c in step_candidates
                                if c.asset == decision.asset and c.direction.value == decision.direction
                            ),
                            None,
                        )
                        if cand and decision.stake_amount:
                            stake = decision.stake_amount
                            entry_delay = timedelta(minutes=cfg.backtest.entry_delay_bars)
                            entry_time = cur_t + entry_delay
                            lookup = fast_lookup_by_asset.get(cand.asset)
                            entry_bar = lookup.get_bar(entry_time) if lookup else None
                            if entry_bar is not None and stake <= cash:
                                open_px = entry_bar["open"]
                                slippage = cfg.backtest.slippage_atr * cand.atr
                                entry_price = open_px + slippage if cand.direction == TradeDirection.CALL else open_px - slippage
                                exit_time = entry_time + timedelta(minutes=expiry_minutes)

                                cash -= stake
                                trade_counter += 1
                                trade_id = f"bt_{variant_name}_{trade_counter}"
                                payout = getattr(cfg.backtest, "default_payout", 0.85)

                                record = TradeRecord(
                                    id=trade_id,
                                    opened_at=entry_time.isoformat(),
                                    asset=cand.asset,
                                    is_otc=False,
                                    direction=cand.direction.value,
                                    expiry_minutes=expiry_minutes,
                                    stake=stake,
                                    payout_at_entry=payout,
                                    result="PENDING",
                                    pnl=0.0,
                                    mode="backtest",
                                    pattern_name=cand.pattern_match.pattern_name,
                                    pattern_type=cand.pattern_match.pattern_type,
                                    risk_score=cand.risk_report.score,
                                    hour_utc=entry_time.hour,
                                )
                                self.repo.insert(record)
                                open_trades.append(
                                    OpenTrade(
                                        trade_id=trade_id,
                                        asset=cand.asset,
                                        direction=cand.direction,
                                        stake=stake,
                                        entry_time=entry_time,
                                        entry_price=entry_price,
                                        expiry_minutes=expiry_minutes,
                                        exit_time=exit_time,
                                        payout=payout,
                                        pattern_name=cand.pattern_match.pattern_name,
                                        pattern_type=cand.pattern_match.pattern_type,
                                        risk_score=cand.risk_report.score,
                                        evidence=cand.evidence,
                                    )
                                )
                    else:
                        skip_counts["llm_skip"] = skip_counts.get("llm_skip", 0) + 1
            else:
                for asset, cands in candidates_by_asset.items():
                    # Decision policy: choose 1 candidate for this asset
                    if cfg.backtest.decision_policy == "highest_similarity":
                        cands.sort(
                            key=lambda c: (
                                -float(c.pattern_match.similarity),
                                float(c.risk_report.score),
                            )
                        )
                    else:  # "lowest_risk" (default)
                        cands.sort(
                            key=lambda c: (
                                float(c.risk_report.score),
                                -float(c.pattern_match.similarity),
                            )
                        )

                    cand = cands[0]

                    # Check portfolio constraints
                    # a) Opposite-direction block
                    has_opposite = any(
                        ot.asset == asset and ot.direction != cand.direction
                        for ot in open_trades
                    )
                    if has_opposite:
                        skip_counts["opposite_direction"] += 1
                        continue

                    # b) Stake sizing (recompute open stakes and equity per committed trade)
                    current_open_stakes = sum(ot.stake for ot in open_trades)
                    equity = compute_equity(cash, current_open_stakes)

                    target_pct = cand.max_stake_pct
                    if cfg.backtest.stake_policy == "mid":
                        target_pct = (cfg.stake.min_pct + target_pct) / 2.0
                    elif cfg.backtest.stake_policy == "min":
                        target_pct = cfg.stake.min_pct

                    stake = stake_amount(
                        equity=equity,
                        pct=target_pct,
                        min_amount=cfg.stake.min_amount,
                        round_to=cfg.stake.round_to,
                    )

                    if stake is None:
                        skip_counts["stake_below_min"] += 1
                        continue

                    # c) Exposure cap (cfg.stake.total_cap_pct converted from percentage)
                    total_cap_fraction = cfg.stake.total_cap_pct / 100.0
                    if (current_open_stakes + stake) > (total_cap_fraction * equity):
                        skip_counts["exposure_cap"] += 1
                        continue

                    # d) Cash sufficiency
                    if stake > cash:
                        skip_counts["insufficient_cash"] += 1
                        continue

                    # Entry execution: price at OPEN of next 1m bar + slippage
                    entry_delay = timedelta(minutes=cfg.backtest.entry_delay_bars)
                    entry_time = cur_t + entry_delay
                    lookup = fast_lookup_by_asset.get(cand.asset)
                    entry_bar = lookup.get_bar(entry_time) if lookup else None
                    if entry_bar is None:
                        skip_counts["data_gap_entry"] += 1
                        continue

                    open_px = entry_bar["open"]
                    slippage = cfg.backtest.slippage_atr * cand.atr
                    if cand.direction == TradeDirection.CALL:
                        entry_price = open_px + slippage
                    else:
                        entry_price = open_px - slippage

                    exit_time = entry_time + timedelta(minutes=expiry_minutes)

                    # Commit trade immediately so open_trades and equity reflect it on next asset
                    cash -= stake
                    trade_counter += 1
                    trade_id = f"bt_{variant_name}_{trade_counter}"

                    payout = getattr(cfg.backtest, "default_payout", 0.85)
                    record = TradeRecord(
                        id=trade_id,
                        opened_at=entry_time.isoformat(),
                        asset=cand.asset,
                        is_otc=False,
                        direction=cand.direction.value,
                        expiry_minutes=expiry_minutes,
                        stake=stake,
                        payout_at_entry=payout,
                        result="PENDING",
                        pnl=0.0,
                        mode="backtest",
                        pattern_name=cand.pattern_match.pattern_name,
                        pattern_type=cand.pattern_match.pattern_type,
                        risk_score=cand.risk_report.score,
                        hour_utc=entry_time.hour,
                    )
                    self.repo.insert(record)

                    open_trades.append(
                        OpenTrade(
                            trade_id=trade_id,
                            asset=cand.asset,
                            direction=cand.direction,
                            stake=stake,
                            entry_time=entry_time,
                            entry_price=entry_price,
                            expiry_minutes=expiry_minutes,
                            exit_time=exit_time,
                            payout=payout,
                            pattern_name=cand.pattern_match.pattern_name,
                            pattern_type=cand.pattern_match.pattern_type,
                            risk_score=cand.risk_report.score,
                            evidence=cand.evidence,
                        )
                    )

            cur_t += check_interval

        # Settle any trades still open at simulation end using shared settle_position
        for ot in open_trades:
            lookup = fast_lookup_by_asset.get(ot.asset)
            exit_bar = lookup.get_bar(ot.exit_time) if lookup else None
            res, pnl, cash_refund = settle_position(ot, exit_bar, cfg.backtest.tie_policy)
            if res == "VOID":
                skip_counts["data_gap_exit"] += 1
            cash += cash_refund

            self.repo.update_result(
                trade_id=ot.trade_id,
                closed_at=ot.exit_time.isoformat(),
                result=res,
                pnl=pnl,
            )
            upd = self.repo.get_trade_by_id(ot.trade_id)
            if upd:
                settled_trades.append(upd)

        final_equity = cash
        total_pnl = sum(t.pnl for t in settled_trades)

        return BacktestResult(
            trades=settled_trades,
            equity_curve=equity_curve,
            starting_balance=cfg.backtest.starting_balance,
            final_equity=final_equity,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            skip_counts=skip_counts,
            total_steps=total_steps,
            variant_name=variant_name,
        )
