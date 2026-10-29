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
    ):
        self.config = config
        self.data_by_asset = data_by_asset
        templates_dir = getattr(config.patterns, "templates_dir", "patterns/templates")
        self.pattern_library = pattern_library or get_pattern_library(templates_dir)

        # In-memory trade repository for backtest trades
        self.store = store or DataStore(db_path=":memory:")
        self.repo = TradeRepository(self.store)

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
        }
        total_steps = 0
        peak_equity = cash
        max_drawdown = 0.0

        trade_counter = 0

        # Pre-build lookup map of 1m bars for fast fill simulation
        bars_1m_by_asset: dict[str, dict[datetime, dict[str, float]]] = {}
        for asset, p_data in self.data_by_asset.items():
            df_1m = p_data.df_1m
            # Map timestamp to open and close prices
            bars_1m_by_asset[asset] = {
                row["ts"]: {"open": float(row["open"]), "close": float(row["close"])}
                for _, row in df_1m.iterrows()
            }

        cur_t = start_t
        while cur_t <= end_t:
            total_steps += 1

            # 1. Settle any open trades whose exit_time <= cur_t
            remaining_open: list[OpenTrade] = []
            for ot in open_trades:
                if ot.exit_time <= cur_t:
                    # Settle trade
                    bars_map = bars_1m_by_asset.get(ot.asset, {})
                    exit_bar = bars_map.get(ot.exit_time)
                    if exit_bar is not None:
                        exit_price = exit_bar["close"]
                    else:
                        # Fallback to latest available bar up to exit_time
                        exit_price = ot.entry_price

                    # Determine outcome
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
                        if cfg.backtest.tie_policy == "refund":
                            pnl = 0.0
                            cash += ot.stake
                        else:  # "loss"
                            result = "LOSS"
                            pnl = -ot.stake
                    elif result == "WIN":
                        pnl = ot.stake * ot.payout
                        cash += ot.stake + pnl
                    else:  # LOSS
                        pnl = -ot.stake

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

                if not eval_result.candidates:
                    continue

                # Decision policy: choose 1 candidate for this asset
                if cfg.backtest.decision_policy == "highest_similarity":
                    eval_result.candidates.sort(
                        key=lambda c: (
                            -float(c.pattern_match.similarity),
                            float(c.risk_report.score),
                        )
                    )
                else:  # "lowest_risk" (default)
                    eval_result.candidates.sort(
                        key=lambda c: (
                            float(c.risk_report.score),
                            -float(c.pattern_match.similarity),
                        )
                    )

                cand = eval_result.candidates[0]

                # Check portfolio constraints
                # a) Opposite-direction block
                has_opposite = any(
                    ot.asset == asset and ot.direction != cand.direction
                    for ot in open_trades
                )
                if has_opposite:
                    skip_counts["opposite_direction"] += 1
                    continue

                # b) Stake sizing
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

                # c) 35% exposure cap
                if current_open_stakes + stake > 0.35 * equity:
                    skip_counts["exposure_cap"] += 1
                    continue

                # d) Cash sufficiency
                if stake > cash:
                    skip_counts["insufficient_cash"] += 1
                    continue

                # Entry execution: price at OPEN of next 1m bar + slippage
                entry_delay = timedelta(minutes=cfg.backtest.entry_delay_bars)
                entry_time = cur_t + entry_delay
                bars_map = bars_1m_by_asset.get(asset, {})
                entry_bar = bars_map.get(entry_time)
                if entry_bar is None:
                    continue

                open_px = entry_bar["open"]
                slippage = cfg.backtest.slippage_atr * cand.atr
                if cand.direction == TradeDirection.CALL:
                    entry_price = open_px + slippage
                else:
                    entry_price = open_px - slippage

                exit_time = entry_time + timedelta(minutes=expiry_minutes)

                # Commit trade
                cash -= stake
                trade_counter += 1
                trade_id = f"bt_{variant_name}_{trade_counter}"

                # Insert into TradeRepository
                record = TradeRecord(
                    id=trade_id,
                    opened_at=entry_time.isoformat(),
                    asset=asset,
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
                        asset=asset,
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

        # Settle any trades still open at simulation end
        for ot in open_trades:
            bars_map = bars_1m_by_asset.get(ot.asset, {})
            exit_bar = bars_map.get(ot.exit_time)
            exit_price = exit_bar["close"] if exit_bar else ot.entry_price

            if ot.direction == TradeDirection.CALL:
                res = "WIN" if exit_price > ot.entry_price else "LOSS"
            else:
                res = "WIN" if exit_price < ot.entry_price else "LOSS"

            pnl = ot.stake * ot.payout if res == "WIN" else -ot.stake
            cash += ot.stake + pnl
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
