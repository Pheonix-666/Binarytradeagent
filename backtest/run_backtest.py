"""Command-line runner executing multi-variant train/test backtests.

PROJECT_SPEC.md Section 11 & Phase 6:
  - Runs 6 variants: reversal_mode off vs sr_bypass x expiry 15/30/60.
  - Chronological train/test split (default 70% train, 30% test).
  - Variants ranked on train set only; test set results shown side-by-side.
  - Baselines: Monte Carlo random direction (>= 200 seeds), always CALL, always PUT.
  - Block statistics for 10, 20, 50-trade blocks vs reporting target 0.80.
  - Selectivity curve exploring similarity and risk thresholds.
  - Sanity alarm on test set if win rate >= 70% with >= 100 trades.
  - Saves markdown report to docs/backtests/<timestamp>/report.md and trades.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Optional

import pandas as pd

from backtest.data_loader import (
    generate_synthetic_candles,
    load_history_data,
    precompute_asset_data,
)
from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import (
    compute_baseline_comparison,
    compute_variant_metrics,
)
from core.config import AppConfig, CalendarConfig, load_config
from core.logging_setup import get_logger

logger = get_logger()


def run_full_backtest(
    config_path: str = "config.yaml",
    synthetic_if_missing: bool = True,
    synthetic_days: int = 50,
) -> Path:
    """Execute the full 6-variant train/test backtesting suite."""
    cfg = load_config(config_path)

    # Resolve market calendars and assets
    calendars = cfg.data.calendars
    configured_assets = cfg.backtest.assets

    data_by_asset = {}
    history_dir = Path("data_history")

    for asset in configured_assets:
        cal_name = cfg.data.assets.get(asset, "forex")
        calendar = calendars.get(cal_name, CalendarConfig(type="always_open"))
        csv_file = history_dir / f"{asset}.csv"

        if csv_file.exists():
            logger.info(f"Loading real history CSV for {asset} from {csv_file}")
            df_1m = load_history_data(csv_file, calendar=calendar)
        elif synthetic_if_missing:
            logger.info(
                f"History CSV {csv_file} not found. Generating synthetic {asset} data ({synthetic_days} days)"
            )
            df_1m = generate_synthetic_candles(
                asset_id=asset,
                days=synthetic_days,
                calendar=calendar,
                seed=42,
            )
        else:
            logger.warning(f"Skipping {asset}: file {csv_file} not found.")
            continue

        p_data = precompute_asset_data(asset, df_1m, calendar=calendar, config=cfg)
        data_by_asset[asset] = p_data

    if not data_by_asset:
        raise RuntimeError("No asset data available to run backtest.")

    # Determine timeline and train/test split
    first_asset = list(data_by_asset.keys())[0]
    df_ref = data_by_asset[first_asset].df_1m
    min_t = df_ref["ts"].min().to_pydatetime()
    max_t = df_ref["ts"].max().to_pydatetime()

    # Skip first 10 days for indicator warm-up
    warmup_days = 10
    sim_start = min_t + pd.Timedelta(days=warmup_days)
    if sim_start >= max_t:
        sim_start = min_t + (max_t - min_t) * 0.2

    total_duration = max_t - sim_start
    split_t = sim_start + total_duration * cfg.backtest.train_fraction

    logger.info(
        f"Timeline: start={sim_start.isoformat()}, split={split_t.isoformat()}, end={max_t.isoformat()} "
        f"(Train: {cfg.backtest.train_fraction:.0%}, Test: {1 - cfg.backtest.train_fraction:.0%})"
    )

    # 6 Backtest Variants
    variants = [
        {"name": "reversal_off_15m", "reversal_mode": "off", "expiry": 15},
        {"name": "reversal_off_30m", "reversal_mode": "off", "expiry": 30},
        {"name": "reversal_off_60m", "reversal_mode": "off", "expiry": 60},
        {"name": "sr_bypass_15m", "reversal_mode": "sr_bypass", "expiry": 15},
        {"name": "sr_bypass_30m", "reversal_mode": "sr_bypass", "expiry": 30},
        {"name": "sr_bypass_60m", "reversal_mode": "sr_bypass", "expiry": 60},
    ]

    variant_results = []
    all_trades = []

    for v in variants:
        v_name = v["name"]
        rev_mode = v["reversal_mode"]
        expiry = v["expiry"]

        logger.info(f"Running variant: {v_name} (reversal_mode={rev_mode}, expiry={expiry}m)")

        # 1. Train set run
        engine_train = BacktestEngine(config=cfg, data_by_asset=data_by_asset)
        res_train = engine_train.run(
            start_t=sim_start,
            end_t=split_t,
            expiry_minutes=expiry,
            reversal_mode=rev_mode,
            variant_name=f"{v_name}_train",
        )
        metrics_train = compute_variant_metrics(
            res_train.trades,
            starting_balance=res_train.starting_balance,
            final_equity=res_train.final_equity,
            max_drawdown=res_train.max_drawdown,
            target_win_rate=cfg.backtest.target_win_rate,
            is_test_set=False,
        )

        # 2. Test set run
        engine_test = BacktestEngine(config=cfg, data_by_asset=data_by_asset)
        res_test = engine_test.run(
            start_t=split_t,
            end_t=max_t,
            expiry_minutes=expiry,
            reversal_mode=rev_mode,
            variant_name=f"{v_name}_test",
        )
        metrics_test = compute_variant_metrics(
            res_test.trades,
            starting_balance=res_test.starting_balance,
            final_equity=res_test.final_equity,
            max_drawdown=res_test.max_drawdown,
            target_win_rate=cfg.backtest.target_win_rate,
            is_test_set=True,
        )

        baseline_test = compute_baseline_comparison(res_test.trades, num_seeds=200)

        variant_results.append({
            "variant": v,
            "train_result": res_train,
            "train_metrics": metrics_train,
            "test_result": res_test,
            "test_metrics": metrics_test,
            "baseline_test": baseline_test,
        })
        all_trades.extend(res_train.trades)
        all_trades.extend(res_test.trades)

    # Rank variants on TRAIN set only (by EV per trade)
    variant_results.sort(key=lambda r: -r["train_metrics"].ev_per_trade)

    best_variant = variant_results[0]
    best_v_name = best_variant["variant"]["name"]

    # Selectivity curve: re-run TEST period across similarity & risk thresholds
    logger.info("Computing selectivity curve on test set...")
    selectivity_rows = []
    test_days = max(1.0, (max_t - split_t).total_seconds() / 86400.0)

    for sim_thresh in [60.0, 70.0, 75.0, 80.0, 85.0]:
        for max_risk in [60.0, 50.0, 40.0, 30.0]:
            sel_cfg = cfg.model_copy(deep=True)
            sel_cfg.patterns.min_similarity = sim_thresh
            sel_cfg.gate.reversal_bypass.max_risk_score = max_risk

            eng = BacktestEngine(config=sel_cfg, data_by_asset=data_by_asset)
            r_sel = eng.run(
                start_t=split_t,
                end_t=max_t,
                expiry_minutes=best_variant["variant"]["expiry"],
                reversal_mode=best_variant["variant"]["reversal_mode"],
                variant_name=f"sel_{sim_thresh}_{max_risk}",
            )
            m_sel = compute_variant_metrics(
                r_sel.trades,
                starting_balance=r_sel.starting_balance,
                final_equity=r_sel.final_equity,
                max_drawdown=r_sel.max_drawdown,
                target_win_rate=cfg.backtest.target_win_rate,
                is_test_set=True,
            )
            trades_per_day = round(m_sel.total_trades / test_days, 1)
            selectivity_rows.append({
                "min_sim": sim_thresh,
                "max_risk": max_risk,
                "trades": m_sel.total_trades,
                "trades_per_day": trades_per_day,
                "win_rate": m_sel.win_rate,
                "wilson_ci": m_sel.wilson_ci,
                "ev_per_trade": m_sel.ev_per_trade,
                "realized_return": m_sel.realized_return,
            })

    # Generate timestamped report directory
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_dir = Path("docs") / "backtests" / timestamp_str
    report_dir.mkdir(parents=True, exist_ok=True)

    report_md_path = report_dir / "report.md"
    trades_csv_path = report_dir / "trades.csv"

    # Export trades.csv
    if all_trades:
        df_trades = pd.DataFrame([t.to_dict() for t in all_trades])
        df_trades.to_csv(trades_csv_path, index=False)
    else:
        pd.DataFrame().to_csv(trades_csv_path, index=False)

    # Render Markdown Report
    report_md = build_markdown_report(
        timestamp=timestamp_str,
        config=cfg,
        timeline={"start": sim_start, "split": split_t, "end": max_t},
        variant_results=variant_results,
        selectivity_rows=selectivity_rows,
    )

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info(f"Backtest report written to {report_md_path}")
    logger.info(f"Trades log exported to {trades_csv_path}")

    return report_md_path


def build_markdown_report(
    timestamp: str,
    config: AppConfig,
    timeline: dict[str, datetime],
    variant_results: list[dict[str, Any]],
    selectivity_rows: list[dict[str, Any]],
) -> str:
    """Build formatted GitHub Markdown report."""
    lines: list[str] = []

    lines.append(f"# Systematic Fixed-Time Trading Backtest Report")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    lines.append(f"**Report Path:** `docs/backtests/{timestamp}/report.md`  ")
    lines.append(f"**Target Win Rate (Reporting Only):** `{config.backtest.target_win_rate:.0%}`  ")
    lines.append(f"**Timeline Split:** Train ({timeline['start'].strftime('%Y-%m-%d')} to {timeline['split'].strftime('%Y-%m-%d')}), Test ({timeline['split'].strftime('%Y-%m-%d')} to {timeline['end'].strftime('%Y-%m-%d')})  ")
    lines.append("")

    # Check for Sanity Alarms across any test variant
    alarms = [r["test_metrics"].sanity_alarm_message for r in variant_results if r["test_metrics"].sanity_alarm]
    if alarms:
        lines.append("> [!WARNING]")
        for a in alarms:
            lines.append(f"> **{a}**  ")
        lines.append("")

    # Section 1: Ranked Variants Overview
    lines.append("## 1. Variant Performance Summary")
    lines.append("Variants are ranked by **Train EV per trade**. Test results are shown side-by-side.")
    lines.append("")
    lines.append("| Rank | Variant | Reversal Mode | Expiry | Train Trades | Train WinRate (95% CI) | Train EV | Test Trades | Test WinRate (95% CI) | Test EV | Test Return | Edge Label |")
    lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for i, r in enumerate(variant_results):
        v = r["variant"]
        tm = r["train_metrics"]
        sm = r["test_metrics"]
        train_ci_str = f"[{tm.wilson_ci[0]:.1%}, {tm.wilson_ci[1]:.1%}]"
        test_ci_str = f"[{sm.wilson_ci[0]:.1%}, {sm.wilson_ci[1]:.1%}]"
        lines.append(
            f"| {i + 1} | `{v['name']}` | `{v['reversal_mode']}` | {v['expiry']}m | "
            f"{tm.total_trades} | {tm.win_rate:.1%} {train_ci_str} | {tm.ev_per_trade:+.4f} | "
            f"{sm.total_trades} | {sm.win_rate:.1%} {test_ci_str} | {sm.ev_per_trade:+.4f} | {sm.realized_return:+.1%} | **{sm.edge_label}** |"
        )
    lines.append("")

    # Section 2: Best Variant In-Depth Analysis
    top = variant_results[0]
    best_name = top["variant"]["name"]
    best_test_m = top["test_metrics"]
    best_baseline = top["baseline_test"]

    lines.append(f"## 2. In-Depth Test Analysis: `{best_name}`")
    lines.append(f"- **Final Equity:** `${best_test_m.final_equity:,.2f}` (Initial: `${config.backtest.starting_balance:,.2f}`)")
    lines.append(f"- **Max Drawdown:** `{best_test_m.max_drawdown:.1%}`")
    lines.append(f"- **Longest Losing Streak:** `{best_test_m.streaks.longest_losing_streak}` trades (avg: `{best_test_m.streaks.average_losing_streak}`)")
    lines.append(f"- **Longest Winning Streak:** `{best_test_m.streaks.longest_winning_streak}` trades (avg: `{best_test_m.streaks.average_winning_streak}`)")
    lines.append(f"- **Break-Even Win Rate:** `{best_test_m.break_even:.1%}`")
    lines.append("")

    # Baseline comparison box
    lines.append("### Baseline Controls (Monte Carlo & Fixed Direction)")
    lines.append(f"**Control Verdict:** {best_baseline.summary}")
    lines.append("")
    lines.append("| Baseline | Win Rate | Comparison |")
    lines.append("|:---|:---:|:---|")
    lines.append(f"| **Strategy (`{best_name}`)** | **{best_test_m.win_rate:.1%}** | Actual Strategy |")
    lines.append(f"| Random Direction (200 seeds, 5th-95th percentile) | [{best_baseline.random_band_p5:.1%}, {best_baseline.random_band_p95:.1%}] (mean: {best_baseline.random_mean:.1%}) | Pure chance control |")
    lines.append(f"| Always CALL | {best_baseline.always_call_win_rate:.1%} | Directional bias control |")
    lines.append(f"| Always PUT | {best_baseline.always_put_win_rate:.1%} | Directional bias control |")
    lines.append("")

    # Block Statistics vs 80% Target
    lines.append(f"### Block Statistics vs Reporting Target (`{config.backtest.target_win_rate:.0%}`)")
    lines.append("Non-overlapping chronological blocks. Compares actual hit rate against expected binomial probability:")
    lines.append("")
    lines.append("| Block Size | Total Blocks | Hit Count (>= 80%) | Actual Hit Rate | Expected Binomial Hit Rate | Interpretation |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---|")
    for b_size in [10, 20, 50]:
        b = best_test_m.blocks.get(b_size)
        if b:
            diff = b.actual_hit_rate - b.expected_binomial_hit_rate
            interp = "Consistent with binomial chance" if abs(diff) <= 0.10 else ("Higher block clustering than chance" if diff > 0 else "Lower block hit rate than chance")
            lines.append(
                f"| {b.block_size} trades | {b.num_blocks} | {b.hit_count} | {b.actual_hit_rate:.1%} | {b.expected_binomial_hit_rate:.1%} | {interp} |"
            )
    lines.append("")

    # Skip reasons breakdown
    test_res = top["test_result"]
    total_steps = test_res.total_steps
    lines.append("### Pipeline Check Step Skip Fractions")
    lines.append(f"Total check intervals evaluated: `{total_steps}`")
    lines.append("")
    lines.append("| Reason | Step Count | Fraction of Total Steps |")
    lines.append("|:---|:---:|:---:|")
    for reason, count in test_res.skip_counts.items():
        frac = count / total_steps if total_steps > 0 else 0.0
        lines.append(f"| `{reason}` | {count} | {frac:.1%} |")
    lines.append("")

    # Risk Bucket Breakdown
    if best_test_m.breakdown_by_risk_bucket:
        lines.append("### Performance by Risk Bucket")
        lines.append("| Risk Bucket | Trades | Wins | Win Rate | Realized PnL |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|")
        for bucket, b_data in best_test_m.breakdown_by_risk_bucket.items():
            lines.append(f"| {bucket} | {b_data['n']} | {b_data['wins']} | {b_data['win_rate']:.1%} | ${b_data['pnl']:+,.2f} |")
        lines.append("")

    # Pattern Breakdown
    if best_test_m.breakdown_by_pattern:
        lines.append("### Performance by Pattern")
        lines.append("| Pattern | Trades | Wins | Win Rate | Realized PnL |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")
        for pat, p_data in best_test_m.breakdown_by_pattern.items():
            lines.append(f"| `{pat}` | {p_data['n']} | {p_data['wins']} | {p_data['win_rate']:.1%} | ${p_data['pnl']:+,.2f} |")
        lines.append("")

    # Hour of Day Breakdown
    if best_test_m.breakdown_by_hour:
        lines.append("### Performance by Hour of Day (UTC)")
        lines.append("| Hour (UTC) | Trades | Wins | Win Rate | Realized PnL |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|")
        for h, h_data in sorted(best_test_m.breakdown_by_hour.items()):
            lines.append(f"| {h:02d}:00 | {h_data['n']} | {h_data['wins']} | {h_data['win_rate']:.1%} | ${h_data['pnl']:+,.2f} |")
        lines.append("")

    # Section 3: Selectivity Curve
    lines.append("## 3. Selectivity Curve (Trade-off: Selectivity vs Sample Size)")
    lines.append("Evaluated on the TEST dataset under different similarity and risk caps:")
    lines.append("")
    lines.append("| Min Similarity | Max Risk Score | Trades | Trades/Day | Win Rate (95% CI) | EV Per Trade | Realized Return |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for s in selectivity_rows:
        ci_str = f"[{s['wilson_ci'][0]:.1%}, {s['wilson_ci'][1]:.1%}]"
        lines.append(
            f"| {s['min_sim']:.0f}% | {s['max_risk']:.0f} | {s['trades']} | {s['trades_per_day']:.1f} | "
            f"{s['win_rate']:.1%} {ci_str} | {s['ev_per_trade']:+.4f} | {s['realized_return']:+.1%} |"
        )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    report_file = run_full_backtest()
    print(f"\nBacktest run complete! Report saved at: {report_file}")
