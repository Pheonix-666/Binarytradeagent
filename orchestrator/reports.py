"""Daily performance report generator.

PROJECT_SPEC.md Phase 9:
  - Generates markdown summaries saved to reports/daily/YYYY-MM-DD.md.
  - Summarizes trades, PnL, win rates, Wilson 95% CI, break-even, EV, realized return.
  - Reports paper gate progress (n of min_paper_trades, CI lower > break-even).
  - Reports adaptive tier and time per tier, edge-decay status.
  - Reports skip counts by reason from cycle execution logs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from core.config import AppConfig
from data.store import TradeRecord, TradeRepository
from selection.stats import asset_mode, break_even as calc_break_even, compute_asset_stats, ev as calc_ev, wilson_interval


def _format_time_per_tier(
    adaptive_controller: Optional[Any],
    state_manager: Any,
    day_start: datetime,
    day_end: datetime,
) -> dict[str, float]:
    """Calculate hours spent in each adaptive tier during the specified day."""
    tier_hours: dict[str, float] = {
        "NORMAL": 0.0,
        "CAUTION": 0.0,
        "DEFENSIVE": 0.0,
        "PAUSED": 0.0,
    }

    state_obj = None
    if adaptive_controller and hasattr(adaptive_controller, "state"):
        state_obj = adaptive_controller.state
    elif hasattr(state_manager, "data"):
        state_obj = state_manager.data.get("adaptive_state")

    history = getattr(state_obj, "history", None) if state_obj else None
    if not history and isinstance(state_obj, dict):
        history = state_obj.get("history", [])

    current_tier = "NORMAL"
    if state_obj:
        current_tier = getattr(state_obj, "tier", None) or (
            state_obj.get("tier", "NORMAL") if isinstance(state_obj, dict) else "NORMAL"
        )

    if not history:
        # Default all day hours (24.0) to current tier
        total_day_hours = min(24.0, max(0.0, (day_end - day_start).total_seconds() / 3600.0))
        tier_hours[current_tier] = round(total_day_hours, 1)
        return tier_hours

    # Parse transitions
    sorted_hist = sorted(
        history,
        key=lambda h: datetime.fromisoformat(h["timestamp"]).replace(tzinfo=timezone.utc)
        if isinstance(h.get("timestamp"), str)
        else day_start,
    )

    prev_time = day_start
    active_tier = "NORMAL"

    for entry in sorted_hist:
        ts_raw = entry.get("timestamp")
        if not ts_raw:
            continue
        try:
            entry_time = datetime.fromisoformat(str(ts_raw))
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
        except Exception:
            continue

        if entry_time < day_start:
            active_tier = entry.get("to_tier", active_tier)
            continue
        if entry_time > day_end:
            break

        duration_hours = (entry_time - prev_time).total_seconds() / 3600.0
        tier_hours[active_tier] = tier_hours.get(active_tier, 0.0) + duration_hours
        prev_time = entry_time
        active_tier = entry.get("to_tier", active_tier)

    # Tail duration to end of day
    if prev_time < day_end:
        duration_hours = (day_end - prev_time).total_seconds() / 3600.0
        tier_hours[active_tier] = tier_hours.get(active_tier, 0.0) + duration_hours

    return {k: round(v, 1) for k, v in tier_hours.items()}


def _extract_skip_counts(
    cycle_log_path: Optional[Path | str],
    date_str: str,
    explicit_skip_counts: Optional[dict[str, int]] = None,
) -> dict[str, int]:
    """Tally skip reasons from cycles.jsonl for the given date."""
    if explicit_skip_counts is not None:
        return dict(explicit_skip_counts)

    counts: dict[str, int] = {}
    if not cycle_log_path:
        return counts

    log_path = Path(cycle_log_path)
    if not log_path.exists():
        return counts

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                ts = rec.get("timestamp", "")
                if not ts.startswith(date_str):
                    continue

                if rec.get("action") == "SKIP":
                    reason = rec.get("reason")
                    if reason:
                        counts[reason] = counts.get(reason, 0) + 1
                    reasons = rec.get("skip_reasons")
                    if isinstance(reasons, list):
                        for r in reasons:
                            counts[str(r)] = counts.get(str(r), 0) + 1
    except Exception:
        pass

    return counts


def generate_daily_summary_markdown(
    date_str: str,
    repository: TradeRepository,
    config: AppConfig,
    state_manager: Any,
    breaker: Any,
    adaptive_controller: Optional[Any] = None,
    assets: Optional[Sequence[str]] = None,
    cycle_log_path: Optional[Path | str] = None,
    skip_counts: Optional[dict[str, int]] = None,
) -> tuple[str, str]:
    """Generate daily summary markdown content and brief Telegram message.

    Returns: (markdown_content, telegram_message)
    """
    target_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    day_start = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = target_dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    day_trades = repository.get_trades(since=day_start, until=day_end)
    settled = [t for t in day_trades if t.result in {"WIN", "LOSS", "TIE"}]
    wins = sum(1 for t in settled if t.result == "WIN")
    losses = sum(1 for t in settled if t.result == "LOSS")
    ties = sum(1 for t in settled if t.result == "TIE")
    total_settled = len(settled)

    win_rate = (wins / total_settled) if total_settled > 0 else 0.0
    ci_low, ci_high = wilson_interval(wins, total_settled, z=1.96) if total_settled > 0 else (0.0, 1.0)

    # Average payout across settled trades
    payouts = [float(t.payout_at_entry) for t in settled if t.payout_at_entry is not None]
    avg_payout = (sum(payouts) / len(payouts)) if payouts else float(getattr(config.trading, "base_stake", 0.85) and 0.85)
    overall_be = calc_break_even(avg_payout)
    overall_ev = calc_ev(win_rate, avg_payout) if total_settled > 0 else 0.0

    total_pnl = sum(t.pnl for t in settled)
    total_stake = sum(t.stake for t in settled)
    overall_return = (total_pnl / total_stake) if total_stake > 0 else 0.0

    state_data = getattr(state_manager, "data", {}) if state_manager else {}
    balance = float(state_data.get("balance", 0.0))
    tier = (
        getattr(adaptive_controller.state, "tier", "NORMAL")
        if adaptive_controller and hasattr(adaptive_controller, "state")
        else (state_data.get("adaptive_state") or {}).get("tier", "NORMAL")
    )
    breaker_status = "TRIPPED" if breaker.is_tripped() else "OK"

    # Edge decay status
    edge_decay_detected = False
    if adaptive_controller and hasattr(adaptive_controller, "state"):
        edge_decay_detected = bool(getattr(adaptive_controller.state, "edge_decay_detected", False))
    elif isinstance(state_data.get("adaptive_state"), dict):
        edge_decay_detected = bool(state_data["adaptive_state"].get("edge_decay_detected", False))

    edge_decay_str = "⚠️ DECAY DETECTED" if edge_decay_detected else "✅ Intact (No decay)"

    # Time per tier
    tier_hours = _format_time_per_tier(
        adaptive_controller=adaptive_controller,
        state_manager=state_manager,
        day_start=day_start,
        day_end=day_end,
    )
    time_tier_lines = [f"`{t}`: {h:.1f}h" for t, h in tier_hours.items() if h > 0]
    time_tier_str = ", ".join(time_tier_lines) if time_tier_lines else f"`{tier}`: 24.0h"

    # Skip counts
    skips = _extract_skip_counts(cycle_log_path, date_str, explicit_skip_counts=skip_counts)

    # Sanity alarm check: when source is replay (never live) and daily or overall win rate >= 70% with >= 100 trades
    evaluated_assets = list(assets or config.source.assets or ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD"])
    all_trades = repository.get_trades(until=day_end)
    all_settled = [t for t in all_trades if t.result in {"WIN", "LOSS", "TIE"}]
    cum_wins = sum(1 for t in all_settled if t.result == "WIN")
    cum_n = len(all_settled)
    cum_wr = (cum_wins / cum_n) if cum_n > 0 else 0.0

    source_name = str(getattr(config.source, "name", "")).lower() if hasattr(config, "source") else ""
    is_replay = source_name == "replay"
    sanity_alarm = is_replay and (
        (total_settled >= 100 and win_rate >= 0.70) or (cum_n >= 100 and cum_wr >= 0.70)
    )
    sanity_alarm_message = (
        "SANITY ALARM: Observed test set win rate >= 70% with >= 100 trades! "
        "Rigorous verification required: check for look-ahead bias, causal index leakage, "
        "or optimistic fill assumptions. Checklist: (1) Look-ahead guard test passed; "
        "(2) Entry delay is >= 1 bar; (3) Slippage penalty applied; (4) Indicators purely causal."
    )

    md_lines = [
        f"# Daily Performance Report — {date_str}",
        "",
        f"- **Generated At**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        f"- **Mode**: `{config.mode}`",
        f"- **Current Balance**: `${balance:,.2f}`",
        f"- **Adaptive Tier**: `{tier}`",
        f"- **Time per Tier**: {time_tier_str}",
        f"- **Edge-Decay Status**: {edge_decay_str}",
        f"- **Circuit Breaker**: `{breaker_status}`",
        "",
    ]

    if sanity_alarm:
        md_lines.extend([
            "> [!WARNING]",
            f"> **{sanity_alarm_message}**",
            "",
        ])

    md_lines.extend([
        "## Overall Performance",
        "",
        f"- **Total Settled Trades**: {total_settled}",
        f"- **Wins / Losses / Ties**: {wins} / {losses} / {ties}",
        f"- **Win Rate (Wilson 95% CI)**: {win_rate:.1%} [{ci_low:.1%}, {ci_high:.1%}]",
        f"- **Break-Even Rate**: {overall_be:.1%}",
        f"- **Expected Value (EV)**: {overall_ev:+.2f} per trade",
        f"- **Total Net PnL**: `${total_pnl:,.2f}`",
        f"- **Realized Return**: {overall_return:+.2%}",
        "",
        "## Asset Performance & Paper Gate Progress",
        "",
        "| Asset | Trades | W / L / T | Win Rate (95% CI) | Break-Even | EV | Return | Paper Gate Progress | CI > BE | Status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    min_paper_trades = getattr(config.paper_gate, "min_paper_trades", 30)

    for asset in evaluated_assets:
        asset_day_trades = [t for t in settled if t.asset == asset]
        a_n = len(asset_day_trades)
        a_wins = sum(1 for t in asset_day_trades if t.result == "WIN")
        a_losses = sum(1 for t in asset_day_trades if t.result == "LOSS")
        a_ties = sum(1 for t in asset_day_trades if t.result == "TIE")
        a_wr = (a_wins / a_n) if a_n > 0 else 0.0
        a_ci_low, a_ci_high = wilson_interval(a_wins, a_n, z=1.96) if a_n > 0 else (0.0, 1.0)

        a_pnl = sum(t.pnl for t in asset_day_trades)
        a_stake = sum(t.stake for t in asset_day_trades)
        a_ret = (a_pnl / a_stake) if a_stake > 0 else 0.0

        stats = compute_asset_stats(trades=all_trades, asset=asset, mode="paper", as_of=day_end)
        if stats:
            be = stats.break_even
            a_ev = stats.ev
            total_n = stats.n
            gate_mode, _ = asset_mode(stats, config)
            ci_above_be = "YES" if stats.wilson_ci[0] > be else "NO"
            cum_ci_low, cum_ci_high = stats.wilson_ci
        else:
            be = calc_break_even(avg_payout)
            a_ev = 0.0
            total_n = a_n
            gate_mode = "paper"
            ci_above_be = "NO"
            cum_ci_low, cum_ci_high = (0.0, 1.0)

        gate_progress = f"{total_n} / {min_paper_trades}"
        md_lines.append(
            f"| `{asset}` | {a_n} | {a_wins}/{a_losses}/{a_ties} | {a_wr:.1%} [{a_ci_low:.1%}, {a_ci_high:.1%}] | "
            f"{be:.1%} | {a_ev:+.2f} | {a_ret:+.1%} | {gate_progress} | {ci_above_be} | `{gate_mode}` |"
        )

    md_lines.append("")
    md_lines.append("## Cycle Evaluation Skips")
    md_lines.append("")
    if skips:
        md_lines.append("| Skip Reason | Count |")
        md_lines.append("|---|---|")
        for reason, count in sorted(skips.items(), key=lambda x: -x[1]):
            md_lines.append(f"| {reason} | {count} |")
    else:
        md_lines.append("No cycle evaluation skips recorded for this day.")

    md_lines.append("")
    md_content = "\n".join(md_lines)

    tg_message = (
        f"📅 Daily Summary for {date_str}\n"
        f"• Mode: {config.mode}\n"
        f"• Trades: {total_settled} ({wins}W / {losses}L / {ties}T)\n"
        f"• Win Rate: {win_rate:.1%} [{ci_low:.1%}, {ci_high:.1%}]\n"
        f"• BE: {overall_be:.1%} | EV: {overall_ev:+.2f} | Return: {overall_return:+.1%}\n"
        f"• Net PnL: ${total_pnl:,.2f} | Balance: ${balance:,.2f}\n"
        f"• Tier: {tier} | Decay: {edge_decay_str}\n"
        f"• Breaker: {breaker_status}"
    )
    if sanity_alarm:
        tg_message += "\n⚠️ SANITY ALARM: Win rate >= 70% with >= 100 trades (verify look-ahead!)"

    return md_content, tg_message
