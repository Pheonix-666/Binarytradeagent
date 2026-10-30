"""Postmortem report generator summarizing losses, tier transitions, and context health.

PROJECT_SPEC.md Adaptive Layer (Phase 8A):
  - Pure reporting module: produces a clean Markdown summary for the user.
  - Never mutates config, adaptive state, or trade records.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from datetime import datetime, timezone

from adaptive.monitor import AdaptiveState
from adaptive.champion import ChampionTracker
from adaptive.contexts import ContextTracker


def generate_postmortem_markdown(
    state: AdaptiveState,
    champion_tracker: Optional[ChampionTracker] = None,
    context_tracker: Optional[ContextTracker] = None,
    recent_losses: Optional[Sequence[Any]] = None,
) -> str:
    """Generate Markdown postmortem diagnostic report for review.

    Pure function: no disk or network I/O, no state mutation.
    """
    lines: list[str] = [
        "# Adaptive Layer Postmortem & System Health Report",
        "",
        f"**Generated At:** {datetime.now(timezone.utc).isoformat()} UTC",
        "",
        "## 1. Current State & Regulator Status",
        "",
        f"- **Active Tier:** `{state.tier}`",
        f"- **Real Trading Allowed:** `{state.real_trading_allowed}`",
        f"- **Current Equity:** `${state.current_equity:,.2f}`",
        f"- **Reference Equity:** `${state.reference_equity:,.2f}`",
        f"- **Peak Drawdown:** `{state.drawdown_pct:.2f}%`",
        f"- **Consecutive Losses:** `{state.consecutive_losses}`",
        f"- **Effective Stake Scale:** `{state.stake_scale:.2f}x`",
        f"- **In Ramp-Up Period:** `{state.in_rampup}` ({state.rampup_trades_remaining} trades remaining)",
        f"- **Latest Reason:** {state.transition_reason}",
        "",
        "## 2. Tier Transition History",
        "",
        "| Timestamp | From Tier | To Tier | Drawdown | Losses | Reason |",
        "|---|---|---|---|---|---|",
    ]

    if not state.history:
        lines.append("| - | - | - | - | - | No transitions recorded |")
    else:
        for item in reversed(state.history[-20:]):
            ts = item.get("timestamp", "-")
            fr = item.get("from_tier", "-")
            to = item.get("to_tier", "-")
            dd = f"{item.get('drawdown_pct', 0.0):.2f}%"
            losses = item.get("consecutive_losses", 0)
            reason = item.get("reason", "")
            lines.append(f"| {ts} | `{fr}` | `{to}` | {dd} | {losses} | {reason} |")

    lines.extend([
        "",
        "## 3. Profile Champion & Shadow Challengers",
        "",
        "| Profile | Status | Trades | Win Rate | 95% CI Lower | EV |",
        "|---|---|---|---|---|---|",
    ])

    if champion_tracker is not None:
        champ_name = champion_tracker.champion_name
        for p_name, perf in champion_tracker.profiles.items():
            status = "**CHAMPION**" if p_name == champ_name else "Shadow"
            lines.append(
                f"| `{p_name}` | {status} | {perf.trades_count} | "
                f"{perf.win_rate * 100:.1f}% | {perf.ci_lower:.4f} | {perf.ev:+.4f} |"
            )
    else:
        lines.append("| standard | Active | - | - | - | - |")

    lines.extend([
        "",
        "## 4. Context Health & Suspended Dimensions",
        "",
    ])

    if context_tracker is not None and context_tracker.contexts:
        suspended = [s for s in context_tracker.contexts.values() if s.is_suspended]
        lines.append(f"- **Total Monitored Contexts:** {len(context_tracker.contexts)}")
        lines.append(f"- **Currently Suspended Contexts:** {len(suspended)}")
        lines.append("")
        lines.append("| Dimension Key | Trades | Win Rate | 99% CI Upper | Suspended Since | Shadow Re-tests |")
        lines.append("|---|---|---|---|---|---|")

        if not suspended:
            lines.append("| None | - | - | - | - | - |")
        else:
            for s in suspended:
                swr = (
                    f"{s.shadow_wins_since_suspension}/{s.shadow_trades_since_suspension}"
                    if s.shadow_trades_since_suspension > 0 else "0"
                )
                lines.append(
                    f"| `{s.context_key}` | {s.trades_count} | {s.win_rate * 100:.1f}% | "
                    f"{s.ci_upper_99:.4f} | {s.suspended_at or '-'} | {swr} |"
                )
    else:
        lines.append("- No context data collected yet.")

    return "\n".join(lines)
