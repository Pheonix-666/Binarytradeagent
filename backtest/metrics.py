"""Statistical metrics, block analysis, streak tracking, and baseline calculations.

PROJECT_SPEC.md Section 11 & Phase 6:
  - Overall win rate with Wilson 95% confidence interval.
  - Non-overlapping 10, 20, 50-trade block win rates vs target_win_rate (0.80).
  - Expected binomial block hit rate vs actual observed hit rate.
  - Longest and average losing streaks.
  - EV per trade, realized return, max drawdown.
  - Baseline controls:
      * Random direction Monte Carlo (>= 200 seeds) with 5th/95th percentile band.
      * Always CALL / Always PUT.
  - Sanity alarm when test win rate >= 70% with >= 100 trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

from data.store import TradeRecord
from selection.stats import break_even, ev, realized_return, wilson_interval


@dataclass
class BlockStats:
    """Metrics for fixed-size non-overlapping trade blocks."""

    block_size: int
    num_blocks: int
    hit_count: int  # Blocks with win rate >= target_win_rate
    actual_hit_rate: float
    expected_binomial_hit_rate: float
    block_win_rates: list[float] = field(default_factory=list)


@dataclass
class StreakStats:
    """Streak metrics for settled trades."""

    longest_losing_streak: int
    average_losing_streak: float
    longest_winning_streak: int
    average_winning_streak: float


@dataclass
class BaselineComparison:
    """Comparison against random and fixed-direction baselines."""

    strategy_win_rate: float
    random_band_p5: float
    random_band_p95: float
    random_mean: float
    always_call_win_rate: float
    always_put_win_rate: float
    beats_random_band: bool
    summary: str


@dataclass
class VariantMetrics:
    """Complete performance report for a backtest variant."""

    total_trades: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    wilson_ci: tuple[float, float]
    break_even: float
    ev_per_trade: float
    realized_return: float
    final_equity: float
    max_drawdown: float
    streaks: StreakStats
    blocks: dict[int, BlockStats]  # 10, 20, 50
    edge_label: str  # "positive_edge", "negative_edge", "edge unproven"
    sanity_alarm: bool = False
    sanity_alarm_message: str = ""
    breakdown_by_asset: dict[str, dict[str, Any]] = field(default_factory=dict)
    breakdown_by_pattern: dict[str, dict[str, Any]] = field(default_factory=dict)
    breakdown_by_hour: dict[int, dict[str, Any]] = field(default_factory=dict)
    breakdown_by_risk_bucket: dict[str, dict[str, Any]] = field(default_factory=dict)
    void_trades: int = 0


def binomial_survival_prob(n: int, k_req: int, p: float) -> float:
    """Compute P(X >= k_req) for X ~ Binomial(n, p)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    if k_req <= 0:
        return 1.0
    if k_req > n:
        return 0.0

    prob = 0.0
    for k in range(k_req, n + 1):
        # Comb(n, k) * p^k * (1-p)^(n-k)
        comb = math.comb(n, k)
        prob += comb * (p ** k) * ((1.0 - p) ** (n - k))
    return float(np.clip(prob, 0.0, 1.0))


def compute_streaks(trades: Sequence[TradeRecord]) -> StreakStats:
    """Calculate longest and average losing and winning streaks in chronological order."""
    valid_trades = [t for t in trades if t.result not in ("VOID", "PENDING")]
    if not valid_trades:
        return StreakStats(0, 0.0, 0, 0.0)

    losing_streaks: list[int] = []
    winning_streaks: list[int] = []

    cur_losing = 0
    cur_winning = 0

    for t in valid_trades:
        if t.result == "WIN":
            if cur_losing > 0:
                losing_streaks.append(cur_losing)
                cur_losing = 0
            cur_winning += 1
        elif t.result == "LOSS":
            if cur_winning > 0:
                winning_streaks.append(cur_winning)
                cur_winning = 0
            cur_losing += 1
        else:  # TIE
            # Does not break streak, or neutral
            pass

    if cur_losing > 0:
        losing_streaks.append(cur_losing)
    if cur_winning > 0:
        winning_streaks.append(cur_winning)

    longest_l = max(losing_streaks) if losing_streaks else 0
    avg_l = float(np.mean(losing_streaks)) if losing_streaks else 0.0

    longest_w = max(winning_streaks) if winning_streaks else 0
    avg_w = float(np.mean(winning_streaks)) if winning_streaks else 0.0

    return StreakStats(
        longest_losing_streak=longest_l,
        average_losing_streak=round(avg_l, 2),
        longest_winning_streak=longest_w,
        average_winning_streak=round(avg_w, 2),
    )


def compute_block_statistics(
    trades: Sequence[TradeRecord],
    block_sizes: Sequence[int] = (10, 20, 50),
    target_win_rate: float = 0.80,
) -> dict[int, BlockStats]:
    """Calculate non-overlapping block win rates and expected vs actual hit rates.

    target_win_rate (default 0.80) is used for reporting only.
    VOID trades are excluded from block win-rate evaluations.
    """
    valid_trades = [t for t in trades if t.result not in ("VOID", "PENDING")]
    n_total = len(valid_trades)
    overall_wins = sum(1 for t in valid_trades if t.result == "WIN")
    overall_p = (overall_wins / n_total) if n_total > 0 else 0.0

    results: dict[int, BlockStats] = {}

    for b_size in block_sizes:
        num_blocks = n_total // b_size
        if num_blocks == 0:
            k_req = math.ceil(b_size * target_win_rate)
            exp_p = binomial_survival_prob(b_size, k_req, overall_p)
            results[b_size] = BlockStats(
                block_size=b_size,
                num_blocks=0,
                hit_count=0,
                actual_hit_rate=0.0,
                expected_binomial_hit_rate=round(exp_p, 4),
                block_win_rates=[],
            )
            continue

        k_req = math.ceil(b_size * target_win_rate)
        exp_p = binomial_survival_prob(b_size, k_req, overall_p)

        block_wrs: list[float] = []
        hits = 0

        for i in range(num_blocks):
            sub_trades = valid_trades[i * b_size : (i + 1) * b_size]
            b_wins = sum(1 for t in sub_trades if t.result == "WIN")
            b_wr = b_wins / b_size
            block_wrs.append(b_wr)
            if b_wr >= target_win_rate:
                hits += 1

        actual_rate = hits / num_blocks
        results[b_size] = BlockStats(
            block_size=b_size,
            num_blocks=num_blocks,
            hit_count=hits,
            actual_hit_rate=round(actual_rate, 4),
            expected_binomial_hit_rate=round(exp_p, 4),
            block_win_rates=[round(w, 4) for w in block_wrs],
        )

    return results


def compute_variant_metrics(
    trades: Sequence[TradeRecord],
    starting_balance: float = 10000.0,
    final_equity: float = 10000.0,
    max_drawdown: float = 0.0,
    target_win_rate: float = 0.80,
    is_test_set: bool = False,
) -> VariantMetrics:
    """Compute comprehensive performance metrics from trade logs."""
    n_all = len(trades)
    valid_trades = [t for t in trades if t.result not in ("VOID", "PENDING")]
    n = len(valid_trades)
    void_count = sum(1 for t in trades if t.result == "VOID")

    if n == 0:
        be = break_even(0.85)
        return VariantMetrics(
            total_trades=n_all,
            wins=0,
            losses=0,
            ties=0,
            win_rate=0.0,
            wilson_ci=(0.0, 1.0),
            break_even=round(be, 4),
            ev_per_trade=0.0,
            realized_return=0.0,
            final_equity=final_equity,
            max_drawdown=max_drawdown,
            streaks=StreakStats(0, 0.0, 0, 0.0),
            blocks=compute_block_statistics([], target_win_rate=target_win_rate),
            edge_label="edge unproven",
            void_trades=void_count,
        )

    wins = sum(1 for t in valid_trades if t.result == "WIN")
    losses = sum(1 for t in valid_trades if t.result == "LOSS")
    ties = sum(1 for t in valid_trades if t.result == "TIE")
    win_rate = wins / n

    payouts = [t.payout_at_entry for t in valid_trades]
    mean_payout = float(np.mean(payouts)) if payouts else 0.85
    be = break_even(mean_payout)
    ev_val = ev(win_rate, mean_payout)
    ret = realized_return(valid_trades)
    ci = wilson_interval(wins, n)

    # Edge classification
    if ci[0] <= be <= ci[1]:
        edge_label = "edge unproven"
    elif ci[0] > be:
        edge_label = "positive_edge"
    else:
        edge_label = "negative_edge"

    streaks = compute_streaks(trades)
    blocks = compute_block_statistics(trades, target_win_rate=target_win_rate)

    # Sanity alarm check: if test set win rate >= 70% with >= 100 trades
    sanity_alarm = False
    alarm_msg = ""
    if is_test_set and n >= 100 and win_rate >= 0.70:
        sanity_alarm = True
        alarm_msg = (
            "SANITY ALARM: Observed test set win rate >= 70% with >= 100 trades! "
            "Rigorous verification required: check for look-ahead bias, causal index leakage, "
            "or optimistic fill assumptions. Checklist: (1) Look-ahead guard test passed; "
            "(2) Entry delay is >= 1 bar; (3) Slippage penalty applied; (4) Indicators purely causal."
        )

    # Breakdowns
    # Asset
    by_asset: dict[str, dict[str, Any]] = {}
    for a in set(t.asset for t in trades):
        a_trades = [t for t in trades if t.asset == a]
        a_w = sum(1 for t in a_trades if t.result == "WIN")
        by_asset[a] = {
            "n": len(a_trades),
            "wins": a_w,
            "win_rate": round(a_w / len(a_trades), 4),
            "pnl": round(sum(t.pnl for t in a_trades), 2),
        }

    # Pattern
    by_pattern: dict[str, dict[str, Any]] = {}
    for p in set(t.pattern_name for t in trades if t.pattern_name):
        p_trades = [t for t in trades if t.pattern_name == p]
        p_w = sum(1 for t in p_trades if t.result == "WIN")
        by_pattern[p] = {
            "n": len(p_trades),
            "wins": p_w,
            "win_rate": round(p_w / len(p_trades), 4),
            "pnl": round(sum(t.pnl for t in p_trades), 2),
        }

    # Hour of day (UTC)
    by_hour: dict[int, dict[str, Any]] = {}
    for h in range(24):
        h_trades = [t for t in trades if t.hour_utc == h]
        if h_trades:
            h_w = sum(1 for t in h_trades if t.result == "WIN")
            by_hour[h] = {
                "n": len(h_trades),
                "wins": h_w,
                "win_rate": round(h_w / len(h_trades), 4),
                "pnl": round(sum(t.pnl for t in h_trades), 2),
            }

    # Risk buckets (1-20, 21-40, 41-60, 61-80, 81-100)
    risk_ranges = [("1-20", 1, 20), ("21-40", 21, 40), ("41-60", 41, 60), ("61-80", 61, 80), ("81-100", 81, 100)]
    by_risk: dict[str, dict[str, Any]] = {}
    for name, r_min, r_max in risk_ranges:
        r_trades = [t for t in trades if t.risk_score is not None and r_min <= t.risk_score <= r_max]
        if r_trades:
            r_w = sum(1 for t in r_trades if t.result == "WIN")
            by_risk[name] = {
                "n": len(r_trades),
                "wins": r_w,
                "win_rate": round(r_w / len(r_trades), 4),
                "pnl": round(sum(t.pnl for t in r_trades), 2),
            }

    return VariantMetrics(
        total_trades=n_all,
        wins=wins,
        losses=losses,
        ties=ties,
        win_rate=round(win_rate, 4),
        wilson_ci=(round(ci[0], 4), round(ci[1], 4)),
        break_even=round(be, 4),
        ev_per_trade=round(ev_val, 4),
        realized_return=round(ret, 4),
        final_equity=round(final_equity, 2),
        max_drawdown=round(max_drawdown, 4),
        streaks=streaks,
        blocks=blocks,
        edge_label=edge_label,
        sanity_alarm=sanity_alarm,
        sanity_alarm_message=alarm_msg,
        breakdown_by_asset=by_asset,
        breakdown_by_pattern=by_pattern,
        breakdown_by_hour=by_hour,
        breakdown_by_risk_bucket=by_risk,
        void_trades=void_count,
    )


def compute_baseline_comparison(
    trades: Sequence[TradeRecord],
    num_seeds: int = 200,
) -> BaselineComparison:
    """Evaluate against random direction Monte Carlo, always CALL, and always PUT baselines.

    Preserves exact entry times, exit times, and stakes.
    """
    n = len(trades)
    if n == 0:
        return BaselineComparison(
            strategy_win_rate=0.0,
            random_band_p5=0.0,
            random_band_p95=0.0,
            random_mean=0.0,
            always_call_win_rate=0.0,
            always_put_win_rate=0.0,
            beats_random_band=False,
            summary="No trades to evaluate against baselines.",
        )

    strategy_wins = sum(1 for t in trades if t.result == "WIN")
    strategy_wr = strategy_wins / n

    # Fixed direction baselines
    call_wins = sum(1 for t in trades if (t.direction == "CALL" and t.result == "WIN") or (t.direction == "PUT" and t.result == "LOSS"))
    put_wins = sum(1 for t in trades if (t.direction == "PUT" and t.result == "WIN") or (t.direction == "CALL" and t.result == "LOSS"))
    call_wr = call_wins / n
    put_wr = put_wins / n

    # Monte Carlo random direction
    rng = np.random.default_rng(42)
    random_wrs: list[float] = []

    for _ in range(num_seeds):
        # 50% random coin flip direction for each trade
        rand_dirs = rng.integers(0, 2, size=n)  # 0 for CALL, 1 for PUT
        r_wins = 0
        for i, t in enumerate(trades):
            chosen_call = (rand_dirs[i] == 0)
            orig_call = (t.direction == "CALL")
            # If chosen matches original direction, outcome is same; if flipped, outcome is inverted
            if chosen_call == orig_call:
                if t.result == "WIN":
                    r_wins += 1
            else:
                if t.result == "LOSS":
                    r_wins += 1
        random_wrs.append(r_wins / n)

    p5 = float(np.percentile(random_wrs, 5))
    p95 = float(np.percentile(random_wrs, 95))
    r_mean = float(np.mean(random_wrs))

    beats_band = strategy_wr > p95
    if beats_band:
        summary = (
            f"Strategy win rate ({strategy_wr:.1%}) is significantly ABOVE the random baseline 95th percentile ({p95:.1%})."
        )
    elif strategy_wr < p5:
        summary = (
            f"Strategy win rate ({strategy_wr:.1%}) is significantly BELOW the random baseline 5th percentile ({p5:.1%})."
        )
    else:
        summary = (
            f"Strategy win rate ({strategy_wr:.1%}) is INSIDE the random baseline band [{p5:.1%}, {p95:.1%}] (consistent with random chance)."
        )

    return BaselineComparison(
        strategy_win_rate=round(strategy_wr, 4),
        random_band_p5=round(p5, 4),
        random_band_p95=round(p95, 4),
        random_mean=round(r_mean, 4),
        always_call_win_rate=round(call_wr, 4),
        always_put_win_rate=round(put_wr, 4),
        beats_random_band=beats_band,
        summary=summary,
    )


def generate_adaptive_comparison_report(
    result_off: Any,
    result_on: Any,
    target_win_rate: float = 0.80,
) -> str:
    """Generate Markdown comparison report of backtest run with adaptive OFF vs ON.

    Compares side by side:
      - Final equity & return
      - Max drawdown
      - Trades taken (total, wins, losses)
      - Win rate with Wilson 95% CI
      - Time spent in each tier
      - Profile switches and context suspensions
      - Explanatory note if ON run traded significantly less
      - Warning: Do NOT tune adaptive thresholds on the test period
    """
    def _metrics(res: Any) -> tuple[int, int, int, float, tuple[float, float], float]:
        trades = getattr(res, "trades", [])
        wins = sum(1 for t in trades if getattr(t, "result", "") == "WIN")
        losses = sum(1 for t in trades if getattr(t, "result", "") == "LOSS")
        valid = wins + losses
        wr = (wins / valid) if valid > 0 else 0.0
        ci = wilson_interval(wins, valid) if valid > 0 else (0.0, 0.0)
        ret = ((res.final_equity - res.starting_balance) / res.starting_balance) * 100.0 if res.starting_balance > 0 else 0.0
        return len(trades), wins, losses, wr, ci, ret

    n_off, w_off, l_off, wr_off, ci_off, ret_off = _metrics(result_off)
    n_on, w_on, l_on, wr_on, ci_on, ret_on = _metrics(result_on)

    tier_times = getattr(result_on, "time_in_tier", {})
    total_tier_steps = sum(tier_times.values()) if tier_times else 1

    lines = [
        "# Adaptive Layer Comparative Evaluation: OFF vs ON",
        "",
        "> [!WARNING]",
        "> **Do NOT tune adaptive thresholds on the test period.**",
        "> Adaptive parameters must be configured a priori from risk management principles,",
        "> not optimized to fit backtest results.",
        "",
        "## Performance Comparison",
        "",
        "| Metric | Adaptive OFF (Baseline) | Adaptive ON | Difference |",
        "|---|---|---|---|",
        f"| **Starting Balance** | ${result_off.starting_balance:,.2f} | ${result_on.starting_balance:,.2f} | $0.00 |",
        f"| **Final Equity** | ${result_off.final_equity:,.2f} | ${result_on.final_equity:,.2f} | ${result_on.final_equity - result_off.final_equity:+,.2f} |",
        f"| **Total Return** | {ret_off:+.2f}% | {ret_on:+.2f}% | {ret_on - ret_off:+.2f}% |",
        f"| **Max Drawdown** | {result_off.max_drawdown * 100:.2f}% | {result_on.max_drawdown * 100:.2f}% | {(result_on.max_drawdown - result_off.max_drawdown) * 100:+.2f}% |",
        f"| **Trades Taken** | {n_off} (W: {w_off}, L: {l_off}) | {n_on} (W: {w_on}, L: {l_on}) | {n_on - n_off:+d} trades |",
        f"| **Win Rate** | {wr_off * 100:.1f}% | {wr_on * 100:.1f}% | {(wr_on - wr_off) * 100:+.1f}% |",
        f"| **Wilson 95% CI** | [{ci_off[0] * 100:.1f}%, {ci_off[1] * 100:.1f}%] | [{ci_on[0] * 100:.1f}%, {ci_on[1] * 100:.1f}%] | - |",
        "",
    ]

    # Explicit note when ON traded much less
    if n_off > 0 and n_on < n_off:
        diff_trades = n_off - n_on
        pct_filtered = (diff_trades / n_off) * 100.0

        switches = getattr(result_on, "profile_switches", 0)
        suspensions = getattr(result_on, "context_suspensions", 0)
        paused_steps = tier_times.get("PAUSED", 0)
        pause_events = 0
        if getattr(result_on, "adaptive_state", None) and hasattr(result_on.adaptive_state, "history"):
            pause_events = sum(1 for h in result_on.adaptive_state.history if h.get("to_tier") == "PAUSED")

        actions = []
        if pause_events > 0 or paused_steps > 0:
            p_count = pause_events if pause_events > 0 else 1
            actions.append(f"{p_count} pauses ({paused_steps} steps in PAUSED)")
        if suspensions > 0:
            actions.append(f"{suspensions} context suspensions")
        if switches > 0:
            actions.append(f"{switches} profile switches")

        if actions:
            action_desc = f" Interventions: {', '.join(actions)}."
        else:
            action_desc = " The adaptive regulator tightened risk thresholds and filtered lower-ranking candidate setups."

        lines.extend([
            "> [!NOTE]",
            f"> **Selective Trading:** Adaptive mode executed {n_on} trades vs {n_off} in baseline "
            f"({diff_trades} fewer trades, {pct_filtered:.1f}% reduction).{action_desc}",
            "",
        ])

    lines.extend([
        "## Adaptive Regulator Telemetry",
        "",
        f"- **Profile Switches:** `{getattr(result_on, 'profile_switches', 0)}`",
        f"- **Active Context Suspensions:** `{getattr(result_on, 'context_suspensions', 0)}`",
        f"- **Final State Tier:** `{getattr(result_on.adaptive_state, 'tier', 'N/A') if getattr(result_on, 'adaptive_state', None) else 'N/A'}`",
        "",
        "### Time Spent in Tiers",
        "",
        "| Tier | Steps Spent | % of Simulation |",
        "|---|---|---|",
    ])

    for t_name in ("NORMAL", "CAUTION", "DEFENSIVE", "PAUSED"):
        steps = tier_times.get(t_name, 0)
        pct = (steps / total_tier_steps) * 100.0 if total_tier_steps > 0 else 0.0
        lines.append(f"| `{t_name}` | {steps} | {pct:.1f}% |")

    return "\n".join(lines)
