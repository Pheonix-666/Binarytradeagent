"""Edge-decay detector for adaptive risk regulation.

PROJECT_SPEC.md Adaptive Layer (Phase 8A & Phase 8 Evidence Pass):
  - Two-level rule:
      * CAUTION uses z_caution (default 1.0, one-sided) over window_trades_caution (default 60).
        CAUTION only tightens, so it may be sensitive.
      * DEFENSIVE uses z (default 1.96) over persist_windows consecutive non-overlapping
        windows of window_trades (default 100).
  - Helper trades_to_detect(shortfall, payout, z) returning approximate number of trades
    needed to detect a given win-rate shortfall.
  - Tier recovery follows existing hysteresis rules (tightens only, never loosens).
  - Pure function, no I/O, no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence

from core.config import AppConfig, EdgeDecayConfig
from selection.stats import break_even, wilson_interval


@dataclass(frozen=True)
class EdgeDecayResult:
    """Telemetry and recommendation from edge decay evaluation."""

    decay_detected: bool
    persistent: bool
    recommended_tier: Optional[str]
    n_trades_evaluated: int
    win_rate: float
    ci_upper: float
    break_even_rate: float
    reason: str


def trades_to_detect(
    shortfall: float,
    payout: float = 0.85,
    z: float = 1.0,
) -> int:
    """Approximate number of trades needed to detect a given win-rate shortfall.

    Formula:
      p0 = break_even(payout) = 1 / (1 + payout)
      p1 = max(0.01, min(0.99, p0 - shortfall))
      n = ceil( (z^2 * p1 * (1 - p1)) / (shortfall^2) )

    Parameters:
      shortfall: Win-rate degradation below break-even (e.g. 0.05 or 5.0 for 5 points).
      payout: Payout rate (e.g. 0.85).
      z: Normal critical value (e.g. 1.0 for CAUTION, 1.96 for DEFENSIVE).

    Returns:
      Approximate integer number of trades N.
    """
    if shortfall <= 0:
        raise ValueError("shortfall must be > 0")
    if shortfall > 1.0:
        shortfall = shortfall / 100.0  # Normalize percentage points (e.g. 5 -> 0.05)

    p0 = break_even(payout)
    p1 = max(0.01, min(0.99, p0 - shortfall))
    var = p1 * (1.0 - p1)
    n = (z * z * var) / (shortfall * shortfall)
    return int(math.ceil(n))


def evaluate_edge_decay(
    trades: Sequence[Any],
    config: AppConfig,
    default_payout: float = 0.85,
) -> EdgeDecayResult:
    """Evaluate whether edge has decayed using two-level CAUTION and DEFENSIVE criteria.

    Parameters:
      trades: Chronological sequence of closed trades for the active profile.
      config: System AppConfig containing adaptive.edge_decay settings.
      default_payout: Fallback payout if individual trade payouts are missing.

    Returns:
      EdgeDecayResult containing decay indicators and recommended minimum tier.
    """
    cfg: EdgeDecayConfig = config.adaptive.edge_decay
    w_caution = cfg.window_trades_caution
    z_caution = cfg.z_caution
    w_defensive = cfg.window_trades
    min_trades = cfg.min_trades
    z_defensive = cfg.z
    persist_windows = cfg.persist_windows
    tier_on_decay = cfg.tier_on_decay.upper()
    tier_on_persistent = cfg.tier_on_persistent.upper()

    # 1. Filter settled binary trades (WIN or LOSS only)
    settled: list[tuple[str, float]] = []
    for t in trades:
        res = getattr(t, "result", None)
        if isinstance(t, dict):
            res = t.get("result")
        if not res:
            continue
        res_str = str(res).upper()
        if res_str not in ("WIN", "LOSS"):
            continue

        payout = getattr(t, "payout_at_entry", getattr(t, "payout", None))
        if isinstance(t, dict):
            payout = t.get("payout_at_entry", t.get("payout"))
        if payout is None or payout <= 0.0:
            payout = default_payout

        settled.append((res_str, float(payout)))

    total_settled = len(settled)

    # 2. Insufficient trades check
    effective_min = min(min_trades, w_caution)
    if total_settled < effective_min:
        return EdgeDecayResult(
            decay_detected=False,
            persistent=False,
            recommended_tier=None,
            n_trades_evaluated=total_settled,
            win_rate=0.0,
            ci_upper=1.0,
            break_even_rate=0.5,
            reason=f"Insufficient trades ({total_settled} < min_trades {effective_min})",
        )

    # 3. Level 1: CAUTION detection
    # Evaluates over rolling window of window_trades_caution using z_caution
    slice_caution = settled[-min(total_settled, w_caution):]
    n_caution = len(slice_caution)
    wins_caution = sum(1 for res, _ in slice_caution if res == "WIN")
    mean_payout_caution = sum(p for _, p in slice_caution) / float(n_caution)
    be_caution = break_even(mean_payout_caution)

    wr_caution = wins_caution / float(n_caution)
    _, ci_upper_caution = wilson_interval(wins_caution, n_caution, z=z_caution)

    decay_detected = bool(ci_upper_caution < be_caution)

    # 4. Level 2: DEFENSIVE persistent detection
    # Evaluates persist_windows consecutive non-overlapping windows of window_trades using z
    persistent = False
    required_trades = persist_windows * w_defensive
    if total_settled >= required_trades:
        all_windows_decayed = True
        for i in range(persist_windows):
            start_idx = total_settled - (persist_windows - i) * w_defensive
            end_idx = total_settled - (persist_windows - i - 1) * w_defensive
            w_slice = settled[start_idx:end_idx]
            w_wins = sum(1 for res, _ in w_slice if res == "WIN")
            w_mean_payout = sum(p for _, p in w_slice) / float(w_defensive)
            w_be = break_even(w_mean_payout)
            _, w_ci_upper = wilson_interval(w_wins, w_defensive, z=z_defensive)
            if not (w_ci_upper < w_be):
                all_windows_decayed = False
                break
        persistent = all_windows_decayed

    # 5. Determine recommended minimum tier
    if persistent:
        rec_tier = tier_on_persistent
        reason = (
            f"Persistent edge decay across {persist_windows} non-overlapping windows of {w_defensive} trades "
            f"(z={z_defensive:.2f})"
        )
    elif decay_detected:
        rec_tier = tier_on_decay
        reason = (
            f"Edge decay detected (CAUTION): Wilson upper bound {ci_upper_caution:.4f} < break-even {be_caution:.4f} "
            f"(window={n_caution}, z={z_caution:.2f}, win rate={wr_caution:.3f})"
        )
    else:
        rec_tier = None
        reason = f"Edge intact: Wilson CI upper {ci_upper_caution:.4f} >= break-even {be_caution:.4f}"

    return EdgeDecayResult(
        decay_detected=decay_detected,
        persistent=persistent,
        recommended_tier=rec_tier,
        n_trades_evaluated=n_caution,
        win_rate=round(wr_caution, 4),
        ci_upper=round(ci_upper_caution, 4),
        break_even_rate=round(be_caution, 4),
        reason=reason,
    )
