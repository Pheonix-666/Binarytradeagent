"""Champion-challenger profile evaluation and shadow tracking.

PROJECT_SPEC.md Adaptive Layer (Phase 8A):
  - Every profile is evaluated continuously as shadow trades.
  - EFFICIENCY: Expensive per-step work (indicators, trend labels, pattern matching)
    is shared across all profiles; profile-specific thresholds are filtered afterwards.
  - The real-money profile changes ONLY under strict profile_switch rules:
      * min_shadow_trades >= 50
      * require_ci_lower_above_break_even: Wilson 95% lower bound > break_even
      * min_ev_advantage >= 0.02
      * cooldown_days >= 3
  - Serializable: to_dict() and from_dict().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional, Sequence

from analysis.candidates import CandidateTrade
from core.config import AppConfig, ProfileSwitchConfig
from selection.stats import break_even, compute_asset_stats, ev, wilson_interval


@dataclass
class ProfilePerformance:
    """Historical shadow performance of a trading profile."""

    profile_name: str
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    voids: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    ev: float = 0.0

    def record_trade(self, result: str, payout: float = 0.85) -> None:
        """Update metrics with a closed trade result."""
        res = result.upper()
        if res == "WIN":
            self.wins += 1
            self.total_pnl += payout
        elif res == "LOSS":
            self.losses += 1
            self.total_pnl -= 1.0
        elif res in ("REFUND", "VOID", "TIE"):
            self.voids += 1

        self.trades_count += 1
        valid_trades = self.wins + self.losses
        if valid_trades > 0:
            self.win_rate = self.wins / valid_trades
            low, high = wilson_interval(self.wins, valid_trades, z=1.96)
            self.ci_lower = round(low, 4)
            self.ci_upper = round(high, 4)
            self.ev = round(ev(self.win_rate, payout), 4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfilePerformance:
        return cls(**data)


def filter_candidates_for_profile(
    candidates: Sequence[CandidateTrade],
    profile_dict: Mapping[str, Any],
) -> list[CandidateTrade]:
    """Filter pre-computed candidates using profile-specific thresholds.

    Zero redundant recomputation: indicators, trend, and DTW similarity
    are computed ONCE and filtered here in O(N).
    """
    min_sim = profile_dict.get("patterns.min_similarity", profile_dict.get("min_similarity"))
    max_risk = profile_dict.get("max_risk_cap")
    rev_mode = profile_dict.get("reversal_mode")

    filtered: list[CandidateTrade] = []
    for cand in candidates:
        # 1. Similarity filter
        if min_sim is not None and float(cand.pattern_match.similarity) < float(min_sim):
            continue

        # 2. Risk cap filter
        if max_risk is not None and float(cand.risk_report.score) > float(max_risk):
            continue

        # 3. Reversal mode filter
        if rev_mode in ("off", False) and cand.pattern_match.pattern_type == "reversal":
            continue

        filtered.append(cand)

    return filtered


class ChampionTracker:
    """Manages active champion profile and evaluates shadow challengers."""

    def __init__(
        self,
        config: AppConfig,
        champion_name: str = "standard",
        last_switch_time: Optional[datetime] = None,
    ):
        self.config = config
        self.champion_name = champion_name
        self.last_switch_time = last_switch_time
        self.profiles: dict[str, ProfilePerformance] = {}

        # Initialize tracking for all defined profiles
        profile_names = list(config.adaptive.profiles.keys())
        if not profile_names:
            profile_names = ["standard", "conservative", "trend_long_expiry"]
        for p in profile_names:
            self.profiles[p] = ProfilePerformance(profile_name=p)

    def record_shadow_result(self, profile_name: str, result: str, payout: float = 0.85) -> None:
        """Record trade result for a specific profile."""
        if profile_name not in self.profiles:
            self.profiles[profile_name] = ProfilePerformance(profile_name=profile_name)
        self.profiles[profile_name].record_trade(result=result, payout=payout)

    def evaluate_switch(self, now: datetime) -> tuple[Optional[str], Optional[str]]:
        """Evaluate whether a challenger qualifies to unseat current champion.

        Returns:
            (new_champion_name, reason) if switch occurred, else (None, None).
        """
        cfg: ProfileSwitchConfig = self.config.adaptive.profile_switch
        champ_perf = self.profiles.get(self.champion_name)
        if not champ_perf:
            return None, None

        # 1. Cooldown check
        if self.last_switch_time is not None:
            elapsed = now - self.last_switch_time
            if elapsed < timedelta(days=cfg.cooldown_days):
                return None, None

        payout = getattr(self.config.backtest, "default_payout", 0.85)
        be = break_even(payout)

        best_challenger: Optional[str] = None
        best_advantage: float = 0.0

        for name, perf in self.profiles.items():
            if name == self.champion_name:
                continue

            # a) Minimum shadow trade requirement
            if perf.trades_count < cfg.min_shadow_trades:
                continue

            # b) Statistical significance: Wilson CI lower bound must exceed break-even
            if cfg.require_ci_lower_above_break_even:
                if perf.ci_lower <= be:
                    continue

            # c) EV Advantage check
            advantage = perf.ev - champ_perf.ev
            if advantage >= cfg.min_ev_advantage and advantage > best_advantage:
                best_advantage = advantage
                best_challenger = name

        if best_challenger is not None:
            old_champ = self.champion_name
            self.champion_name = best_challenger
            self.last_switch_time = now
            reason = (
                f"Profile switched from {old_champ} to {best_challenger}: "
                f"EV advantage +{best_advantage:.4f} >= {cfg.min_ev_advantage}, "
                f"trades={self.profiles[best_challenger].trades_count}, "
                f"CI lower {self.profiles[best_challenger].ci_lower:.4f} > break_even {be:.4f}"
            )
            return best_challenger, reason

        return None, None

    def to_dict(self) -> dict[str, Any]:
        """Serialize champion tracker state."""
        return {
            "champion_name": self.champion_name,
            "last_switch_time": self.last_switch_time.isoformat() if self.last_switch_time else None,
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: AppConfig) -> ChampionTracker:
        """Restore champion tracker from serialized state."""
        champ_name = data.get("champion_name", "standard")
        last_sw_str = data.get("last_switch_time")
        last_sw = datetime.fromisoformat(last_sw_str) if last_sw_str else None
        tracker = cls(config=config, champion_name=champ_name, last_switch_time=last_sw)

        profiles_data = data.get("profiles", {})
        for name, p_data in profiles_data.items():
            tracker.profiles[name] = ProfilePerformance.from_dict(p_data)
        return tracker
