"""Context tagging, performance tracking, and statistical context suspension.

PROJECT_SPEC.md Adaptive Layer (Phase 8A):
  - Tag every trade (real and shadow) with its context:
      [asset, hour_bucket, pattern, risk_bucket, volatility_bucket, expiry]
  - Wilson upper bound against break-even:
      If n >= min_trades (30) and Wilson 99% upper bound (z=2.58) < break_even,
      the context is statistically proven negative EV and is suspended.
  - Cooldown: suspended for at least cooldown_days (3 days).
  - Cap: at most max_suspended_fraction (0.3 = 30%) of observed contexts suspended.
  - Retest in shadow: shadow performance tracked during suspension; reinstated if performance recovers.
  - Candidate filtering: build_candidates skips suspended contexts with explicit reason.
  - Serializable: to_dict() and from_dict().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional, Sequence

from analysis.candidates import CandidateTrade
from core.config import AppConfig, ContextLearningConfig
from selection.stats import break_even, wilson_interval


def get_risk_bucket(risk_score: float) -> str:
    """Bucket risk score into discrete low/med/high categories."""
    if risk_score <= 30.0:
        return "risk_low"
    elif risk_score <= 60.0:
        return "risk_med"
    return "risk_high"


def get_volatility_bucket(vol_pct: float) -> str:
    """Bucket volatility percentile into discrete low/med/high categories."""
    if vol_pct <= 33.3:
        return "vol_low"
    elif vol_pct <= 66.6:
        return "vol_med"
    return "vol_high"


def extract_context_tags(
    asset: str,
    hour_utc: int,
    pattern_name: str,
    risk_score: float,
    volatility_pct: float,
    expiry_minutes: int,
) -> dict[str, str]:
    """Extract individual and compound context dimension tags for a trade or candidate."""
    tags = {
        "asset": asset,
        "hour_bucket": f"h_{int(hour_utc):02d}",
        "pattern": pattern_name,
        "risk_bucket": get_risk_bucket(risk_score),
        "volatility_bucket": get_volatility_bucket(volatility_pct),
        "expiry": f"{int(expiry_minutes)}m",
    }
    return tags


@dataclass
class ContextStats:
    """Performance tracking for a single context dimension."""

    context_key: str  # e.g. "asset:EURUSD" or "pattern:double_top"
    dimension: str    # e.g. "asset", "pattern", etc.
    value: str
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    voids: int = 0
    win_rate: float = 0.0
    ci_upper_99: float = 1.0
    is_suspended: bool = False
    suspended_at: Optional[str] = None
    shadow_trades_since_suspension: int = 0
    shadow_wins_since_suspension: int = 0

    def record_trade(self, result: str, z: float = 2.58) -> None:
        """Record trade result and update Wilson upper bound."""
        res = result.upper()
        if res == "WIN":
            self.wins += 1
            if self.is_suspended:
                self.shadow_wins_since_suspension += 1
        elif res == "LOSS":
            self.losses += 1
        elif res in ("REFUND", "VOID", "TIE"):
            self.voids += 1

        self.trades_count += 1
        if self.is_suspended:
            self.shadow_trades_since_suspension += 1

        valid_trades = self.wins + self.losses
        if valid_trades > 0:
            self.win_rate = self.wins / valid_trades
            _, upper = wilson_interval(self.wins, valid_trades, z=z)
            self.ci_upper_99 = round(upper, 4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextStats:
        return cls(**data)


class ContextTracker:
    """Tracks context performance, enforces suspension limits, and filters candidates."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.contexts: dict[str, ContextStats] = {}
        self.suspended_contexts: set[str] = set()

    def _get_or_create(self, dim: str, val: str) -> ContextStats:
        key = f"{dim}:{val}"
        if key not in self.contexts:
            self.contexts[key] = ContextStats(context_key=key, dimension=dim, value=val)
        return self.contexts[key]

    def record_trade(
        self,
        asset: str,
        hour_utc: int,
        pattern_name: str,
        risk_score: float,
        volatility_pct: float,
        expiry_minutes: int,
        result: str,
        now: Optional[datetime] = None,
    ) -> None:
        """Tag trade with context features and record outcome across all dimensions."""
        cfg: ContextLearningConfig = self.config.adaptive.context_learning
        tags = extract_context_tags(
            asset=asset,
            hour_utc=hour_utc,
            pattern_name=pattern_name,
            risk_score=risk_score,
            volatility_pct=volatility_pct,
            expiry_minutes=expiry_minutes,
        )

        for dim in cfg.keys:
            if dim in tags:
                stats = self._get_or_create(dim, tags[dim])
                stats.record_trade(result=result, z=cfg.z)

        # Periodically re-evaluate suspensions
        if now is not None:
            self.evaluate_suspensions(now=now)

    def evaluate_suspensions(self, now: datetime) -> None:
        """Evaluate context suspension candidates, respect cap and cooldowns."""
        cfg: ContextLearningConfig = self.config.adaptive.context_learning
        payout = getattr(self.config.backtest, "default_payout", 0.85)
        be = break_even(payout)
        cooldown = timedelta(days=cfg.cooldown_days)

        # 1. Check existing suspensions for shadow recovery
        to_reinstate: list[str] = []
        for key in list(self.suspended_contexts):
            stats = self.contexts.get(key)
            if not stats or not stats.suspended_at:
                continue

            suspended_time = datetime.fromisoformat(stats.suspended_at)
            # Must exceed cooldown period before potential reinstatement
            if (now - suspended_time) >= cooldown:
                # Check shadow recovery: either Wilson upper bound recovered >= break_even
                # or shadow win rate in shadow trades >= break_even
                if stats.shadow_trades_since_suspension >= 10:
                    shadow_wr = stats.shadow_wins_since_suspension / stats.shadow_trades_since_suspension
                    if shadow_wr >= be or stats.ci_upper_99 >= be:
                        to_reinstate.append(key)
                elif stats.ci_upper_99 >= be:
                    to_reinstate.append(key)

        for key in to_reinstate:
            stats = self.contexts[key]
            stats.is_suspended = False
            stats.suspended_at = None
            stats.shadow_trades_since_suspension = 0
            stats.shadow_wins_since_suspension = 0
            self.suspended_contexts.discard(key)

        # 2. Check candidates qualifying for new suspension
        # Qualifies if n >= min_trades and Wilson 99% upper bound < break_even
        qualifying_keys: list[str] = []
        for key, stats in self.contexts.items():
            if stats.is_suspended:
                qualifying_keys.append(key)
                continue

            if stats.trades_count >= cfg.min_trades and stats.ci_upper_99 < be:
                qualifying_keys.append(key)

        # 3. Enforce maximum suspended fraction cap
        total_contexts = len(self.contexts)
        if total_contexts == 0:
            return

        max_allowed_suspended = max(1, int(total_contexts * cfg.max_suspended_fraction))

        if len(qualifying_keys) > max_allowed_suspended:
            # Sort by worst win rate / lowest upper bound to suspend the worst
            qualifying_keys.sort(key=lambda k: (self.contexts[k].ci_upper_99, self.contexts[k].win_rate))
            qualifying_keys = qualifying_keys[:max_allowed_suspended]

        # Apply suspensions
        newly_suspended = set(qualifying_keys)
        for key in newly_suspended:
            if key not in self.suspended_contexts:
                stats = self.contexts[key]
                stats.is_suspended = True
                stats.suspended_at = now.isoformat()
                stats.shadow_trades_since_suspension = 0
                stats.shadow_wins_since_suspension = 0

        self.suspended_contexts = newly_suspended

    def is_candidate_suspended(self, candidate: CandidateTrade, expiry_minutes: int) -> tuple[bool, Optional[str]]:
        """Check if any context dimension of candidate is currently suspended."""
        cfg: ContextLearningConfig = self.config.adaptive.context_learning
        hour_utc = candidate.t.hour
        vol_pct = candidate.volatility_percentile
        pat_name = candidate.pattern_match.pattern_name
        risk_score = float(candidate.risk_report.score)

        tags = extract_context_tags(
            asset=candidate.asset,
            hour_utc=hour_utc,
            pattern_name=pat_name,
            risk_score=risk_score,
            volatility_pct=vol_pct,
            expiry_minutes=expiry_minutes,
        )

        for dim in cfg.keys:
            val = tags.get(dim)
            if val is not None:
                key = f"{dim}:{val}"
                if key in self.suspended_contexts:
                    return True, f"context_suspended: {key}"

        return False, None

    def filter_candidates(
        self,
        candidates: Sequence[CandidateTrade],
        expiry_minutes: int,
    ) -> tuple[list[CandidateTrade], list[str]]:
        """Filter candidates skipping suspended contexts with explicit reasons."""
        qualifying: list[CandidateTrade] = []
        skip_reasons: list[str] = []

        for cand in candidates:
            suspended, reason = self.is_candidate_suspended(cand, expiry_minutes=expiry_minutes)
            if suspended:
                skip_reasons.append(reason or "context_suspended")
            else:
                qualifying.append(cand)

        return qualifying, skip_reasons

    def to_dict(self) -> dict[str, Any]:
        """Serialize context tracker state."""
        return {
            "contexts": {k: v.to_dict() for k, v in self.contexts.items()},
            "suspended_contexts": list(self.suspended_contexts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: AppConfig) -> ContextTracker:
        """Restore context tracker from serialized state."""
        tracker = cls(config=config)
        ctx_data = data.get("contexts", {})
        for k, v in ctx_data.items():
            tracker.contexts[k] = ContextStats.from_dict(v)
        tracker.suspended_contexts = set(data.get("suspended_contexts", []))
        return tracker
