"""Adaptive risk, profile champion, and context learning package.

Phase 8A: Adaptive core.
"""

from __future__ import annotations

from adaptive.champion import (
    ChampionTracker,
    ProfilePerformance,
    filter_candidates_for_profile,
)
from adaptive.contexts import (
    ContextStats,
    ContextTracker,
    extract_context_tags,
)
from adaptive.decay import (
    EdgeDecayResult,
    evaluate_edge_decay,
    trades_to_detect,
)
from adaptive.monitor import (
    AdaptiveController,
    AdaptiveState,
    TierName,
    update_monitor,
)
from adaptive.postmortem import generate_postmortem_markdown
from adaptive.profiles import (
    PREAPPROVED_PROFILES,
    effective_config,
    validate_profile_dict,
)

__all__ = [
    "AdaptiveController",
    "AdaptiveState",
    "TierName",
    "update_monitor",
    "evaluate_edge_decay",
    "trades_to_detect",
    "EdgeDecayResult",
    "PREAPPROVED_PROFILES",
    "effective_config",
    "validate_profile_dict",
    "ChampionTracker",
    "ProfilePerformance",
    "filter_candidates_for_profile",
    "ContextTracker",
    "ContextStats",
    "extract_context_tags",
    "generate_postmortem_markdown",
]

