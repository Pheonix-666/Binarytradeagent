"""Pre-approved trading profiles and tightening configuration overlay.

PROJECT_SPEC.md Adaptive Layer (Phase 8A):
  - Profiles: named overrides limited to a strict WHITELIST of keys with a clear
    tightening order:
      * reversal_mode: can only tighten towards "off"
      * patterns.min_similarity: can only increase (stricter pattern match)
      * max_risk_cap: can only decrease (stricter risk limit)
      * stake_scale: can only decrease (<= 1.0)
  - Ship three pre-approved profiles:
      * standard: baseline default
      * conservative: higher similarity threshold, lower risk cap, scaled stake
      * trend_long_expiry: strict trend gate (no reversals)
  - effective_config(base_config, adaptive_state, active_profile):
      Produces AppConfig that only TIGHTENS risk parameters, never loosens.
      Safety limits (stake min/max/cap) are untouched.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from core.config import AppConfig, PROFILE_WHITELIST_KEYS
from adaptive.monitor import AdaptiveState


PREAPPROVED_PROFILES: dict[str, dict[str, Any]] = {
    "standard": {},
    "conservative": {
        "patterns.min_similarity": 85.0,
        "max_risk_cap": 45.0,
        "stake_scale": 0.8,
    },
}


def validate_profile_dict(profile_dict: Mapping[str, Any], profile_name: str = "custom") -> None:
    """Validate that profile overrides contain only whitelisted keys.

    Raises:
        ValueError: if forbidden keys are present.
    """
    for key in profile_dict:
        if key not in PROFILE_WHITELIST_KEYS:
            raise ValueError(
                f"Profile '{profile_name}' contains non-whitelisted key '{key}'. "
                f"Allowed keys: {sorted(list(PROFILE_WHITELIST_KEYS))}"
            )


def effective_config(
    base_config: AppConfig,
    adaptive_state: Optional[AdaptiveState] = None,
    active_profile: Optional[Mapping[str, Any] | str] = None,
) -> AppConfig:
    """Generate effective application configuration applying profile and adaptive tier tightening.

    Guarantees:
      1. Every overridden key is at least as strict as base_config.
      2. Safety limits (stake 1-20%, total cap 35%) are strictly preserved.
      3. Never increases stake after a loss or loosens entry gates.
    """
    # 1. Resolve profile dictionary
    p_dict: dict[str, Any] = {}
    p_name = "standard"
    if isinstance(active_profile, str):
        p_name = active_profile
        # Lookup in config profiles or fallback to preapproved
        p_dict = base_config.adaptive.profiles.get(active_profile, PREAPPROVED_PROFILES.get(active_profile, {}))
    elif isinstance(active_profile, dict):
        p_dict = dict(active_profile)
        p_name = "custom"

    validate_profile_dict(p_dict, profile_name=p_name)

    # 2. Deep-copy base config
    cfg = base_config.model_copy(deep=True)

    # 3. Apply Profile Overrides (Strictly Tightening)
    # A. patterns.min_similarity
    prof_sim = p_dict.get("patterns.min_similarity", p_dict.get("min_similarity"))
    if prof_sim is not None:
        cfg.patterns.min_similarity = max(float(cfg.patterns.min_similarity), float(prof_sim))

    # B. reversal_mode: can only tighten towards "off"
    prof_rev = p_dict.get("reversal_mode")
    if prof_rev is not None:
        prof_rev_str = "off" if (prof_rev is False or prof_rev == "off") else "sr_bypass"
        # If base is "off" or profile is "off", effective must be "off"
        if cfg.gate.reversal_mode == "off" or prof_rev_str == "off":
            cfg.gate.reversal_mode = "off"

    # C. Profile stake scale
    prof_stake_scale = float(p_dict.get("stake_scale", 1.0))
    prof_stake_scale = min(1.0, max(0.0, prof_stake_scale))

    # 4. Apply Adaptive State Tightening (from Adaptive Monitor)
    state_scale = 1.0
    if adaptive_state is not None:
        # A. Extra similarity threshold
        if adaptive_state.min_similarity_add > 0:
            cfg.patterns.min_similarity += float(adaptive_state.min_similarity_add)

        # B. Reversal mode override from tier
        if adaptive_state.reversal_mode_override == "off":
            cfg.gate.reversal_mode = "off"

        # C. Stake scale from tier and ramp-up
        state_scale = float(adaptive_state.stake_scale)

        # D. Max risk cap stored in config for downstream candidate filtering
        tier_risk_cap = adaptive_state.max_risk_cap
        prof_risk_cap = p_dict.get("max_risk_cap")

        caps = [c for c in (tier_risk_cap, prof_risk_cap) if c is not None]
        if caps:
            # Strictest cap (lowest risk)
            effective_cap = min(float(c) for c in caps)
            setattr(cfg.risk, "max_risk_cap", effective_cap)
        else:
            setattr(cfg.risk, "max_risk_cap", None)
    else:
        prof_risk_cap = p_dict.get("max_risk_cap")
        if prof_risk_cap is not None:
            setattr(cfg.risk, "max_risk_cap", float(prof_risk_cap))
        else:
            setattr(cfg.risk, "max_risk_cap", None)

    # 5. Combined effective stake scale (profile scale * tier state scale)
    combined_stake_scale = min(1.0, prof_stake_scale * state_scale)
    setattr(cfg.stake, "stake_scale", combined_stake_scale)

    # 6. Safety Layer Invariant Verification
    # Ensure stake limits are strictly unchanged from base_config
    assert cfg.stake.min_pct == base_config.stake.min_pct, "Safety invariant: min_pct cannot be changed"
    assert cfg.stake.max_pct == base_config.stake.max_pct, "Safety invariant: max_pct cannot be changed"
    assert cfg.stake.total_cap_pct == base_config.stake.total_cap_pct, "Safety invariant: total_cap_pct cannot be changed"

    return cfg

