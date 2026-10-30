"""Configuration loader with full validation using Pydantic v2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def resolve_project_path(p: str | Path) -> Path:
    """Resolve a relative path against the absolute project root."""
    path = Path(p)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


# ── Nested config models ──────────────────────────────────────────────


class CalendarWindowConfig(BaseModel):
    """Weekly trading window definition."""

    open_day: str = "Sun"
    open_time: str = "17:00"
    close_day: str = "Fri"
    close_time: str = "17:00"
    tz: str = "America/New_York"


class CalendarConfig(BaseModel):
    """Market calendar configuration."""

    type: Literal["always_open", "weekly_windows"] = "always_open"
    windows: list[CalendarWindowConfig] = Field(default_factory=list)
    daily_boundary_tz: str = "UTC"
    daily_boundary_hour: int = Field(ge=0, le=23, default=0)


class DataConfig(BaseModel):
    """Market data parameters."""

    daily_boundary_tz: str = "UTC"
    min_bar_coverage: float = Field(ge=0.5, le=1.0, default=0.98)
    calendars: dict[str, CalendarConfig] = Field(default_factory=dict)
    assets: dict[str, str] = Field(default_factory=dict)


class TradingConfig(BaseModel):
    """Trading timing and asset selection parameters."""

    expiry_minutes: int = Field(ge=1)
    allowed_expiries: list[int]
    check_interval_minutes: int = Field(gt=0)
    asset_reselect_hours: int = Field(gt=0)
    top_assets: int = Field(ge=1)
    max_measured_slots: int = Field(ge=0)
    min_trades_for_stats: int = Field(ge=1)
    tie_policy: Literal["refund", "loss"] = Field(default="loss")

    @model_validator(mode="after")
    def _validate_expiry(self) -> "TradingConfig":
        valid = {15, 30, 60}
        for e in self.allowed_expiries:
            if e not in valid:
                raise ValueError(
                    f"allowed_expiries contains {e}; must be subset of {valid}"
                )
        if self.expiry_minutes not in self.allowed_expiries:
            raise ValueError(
                f"expiry_minutes={self.expiry_minutes} not in "
                f"allowed_expiries={self.allowed_expiries}"
            )
        if self.max_measured_slots > self.top_assets:
            raise ValueError(
                f"max_measured_slots={self.max_measured_slots} must be "
                f"<= top_assets={self.top_assets}"
            )
        return self


class StakeConfig(BaseModel):
    """Stake sizing parameters."""

    min_pct: float = Field(gt=0, le=100)
    max_pct: float = Field(gt=0, le=100)
    total_cap_pct: float = Field(gt=0, le=100)
    daily_loss_limit_pct: Optional[float] = Field(default=None)
    min_amount: float = Field(gt=0, default=1.0)
    max_amount: Optional[float] = Field(default=1000.0)
    round_to: float = Field(gt=0, default=1.0)
    stake_scale: float = Field(gt=0, le=1.0, default=1.0)

    @model_validator(mode="after")
    def _validate_ordering(self) -> "StakeConfig":
        if self.min_pct > self.max_pct:
            raise ValueError(
                f"min_pct={self.min_pct} must be <= max_pct={self.max_pct}"
            )
        if self.max_pct > self.total_cap_pct:
            raise ValueError(
                f"max_pct={self.max_pct} must be <= total_cap_pct={self.total_cap_pct}"
            )
        if self.max_amount is not None and self.max_amount < self.min_amount:
            raise ValueError(
                f"max_amount={self.max_amount} must be >= min_amount={self.min_amount}"
            )
        if self.daily_loss_limit_pct is not None:
            if self.daily_loss_limit_pct <= 0 or self.daily_loss_limit_pct > 100:
                raise ValueError(
                    f"daily_loss_limit_pct={self.daily_loss_limit_pct} "
                    "must be >0 and <=100 when set"
                )
        return self


class TimeframeTrendConfig(BaseModel):
    """Trend parameters for a single timeframe (daily, h1, m15)."""

    ema_fast: int = Field(ge=2, default=20)
    ema_slow: int = Field(ge=2, default=50)
    adx_period: int = Field(ge=2, default=14)
    adx_min: int = Field(ge=0, le=100, default=20)
    warmup_multiplier: int = Field(ge=1, default=3)

    @model_validator(mode="after")
    def _validate_ema_ordering(self) -> "TimeframeTrendConfig":
        if self.ema_fast >= self.ema_slow:
            raise ValueError(
                f"ema_fast={self.ema_fast} must be < ema_slow={self.ema_slow}"
            )
        return self


class TrendConfig(BaseModel):
    """Per-timeframe trend indicator parameters."""

    daily: TimeframeTrendConfig = Field(default_factory=TimeframeTrendConfig)
    h1: TimeframeTrendConfig = Field(default_factory=TimeframeTrendConfig)
    m15: TimeframeTrendConfig = Field(default_factory=TimeframeTrendConfig)


class ReversalBypassConfig(BaseModel):
    """Reversal bypass gate parameters."""

    min_similarity: float = Field(ge=0, le=100)
    max_risk_score: float = Field(ge=0, le=100)
    require_sr_level: bool
    sr_level_tolerance_atr: Optional[float] = Field(default=None)
    require_confirmation_candle: bool
    max_volatility_percentile: float = Field(ge=0, le=100)


class LevelsConfig(BaseModel):
    """Support and resistance level detection parameters."""

    swing_left_bars: int = Field(default=5, ge=1)
    swing_right_bars: int = Field(default=2, ge=1)
    cluster_tolerance_atr: float = Field(default=0.5, gt=0)
    at_level_tolerance_atr: float = Field(default=0.5, gt=0)


class GateConfig(BaseModel):
    """Trend/reversal gate parameters."""

    strict_trend_gate: bool
    reversal_mode: Literal["off", "sr_bypass"]
    trend: TrendConfig
    reversal_bypass: ReversalBypassConfig

    @field_validator("reversal_mode", mode="before")
    @classmethod
    def _coerce_reversal_mode(cls, v: Any) -> Any:
        # YAML parses unquoted `off` as boolean False
        if v is False or v == "off" or v == "False":
            return "off"
        return v


class PatternsConfig(BaseModel):
    """Pattern matching parameters."""

    templates_dir: str = Field(default="patterns/templates")
    timeframes: list[str] = Field(default=["5m", "15m"])
    min_similarity: float = Field(ge=0, le=100, default=60.0)
    template_points: int = Field(ge=8, default=64)
    window_lengths: list[int] = Field(default=[30, 45, 60, 90])
    dtw_shortlist: int = Field(ge=1, default=3)
    dtw_weight: float = Field(ge=0, le=1, default=0.3)
    dtw_sakoe_chiba_radius: int = Field(ge=1, default=6)
    allow_trade_without_pattern: bool = Field(default=False)

    @model_validator(mode="after")
    def _validate_windows(self) -> "PatternsConfig":
        for w in self.window_lengths:
            if w <= 0:
                raise ValueError(
                    f"window_lengths contains non-positive value {w}"
                )
        return self


class RiskWeightsConfig(BaseModel):
    """Risk scoring weight factors — must sum to 1.0."""

    volatility: float = Field(ge=0, le=1)
    pattern: float = Field(ge=0, le=1)
    win_rate: float = Field(ge=0, le=1)
    session: float = Field(ge=0, le=1)
    sr_distance: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_sum(self) -> "RiskWeightsConfig":
        total = (
            self.volatility
            + self.pattern
            + self.win_rate
            + self.session
            + self.sr_distance
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Risk weights must sum to 1.0 (±0.001), got {total}"
            )
        return self


class RiskDefaultsConfig(BaseModel):
    """Defaults for missing inputs in risk calculation."""

    pattern_missing: float = Field(ge=0, le=100, default=100.0)
    untested_asset: float = Field(ge=0, le=100, default=100.0)
    session_missing: float = Field(ge=0, le=100, default=50.0)
    no_sr_level: float = Field(ge=0, le=100, default=10.0)
    sr_unknown: float = Field(ge=0, le=100, default=50.0)
    volatility_missing: float = Field(ge=0, le=100, default=50.0)


class RiskConfig(BaseModel):
    """Risk scoring parameters."""

    weights: RiskWeightsConfig
    volatility_lookback_days: int = Field(ge=1, default=30)
    atr_period: int = Field(ge=2, default=14)
    session_min_samples: int = Field(ge=1, default=20)
    stats_modes: list[str] = Field(default_factory=lambda: ["paper", "demo", "live"])
    defaults: RiskDefaultsConfig = Field(default_factory=RiskDefaultsConfig)
    sr_distance_min_atr: float = Field(gt=0, default=0.5)
    sr_distance_max_atr: float = Field(gt=0, default=3.0)
    sr_distance_near_score: float = Field(ge=0, le=100, default=90.0)
    sr_distance_far_score: float = Field(ge=0, le=100, default=10.0)
    max_risk_cap: Optional[float] = Field(default=None)

    @model_validator(mode="after")
    def _validate_sr_distance_order(self) -> "RiskConfig":
        if self.sr_distance_min_atr >= self.sr_distance_max_atr:
            raise ValueError(
                f"sr_distance_min_atr ({self.sr_distance_min_atr}) must be < "
                f"sr_distance_max_atr ({self.sr_distance_max_atr})"
            )
        return self


class PaperGateConfig(BaseModel):
    """Paper-trading graduation gate."""

    min_paper_trades: int = Field(ge=1, default=200)
    min_edge_margin: float = Field(gt=0, lt=1, default=0.02)
    require_ci_lower_bound: bool = Field(default=False)


class SelectionConfig(BaseModel):
    """Asset selection and correlation parameters."""

    correlation_lookback: int = Field(ge=5, default=50)
    correlation_min: float = Field(ge=0, le=1, default=0.7)
    rolling_window: int = Field(ge=10, default=100)
    shrinkage_k: float = Field(ge=0, default=50.0)
    ranking_modes: list[str] = Field(default_factory=lambda: ["paper", "demo", "live"])


class SafetyConfig(BaseModel):
    """Safety and circuit-breaker parameters."""

    model_config = ConfigDict(extra="forbid")

    max_orders_per_window: int = Field(ge=1, default=6)
    stale_data_multiplier: float = Field(gt=0)
    max_consecutive_exec_errors: int = Field(ge=1)
    kill_file: str
    balance_tolerance: float = Field(ge=0, default=1.0)
    rounding_tolerance: float = Field(ge=0, default=0.001)
    stake_verification_tolerance: float = Field(ge=0, default=0.01)
    rate_limit_window_minutes: int = Field(ge=1, default=10)
    max_alert_failures: int = Field(ge=1, default=5)


class LLMConfig(BaseModel):
    """LLM integration parameters."""

    provider: Literal["anthropic", "gemini"]
    model: str
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    max_decision_age_seconds: float = Field(gt=0)
    max_price_move_atr: float = Field(gt=0, default=0.25)
    evidence_float_tolerance: float = Field(ge=0, default=0.01)
    max_calls_per_hour: int = Field(ge=1, default=60)
    default_atr: Optional[float] = Field(default=None)


class AlertsConfig(BaseModel):
    """Alert channel parameters."""

    telegram_enabled: bool


class BacktestConfig(BaseModel):
    """Backtesting parameters."""

    default_payout: float = Field(gt=0, lt=1, default=0.85)
    train_fraction: float = Field(gt=0, lt=1, default=0.7)
    target_win_rate: float = Field(gt=0, le=1.0, default=0.80)
    assets: list[str] = Field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
    starting_balance: float = Field(gt=0, default=10000.0)
    decision_policy: Literal["lowest_risk", "highest_similarity", "llm"] = "lowest_risk"
    stake_policy: Literal["max", "mid", "min"] = "max"
    entry_delay_bars: int = Field(ge=0, default=1)
    slippage_atr: float = Field(ge=0, default=0.02)
    tie_policy: Optional[Literal["refund", "loss"]] = Field(default=None)
    ignore_paper_gate: bool = Field(default=True)
    max_llm_steps: int = Field(ge=1, default=200)


# ── Adaptive layer config ─────────────────────────────────────────────


class AdaptiveTierCaution(BaseModel):
    drawdown_pct: float = Field(gt=0, le=100, default=15.0)
    consecutive_losses: int = Field(ge=1, default=4)
    stake_scale: float = Field(gt=0, le=1.0, default=0.5)
    min_similarity_add: float = Field(ge=0, default=5.0)
    max_risk_cap: float = Field(gt=0, le=100, default=50.0)


class AdaptiveTierDefensive(BaseModel):
    drawdown_pct: float = Field(gt=0, le=100, default=25.0)
    consecutive_losses: Optional[int] = Field(default=None)
    stake_scale: float = Field(gt=0, le=1.0, default=0.35)
    reversal_mode: str = Field(default="off")
    min_similarity_add: float = Field(ge=0, default=10.0)
    max_risk_cap: float = Field(gt=0, le=100, default=40.0)

    @field_validator("reversal_mode", mode="before")
    @classmethod
    def _coerce_reversal_mode(cls, v: Any) -> str:
        if isinstance(v, bool):
            return "off" if not v else "on"
        return str(v)


class AdaptiveTierPaused(BaseModel):
    drawdown_pct: float = Field(gt=0, le=100, default=40.0)
    real_trading: bool = Field(default=False)


class AdaptiveTiersConfig(BaseModel):
    caution: AdaptiveTierCaution = Field(default_factory=AdaptiveTierCaution)
    defensive: AdaptiveTierDefensive = Field(default_factory=AdaptiveTierDefensive)
    paused: AdaptiveTierPaused = Field(default_factory=AdaptiveTierPaused)

    @model_validator(mode="after")
    def _validate_tier_drawdown_ordering(self) -> "AdaptiveTiersConfig":
        if not (self.caution.drawdown_pct < self.defensive.drawdown_pct < self.paused.drawdown_pct):
            raise ValueError(
                f"Adaptive tier drawdown thresholds must be strictly increasing: "
                f"caution ({self.caution.drawdown_pct}) < defensive ({self.defensive.drawdown_pct}) < paused ({self.paused.drawdown_pct})"
            )
        return self


class RequalificationConfig(BaseModel):
    min_shadow_trades: int = Field(ge=1, default=30)
    win_rate_margin: float = Field(ge=0, default=0.0)
    rampup_trades: int = Field(ge=1, default=20)
    rampup_stake_scale: float = Field(gt=0, le=1.0, default=0.5)


class ProfileSwitchConfig(BaseModel):
    min_shadow_trades: int = Field(ge=1, default=50)
    require_ci_lower_above_break_even: bool = Field(default=True)
    min_ev_advantage: float = Field(ge=0, default=0.02)
    cooldown_days: float = Field(ge=0, default=3.0)


class EdgeDecayConfig(BaseModel):
    """Edge decay detection parameters."""

    window_trades_caution: int = Field(ge=1, default=60)
    z_caution: float = Field(gt=0, default=1.0)
    window_trades: int = Field(ge=1, default=100)
    min_trades: int = Field(ge=1, default=30)
    z: float = Field(gt=0, default=1.96)
    tier_on_decay: Literal["caution", "defensive", "paused"] = "caution"
    tier_on_persistent: Literal["caution", "defensive", "paused"] = "defensive"
    persist_windows: int = Field(ge=1, default=2)

    @field_validator("tier_on_decay", "tier_on_persistent", mode="before")
    @classmethod
    def _normalize_tier(cls, v: Any) -> str:
        return str(v).lower()


class ContextLearningConfig(BaseModel):
    min_trades: int = Field(ge=1, default=30)
    z: float = Field(gt=0, default=2.58)
    cooldown_days: float = Field(ge=0, default=3.0)
    max_suspended_fraction: float = Field(gt=0, le=1.0, default=0.3)
    keys: list[str] = Field(
        default_factory=lambda: [
            "asset",
            "hour_bucket",
            "pattern",
            "risk_bucket",
            "volatility_bucket",
            "expiry",
        ]
    )


PROFILE_WHITELIST_KEYS: set[str] = {
    "reversal_mode",
    "patterns.min_similarity",
    "min_similarity",
    "max_risk_cap",
    "stake_scale",
}


class AdaptiveConfig(BaseModel):
    enabled: bool = Field(default=True)
    reference: Literal["peak", "session_start", "checkpoint"] = "peak"
    tiers: AdaptiveTiersConfig = Field(default_factory=AdaptiveTiersConfig)
    hysteresis_recovery_fraction: float = Field(gt=0, lt=1.0, default=0.5)
    requalification: RequalificationConfig = Field(default_factory=RequalificationConfig)
    profiles: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            "standard": {},
            "conservative": {
                "patterns.min_similarity": 85.0,
                "max_risk_cap": 45,
                "stake_scale": 0.8,
            },
        }
    )
    profile_switch: ProfileSwitchConfig = Field(default_factory=ProfileSwitchConfig)
    context_learning: ContextLearningConfig = Field(default_factory=ContextLearningConfig)
    edge_decay: EdgeDecayConfig = Field(default_factory=EdgeDecayConfig)

    @field_validator("profiles")
    @classmethod
    def _validate_profile_keys(cls, v: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        for pname, pdict in v.items():
            if not isinstance(pdict, dict):
                raise ValueError(f"Profile '{pname}' must be a dictionary")
            for k in list(pdict.keys()):
                if k not in PROFILE_WHITELIST_KEYS:
                    raise ValueError(
                        f"Profile '{pname}' contains forbidden key '{k}'. "
                        f"Allowed whitelist: {sorted(list(PROFILE_WHITELIST_KEYS))}"
                    )
                if k == "reversal_mode" and isinstance(pdict[k], bool):
                    pdict[k] = "off" if not pdict[k] else "on"
        return v


# ── Root config ───────────────────────────────────────────────────────


class AppConfig(BaseModel):
    """Root application configuration."""

    mode: Literal["record", "paper", "demo", "live"]
    live_requires_env: str

    data: DataConfig = Field(default_factory=DataConfig)
    trading: TradingConfig
    stake: StakeConfig
    gate: GateConfig
    levels: LevelsConfig = Field(default_factory=LevelsConfig)
    patterns: PatternsConfig
    risk: RiskConfig
    paper_gate: PaperGateConfig
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    safety: SafetyConfig
    llm: LLMConfig
    alerts: AlertsConfig
    backtest: BacktestConfig
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)

    @model_validator(mode="after")
    def _sync_tie_policy(self) -> "AppConfig":
        if hasattr(self, "backtest") and hasattr(self, "trading"):
            if self.backtest.tie_policy is None:
                self.backtest.tie_policy = self.trading.tie_policy
            else:
                self.trading.tie_policy = self.backtest.tie_policy
        return self


# ── Loader ────────────────────────────────────────────────────────────


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Read *path*, validate with Pydantic, and return an ``AppConfig``.

    Raises ``ValueError`` when ``mode`` is ``"live"`` but the environment
    variable named by ``live_requires_env`` is not set to ``"YES"``.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = AppConfig(**raw)

    if cfg.mode == "live":
        env_name = cfg.live_requires_env
        if os.environ.get(env_name) != "YES":
            raise ValueError(
                f"mode is 'live' but env var {env_name} is not 'YES'"
            )

    return cfg
