"""Configuration loader with full validation using Pydantic v2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Nested config models ──────────────────────────────────────────────


class TradingConfig(BaseModel):
    """Trading timing and asset selection parameters."""

    expiry_minutes: int = Field(ge=1)
    allowed_expiries: list[int]
    check_interval_minutes: int = Field(gt=0)
    asset_reselect_hours: int = Field(gt=0)
    top_assets: int = Field(ge=1)
    max_measured_slots: int = Field(ge=0)
    min_trades_for_stats: int = Field(ge=1)

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
        if self.daily_loss_limit_pct is not None:
            if self.daily_loss_limit_pct <= 0 or self.daily_loss_limit_pct > 100:
                raise ValueError(
                    f"daily_loss_limit_pct={self.daily_loss_limit_pct} "
                    "must be >0 and <=100 when set"
                )
        return self


class TrendConfig(BaseModel):
    """Trend indicator parameters."""

    ema_fast: int = Field(ge=2)
    ema_slow: int = Field(ge=2)
    adx_period: int = Field(ge=2)
    adx_min: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _validate_ema_ordering(self) -> "TrendConfig":
        if self.ema_fast >= self.ema_slow:
            raise ValueError(
                f"ema_fast={self.ema_fast} must be < ema_slow={self.ema_slow}"
            )
        return self


class ReversalBypassConfig(BaseModel):
    """Reversal bypass gate parameters."""

    min_similarity: float = Field(ge=0, le=100)
    max_risk_score: float = Field(ge=0, le=100)
    require_sr_level: bool
    require_confirmation_candle: bool
    max_volatility_percentile: float = Field(ge=0, le=100)


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

    min_similarity: float = Field(ge=0, le=100)
    template_points: int = Field(ge=8)
    window_lengths: list[int]
    dtw_shortlist: int = Field(ge=1)
    allow_trade_without_pattern: bool

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


class RiskConfig(BaseModel):
    """Risk scoring parameters."""

    weights: RiskWeightsConfig
    volatility_lookback_days: int = Field(ge=1)
    atr_period: int = Field(ge=2)


class PaperGateConfig(BaseModel):
    """Paper-trading graduation gate."""

    min_paper_trades: int = Field(ge=1)
    min_edge_margin: float = Field(gt=0, lt=1)


class SafetyConfig(BaseModel):
    """Safety and circuit-breaker parameters."""

    max_orders_per_10min: int = Field(ge=1)
    stale_data_multiplier: float = Field(gt=0)
    max_consecutive_exec_errors: int = Field(ge=1)
    kill_file: str


class LLMConfig(BaseModel):
    """LLM integration parameters."""

    provider: Literal["anthropic", "gemini"]
    model: str
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    max_decision_age_seconds: float = Field(gt=0)


class AlertsConfig(BaseModel):
    """Alert channel parameters."""

    telegram_enabled: bool


class BacktestConfig(BaseModel):
    """Backtesting parameters."""

    default_payout: float = Field(gt=0, lt=1)
    train_fraction: float = Field(gt=0, lt=1)


# ── Root config ───────────────────────────────────────────────────────


class AppConfig(BaseModel):
    """Root application configuration."""

    mode: Literal["record", "paper", "demo", "live"]
    live_requires_env: str

    trading: TradingConfig
    stake: StakeConfig
    gate: GateConfig
    patterns: PatternsConfig
    risk: RiskConfig
    paper_gate: PaperGateConfig
    safety: SafetyConfig
    llm: LLMConfig
    alerts: AlertsConfig
    backtest: BacktestConfig


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
