"""Candle builder — aggregates raw ticks into 1m candles and resamples to higher timeframes.

Rules:
- Closed candles only. Any partial or in-progress candle is excluded from signals.
- Supports 1m, 5m, 15m, 1h, and 1d timeframes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd


TIMEFRAME_DELTAS = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}

TIMEFRAME_RULES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "1d": "1D",
}


def ticks_to_1m(
    ticks: pd.DataFrame,
    current_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """Aggregate raw ticks into 1-minute OHLCV candles (closed only).

    Expected ticks DataFrame columns:
      - 'ts': datetime or int/float timestamp (seconds or ms)
      - 'price': float
      - 'volume': optional float (defaults to 0.0)

    Returns DataFrame with columns:
      ['ts', 'open', 'high', 'low', 'close', 'volume']
    sorted ascending by 'ts'.
    """
    if ticks.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    df = ticks.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    elif df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(timezone.utc)
    else:
        df["ts"] = df["ts"].dt.tz_convert(timezone.utc)

    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = df["volume"].fillna(0.0)

    df = df.sort_values("ts")

    max_tick_time = df["ts"].max()
    effective_now = current_time
    if effective_now is not None:
        if effective_now.tzinfo is None:
            effective_now = effective_now.replace(tzinfo=timezone.utc)
        else:
            effective_now = effective_now.astimezone(timezone.utc)

    # Floor to 1-minute bucket
    df["bucket"] = df["ts"].dt.floor("1min")

    agg_dict = {
        "price": ["first", "max", "min", "last"],
        "volume": "sum",
    }
    grouped = df.groupby("bucket").agg(agg_dict)
    grouped.columns = ["open", "high", "low", "close", "volume"]
    grouped = grouped.reset_index().rename(columns={"bucket": "ts"})

    # Filter out partial (open) candles:
    # A 1m candle starting at T covers [T, T + 1m).
    # It is closed only if current_time >= T + 1m (or max_tick_time >= T + 1m).
    one_min = timedelta(minutes=1)
    if effective_now is not None:
        grouped = grouped[grouped["ts"] + one_min <= effective_now]
    else:
        # Without explicit current_time, the bucket containing the latest tick
        # is only considered closed if a tick exists at or past T + 1m.
        # Since max_tick_time is in the dataset, T + 1m <= max_tick_time
        grouped = grouped[grouped["ts"] + one_min <= max_tick_time]

    return grouped.sort_values("ts").reset_index(drop=True)


def resample_candles(
    df_1m: pd.DataFrame,
    target_timeframe: str,
    current_time: Optional[datetime] = None,
    daily_boundary_tz: str = "UTC",
) -> pd.DataFrame:
    """Resample 1-minute closed candles into higher timeframe closed candles.

    Supported timeframes: '1m', '5m', '15m', '1h', '1d'.

    Rules:
      - Closed candles only. Partial or in-progress intervals are strictly excluded.
      - Daily boundary alignment: If target_timeframe is '1d', boundaries are
        aligned according to daily_boundary_tz (e.g. 'UTC' or market local).
      - Gap handling: Periods with zero tick/candle data are not filled with
        phantom zero bars; they are dropped. For buckets with internal missing
        minutes, open/high/low/close are derived from all available bars.
    """
    if target_timeframe not in TIMEFRAME_RULES:
        raise ValueError(
            f"Unsupported timeframe: {target_timeframe}. "
            f"Must be one of {list(TIMEFRAME_RULES.keys())}"
        )

    if df_1m.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    df = df_1m.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    elif df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(timezone.utc)
    else:
        df["ts"] = df["ts"].dt.tz_convert(timezone.utc)

    if target_timeframe == "1m":
        return df.sort_values("ts").reset_index(drop=True)

    rule = TIMEFRAME_RULES[target_timeframe]
    delta = TIMEFRAME_DELTAS[target_timeframe]

    # Convert to daily_boundary_tz if daily timeframe requested
    if target_timeframe == "1d" and daily_boundary_tz != "UTC":
        df["ts"] = df["ts"].dt.tz_convert(daily_boundary_tz)

    df = df.set_index("ts").sort_index()

    resampled = df.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    # Gap handling: Drop empty buckets (periods with zero market data)
    resampled = resampled.dropna(subset=["open", "close"]).reset_index()

    if target_timeframe == "1d" and daily_boundary_tz != "UTC":
        resampled["ts"] = resampled["ts"].dt.tz_convert(timezone.utc)
        df.index = df.index.tz_convert(timezone.utc)

    # Determine end of available 1m data
    data_end_time = df.index.max() + timedelta(minutes=1)

    effective_now = current_time
    if effective_now is not None:
        if effective_now.tzinfo is None:
            effective_now = effective_now.replace(tzinfo=timezone.utc)
        else:
            effective_now = effective_now.astimezone(timezone.utc)
        cutoff = min(effective_now, data_end_time)
    else:
        cutoff = data_end_time

    # A bucket starting at T and spanning delta is closed if and only if T + delta <= cutoff
    resampled = resampled[resampled["ts"] + delta <= cutoff]

    return resampled.sort_values("ts").reset_index(drop=True)


def detect_gaps(
    df: pd.DataFrame,
    timeframe: str,
) -> list[tuple[datetime, datetime]]:
    """Scan candle DataFrame for gaps larger than the expected timeframe interval.

    Returns list of (gap_start, gap_end) timestamps.
    """
    if len(df) < 2 or timeframe not in TIMEFRAME_DELTAS:
        return []

    delta = TIMEFRAME_DELTAS[timeframe]
    ts_series = pd.to_datetime(df["ts"], utc=True).sort_values()
    diffs = ts_series.diff()

    gaps = []
    for prev_ts, curr_ts, diff in zip(ts_series.iloc[:-1], ts_series.iloc[1:], diffs.iloc[1:]):
        if diff > delta:
            gaps.append((prev_ts.to_pydatetime(), curr_ts.to_pydatetime()))

    return gaps

    return resampled.sort_values("ts").reset_index(drop=True)


class CandleBuilder:
    """Stateful candle builder tracking ticks and generating closed candles per asset."""

    def __init__(self):
        self._ticks: dict[str, list[dict]] = {}
        self._candles_1m: dict[str, list[dict]] = {}

    def add_tick(
        self,
        asset_id: str,
        ts: datetime,
        price: float,
        volume: float = 0.0,
    ) -> list[dict]:
        """Add a tick for asset_id.

        If adding this tick closes one or more 1m candles, return the newly
        closed 1m candles as a list of dicts.
        """
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        if asset_id not in self._ticks:
            self._ticks[asset_id] = []
            self._candles_1m[asset_id] = []

        self._ticks[asset_id].append({"ts": ts, "price": price, "volume": volume})

        # Check if we can form new closed 1m candle(s)
        df_ticks = pd.DataFrame(self._ticks[asset_id])
        df_1m = ticks_to_1m(df_ticks, current_time=ts)

        existing_count = len(self._candles_1m[asset_id])
        if len(df_1m) > existing_count:
            new_closed = df_1m.iloc[existing_count:].to_dict(orient="records")
            self._candles_1m[asset_id].extend(new_closed)
            # Prune old ticks that are already incorporated into closed candles
            latest_closed_ts = df_1m.iloc[-1]["ts"]
            self._ticks[asset_id] = [
                t for t in self._ticks[asset_id] if t["ts"] >= latest_closed_ts + timedelta(minutes=1)
            ]
            return new_closed

        return []

    def get_candles(
        self,
        asset_id: str,
        timeframe: str,
        n: Optional[int] = None,
        current_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Return the latest closed candles for asset_id and timeframe."""
        if asset_id not in self._candles_1m or not self._candles_1m[asset_id]:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        df_1m = pd.DataFrame(self._candles_1m[asset_id])
        df_resampled = resample_candles(df_1m, timeframe, current_time=current_time)

        if n is not None and n > 0:
            return df_resampled.tail(n).reset_index(drop=True)
        return df_resampled
