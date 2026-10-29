"""Candle builder — aggregates raw ticks into 1m candles and resamples to higher timeframes.

Rules:
- Closed candles only. Any partial or in-progress candle is excluded from signals.
- Supports 1m, 5m, 15m, 1h, and 1d timeframes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from core.config import CalendarConfig
from data.calendar import (
    expected_bars as get_expected_bars,
    get_daily_bucket_end,
    get_daily_bucket_start,
    is_market_open,
)


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
        return pd.DataFrame(columns=["ts", "ts_close", "open", "high", "low", "close", "volume"])

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
    # A 1m candle starting at T covers [T, T + 1m) and closes at T + 1m.
    # It is closed only if current_time >= T + 1m (or max_tick_time >= T + 1m).
    one_min = timedelta(minutes=1)
    grouped["ts_close"] = grouped["ts"] + one_min

    if effective_now is not None:
        grouped = grouped[grouped["ts_close"] <= effective_now]
    else:
        grouped = grouped[grouped["ts_close"] <= max_tick_time]

    cols = ["ts", "ts_close", "open", "high", "low", "close", "volume"]
    return grouped[cols].sort_values("ts").reset_index(drop=True)


def resample_candles(
    df_1m: pd.DataFrame,
    target_timeframe: str,
    current_time: Optional[datetime] = None,
    daily_boundary_tz: str = "UTC",
    calendar: Optional[CalendarConfig] = None,
    min_bar_coverage: float = 0.98,
) -> pd.DataFrame:
    """Resample 1-minute closed candles into higher timeframe closed candles.

    Supported timeframes: '1m', '5m', '15m', '1h', '1d'.

    Rules:
      - Closed candles only. Partial or in-progress intervals are strictly excluded.
      - ts: open timestamp of the bar.
      - ts_close: close timestamp (bucket end).
      - Calendar-aware coverage: expected_bars(bucket, calendar) determines the number
        of open 1m bars in that bucket.
      - A bar is emitted only if expected > 0 AND (present / expected) >= min_bar_coverage.
      - Buckets where the market is closed (expected == 0) are not emitted and are not gaps.
      - Daily boundary alignment: If target_timeframe is '1d', boundaries are aligned to
        the calendar's daily_boundary_tz and daily_boundary_hour (e.g. 17:00 New York).
    """
    if target_timeframe not in TIMEFRAME_RULES:
        raise ValueError(
            f"Unsupported timeframe: {target_timeframe}. "
            f"Must be one of {list(TIMEFRAME_RULES.keys())}"
        )

    if df_1m.empty:
        return pd.DataFrame(columns=["ts", "ts_close", "open", "high", "low", "close", "volume"])

    df = df_1m.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    elif df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(timezone.utc)
    else:
        df["ts"] = df["ts"].dt.tz_convert(timezone.utc)

    if target_timeframe == "1m":
        if "ts_close" not in df.columns:
            df["ts_close"] = df["ts"] + timedelta(minutes=1)
        cols = ["ts", "ts_close", "open", "high", "low", "close", "volume"]
        extra = [c for c in df.columns if c not in cols]
        return df[cols + extra].sort_values("ts").reset_index(drop=True)

    delta = TIMEFRAME_DELTAS[target_timeframe]

    # Assign bucket timestamps
    if target_timeframe == "1d":
        if calendar is not None and (calendar.daily_boundary_hour != 0 or calendar.daily_boundary_tz != "UTC"):
            df["bucket"] = df["ts"].apply(lambda t: get_daily_bucket_start(t, calendar))
        elif daily_boundary_tz != "UTC":
            # Backward-compatibility for daily_boundary_tz argument without calendar
            df["ts_local"] = df["ts"].dt.tz_convert(daily_boundary_tz)
            df["bucket"] = df["ts_local"].dt.floor("1D").dt.tz_convert(timezone.utc)
        else:
            df["bucket"] = df["ts"].dt.floor("1D")
    else:
        rule_str = TIMEFRAME_RULES[target_timeframe]
        df["bucket"] = df["ts"].dt.floor(rule_str)

    # Group by bucket and compute OHLCV
    rows = []
    for bucket_start, group in df.groupby("bucket"):
        if bucket_start.tzinfo is None:
            b_utc = bucket_start.replace(tzinfo=timezone.utc)
        else:
            b_utc = bucket_start.astimezone(timezone.utc)

        present_count = len(group)
        expected = get_expected_bars(b_utc, target_timeframe, calendar)

        if expected <= 0:
            continue

        coverage = float(present_count) / float(expected)
        if coverage < min_bar_coverage:
            continue

        if target_timeframe == "1d":
            if calendar is not None:
                b_close = get_daily_bucket_end(b_utc, calendar)
            else:
                b_close = b_utc + timedelta(days=1)
        else:
            b_close = b_utc + delta

        sorted_group = group.sort_values("ts")
        rows.append(
            {
                "ts": b_utc,
                "ts_close": b_close,
                "open": float(sorted_group["open"].iloc[0]),
                "high": float(sorted_group["high"].max()),
                "low": float(sorted_group["low"].min()),
                "close": float(sorted_group["close"].iloc[-1]),
                "volume": float(sorted_group["volume"].sum()),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["ts", "ts_close", "open", "high", "low", "close", "volume"])

    resampled = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

    # Determine end of available 1m data
    data_end_time = df["ts"].max() + timedelta(minutes=1)

    effective_now = current_time
    if effective_now is not None:
        if effective_now.tzinfo is None:
            effective_now = effective_now.replace(tzinfo=timezone.utc)
        else:
            effective_now = effective_now.astimezone(timezone.utc)
        cutoff = min(effective_now, data_end_time)
    else:
        cutoff = data_end_time

    # A bucket starting at T is closed if and only if its close timestamp ts_close <= cutoff
    resampled = resampled[resampled["ts_close"] <= cutoff]

    return resampled.sort_values("ts").reset_index(drop=True)


def detect_gaps(
    df: pd.DataFrame,
    timeframe: str,
    calendar: Optional[CalendarConfig] = None,
) -> list[tuple[datetime, datetime]]:
    """Scan candle DataFrame for missing intervals where the market was expected to be open.

    Calendar-aware rule:
      - Only missing intervals where expected_bars(bucket, calendar) > 0 are reported as gaps.
      - Intervals where the market is closed (e.g. forex weekends) are NOT gaps.
    """
    if len(df) < 2 or timeframe not in TIMEFRAME_DELTAS:
        return []

    delta = TIMEFRAME_DELTAS[timeframe]
    ts_series = pd.to_datetime(df["ts"], utc=True).sort_values()

    gaps: list[tuple[datetime, datetime]] = []

    for prev_ts, curr_ts in zip(ts_series.iloc[:-1], ts_series.iloc[1:]):
        if curr_ts <= prev_ts + delta:
            continue

        # There is a jump from prev_ts to curr_ts. Check if any bucket in between had expected > 0.
        check_time = prev_ts + delta
        gap_has_open_market = False

        while check_time < curr_ts:
            b_utc = check_time.to_pydatetime() if hasattr(check_time, "to_pydatetime") else check_time
            if get_expected_bars(b_utc, timeframe, calendar) > 0:
                gap_has_open_market = True
                break
            check_time += delta

        if gap_has_open_market:
            p_dt = prev_ts.to_pydatetime() if hasattr(prev_ts, "to_pydatetime") else prev_ts
            c_dt = curr_ts.to_pydatetime() if hasattr(curr_ts, "to_pydatetime") else curr_ts
            gaps.append((p_dt, c_dt))

    return gaps


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
