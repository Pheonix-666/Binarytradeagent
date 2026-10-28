"""Market trading calendars — trading hours, weekly windows, and bar coverage expectations.

Section 6.1 & Section 12 (Phase 5 fixes):
  - Supported calendar types: 'always_open' (crypto) and 'weekly_windows' (forex, equities).
  - Determines open/closed market status for any UTC timestamp.
  - Computes expected_bars for any timeframe bucket.
  - Aligns daily buckets to daily_boundary_tz and daily_boundary_hour (e.g. 17:00 America/New_York for forex).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import math
from typing import Optional, Union
import zoneinfo

import pandas as pd

from core.config import CalendarConfig, CalendarWindowConfig

DAY_NAME_TO_INT = {
    "mon": 0, "monday": 0,
    "tue": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}


def parse_time_str(t_str: str) -> time:
    """Parse HH:MM string into datetime.time."""
    parts = t_str.strip().split(":")
    return time(hour=int(parts[0]), minute=int(parts[1]))


def is_market_open(
    dt: datetime,
    calendar: CalendarConfig,
) -> bool:
    """Determine whether the market is open at timestamp dt (UTC)."""
    if calendar.type == "always_open":
        return True

    if not calendar.windows:
        return True

    # Ensure dt is timezone-aware in UTC
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)

    # Check each configured weekly window
    for win in calendar.windows:
        tz_name = getattr(win, "tz", calendar.daily_boundary_tz) or "UTC"
        try:
            tz = zoneinfo.ZoneInfo(tz_name)
        except Exception:
            tz = timezone.utc

        dt_local = dt_utc.astimezone(tz)

        # Weekly minute from Monday 00:00 (0 to 10079)
        cur_weekly_min = dt_local.weekday() * 1440 + dt_local.hour * 60 + dt_local.minute

        open_day = DAY_NAME_TO_INT[win.open_day.lower()[:3]]
        open_t = parse_time_str(win.open_time)
        open_weekly_min = open_day * 1440 + open_t.hour * 60 + open_t.minute

        close_day = DAY_NAME_TO_INT[win.close_day.lower()[:3]]
        close_t = parse_time_str(win.close_time)
        close_weekly_min = close_day * 1440 + close_t.hour * 60 + close_t.minute

        if open_weekly_min <= close_weekly_min:
            # Standard window (e.g. Mon 08:00 to Fri 17:00)
            if open_weekly_min <= cur_weekly_min < close_weekly_min:
                return True
        else:
            # Wrap-around window (e.g. Sun 17:00 to Fri 17:00)
            if cur_weekly_min >= open_weekly_min or cur_weekly_min < close_weekly_min:
                return True

    return False


def get_daily_bucket_start(
    dt: datetime,
    calendar: CalendarConfig,
) -> datetime:
    """Calculate the UTC timestamp for the start of the daily candle containing dt.

    Aligned to daily_boundary_tz and daily_boundary_hour (e.g. 17:00 America/New_York).
    """
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)

    tz_name = calendar.daily_boundary_tz or "UTC"
    try:
        tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    dt_local = dt_utc.astimezone(tz)
    b_hour = calendar.daily_boundary_hour

    if dt_local.hour < b_hour:
        # Belongs to previous day's boundary
        start_date = dt_local.date() - timedelta(days=1)
    else:
        start_date = dt_local.date()

    start_local = datetime(
        start_date.year,
        start_date.month,
        start_date.day,
        b_hour,
        0,
        0,
        tzinfo=tz,
    )
    return start_local.astimezone(timezone.utc)


def expected_bars(
    bucket_start: datetime,
    timeframe: str,
    calendar: Optional[CalendarConfig] = None,
) -> int:
    """Calculate the number of open 1m bars the calendar expects in this bucket.

    Returns 0 if the market is entirely closed throughout the bucket.
    """
    if calendar is None or calendar.type == "always_open":
        if timeframe == "1m":
            return 1
        elif timeframe == "5m":
            return 5
        elif timeframe == "15m":
            return 15
        elif timeframe == "1h":
            return 60
        elif timeframe == "1d":
            return 1440
        return 1

    if bucket_start.tzinfo is None:
        b_utc = bucket_start.replace(tzinfo=timezone.utc)
    else:
        b_utc = bucket_start.astimezone(timezone.utc)

    if timeframe == "1m":
        return 1 if is_market_open(b_utc, calendar) else 0

    if timeframe == "5m":
        total_open = 0
        for m in range(5):
            if is_market_open(b_utc + timedelta(minutes=m), calendar):
                total_open += 1
        return total_open

    if timeframe == "15m":
        total_open = 0
        for m in range(15):
            if is_market_open(b_utc + timedelta(minutes=m), calendar):
                total_open += 1
        return total_open

    if timeframe == "1h":
        total_open = 0
        for m in range(60):
            if is_market_open(b_utc + timedelta(minutes=m), calendar):
                total_open += 1
        return total_open

    if timeframe == "1d":
        # 1440 minutes from daily boundary
        total_open = 0
        for m in range(1440):
            if is_market_open(b_utc + timedelta(minutes=m), calendar):
                total_open += 1
        return total_open

    return 1

