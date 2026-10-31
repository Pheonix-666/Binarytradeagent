"""Tests for swappable PriceSource interfaces and incremental candle cache.

PROJECT_SPEC.md Phase 9:
  - ReplaySource: replays CSV history driven by injected clock, never returns ts_close > clock.now.
  - CryptoPublicSource: public REST with rate limits, retries, backoff, closed candles only.
  - BinomoReadOnlySource: raises NotImplementedError.
  - Incremental candle caching in DataStore: fetches only new bars.
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

import numpy as np
import pandas as pd
import pytest

from core.clock import FakeClock
from core.config import AppConfig, SourceConfig
from core.models import AssetQuote
from data.price_source import (
    BinomoReadOnlySource,
    CryptoPublicSource,
    PriceSource,
    ReplaySource,
)
from data.store import DataStore


@pytest.fixture
def sample_csv_dir(tmp_path: Path) -> Path:
    """Create a temporary data_history directory with synthetic 1m CSV candles."""
    history_dir = tmp_path / "data_history"
    history_dir.mkdir(parents=True)

    # Base time: 2026-10-01 10:00:00 UTC
    base_t = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for i in range(120):  # 2 hours of 1m bars
        bar_t = base_t + timedelta(minutes=i)
        bar_close_t = bar_t + timedelta(minutes=1)
        o = price + i * 0.1
        h = o + 0.2
        l = o - 0.2
        c = o + 0.05
        rows.append({
            "ts": bar_t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ts_close": bar_close_t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": 10.0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(history_dir / "EURUSD.csv", index=False)
    return history_dir


def test_replay_source_never_returns_future_candles(sample_csv_dir: Path):
    """ReplaySource must never return candles where ts_close > clock.now()."""
    # Set clock to 10:15:00 UTC (15 closed 1m bars exist: 10:00 to 10:15)
    clock = FakeClock(initial_time=datetime(2026, 10, 1, 10, 15, tzinfo=timezone.utc))
    source = ReplaySource(
        clock=clock,
        data_dir=sample_csv_dir,
        assets=["EURUSD"],
    )

    candles = source.get_candles(asset="EURUSD", timeframe="1m", n=100)
    assert len(candles) == 15
    assert (candles["ts_close"] <= clock.now()).all()
    # Latest closed candle is from 10:14 with ts_close 10:15
    assert candles.iloc[-1]["ts_close"] == datetime(2026, 10, 1, 10, 15, tzinfo=timezone.utc)

    # Intermediate time: 10:15:30 UTC -> Still only 15 bars (10:15 bar closes at 10:16)
    clock.set_time(datetime(2026, 10, 1, 10, 15, 30, tzinfo=timezone.utc))
    candles_half = source.get_candles(asset="EURUSD", timeframe="1m", n=100)
    assert len(candles_half) == 15
    assert (candles_half["ts_close"] <= clock.now()).all()

    # Advance clock to 10:16:00 UTC -> exactly 16 bars
    clock.advance(seconds=30)
    candles_16 = source.get_candles(asset="EURUSD", timeframe="1m", n=100)
    assert len(candles_16) == 16
    assert (candles_16["ts_close"] <= clock.now()).all()
    assert candles_16.iloc[-1]["ts_close"] == datetime(2026, 10, 1, 10, 16, tzinfo=timezone.utc)


def test_replay_source_resampled_timeframes(sample_csv_dir: Path):
    """ReplaySource resamples to higher timeframes and excludes forming bars."""
    # At 10:25:00 UTC, five closed 5m bars exist: 10:00-10:05, 10:05-10:10, 10:10-10:15, 10:15-10:20, 10:20-10:25
    clock = FakeClock(initial_time=datetime(2026, 10, 1, 10, 25, tzinfo=timezone.utc))
    source = ReplaySource(
        clock=clock,
        data_dir=sample_csv_dir,
        assets=["EURUSD"],
    )

    c5m = source.get_candles(asset="EURUSD", timeframe="5m", n=10)
    assert len(c5m) == 5
    assert (c5m["ts_close"] <= clock.now()).all()
    assert c5m.iloc[-1]["ts_close"] == datetime(2026, 10, 1, 10, 25, tzinfo=timezone.utc)

    # At 10:28:00 UTC, 10:25-10:30 bar is still forming! Must NOT be returned
    clock.set_time(datetime(2026, 10, 1, 10, 28, tzinfo=timezone.utc))
    c5m_forming = source.get_candles(asset="EURUSD", timeframe="5m", n=10)
    assert len(c5m_forming) == 5
    assert (c5m_forming["ts_close"] <= clock.now()).all()


def test_replay_source_assets(sample_csv_dir: Path):
    """ReplaySource returns quotes with payout and is_otc."""
    clock = FakeClock(initial_time=datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc))
    source = ReplaySource(
        clock=clock,
        data_dir=sample_csv_dir,
        assets=["EURUSD", "BTCUSD_otc"],
        payout_map={"EURUSD": 0.88, "BTCUSD_otc": 0.90},
    )
    quotes = source.get_assets()
    assert len(quotes) == 2
    q0 = next(q for q in quotes if q.asset_id == "EURUSD")
    assert q0.payout == 0.88
    assert not q0.is_otc

    q1 = next(q for q in quotes if q.asset_id == "BTCUSD_otc")
    assert q1.payout == 0.90
    assert q1.is_otc


def test_binomo_readonly_source_raises():
    """BinomoReadOnlySource raises NotImplementedError for all operations."""
    source = BinomoReadOnlySource()
    with pytest.raises(NotImplementedError, match="BinomoReadOnlySource not implemented until Phase 9B"):
        source.get_assets()

    with pytest.raises(NotImplementedError, match="BinomoReadOnlySource not implemented until Phase 9B"):
        source.get_candles("EURUSD", "1m", 10)

    with pytest.raises(NotImplementedError, match="BinomoReadOnlySource not implemented until Phase 9B"):
        source.now()


def test_crypto_public_source_closed_candles_and_retry():
    """CryptoPublicSource fetches closed candles, ignores in-progress bars, and retries on failure."""
    cur_time = datetime(2026, 10, 1, 12, 5, 30, tzinfo=timezone.utc)
    clock = FakeClock(initial_time=cur_time)

    # 3 bars: 12:03-12:04 (closed), 12:04-12:05 (closed), 12:05-12:06 (in-progress)
    mock_binance_data = [
        [1790856180000, "100.0", "101.0", "99.0", "100.5", "10.0", 1790856239999],  # close 12:04:00
        [1790856240000, "100.5", "102.0", "100.0", "101.5", "15.0", 1790856299999],  # close 12:05:00
        [1790856300000, "101.5", "103.0", "101.0", "102.5", "12.0", 1790856359999],  # close 12:06:00 (future)
    ]

    source = CryptoPublicSource(
        clock=clock,
        assets=["BTCUSDT"],
        min_request_interval=0.0,
        backoff_base=0.01,
        max_retries=2,
    )

    with patch.object(source, "_http_get_json", return_value=mock_binance_data) as mock_get:
        candles = source.get_candles(asset="BTCUSDT", timeframe="1m", n=10)
        assert mock_get.called
        # Only the 2 closed bars are returned; the forming bar (close 12:06 > 12:05:30) is dropped
        assert len(candles) == 2
        assert (candles["ts_close"] <= clock.now()).all()


def test_crypto_public_source_retries_and_fails():
    """CryptoPublicSource retries up to max_retries on persistent error."""
    source = CryptoPublicSource(
        clock=FakeClock(initial_time=datetime.now(timezone.utc)),
        assets=["BTCUSDT"],
        min_request_interval=0.0,
        backoff_base=0.01,
        max_retries=2,
    )

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network down")):
        with pytest.raises(Exception):
            source.get_candles(asset="BTCUSDT", timeframe="1m", n=10)


def test_incremental_candle_cache_fetches_only_new_bars(tmp_path: Path, sample_csv_dir: Path):
    """DataStore.sync_candles_from_source fetches all bars on cold start, then only incremental new bars."""
    db_path = tmp_path / "market.db"
    store = DataStore(db_path=db_path)

    clock = FakeClock(initial_time=datetime(2026, 10, 1, 10, 10, tzinfo=timezone.utc))
    source = ReplaySource(
        clock=clock,
        data_dir=sample_csv_dir,
        assets=["EURUSD"],
    )

    # 1. Cold start: 10 closed bars exist (10:00 to 10:10)
    c1 = store.sync_candles_from_source(source=source, asset_id="EURUSD", timeframe="1m", n_needed=50)
    assert len(c1) == 10
    latest_ts = store.get_latest_candle_time("EURUSD", "1m")
    assert latest_ts == datetime(2026, 10, 1, 10, 9, tzinfo=timezone.utc)

    # Spy on source.get_candles to check how many bars are requested on incremental sync
    orig_get_candles = source.get_candles
    calls = []

    def spied_get_candles(asset: str, timeframe: str, n: int):
        calls.append(n)
        return orig_get_candles(asset, timeframe, n)

    source.get_candles = spied_get_candles

    # 2. Advance clock by 3 minutes (10:13:00 UTC) -> 3 new closed bars (10:10, 10:11, 10:12)
    clock.advance(seconds=180)
    c2 = store.sync_candles_from_source(source=source, asset_id="EURUSD", timeframe="1m", n_needed=50)

    # Total cached bars in SQLite should now be 13
    assert len(c2) == 13
    assert calls[0] <= 10  # Incremental fetch requested small window (fetch_n), not full history!
    assert store.get_latest_candle_time("EURUSD", "1m") == datetime(2026, 10, 1, 10, 12, tzinfo=timezone.utc)

