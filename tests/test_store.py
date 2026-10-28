"""Unit tests for SQLite DataStore and DataRecorder."""

from datetime import datetime, timedelta, timezone
import pandas as pd
import pytest

from core.models import AssetQuote
from data.recorder import DataRecorder
from data.store import DataStore
from adapters.paper import PaperAdapter


@pytest.fixture
def test_store(tmp_path):
    """Create a temporary SQLite data store."""
    db_file = tmp_path / "test_trades.sqlite"
    return DataStore(db_path=db_file)


def test_save_and_get_candles(test_store):
    base = datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([
        {"ts": base, "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, "volume": 10},
        {"ts": base + timedelta(minutes=5), "open": 102.0, "high": 107.0, "low": 101.0, "close": 106.0, "volume": 12},
    ])

    saved = test_store.save_candles("EURUSD", "5m", df)
    assert saved == 2

    loaded = test_store.get_candles("EURUSD", "5m")
    assert len(loaded) == 2
    assert loaded.iloc[0]["close"] == 102.0
    assert loaded.iloc[1]["close"] == 106.0


def test_upsert_candles_no_duplicates(test_store):
    base = datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc)
    df1 = pd.DataFrame([
        {"ts": base, "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, "volume": 10},
    ])
    test_store.save_candles("EURUSD", "5m", df1)

    # Upsert with updated close
    df2 = pd.DataFrame([
        {"ts": base, "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.5, "volume": 15},
    ])
    test_store.save_candles("EURUSD", "5m", df2)

    loaded = test_store.get_candles("EURUSD", "5m")
    assert len(loaded) == 1
    assert loaded.iloc[0]["close"] == 104.5


def test_save_and_get_payouts(test_store):
    now = datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc)
    quote1 = AssetQuote(
        asset_id="EURUSD",
        name="EUR/USD",
        is_otc=False,
        payout=0.85,
        is_open=True,
    )
    quote2 = AssetQuote(
        asset_id="BTCUSD",
        name="BTC/USD (OTC)",
        is_otc=True,
        payout=0.90,
        is_open=True,
    )

    count = test_store.save_payouts([quote1, quote2], ts=now)
    assert count == 2

    payout_eur = test_store.get_latest_payout("EURUSD")
    assert payout_eur == 0.85

    payout_btc = test_store.get_latest_payout("BTCUSD")
    assert payout_btc == 0.90

    history = test_store.get_payout_history("EURUSD")
    assert len(history) == 1
    assert history.iloc[0]["payout"] == 0.85


def test_data_recorder_integration(test_store):
    adapter = PaperAdapter(initial_balance=1000.0)
    quote = AssetQuote(
        asset_id="EURUSD",
        name="EUR/USD",
        is_otc=False,
        payout=0.87,
        is_open=True,
    )
    adapter.set_assets([quote])

    base = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    df_candles = pd.DataFrame([
        {"ts": base, "open": 1.10, "high": 1.12, "low": 1.09, "close": 1.11, "volume": 50},
    ])
    adapter.set_candles("EURUSD", "5m", df_candles)

    recorder = DataRecorder(adapter=adapter, store=test_store)
    recorded_quotes = recorder.record_payouts()
    assert len(recorded_quotes) == 1
    assert test_store.get_latest_payout("EURUSD") == 0.87

    saved_candles = recorder.record_candles("EURUSD", "5m")
    assert saved_candles == 1
    loaded = test_store.get_candles("EURUSD", "5m")
    assert len(loaded) == 1
    assert loaded.iloc[0]["close"] == 1.11
