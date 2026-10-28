"""Data recorder — records candles, quotes, and payouts from an adapter into SQLite."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

import pandas as pd

from adapters.base import PlatformAdapter
from core.logging_setup import get_logger
from core.models import AssetQuote
from data.store import DataStore


logger = get_logger()


class DataRecorder:
    """Orchestrates polling and persisting platform market data and payouts."""

    def __init__(self, adapter: PlatformAdapter, store: DataStore):
        self.adapter = adapter
        self.store = store

    def record_payouts(self) -> list[AssetQuote]:
        """Fetch current asset quotes and payouts from the adapter and save to store."""
        try:
            quotes = self.adapter.get_assets()
            if quotes:
                count = self.store.save_payouts(quotes)
                logger.info(f"Recorded payouts for {count} assets")
            return quotes
        except Exception as exc:
            logger.error(f"Failed to record payouts: {exc}")
            return []

    def record_candles(
        self,
        asset_id: str,
        timeframe: str,
        n: int = 100,
    ) -> int:
        """Fetch latest candles for asset_id from adapter and persist them."""
        try:
            df = self.adapter.get_candles(asset_id, timeframe, n)
            if not df.empty:
                count = self.store.save_candles(asset_id, timeframe, df)
                return count
            return 0
        except Exception as exc:
            logger.error(f"Failed to record candles for {asset_id} ({timeframe}): {exc}")
            return 0

    def record_universe(
        self,
        assets: Optional[Sequence[str]] = None,
        timeframes: Sequence[str] = ("1m", "5m", "15m", "1h", "1d"),
        n: int = 100,
    ) -> dict[str, int]:
        """Record payouts and candles for specified assets (or all open assets)."""
        quotes = self.record_payouts()

        if assets is None:
            target_assets = [q.asset_id for q in quotes if q.is_open]
        else:
            target_assets = list(assets)

        saved_counts: dict[str, int] = {}
        for asset_id in target_assets:
            for tf in timeframes:
                key = f"{asset_id}_{tf}"
                saved = self.record_candles(asset_id, tf, n=n)
                saved_counts[key] = saved

        return saved_counts
