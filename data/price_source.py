"""Swappable price sources and market data interfaces.

PROJECT_SPEC.md Phase 9:
  - Unified PriceSource interface:
      get_assets() -> list[AssetQuote] with payout and is_otc.
      get_candles(asset, timeframe, n) -> CLOSED candles only with ts and ts_close.
      now() -> datetime.
  - ReplaySource: replays historical CSV from data_history/ as if live, driven by injected clock.
    Strict invariant: NEVER returns data with ts_close > clock.now().
  - CryptoPublicSource: public market data REST (no key) for real-market crypto pairs.
    Rate-limited, timeouts, retries with backoff. Closed candles only.
  - BinomoReadOnlySource: stub raising NotImplementedError until Phase 9B.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import time
from typing import Mapping, Optional, Union
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from core.clock import Clock, RealClock
from core.config import AppConfig, CalendarConfig
from core.models import AssetQuote
from data.calendar import is_market_open
from data.candle_builder import TIMEFRAME_DELTAS, resample_candles

logger = logging.getLogger("trading_system.price_source")


class PriceSource(ABC):
    """Abstract base class for all market data and price sources."""

    @abstractmethod
    def get_assets(self) -> list[AssetQuote]:
        """Return quotes for configured assets with payout and is_otc."""
        pass

    @abstractmethod
    def get_candles(self, asset: str, timeframe: str, n: int) -> pd.DataFrame:
        """Return CLOSED candles only with ts and ts_close, ordered by ts ASC.

        Length <= n. Guaranteed: ts_close <= self.now() (zero look-ahead).
        """
        pass

    @abstractmethod
    def now(self) -> datetime:
        """Return the current timestamp from the source's clock."""
        pass


class ReplaySource(PriceSource):
    """Replays historical 1-minute CSVs from data_history/ as if live.

    Driven by an injected Clock. Never returns candles with ts_close > clock.now().
    """

    def __init__(
        self,
        clock: Clock,
        config: Optional[AppConfig] = None,
        data_dir: Optional[Union[str, Path]] = None,
        assets: Optional[list[str]] = None,
        payout_map: Optional[Mapping[str, float]] = None,
        default_payout: float = 0.85,
    ) -> None:
        self.clock = clock
        self.config = config
        self.data_dir = Path(
            data_dir
            or (config.source.replay_dir if config and hasattr(config, "source") else "data_history")
        )
        self.assets = (
            assets
            or (config.source.assets if config and hasattr(config, "source") else ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD"])
        )
        self.payout_map = dict(payout_map or {})
        self.default_payout = default_payout
        self._raw_cache: dict[str, pd.DataFrame] = {}

    def now(self) -> datetime:
        return self.clock.now()

    def get_assets(self) -> list[AssetQuote]:
        current_time = self.now()
        calendar = self.config.calendar if self.config and hasattr(self.config, "calendar") else CalendarConfig()
        quotes: list[AssetQuote] = []

        for asset in self.assets:
            is_otc = "_otc" in asset.lower() or "otc" in asset.lower()
            is_open = True if is_otc else is_market_open(current_time, calendar)
            payout = self.payout_map.get(asset, self.default_payout)

            quotes.append(
                AssetQuote(
                    asset_id=asset,
                    name=asset,
                    is_otc=is_otc,
                    payout=payout,
                    is_open=is_open,
                )
            )
        return quotes

    def _load_raw_1m(self, asset: str) -> pd.DataFrame:
        if asset in self._raw_cache:
            return self._raw_cache[asset]

        csv_path = self.data_dir / f"{asset}.csv"
        if not csv_path.exists():
            # Try lowercase or search
            matches = list(self.data_dir.glob(f"*{asset}*.csv"))
            if matches:
                csv_path = matches[0]
            else:
                logger.warning("Replay CSV not found for asset %s at %s", asset, csv_path)
                empty_df = pd.DataFrame(columns=["ts", "ts_close", "open", "high", "low", "close", "volume"])
                self._raw_cache[asset] = empty_df
                return empty_df

        df = pd.read_csv(csv_path)
        if "ts" not in df.columns:
            raise ValueError(f"CSV for {asset} missing required 'ts' column")

        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        if "ts_close" not in df.columns:
            df["ts_close"] = df["ts"] + timedelta(minutes=1)
        else:
            df["ts_close"] = pd.to_datetime(df["ts_close"], utc=True)

        if "volume" not in df.columns:
            df["volume"] = 0.0

        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)

        df = df.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
        self._raw_cache[asset] = df
        return df

    def get_candles(self, asset: str, timeframe: str, n: int) -> pd.DataFrame:
        current_time = self.now()
        df_1m = self._load_raw_1m(asset)
        if df_1m.empty:
            return pd.DataFrame(columns=["ts", "ts_close", "open", "high", "low", "close", "volume"])

        # Strict boundary: ts_close <= clock.now()
        closed_1m = df_1m[df_1m["ts_close"] <= current_time]
        if closed_1m.empty:
            return pd.DataFrame(columns=["ts", "ts_close", "open", "high", "low", "close", "volume"])

        if timeframe == "1m":
            res = closed_1m.tail(n).copy()
            return res.reset_index(drop=True)

        calendar = self.config.calendar if self.config and hasattr(self.config, "calendar") else None
        multiplier = {"5m": 5, "15m": 15, "1h": 60, "1d": 1440}.get(timeframe, 60)
        needed_1m = max(100, (n + 10) * multiplier)
        source_1m = closed_1m.tail(needed_1m)
        resampled = resample_candles(
            df_1m=source_1m,
            target_timeframe=timeframe,
            current_time=current_time,
            calendar=calendar,
        )
        if resampled.empty:
            return pd.DataFrame(columns=["ts", "ts_close", "open", "high", "low", "close", "volume"])

        # Final invariant assertion: ts_close <= current_time
        resampled_closed = resampled[resampled["ts_close"] <= current_time]
        return resampled_closed.tail(n).reset_index(drop=True)


class CryptoPublicSource(PriceSource):
    """Public REST market-data source for major crypto pairs (no API key required).

    Features:
      - Closed candles only (ts_close <= clock.now()).
      - Rate-limited requests with backoff and retry.
      - Uses Binance Public REST /api/v3/klines.
    """

    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "1d": "1d",
    }

    def __init__(
        self,
        clock: Optional[Clock] = None,
        config: Optional[AppConfig] = None,
        assets: Optional[list[str]] = None,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        min_request_interval: float = 0.2,
    ) -> None:
        self.clock = clock or RealClock()
        self.config = config
        src_cfg = getattr(config, "source", None)
        self.assets = (
            assets
            or (getattr(src_cfg, "assets", None))
            or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        )
        self.base_url = (
            base_url
            or getattr(src_cfg, "public_api_url", "https://data-api.binance.vision/api/v3")
        ).rstrip("/")
        self.timeout = timeout if timeout > 0 else getattr(src_cfg, "rest_timeout_seconds", 10.0)
        self.max_retries = max_retries if max_retries > 0 else getattr(src_cfg, "max_retries", 3)
        self.backoff_base = backoff_base if backoff_base > 0 else getattr(src_cfg, "backoff_base_seconds", 1.0)
        self.min_request_interval = min_request_interval
        self._last_request_time: float = 0.0

    def now(self) -> datetime:
        return self.clock.now()

    def get_assets(self) -> list[AssetQuote]:
        quotes = []
        for asset in self.assets:
            quotes.append(
                AssetQuote(
                    asset_id=asset,
                    name=asset,
                    is_otc=False,
                    payout=0.85,
                    is_open=True,  # Crypto markets trade 24/7
                )
            )
        return quotes

    def _normalize_symbol(self, asset: str) -> str:
        s = asset.upper().replace("/", "").replace("-", "")
        if s.endswith("USD") and not s.endswith("USDT"):
            s = s + "T"
        return s

    def _http_get_json(self, url: str) -> list[Any]:
        attempt = 0
        while attempt <= self.max_retries:
            # Enforce minimum interval between calls
            elapsed = time.time() - self._last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "TradeAgent/1.0", "Accept": "application/json"},
            )
            try:
                self._last_request_time = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        return data
                    raise urllib.error.HTTPError(url, resp.status, "HTTP Error", resp.headers, None)
            except Exception as exc:
                attempt += 1
                if attempt > self.max_retries:
                    logger.error("CryptoPublicSource failed request after %d attempts to %s: %s", attempt, url, exc)
                    raise
                backoff = self.backoff_base * (2 ** (attempt - 1))
                logger.warning("CryptoPublicSource retrying in %.2fs (attempt %d/%d): %s", backoff, attempt, self.max_retries, exc)
                time.sleep(backoff)
        return []

    def get_candles(self, asset: str, timeframe: str, n: int) -> pd.DataFrame:
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        symbol = self._normalize_symbol(asset)
        interval = self.TIMEFRAME_MAP[timeframe]
        # Request slightly more than n to guarantee n closed candles
        limit = min(1000, max(n + 10, 20))

        params = urllib.parse.urlencode({
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        })
        url = f"{self.base_url}/klines?{params}"

        raw_klines = self._http_get_json(url)
        if not raw_klines:
            return pd.DataFrame(columns=["ts", "ts_close", "open", "high", "low", "close", "volume"])

        rows = []
        cur_now = self.now()
        for k in raw_klines:
            # Binance kline format:
            # 0: Open time (ms)
            # 1: Open
            # 2: High
            # 3: Low
            # 4: Close
            # 5: Volume
            # 6: Close time (ms)
            open_ms = int(k[0])
            close_ms = int(k[6])

            ts_open = datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc)
            # Binance close_time is e.g. 59999ms, so close bucket boundary is close_ms + 1ms
            ts_close = datetime.fromtimestamp((close_ms + 1) / 1000.0, tz=timezone.utc)

            # Strictly CLOSED candle check
            if ts_close > cur_now:
                continue

            rows.append({
                "ts": ts_open,
                "ts_close": ts_close,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["ts", "ts_close", "open", "high", "low", "close", "volume"])

        df = df.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
        return df.tail(n).reset_index(drop=True)


class BinomoReadOnlySource(PriceSource):
    """Stub price source raising NotImplementedError until Phase 9B."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_assets(self) -> list[AssetQuote]:
        raise NotImplementedError("BinomoReadOnlySource not implemented until Phase 9B")

    def get_candles(self, asset: str, timeframe: str, n: int) -> pd.DataFrame:
        raise NotImplementedError("BinomoReadOnlySource not implemented until Phase 9B")

    def now(self) -> datetime:
        raise NotImplementedError("BinomoReadOnlySource not implemented until Phase 9B")
