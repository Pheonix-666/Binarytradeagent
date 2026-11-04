"""Binomo passive price and asset reader implementing PriceSource.

Adheres strictly to PROJECT_SPEC.md:
- Read-only: executes only GET requests via the page's authenticated context.
- Zero clicks, zero fills, zero key presses, zero button presses.
- Enforces closed candles only (ts_close <= now, zero look-ahead).
- Rate limits requests and immediately stops with an alert on HTTP 401, 403, or 429 (no retry loops).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import time
from typing import Any, Mapping, Optional
import urllib.parse

import pandas as pd

from adapters.binomo.session import BinomoSession
from alerts.telegram import TelegramAlerter
from core.config import AppConfig
from core.models import AssetQuote
from data.candle_builder import TIMEFRAME_DELTAS
from data.price_source import PriceSource

logger = logging.getLogger("trading_system.adapters.binomo.reader")


def is_otc_instrument(ric: str, name: str) -> bool:
    """Return True if asset is OTC (ric starts with Z- or name contains (OTC))."""
    r = (ric or "").upper()
    n = (name or "").upper()
    return r.startswith("Z-") or "(OTC)" in n or "(otc)" in name.lower()


def is_schedule_open(schedule_data: Any, now_utc: datetime) -> bool:
    """Check if schedule indicates the asset is currently open."""
    if not schedule_data:
        return True

    if isinstance(schedule_data, dict):
        status = schedule_data.get("status") or schedule_data.get("trading_status")
        if status and str(status).lower() in ("closed", "disabled", "maintenance"):
            return False

        # Check day-of-week schedule: {"mon": [...], "tue": [...]} or {"mon": {"from": ..., "to": ...}}
        day_key = now_utc.strftime("%a").lower()[:3]  # "mon", "tue", ...
        if day_key in schedule_data:
            day_rules = schedule_data[day_key]
            if day_rules is False or day_rules is None:
                return False
            if isinstance(day_rules, dict):
                # e.g. {"open": "00:00", "close": "23:59"}
                open_str = day_rules.get("open") or day_rules.get("from")
                close_str = day_rules.get("close") or day_rules.get("to")
                if open_str and close_str:
                    try:
                        cur_time = now_utc.strftime("%H:%M")
                        return open_str <= cur_time <= close_str
                    except Exception:
                        pass
            elif isinstance(day_rules, list):
                if not day_rules:
                    return False
                cur_time = now_utc.strftime("%H:%M")
                matched_any = False
                for interval in day_rules:
                    if isinstance(interval, dict):
                        o = interval.get("from") or interval.get("open")
                        c = interval.get("to") or interval.get("close")
                        if o and c and (o <= cur_time <= c):
                            matched_any = True
                            break
                return matched_any

    return True


class BinomoReadOnlySource(PriceSource):
    """Binomo read-only PriceSource implementation for Assets and Candles."""

    def __init__(
        self,
        session: Optional[BinomoSession] = None,
        config: Optional[AppConfig] = None,
        alerter: Optional[TelegramAlerter] = None,
        base_api_url: str = "https://api.binomo.com",
    ) -> None:
        self.session = session
        self.config = config
        self.alerter = alerter
        self.base_api_url = base_api_url.rstrip("/")

        self._stopped = False
        self._request_history: list[float] = []

        # Rate limit and backoff settings
        self.requests_per_minute = 60
        self.option_type = "turbo"
        self.demo_required = True
        self.timeframe_segments: dict[str, int] = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "60m": 3600,
        }

        if config and hasattr(config, "binomo"):
            b_cfg = config.binomo
            self.requests_per_minute = getattr(b_cfg, "requests_per_minute", 60)
            self.option_type = getattr(b_cfg, "option_type", "turbo").lower()
            self.demo_required = getattr(b_cfg, "demo_required", True)
            if getattr(b_cfg, "candle_timeframe_segments", None):
                self.timeframe_segments = {
                    str(k): int(v) for k, v in b_cfg.candle_timeframe_segments.items()
                }

    def now(self) -> datetime:
        """Return current timestamp in UTC."""
        return datetime.now(timezone.utc)

    def _enforce_rate_limit(self) -> None:
        """Sliding window rate limit enforcing configured requests_per_minute."""
        if self._stopped:
            raise RuntimeError("BinomoReadOnlySource is stopped due to prior 401/403/429 error.")

        now_t = time.time()
        # Prune older than 60 seconds
        self._request_history = [t for t in self._request_history if now_t - t < 60.0]

        if len(self._request_history) >= self.requests_per_minute:
            sleep_needed = 60.0 - (now_t - self._request_history[0]) + 0.1
            if sleep_needed > 0:
                logger.info("Rate limit reached (%d rpm). Sleeping %.2fs", self.requests_per_minute, sleep_needed)
                time.sleep(sleep_needed)

        self._request_history.append(time.time())

    def _execute_get(self, url: str) -> dict[str, Any]:
        """Execute GET request via Playwright request context with backoff & stop-on-429."""
        if self._stopped:
            raise RuntimeError("BinomoReadOnlySource is stopped due to prior fatal error.")

        self._enforce_rate_limit()

        max_retries = 3
        backoff_sec = 1.0

        for attempt in range(max_retries):
            try:
                page = self.session.page if self.session else None
                if not page or not hasattr(page, "request"):
                    raise RuntimeError("No active Playwright page request context available.")

                response = page.request.get(url, timeout=15000)
                status = response.status

                # Stop on 401, 403, 429 immediately without retry
                if status in (401, 403, 429):
                    self._stopped = True
                    msg = f"CRITICAL: Binomo API returned HTTP {status} on {url}. Stopping reader immediately."
                    logger.critical(msg)
                    if self.alerter:
                        try:
                            self.alerter.send_alert(msg)
                        except Exception:
                            pass
                    raise RuntimeError(f"Binomo API HTTP {status}: {response.text()}")

                if status == 200:
                    return response.json()

                # Other HTTP errors: transient retry
                logger.warning("HTTP %d on %s (attempt %d/%d)", status, url, attempt + 1, max_retries)
            except RuntimeError:
                raise
            except Exception as exc:
                logger.warning("Request error on %s (attempt %d/%d): %s", url, attempt + 1, max_retries, exc)

            if attempt < max_retries - 1:
                time.sleep(backoff_sec)
                backoff_sec *= 2.0

        raise RuntimeError(f"Failed GET {url} after {max_retries} attempts.")

    # ── Asset Discovery and Quotes ──────────────────────────────────────────

    def get_assets(self) -> list[AssetQuote]:
        """Query /bo-assets/v7/assets and parse available assets with payout and OTC status."""
        url = f"{self.base_api_url}/bo-assets/v7/assets"
        data = self._execute_get(url)

        asset_list: list[dict[str, Any]] = []
        if isinstance(data, list):
            asset_list = data
        elif isinstance(data, dict):
            if isinstance(data.get("data"), list):
                asset_list = data["data"]
            elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("assets"), list):
                asset_list = data["data"]["assets"]
            elif isinstance(data.get("assets"), list):
                asset_list = data["assets"]

        quotes: list[AssetQuote] = []
        now_dt = self.now()

        for item in asset_list:
            if not isinstance(item, dict):
                continue

            ric = str(item.get("ric") or item.get("symbol") or "")
            name = str(item.get("name") or item.get("asset_name") or ric)
            if not ric:
                continue

            otc = is_otc_instrument(ric, name)

            tts = item.get("trading_tools_settings") if isinstance(item.get("trading_tools_settings"), dict) else {}
            ftt = tts.get("ftt") if isinstance(tts.get("ftt"), dict) else {}

            # Check demo eligibility if required
            enabled_accounts = (
                item.get("enabled_account_types")
                or ftt.get("enabled_account_types")
                or []
            )
            if self.demo_required and "demo" not in [str(a).lower() for a in enabled_accounts]:
                continue

            # Check schedule / open status
            sched = ftt.get("schedule") or item.get("schedule") or {}
            if not is_schedule_open(sched, now_dt):
                logger.debug("Asset %s is currently closed according to schedule. Skipping.", ric)
                continue

            # Extract payout for configured option_type (blitz, turbo, standard)
            rate_key = f"base_payment_rate_{self.option_type}"
            payout_val = ftt.get(rate_key)
            if payout_val is None or payout_val == "-" or payout_val == "<num>":
                payout_val = item.get("payment_rate") or item.get("payout")

            if payout_val is None:
                logger.debug("Asset %s has no payout rate for option_type '%s'. Skipping.", ric, self.option_type)
                continue

            try:
                payout_float = float(payout_val)
                # If expressed as percentage (e.g. 83), convert to decimal 0.83
                if payout_float > 1.0:
                    payout_float /= 100.0
            except (ValueError, TypeError):
                continue

            if payout_float <= 0.0:
                logger.debug("Asset %s payout <= 0. Skipping.", ric)
                continue

            quote = AssetQuote(
                asset_id=ric,
                name=name,
                is_otc=otc,
                payout=payout_float,
                is_open=True,
            )
            quotes.append(quote)

        logger.info("BinomoReadOnlySource loaded %d valid assets (option_type=%s)", len(quotes), self.option_type)
        return quotes

    # ── Closed Candles Fetching ─────────────────────────────────────────────

    def get_candles(self, asset: str, timeframe: str, n: int) -> pd.DataFrame:
        """Fetch historical closed candles for asset and timeframe.
        
        Guaranteed: ts_close <= self.now() (zero look-ahead).
        Paginates by day if n exceeds a single day's window.
        """
        tf_delta = TIMEFRAME_DELTAS.get(timeframe)
        if not tf_delta:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        segment = self.timeframe_segments.get(timeframe)
        if not segment:
            # Fallback based on timedelta seconds
            segment = int(tf_delta.total_seconds())

        encoded_ric = urllib.parse.quote(asset, safe="")
        now_dt = self.now()

        # Calculate how many days we might need
        seconds_per_bar = tf_delta.total_seconds()
        total_seconds_needed = n * seconds_per_bar
        days_needed = max(1, int(total_seconds_needed // 86400) + 1)

        raw_bars: list[dict[str, Any]] = []

        # Paginate backwards from today
        for day_offset in range(days_needed, -1, -1):
            target_date = (now_dt - timedelta(days=day_offset)).strftime("%Y-%m-%dT00:00:00")
            url = f"{self.base_api_url}/candles/v1/{encoded_ric}/{target_date}/{segment}"

            try:
                res = self._execute_get(url)
            except RuntimeError:
                raise
            except Exception as exc:
                logger.warning("Candle request failed for %s on %s: %s", asset, target_date, exc)
                continue

            day_bars = []
            if isinstance(res, list):
                day_bars = res
            elif isinstance(res, dict):
                if isinstance(res.get("data"), list):
                    day_bars = res["data"]
                elif isinstance(res.get("candles"), list):
                    day_bars = res["candles"]

            raw_bars.extend(day_bars)

        if not raw_bars:
            return pd.DataFrame(columns=["ts", "ts_close", "open", "high", "low", "close", "volume"])

        parsed_rows: list[dict[str, Any]] = []
        for b in raw_bars:
            if not isinstance(b, dict):
                continue

            created_at = b.get("created_at") or b.get("time") or b.get("ts")
            if not created_at:
                continue

            try:
                if isinstance(created_at, (int, float)):
                    ts = datetime.fromtimestamp(created_at, tz=timezone.utc)
                else:
                    ts = pd.to_datetime(created_at, utc=True).to_pydatetime()
            except Exception:
                continue

            ts_close = ts + tf_delta

            # ZERO LOOK-AHEAD GUARANTEE: Candle must be fully closed
            if ts_close > now_dt:
                continue

            parsed_rows.append({
                "ts": ts,
                "ts_close": ts_close,
                "open": float(b.get("open") or b.get("o", 0.0)),
                "high": float(b.get("high") or b.get("h", 0.0)),
                "low": float(b.get("low") or b.get("l", 0.0)),
                "close": float(b.get("close") or b.get("c", 0.0)),
                "volume": float(b.get("volume") or b.get("v", 0.0)),
            })

        df = pd.DataFrame(parsed_rows)
        if df.empty:
            return pd.DataFrame(columns=["ts", "ts_close", "open", "high", "low", "close", "volume"])

        df = df.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
        return df.tail(n).reset_index(drop=True)