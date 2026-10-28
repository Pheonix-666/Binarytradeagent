"""Data store — persistent storage for candles, payouts, and trades using SQLite."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from core.models import AssetQuote


DEFAULT_DB_PATH = Path("state") / "trades.sqlite"


class DataStore:
    """SQLite-backed data store for candles, quotes/payouts, and trade records."""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Candles table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS candles (
                    asset_id TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL DEFAULT 0.0,
                    PRIMARY KEY (asset_id, timeframe, ts)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_candles_asset_tf_ts 
                ON candles (asset_id, timeframe, ts)
                """
            )

            # Payouts and quotes table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS payouts (
                    asset_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    payout REAL NOT NULL,
                    is_otc INTEGER NOT NULL,
                    is_open INTEGER NOT NULL,
                    PRIMARY KEY (asset_id, ts)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_payouts_asset_ts 
                ON payouts (asset_id, ts)
                """
            )
            conn.commit()

    def save_candles(
        self,
        asset_id: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> int:
        """Insert or replace closed candles for an asset and timeframe.

        Returns the number of rows inserted/updated.
        """
        if df.empty:
            return 0

        candles_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(candles_df["ts"]):
            candles_df["ts"] = pd.to_datetime(candles_df["ts"], utc=True)
        elif candles_df["ts"].dt.tz is None:
            candles_df["ts"] = candles_df["ts"].dt.tz_localize(timezone.utc)
        else:
            candles_df["ts"] = candles_df["ts"].dt.tz_convert(timezone.utc)

        ts_strings = candles_df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        if "volume" not in candles_df.columns:
            volume_series = 0.0
        else:
            volume_series = candles_df["volume"].fillna(0.0)

        rows = [
            (
                asset_id,
                timeframe,
                ts_str,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(vol),
            )
            for ts_str, (_, row), vol in zip(
                ts_strings,
                candles_df.iterrows(),
                volume_series if isinstance(volume_series, pd.Series) else [0.0] * len(candles_df),
            )
        ]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO candles (asset_id, timeframe, ts, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return len(rows)

    def get_candles(
        self,
        asset_id: str,
        timeframe: str,
        n: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch candles for an asset and timeframe, sorted ascending by ts."""
        query = "SELECT ts, open, high, low, close, volume FROM candles WHERE asset_id = ? AND timeframe = ?"
        params: list[Union[str, int]] = [asset_id, timeframe]

        if since is not None:
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)
            else:
                since = since.astimezone(timezone.utc)
            query += " AND ts >= ?"
            params.append(since.strftime("%Y-%m-%dT%H:%M:%SZ"))

        query += " ORDER BY ts ASC"

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        if n is not None and n > 0:
            df = df.tail(n).reset_index(drop=True)

        return df

    def save_payout(self, quote: AssetQuote, ts: Optional[datetime] = None) -> None:
        """Save a single asset quote and payout snapshot."""
        self.save_payouts([quote], ts=ts)

    def save_payouts(
        self,
        quotes: list[AssetQuote],
        ts: Optional[datetime] = None,
    ) -> int:
        """Save quotes and payouts snapshot. Returns number of saved records."""
        if not quotes:
            return 0

        effective_ts = ts or datetime.now(timezone.utc)
        if effective_ts.tzinfo is None:
            effective_ts = effective_ts.replace(tzinfo=timezone.utc)
        else:
            effective_ts = effective_ts.astimezone(timezone.utc)

        ts_str = effective_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        rows = [
            (
                q.asset_id,
                ts_str,
                float(q.payout),
                1 if q.is_otc else 0,
                1 if q.is_open else 0,
            )
            for q in quotes
        ]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO payouts (asset_id, ts, payout, is_otc, is_open)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return len(rows)

    def get_latest_payout(self, asset_id: str) -> Optional[float]:
        """Return the most recent recorded payout for asset_id, or None."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT payout FROM payouts WHERE asset_id = ? ORDER BY ts DESC LIMIT 1",
                (asset_id,),
            )
            row = cursor.fetchone()
            return float(row["payout"]) if row else None

    def get_payout_history(
        self,
        asset_id: str,
        since: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Return payout history for asset_id."""
        query = "SELECT ts, payout, is_otc, is_open FROM payouts WHERE asset_id = ?"
        params: list[str] = [asset_id]
        if since is not None:
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)
            else:
                since = since.astimezone(timezone.utc)
            query += " AND ts >= ?"
            params.append(since.strftime("%Y-%m-%dT%H:%M:%SZ"))

        query += " ORDER BY ts ASC"

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df["is_otc"] = df["is_otc"].astype(bool)
            df["is_open"] = df["is_open"].astype(bool)

        return df
