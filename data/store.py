"""Data store — persistent storage for candles, payouts, and trades using SQLite."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from core.models import AssetQuote
from dataclasses import dataclass


@dataclass
class TradeRecord:
    """Persistent database record for a single trade."""

    id: str
    opened_at: str  # ISO 8601 string
    asset: str
    is_otc: bool
    direction: str  # "CALL" or "PUT"
    expiry_minutes: int
    stake: float
    payout_at_entry: float
    result: str = "PENDING"  # "WIN", "LOSS", "TIE", "PENDING"
    pnl: float = 0.0
    mode: str = "paper"  # "paper", "demo", "live"
    closed_at: Optional[str] = None
    pattern_name: Optional[str] = None
    pattern_type: Optional[str] = None
    risk_score: Optional[int] = None
    hour_utc: Optional[int] = None

    def __post_init__(self) -> None:
        if self.hour_utc is None and self.opened_at:
            try:
                dt = pd.to_datetime(self.opened_at, utc=True)
                self.hour_utc = int(dt.hour)
            except Exception:
                self.hour_utc = 0


DEFAULT_DB_PATH = Path("state") / "trades.sqlite"


class DataStore:
    """SQLite-backed data store for candles, quotes/payouts, and trade records."""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH):
        if str(db_path) == ":memory:":
            self.db_path = ":memory:"
            self._mem_conn = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._mem_conn = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
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
                    ts_close TEXT,
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

            # Trades table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    asset TEXT NOT NULL,
                    is_otc INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    expiry_minutes INTEGER NOT NULL,
                    stake REAL NOT NULL,
                    payout_at_entry REAL NOT NULL,
                    result TEXT NOT NULL,
                    pnl REAL DEFAULT 0.0,
                    mode TEXT NOT NULL,
                    pattern_name TEXT,
                    pattern_type TEXT,
                    risk_score INTEGER,
                    hour_utc INTEGER NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trades_asset_mode_closed 
                ON trades (asset, mode, closed_at)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trades_opened_at 
                ON trades (opened_at)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trades_mode 
                ON trades (mode)
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
        if "ts_close" in candles_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(candles_df["ts_close"]):
                candles_df["ts_close"] = pd.to_datetime(candles_df["ts_close"], utc=True)
            ts_close_strings = candles_df["ts_close"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            ts_close_strings = [None] * len(candles_df)

        if "volume" not in candles_df.columns:
            volume_series = 0.0
        else:
            volume_series = candles_df["volume"].fillna(0.0)

        rows = [
            (
                asset_id,
                timeframe,
                ts_str,
                ts_close_str,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(vol),
            )
            for ts_str, ts_close_str, (_, row), vol in zip(
                ts_strings,
                ts_close_strings,
                candles_df.iterrows(),
                volume_series if isinstance(volume_series, pd.Series) else [0.0] * len(candles_df),
            )
        ]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO candles (asset_id, timeframe, ts, ts_close, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        query = "SELECT ts, ts_close, open, high, low, close, volume FROM candles WHERE asset_id = ? AND timeframe = ?"
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
            return pd.DataFrame(columns=["ts", "ts_close", "open", "high", "low", "close", "volume"])

        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        if "ts_close" in df.columns and not df["ts_close"].isna().all():
            df["ts_close"] = pd.to_datetime(df["ts_close"], utc=True)
        else:
            # Fallback for legacy rows
            from data.candle_builder import TIMEFRAME_DELTAS
            delta = TIMEFRAME_DELTAS.get(timeframe, timedelta(minutes=1))
            df["ts_close"] = df["ts"] + delta
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


class TradeRepository:
    """Repository for CRUD operations on the SQLite trades table."""

    def __init__(self, data_store: DataStore):
        self.store = data_store

    def insert(self, trade: TradeRecord) -> None:
        """Insert a new trade record."""
        with self.store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO trades (
                    id, opened_at, closed_at, asset, is_otc, direction,
                    expiry_minutes, stake, payout_at_entry, result, pnl,
                    mode, pattern_name, pattern_type, risk_score, hour_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.id,
                    trade.opened_at,
                    trade.closed_at,
                    trade.asset,
                    1 if trade.is_otc else 0,
                    trade.direction,
                    trade.expiry_minutes,
                    trade.stake,
                    trade.payout_at_entry,
                    trade.result,
                    trade.pnl,
                    trade.mode,
                    trade.pattern_name,
                    trade.pattern_type,
                    trade.risk_score,
                    trade.hour_utc if trade.hour_utc is not None else 0,
                ),
            )
            conn.commit()

    def update_result(
        self,
        trade_id: str,
        closed_at: Union[str, datetime],
        result: str,
        pnl: float,
    ) -> bool:
        """Update closed_at, result (WIN/LOSS/TIE), and pnl for an existing trade."""
        if isinstance(closed_at, datetime):
            if closed_at.tzinfo is None:
                closed_at = closed_at.replace(tzinfo=timezone.utc)
            else:
                closed_at = closed_at.astimezone(timezone.utc)
            closed_str = closed_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            closed_str = str(closed_at)

        with self.store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE trades
                SET closed_at = ?, result = ?, pnl = ?
                WHERE id = ?
                """,
                (closed_str, result, float(pnl), trade_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_trade_by_id(self, trade_id: str) -> Optional[TradeRecord]:
        """Fetch a single trade record by id, or None."""
        with self.store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def get_trades(
        self,
        asset: Optional[str] = None,
        mode: Optional[str] = None,
        since: Optional[Union[str, datetime]] = None,
        until: Optional[Union[str, datetime]] = None,
    ) -> list[TradeRecord]:
        """Query trade records with optional filtering."""
        query = "SELECT * FROM trades WHERE 1=1"
        params: list[Union[str, int]] = []

        if asset:
            query += " AND asset = ?"
            params.append(asset)
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        if since:
            since_str = (
                since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if isinstance(since, datetime)
                else str(since)
            )
            query += " AND opened_at >= ?"
            params.append(since_str)
        if until:
            until_str = (
                until.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if isinstance(until, datetime)
                else str(until)
            )
            query += " AND opened_at <= ?"
            params.append(until_str)

        query += " ORDER BY opened_at ASC"

        with self.store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TradeRecord:
        return TradeRecord(
            id=str(row["id"]),
            opened_at=str(row["opened_at"]),
            closed_at=str(row["closed_at"]) if row["closed_at"] else None,
            asset=str(row["asset"]),
            is_otc=bool(row["is_otc"]),
            direction=str(row["direction"]),
            expiry_minutes=int(row["expiry_minutes"]),
            stake=float(row["stake"]),
            payout_at_entry=float(row["payout_at_entry"]),
            result=str(row["result"]),
            pnl=float(row["pnl"]),
            mode=str(row["mode"]),
            pattern_name=str(row["pattern_name"]) if row["pattern_name"] else None,
            pattern_type=str(row["pattern_type"]) if row["pattern_type"] else None,
            risk_score=int(row["risk_score"]) if row["risk_score"] is not None else None,
            hour_utc=int(row["hour_utc"]),
        )
