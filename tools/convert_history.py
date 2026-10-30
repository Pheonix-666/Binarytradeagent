"""History data converter: converts Binance kline CSVs and HistData ASCII M1 files
to data_history/<ASSET>.csv with columns ts,open,high,low,close in UTC.

Timestamp Convention:
  - `ts` in the output CSV MUST always be the candle OPEN time in UTC.
  - For Binance klines, `open_time` is explicitly used as the candle open time.
  - For HistData M1, timestamps represent the start of the minute bucket (open time).
  - For any data source providing candle close times instead, specify `--ts-is-close`
    (or ts_is_close=True) to shift timestamps back by one minute (ts = ts - 1 minute)
    so that all stored candles consistently represent candle open time.

Capabilities:
  (a) Binance kline CSVs:
      - Auto-detects millisecond (13-digit) vs microsecond (16-digit) timestamps.
      - Handles files with or without standard Binance headers.
      - Extracts open_time, open, high, low, close.
  (b) HistData ASCII M1 files:
      - Supports semicolon and comma delimited formats.
      - Supports 5, 6, and 7-column variants (including separate date and time).
      - Converts from source timezone (default EST: fixed UTC-5 without DST) to UTC.
      - Configurable source timezone parameter (--source-tz).
  (c) Pipeline:
      - Sorts chronologically.
      - De-duplicates identical timestamps, reporting both total duplicate count
        and count of duplicates having conflicting/different prices (keeping first row).
      - Validates price bounds: high >= max(open, close), low <= min(open, close), prices > 0.
        Fails (exit non-zero) with count and first 10 offending rows by default, or drops
        them and prints dropped count when --drop-invalid is specified.
      - Emits output to data_history/<ASSET>.csv with ts formatted in ISO-8601 UTC.
      - Evaluates and prints a calendar-aware gap report using config.yaml calendar rules
        (scheduled closures like forex weekends are suppressed and not flagged as gaps).
      - Prints a final one-line summary: rows read, duplicates, invalid, rows written,
        date range, gap count.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Union
import zoneinfo

# Ensure repository root is in sys.path when script is invoked directly
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd

from core.config import AppConfig, CalendarConfig, load_config
from data.candle_builder import detect_gaps

logger = logging.getLogger(__name__)


def parse_tz(tz_str: str) -> datetime.tzinfo:
    """Parse a timezone string into a tzinfo object.

    Handles:
      - 'EST': Eastern Standard Time fixed at UTC-5 without DST (as used by HistData).
      - 'UTC' or 'Z': UTC.
      - Fixed offsets: 'UTC-5', 'UTC+02:00', '-05:00', '+0200', etc.
      - IANA timezone names: 'America/New_York', 'Europe/London', etc.
    """
    if not tz_str:
        return timezone.utc
    s = tz_str.strip()
    if s.upper() in ("UTC", "Z"):
        return timezone.utc
    if s.upper() == "EST":
        return zoneinfo.ZoneInfo("EST")
    # Check fixed offset patterns like UTC-5, GMT+2, -05:00, +0200
    m = re.match(r"^(?:UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2}))?$", s, re.IGNORECASE)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        hours = int(m.group(2))
        mins = int(m.group(3) or 0)
        return timezone(sign * timedelta(hours=hours, minutes=mins))
    try:
        return zoneinfo.ZoneInfo(s)
    except Exception as e:
        raise ValueError(f"Unknown or invalid timezone: '{tz_str}'") from e


def detect_format(file_path: Union[str, Path]) -> str:
    """Detect whether a historical data file is 'binance' or 'histdata'.

    Inspects delimiter and content of the first few non-empty lines.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = []
        for _ in range(20):
            line = f.readline()
            if not line:
                break
            line_str = line.strip()
            if line_str and not line_str.startswith("#"):
                lines.append(line_str)

    if not lines:
        raise ValueError(f"File {file_path} is empty")

    first_line = lines[0]
    first_line_lower = first_line.lower()

    # Semicolon delimiter is characteristic of HistData M1
    if ";" in first_line:
        return "histdata"

    # Header-based detection
    if "open_time" in first_line_lower or "quote_volume" in first_line_lower or "close_time" in first_line_lower:
        return "binance"
    if "datetime" in first_line_lower or "date_time" in first_line_lower or "ticker" in first_line_lower:
        return "histdata"

    # Inspect first token
    tokens = [t.strip() for t in re.split(r"[,;\s]+", first_line) if t.strip()]
    if tokens:
        first_token = tokens[0]
        # Pure integer Unix timestamp (Binance): 10, 13, or 16 digits
        if first_token.isdigit() and len(first_token) >= 10:
            return "binance"
        # Date string (HistData): YYYYMMDD or YYYY-MM-DD or YYYY.MM.DD
        if re.match(r"^\d{4}[-./]?\d{2}[-./]?\d{2}", first_token):
            return "histdata"

    # Check second line if first was a header
    if len(lines) > 1:
        tokens_l2 = [t.strip() for t in re.split(r"[,;\s]+", lines[1]) if t.strip()]
        if tokens_l2:
            t0 = tokens_l2[0]
            if t0.isdigit() and len(t0) >= 10:
                return "binance"
            if re.match(r"^\d{4}[-./]?\d{2}[-./]?\d{2}", t0):
                return "histdata"

    # Default fallback
    return "binance"


def parse_binance(file_path: Union[str, Path]) -> pd.DataFrame:
    """Parse Binance kline CSV file.

    Auto-detects:
      - millisecond (13-digit) vs microsecond (16-digit) vs second (10-digit) timestamps.
      - headers present vs headerless raw kline exports.
    Returns DataFrame with columns ['ts', 'open', 'high', 'low', 'close'] with ts in UTC.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Inspect first non-empty line to see if there is a header
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = ""
        for line in f:
            if line.strip() and not line.strip().startswith("#"):
                first_line = line.strip()
                break

    if not first_line:
        raise ValueError(f"File {file_path} contains no data")

    # If first token contains non-digits, it's a header
    first_token = re.split(r"[,;\s]+", first_line)[0].strip()
    has_header = not (first_token.isdigit() and len(first_token) >= 10)

    if has_header:
        df = pd.read_csv(path)
        col_map = {str(c).lower().strip(): c for c in df.columns}
        # Look for timestamp column
        ts_col = None
        for candidate in ["open_time", "opentime", "open time", "timestamp", "time", "ts"]:
            if candidate in col_map:
                ts_col = col_map[candidate]
                break
        if ts_col is None:
            ts_col = df.columns[0]

        open_col = col_map.get("open", df.columns[1])
        high_col = col_map.get("high", df.columns[2])
        low_col = col_map.get("low", df.columns[3])
        close_col = col_map.get("close", df.columns[4])
    else:
        df = pd.read_csv(path, header=None)
        ts_col = 0
        open_col = 1
        high_col = 2
        low_col = 3
        close_col = 4

    # Timestamp unit detection
    ts_numeric = pd.to_numeric(df[ts_col], errors="coerce")
    valid_ts = ts_numeric.dropna()
    if valid_ts.empty:
        raise ValueError(f"No valid numeric timestamps found in {file_path}")
    first_valid = float(valid_ts.iloc[0])

    if first_valid >= 1e14:
        unit = "us"
    elif first_valid >= 1e11:
        unit = "ms"
    else:
        unit = "s"

    ts_series = pd.to_datetime(ts_numeric, unit=unit, utc=True)

    result = pd.DataFrame({
        "ts": ts_series,
        "open": pd.to_numeric(df[open_col], errors="coerce"),
        "high": pd.to_numeric(df[high_col], errors="coerce"),
        "low": pd.to_numeric(df[low_col], errors="coerce"),
        "close": pd.to_numeric(df[close_col], errors="coerce"),
    })
    return result


def parse_histdata(file_path: Union[str, Path], source_tz: str = "EST") -> pd.DataFrame:
    """Parse HistData ASCII M1 file.

    Features:
      - Supports semicolon or comma delimiters.
      - Supports 5, 6, and 7-column variants (including separate date and time).
      - Converts from source_tz (default 'EST' fixed UTC-5 without DST) to UTC.
    Returns DataFrame with columns ['ts', 'open', 'high', 'low', 'close'] with ts in UTC.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Inspect first non-empty lines to detect delimiter and header
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = ""
        for line in f:
            if line.strip() and not line.strip().startswith("#"):
                first_line = line.strip()
                break

    if not first_line:
        raise ValueError(f"File {file_path} contains no data")

    sep = ";" if ";" in first_line else ","

    # Check if first line is a header: any letters in first two fields
    tokens = [t.strip() for t in first_line.split(sep)]
    has_header = any(re.search(r"[a-zA-Z]", t) for t in tokens[:2])

    if has_header:
        df = pd.read_csv(path, sep=sep)
        col_map = {str(c).lower().strip(): c for c in df.columns}
        if "datetime" in col_map:
            dt_series = df[col_map["datetime"]].astype(str)
        elif "date_time" in col_map:
            dt_series = df[col_map["date_time"]].astype(str)
        elif "date" in col_map and "time" in col_map:
            dt_series = df[col_map["date"]].astype(str).str.strip() + " " + df[col_map["time"]].astype(str).str.strip()
        elif "timestamp" in col_map:
            dt_series = df[col_map["timestamp"]].astype(str)
        elif "ts" in col_map:
            dt_series = df[col_map["ts"]].astype(str)
        else:
            dt_series = df[df.columns[0]].astype(str)

        open_col = col_map.get("open", df.columns[1])
        high_col = col_map.get("high", df.columns[2])
        low_col = col_map.get("low", df.columns[3])
        close_col = col_map.get("close", df.columns[4])
        open_series = df[open_col]
        high_series = df[high_col]
        low_series = df[low_col]
        close_series = df[close_col]
    else:
        df = pd.read_csv(path, sep=sep, header=None)
        num_cols = len(df.columns)
        # Check if Col 0 is date and Col 1 is time
        col0_sample = str(df.iloc[0, 0]).strip().split(".")[0]
        col1_sample = str(df.iloc[0, 1]).strip().split(".")[0] if num_cols > 1 else ""

        is_split_datetime = (
            num_cols >= 6 and
            (
                ":" in col1_sample or
                (len(col0_sample) == 8 and len(col1_sample) in (4, 6) and col0_sample.isdigit() and col1_sample.isdigit())
            )
        )

        if is_split_datetime:
            # Col 0: date, Col 1: time, Col 2: open, Col 3: high, Col 4: low, Col 5: close
            date_col = df[0].astype(str).str.strip().str.split(".").str[0]
            time_col = df[1].astype(str).str.strip().str.split(".").str[0].str.zfill(6)
            dt_series = date_col + " " + time_col
            open_series = df[2]
            high_series = df[3]
            low_series = df[4]
            close_series = df[5]
        else:
            # Col 0 is combined datetime
            dt_series = df[0].astype(str).str.strip()
            open_series = df[1]
            high_series = df[2]
            low_series = df[3]
            close_series = df[4]

    # Parse datetime and convert to UTC
    tz = parse_tz(source_tz)
    dt_parsed = pd.to_datetime(dt_series, format="mixed")
    if dt_parsed.dt.tz is None:
        utc_ts = dt_parsed.dt.tz_localize(tz).dt.tz_convert(timezone.utc)
    else:
        utc_ts = dt_parsed.dt.tz_convert(timezone.utc)

    result = pd.DataFrame({
        "ts": utc_ts,
        "open": pd.to_numeric(open_series, errors="coerce"),
        "high": pd.to_numeric(high_series, errors="coerce"),
        "low": pd.to_numeric(low_series, errors="coerce"),
        "close": pd.to_numeric(close_series, errors="coerce"),
    })
    return result


def validate_candles(
    df: pd.DataFrame,
    drop_invalid: bool = False,
) -> tuple[pd.DataFrame, int, list[dict]]:
    """Validate price relationships and positive prices for all candles.

    Rules:
      - high >= max(open, close)
      - low <= min(open, close)
      - open > 0, high > 0, low > 0, close > 0 (no negative or zero prices)
      - no NaN / null values in ts, open, high, low, close

    If drop_invalid=False, raises ValueError with count and the first 10 offending rows.
    If drop_invalid=True, drops invalid rows, prints the dropped count, and returns (cleaned_df, count, first_10).
    """
    if df.empty:
        return df, 0, []

    invalid_mask = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
        | df[["ts", "open", "high", "low", "close"]].isna().any(axis=1)
    )

    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        first_10 = df[invalid_mask].head(10).to_dict(orient="records")
        rows_str = "\n".join(
            f"  {idx + 1:2d}. ts={r.get('ts')} open={r.get('open')} high={r.get('high')} low={r.get('low')} close={r.get('close')}"
            for idx, r in enumerate(first_10)
        )
        msg = (
            f"Validation failed: found {invalid_count} invalid candle row(s) violating price rules "
            f"(high >= max(open,close), low <= min(open,close), prices > 0).\n"
            f"First {len(first_10)} offending row(s):\n{rows_str}"
        )
        if not drop_invalid:
            raise ValueError(msg)
        else:
            print(f"Dropped {invalid_count} invalid candle row(s) failing price rules.")
            logger.warning(msg)
            df = df[~invalid_mask].copy().reset_index(drop=True)
            return df, invalid_count, first_10

    return df, 0, []


def generate_gap_report(
    df: pd.DataFrame,
    asset: str,
    config: Optional[AppConfig] = None,
) -> tuple[str, list[tuple[datetime, datetime]]]:
    """Scan DataFrame for missing intervals per asset calendar rules and format report.

    Expected calendar closures (e.g., forex weekends) are suppressed and not flagged as gaps.
    Returns (report_text, list_of_gaps).
    """
    if config is None:
        try:
            config = load_config("config.yaml")
        except Exception:
            config = None

    calendar = None
    cal_name = "always_open (default)"
    if config is not None and hasattr(config, "data"):
        cal_key = config.data.assets.get(asset)
        if cal_key and cal_key in config.data.calendars:
            calendar = config.data.calendars[cal_key]
            cal_name = f"{cal_key} (type: {calendar.type})"

    gaps = detect_gaps(df, timeframe="1m", calendar=calendar)

    total_bars = len(df)
    min_ts = df["ts"].min() if not df.empty else "N/A"
    max_ts = df["ts"].max() if not df.empty else "N/A"

    lines = [
        "=" * 72,
        f"CALENDAR GAP REPORT: {asset}",
        "=" * 72,
        f"Calendar:    {cal_name}",
        f"Total bars:  {total_bars:,}",
        f"Start (UTC): {min_ts}",
        f"End   (UTC): {max_ts}",
    ]

    if not gaps:
        lines.extend([
            "Status:      CLEAN (0 unexpected market gaps)",
            "Note:        Expected calendar closures (e.g. weekends) are excluded.",
        ])
    else:
        total_gap_minutes = sum(int((end - start).total_seconds() / 60) - 1 for start, end in gaps)
        lines.extend([
            f"Status:      {len(gaps)} UNEXPECTED GAP(S) DETECTED ({total_gap_minutes:,} total missing minutes)",
            "Note:        Expected calendar closures (e.g. weekends) are excluded.",
            "",
            "Gap details (showing up to 20):",
        ])
        for idx, (p_dt, c_dt) in enumerate(gaps[:20], 1):
            missing_mins = int((c_dt - p_dt).total_seconds() / 60) - 1
            lines.append(f"  {idx:2d}. {p_dt.isoformat()} -> {c_dt.isoformat()} ({missing_mins} missing minutes)")
        if len(gaps) > 20:
            lines.append(f"  ... and {len(gaps) - 20} more gap(s)")

    lines.append("=" * 72)
    return "\n".join(lines), gaps


def convert_history(
    input_files: Union[str, Path, list[Union[str, Path]]],
    asset: str,
    output_dir: Union[str, Path] = "data_history",
    fmt: str = "auto",
    source_tz: str = "EST",
    config_path: Union[str, Path] = "config.yaml",
    drop_invalid: bool = False,
    ts_is_close: bool = False,
    print_report: bool = True,
) -> tuple[Path, pd.DataFrame, str]:
    """Convert input 1-minute historical data files into data_history/<ASSET>.csv in UTC.

    Returns (output_path, clean_df, gap_report).
    """
    if isinstance(input_files, (str, Path)):
        raw_inputs = [input_files]
    else:
        raw_inputs = list(input_files)

    # Expand any glob patterns
    resolved_paths: list[Path] = []
    for item in raw_inputs:
        item_str = str(item)
        if any(c in item_str for c in "*?[]"):
            matched = [Path(p) for p in sorted(glob.glob(item_str))]
            if not matched:
                raise FileNotFoundError(f"Glob pattern matched no files: {item_str}")
            resolved_paths.extend(matched)
        else:
            p = Path(item)
            if not p.is_file():
                raise FileNotFoundError(f"Input file not found: {p}")
            resolved_paths.append(p)

    if not resolved_paths:
        raise ValueError("No input files provided")

    dfs: list[pd.DataFrame] = []
    for p in resolved_paths:
        file_fmt = fmt.lower()
        if file_fmt == "auto":
            file_fmt = detect_format(p)

        if file_fmt == "binance":
            df = parse_binance(p)
        elif file_fmt == "histdata":
            df = parse_histdata(p, source_tz=source_tz)
        else:
            raise ValueError(f"Unsupported format: '{fmt}'. Expected 'auto', 'binance', or 'histdata'.")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    if combined.empty:
        raise ValueError("Input files contained no candle rows")

    rows_read = len(combined)

    # If timestamps represent close time, shift back by 1 minute to represent open time
    if ts_is_close:
        combined["ts"] = combined["ts"] - timedelta(minutes=1)

    # Sort chronologically by ts
    combined = combined.sort_values("ts").reset_index(drop=True)

    # Count duplicate timestamps and differing price duplicates
    total_dups = int(combined.duplicated(subset=["ts"], keep="first").sum())
    exact_dups = int(combined.duplicated(subset=["ts", "open", "high", "low", "close"], keep="first").sum())
    diff_price_dups = total_dups - exact_dups

    if print_report:
        if total_dups > 0:
            print(
                f"Duplicate timestamps: {total_dups} "
                f"({diff_price_dups} with different prices, keeping first row; {exact_dups} identical)"
            )
        else:
            print("Duplicate timestamps: 0")

    # Keep first row for each timestamp
    combined = combined.drop_duplicates(subset=["ts"], keep="first").reset_index(drop=True)

    # Validate price relationships
    combined, invalid_count, _ = validate_candles(combined, drop_invalid=drop_invalid)

    # Load config for gap detection
    cfg: Optional[AppConfig] = None
    try:
        cfg = load_config(config_path)
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")

    # Generate gap report (using datetime ts before string formatting)
    report, gaps = generate_gap_report(combined, asset=asset, config=cfg)
    gap_count = len(gaps)

    if print_report:
        print(report)

    # Format output DataFrame with ISO 8601 UTC ts
    output_df = combined.copy()
    output_df["ts"] = output_df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    output_df = output_df[["ts", "open", "high", "low", "close"]]

    # Write to target directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{asset.upper()}.csv"
    output_df.to_csv(out_file, index=False)

    # Final one-line summary
    min_ts_str = output_df["ts"].min() if not output_df.empty else "N/A"
    max_ts_str = output_df["ts"].max() if not output_df.empty else "N/A"
    summary_line = (
        f"Summary: rows_read={rows_read:,}, duplicates={total_dups:,} (different_prices={diff_price_dups:,}), "
        f"invalid={invalid_count:,}, rows_written={len(output_df):,}, "
        f"date_range={min_ts_str} -> {max_ts_str}, gaps={gap_count:,}"
    )
    if print_report:
        print(summary_line)

    return out_file, output_df, report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert 1-minute historical data (Binance or HistData) to data_history/<ASSET>.csv in UTC."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input historical data file(s) or glob pattern(s)",
    )
    parser.add_argument(
        "--asset",
        "-a",
        required=True,
        help="Asset symbol (e.g., EURUSD, BTCUSD)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["auto", "binance", "histdata"],
        default="auto",
        help="Input format (default: auto)",
    )
    parser.add_argument(
        "--source-tz",
        default="EST",
        help="Source timezone for HistData / ASCII files (default: EST, fixed UTC-5 without DST)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data_history",
        help="Output directory (default: data_history)",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--drop-invalid",
        action="store_true",
        help="Drop invalid candle rows instead of failing validation",
    )
    parser.add_argument(
        "--ts-is-close",
        action="store_true",
        default=False,
        help="Source timestamps are candle close times; shift back 1 minute to represent open time (default: false)",
    )

    args = parser.parse_args(argv)

    try:
        out_file, df, _ = convert_history(
            input_files=args.input_files,
            asset=args.asset,
            output_dir=args.output_dir,
            fmt=args.format,
            source_tz=args.source_tz,
            config_path=args.config,
            drop_invalid=args.drop_invalid,
            ts_is_close=args.ts_is_close,
            print_report=True,
        )
        print(f"\nSuccessfully wrote {len(df):,} bars to {out_file}")
        return 0
    except Exception as e:
        print(f"Error converting history: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
