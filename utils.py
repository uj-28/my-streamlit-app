from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config import CACHE_DIR, INDIA_TZ, MAX_NULL_PCT, MIN_AVG_VOLUME, MIN_HISTORY_BARS


def configure_logging(log_name: str = "scanner.log") -> None:
    """Configure file logging once per session."""
    CACHE_DIR.mkdir(exist_ok=True)
    log_path = CACHE_DIR / log_name
    if getattr(configure_logging, "_configured", False):
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=str(log_path),
    )
    configure_logging._configured = True
    logging.info("Logging initialized at %s", datetime.now().isoformat())


def ensure_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
    """Normalize yfinance output to Series for indicator calculations."""
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(index=obj.index, dtype="float64")
        return obj.iloc[:, 0]
    return obj


def normalize_timezone(index: pd.DatetimeIndex, tz: ZoneInfo = INDIA_TZ) -> pd.DatetimeIndex:
    """Ensure index is timezone-aware and converted to target timezone."""
    idx = pd.to_datetime(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(tz)


def validate_ohlcv(df: pd.DataFrame, min_rows: int = MIN_HISTORY_BARS) -> bool:
    """Basic OHLCV validation to avoid corrupt data."""
    if df is None or df.empty:
        return False
    if len(df) < min_rows:
        return False
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            return False
        null_pct = float(df[col].isna().mean())
        if null_pct > MAX_NULL_PCT:
            return False
    return True


def pre_filter_ohlcv(df: pd.DataFrame, min_rows: int = MIN_HISTORY_BARS) -> bool:
    """Fast pre-filter to skip illiquid or short-history symbols."""
    if not validate_ohlcv(df, min_rows=min_rows):
        return False
    avg_volume = float(df["Volume"].tail(20).mean())
    return avg_volume >= MIN_AVG_VOLUME


def bool_to_text(value: object) -> str:
    if pd.isna(value):
        return "-"
    return "YES" if bool(value) else "NO"

