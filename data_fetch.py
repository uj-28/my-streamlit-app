from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    CACHE_DIR,
    CACHE_INTERVALS,
    INDIA_TZ,
    NSE_INDEX_BASE_URL,
    NSE_INDEX_CSV_FILES,
    SYMBOL_COLUMN_CANDIDATES,
    YF_BATCH_SIZE,
    YF_MAX_WORKERS,
    BANK_NIFTY_FALLBACK,
    NIFTY_50_FALLBACK,
    NIFTY_IT_FALLBACK,
    NIFTY_MIDCAP_50_FALLBACK,
    NIFTY_SMALLCAP_50_FALLBACK,
    NIFTY_750_FALLBACK,
    SENSEX_30_FALLBACK,
)
from utils import normalize_timezone


def set_yf_cache() -> None:
    """Set local writable cache directory to avoid yfinance sqlite path issues."""
    cache_dir = CACHE_DIR / "yf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))


def indicator_cache_path(
    symbol: str,
    interval: str,
    resample_rule: str | None,
    ema_period: int,
    rsi_length: int,
    atr_period: int,
    atr_multiplier: float,
    halftrend_amplitude: int,
    halftrend_channel_deviation: int,
    show_halftrend: bool,
) -> Path:
    """Stable path for per-symbol indicator cache."""
    safe_symbol = symbol.replace("/", "_")
    key = (
        f"{interval}|{resample_rule}|{ema_period}|{rsi_length}|{atr_period}|"
        f"{atr_multiplier}|{halftrend_amplitude}|{halftrend_channel_deviation}|{show_halftrend}"
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / "indicators" / interval / safe_symbol / f"ind_{digest}.parquet"


def read_indicator_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception:
        return None


def write_indicator_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def fetch_csv_from_url(url: str) -> pd.DataFrame | None:
    """Fetch CSV with browser-like headers; return None on network/parse failure."""
    try:
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/csv,*/*;q=0.9",
            },
        )
        with urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        return pd.read_csv(StringIO(raw))
    except (URLError, HTTPError, TimeoutError, ValueError):
        return None


def to_nse_ticker(symbol: str) -> str:
    sym = symbol.strip().upper()
    if not sym:
        return ""
    return sym if sym.endswith(".NS") else f"{sym}.NS"


def load_nse_index_constituents(universe: str) -> list[str]:
    csv_file = NSE_INDEX_CSV_FILES.get(universe)
    if not csv_file:
        return []

    df = fetch_csv_from_url(f"{NSE_INDEX_BASE_URL}{csv_file}")
    if df is None or df.empty:
        return []

    symbol_col = next((c for c in df.columns if str(c).strip().lower() == "symbol"), None)
    if symbol_col is None:
        return []

    out: list[str] = []
    for raw_symbol in df[symbol_col].astype(str):
        ticker = to_nse_ticker(raw_symbol)
        if ticker:
            out.append(ticker)
    return list(dict.fromkeys(out))


def load_symbols_from_file(uploaded_file) -> list[str]:
    """Load symbols from uploaded CSV/XLSX and normalize to NSE tickers."""
    if uploaded_file is None:
        return []

    filename = (uploaded_file.name or "").lower()
    try:
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except ImportError:
        st.error("Excel upload requires the 'openpyxl' package. Please upload a CSV or install openpyxl.")
        return []

    if df.empty:
        return []

    symbol_col = next(
        (c for c in df.columns if str(c).strip().lower() in SYMBOL_COLUMN_CANDIDATES),
        None,
    )
    if symbol_col is None:
        symbol_col = df.columns[0]

    out: list[str] = []
    for raw_symbol in df[symbol_col].astype(str):
        ticker = to_nse_ticker(raw_symbol)
        if ticker:
            out.append(ticker)
    return list(dict.fromkeys(out))


@st.cache_data(ttl=60 * 60, show_spinner=False)
def resolve_universe(universe: str) -> tuple[list[str], str]:
    """Resolve universe constituents with NSE CSVs + fallback lists."""
    live = load_nse_index_constituents(universe)
    if live:
        return live, "NSE live constituents"

    if universe == "NIFTY 50":
        return NIFTY_50_FALLBACK, "Fallback list"
    if universe == "NIFTY IT":
        return NIFTY_IT_FALLBACK, "Fallback list"
    if universe == "NIFTY MIDCAP 50":
        return NIFTY_MIDCAP_50_FALLBACK, "Fallback list"
    if universe == "NIFTY SMALLCAP 50":
        return NIFTY_SMALLCAP_50_FALLBACK, "Fallback list"
    if universe == "NIFTY 750":
        return NIFTY_750_FALLBACK, "Fallback list"
    if universe == "BANK NIFTY":
        return BANK_NIFTY_FALLBACK, "Fallback list"
    if universe == "SENSEX 30":
        return SENSEX_30_FALLBACK, "Fallback list"

    return [], "Unavailable (could not load index constituents)"


def resample_ohlcv(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe."""
    if data.empty:
        return data
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = data.resample(rule, label="right", closed="right").agg(agg)
    return out.dropna()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_bulk_history(
    symbols: tuple[str, ...],
    interval: str,
    period: str,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Download OHLCV data for multiple symbols with disk cache fallback."""
    if not symbols:
        return pd.DataFrame()
    symbols_list = list(symbols)

    def normalize_batch(df: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            return df
        if len(batch) == 1:
            symbol = batch[0]
            df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
        return df

    def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if not cols:
            return pd.DataFrame(index=df.index)
        out = df[cols].copy()
        out.index = normalize_timezone(out.index)
        return out.sort_index()

    def cache_path(symbol: str, tf_interval: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return CACHE_DIR / tf_interval / f"{safe_symbol}.parquet"

    def read_cache(symbol: str, tf_interval: str) -> pd.DataFrame | None:
        path = cache_path(symbol, tf_interval)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            return sanitize_ohlcv(df)
        except Exception:
            return None

    def write_cache(symbol: str, tf_interval: str, df: pd.DataFrame) -> None:
        path = cache_path(symbol, tf_interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def merge_frames(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
        if existing is None or existing.empty:
            return incoming.copy()
        if incoming.empty:
            return existing.copy()
        merged = pd.concat([existing, incoming]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        return merged

    def fetch_batch(batch: list[str], rng_start=None, rng_end=None, use_period: bool = False) -> pd.DataFrame:
        try:
            if use_period:
                data = yf.download(
                    batch,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    actions=False,
                    threads=True,
                    progress=False,
                )
            else:
                data = yf.download(
                    batch,
                    start=rng_start,
                    end=rng_end,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    actions=False,
                    threads=True,
                    progress=False,
                )
            return normalize_batch(data, batch)
        except Exception as exc:
            logging.warning("Batch download failed: %s", str(exc))
            return pd.DataFrame()

    # Fallback to direct yfinance for unsupported intervals.
    if interval not in CACHE_INTERVALS:
        batches = [symbols_list[i : i + YF_BATCH_SIZE] for i in range(0, len(symbols_list), YF_BATCH_SIZE)]
        if len(batches) == 1:
            return fetch_batch(batches[0], rng_start=start, rng_end=end, use_period=start is None and end is None)

        results: list[pd.DataFrame] = []
        max_workers = min(YF_MAX_WORKERS, len(batches))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(fetch_batch, batch, start, end, start is None and end is None)
                for batch in batches
            ]
            for fut in as_completed(futures):
                df = fut.result()
                if not df.empty:
                    results.append(df)
        if not results:
            return pd.DataFrame()
        if len(results) == 1:
            return results[0]
        return pd.concat(results, axis=1)

    CACHE_DIR.mkdir(exist_ok=True)
    today = datetime.now(INDIA_TZ).date()
    cached_frames: dict[str, pd.DataFrame] = {}
    fetch_groups: dict[tuple[str | None, str | None, bool], list[str]] = {}

    for symbol in symbols_list:
        cached = read_cache(symbol, interval)
        if cached is None or cached.empty:
            key = (start if start is not None else None, end if end is not None else None, True)
            fetch_groups.setdefault(key, []).append(symbol)
            continue

        cached_frames[symbol] = cached
        cached_start = cached.index.min()
        cached_end = cached.index.max()
        if cached_start is None or cached_end is None:
            key = (start if start is not None else None, end if end is not None else None, True)
            fetch_groups.setdefault(key, []).append(symbol)
            continue

        # Determine missing ranges.
        missing_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if start is not None:
            start_ts = pd.Timestamp(start)
            if cached_start > start_ts:
                missing_ranges.append((start_ts, cached_start - pd.Timedelta(days=1)))
        if end is not None:
            end_ts = pd.Timestamp(end)
            if cached_end < end_ts:
                missing_ranges.append((cached_end + pd.Timedelta(days=1), end_ts))
        if start is None and end is None:
            if cached_end.date() < today:
                missing_ranges.append((cached_end + pd.Timedelta(days=1), pd.Timestamp(today)))

        if not missing_ranges:
            continue

        for rng_start, rng_end in missing_ranges:
            key = (rng_start, rng_end, False)
            fetch_groups.setdefault(key, []).append(symbol)

    # Fetch missing data in grouped batches.
    for (rng_start, rng_end, use_period), batch in fetch_groups.items():
        if not batch:
            continue
        if use_period:
            data = fetch_batch(batch, use_period=True)
        else:
            data = fetch_batch(batch, rng_start=rng_start, rng_end=rng_end, use_period=False)
        if data.empty:
            continue
        for symbol in batch:
            if not isinstance(data.columns, pd.MultiIndex):
                sym_df = data.copy()
            else:
                if symbol not in data.columns.get_level_values(0):
                    continue
                sym_df = data[symbol].dropna(how="all")
            sym_df = sanitize_ohlcv(sym_df)
            if sym_df.empty:
                continue
            merged = merge_frames(cached_frames.get(symbol), sym_df)
            cached_frames[symbol] = merged
            write_cache(symbol, interval, merged)

    if not cached_frames:
        return pd.DataFrame()

    frames = []
    for symbol in symbols_list:
        df = cached_frames.get(symbol)
        if df is None or df.empty:
            continue
        df = df.copy()
        df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, axis=1)
