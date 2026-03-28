from __future__ import annotations

import re
from datetime import date, datetime
import logging

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from ai_model import (
    build_training_matrix,
    compute_feature_frame,
    get_feature_columns,
    load_ai_model,
    save_ai_model,
    save_versioned_model,
    score_to_signal,
    train_xgb_model,
    walk_forward_validate,
)
from config import AI_CONFIDENCE_DEFAULT, INDIA_TZ, NSE_MARKET_CLOSE
from data_fetch import (
    fetch_bulk_history,
    indicator_cache_path,
    read_indicator_cache,
    resample_ohlcv,
    write_indicator_cache,
)
from indicators import compute_halftrend, compute_rsi, compute_supertrend
from utils import ensure_series, pre_filter_ohlcv, validate_ohlcv


def get_signal_index(
    index: pd.Index,
    use_last_closed_candle: bool,
) -> pd.Timestamp | None:
    """
    Pick candle used for signal generation.
    - If enabled, during market hours use previous fully closed daily candle.
    - Otherwise use latest available daily candle.
    """
    if len(index) == 0:
        return None

    latest = pd.Timestamp(index[-1])
    if not use_last_closed_candle or len(index) < 2:
        return latest

    now_ist = datetime.now(INDIA_TZ)
    latest_date = latest.date()
    if latest_date == now_ist.date() and now_ist.time() < NSE_MARKET_CLOSE:
        return pd.Timestamp(index[-2])
    return latest


@st.cache_data(ttl=900, show_spinner=False)
def scan_universe(
    universe_name: str,
    symbols: tuple[str, ...],
    interval: str,
    period: str,
    resample_rule: str | None,
    timeframe_label: str,
    rsi_threshold: float,
    rsi_length: int,
    ema_period: int,
    ema_direction: str,
    atr_period: int,
    atr_multiplier: float,
    halftrend_amplitude: int,
    halftrend_channel_deviation: int,
    show_halftrend: bool,
    use_ema: bool,
    use_rsi: bool,
    use_supertrend: bool,
    use_last_closed_candle: bool,
    enable_ai: bool,
    ai_cache_key: str,
    ai_conf_threshold: float = AI_CONFIDENCE_DEFAULT,
) -> pd.DataFrame:
    results = []
    ema_is_above = ema_direction == "Above EMA"
    ema_op = ">" if ema_is_above else "<"
    cond_ema_col = f"Close {ema_op} EMA{ema_period}"
    cond_rsi_col = f"RSI({rsi_length}) > {rsi_threshold:g}"
    ai_feature_rows: dict[str, pd.Series] = {}
    training_X: list[pd.DataFrame] = []
    training_y: list[pd.Series] = []
    feature_cols = get_feature_columns()

    bulk_data = fetch_bulk_history(symbols, interval=interval, period=period)

    for symbol in symbols:
        try:
            if isinstance(bulk_data.columns, pd.MultiIndex):
                if symbol not in bulk_data.columns.get_level_values(0):
                    continue
                data = bulk_data[symbol].dropna(how="all")
            else:
                data = bulk_data.copy()
        except Exception as exc:
            logging.warning("Failed to read symbol %s: %s", symbol, str(exc))
            continue

        if resample_rule:
            data = resample_ohlcv(data, resample_rule)

        if not validate_ohlcv(data):
            continue
        if not pre_filter_ohlcv(data):
            continue

        min_required = max(
            ema_period + 5,
            atr_period + 5,
            rsi_length + 20,
            105,
            halftrend_amplitude + 5,
        )
        if data.empty or len(data) < min_required:
            continue

        close = ensure_series(data["Close"]).astype("float64")
        high = ensure_series(data["High"]).astype("float64")
        low = ensure_series(data["Low"]).astype("float64")

        ohlc = pd.concat([high, low, close], axis=1, keys=["High", "Low", "Close"]).dropna()
        if len(ohlc) < min_required:
            continue

        high = ohlc["High"]
        low = ohlc["Low"]
        close = ohlc["Close"]
        symbol_clean = symbol.replace(".NS", "")

        ind_cache_path = indicator_cache_path(
            symbol=symbol,
            interval=interval,
            resample_rule=resample_rule,
            ema_period=ema_period,
            rsi_length=rsi_length,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            halftrend_amplitude=halftrend_amplitude,
            halftrend_channel_deviation=halftrend_channel_deviation,
            show_halftrend=show_halftrend,
        )
        ind_cache = read_indicator_cache(ind_cache_path)

        if ind_cache is not None and ind_cache.index.equals(ohlc.index):
            ema = ind_cache["ema"].astype("float64")
            rsi = ind_cache["rsi"].astype("float64")
            st_dir = ind_cache["st_dir"].astype("float64")
            if show_halftrend:
                ht_buy = ind_cache["ht_buy"].astype("bool")
                ht_sell = ind_cache["ht_sell"].astype("bool")
            else:
                ht_buy = pd.Series(False, index=close.index, dtype="bool")
                ht_sell = pd.Series(False, index=close.index, dtype="bool")
        else:
            ema = close.ewm(span=ema_period, adjust=False).mean()
            rsi = compute_rsi(close, period=rsi_length)
            _, st_dir = compute_supertrend(
                high=high,
                low=low,
                close=close,
                atr_period=atr_period,
                multiplier=atr_multiplier,
            )
            ht_buy = pd.Series(False, index=close.index, dtype="bool")
            ht_sell = pd.Series(False, index=close.index, dtype="bool")
            if show_halftrend:
                _, ht_buy, ht_sell, _ = compute_halftrend(
                    high=high,
                    low=low,
                    close=close,
                    amplitude=halftrend_amplitude,
                    channel_deviation=halftrend_channel_deviation,
                    atr_period=100,
                )

            cache_frame = pd.DataFrame(
                {
                    "ema": ema,
                    "rsi": rsi,
                    "st_dir": st_dir,
                    "ht_buy": ht_buy.astype("bool"),
                    "ht_sell": ht_sell.astype("bool"),
                },
                index=close.index,
            )
            write_indicator_cache(ind_cache_path, cache_frame)

        signal_idx = get_signal_index(
            close.index,
            use_last_closed_candle=use_last_closed_candle,
        )
        if signal_idx is None:
            continue

        if enable_ai:
            end_date = pd.Timestamp(ohlc.index[-1])
            start_date = end_date - pd.DateOffset(years=2)
            data_ai = data.loc[data.index >= start_date].copy()

            X_sym, y_sym = build_training_matrix(
                data=data_ai,
                rsi_length=rsi_length,
                atr_period=atr_period,
                atr_multiplier=atr_multiplier,
            )
            if not X_sym.empty and not y_sym.empty:
                training_X.append(X_sym)
                training_y.append(y_sym)

            features_sym = compute_feature_frame(
                data=data_ai,
                rsi_length=rsi_length,
                atr_period=atr_period,
                atr_multiplier=atr_multiplier,
            )
            if not features_sym.empty and signal_idx in features_sym.index:
                ai_feature_rows[symbol_clean] = features_sym.loc[signal_idx]

        latest_close = float(close.loc[signal_idx])
        latest_ema = float(ema.loc[signal_idx])
        latest_rsi = float(rsi.loc[signal_idx]) if pd.notna(rsi.loc[signal_idx]) else None
        latest_st_dir = int(st_dir.loc[signal_idx]) if pd.notna(st_dir.loc[signal_idx]) else -1
        latest_ht_buy = bool(ht_buy.loc[signal_idx]) if pd.notna(ht_buy.loc[signal_idx]) else False
        latest_ht_sell = bool(ht_sell.loc[signal_idx]) if pd.notna(ht_sell.loc[signal_idx]) else False

        ema_cond = latest_close > latest_ema if ema_is_above else latest_close < latest_ema
        rsi_cond = latest_rsi is not None and latest_rsi > rsi_threshold
        supertrend_cond = latest_st_dir == 1
        ema_ok = ema_cond if use_ema else True
        rsi_ok = rsi_cond if use_rsi else True
        supertrend_ok = supertrend_cond if use_supertrend else True

        results.append(
            {
                "Universe": universe_name,
                "Symbol": symbol_clean,
                "Close": round(latest_close, 2),
                f"EMA{ema_period}": round(latest_ema, 2),
                "RSI": round(latest_rsi, 2) if latest_rsi is not None else None,
                "Supertrend": "Green" if supertrend_cond else "Red",
                "HalfTrend Signal": "Buy" if latest_ht_buy else "Sell" if latest_ht_sell else "Neutral",
                "EMA Filter Enabled": use_ema,
                "RSI Filter Enabled": use_rsi,
                "Supertrend Filter Enabled": use_supertrend,
                cond_ema_col: ema_cond if use_ema else pd.NA,
                cond_rsi_col: rsi_cond if use_rsi else pd.NA,
                "Supertrend Green": supertrend_cond if use_supertrend else pd.NA,
                "Pass": ema_ok and rsi_ok and supertrend_ok,
                "Signal Candle": signal_idx.date().isoformat(),
                "Latest Available Candle": pd.Timestamp(close.index[-1]).date().isoformat(),
                "Candle Mode": "Last Closed" if use_last_closed_candle else "Live",
                "Timeframe": timeframe_label,
            }
        )

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    if enable_ai:
        model = None
        cached_meta = st.session_state.get("ai_model_meta", {})
        if st.session_state.get("ai_model") is not None and cached_meta.get("key") == ai_cache_key:
            model = st.session_state.get("ai_model")
        else:
            model = load_ai_model(ai_cache_key)
            if model is None:
                if training_X and training_y:
                    X_train = pd.concat(training_X, ignore_index=True)
                    y_train = pd.concat(training_y, ignore_index=True)
                else:
                    X_train = pd.DataFrame()
                    y_train = pd.Series(dtype="int64")

                wf_scores = walk_forward_validate(X_train, y_train)
                wf_mean = float(np.mean(wf_scores)) if wf_scores else None
                wf_std = float(np.std(wf_scores)) if wf_scores else None

                with st.spinner("Training AI model (XGBoost)..."):
                    model = train_xgb_model(X_train, y_train)

                metadata = {
                    "key": ai_cache_key,
                    "rows": int(len(X_train)),
                    "symbols": int(len(symbols)),
                    "walk_forward_mean": wf_mean,
                    "walk_forward_std": wf_std,
                    "features": feature_cols,
                }
                save_ai_model(ai_cache_key, model, metadata=metadata)
                save_versioned_model(model, metadata)
                st.session_state["ai_model_meta"] = metadata

            st.session_state["ai_model"] = model
            if "ai_model_meta" not in st.session_state:
                st.session_state["ai_model_meta"] = {"key": ai_cache_key}

        ai_probs_pct = []
        ai_signals = []
        ai_confident = []
        ai_dist_ma50 = []
        ai_dist_ma200 = []
        ai_vol_ratio = []
        ai_atr_pct = []
        ai_st_dir = []
        ai_adx = []
        ai_ema_dist = []
        ai_52w_high = []
        ai_52w_low = []

        with st.spinner("Scoring AI probabilities..."):
            probs = np.full(len(out), 0.5, dtype="float64")
            if model is not None:
                feature_matrix: list[np.ndarray] = []
                valid_indices: list[int] = []
                for idx, row in out.iterrows():
                    feat = ai_feature_rows.get(row["Symbol"])
                    if feat is None:
                        continue
                    feat_slice = feat[feature_cols]
                    if feat_slice.isna().any():
                        continue
                    feature_matrix.append(feat_slice.astype("float64").to_numpy())
                    valid_indices.append(int(idx))

                if feature_matrix:
                    X_all = np.vstack(feature_matrix)
                    dtest = xgb.DMatrix(X_all)
                    preds = model.predict(dtest)
                    for i, prob in zip(valid_indices, preds):
                        probs[i] = float(prob)

            for idx, row in out.iterrows():
                feat = ai_feature_rows.get(row["Symbol"])
                prob = float(probs[int(idx)])
                ai_probs_pct.append(round(prob * 100.0))
                ai_signals.append(score_to_signal(prob))
                ai_confident.append(prob >= ai_conf_threshold)
                ai_dist_ma50.append(
                    round(float(feat["dist_ma50"]), 2) if feat is not None and pd.notna(feat.get("dist_ma50")) else None
                )
                ai_dist_ma200.append(
                    round(float(feat["dist_ma200"]), 2) if feat is not None and pd.notna(feat.get("dist_ma200")) else None
                )
                ai_vol_ratio.append(
                    round(float(feat["vol_ratio"]), 2) if feat is not None and pd.notna(feat.get("vol_ratio")) else None
                )
                ai_atr_pct.append(
                    round(float(feat["atr_pct"]), 2) if feat is not None and pd.notna(feat.get("atr_pct")) else None
                )
                ai_st_dir.append(
                    int(feat["supertrend_dir"]) if feat is not None and pd.notna(feat.get("supertrend_dir")) else None
                )
                ai_adx.append(
                    round(float(feat["adx"]), 2) if feat is not None and pd.notna(feat.get("adx")) else None
                )
                ai_ema_dist.append(
                    round(float(feat["ema_dist_pct"]), 2)
                    if feat is not None and pd.notna(feat.get("ema_dist_pct"))
                    else None
                )
                ai_52w_high.append(
                    round(float(feat["pct_from_52w_high"]), 2)
                    if feat is not None and pd.notna(feat.get("pct_from_52w_high"))
                    else None
                )
                ai_52w_low.append(
                    round(float(feat["pct_from_52w_low"]), 2)
                    if feat is not None and pd.notna(feat.get("pct_from_52w_low"))
                    else None
                )

        out["AI Probability %"] = ai_probs_pct
        out["AI Signal"] = ai_signals
        out["AI Confident"] = ai_confident
        out["Dist MA50 %"] = ai_dist_ma50
        out["Dist MA200 %"] = ai_dist_ma200
        out["Volume Ratio"] = ai_vol_ratio
        out["ATR %"] = ai_atr_pct
        out["Supertrend Signal"] = ai_st_dir
        out["ADX"] = ai_adx
        out["EMA Dist %"] = ai_ema_dist
        out["52W High %"] = ai_52w_high
        out["52W Low %"] = ai_52w_low

    out["RSI"] = pd.to_numeric(out["RSI"], errors="coerce")
    out.sort_values(["Pass", "RSI"], ascending=[False, False], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def backtest_universe(
    universe_name: str,
    symbols: tuple[str, ...],
    interval: str,
    period: str,
    timeframe_label: str,
    resample_rule: str | None,
    start_date: date | None,
    end_date: date | None,
    trade_direction: str,
    use_ema: bool,
    ema_period: int,
    ema_direction: str,
    use_rsi: bool,
    rsi_length: int,
    rsi_threshold: float,
    rsi_mode: str,
    use_supertrend: bool,
    supertrend_mode: str,
    atr_period: int,
    atr_multiplier: float,
    exit_mode: str,
    exit_indicator: str,
    target_pct: float,
    stop_pct: float,
    hold_candles: int,
    entry_expr: str | None,
    exit_expr: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades: list[dict] = []
    total_bars = 0
    total_entry_hits = 0
    symbols_used = 0
    symbols_skipped = 0

    if not symbols:
        return pd.DataFrame(), pd.DataFrame()

    bulk_data = fetch_bulk_history(symbols, interval=interval, period=period, start=start_date, end=end_date)
    ema_is_above = ema_direction == "Above EMA"

    for symbol in symbols:
        try:
            if isinstance(bulk_data.columns, pd.MultiIndex):
                if symbol not in bulk_data.columns.get_level_values(0):
                    continue
                data = bulk_data[symbol].dropna(how="all")
            else:
                data = bulk_data.copy()
        except Exception:
            continue

        if data.empty:
            symbols_skipped += 1
            continue

        if resample_rule:
            data = resample_ohlcv(data, resample_rule)

        if start_date is not None:
            data = data.loc[data.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            data = data.loc[data.index <= pd.Timestamp(end_date)]

        min_required = max(ema_period + 5, atr_period + 5, rsi_length + 20, 105)
        if data.empty or len(data) < min_required + 2:
            symbols_skipped += 1
            continue

        close = ensure_series(data["Close"]).astype("float64")
        high = ensure_series(data["High"]).astype("float64")
        low = ensure_series(data["Low"]).astype("float64")

        ohlc = pd.concat([high, low, close], axis=1, keys=["High", "Low", "Close"]).dropna()
        if len(ohlc) < min_required + 2:
            symbols_skipped += 1
            continue

        high = ohlc["High"]
        low = ohlc["Low"]
        close = ohlc["Close"]

        indicator_cache: dict[str, pd.Series] = {}

        def ema_fn(length: int) -> pd.Series:
            key = f"ema_{length}"
            if key not in indicator_cache:
                indicator_cache[key] = close.ewm(span=length, adjust=False).mean()
            return indicator_cache[key]

        def rsi_fn(length: int) -> pd.Series:
            key = f"rsi_{length}"
            if key not in indicator_cache:
                indicator_cache[key] = compute_rsi(close, period=length)
            return indicator_cache[key]

        def supertrend_fn(atr_len: int, mult: float) -> pd.Series:
            key = f"st_{atr_len}_{mult}"
            if key not in indicator_cache:
                _, st_local = compute_supertrend(
                    high=high,
                    low=low,
                    close=close,
                    atr_period=atr_len,
                    multiplier=mult,
                )
                indicator_cache[key] = st_local
            return indicator_cache[key]

        def to_series(value: pd.Series | float | int) -> pd.Series:
            if isinstance(value, pd.Series):
                return value
            return pd.Series(value, index=close.index, dtype="float64")

        def cross_above(a: pd.Series | float | int, b: pd.Series | float | int) -> pd.Series:
            a_s = to_series(a)
            b_s = to_series(b)
            return (a_s.shift(1) <= b_s.shift(1)) & (a_s > b_s)

        def cross_below(a: pd.Series | float | int, b: pd.Series | float | int) -> pd.Series:
            a_s = to_series(a)
            b_s = to_series(b)
            return (a_s.shift(1) >= b_s.shift(1)) & (a_s < b_s)

        ema = ema_fn(ema_period)
        rsi = rsi_fn(rsi_length)
        _, st_dir = compute_supertrend(
            high=high,
            low=low,
            close=close,
            atr_period=atr_period,
            multiplier=atr_multiplier,
        )

        def build_signal(expr: str) -> pd.Series:
            expr_clean = expr.strip()
            if not expr_clean:
                raise ValueError("Rule cannot be empty.")
            expr_clean = re.sub(r"\s+", " ", expr_clean)
            expr_clean = re.sub(
                r"([A-Za-z0-9_.()]+)\s+crosses\s+above\s+([A-Za-z0-9_.()]+)",
                r"cross_above(\1, \2)",
                expr_clean,
                flags=re.IGNORECASE,
            )
            expr_clean = re.sub(
                r"([A-Za-z0-9_.()]+)\s+crosses\s+below\s+([A-Za-z0-9_.()]+)",
                r"cross_below(\1, \2)",
                expr_clean,
                flags=re.IGNORECASE,
            )
            expr_clean = re.sub(r"\band\b", "&", expr_clean, flags=re.IGNORECASE)
            expr_clean = re.sub(r"\bor\b", "|", expr_clean, flags=re.IGNORECASE)
            expr_clean = re.sub(r"\s+and\s+", " & ", expr_clean, flags=re.IGNORECASE)
            expr_clean = re.sub(r"\s+or\s+", " | ", expr_clean, flags=re.IGNORECASE)
            expr_clean = re.sub(r"\bgreen\b", "1", expr_clean, flags=re.IGNORECASE)
            expr_clean = re.sub(r"\bred\b", "-1", expr_clean, flags=re.IGNORECASE)
            expr_clean = re.sub(r"\s*&\s*", " & ", expr_clean)
            expr_clean = re.sub(r"\s*\|\s*", " | ", expr_clean)
            if " & " in expr_clean or " | " in expr_clean:
                expr_clean = "(" + expr_clean.replace(" & ", ") & (").replace(" | ", ") | (") + ")"
            env = {
                "close": close,
                "open": ensure_series(data["Open"]).astype("float64"),
                "high": high,
                "low": low,
                "volume": ensure_series(data["Volume"]).astype("float64"),
                "ema": ema_fn,
                "rsi": rsi_fn,
                "supertrend": supertrend_fn,
                "cross_above": cross_above,
                "cross_below": cross_below,
            }
            try:
                result = eval(expr_clean, {"__builtins__": {}}, env)
            except Exception as exc:
                raise ValueError(f"Invalid rule: {exc}. Parsed: {expr_clean}") from exc
            if isinstance(result, pd.Series):
                return result.fillna(False).astype(bool)
            raise ValueError("Rule did not produce a series result.")

        if entry_expr:
            signal = build_signal(entry_expr)
            exit_signal = build_signal(exit_expr) if exit_expr else None
        else:
            ema_cond = close > ema if ema_is_above else close < ema

            if rsi_mode == "RSI > Threshold":
                rsi_cond = rsi > rsi_threshold
            elif rsi_mode == "RSI < Threshold":
                rsi_cond = rsi < rsi_threshold
            elif rsi_mode == "RSI Crosses Above":
                rsi_cond = (rsi.shift(1) <= rsi_threshold) & (rsi > rsi_threshold)
            else:
                rsi_cond = (rsi.shift(1) >= rsi_threshold) & (rsi < rsi_threshold)

            supertrend_cond = st_dir == (1 if supertrend_mode == "Green (Bullish)" else -1)

            signal = pd.Series(True, index=close.index)
            if use_ema:
                signal &= ema_cond
            if use_rsi:
                signal &= rsi_cond
            if use_supertrend:
                signal &= supertrend_cond
            exit_signal = None

        symbols_used += 1
        total_bars += int(len(signal))
        total_entry_hits += int(signal.fillna(False).sum())

        symbol_clean = symbol.replace(".NS", "")
        in_trade = False
        entry_idx = None
        entry_price = None
        entry_is_short = False

        for idx in signal.index:
            if not in_trade:
                if bool(signal.loc[idx]) and pd.notna(close.loc[idx]):
                    in_trade = True
                    entry_idx = idx
                    entry_price = float(close.loc[idx])
                    if trade_direction == "Auto (Supertrend)" and use_supertrend:
                        st_value = st_dir.loc[idx] if idx in st_dir.index else 1
                        entry_is_short = int(st_value) == -1
                    else:
                        entry_is_short = trade_direction == "Short"
                continue

            if entry_idx is None or entry_price is None:
                in_trade = False
                continue

            exit_now = False
            exit_reason = ""
            is_short = entry_is_short if trade_direction == "Auto (Supertrend)" else trade_direction == "Short"

            if exit_signal is not None and bool(exit_signal.loc[idx]):
                exit_now = True
                exit_reason = "Custom Exit"
            elif exit_mode == "Fixed Target/SL":
                if is_short:
                    curr_ret = (entry_price / float(close.loc[idx]) - 1.0) * 100.0
                else:
                    curr_ret = (float(close.loc[idx]) / entry_price - 1.0) * 100.0
                if curr_ret >= target_pct:
                    exit_now = True
                    exit_reason = f"Target {target_pct:g}%"
                elif curr_ret <= -abs(stop_pct):
                    exit_now = True
                    exit_reason = f"Stop {-abs(stop_pct):g}%"
            elif exit_mode == "Indicator Flip":
                if exit_indicator == "EMA":
                    exit_now = not bool(ema_cond.loc[idx])
                elif exit_indicator == "RSI":
                    exit_now = not bool(rsi_cond.loc[idx])
                else:
                    exit_now = not bool(supertrend_cond.loc[idx])
                exit_reason = f"{exit_indicator} Flip"
            else:
                if entry_idx is not None:
                    held = (signal.index.get_loc(idx) - signal.index.get_loc(entry_idx))
                    if held >= max(hold_candles, 1):
                        exit_now = True
                        exit_reason = f"Time {hold_candles} candles"

            if exit_now:
                exit_price = float(close.loc[idx])
                if is_short:
                    ret_pct = (entry_price / exit_price - 1.0) * 100.0
                else:
                    ret_pct = (exit_price / entry_price - 1.0) * 100.0
                trades.append(
                    {
                        "Universe": universe_name,
                        "Symbol": symbol_clean,
                        "Entry Date": pd.Timestamp(entry_idx).date().isoformat(),
                        "Exit Date": pd.Timestamp(idx).date().isoformat(),
                        "Entry Close": round(entry_price, 2),
                        "Exit Close": round(exit_price, 2),
                        "Return %": round(ret_pct, 2),
                        "Exit Reason": exit_reason,
                        "Timeframe": timeframe_label,
                        "Direction": "Short" if is_short else "Long",
                    }
                )
                in_trade = False
                entry_idx = None
                entry_price = None

    if not trades:
        stats = {
            "Total Signals": 0,
            "Win Rate %": 0.0,
            "Avg Return %": 0.0,
            "Median Return %": 0.0,
            "Best Return %": 0.0,
            "Worst Return %": 0.0,
            "Cumulative Return %": 0.0,
            "Raw Entry Signals": int(total_entry_hits),
            "Bars Scanned": int(total_bars),
            "Symbols Used": int(symbols_used),
            "Symbols Skipped": int(symbols_skipped),
        }
        stats_df = pd.DataFrame([stats])
        return stats_df, pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    trades_df.sort_values(["Entry Date", "Symbol"], inplace=True, ignore_index=True)

    stats = {
        "Total Signals": int(len(trades_df)),
        "Win Rate %": round(float((trades_df["Return %"] > 0).mean() * 100.0), 2),
        "Avg Return %": round(float(trades_df["Return %"].mean()), 2),
        "Median Return %": round(float(trades_df["Return %"].median()), 2),
        "Best Return %": round(float(trades_df["Return %"].max()), 2),
        "Worst Return %": round(float(trades_df["Return %"].min()), 2),
        "Cumulative Return %": round(float(((trades_df["Return %"] / 100.0 + 1.0).prod() - 1.0) * 100.0), 2),
        "Raw Entry Signals": int(total_entry_hits),
        "Bars Scanned": int(total_bars),
        "Symbols Used": int(symbols_used),
        "Symbols Skipped": int(symbols_skipped),
    }
    stats_df = pd.DataFrame([stats])
    return stats_df, trades_df

