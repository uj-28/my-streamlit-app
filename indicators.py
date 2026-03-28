from __future__ import annotations

import numpy as np
import pandas as pd

from utils import ensure_series


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    TradingView-style RSI:
    - Up/Down from close-to-close change
    - Wilder's RMA smoothing with SMA seed
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    def rma(series: pd.Series, length: int) -> pd.Series:
        out = pd.Series(index=series.index, dtype="float64")
        valid = series.dropna()
        if len(valid) < length:
            return out

        seed_idx = valid.index[length - 1]
        prev = float(valid.iloc[:length].mean())
        out.loc[seed_idx] = prev

        for idx in valid.index[length:]:
            curr = float(valid.loc[idx])
            prev = ((length - 1) * prev + curr) / length
            out.loc[idx] = prev
        return out

    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    atr = compute_atr(high, low, close, period=atr_period)

    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    for i in range(1, len(close)):
        if upper_band.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if lower_band.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    supertrend = pd.Series(index=close.index, dtype="float64")
    direction = pd.Series(index=close.index, dtype="int64")

    if len(close) == 0:
        return supertrend, direction

    supertrend.iloc[0] = final_upper.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(close)):
        if supertrend.iloc[i - 1] == final_upper.iloc[i - 1]:
            if close.iloc[i] <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
        else:
            if close.iloc[i] >= final_lower.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1

    return supertrend, direction


def compute_halftrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    amplitude: int = 2,
    channel_deviation: int = 2,
    atr_period: int = 100,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    HalfTrend indicator (Pine v6 reference):
    Returns: ht line, buy signal, sell signal, trend (0=up, 1=down).
    """
    n = len(close)
    ht = pd.Series(index=close.index, dtype="float64")
    buy_signal = pd.Series(False, index=close.index, dtype="bool")
    sell_signal = pd.Series(False, index=close.index, dtype="bool")
    trend = pd.Series(index=close.index, dtype="int64")

    if n == 0:
        return ht, buy_signal, sell_signal, trend

    atr = compute_atr(high, low, close, period=atr_period)
    atr2 = atr / 2.0
    dev = channel_deviation * atr2

    high_price = high.rolling(window=amplitude, min_periods=1).max()
    low_price = low.rolling(window=amplitude, min_periods=1).min()
    high_ma = high.rolling(window=amplitude, min_periods=1).mean()
    low_ma = low.rolling(window=amplitude, min_periods=1).mean()

    trend_val = 0
    next_trend = 0
    max_low_price = float(low.shift(1).iloc[0]) if n > 0 else float(low.iloc[0])
    min_high_price = float(high.shift(1).iloc[0]) if n > 0 else float(high.iloc[0])
    up = float("nan")
    down = float("nan")
    prev_up = float("nan")
    prev_down = float("nan")

    for i in range(n):
        if next_trend == 1:
            max_low_price = max(float(low_price.iloc[i]), max_low_price)
            low_prev = float(low.shift(1).iloc[i]) if i > 0 else float(low.iloc[i])
            if float(high_ma.iloc[i]) < max_low_price and float(close.iloc[i]) < low_prev:
                trend_val = 1
                next_trend = 0
                min_high_price = float(high_price.iloc[i])
        else:
            min_high_price = min(float(high_price.iloc[i]), min_high_price)
            high_prev = float(high.shift(1).iloc[i]) if i > 0 else float(high.iloc[i])
            if float(low_ma.iloc[i]) > min_high_price and float(close.iloc[i]) > high_prev:
                trend_val = 0
                next_trend = 1
                max_low_price = float(low_price.iloc[i])

        if trend_val == 0:
            if i > 0 and trend.iloc[i - 1] != 0:
                up = prev_down if not pd.isna(prev_down) else down
                buy_signal.iloc[i] = True
            else:
                base_up = max_low_price
                up = base_up if pd.isna(prev_up) else max(base_up, prev_up)
            ht.iloc[i] = up + dev.iloc[i] if i < len(dev) else up
        else:
            if i > 0 and trend.iloc[i - 1] != 1:
                down = prev_up if not pd.isna(prev_up) else up
                sell_signal.iloc[i] = True
            else:
                base_down = min_high_price
                down = base_down if pd.isna(prev_down) else min(base_down, prev_down)
            ht.iloc[i] = down - dev.iloc[i] if i < len(dev) else down

        trend.iloc[i] = trend_val
        prev_up = up
        prev_down = down

    return ht, buy_signal, sell_signal, trend


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX)."""
    high_diff = high.diff()
    low_diff = low.diff().abs()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

    atr = compute_atr(high, low, close, period=period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).abs()
    adx = 100 * dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.replace([np.inf, -np.inf], np.nan)


def week52_analysis(close: pd.Series) -> dict[str, float]:
    """Return 52-week high/low and distance percentages."""
    if len(close) < 252:
        return {"52w_high": np.nan, "52w_low": np.nan, "pct_from_high": np.nan, "pct_from_low": np.nan}
    high_52 = float(close.tail(252).max())
    low_52 = float(close.tail(252).min())
    current = float(close.iloc[-1])
    pct_from_high = ((current - high_52) / high_52) * 100.0 if high_52 else np.nan
    pct_from_low = ((current - low_52) / low_52) * 100.0 if low_52 else np.nan
    return {
        "52w_high": round(high_52, 2),
        "52w_low": round(low_52, 2),
        "pct_from_high": round(pct_from_high, 2),
        "pct_from_low": round(pct_from_low, 2),
    }

