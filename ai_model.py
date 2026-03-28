from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from config import AI_MIN_TRAIN_ROWS, AI_MODEL_DIR, AI_WALK_FORWARD_SPLITS
from indicators import compute_adx, compute_atr, compute_rsi, compute_supertrend
from utils import ensure_series


def ai_model_path(cache_key: str) -> Path:
    """Stable path for cached model (per universe/timeframe key)."""
    digest = str(abs(hash(cache_key)))
    return AI_MODEL_DIR / f"model_{digest}.joblib"


def load_ai_model(cache_key: str):
    path = ai_model_path(cache_key)
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        return payload.get("model")
    except Exception:
        return None


def save_ai_model(cache_key: str, model, metadata: dict | None = None) -> None:
    if model is None:
        return
    AI_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = ai_model_path(cache_key)
    payload = {"model": model, "metadata": metadata or {}}
    joblib.dump(payload, path)


def save_versioned_model(model, metadata: dict) -> Path | None:
    if model is None:
        return None
    AI_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    version_path = AI_MODEL_DIR / f"model_{ts}.joblib"
    joblib.dump({"model": model, "metadata": metadata}, version_path)
    return version_path


def load_latest_model():
    if not AI_MODEL_DIR.exists():
        return None, None
    files = sorted(AI_MODEL_DIR.glob("model_*.joblib"))
    if not files:
        return None, None
    payload = joblib.load(files[-1])
    return payload.get("model"), payload.get("metadata")


def get_feature_columns() -> list[str]:
    return [
        "rsi",
        "ema20",
        "ema_dist_pct",
        "dist_ma50",
        "dist_ma200",
        "vol_ratio",
        "atr_pct",
        "supertrend_dir",
        "adx",
        "day_of_week",
        "return_1d",
        "pct_from_52w_high",
        "pct_from_52w_low",
    ]


def compute_feature_frame(
    data: pd.DataFrame,
    rsi_length: int,
    atr_period: int,
    atr_multiplier: float,
) -> pd.DataFrame:
    """Compute ML feature frame from OHLCV data."""
    close = ensure_series(data["Close"]).astype("float64")
    high = ensure_series(data["High"]).astype("float64")
    low = ensure_series(data["Low"]).astype("float64")
    volume = ensure_series(data["Volume"]).astype("float64")

    ohlc = pd.concat([high, low, close, volume], axis=1, keys=["High", "Low", "Close", "Volume"]).dropna()
    if ohlc.empty:
        return pd.DataFrame()

    high = ohlc["High"]
    low = ohlc["Low"]
    close = ohlc["Close"]
    volume = ohlc["Volume"]

    rsi = compute_rsi(close, period=rsi_length)
    ema20 = close.ewm(span=20, adjust=False).mean()
    dist_ma50 = (close - close.rolling(50, min_periods=1).mean()) / close.rolling(50, min_periods=1).mean() * 100.0
    dist_ma200 = (close - close.rolling(200, min_periods=1).mean()) / close.rolling(200, min_periods=1).mean() * 100.0
    ema_dist_pct = (close - ema20) / ema20 * 100.0
    vol_ratio = volume / volume.rolling(20, min_periods=1).mean()

    atr = compute_atr(high, low, close, period=atr_period)
    atr_pct = (atr / close) * 100.0
    _, st_dir = compute_supertrend(
        high=high,
        low=low,
        close=close,
        atr_period=atr_period,
        multiplier=atr_multiplier,
    )
    adx = compute_adx(high, low, close, period=14)
    day_of_week = close.index.dayofweek
    return_1d = close.pct_change() * 100.0

    high_52w = close.rolling(252, min_periods=1).max()
    low_52w = close.rolling(252, min_periods=1).min()
    pct_from_52w_high = (close - high_52w) / high_52w * 100.0
    pct_from_52w_low = (close - low_52w) / low_52w * 100.0

    features = pd.DataFrame(
        {
            "rsi": rsi,
            "ema20": ema20,
            "ema_dist_pct": ema_dist_pct,
            "dist_ma50": dist_ma50,
            "dist_ma200": dist_ma200,
            "vol_ratio": vol_ratio,
            "atr_pct": atr_pct,
            "supertrend_dir": st_dir,
            "adx": adx,
            "day_of_week": day_of_week,
            "return_1d": return_1d,
            "pct_from_52w_high": pct_from_52w_high,
            "pct_from_52w_low": pct_from_52w_low,
        }
    )
    return features


def build_training_matrix(
    data: pd.DataFrame,
    rsi_length: int,
    atr_period: int,
    atr_multiplier: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix and labels for next-day direction."""
    features = compute_feature_frame(
        data=data,
        rsi_length=rsi_length,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
    )
    if features.empty:
        return pd.DataFrame(), pd.Series(dtype="int64")

    close = ensure_series(data["Close"]).astype("float64").reindex(features.index)
    next_return = close.pct_change().shift(-1)
    labels = (next_return > 0).astype("int64")

    frame = features.copy()
    frame["label"] = labels
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    if frame.empty:
        return pd.DataFrame(), pd.Series(dtype="int64")

    y = frame.pop("label")
    return frame, y


def walk_forward_validate(X: pd.DataFrame, y: pd.Series, n_splits: int = AI_WALK_FORWARD_SPLITS) -> list[float]:
    if X.empty or y.empty or y.nunique() < 2:
        return []
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X) // 50)))
    scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if y_train.nunique() < 2 or len(X_train) < AI_MIN_TRAIN_ROWS:
            continue
        model = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=150,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
        )
        model.fit(X_train, y_train)
        score = float(model.score(X_test, y_test))
        scores.append(score)
        logging.info("Walk-forward fold %d accuracy: %.4f", fold, score)
    return scores


def train_xgb_model(X: pd.DataFrame, y: pd.Series):
    """Train an XGBoost classifier if data is sufficient."""
    if X.empty or y.empty:
        return None
    if y.nunique() < 2:
        return None
    if len(X) < AI_MIN_TRAIN_ROWS:
        return None

    dtrain = xgb.DMatrix(X.to_numpy(dtype="float32"), label=y.to_numpy(dtype="float32"))
    params = {
        "max_depth": 4,
        "eta": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": 42,
    }
    booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=150)
    return booster


def score_to_signal(score: float) -> str:
    if score >= 0.75:
        return "Strong Buy"
    if score >= 0.60:
        return "Buy"
    if score >= 0.45:
        return "Neutral"
    return "Avoid"

