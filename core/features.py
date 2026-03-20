"""
core/features.py - 공통 기술적 지표 피처 계산 모듈
────────────────────────────────────────────────────────────────
역할:
  - core/inference.py 와 models/train_model.py 가 동일하게 사용
  - 이 파일 하나만 수정하면 학습/추론 피처가 동시에 반영됨
  - Train-Serving Skew 방지를 위해 모든 rolling 파라미터 고정

피처 목록 (19개):
  이동평균   : ma5, ma10, ma20, ma60 + 비율 3개
  모멘텀     : rsi14, macd, macd_signal, macd_hist
  변동성     : bb_width, bb_pct, atr14
  거래량     : vol_ratio5, vol_ratio20, obv_norm
  가격       : price_range, change_rt
────────────────────────────────────────────────────────────────
"""

from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 피처 컬럼 정의 (학습/추론 공통)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "ma5", "ma10", "ma20", "ma60",
    "ma5_ratio", "ma10_ratio", "ma20_ratio",
    "rsi14",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pct",
    "atr14",
    "vol_ratio5", "vol_ratio20",
    "price_range",
    "change_rt",
    "obv_norm",
]

# 피처 계산에 필요한 최소 행 수
# ma60 = 60봉, ATR = 14봉, MACD = 26+9봉 → 60이 지배적
MIN_ROWS_REQUIRED = 60


def compute_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    OHLCV DataFrame으로부터 기술적 지표 피처를 계산합니다.

    학습(train_model.py)과 추론(inference.py) 양쪽에서 동일하게 호출합니다.
    rolling 윈도우 크기는 고정값을 사용하여 Train-Serving Skew를 방지합니다.

    Args:
        df: price, high, low, volume, change_rt 컬럼 포함 DataFrame
            (최소 MIN_ROWS_REQUIRED=60 행 권장)

    Returns:
        FEATURE_COLS 컬럼만 포함한 DataFrame, 또는 유효 행 없으면 None
    """
    if len(df) < MIN_ROWS_REQUIRED:
        return None

    df = df.copy()
    price  = df["price"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # ── 이동평균 (고정 윈도우) ────────────────
    df["ma5"]  = price.rolling(5).mean()
    df["ma10"] = price.rolling(10).mean()
    df["ma20"] = price.rolling(20).mean()
    df["ma60"] = price.rolling(60).mean()   # ← 학습/추론 동일하게 고정

    df["ma5_ratio"]  = price / df["ma5"]  - 1
    df["ma10_ratio"] = price / df["ma10"] - 1
    df["ma20_ratio"] = price / df["ma20"] - 1

    # ── RSI(14) ───────────────────────────────
    delta  = price.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # ── MACD ──────────────────────────────────
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands (20, 2σ) ──────────────
    bb_mid = price.rolling(20).mean()
    bb_std = price.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_up - bb_dn) / (bb_mid + 1e-9)
    df["bb_pct"]   = (price - bb_dn) / (bb_up - bb_dn + 1e-9)

    # ── ATR(14) ───────────────────────────────
    tr = pd.concat([
        high - low,
        (high - price.shift()).abs(),
        (low  - price.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean() / (price + 1e-9)

    # ── 거래량 비율 ───────────────────────────
    vol_ma5  = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()
    df["vol_ratio5"]  = volume / (vol_ma5  + 1e-9)
    df["vol_ratio20"] = volume / (vol_ma20 + 1e-9)

    # ── 가격 범위 ─────────────────────────────
    df["price_range"] = (high - low) / (price + 1e-9)

    # ── 등락률 ────────────────────────────────
    df["change_rt"] = df["change_rt"].astype(float)

    # ── OBV 정규화 ────────────────────────────
    direction = np.sign(price.diff().fillna(0))
    obv = (direction * volume).cumsum()
    df["obv_norm"] = (
        (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-9)
    )

    feat_df = df[FEATURE_COLS].dropna()
    if len(feat_df) == 0:
        return None
    return feat_df
