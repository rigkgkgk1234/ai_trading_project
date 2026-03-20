"""
core/inference.py - ONNX 기반 경량 모델 추론 엔진
────────────────────────────────────────────────────────────────
역할:
  - 버퍼에서 최신 데이터를 가져와 기술적 지표를 계산
  - ONNX Runtime으로 로컬 추론 (저지연)
  - 예측 결과: BUY / HOLD / SELL + 신뢰도(%)
────────────────────────────────────────────────────────────────
기술적 지표 (Feature Engineering):
  이동평균   : MA5, MA10, MA20, MA60
  모멘텀     : RSI(14), MACD, MACD Signal
  변동성     : Bollinger Band Width, ATR(14)
  거래량     : Volume Ratio, OBV
  가격 위치  : Price vs MA20, High-Low Range
"""

import os
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort
from loguru import logger

from config import (
    MODEL_PATH,
    SCALER_PATH,
    LABEL_MAP,
    MIN_ROWS_FOR_PRED,
    PREDICTION_WINDOW,
)
from core.buffer import buffer


# ─────────────────────────────────────────────
# 피처 컬럼 정의 (학습/추론 일치 필수)
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


def compute_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    틱 DataFrame으로부터 기술적 지표 피처를 계산합니다.

    Args:
        df: price, high, low, volume, change_rt 컬럼 포함 DataFrame
    Returns:
        피처 DataFrame (마지막 유효 행 기준) 또는 None
    """
    if len(df) < MIN_ROWS_FOR_PRED:
        return None

    df = df.copy()
    price  = df["price"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # ── 이동평균 ─────────────────────────────
    df["ma5"]  = price.rolling(5).mean()
    df["ma10"] = price.rolling(10).mean()
    df["ma20"] = price.rolling(20).mean()
    df["ma60"] = price.rolling(min(60, len(df))).mean()

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

    # ── Bollinger Bands ───────────────────────
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
    df["obv_norm"] = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-9)

    feat_df = df[FEATURE_COLS].dropna()
    if len(feat_df) == 0:
        return None
    return feat_df


class InferenceEngine:
    """ONNX 모델을 로드하고 추론을 수행하는 엔진."""

    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.scaler = None
        self._loaded = False
        self._load()

    def _load(self) -> None:
        """모델과 스케일러를 디스크에서 로드합니다."""
        if not os.path.exists(MODEL_PATH):
            logger.warning(
                f"[Inference] ONNX 모델 없음: {MODEL_PATH} "
                "→ models/train_model.py 를 먼저 실행하세요."
            )
            return
        if not os.path.exists(SCALER_PATH):
            logger.warning(f"[Inference] 스케일러 없음: {SCALER_PATH}")
            return

        try:
            self.session = ort.InferenceSession(
                MODEL_PATH,
                providers=["CPUExecutionProvider"],
            )
            self.scaler = joblib.load(SCALER_PATH)
            self._loaded = True
            logger.info(f"[Inference] 모델 로드 완료: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"[Inference] 모델 로드 실패: {e}")

    def reload(self) -> bool:
        """모델을 재로드합니다 (재학습 후 호출)."""
        self._loaded = False
        self._load()
        return self._loaded

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def predict(self, code: str) -> dict:
        """
        특정 종목에 대한 예측을 수행합니다.

        Returns:
            {
              "code"       : str,
              "signal"     : "BUY" | "HOLD" | "SELL",
              "confidence" : float (0~100),
              "probas"     : {"BUY": float, "HOLD": float, "SELL": float},
              "price"      : float,
              "rows_used"  : int,
              "error"      : str | None,
            }
        """
        base = {"code": code, "signal": "HOLD", "confidence": 0.0,
                "probas": {}, "price": 0.0, "rows_used": 0, "error": None}

        if not self._loaded:
            base["error"] = "모델 미로드 (train_model.py 먼저 실행)"
            return base

        df = buffer.get_df(code)
        if df.empty:
            base["error"] = "버퍼 데이터 없음"
            return base

        latest = buffer.get_latest(code)
        base["price"] = latest["price"] if latest else 0.0

        feat_df = compute_features(df)
        if feat_df is None:
            base["error"] = f"피처 계산 불가 (최소 {MIN_ROWS_FOR_PRED}행 필요)"
            return base

        # 마지막 행만 사용 (현재 시점 예측)
        X_raw = feat_df.iloc[[-1]].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_raw).astype(np.float32)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: X_scaled})

        # outputs[0]: label, outputs[1]: probabilities dict
        label_idx = int(outputs[0][0])
        # ONNX XGBoost 출력 형식에 따라 확률 파싱
        if len(outputs) > 1 and isinstance(outputs[1], list):
            proba_dict = outputs[1][0]  # [{0: p0, 1: p1, 2: p2}]
            probas_arr = np.array([proba_dict.get(i, 0.0) for i in range(3)])
        elif len(outputs) > 1 and hasattr(outputs[1], "shape"):
            probas_arr = outputs[1][0]
        else:
            probas_arr = np.array([0.0, 0.0, 0.0])
            probas_arr[label_idx] = 1.0

        signal = LABEL_MAP.get(label_idx, "HOLD")
        confidence = float(probas_arr[label_idx]) * 100

        base.update({
            "signal"    : signal,
            "confidence": round(confidence, 2),
            "probas"    : {
                "BUY" : round(float(probas_arr[1]) * 100, 2),
                "HOLD": round(float(probas_arr[0]) * 100, 2),
                "SELL": round(float(probas_arr[2]) * 100, 2),
            },
            "rows_used" : len(feat_df),
        })
        return base

    def predict_batch(self, codes: list) -> list:
        """복수 종목 일괄 예측."""
        return [self.predict(code) for code in codes]


# ─────────────────────────────────────────────
# 싱글톤 인스턴스
# ─────────────────────────────────────────────
engine = InferenceEngine()
