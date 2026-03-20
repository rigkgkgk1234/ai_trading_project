"""
core/inference.py - ONNX 기반 경량 모델 추론 엔진
────────────────────────────────────────────────────────────────
역할:
  - 버퍼에서 최신 데이터를 가져와 피처를 계산 (core/features.py 사용)
  - ONNX Runtime으로 로컬 추론 (저지연)
  - 예측 결과: BUY / HOLD / SELL + 신뢰도(%)
────────────────────────────────────────────────────────────────
"""

import os
from typing import Optional

import joblib
import numpy as np
import onnxruntime as ort
from loguru import logger

from config import (
    MODEL_PATH,
    SCALER_PATH,
    LABEL_MAP,
    MIN_ROWS_FOR_PRED,
)
from core.buffer import buffer
from core.features import FEATURE_COLS, compute_features, MIN_ROWS_REQUIRED

# config의 MIN_ROWS_FOR_PRED 가 features의 MIN_ROWS_REQUIRED 보다 작으면 경고
if MIN_ROWS_FOR_PRED < MIN_ROWS_REQUIRED:
    logger.warning(
        f"[Inference] MIN_ROWS_FOR_PRED({MIN_ROWS_FOR_PRED}) < "
        f"MIN_ROWS_REQUIRED({MIN_ROWS_REQUIRED}). "
        f"ma60 등 장기 지표 계산이 불가능할 수 있습니다."
    )


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
        """
        재학습 후 새 ONNX 모델을 핫-로드합니다.
        이전 세션을 명시적으로 해제하여 메모리 누수를 방지합니다.
        """
        # 이전 세션 명시적 해제 (메모리 누수 방지)
        if self.session is not None:
            del self.session
            self.session = None
        if self.scaler is not None:
            del self.scaler
            self.scaler = None

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
        base = {
            "code": code, "signal": "HOLD", "confidence": 0.0,
            "probas": {}, "price": 0.0, "rows_used": 0, "error": None,
        }

        if not self._loaded:
            base["error"] = "모델 미로드 (train_model.py 먼저 실행)"
            return base

        df = buffer.get_df(code)
        if df.empty:
            base["error"] = "버퍼 데이터 없음"
            return base

        latest = buffer.get_latest(code)
        base["price"] = latest["price"] if latest else 0.0

        # 피처 계산 — core/features.py 공통 함수 사용
        feat_df = compute_features(df)
        if feat_df is None:
            base["error"] = (
                f"피처 계산 불가 "
                f"(최소 {MIN_ROWS_REQUIRED}행 필요, 현재 {len(df)}행)"
            )
            return base

        # 마지막 행만 사용 (현재 시점 예측)
        X_raw    = feat_df.iloc[[-1]].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_raw).astype(np.float32)

        input_name = self.session.get_inputs()[0].name
        outputs    = self.session.run(None, {input_name: X_scaled})

        label_idx = int(outputs[0][0])

        # ONNX XGBoost 출력 형식에 따라 확률 파싱
        if len(outputs) > 1 and isinstance(outputs[1], list):
            proba_dict = outputs[1][0]
            probas_arr = np.array([proba_dict.get(i, 0.0) for i in range(3)])
        elif len(outputs) > 1 and hasattr(outputs[1], "shape"):
            probas_arr = outputs[1][0]
        else:
            probas_arr = np.zeros(3)
            probas_arr[label_idx] = 1.0

        signal     = LABEL_MAP.get(label_idx, "HOLD")
        confidence = float(probas_arr[label_idx]) * 100

        base.update({
            "signal"    : signal,
            "confidence": round(confidence, 2),
            "probas": {
                "BUY" : round(float(probas_arr[1]) * 100, 2),
                "HOLD": round(float(probas_arr[0]) * 100, 2),
                "SELL": round(float(probas_arr[2]) * 100, 2),
            },
            "rows_used": len(feat_df),
        })
        return base

    def predict_batch(self, codes: list) -> list:
        """복수 종목 일괄 예측 (순차 실행, CPU 바운드 특성상 적절)."""
        return [self.predict(code) for code in codes]


# ─────────────────────────────────────────────
# 싱글톤 인스턴스
# ─────────────────────────────────────────────
engine = InferenceEngine()
