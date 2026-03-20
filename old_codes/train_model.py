"""
models/train_model.py - XGBoost 모델 학습 및 ONNX 변환
────────────────────────────────────────────────────────────────
역할:
  1. PyKRX로 국내 주식 일봉 데이터 수집 (TRAIN_START_DATE ~ TRAIN_END_DATE)
  2. 기술적 지표 피처 계산 (core/inference.py 와 동일 로직)
  3. 레이블 생성: PREDICTION_WINDOW 봉 후 수익률 기준
     - +0.5% 이상  → BUY(1)
     - -0.5% 이하  → SELL(2)
     - 그 외       → HOLD(0)
  4. XGBoost 분류기 학습
  5. ONNX 변환 및 저장 (onnxmltools)
  6. 모델 성능 리포트 출력

Usage:
    python models/train_model.py
    python models/train_model.py --codes 005930 000660 035420
────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from pykrx import stock as krx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ONNX 변환
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

from config import (
    DEFAULT_STOCKS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    MODEL_PATH,
    SCALER_PATH,
    PREDICTION_WINDOW,
    MODEL_DIR,
)
from core.inference import FEATURE_COLS


# ─────────────────────────────────────────────
# 1. 데이터 수집 (PyKRX)
# ─────────────────────────────────────────────

def fetch_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    """PyKRX로 일봉 OHLCV 데이터를 수집합니다."""
    try:
        df = krx.get_market_ohlcv(start, end, code)
        if df is None or df.empty:
            logger.warning(f"[Train] 데이터 없음: {code}")
            return pd.DataFrame()
        df = df.rename(columns={
            "시가": "open", "고가": "high", "저가": "low",
            "종가": "price", "거래량": "volume",
        })
        df.index = pd.to_datetime(df.index)
        df.index.name = "timestamp"
        df = df.reset_index()
        df["code"]      = code
        df["tick_vol"]  = df["volume"]
        df["change_rt"] = df["price"].pct_change() * 100
        logger.info(f"[Train] {code}: {len(df)}행 수집 ({start}~{end})")
        return df[["timestamp","code","price","open","high","low","volume","tick_vol","change_rt"]]
    except Exception as e:
        logger.error(f"[Train] {code} 수집 실패: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# 2. 피처 계산 (core/inference.py 와 동일)
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame에서 기술적 지표 피처를 생성합니다."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    price  = df["price"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    df["ma5"]  = price.rolling(5).mean()
    df["ma10"] = price.rolling(10).mean()
    df["ma20"] = price.rolling(20).mean()
    df["ma60"] = price.rolling(60).mean()

    df["ma5_ratio"]  = price / df["ma5"]  - 1
    df["ma10_ratio"] = price / df["ma10"] - 1
    df["ma20_ratio"] = price / df["ma20"] - 1

    delta = price.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    bb_mid = price.rolling(20).mean()
    bb_std = price.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_up - bb_dn) / (bb_mid + 1e-9)
    df["bb_pct"]   = (price - bb_dn) / (bb_up - bb_dn + 1e-9)

    tr = pd.concat([
        high - low,
        (high - price.shift()).abs(),
        (low  - price.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean() / (price + 1e-9)

    vol_ma5  = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()
    df["vol_ratio5"]  = volume / (vol_ma5  + 1e-9)
    df["vol_ratio20"] = volume / (vol_ma20 + 1e-9)

    df["price_range"] = (high - low) / (price + 1e-9)
    df["change_rt"]   = df["change_rt"].astype(float)

    direction = np.sign(price.diff().fillna(0))
    obv = (direction * volume).cumsum()
    df["obv_norm"] = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-9)

    return df


# ─────────────────────────────────────────────
# 3. 레이블 생성
# ─────────────────────────────────────────────

BUY_THRESHOLD  =  0.005   # +0.5%
SELL_THRESHOLD = -0.005   # -0.5%

def make_labels(df: pd.DataFrame, window: int = PREDICTION_WINDOW) -> pd.Series:
    """
    window 봉 후 수익률에 따라 레이블을 생성합니다.
    HOLD=0, BUY=1, SELL=2
    """
    future_ret = df["price"].pct_change(window).shift(-window)
    labels = pd.Series(0, index=df.index, name="label")
    labels[future_ret >= BUY_THRESHOLD]  = 1
    labels[future_ret <= SELL_THRESHOLD] = 2
    return labels


# ─────────────────────────────────────────────
# 4. 학습 파이프라인
# ─────────────────────────────────────────────

def train(codes: list) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 데이터 수집 ───────────────────────────
    all_frames = []
    for code in codes:
        df = fetch_ohlcv(code, TRAIN_START_DATE, TRAIN_END_DATE)
        if df.empty:
            continue
        df = build_features(df)
        df["label"] = make_labels(df)
        all_frames.append(df)

    if not all_frames:
        logger.error("[Train] 수집된 데이터가 없습니다. 종료.")
        return

    full_df = pd.concat(all_frames, ignore_index=True)
    logger.info(f"[Train] 전체 데이터: {len(full_df):,}행")

    # ── 피처/레이블 분리 ──────────────────────
    available_cols = [c for c in FEATURE_COLS if c in full_df.columns]
    data = full_df[available_cols + ["label"]].dropna()
    logger.info(f"[Train] dropna 후: {len(data):,}행 | 피처: {len(available_cols)}개")

    X = data[available_cols].values.astype(np.float32)
    y = data["label"].values.astype(int)

    label_counts = pd.Series(y).value_counts().sort_index()
    logger.info(f"[Train] 레이블 분포: HOLD={label_counts.get(0,0)}, "
                f"BUY={label_counts.get(1,0)}, SELL={label_counts.get(2,0)}")

    # ── 스케일링 ──────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    # ── XGBoost 학습 ──────────────────────────
    logger.info("[Train] XGBoost 학습 시작...")
    model = XGBClassifier(
        n_estimators   = 300,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        use_label_encoder=False,
        eval_metric     = "mlogloss",
        random_state    = 42,
        n_jobs          = -1,
    )
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=50,
    )

    # ── 성능 평가 ─────────────────────────────
    y_pred = model.predict(X_test_s)
    logger.info("\n" + classification_report(
        y_test, y_pred,
        target_names=["HOLD", "BUY", "SELL"],
    ))
    logger.info(f"[Train] Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # ── 스케일러 저장 ─────────────────────────
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"[Train] 스케일러 저장: {SCALER_PATH}")

    # ── ONNX 변환 & 저장 ──────────────────────
    logger.info("[Train] ONNX 변환 중...")
    initial_type = [("float_input", FloatTensorType([None, len(available_cols)]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    onnxmltools.utils.save_model(onnx_model, MODEL_PATH)
    logger.success(f"[Train] ONNX 모델 저장 완료: {MODEL_PATH}")
    logger.success("=" * 60)
    logger.success("✅  학습 완료! FastAPI 서버를 재시작하면 새 모델이 적용됩니다.")
    logger.success("=" * 60)


# ─────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="국내 주식 AI 모델 학습")
    parser.add_argument(
        "--codes", nargs="+",
        default=[s["code"] for s in DEFAULT_STOCKS],
        help="학습할 종목코드 (예: 005930 000660)",
    )
    args = parser.parse_args()
    train(args.codes)
