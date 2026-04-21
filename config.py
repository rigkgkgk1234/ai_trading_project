"""
config.py - 중앙 설정 관리
모든 환경변수 및 시스템 파라미터를 통합 관리합니다.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# KIS (한국투자증권) API 설정
# ─────────────────────────────────────────────
KIS_APP_KEY    = os.getenv("KIS_APP_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_APP_SECRET", "")
KIS_ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO", "")      # 계좌번호 앞 8자리
KIS_ACCOUNT_CD = os.getenv("KIS_ACCOUNT_CD", "01")    # 계좌 상품코드

# 실전: "https://openapi.koreainvestment.com:9443"
# 모의: "https://openapivts.koreainvestment.com:29443"
KIS_BASE_URL   = os.getenv("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443")

# WebSocket
# 실전: "ws://ops.koreainvestment.com:21000"
# 모의: "ws://ops.koreainvestment.com:31000"
KIS_WS_URL     = os.getenv("KIS_WS_URL", "ws://ops.koreainvestment.com:21000")

# ─────────────────────────────────────────────
# 시뮬레이션 모드 (API 키 없을 때 테스트용)
# ─────────────────────────────────────────────
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "true").lower() == "true"

# ─────────────────────────────────────────────
# 기본 관심 종목 (KOSPI/KOSDAQ 종목코드)
# ─────────────────────────────────────────────
DEFAULT_STOCKS = [
    {"code": "005930", "name": "삼성전자",   "market": "KOSPI"},
    {"code": "000660", "name": "SK하이닉스", "market": "KOSPI"},
    {"code": "035420", "name": "NAVER",      "market": "KOSPI"},
    {"code": "051910", "name": "LG화학",     "market": "KOSPI"},
    {"code": "006400", "name": "삼성SDI",    "market": "KOSPI"},
    {"code": "035720", "name": "카카오",     "market": "KOSPI"},
    {"code": "068270", "name": "셀트리온",   "market": "KOSPI"},
    {"code": "207940", "name": "삼성바이오로직스", "market": "KOSPI"},
]

# ─────────────────────────────────────────────
# 인메모리 버퍼 설정 (슬라이딩 윈도우)
# ─────────────────────────────────────────────
BUFFER_MAX_ROWS      = 5_000   # 종목당 최대 틱 수
BUFFER_MAX_SECONDS   = 86_400  # 최대 보관 시간 (24시간)
BUFFER_CLEANUP_EVERY = 60      # 자동 정리 주기 (초)

# ─────────────────────────────────────────────
# 모델 / 추론 설정
# ─────────────────────────────────────────────
MODEL_DIR         = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH        = os.path.join(MODEL_DIR, "stock_model.onnx")
SCALER_PATH       = os.path.join(MODEL_DIR, "scaler.pkl")
LABEL_MAP         = {0: "HOLD", 1: "BUY", 2: "SELL"}
PREDICTION_WINDOW = 5   # 몇 틱 후 방향 예측
MIN_ROWS_FOR_PRED = 60  # ma60 계산을 위해 최소 60행 필요


# ─────────────────────────────────────────────
# 4. 피처 이름
# ─────────────────────────────────────────────

FEATURE_COLS: list[str] = [
    "prev_1_close",       # 전 봉 종가
    "prev_2_close",       # 전전 봉 종가
    "prev_5_close",       # 5 일 전 종가
    "prev_10_close",      # 10 일 전 종가
    "vol_ma_5",           # 5 일 이동평균량
    "vol_ma_10",          # 10 일 이동평균량
    "vol_ratio_5",        # 5 일 평균 대비 거래량 비율
    "vol_ratio_10",       # 10 일 평균 대비 거래량 비율
    "volatility",         # ATR
    "rsi",                # RSI
    "cci",                # CCI
    "wma",                # WMA
    "ema_12",             # EMA (단기)
    "ema_26",             # EMA (장기)
    "macd",               # MACD
    "macd_signal",        # MACD 신호선
    "macd_hist",          # MACD 히스토그램
    "adx",                # ADX
    "ao",                 # AO
    "mom",                # 모멘텀
    "roc",                # ROC
    "upper_band",         # 볼린저밴드 상단
    "lower_band",         # 볼린저밴드 하단
    "price_pos",          # 가격 위치
    "price_log_return",   # 로그 수익률
]

# ─────────────────────────────────────────────
# FastAPI 서버 설정
# ─────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# ─────────────────────────────────────────────
# Streamlit 대시보드 설정
# ─────────────────────────────────────────────
STREAMLIT_REFRESH_SEC = 3   # 대시보드 갱신 주기 (초)
CHART_CANDLE_LIMIT    = 200 # 캔들 차트 최대 표시 개수

# ─────────────────────────────────────────────
# PyKRX 학습 데이터 기간
# ─────────────────────────────────────────────
TRAIN_START_DATE = "20230101"
TRAIN_END_DATE   = "20241231"
