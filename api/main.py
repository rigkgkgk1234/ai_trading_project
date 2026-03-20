"""
api/main.py - FastAPI 추론 서버
────────────────────────────────────────────────────────────────
엔드포인트:
  GET  /                       - 서버 상태 확인
  GET  /health                 - 헬스체크
  GET  /stocks                 - 관심 종목 목록
  GET  /buffer/stats           - 버퍼 상태
  GET  /predict/{code}         - 단일 종목 예측
  GET  /predict                - 전체 종목 일괄 예측
  GET  /data/{code}            - 종목 최근 틱 데이터
  POST /model/reload           - 모델 재로드 (재학습 후 호출)
  GET  /stream/{code}          - SSE 실시간 예측 스트림

시작 방법:
  uvicorn api.main:app --reload --port 8000
────────────────────────────────────────────────────────────────
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from config import DEFAULT_STOCKS, SIMULATION_MODE
from core.buffer import buffer
from core.inference import engine
from collector import start_collector


# ─────────────────────────────────────────────
# 앱 수명주기: 수집기 + 버퍼 정리 백그라운드 시작
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("🚀  AI 국내 주식 예측 서버 시작")
    logger.info(f"    모드: {'시뮬레이션' if SIMULATION_MODE else 'KIS 실전'}")
    logger.info(f"    종목: {[s['name'] for s in DEFAULT_STOCKS]}")
    logger.info("=" * 60)

    # 데이터 수집기를 백그라운드 태스크로 시작
    collector_task = asyncio.create_task(start_collector(DEFAULT_STOCKS))

    yield  # 서버 실행

    collector_task.cancel()
    try:
        await collector_task
    except asyncio.CancelledError:
        pass   # start_collector 내부에서 re-raise 된 CancelledError 수거
    buffer.stop()
    logger.info("👋  서버 종료")


# ─────────────────────────────────────────────
# FastAPI 앱 초기화
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "AI 국내 주식 예측 API",
    description = "KIS WebSocket 기반 실시간 국내 주가 예측 시스템",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ─────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────

def _stock_info(code: str) -> dict:
    return next((s for s in DEFAULT_STOCKS if s["code"] == code), {})

def _require_code(code: str):
    if not any(s["code"] == code for s in DEFAULT_STOCKS):
        raise HTTPException(status_code=404, detail=f"종목 {code} 없음")


# ─────────────────────────────────────────────
# 기본 엔드포인트
# ─────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "service" : "AI 국내 주식 예측 시스템",
        "version" : "1.0.0",
        "mode"    : "simulation" if SIMULATION_MODE else "live",
        "model"   : "ready" if engine.is_ready else "not_loaded",
        "time"    : datetime.now().isoformat(),
    }


@app.get("/health", tags=["System"])
async def health():
    stats = buffer.buffer_stats()
    return {
        "status"        : "ok",
        "model_ready"   : engine.is_ready,
        "buffer_codes"  : stats["total_codes"],
        "buffer_rows"   : stats["total_rows"],
        "timestamp"     : datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────
# 종목 정보
# ─────────────────────────────────────────────

@app.get("/stocks", tags=["Stocks"])
async def list_stocks():
    """관심 종목 목록과 최신 가격을 반환합니다."""
    result = []
    for s in DEFAULT_STOCKS:
        latest = buffer.get_latest(s["code"])
        result.append({
            **s,
            "price"    : latest["price"]     if latest else None,
            "change_rt": latest["change_rt"] if latest else None,
            "volume"   : latest["volume"]    if latest else None,
            "buffered" : buffer.row_count(s["code"]),
        })
    return result


# ─────────────────────────────────────────────
# 버퍼 정보
# ─────────────────────────────────────────────

@app.get("/buffer/stats", tags=["Buffer"])
async def buffer_stats():
    """인메모리 버퍼 현황을 반환합니다."""
    stats = buffer.buffer_stats()
    stats["timestamp"] = datetime.now().isoformat()
    return stats


@app.delete("/buffer/{code}", tags=["Buffer"])
async def clear_buffer(code: str):
    """특정 종목의 버퍼를 초기화합니다."""
    _require_code(code)
    buffer.clear(code)
    return {"message": f"{code} 버퍼 초기화 완료"}


# ─────────────────────────────────────────────
# 틱 데이터 조회
# ─────────────────────────────────────────────

@app.get("/data/{code}", tags=["Data"])
async def get_data(code: str, n: int = 100):
    """
    종목의 최근 틱 데이터를 반환합니다.
    n: 반환할 최근 행 수 (기본 100, 최대 500)
    """
    _require_code(code)
    n = min(n, 500)
    df = buffer.get_df(code, n=n)
    if df.empty:
        return {"code": code, "rows": 0, "data": []}

    records = df.copy()
    records["timestamp"] = records["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "code" : code,
        "name" : _stock_info(code).get("name", code),
        "rows" : len(records),
        "data" : records.to_dict(orient="records"),
    }


# ─────────────────────────────────────────────
# AI 예측
# ─────────────────────────────────────────────

@app.get("/predict/{code}", tags=["Prediction"])
async def predict_single(code: str):
    """단일 종목 AI 예측을 수행합니다."""
    _require_code(code)
    result = engine.predict(code)
    result["name"] = _stock_info(code).get("name", code)
    result["timestamp"] = datetime.now().isoformat()
    return result


@app.get("/predict", tags=["Prediction"])
async def predict_all():
    """모든 관심 종목에 대한 일괄 예측을 수행합니다."""
    codes   = [s["code"] for s in DEFAULT_STOCKS]
    results = engine.predict_batch(codes)
    for r in results:
        r["name"]      = _stock_info(r["code"]).get("name", r["code"])
        r["timestamp"] = datetime.now().isoformat()
    return results


# ─────────────────────────────────────────────
# 모델 관리
# ─────────────────────────────────────────────

@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """
    재학습 후 새 ONNX 모델을 핫-로드합니다.
    서버 재시작 없이 반영됩니다.
    """
    success = engine.reload()
    return {
        "success"  : success,
        "model_ready": engine.is_ready,
        "message"  : "모델 재로드 완료" if success else "모델 파일 없음 (학습 필요)",
        "timestamp": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────
# SSE 실시간 스트림 (대시보드 연동용)
# ─────────────────────────────────────────────

@app.get("/stream/{code}", tags=["Stream"])
async def stream_prediction(code: str, interval: float = 2.0):
    """
    Server-Sent Events로 실시간 예측을 스트리밍합니다.
    interval: 푸시 간격(초), 기본 2초
    """
    _require_code(code)
    interval = max(0.5, min(interval, 10.0))

    async def event_generator():
        import json
        while True:
            result = engine.predict(code)
            latest = buffer.get_latest(code)
            data = {
                **result,
                "name"     : _stock_info(code).get("name", code),
                "timestamp": datetime.now().isoformat(),
                "price"    : latest["price"] if latest else 0,
                "change_rt": latest["change_rt"] if latest else 0,
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            await asyncio.sleep(interval)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
