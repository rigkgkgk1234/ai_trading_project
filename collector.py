"""
collector.py - KIS WebSocket 실시간 국내 주식 데이터 수집기
────────────────────────────────────────────────────────────────
역할:
  - KIS OpenAPI WebSocket 에 연결하여 실시간 체결가 수신
  - SIMULATION_MODE=true 시 랜덤 시뮬레이션 데이터 생성 (API 없이 테스트)
  - 수신한 틱을 core/buffer.py 에 push
  - 자동 재접속(Exponential Backoff) 내장

KIS WebSocket 참고:
  TR_ID: H0STCNT0 (주식 현재가 체결)
  데이터: 거래소코드^종목코드^시간^현재가^전일대비...
────────────────────────────────────────────────────────────────
"""

import asyncio
import json
import random
from datetime import datetime
from typing import List

import requests
import websockets
from loguru import logger

from config import (
    KIS_APP_KEY,
    KIS_APP_SECRET,
    KIS_BASE_URL,
    KIS_WS_URL,
    SIMULATION_MODE,
    DEFAULT_STOCKS,
)
from core.buffer import buffer


# ─────────────────────────────────────────────
# KIS API: OAuth 접속키 발급
# ─────────────────────────────────────────────

def get_approval_key() -> str:
    """KIS WebSocket 접속용 승인키를 발급받습니다."""
    url  = f"{KIS_BASE_URL}/oauth2/Approval"
    body = {
        "grant_type": "client_credentials",
        "appkey"    : KIS_APP_KEY,
        "secretkey" : KIS_APP_SECRET,
    }
    resp = requests.post(url, json=body, timeout=10)
    resp.raise_for_status()
    key = resp.json().get("approval_key", "")
    logger.info(f"[Collector] 승인키 발급 완료: {key[:8]}...")
    return key


# ─────────────────────────────────────────────
# KIS WebSocket 구독 메시지 생성
# ─────────────────────────────────────────────

def make_subscribe_msg(approval_key: str, stock_code: str, tr_type: str = "1") -> str:
    """
    종목 구독/해제 메시지를 생성합니다.
    tr_type: "1" = 등록, "2" = 해제
    """
    return json.dumps({
        "header": {
            "approval_key": approval_key,
            "custtype"    : "P",
            "tr_type"     : tr_type,
            "content-type": "utf-8",
        },
        "body": {
            "input": {
                "tr_id" : "H0STCNT0",   # 주식 현재가 체결
                "tr_key": stock_code,
            }
        },
    })


# ─────────────────────────────────────────────
# KIS 실시간 체결 데이터 파싱
# ─────────────────────────────────────────────
# H0STCNT0 필드 순서 (주요 필드만 추출)
_H0STCNT0_IDX = {
    "mksc_shrn_iscd": 0,   # 종목코드
    "stck_cntg_hour": 1,   # 체결시간 (HHMMSS)
    "stck_prpr"     : 2,   # 현재가
    "prdy_vrss"     : 3,   # 전일 대비
    "prdy_ctrt"     : 4,   # 전일 대비율
    "acml_vol"      : 7,   # 누적 거래량
    "cntg_vol"      : 12,  # 체결 거래량
    "stck_oprc"     : 13,  # 시가
    "stck_hgpr"     : 14,  # 고가
    "stck_lwpr"     : 15,  # 저가
}

def _parse_tick(raw: str, code_name_map: dict) -> dict | None:
    """KIS H0STCNT0 데이터 문자열을 틱 dict로 변환합니다."""
    try:
        parts = raw.split("^")
        code  = parts[_H0STCNT0_IDX["mksc_shrn_iscd"]]
        now_str = parts[_H0STCNT0_IDX["stck_cntg_hour"]]  # HHMMSS
        today = datetime.now().strftime("%Y%m%d")
        ts = datetime.strptime(f"{today}{now_str}", "%Y%m%d%H%M%S")

        return {
            "code"     : code,
            "name"     : code_name_map.get(code, code),
            "timestamp": ts,
            "price"    : float(parts[_H0STCNT0_IDX["stck_prpr"]]),
            "open"     : float(parts[_H0STCNT0_IDX["stck_oprc"]]),
            "high"     : float(parts[_H0STCNT0_IDX["stck_hgpr"]]),
            "low"      : float(parts[_H0STCNT0_IDX["stck_lwpr"]]),
            "volume"   : int(parts[_H0STCNT0_IDX["acml_vol"]]),
            "tick_vol" : int(parts[_H0STCNT0_IDX["cntg_vol"]]),
            "change_rt": float(parts[_H0STCNT0_IDX["prdy_ctrt"]]),
        }
    except Exception as e:
        logger.debug(f"[Collector] 파싱 오류: {e} | raw={raw[:60]}")
        return None


# ─────────────────────────────────────────────
# 실제 KIS WebSocket 수집기
# ─────────────────────────────────────────────

async def run_kis_collector(stocks: List[dict]) -> None:
    """KIS WebSocket에 접속하여 실시간 체결 데이터를 수집합니다."""
    code_name_map = {s["code"]: s["name"] for s in stocks}
    codes = [s["code"] for s in stocks]

    retry_delay = 5
    max_delay   = 120

    while True:
        try:
            logger.info("[Collector] KIS 승인키 발급 중...")
            approval_key = get_approval_key()

            logger.info(f"[Collector] WebSocket 연결: {KIS_WS_URL}")
            async with websockets.connect(
                KIS_WS_URL,
                ping_interval=30,
                ping_timeout=10,
            ) as ws:
                # 모든 종목 구독 등록
                for code in codes:
                    await ws.send(make_subscribe_msg(approval_key, code, "1"))
                    await asyncio.sleep(0.1)
                logger.success(f"[Collector] {len(codes)}개 종목 구독 완료")
                retry_delay = 5  # 성공 시 재시도 딜레이 초기화

                async for message in ws:
                    if not isinstance(message, str):
                        continue
                    # 암호화 PINGPONG / 시스템 메시지 무시
                    if message.startswith("{"):
                        msg_json = json.loads(message)
                        rt_cd = msg_json.get("header", {}).get("tr_id", "")
                        if rt_cd == "PINGPONG":
                            await ws.send(message)
                        continue

                    # 실시간 체결 데이터: "0|H0STCNT0|001|..."
                    parts = message.split("|")
                    if len(parts) < 4 or parts[1] != "H0STCNT0":
                        continue

                    tick = _parse_tick(parts[3], code_name_map)
                    if tick:
                        buffer.push(tick)
                        logger.debug(
                            f"[Collector] {tick['name']}({tick['code']}) "
                            f"₩{tick['price']:,.0f} | 거래량 {tick['tick_vol']:,}"
                        )

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"[Collector] 연결 끊김: {e}. {retry_delay}s 후 재접속...")
        except Exception as e:
            logger.error(f"[Collector] 오류: {e}. {retry_delay}s 후 재시도...")

        await asyncio.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, max_delay)


# ─────────────────────────────────────────────
# 시뮬레이션 수집기 (SIMULATION_MODE=true)
# ─────────────────────────────────────────────

# 종목별 기준가 (시뮬레이션용)
_SIM_BASE_PRICES = {
    "005930": 75000,   # 삼성전자
    "000660": 185000,  # SK하이닉스
    "035420": 195000,  # NAVER
    "051910": 420000,  # LG화학
    "006400": 270000,  # 삼성SDI
    "035720": 48000,   # 카카오
    "068270": 165000,  # 셀트리온
    "207940": 780000,  # 삼성바이오로직스
}

async def run_simulation_collector(stocks: List[dict]) -> None:
    """
    시뮬레이션 틱 데이터 생성기.
    실제 API 없이도 대시보드/추론을 테스트할 수 있습니다.
    랜덤 워크(GBM) 기반으로 현실적인 주가 흐름을 모방합니다.
    """
    logger.info(
        f"[Collector] ⚠️  시뮬레이션 모드 시작 "
        f"({len(stocks)}개 종목 | 랜덤 틱 생성)"
    )

    # 종목별 상태 초기화
    state = {}
    for s in stocks:
        code = s["code"]
        base = _SIM_BASE_PRICES.get(code, 50000)
        state[code] = {
            "price"    : float(base),
            "open"     : float(base),
            "high"     : float(base),
            "low"      : float(base),
            "volume"   : 0,
            "name"     : s["name"],
        }

    while True:
        for s in stocks:
            code = s["code"]
            st   = state[code]

            # 기하 브라운 운동 (μ=0, σ=0.002)
            drift = random.gauss(0, 0.002)
            st["price"] = max(1, st["price"] * (1 + drift))
            st["high"]  = max(st["high"], st["price"])
            st["low"]   = min(st["low"],  st["price"])

            tick_vol = random.randint(100, 5000)
            st["volume"] += tick_vol

            change_rt = (st["price"] - st["open"]) / st["open"] * 100

            tick = {
                "code"     : code,
                "name"     : st["name"],
                "timestamp": datetime.now(),
                "price"    : round(st["price"], 0),
                "open"     : round(st["open"],  0),
                "high"     : round(st["high"],  0),
                "low"      : round(st["low"],   0),
                "volume"   : st["volume"],
                "tick_vol" : tick_vol,
                "change_rt": round(change_rt, 2),
            }
            buffer.push(tick)

        await asyncio.sleep(0.5)   # 0.5초마다 틱 생성


# ─────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────

async def start_collector(stocks: List[dict] = None) -> None:
    """
    수집기를 시작합니다. (버퍼 정리 태스크 포함)

    CancelledError 처리:
      asyncio.create_task() 로 생성된 하위 태스크는 부모가 취소돼도
      자동으로 취소되지 않습니다. 명시적으로 cancel() 을 호출해야
      서버 종료 시 좀비 태스크가 남지 않습니다.
    """
    if stocks is None:
        stocks = DEFAULT_STOCKS

    # 버퍼 자동 정리 태스크
    cleanup_task = asyncio.create_task(buffer.start_cleanup_task())

    if SIMULATION_MODE:
        collector_task = asyncio.create_task(run_simulation_collector(stocks))
    else:
        collector_task = asyncio.create_task(run_kis_collector(stocks))

    try:
        await asyncio.gather(cleanup_task, collector_task)
    except asyncio.CancelledError:
        # 외부(main.py lifespan)에서 취소 시 → 하위 태스크 명시적 취소
        cleanup_task.cancel()
        collector_task.cancel()
        await asyncio.gather(cleanup_task, collector_task, return_exceptions=True)
        logger.info("[Collector] 모든 태스크 정상 종료")
        raise


if __name__ == "__main__":
    asyncio.run(start_collector())
