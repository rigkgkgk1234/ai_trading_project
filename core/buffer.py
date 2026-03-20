"""
core/buffer.py - 인메모리 슬라이딩 윈도우 버퍼
────────────────────────────────────────────────────────────────
역할:
  - 실시간 수신 틱 데이터를 종목별로 메모리에 저장
  - BUFFER_MAX_ROWS / BUFFER_MAX_SECONDS 초과 시 오래된 데이터 자동 삭제
  - 스레드 안전(thread-safe) deque 기반 O(1) 삽입/삭제
  - 백그라운드 정리 태스크(asyncio) 내장
────────────────────────────────────────────────────────────────
틱 데이터 스키마 (dict):
  code       : str      종목코드
  name       : str      종목명
  timestamp  : datetime
  price      : float    현재가
  open       : float    시가
  high       : float    고가
  low        : float    저가
  volume     : int      누적거래량
  tick_vol   : int      틱 거래량
  change_rt  : float    등락률(%)
"""

import asyncio
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from config import (
    BUFFER_MAX_ROWS,
    BUFFER_MAX_SECONDS,
    BUFFER_CLEANUP_EVERY,
)


class StockBuffer:
    """종목별 실시간 틱 데이터를 보관하는 슬라이딩 윈도우 버퍼."""

    def __init__(self):
        self._data: Dict[str, deque] = {}
        self._lock = threading.Lock()
        self._running = False

    # ── 데이터 추가 ───────────────────────────────────────────
    def push(self, tick: dict) -> None:
        """새 틱 데이터를 버퍼에 추가합니다."""
        code = tick.get("code")
        if not code:
            return
        with self._lock:
            if code not in self._data:
                self._data[code] = deque(maxlen=BUFFER_MAX_ROWS)
            self._data[code].append(tick)

    def push_many(self, ticks: List[dict]) -> None:
        """복수 틱을 한 번에 추가합니다."""
        for tick in ticks:
            self.push(tick)

    # ── 데이터 조회 ───────────────────────────────────────────
    def get_df(self, code: str, n: Optional[int] = None) -> pd.DataFrame:
        """
        특정 종목의 버퍼 데이터를 DataFrame으로 반환합니다.
        Args:
            code: 종목코드 (예: "005930")
            n   : 최근 n개 행만 반환 (None 이면 전체)
        """
        with self._lock:
            if code not in self._data or len(self._data[code]) == 0:
                return pd.DataFrame()
            rows = list(self._data[code])

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        if n is not None:
            df = df.tail(n).reset_index(drop=True)
        return df

    def get_latest(self, code: str) -> Optional[dict]:
        """종목의 가장 최신 틱 1개를 반환합니다."""
        with self._lock:
            if code not in self._data or len(self._data[code]) == 0:
                return None
            return dict(self._data[code][-1])

    def get_all_codes(self) -> List[str]:
        """버퍼에 데이터가 있는 모든 종목코드를 반환합니다."""
        with self._lock:
            return [c for c, d in self._data.items() if len(d) > 0]

    def row_count(self, code: str) -> int:
        """종목별 현재 버퍼 행 수를 반환합니다."""
        with self._lock:
            return len(self._data.get(code, []))

    def buffer_stats(self) -> dict:
        """전체 버퍼 상태 요약을 반환합니다."""
        with self._lock:
            return {
                "total_codes": len(self._data),
                "total_rows": sum(len(d) for d in self._data.values()),
                "per_code": {code: len(d) for code, d in self._data.items()},
            }

    # ── 슬라이딩 윈도우 정리 ─────────────────────────────────
    def _cleanup_once(self) -> None:
        """
        시간 기준 슬라이딩 윈도우 정리.
        BUFFER_MAX_SECONDS 이전 데이터를 제거합니다.
        용량(maxlen) 초과 정리는 deque가 자동 처리합니다.
        """
        cutoff = datetime.now() - timedelta(seconds=BUFFER_MAX_SECONDS)
        removed_total = 0
        with self._lock:
            for code, dq in self._data.items():
                original = len(dq)
                while dq and dq[0]["timestamp"] < cutoff:
                    dq.popleft()
                removed_total += original - len(dq)
        if removed_total > 0:
            logger.debug(f"[Buffer] 슬라이딩 윈도우 정리: {removed_total}개 틱 삭제")

    async def start_cleanup_task(self) -> None:
        """백그라운드 정리 태스크를 시작합니다 (asyncio)."""
        self._running = True
        logger.info(
            f"[Buffer] 자동 정리 태스크 시작 "
            f"(주기={BUFFER_CLEANUP_EVERY}s | 보관={BUFFER_MAX_SECONDS}s | "
            f"최대={BUFFER_MAX_ROWS}행/종목)"
        )
        while self._running:
            await asyncio.sleep(BUFFER_CLEANUP_EVERY)
            self._cleanup_once()

    def stop(self) -> None:
        self._running = False

    def clear(self, code: Optional[str] = None) -> None:
        """버퍼를 초기화합니다. code 지정 시 해당 종목만 초기화."""
        with self._lock:
            if code:
                self._data.pop(code, None)
            else:
                self._data.clear()
        logger.info(f"[Buffer] 초기화 완료 ({'전체' if not code else code})")


# ─────────────────────────────────────────────
# 싱글톤 인스턴스 (전 모듈에서 공유)
# ─────────────────────────────────────────────
buffer = StockBuffer()
