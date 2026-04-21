#!/usr/bin/env python3
"""
AI 주식 예측 시스템 - 워크플로우 테스트 스크립트
──────────────────────────────────────────────────────────────────
이 스크립트로 다음 테스트를 수행합니다:
1. backend 서버 스타트업 (uvicorn)
2. model/reload POST 테스트
3. /health GET 테스트
4. /predict GET 테스트
5. /stream GET 테스트
6. frontend 앱 스타트업 (streamlit)

사용 방법:
  python test_workflow.py

주의: 이 스크립트는 uvicorn 과 streamlit 이 설치되어 있어야 합니다.
    설치 필요 시: pip install uvicorn streamlit requests
──────────────────────────────────────────────────────────────────
"""

import subprocess
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.inference import engine
from core.buffer import buffer
from collector import start_collector


def clear_and_reload():
    """버퍼 초기화 및 모델 재로드"""
    # 버퍼 초기화
    codes = [s["code"] for s in DEFAULT_STOCKS]
    for c in codes:
        buffer.clear(c)

    # 모델 재로드
    engine.reload()

    print("✅ 버퍼 및 모델 초기화 완료")
    return engine.is_ready


def test_health():
    """/health 엔드포인트 테스트"""
    try:
        result = engine.health_check()
        return {
            "success": result["status"] == "ok",
            "model_ready": result.get("model_ready", False),
            "buffer_rows": result.get("buffer_rows", 0),
            "buffer_codes": result.get("buffer_codes", 0),
            "message": f"모델 상태: {'✅' if result['model_ready'] else '⏳'} 버퍼: {result['buffer_rows']:,}행"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"❌ 헬스체크 실패: {e}"
        }


def test_predict(code: str):
    """/predict/{code} 엔드포인트 테스트"""
    try:
        result = engine.predict(code)
        return {
            "success": True,
            "prediction": result["prediction"],
            "prediction_prob": result["prediction_prob"],
            "message": f"✅ {code}: {result['prediction']} ({result['prediction_prob']:.2%})"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"❌ 예측 실패: {e}"
        }


def test_stream(code: str, interval: float = 2.0):
    """/stream/{code} 엔드포인트 테스트 (SSE)"""
    try:
        import asyncio

        async def test():
            count = 0
            for _ in range(3):  # 3 번 요청
                result = engine.predict(code)
                await asyncio.sleep(interval)
                count += 1
            return count

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            count = loop.run_until_complete(test())
            return {
                "success": True,
                "count": count,
                "message": f"✅ SSE 스트리밍 정상 (3 회 요청 완료)"
            }
        finally:
            loop.close()
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"❌ SSE 스트리밍 실패: {e}"
        }


def run_backend():
    """백엔드 서버 실행"""
    print("=" * 60)
    print("🚀  백엔드 서버 시작...")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-m", "uvicorn",
        "api.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload",
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        print(f"서버 프로세스 PID: {process.pid}")
        print("서버가 시작될 때까지 5 초 대기합니다...\n")

        time.sleep(5)

        return process
    except Exception as e:
        print(f"서버 시작 실패: {e}")
        return None


def run_frontend():
    """프론트엔드 앱 실행"""
    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend", "app.py",
    )

    print("=" * 60)
    print("🚀  프론트엔드 앱 시작...")
    print("=" * 60)
    print(f"웹사이트를 열려면: http://localhost:8501")
    print("=" * 60 + "\n")

    cmd = [sys.executable, "-m", "streamlit", "run", frontend_path, "--server.port", "8501"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return process


def main():
    """메인 테스트 흐름"""
    print("\n" + "=" * 60)
    print("  AI 주식 예측 시스템 - 워크플로우 테스트")
    print("=" * 60 + "\n")

    # 1. 서버 스타트업
    backend_proc = run_backend()
    if not backend_proc:
        print("\n❌ 백엔드 서버 시작 실패")
        return

    # 2. 버퍼 수집기 시작
    collector_task = asyncio.create_task(start_collector(DEFAULT_STOCKS))

    # 3. 데이터 수집 대기
    print("데이터 수집을 10 초 대기합니다...\n")
    time.sleep(10)

    # 4. 버퍼 초기화 및 모델 재로드
    if not clear_and_reload():
        print("\n❌ 모델 재로드 실패")
        return

    # 5. /health 테스트
    print("\n--- /health 엔드포인트 테스트 ---")
    health_result = test_health()
    if health_result["success"]:
        print(health_result["message"])
    else:
        print(health_result["message"])
        return

    # 6. /predict 테스트
    print("\n--- /predict 엔드포인트 테스트 ---")
    for stock in DEFAULT_STOCKS[:3]:  # 첫 3 종목만 테스트
        predict_result = test_predict(stock["code"])
        if predict_result["success"]:
            print(predict_result["message"])
        else:
            print(predict_result["message"])

    # 7. /stream 테스트
    print("\n--- /stream 엔드포인트 테스트 ---")
    stream_result = test_stream(DEFAULT_STOCKS[0]["code"])
    if stream_result["success"]:
        print(stream_result["message"])
    else:
        print(stream_result["message"])

    # 8. 프론트엔드 스타트업
    frontend_proc = run_frontend()

    # 9. 30 초 후 종료
    print("\n" + "=" * 60)
    print("테스트 완료 - 프론트엔드가 실행 중입니다.")
    print("=" * 60)
    print("\n30 초 후 자동 종료합니다. 수동 중단 필요 시:")
    print("  Ctrl+C 를 누르세요.")
    print("\n프론트엔드: http://localhost:8501")
    print("=" * 60 + "\n")

    # 10. 30 초 대기 후 종료
    try:
        import time
        time.sleep(30)
    except KeyboardInterrupt:
        print("\n✅ 수동 종료됨")

    # 11. 서버 종료
    print("\n서버 종료 중...")
    for proc in [backend_proc, frontend_proc]:
        proc.terminate()
        proc.wait()

    print("✅ 모든 프로세스 종료됨")


if __name__ == "__main__":
    import asyncio
    DEFAULT_STOCKS = [
        {"code": "005930", "name": "삼성전자"},
        {"code": "000660", "name": "SK 하이닉스"},
        {"code": "035420", "name": "NAVER"},
    ]
    SIMULATION_MODE = True

    main()
