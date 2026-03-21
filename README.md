# AI 기반 국내 주가 예측 시스템

> KIS WebSocket × ONNX SLM × FastAPI × Streamlit  
> **Python 풀스택** 실시간 국내 주식 AI 예측 도구

───────────────────────────────────────────────────────────

## 프로젝트 구조

```
ai_trading_project/
├── api/
│   └── main.py           # FastAPI 추론 서버 (비동기 REST + SSE)
├── core/
│   ├── buffer.py         # 인메모리 슬라이딩 윈도우 버퍼
│   ├── features.py       # 기술적 지표 피처 계산 (학습/추론 공통)
│   └── inference.py      # ONNX 경량 모델 추론 엔진
├── data/                 # (자동 생성) 학습 캐시
├── frontend/
│   └── app.py            # Streamlit 실시간 대시보드
├── models/
│   ├── train_model.py    # PyKRX 데이터 수집 + XGBoost 학습 + ONNX 변환
│   ├── stock_model.onnx  # (학습 후 생성)
│   └── scaler.pkl        # (학습 후 생성)
├── collector.py          # KIS WebSocket 수집기 (시뮬 모드 내장)
├── config.py             # 중앙 설정 관리
├── requirements.txt
└── .env.example
```

───────────────────────────────────────────────────────────

## 빠른 시작

### 환경 설정

```bash
# 의존 패키지 설치 (최초 1회)
pip3 install -r requirements.txt

# 환경변수 파일 생성
cp .env.example .env
# .env 파일을 열어 KIS_APP_KEY, KIS_APP_SECRET 등 입력
# API 없이 테스트하려면 SIMULATION_MODE=true 유지 (기본값)
```

───────────────────────────────────────────────────────────

## 시뮬레이션 모드 실행 (KIS API 없이 테스트)

> API 키 없이 랜덤 주가 데이터로 전체 파이프라인을 테스트할 수 있습니다.  
> 터미널(Mac) 또는 명령 프롬프트/PowerShell(Windows)을 **3개** 열어서 순서대로 실행합니다.

───────────────────────────────────────────────────────────

### Mac

**터미널 1 — AI 모델 학습 (최초 1회, 약 3~5분 소요)**
```bash
cd ai_trading_project
python3 models/train_model.py
# 완료 메시지: ✅ 학습 완료! POST /model/reload 호출로 즉시 반영됩니다.
```

**터미널 2 — FastAPI 서버 시작**
```bash
cd ai_trading_project
python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# 정상 실행 시: 🚀 AI 국내 주식 예측 서버 시작 / ⚠️ 시뮬레이션 모드 시작
```

**터미널 3 — Streamlit 대시보드 시작**
```bash
cd ai_trading_project
python3 -m streamlit run frontend/app.py
# 브라우저 자동 열림: http://localhost:8501
```

**모델 학습 완료 후 반영 (서버 재시작 없이)**
```bash
curl -X POST http://localhost:8000/model/reload
```

**종료:** 각 터미널에서 `Ctrl + C`

───────────────────────────────────────────────────────────

### Windows

**명령 프롬프트 1 — AI 모델 학습 (최초 1회, 약 3~5분 소요)**
```cmd
cd ai_trading_project
python models/train_model.py
# 완료 메시지: ✅ 학습 완료! POST /model/reload 호출로 즉시 반영됩니다.
```

**명령 프롬프트 2 — FastAPI 서버 시작**
```cmd
cd ai_trading_project
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# 정상 실행 시: 🚀 AI 국내 주식 예측 서버 시작 / ⚠️ 시뮬레이션 모드 시작
```

**명령 프롬프트 3 — Streamlit 대시보드 시작**
```cmd
cd ai_trading_project
python -m streamlit run frontend/app.py
# 브라우저 자동 열림: http://localhost:8501
```

**모델 학습 완료 후 반영 (서버 재시작 없이)**
```cmd
curl -X POST http://localhost:8000/model/reload
```

**종료:** 각 명령 프롬프트에서 `Ctrl + C`

───────────────────────────────────────────────────────────

## 실전 모드 전환 (KIS API 연결)

`.env` 파일에서 아래 항목을 수정합니다:
```
SIMULATION_MODE=false
KIS_APP_KEY=발급받은_앱키
KIS_APP_SECRET=발급받은_앱시크릿
KIS_ACCOUNT_NO=계좌번호_앞8자리
```

───────────────────────────────────────────────────────────

## 모델 학습 상세

**Mac:**
```bash
# 기본 8개 종목 학습
python3 models/train_model.py

# 특정 종목만 학습
python3 models/train_model.py --codes 005930 000660 035420
```

**Windows:**
```cmd
# 기본 8개 종목 학습
python models/train_model.py

# 특정 종목만 학습
python models/train_model.py --codes 005930 000660 035420
```

───────────────────────────────────────────────────────────

## 아키텍처

```
KIS WebSocket ─→ collector.py ─→ core/buffer.py (슬라이딩 윈도우)
                                        │
                              api/main.py (FastAPI)
                                   │
                          core/inference.py (ONNX)
                                   │
                         frontend/app.py (Streamlit)
```

### 핵심 설계 결정

| 컴포넌트 | 선택 | 이유 |
|----------|------|------|
| 데이터 수집 | WebSocket (Push) | REST 폴링 대비 지연 없음 |
| 메모리 관리 | deque + 슬라이딩 윈도우 | OOM 방지, O(1) 삽입/삭제 |
| AI 모델 | XGBoost → ONNX | CPU 로컬 추론 최적화 |
| 백엔드 | FastAPI (async) | 비동기 I/O, 동시 추론 지원 |
| 프론트엔드 | Streamlit | pandas 직결, 빠른 PoC |

───────────────────────────────────────────────────────────

## AI 모델 상세

### 입력 피처 (19개)

| 카테고리 | 피처 |
|----------|------|
| 이동평균 | MA5, MA10, MA20, MA60, MA비율 3개 |
| 모멘텀 | RSI(14), MACD, MACD Signal, MACD Hist |
| 변동성 | Bollinger Band Width/%, ATR(14) |
| 거래량 | Volume Ratio 5일/20일, OBV(정규화) |
| 가격 | 가격범위, 등락률 |

### 예측 레이블

| 레이블 | 조건 |
|--------|------|
| 🟢 BUY  | N봉 후 수익률 ≥ +0.5% |
| 🔴 SELL | N봉 후 수익률 ≤ -0.5% |
| 🔵 HOLD | 그 외 |

> `PREDICTION_WINDOW` (기본값 5봉) 조정 가능 (`config.py`)

───────────────────────────────────────────────────────────

## 주요 API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/predict/{code}` | 단일 종목 AI 예측 |
| GET | `/predict` | 전체 종목 일괄 예측 |
| GET | `/data/{code}?n=200` | 최근 틱 데이터 |
| GET | `/buffer/stats` | 버퍼 상태 |
| POST | `/model/reload` | 재학습 후 핫-리로드 |
| GET | `/stream/{code}` | SSE 실시간 스트림 |

───────────────────────────────────────────────────────────

## 향후 업데이트 로드맵

- [ ] 해외 주식 (IBKR API / Alpaca) 연동
- [ ] LSTM / Transformer 모델 추가
- [ ] 자동매매 주문 연동 (KIS 주문 API)
- [ ] 포트폴리오 백테스팅 모듈
- [ ] 알림 (텔레그램 / 이메일)
- [ ] Docker Compose 배포 구성

───────────────────────────────────────────────────────────

## 면책 조항

> 이 프로그램은 **교육/연구 목적**으로 제작되었습니다.  
> AI 예측 결과는 투자 권유가 아니며, 실제 투자에 따른 손익은 사용자 본인의 책임입니다.
