"""
frontend/app.py - Streamlit 실시간 대시보드
────────────────────────────────────────────────────────────────
주요 화면:
  [1] 사이드바  - 종목 선택, 갱신 주기 설정, 시스템 상태
  [2] 상단 KPI - 현재가 / 등락률 / AI 신호 / 신뢰도
  [3] 메인 차트 - 실시간 캔들(틱) + MA5/MA20 오버레이
  [4] 지표 탭  - RSI, MACD, Bollinger Band 시각화
  [5] 포트폴리오 - 전체 종목 요약 & 신호 테이블

실행:
  streamlit run frontend/app.py
────────────────────────────────────────────────────────────────
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from config import API_HOST, API_PORT, DEFAULT_STOCKS

API_BASE = f"http://{API_HOST}:{API_PORT}"


# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────

st.set_page_config(
    page_title = "AI 주식 예측 대시보드",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS 스타일 ─────────────────────────────────
st.markdown("""
<style>
  .kpi-box {
    background: #1e1e2e; border-radius: 12px; padding: 16px 20px;
    text-align: center; border: 1px solid #313244;
  }
  .kpi-label { color: #a6adc8; font-size: 0.8rem; margin-bottom: 4px; }
  .kpi-value { color: #cdd6f4; font-size: 1.6rem; font-weight: 700; }
  .kpi-up    { color: #a6e3a1; }
  .kpi-dn    { color: #f38ba8; }
  .kpi-hold  { color: #89b4fa; }
  .signal-buy  { background:#1e4620; color:#a6e3a1; padding:4px 10px;
                 border-radius:6px; font-weight:700; }
  .signal-sell { background:#4a1c1c; color:#f38ba8; padding:4px 10px;
                 border-radius:6px; font-weight:700; }
  .signal-hold { background:#1e2a3a; color:#89b4fa; padding:4px 10px;
                 border-radius:6px; font-weight:700; }
  .stMetric > div > div { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# API 헬퍼
# ─────────────────────────────────────────────

@st.cache_data(ttl=2)
def fetch_json(path: str):
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(f"{API_BASE}{path}")
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


def fetch_predict(code: str):
    return fetch_json(f"/predict/{code}")


def fetch_data(code: str, n: int = 200):
    r = fetch_json(f"/data/{code}?n={n}")
    if r and r.get("data"):
        df = pd.DataFrame(r["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return pd.DataFrame()


def fetch_all_predictions():
    return fetch_json("/predict") or []


def fetch_health():
    return fetch_json("/health") or {}


# ─────────────────────────────────────────────
# 차트 함수
# ─────────────────────────────────────────────

_COLORS = {
    "bg"     : "#1e1e2e",
    "grid"   : "#313244",
    "text"   : "#cdd6f4",
    "up"     : "#a6e3a1",
    "down"   : "#f38ba8",
    "ma5"    : "#89dceb",
    "ma20"   : "#fab387",
    "volume" : "#585b70",
    "macd"   : "#89b4fa",
    "signal" : "#f5c2e7",
    "rsi"    : "#cba6f7",
}

def _base_layout(title: str = "") -> dict:
    return dict(
        title      = title,
        paper_bgcolor = _COLORS["bg"],
        plot_bgcolor  = _COLORS["bg"],
        font = dict(color=_COLORS["text"], size=11),
        xaxis = dict(gridcolor=_COLORS["grid"], showgrid=True),
        yaxis = dict(gridcolor=_COLORS["grid"], showgrid=True),
        margin = dict(l=50, r=20, t=35, b=30),
        showlegend = True,
        legend = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )


def render_price_chart(df: pd.DataFrame, code: str, name: str):
    """캔들스틱 + MA5/MA20 + 거래량 복합 차트."""
    if df.empty:
        st.info("데이터 수집 중... 잠시 기다려 주세요.")
        return

    df = df.tail(200).copy()
    price = df["price"]

    df["ma5"]  = price.rolling(5).mean()
    df["ma20"] = price.rolling(20).mean()

    fig = go.Figure()

    # ── 캔들스틱 ──────────────────────────────
    fig.add_trace(go.Candlestick(
        x     = df["timestamp"],
        open  = df["open"],
        high  = df["high"],
        low   = df["low"],
        close = df["price"],
        name  = f"{name}",
        increasing_line_color = _COLORS["up"],
        decreasing_line_color = _COLORS["down"],
        increasing_fillcolor  = _COLORS["up"],
        decreasing_fillcolor  = _COLORS["down"],
    ))

    # ── 이동평균 ──────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["ma5"],
        mode="lines", name="MA5",
        line=dict(color=_COLORS["ma5"], width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["ma20"],
        mode="lines", name="MA20",
        line=dict(color=_COLORS["ma20"], width=1.2),
    ))

    layout = _base_layout(f"{name}({code}) 실시간 차트")
    layout["xaxis_rangeslider_visible"] = False
    layout["height"] = 420
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # ── 거래량 바 차트 ────────────────────────
    colors = [
        _COLORS["up"] if c >= o else _COLORS["down"]
        for c, o in zip(df["price"], df["open"])
    ]
    fig_vol = go.Figure(go.Bar(
        x=df["timestamp"], y=df["tick_vol"],
        marker_color=colors, name="거래량",
    ))
    layout_vol = _base_layout("거래량")
    layout_vol["height"] = 120
    fig_vol.update_layout(**layout_vol)
    st.plotly_chart(fig_vol, use_container_width=True)


def render_indicators(df: pd.DataFrame):
    """RSI / MACD / 볼린저 밴드 지표 탭."""
    if df.empty or len(df) < 30:
        st.info("지표 계산을 위해 데이터 수집 중...")
        return

    df = df.copy()
    price = df["price"].astype(float)

    # RSI
    delta = price.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    df["macd"]   = ema12 - ema26
    df["macd_s"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_h"] = df["macd"] - df["macd_s"]

    # Bollinger
    bb_mid = price.rolling(20).mean()
    bb_std = price.rolling(20).std()
    df["bb_up"]  = bb_mid + 2 * bb_std
    df["bb_dn"]  = bb_mid - 2 * bb_std
    df["bb_mid"] = bb_mid

    tab1, tab2, tab3 = st.tabs(["📊 RSI", "📉 MACD", "🎯 볼린저 밴드"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["rsi"],
            name="RSI(14)", line=dict(color=_COLORS["rsi"], width=1.5)
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="#f38ba8", annotation_text="과매수(70)")
        fig.add_hline(y=30, line_dash="dash", line_color="#a6e3a1", annotation_text="과매도(30)")
        fig.update_layout(**_base_layout("RSI (14)"), height=250, yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        colors_hist = [
            _COLORS["up"] if v >= 0 else _COLORS["down"]
            for v in df["macd_h"].fillna(0)
        ]
        fig.add_trace(go.Bar(
            x=df["timestamp"], y=df["macd_h"],
            name="Histogram", marker_color=colors_hist
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["macd"],
            name="MACD", line=dict(color=_COLORS["macd"], width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["macd_s"],
            name="Signal", line=dict(color=_COLORS["signal"], width=1.5)
        ))
        fig.update_layout(**_base_layout("MACD"), height=250)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["bb_up"],
            name="상단밴드", line=dict(color="#585b70", width=1),
            fill=None,
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["bb_dn"],
            name="하단밴드", line=dict(color="#585b70", width=1),
            fill="tonexty", fillcolor="rgba(88,91,112,0.15)",
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["bb_mid"],
            name="중간밴드", line=dict(color=_COLORS["ma20"], width=1),
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=price,
            name="현재가", line=dict(color=_COLORS["up"], width=1.5),
        ))
        fig.update_layout(**_base_layout("볼린저 밴드 (20, 2σ)"), height=300)
        st.plotly_chart(fig, use_container_width=True)


def render_portfolio_table(predictions: list):
    """전체 종목 요약 테이블."""
    if not predictions:
        st.info("예측 데이터 없음")
        return

    rows = []
    for p in predictions:
        signal = p.get("signal", "HOLD")
        emoji  = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🔵"}.get(signal, "⚪")
        rows.append({
            "종목명"  : p.get("name", p.get("code", "")),
            "코드"    : p.get("code", ""),
            "현재가"  : f"₩{p.get('price', 0):,.0f}",
            "AI신호"  : f"{emoji} {signal}",
            "신뢰도"  : f"{p.get('confidence', 0):.1f}%",
            "BUY%"    : f"{p.get('probas', {}).get('BUY', 0):.1f}",
            "HOLD%"   : f"{p.get('probas', {}).get('HOLD', 0):.1f}",
            "SELL%"   : f"{p.get('probas', {}).get('SELL', 0):.1f}",
            "데이터행" : p.get("rows_used", 0),
            "상태"    : "✅" if not p.get("error") else f"⚠️ {p['error']}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df, use_container_width=True,
        hide_index=True,
        column_config={
            "신뢰도"   : st.column_config.TextColumn("신뢰도"),
            "BUY%"    : st.column_config.ProgressColumn("BUY%", min_value=0, max_value=100, format="%.1f"),
            "HOLD%"   : st.column_config.ProgressColumn("HOLD%", min_value=0, max_value=100, format="%.1f"),
            "SELL%"   : st.column_config.ProgressColumn("SELL%", min_value=0, max_value=100, format="%.1f"),
        }
    )


# ─────────────────────────────────────────────
# 메인 대시보드
# ─────────────────────────────────────────────

def main():
    # ── 자동 갱신 ─────────────────────────────
    REFRESH_SEC = st.sidebar.slider("갱신 주기 (초)", 2, 30, 5)
    count = st_autorefresh(interval=REFRESH_SEC * 1000, key="autorefresh")

    # ── 사이드바 ──────────────────────────────
    st.sidebar.title("📈 AI 주식 예측")
    st.sidebar.caption(f"업데이트: {datetime.now().strftime('%H:%M:%S')}")

    # 서버 상태
    health = fetch_health()
    if health:
        status_color = "🟢" if health.get("model_ready") else "🟡"
        st.sidebar.markdown(
            f"{status_color} 서버 연결됨  \n"
            f"모델: {'로드됨' if health.get('model_ready') else '미로드 (학습 필요)'}  \n"
            f"버퍼: {health.get('buffer_rows', 0):,}행 / {health.get('buffer_codes', 0)}종목"
        )
    else:
        st.sidebar.error("🔴 서버 연결 실패\nuvicorn api.main:app 실행 확인")

    st.sidebar.divider()

    # 종목 선택
    stock_options = {s["name"]: s["code"] for s in DEFAULT_STOCKS}
    selected_name = st.sidebar.selectbox("종목 선택", list(stock_options.keys()))
    selected_code = stock_options[selected_name]

    # 표시 행 수
    n_rows = st.sidebar.slider("표시 틱 수", 50, 500, 200)

    st.sidebar.divider()
    st.sidebar.markdown(
        "**시작 방법**\n"
        "```bash\n"
        "# 1. 서버 시작\n"
        "uvicorn api.main:app --port 8000\n\n"
        "# 2. 모델 학습 (최초 1회)\n"
        "python models/train_model.py\n"
        "```"
    )

    # ── 메인 타이틀 ───────────────────────────
    st.title(f"📊 {selected_name} ({selected_code})")

    # ── 예측 & 데이터 수집 ─────────────────────
    pred = fetch_predict(selected_code)
    df   = fetch_data(selected_code, n=n_rows)

    # ── KPI 카드 ──────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    latest = df.iloc[-1] if not df.empty else None

    with c1:
        price     = latest["price"]     if latest is not None else 0
        change_rt = latest["change_rt"] if latest is not None else 0
        delta_cls = "kpi-up" if change_rt >= 0 else "kpi-dn"
        sign      = "+" if change_rt >= 0 else ""
        st.markdown(f"""
        <div class="kpi-box">
          <div class="kpi-label">현재가</div>
          <div class="kpi-value">₩{price:,.0f}</div>
          <div class="{delta_cls}">{sign}{change_rt:.2f}%</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        signal     = pred.get("signal", "N/A")  if pred else "N/A"
        confidence = pred.get("confidence", 0)   if pred else 0
        sig_cls    = {"BUY": "kpi-up", "SELL": "kpi-dn", "HOLD": "kpi-hold"}.get(signal, "kpi-hold")
        sig_emoji  = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🔵"}.get(signal, "⚪")
        st.markdown(f"""
        <div class="kpi-box">
          <div class="kpi-label">AI 신호</div>
          <div class="kpi-value {sig_cls}">{sig_emoji} {signal}</div>
          <div class="kpi-label">신뢰도 {confidence:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        buy_p  = pred.get("probas", {}).get("BUY",  0) if pred else 0
        hold_p = pred.get("probas", {}).get("HOLD", 0) if pred else 0
        sell_p = pred.get("probas", {}).get("SELL", 0) if pred else 0
        st.markdown(f"""
        <div class="kpi-box">
          <div class="kpi-label">확률 분포</div>
          <div style="font-size:0.85rem; color:#a6e3a1">▲ BUY  {buy_p:.1f}%</div>
          <div style="font-size:0.85rem; color:#89b4fa">● HOLD {hold_p:.1f}%</div>
          <div style="font-size:0.85rem; color:#f38ba8">▼ SELL {sell_p:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        vol = latest["volume"] if latest is not None else 0
        st.markdown(f"""
        <div class="kpi-box">
          <div class="kpi-label">누적 거래량</div>
          <div class="kpi-value">{vol:,}</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        rows_used = pred.get("rows_used", 0) if pred else 0
        buf_rows  = len(df)
        st.markdown(f"""
        <div class="kpi-box">
          <div class="kpi-label">버퍼 / 추론 데이터</div>
          <div class="kpi-value">{buf_rows:,}</div>
          <div class="kpi-label">추론 사용: {rows_used}행</div>
        </div>""", unsafe_allow_html=True)

    if pred and pred.get("error"):
        st.warning(f"⚠️ 예측 오류: {pred['error']}")

    st.divider()

    # ── 메인 차트 ─────────────────────────────
    render_price_chart(df, selected_code, selected_name)

    # ── 기술적 지표 ───────────────────────────
    with st.expander("📉 기술적 지표", expanded=True):
        render_indicators(df)

    st.divider()

    # ── 포트폴리오 전체 요약 ──────────────────
    st.subheader("📋 전체 종목 AI 신호 요약")
    all_preds = fetch_all_predictions()
    render_portfolio_table(all_preds)

    # ── 원시 데이터 (선택) ────────────────────
    with st.expander("🔍 원시 틱 데이터"):
        if not df.empty:
            st.dataframe(df.tail(50), use_container_width=True)
        else:
            st.info("데이터 없음")


if __name__ == "__main__":
    main()
