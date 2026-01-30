import re
import io
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock as krx
import streamlit as st
from datetime import datetime, timedelta, date
import altair as alt

# -----------------------------
# 1. Defaults & Setup
# -----------------------------
DEFAULTS = dict(
    MA_FAST=20, MA_SLOW=60, ATR_PERIOD=14, VOL_LOOKBACK=20,
    VOL_SPIKE=1.5, ATR_PCT_MIN=0.008, ATR_PCT_MAX=0.060,
    STOP_ATR_MULT=1.8, HOLD_DAYS=20, LOOKBACK_YEARS=2,
    ACCOUNT_SIZE=10_000_000, RISK_PER_TRADE=0.01, TOP_N=10,
)

st.set_page_config(page_title="Swing Scanner Pro", layout="wide")

# -----------------------------
# 2. 유틸리티 함수
# -----------------------------
def is_kr_code(x: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", str(x).strip()))

def parse_entry_val(market, text):
    if not text or pd.isna(text) or str(text).strip() == "": return 0.0
    raw = str(text).replace("₩","").replace("$","").replace(",","").strip()
    try:
        val = float(raw)
        return float(int(val)) if market == "KR" else round(val, 2)
    except: return 0.0

def format_curr(mkt, v):
    if not v or v == 0: return ""
    return f"₩{int(v):,}" if mkt == "KR" else f"${float(v):,.2f}"

# -----------------------------
# 3. 세션 상태 초기화 (에러 방지의 핵심)
# -----------------------------
if "pos_df" not in st.session_state:
    # 타입을 명확하게 지정하여 생성 (에러 방지)
    st.session_state.pos_df = pd.DataFrame({
        "market": pd.Series([], dtype="str"),
        "ticker": pd.Series([], dtype="str"),
        "entry_text": pd.Series([], dtype="str"),
        "entry_price": pd.Series([], dtype="float"),
        "entry_display": pd.Series([], dtype="str"),
        "entry_date": pd.Series([], dtype="datetime64[ns]")
    })

def update_positions():
    """데이터 에디터 변경 시 호출되는 콜백"""
    # state에서 직접 편집된 내용을 가져옴
    if "pos_editor" in st.session_state:
        edited_rows = st.session_state["pos_editor"]["edited_rows"]
        for idx, changes in edited_rows.items():
            # 인덱스를 통해 원본 행 접근
            idx_int = int(idx)
            mkt = st.session_state.pos_df.at[idx_int, "market"]
            
            if "entry_text" in changes:
                new_text = changes["entry_text"]
                new_val = parse_entry_val(mkt, new_text)
                st.session_state.pos_df.at[idx_int, "entry_text"] = str(new_text)
                st.session_state.pos_df.at[idx_int, "entry_price"] = float(new_val)
                st.session_state.pos_df.at[idx_int, "entry_display"] = format_curr(mkt, new_val)
            
            if "entry_date" in changes:
                # DateColumn은 문자열로 들어올 수 있으므로 처리
                st.session_state.pos_df.at[idx_int, "entry_date"] = pd.to_datetime(changes["entry_date"])

# -----------------------------
# 4. 데이터 로딩 및 분석 (생략 가능하나 구조 유지를 위해 포함)
# -----------------------------
@st.cache_data(ttl=3600)
def load_stock_data(ticker, years):
    try:
        if is_kr_code(ticker):
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=365*years)).strftime("%Y%m%d")
            df = krx.get_market_ohlcv_by_date(start, end, ticker)
            df = df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
        else:
            df = yf.download(ticker, period=f"{years}y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=lambda x: x.title())
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except: return pd.DataFrame()

def analyze_ticker(ticker, p):
    df = load_stock_data(ticker, p["LOOKBACK_YEARS"])
    if df.empty: return {"error": "Data Error", "ticker": ticker, "market": "KR" if is_kr_code(ticker) else "US"}, None
    
    # 지표 계산 (단순화)
    close = df["Close"].iloc[-1]
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    res = {
        "market": "KR" if is_kr_code(ticker) else "US",
        "ticker": ticker,
        "OX": "O" if close > ma20 else "X",
        "close": float(close),
        "error": ""
    }
    return res, df

# -----------------------------
# 5. UI 메인
# -----------------------------
st.title("🚀 Swing Scanner Pro")

with st.sidebar:
    params = {k: st.number_input(k, value=v) for k, v in DEFAULTS.items() if k != "LOOKBACK_YEARS"}
    params["LOOKBACK_YEARS"] = DEFAULTS["LOOKBACK_YEARS"]

ticker_raw = st.text_area("티커 입력", "005930 NVDA")

if st.button("분석 실행"):
    tickers = list(set(re.split(r"[,\s\n]+", ticker_raw.strip().upper())))
    results = []
    
    for t in tickers:
        if not t: continue
        res, _ = analyze_ticker(t, params)
        results.append(res)
    
    st.session_state.analysis_results = pd.DataFrame(results)
    
    # 포지션 테이블 업데이트 (기존 데이터 유지 로직)
    new_data = []
    for _, row in st.session_state.analysis_results.iterrows():
        # 기존에 있던 티커인지 확인
        exists = st.session_state.pos_df[st.session_state.pos_df["ticker"] == row["ticker"]]
        if not exists.empty:
            new_data.append(exists.iloc[0].to_dict())
        else:
            new_data.append({
                "market": row["market"], "ticker": row["ticker"],
                "entry_text": "", "entry_price": 0.0, "entry_display": "", "entry_date": None
            })
    
    st.session_state.pos_df = pd.DataFrame(new_data)
    # 타입 재강제
    st.session_state.pos_df["entry_date"] = pd.to_datetime(st.session_state.pos_df["entry_date"])
    st.session_state.pos_df["entry_price"] = st.session_state.pos_df["entry_price"].astype(float)

# -----------------------------
# 6. 보유 종목 입력창 (에러 수정 지점)
# -----------------------------
if not st.session_state.pos_df.empty:
    st.subheader("📋 보유 종목 관리")
    st.info("💡 '평단가 입력'란에 숫자를 입력하고 Enter를 누르면 옆 칸에 즉시 계산됩니다.")
    
    # 에러 방지: 데이터프레임 복사본을 전달
    edited_df = st.data_editor(
        st.session_state.pos_df,
        key="pos_editor",
        on_change=update_positions,
        column_config={
            "market": st.column_config.TextColumn("시장", disabled=True),
            "ticker": st.column_config.TextColumn("티커", disabled=True),
            "entry_text": st.column_config.TextColumn("평단가 입력 (직접 입력)"),
            "entry_display": st.column_config.TextColumn("✅ 계산된 평단", disabled=True),
            "entry_date": st.column_config.DateColumn("진입일"),
            "entry_price": None # 숨김 처리
        },
        hide_index=True,
        use_container_width=True
    )

# -----------------------------
# 7. 결과 출력
# -----------------------------
if "analysis_results" in st.session_state:
    st.subheader("🔍 분석 결과 요약")
    st.dataframe(st.session_state.analysis_results, use_container_width=True, hide_index=True)
