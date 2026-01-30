import re
import io
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock as krx
import streamlit as st
from datetime import datetime, timedelta
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

KR_UNIVERSE = ["005930","000660","035420","035720","051910","068270","207940","005380","000270"]
US_UNIVERSE = ["SPY","QQQ","NVDA","AAPL","MSFT","TSLA","AMZN","GOOGL","META"]

st.set_page_config(page_title="Swing Scanner Pro", layout="wide")

# -----------------------------
# 2. 핵심 유틸리티 (평단 계산 및 포맷)
# -----------------------------
def is_kr_code(x: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", str(x).strip()))

def parse_entry_val(market, text):
    if not text or pd.isna(text): return np.nan
    raw = str(text).replace("₩","").replace("$","").replace(",","").strip()
    try:
        val = float(raw)
        return int(val) if market == "KR" else round(val, 2)
    except: return np.nan

def format_curr(mkt, v):
    if pd.isna(v) or v == "": return ""
    return f"₩{int(v):,}" if mkt == "KR" else f"${float(v):,.2f}"

# -----------------------------
# 3. 데이터 로딩 (안정성 강화)
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
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=lambda x: x.title())
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except:
        return pd.DataFrame()

# -----------------------------
# 4. 분석 엔진 (지표 & 시그널)
# -----------------------------
def analyze_ticker(ticker, p):
    df = load_stock_data(ticker, p["LOOKBACK_YEARS"])
    if df.empty: return {"error": "Data Error"}, None
    
    df["MA_FAST"] = df["Close"].rolling(p["MA_FAST"]).mean()
    df["MA_SLOW"] = df["Close"].rolling(p["MA_SLOW"]).mean()
    df["VOL_AVG"] = df["Volume"].rolling(p["VOL_LOOKBACK"]).mean()
    df["ATR"] = (df["High"]-df["Low"]).rolling(p["ATR_PERIOD"]).mean() # 간략화된 ATR
    
    last = df.iloc[-1]
    vol_ratio = last["Volume"]/last["VOL_AVG"] if last["VOL_AVG"] > 0 else 0
    atr_pct = last["ATR"]/last["Close"]
    
    c1 = last["MA_FAST"] > last["MA_SLOW"]
    c2 = last["Close"] > last["MA_FAST"]
    c3 = vol_ratio >= p["VOL_SPIKE"]
    c4 = p["ATR_PCT_MIN"] <= atr_pct <= p["ATR_PCT_MAX"]
    
    cand = 1 if (c1 and c2 and c3 and c4) else 0
    stop_buy = last["Close"] - (p["STOP_ATR_MULT"] * last["ATR"])
    qty = int((p["ACCOUNT_SIZE"] * p["RISK_PER_TRADE"]) / (last["Close"] - stop_buy)) if last["Close"] > stop_buy else 0

    res = {
        "market": "KR" if is_kr_code(ticker) else "US",
        "ticker": ticker,
        "OX": "O" if cand else "X",
        "candidate": cand,
        "close": last["Close"],
        "stop": stop_buy,
        "target": last["Close"] + (last["Close"] - stop_buy) * 2,
        "qty": qty,
        "vol_ratio": vol_ratio,
        "atr_pct": atr_pct * 100,
        "error": ""
    }
    return res, df

# -----------------------------
# 5. 세션 상태 관리 (동기화 핵심)
# -----------------------------
if "pos_df" not in st.session_state:
    st.session_state.pos_df = pd.DataFrame(columns=["market","ticker","entry_text","entry_price","entry_display","entry_date"])

def update_positions():
    """에디터 변경 시 즉시 평단 계산 및 디스플레이 업데이트"""
    ed = st.session_state["pos_editor"]["edited_rows"]
    for idx, changes in ed.items():
        curr_row = st.session_state.pos_df.iloc[idx]
        mkt = curr_row["market"]
        
        if "entry_text" in changes:
            new_val = parse_entry_val(mkt, changes["entry_text"])
            st.session_state.pos_df.at[idx, "entry_price"] = new_val
            st.session_state.pos_df.at[idx, "entry_display"] = format_curr(mkt, new_val)
        if "entry_date" in changes:
            st.session_state.pos_df.at[idx, "entry_date"] = changes["entry_date"]

# -----------------------------
# 6. UI 레이아웃
# -----------------------------
st.title("🚀 Swing Scanner Pro")

with st.sidebar:
    st.header("⚙️ 전략 파라미터")
    params = {k: st.number_input(k, value=v) for k, v in DEFAULTS.items() if k != "LOOKBACK_YEARS"}
    params["LOOKBACK_YEARS"] = DEFAULTS["LOOKBACK_YEARS"]

# 티커 입력 및 분석
ticker_raw = st.text_area("티커 입력 (공백/줄바꿈 구분)", "005930 NVDA AAPL 000660")
if st.button("분석 실행"):
    tickers = list(set(re.split(r"[,\s\n]+", ticker_raw.strip().upper())))
    results = []
    details = {}
    
    for t in tickers:
        if not t: continue
        res, df = analyze_ticker(t, params)
        results.append(res)
        details[t] = df
    
    st.session_state.analysis_results = pd.DataFrame(results)
    st.session_state.details = details
    
    # 분석 시점에 포지션 테이블 초기화/업데이트
    new_pos = st.session_state.analysis_results[["market","ticker"]].copy()
    # 기존 데이터 보존 병합
    st.session_state.pos_df = new_pos.merge(st.session_state.pos_df, on=["market","ticker"], how="left").fillna("")

# -----------------------------
# 7. 보유 종목 입력창 (즉시 반영 로직 적용)
# -----------------------------
if not st.session_state.pos_df.empty:
    st.subheader("📋 보유 종목 관리 (입력 즉시 계산)")
    st.data_editor(
        st.session_state.pos_df,
        key="pos_editor",
        on_change=update_positions,
        column_config={
            "entry_text": st.column_config.TextColumn("평단가 입력 (직접 입력)"),
            "entry_display": st.column_config.TextColumn("✅ 계산된 평단", disabled=True),
            "entry_date": st.column_config.DateColumn("진입일"),
            "market": st.column_config.TextColumn(disabled=True),
            "ticker": st.column_config.TextColumn(disabled=True),
            "entry_price": st.column_config.NumberColumn(disabled=True)
        },
        hide_index=True,
        use_container_width=True
    )

# -----------------------------
# 8. 결과 및 차트 출력
# -----------------------------
if "analysis_results" in st.session_state:
    df_res = st.session_state.analysis_results
    st.subheader("🔍 스캔 결과")
    
    # 매도 시그널 계산 로직 결합
    def get_sell_sig(row):
        tkr = row["ticker"]
        pos_row = st.session_state.pos_df[st.session_state.pos_df["ticker"] == tkr]
        if pos_row.empty or not pos_row.iloc[0]["entry_price"]: return "N/A"
        
        price = row["close"]
        entry = pos_row.iloc[0]["entry_price"]
        # 예시: 평단 대비 -5% 손절
        if price < entry * 0.95: return "🔴 SELL (Stop)"
        if price > entry * 1.15: return "🟢 TAKE PROFIT"
        return "⚪ HOLD"

    df_res["Sell_Signal"] = df_res.apply(get_sell_sig, axis=1)
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    # 개별 종목 상세 (차트)
    for _, r in df_res.iterrows():
        with st.expander(f"{r['ticker']} 상세 분석"):
            df_plot = st.session_state.details[r['ticker']]
            c = alt.Chart(df_plot.reset_index()).mark_line().encode(x='Date:T', y='Close:Q')
            st.altair_chart(c, use_container_width=True)

# -----------------------------
# 9. 엑셀 다운로드
# -----------------------------
if "analysis_results" in st.session_state:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        st.session_state.analysis_results.to_excel(writer, index=False, sheet_name='Results')
    st.download_button("엑셀 다운로드", output.getvalue(), "swing_report.xlsx")

st.markdown("---")
st.caption("Would you like me to add Relative Strength (RS) comparison or more technical indicators?")
