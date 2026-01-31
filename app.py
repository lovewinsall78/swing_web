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
# 1. Defaults & Config
# -----------------------------
DEFAULTS = dict(
    MA_FAST=20, MA_SLOW=60, ATR_PERIOD=14, VOL_LOOKBACK=20,
    VOL_SPIKE=1.5, ATR_PCT_MIN=0.008, ATR_PCT_MAX=0.060,
    STOP_ATR_MULT=1.8, ACCOUNT_SIZE=10_000_000, RISK_PER_TRADE=0.01,
    LOOKBACK_YEARS=2
)

KR_UNIVERSE = ["005930","000660","035420","035720","051910","068270","207940","005380","000270","012330","066570","003550"]
US_UNIVERSE = ["SPY","QQQ","NVDA","AAPL","MSFT","TSLA","AMZN","GOOGL","META","AMD","AVGO","NFLX"]

st.set_page_config(page_title="Swing Scanner Final Pro", layout="wide")

# -----------------------------
# 2. Utility Functions
# -----------------------------
def is_kr_code(x: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", str(x).strip()))

def normalize_tickers(raw: str):
    items = re.split(r"[,\n\s]+", (raw or "").strip())
    return [x.strip().upper() for x in items if x.strip()]

@st.cache_data(ttl=3600)
def get_company_name(t):
    if t == "BTC-USD": return "Bitcoin"
    try:
        if is_kr_code(t): return krx.get_market_ticker_name(t) or t
        return yf.Ticker(t).info.get("shortName", t)
    except: return t

def parse_entry_val(market, text):
    if not text or pd.isna(text) or str(text).strip() == "": return 0.0
    raw = str(text).replace("₩","").replace("$","").replace(",","").strip()
    try:
        val = float(raw)
        return float(int(val)) if market == "KR" else round(val, 2)
    except: return 0.0

def format_curr(mkt, v):
    if not v or v == 0 or pd.isna(v): return ""
    try:
        return f"₩{int(round(float(v))):,}" if mkt == "KR" else f"${float(v):,.2f}"
    except: return str(v)

# -----------------------------
# 3. Session State & Callbacks
# -----------------------------
if "pos_df" not in st.session_state:
    st.session_state.pos_df = pd.DataFrame(columns=["market","ticker","name","entry_text","entry_price","entry_display","entry_date"])
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = "BTC-USD 005930 NVDA"

def on_pos_edit():
    if "pos_editor" in st.session_state:
        ed = st.session_state["pos_editor"]["edited_rows"]
        for idx, changes in ed.items():
            idx_int = int(idx)
            mkt = st.session_state.pos_df.at[idx_int, "market"]
            if "entry_text" in changes:
                new_text = changes["entry_text"]
                new_val = parse_entry_val(mkt, new_text)
                st.session_state.pos_df.at[idx_int, "entry_text"] = str(new_text)
                st.session_state.pos_df.at[idx_int, "entry_price"] = float(new_val)
                st.session_state.pos_df.at[idx_int, "entry_display"] = format_curr(mkt, new_val)

# -----------------------------
# 4. Analysis Engine
# -----------------------------
@st.cache_data(ttl=1200)
def load_data(ticker, years):
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

def analyze_one(ticker, p):
    df = load_data(ticker, p["LOOKBACK_YEARS"])
    if df.empty: return {"candidate": 0, "ticker": ticker, "score": 0, "error": "Data Error"}, None
    
    df["MA_FAST"] = df["Close"].rolling(int(p["MA_FAST"])).mean()
    df["MA_SLOW"] = df["Close"].rolling(int(p["MA_SLOW"])).mean()
    df["VOL_AVG"] = df["Volume"].rolling(int(p["VOL_LOOKBACK"])).mean()
    tr = pd.concat([(df["High"]-df["Low"]), (df["High"]-df["Close"].shift()).abs(), (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(int(p["ATR_PERIOD"])).mean()
    
    last = df.iloc[-1]
    vol_ratio = last["Volume"] / last["VOL_AVG"] if last["VOL_AVG"] > 0 else 0
    atr_pct = last["ATR"] / last["Close"]
    
    c1 = (last["MA_FAST"] > last["MA_SLOW"]) and (last["Close"] > last["MA_FAST"])
    c2 = vol_ratio >= p["VOL_SPIKE"]
    c3 = p["ATR_PCT_MIN"] <= atr_pct <= p["ATR_PCT_MAX"]
    
    score = (40 if c1 else 0) + int(min(30, vol_ratio * 10)) + (30 if c3 else 0)
    cand = 1 if (c1 and c2 and c3) else 0
    stop_dist = p["STOP_ATR_MULT"] * last["ATR"]
    
    res = {
        "market": "KR" if is_kr_code(ticker) else "US",
        "ticker": ticker,
        "name": get_company_name(ticker),
        "OX": "O" if cand else "X",
        "candidate": cand,
        "score": score,
        "close": float(last["Close"]),
        "stop": float(last["Close"] - stop_dist),
        "target": float(last["Close"] + stop_dist * 2),
        "vol_ratio": float(vol_ratio),
        "atr_pct": float(atr_pct * 100),
        "error": ""
    }
    return res, df

# -----------------------------
# 5. Sidebar (Params)
# -----------------------------
with st.sidebar:
    st.header("⚙️ 전략 파라미터 설정")
    p = {}
    p["MA_FAST"] = st.number_input("단기 이평선", value=DEFAULTS["MA_FAST"], help="단기 추세선 (예: 20일)")
    p["MA_SLOW"] = st.number_input("장기 이평선", value=DEFAULTS["MA_SLOW"], help="장기 추세선 (예: 60일)")
    p["ATR_PERIOD"] = st.number_input("ATR 기간", value=DEFAULTS["ATR_PERIOD"], help="변동성 평균 기간")
    p["VOL_LOOKBACK"] = st.number_input("거래량 평균 기간", value=DEFAULTS["VOL_LOOKBACK"])
    p["VOL_SPIKE"] = st.number_input("거래량 급증 배수", value=DEFAULTS["VOL_SPIKE"], help="평균 대비 돌파 배수")
    p["ATR_PCT_MIN"] = st.number_input("최소 변동성(ATR%)", value=DEFAULTS["ATR_PCT_MIN"], format="%.3f")
    p["ATR_PCT_MAX"] = st.number_input("최대 변동성(ATR%)", value=DEFAULTS["ATR_PCT_MAX"], format="%.3f")
    p["STOP_ATR_MULT"] = st.number_input("손절 ATR 배수", value=DEFAULTS["STOP_ATR_MULT"])
    p["ACCOUNT_SIZE"] = st.number_input("총 투자 원금", value=DEFAULTS["ACCOUNT_SIZE"])
    p["RISK_PER_TRADE"] = st.number_input("회당 리스크(%)", value=DEFAULTS["RISK_PER_TRADE"], format="%.2f")
    p["LOOKBACK_YEARS"] = DEFAULTS["LOOKBACK_YEARS"]
    params = p

# -----------------------------
# 6. Main UI
# -----------------------------
st.title("⚖️ Swing Scanner Final Pro")

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("🌟 국산5+외산5 추천"):
        with st.spinner("조건에 맞는 종목 스캔 중..."):
            # KR 추천 (에러 방지용 빈 리스트 체크 포함)
            kr_candidates = [analyze_one(t, params)[0] for t in KR_UNIVERSE]
            kr_filtered = [x for x in kr_candidates if x["candidate"] == 1]
            kr_top = pd.DataFrame(kr_filtered).sort_values("score", ascending=False).head(5)["ticker"].tolist() if kr_filtered else []
            
            # US 추천
            us_candidates = [analyze_one(t, params)[0] for t in US_UNIVERSE]
            us_filtered = [x for x in us_candidates if x["candidate"] == 1]
            us_top = pd.DataFrame(us_filtered).sort_values("score", ascending=False).head(5)["ticker"].tolist() if us_filtered else []
            
            # 비트코인 상시 포함하여 업데이트
            st.session_state.ticker_input = " ".join(["BTC-USD"] + kr_top + us_top)
            st.rerun()

ticker_area = st.text_area("분석 티커 입력", value=st.session_state.ticker_input, height=100)

if st.button("🚀 분석 실행", type="primary"):
    tickers = normalize_tickers(ticker_area)
    results = []
    for t in tickers:
        res, _ = analyze_one(t, params)
        results.append(res)
    st.session_state.analysis_df = pd.DataFrame(results)
    
    new_rows = []
    for _, row in st.session_state.analysis_df.iterrows():
        exist = st.session_state.pos_df[st.session_state.pos_df["ticker"] == row["ticker"]]
        if not exist.empty:
            new_rows.append(exist.iloc[0].to_dict())
        else:
            new_rows.append({
                "market": row["market"], "ticker": row["ticker"], "name": row["name"],
                "entry_text": "", "entry_price": 0.0, "entry_display": "", "entry_date": None
            })
    st.session_state.pos_df = pd.DataFrame(new_rows)
    st.session_state.pos_df["entry_date"] = pd.to_datetime(st.session_state.pos_df["entry_date"])

# -----------------------------
# 7. 결과 화면 및 엑셀 다운로드
# -----------------------------
if st.session_state.analysis_df is not None:
    st.subheader("📥 보유 종목 평단 관리")
    st.data_editor(st.session_state.pos_df, key="pos_editor", on_change=on_pos_edit,
        column_config={
            "entry_text": st.column_config.TextColumn("평단가 입력"),
            "entry_display": st.column_config.TextColumn("✅ 계산된 평단", disabled=True),
            "entry_date": st.column_config.DateColumn("진입일"),
            "market": None, "entry_price": None, "name": st.column_config.TextColumn("종목명", disabled=True)
        }, hide_index=True, use_container_width=True)

    st.subheader("🔍 분석 결과 및 매도 추천")
    df_view = st.session_state.analysis_df.copy()
    
    def get_signal_info(r):
        pos = st.session_state.pos_df[st.session_state.pos_df["ticker"] == r["ticker"]]
        if pos.empty or not pos.iloc[0]["entry_price"]: return "HOLD", "-", 0.0
        entry = pos.iloc[0]["entry_price"]
        curr = r["close"]
        profit_pct = (curr - entry) / entry * 100
        if curr < entry * 0.95: return "🔴 SELL", "손절", profit_pct
        if curr > entry * 1.15: return "🟢 TAKE", "익절", profit_pct
        return "⚪ HOLD", "유지", profit_pct

    sig_data = df_view.apply(lambda r: pd.Series(get_signal_info(r)), axis=1)
    df_view[["Signal", "Reason", "Profit%"]] = sig_data
    
    disp_df = df_view.copy()
    for col in ["close", "stop", "target"]:
        disp_df[col] = disp_df.apply(lambda r: format_curr(r["market"], r[col]), axis=1)
    st.dataframe(disp_df, use_container_width=True, hide_index=True)

    # 엑셀 다운로드 (보고서 + 보유평단 포함)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_view.to_excel(writer, index=False, sheet_name='분석_결과')
        export_pos = st.session_state.pos_df.merge(df_view[["ticker", "close", "Profit%"]], on="ticker", how="left")
        export_pos.to_excel(writer, index=False, sheet_name='나의_포트폴리오')
    
    st.download_button("📂 분석 결과 + 포트폴리오 엑셀 다운로드", output.getvalue(), f"Swing_Report_{datetime.now().strftime('%Y%m%d')}.xlsx")

st.markdown("---")
st.caption("Swing Scanner Final Pro | 추천 종목이 없을 때 발생하는 정렬 오류를 수정했습니다.")
