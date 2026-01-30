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
    STOP_ATR_MULT=1.8, HOLD_DAYS=20, LOOKBACK_YEARS=2,
    ACCOUNT_SIZE=10_000_000, RISK_PER_TRADE=0.01, TOP_N=10,
)

KR_UNIVERSE = ["005930","000660","035420","035720","051910","068270","207940","005380","000270","012330"]
US_UNIVERSE = ["SPY","QQQ","NVDA","AAPL","MSFT","TSLA","AMZN","GOOGL","META","AMD"]

st.set_page_config(page_title="Swing Scanner Final", layout="wide")

# -----------------------------
# 2. Utils
# -----------------------------
def is_kr_code(x: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", str(x).strip()))

def normalize_tickers(raw: str):
    items = re.split(r"[,\n\s]+", (raw or "").strip())
    return [x.strip().upper() for x in items if x.strip()]

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
        return f"₩{int(v):,}" if mkt == "KR" else f"${float(v):,.2f}"
    except: return str(v)

# -----------------------------
# 3. Session State & Callbacks
# -----------------------------
if "pos_df" not in st.session_state:
    st.session_state.pos_df = pd.DataFrame(columns=["market","ticker","name","entry_text","entry_price","entry_display","entry_date"])
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "analysis_detail" not in st.session_state:
    st.session_state.analysis_detail = {}
if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = "005930 NVDA"

def on_pos_edit():
    if "pos_editor" in st.session_state:
        ed = st.session_state["pos_editor"]["edited_rows"]
        for idx, changes in ed.items():
            idx_int = int(idx)
            mkt = st.session_state.pos_df.at[idx_int, "market"]
            if "entry_text" in changes:
                new_text = changes["entry_text"]
                new_val = parse_entry_val(mkt, new_text)
                st.session_state.pos_df.at[idx_int, "entry_price"] = float(new_val)
                st.session_state.pos_df.at[idx_int, "entry_display"] = format_curr(mkt, new_val)

# -----------------------------
# 4. Data Engine
# -----------------------------
@st.cache_data(ttl=3600)
def get_company_name(ticker):
    try:
        if is_kr_code(ticker): return krx.get_market_ticker_name(ticker) or ticker
        return yf.Ticker(ticker).info.get("shortName", ticker)
    except: return ticker

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
    if df.empty: return {"error": f"Data Error: {ticker}", "ticker": ticker, "name": ticker, "OX": "X", "candidate": 0}, None, None
    
    # 지표 계산
    df["MA_FAST"] = df["Close"].rolling(int(p["MA_FAST"])).mean()
    df["MA_SLOW"] = df["Close"].rolling(int(p["MA_SLOW"])).mean()
    df["VOL_AVG"] = df["Volume"].rolling(int(p["VOL_LOOKBACK"])).mean()
    tr = pd.concat([(df["High"]-df["Low"]), (df["High"]-df["Close"].shift()).abs(), (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(int(p["ATR_PERIOD"])).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_AVG"]
    df["ATR_PCT"] = df["ATR"] / df["Close"]
    
    last = df.iloc[-1]
    c1 = (last["MA_FAST"] > last["MA_SLOW"]) and (last["Close"] > last["MA_FAST"])
    c2 = last["VOL_RATIO"] >= p["VOL_SPIKE"]
    c3 = p["ATR_PCT_MIN"] <= last["ATR_PCT"] <= p["ATR_PCT_MAX"]
    cand = 1 if (c1 and c2 and c3) else 0
    
    stop_dist = p["STOP_ATR_MULT"] * last["ATR"]
    qty = int((p["ACCOUNT_SIZE"] * p["RISK_PER_TRADE"]) / stop_dist) if stop_dist > 0 else 0
    
    res = {
        "market": "KR" if is_kr_code(ticker) else "US",
        "ticker": ticker,
        "name": get_company_name(ticker),
        "OX": "O" if cand else "X",
        "candidate": cand,
        "score": (40 if c1 else 0) + int(min(25, max(0, (last["VOL_RATIO"]-1)*20))),
        "close": float(last["Close"]),
        "stop": float(last["Close"] - stop_dist),
        "target_2R": float(last["Close"] + (stop_dist * 2)),
        "qty": qty,
        "vol_ratio": float(last["VOL_RATIO"]),
        "atr_pct": float(last["ATR_PCT"] * 100),
        "date": str(df.index[-1].date()),
        "error": ""
    }
    
    rereasons = pd.DataFrame([
        {"조건": "이평선 정배열", "값": f"{last['MA_FAST']:.0f} > {last['MA_SLOW']:.0f}", "통과": last['MA_FAST'] > last['MA_SLOW']},
        {"조건": "거래량 급증", "값": f"{last['VOL_RATIO']:.2f}x", "통과": c2},
        {"조건": "변동성(ATR%)", "값": f"{last['ATR_PCT']*100:.2f}%", "통과": c3}
    ])
    return res, df, rereasons

# -----------------------------
# 5. UI Layout
# -----------------------------
with st.sidebar:
    st.header("⚙️ 전략 설정")
    params = {k: st.number_input(k, value=v) for k, v in DEFAULTS.items() if k not in ["LOOKBACK_YEARS", "TOP_N"]}
    params["LOOKBACK_YEARS"] = DEFAULTS["LOOKBACK_YEARS"]
    
    if st.button("추천 종목 스캔"):
        with st.spinner("스캔 중..."):
            all_u = KR_UNIVERSE + US_UNIVERSE
            picks = []
            for t in all_u:
                r, _, _ = analyze_one(t, params)
                if r.get("candidate"): picks.append(r)
            if picks:
                st.session_state.ticker_input = " ".join(pd.DataFrame(picks).sort_values("score", ascending=False).head(10)["ticker"].tolist())
                st.rerun()

st.title("⚖️ Swing Scanner FINAL")
raw_input = st.text_area("티커 입력", value=st.session_state.ticker_input)

if st.button("🚀 분석 실행"):
    tickers = normalize_tickers(raw_input)
    results = []
    details = {}
    for t in tickers:
        res, df, reason = analyze_one(t, params)
        results.append(res)
        if df is not None: details[t] = (df, reason)
    st.session_state.analysis_df = pd.DataFrame(results)
    st.session_state.analysis_detail = details
    
    new_rows = []
    for _, row in st.session_state.analysis_df.iterrows():
        exist = st.session_state.pos_df[st.session_state.pos_df["ticker"] == row["ticker"]]
        if not exist.empty: new_rows.append(exist.iloc[0].to_dict())
        else: new_rows.append({"market": row["market"], "ticker": row["ticker"], "name": row["name"], "entry_text": "", "entry_price": 0.0, "entry_display": "", "entry_date": None})
    st.session_state.pos_df = pd.DataFrame(new_rows)

# -----------------------------
# 6. 결과 출력 (포지션 입력 즉시 반영)
# -----------------------------
if st.session_state.analysis_df is not None:
    st.subheader("📥 보유 종목 평단 입력")
    st.data_editor(st.session_state.pos_df, key="pos_editor", on_change=on_pos_edit,
        column_config={"entry_text": st.column_config.TextColumn("평단가 입력"), "entry_display": st.column_config.TextColumn("✅ 계산된 평단", disabled=True), "entry_date": st.column_config.DateColumn("진입일"), "market": None, "entry_price": None},
        hide_index=True, use_container_width=True)

    st.subheader("🔍 분석 결과 요약")
    df_view = st.session_state.analysis_df.copy()
    
    def get_sell_sig(r):
        pos = st.session_state.pos_df[st.session_state.pos_df["ticker"] == r["ticker"]]
        if pos.empty or not pos.iloc[0]["entry_price"]: return "N/A", "-"
        entry = pos.iloc[0]["entry_price"]
        curr = r["close"]
        stop_val = entry * 0.95 # 예시: 단순 -5% 손절
        if curr < stop_val: return "SELL", "손절가 이탈"
        if curr > entry * 1.2: return "PARTIAL", "목표가 도달"
        return "HOLD", "유지"

    df_view[["매도신호", "매도근거"]] = df_view.apply(lambda r: pd.Series(get_sell_sig(r)), axis=1)
    
    # 통화 포맷팅
    for col in ["close", "stop", "target_2R"]:
        df_view[col] = df_view.apply(lambda r: format_curr(r["market"], r[col]), axis=1)
        
    st.dataframe(df_view, use_container_width=True, hide_index=True)

    # 엑셀 다운로드 (엔진 변경: xlsxwriter -> openpyxl)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_view.to_excel(writer, index=False, sheet_name='Summary')
    st.download_button("📂 엑셀 다운로드", output.getvalue(), "Swing_Report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
