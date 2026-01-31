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
# 3. Session State
# -----------------------------
if "pos_df" not in st.session_state:
    st.session_state.pos_df = pd.DataFrame(columns=["market","ticker","name","entry_text","entry_price","entry_display","entry_date"])
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = "BTC-USD 005930 NVDA"
if "msg" not in st.session_state:
    st.session_state.msg = ""

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
    df["VOL_AVG"] = df
