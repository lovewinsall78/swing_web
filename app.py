# ============================================
# Swing Scanner - Final Stable Version
# ============================================

import re
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock as krx
import streamlit as st
from datetime import datetime, timedelta

# ============================================
# 기본 설정
# ============================================
DEFAULTS = dict(
    MA_FAST=20,
    MA_SLOW=60,
    ATR_PERIOD=14,
    VOL_LOOKBACK=20,
    VOL_SPIKE=1.5,
    ATR_PCT_MIN=0.008,
    ATR_PCT_MAX=0.060,
)

KR_UNIVERSE = [
    "005930","000660","035420","035720","051910","068270",
    "207940","005380","000270","012330"
]

US_UNIVERSE = [
    "SPY","QQQ","AAPL","MSFT","NVDA",
    "AMZN","META","TSLA","AVGO","AMD"
]

# ============================================
# 유틸
# ============================================
def is_kr_code(x: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", x.strip()))

def normalize_tickers(raw: str):
    return [x.strip().upper() for x in re.split(r"[,\n\s]+", raw.strip()) if x.strip()]

def parse_entry_text(market: str, s: str):
    if not s:
        return np.nan
    t = str(s).replace("₩","").replace("$","").replace(",","").strip()
    try:
        return int(float(t)) if market=="KR" else round(float(t),2)
    except:
        return np.nan

def format_currency(mkt, v):
    if pd.isna(v): return ""
    return f"₩{int(v):,}" if mkt=="KR" else f"${float(v):,.2f}"

# ============================================
# 지표 계산
# ============================================
def compute_atr(df, p):
    h,l,c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def add_indicators(df, p):
    df = df.copy()
    df["MA_FAST"] = df["Close"].rolling(p["MA_FAST"]).mean()
    df["MA_SLOW"] = df["Close"].rolling(p["MA_SLOW"]).mean()
    df["VOL_AVG"] = df["Volume"].rolling(p["VOL_LOOKBACK"]).mean()
    df["VOL_RATIO"] = df["Volume"]/df["VOL_AVG"]
    df["ATR"] = compute_atr(df, p["ATR_PERIOD"])
    df["ATR_PCT"] = df["ATR"]/df["Close"]
    return df

def buy_signal(last, p):
    return (
        last["MA_FAST"] > last["MA_SLOW"] and
        last["Close"] > last["MA_FAST"] and
        last["VOL_RATIO"] >= p["VOL_SPIKE"] and
        p["ATR_PCT_MIN"] <= last["ATR_PCT"] <= p["ATR_PCT_MAX"]
    )

def score_row(last):
    score = 0
    if last["MA_FAST"] > last["MA_SLOW"]:
        score += 40
    score += min(25, max(0, int((last["VOL_RATIO"]-1)*20)))
    score += min(20, max(0, int(last["ATR_PCT"]*200)))
    return score

# ============================================
# 데이터 로드
# ============================================
def load_us(t):
    df = yf.download(t, period="2y", progress=False)
    if isinstance(df.columns,pd.MultiIndex):
        df = df.xs(t,axis=1,level=1)
    df.columns=[c.title() for c in df.columns]
    return df[["Open","High","Low","Close","Volume"]].dropna()

def load_kr(t):
    e=datetime.now().strftime("%Y-%m-%d")
    s=(datetime.now()-timedelta(days=365*2)).strftime("%Y-%m-%d")
    df=krx.get_market_ohlcv_by_date(s,e,t)
    df=df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
    return df[["Open","High","Low","Close","Volume"]].dropna()

# ============================================
# 단일 종목 분석
# ============================================
def analyze_one(t, params):
    try:
        market = "KR" if is_kr_code(t) else "US"
        df = load_kr(t) if market=="KR" else load_us(t)
        df = add_indicators(df, params)
        last = df.iloc[-1]
        cand = buy_signal(last, params)
        score = score_row(last) if cand else 0

        return {
            "market": market,
            "ticker": t,
            "candidate": int(cand),
            "score": score,
            "close": round(last["Close"],2),
            "error": ""
        }
    except Exception as e:
        return {
            "market": "?",
            "ticker": t,
            "candidate": 0,
            "score": 0,
            "close": None,
            "error": str(e)
        }

# ============================================
# 추천 TOP10
# ============================================
def recommend_top10(params, top_n=10):
    universe = KR_UNIVERSE + US_UNIVERSE
    rows=[]
    for t in universe:
        rows.append(analyze_one(t, params))
    df = pd.DataFrame(rows)
    df_ok = df[df["candidate"]==1].sort_values("score", ascending=False)
    picks = df_ok["ticker"].tolist()[:top_n]

    if len(picks)<top_n:
        rest = df.sort_values("score", ascending=False)["ticker"].tolist()
        for t in rest:
            if t not in picks:
                picks.append(t)
            if len(picks)>=top_n:
                break
    return picks, df

# ============================================
# Streamlit UI
# ============================================
st.set_page_config(layout="wide")
st.markdown("<h1 style='font-size:32px;font-weight:700'>스윙 전략 자동 추천 시스템</h1>", unsafe_allow_html=True)

# Session
if "ticker_input_text" not in st.session_state:
    st.session_state["ticker_input_text"] = ""

if "positions_editor_df" not in st.session_state:
    st.session_state["positions_editor_df"] = pd.DataFrame(
        columns=["ticker","market","entry_text","entry_date"]
    )

# Sidebar
with st.sidebar:
    st.header("전략 설정")
    params={}
    params["MA_FAST"]=st.number_input("MA_FAST",5,200,DEFAULTS["MA_FAST"])
    params["MA_SLOW"]=st.number_input("MA_SLOW",10,300,DEFAULTS["MA_SLOW"])
    params["VOL_LOOKBACK"]=st.number_input("VOL_LOOKBACK",5,200,DEFAULTS["VOL_LOOKBACK"])
    params["VOL_SPIKE"]=st.number_input("VOL_SPIKE",1.0,5.0,DEFAULTS["VOL_SPIKE"])
    params["ATR_PERIOD"]=st.number_input("ATR_PERIOD",5,100,DEFAULTS["ATR_PERIOD"])
    params["ATR_PCT_MIN"]=st.number_input("ATR_PCT_MIN",0.0,0.2,DEFAULTS["ATR_PCT_MIN"])
    params["ATR_PCT_MAX"]=st.number_input("ATR_PCT_MAX",0.0,0.5,DEFAULTS["ATR_PCT_MAX"])

# 추천 버튼 + 입력
col1,col2 = st.columns([2,1])
with col1:
    if st.button("스윙 전략 기준 TOP10 추천"):
        picks, df_scan = recommend_top10(params)
        kr = [t for t in picks if is_kr_code(t)]
        us = [t for t in picks if not is_kr_code(t)]
        txt=""
        if kr: txt+=" ".join(kr)
        if us: txt+=("\n" if txt else "")+" ".join(us)
        st.session_state["ticker_input_text"]=txt

with col2:
    if st.button("입력 초기화"):
        st.session_state["ticker_input_text"]=""

raw = st.text_area(
    "티커 입력",
    value=st.session_state["ticker_input_text"],
    height=120
)
st.session_state["ticker_input_text"]=raw

# ============================================
# 분석 실행
# ============================================
if st.button("분석 실행"):
    rows=[]
    for t in normalize_tickers(raw):
        rows.append(analyze_one(t, params))
    df_res=pd.DataFrame(rows)
    st.subheader("분석 결과")
    st.dataframe(df_res.sort_values(["candidate","score"],ascending=[False,False]), use_container_width=True)

    # 보유 입력 동기화
    base=df_res[["ticker","market"]]
    st.session_state["positions_editor_df"]=base.merge(
        st.session_state["positions_editor_df"],
        on=["ticker","market"],
        how="left"
    ).fillna("")

# ============================================
# 보유 입력
# ============================================
if not st.session_state["positions_editor_df"].empty:
    st.subheader("보유 입력 (KR/US 통화 구분)")
    edited=st.data_editor(
        st.session_state["positions_editor_df"],
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "ticker":st.column_config.TextColumn(disabled=True),
            "market":st.column_config.TextColumn(disabled=True),
            "entry_text":st.column_config.TextColumn("평단 입력"),
            "entry_date":st.column_config.TextColumn("진입일"),
        }
    )
    st.session_state["positions_editor_df"]=edited

    calc=edited.copy()
    calc["entry_price"]=[
        parse_entry_text(m,t)
        for m,t in zip(calc["market"],calc["entry_text"])
    ]
    calc["표시"]=calc.apply(lambda r:format_currency(r["market"],r["entry_price"]),axis=1)

    st.caption("계산된 평단 (읽기 전용)")
    st.dataframe(calc[["ticker","market","표시"]], use_container_width=True)
