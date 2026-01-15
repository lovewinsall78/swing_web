import re
import io
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock as krx
import streamlit as st
from datetime import datetime, timedelta
import altair as alt
from openpyxl import load_workbook

# =========================
# 기본값
# =========================
DEFAULTS = dict(
    MA_FAST=20,
    MA_SLOW=60,
    ATR_PERIOD=14,
    VOL_LOOKBACK=20,
    VOL_SPIKE=1.5,
    ATR_PCT_MIN=0.008,
    ATR_PCT_MAX=0.060,
    ACCOUNT_SIZE=10_000_000,
    RISK_PER_TRADE=0.01,
    STOP_ATR_MULT=1.8,
    HOLD_DAYS=20,
    LOOKBACK_YEARS=2,
)

# =========================
# 유틸
# =========================
def is_kr_code(x: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", x.strip()))

def normalize_tickers(raw: str):
    return [x for x in re.split(r"[,\n\s]+", raw.strip()) if x]

def compute_atr(df, p):
    h,l,c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def add_indicators(df, params):
    df = df.copy()
    df["MA_FAST"] = df["Close"].rolling(params["MA_FAST"]).mean()
    df["MA_SLOW"] = df["Close"].rolling(params["MA_SLOW"]).mean()
    df["VOL_AVG"] = df["Volume"].rolling(params["VOL_LOOKBACK"]).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_AVG"]
    df["ATR"] = compute_atr(df, params["ATR_PERIOD"])
    df["ATR_PCT"] = df["ATR"] / df["Close"]
    return df

def buy_signal(last, p):
    return (
        last["MA_FAST"] > last["MA_SLOW"]
        and last["Close"] > last["MA_FAST"]
        and last["VOL_RATIO"] >= p["VOL_SPIKE"]
        and p["ATR_PCT_MIN"] <= last["ATR_PCT"] <= p["ATR_PCT_MAX"]
    )

def sell_signal(last, entry, p, entry_date):
    atr = last["ATR"]
    close = last["Close"]
    stop = entry - p["STOP_ATR_MULT"] * atr
    target = entry + 2*(entry-stop)

    hold_days = None
    if entry_date:
        try:
            hold_days = (datetime.now().date() - datetime.strptime(entry_date,"%Y-%m-%d").date()).days
        except:
            pass

    if close < stop:
        return "SELL","손절가 이탈",stop,target,hold_days
    if close >= target:
        return "PARTIAL SELL","목표가 도달",stop,target,hold_days
    if close < last["MA_FAST"] or last["MA_FAST"] < last["MA_SLOW"]:
        return "PARTIAL SELL","추세 이탈",stop,target,hold_days
    if hold_days and hold_days >= p["HOLD_DAYS"]:
        return "SELL","보유기간 초과",stop,target,hold_days

    return "HOLD","추세 유지",stop,target,hold_days

# =========================
# 데이터 로드
# =========================
def load_us(t):
    df = yf.download(t, period=f"{DEFAULTS['LOOKBACK_YEARS']}y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(t, axis=1, level=1)
    df.columns = [c.title() for c in df.columns]
    return df[["Open","High","Low","Close","Volume"]].dropna()

def load_kr(t):
    e = datetime.now().strftime("%Y-%m-%d")
    s = (datetime.now()-timedelta(days=365*DEFAULTS["LOOKBACK_YEARS"])).strftime("%Y-%m-%d")
    df = krx.get_market_ohlcv_by_date(s,e,t)
    df = df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
    return df[["Open","High","Low","Close","Volume"]].dropna()

# =========================
# 페이지 설정 & 타이틀
# =========================
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='font-size:36px;font-weight:700'>스윙 트레이딩 분석기 (매수·매도·근거·엑셀)</h1>",
    unsafe_allow_html=True
)

# =========================
# 세션 상태
# =========================
if "analysis_df" not in st.session_state:
    st.session_state["analysis_df"] = None
if "detail" not in st.session_state:
    st.session_state["detail"] = {}
if "positions" not in st.session_state:
    st.session_state["positions"] = pd.DataFrame(columns=["ticker","entry_price","entry_date"])
if "ACCOUNT_SIZE" not in st.session_state:
    st.session_state["ACCOUNT_SIZE"] = DEFAULTS["ACCOUNT_SIZE"]

# =========================
# 사이드바
# =========================
with st.sidebar:
    st.header("📊 스윙 전략 설정")
    params = {}

    with st.expander("① 추세 판단", True):
        params["MA_FAST"] = st.number_input("MA_FAST", 5, 200, DEFAULTS["MA_FAST"])
        params["MA_SLOW"] = st.number_input("MA_SLOW", 10, 300, DEFAULTS["MA_SLOW"])

    with st.expander("② 거래량 · 변동성"):
        params["VOL_LOOKBACK"] = st.number_input("VOL_LOOKBACK", 5, 200, DEFAULTS["VOL_LOOKBACK"])
        params["VOL_SPIKE"] = st.number_input("VOL_SPIKE", 1.0, 5.0, DEFAULTS["VOL_SPIKE"], step=0.05)
        params["ATR_PERIOD"] = st.number_input("ATR_PERIOD", 5, 100, DEFAULTS["ATR_PERIOD"])
        params["ATR_PCT_MIN"] = st.number_input("ATR_PCT_MIN", 0.0, 0.2, DEFAULTS["ATR_PCT_MIN"], step=0.001)
        params["ATR_PCT_MAX"] = st.number_input("ATR_PCT_MAX", 0.0, 0.5, DEFAULTS["ATR_PCT_MAX"], step=0.001)

    with st.expander("③ 리스크 · 보유"):
        params["STOP_ATR_MULT"] = st.number_input("STOP_ATR_MULT", 0.5, 5.0, DEFAULTS["STOP_ATR_MULT"], step=0.1)
        params["HOLD_DAYS"] = st.number_input("HOLD_DAYS", 1, 200, DEFAULTS["HOLD_DAYS"])

    with st.expander("④ 계좌 가정값 (천단위 콤마 입력)"):
        acc_str = st.text_input(
            "ACCOUNT_SIZE",
            value=f"{int(st.session_state['ACCOUNT_SIZE']):,}"
        )
        try:
            params["ACCOUNT_SIZE"] = int(acc_str.replace(",",""))
            st.session_state["ACCOUNT_SIZE"] = params["ACCOUNT_SIZE"]
        except:
            params["ACCOUNT_SIZE"] = st.session_state["ACCOUNT_SIZE"]

        params["RISK_PER_TRADE"] = st.number_input(
            "RISK_PER_TRADE",
            0.001, 0.05, DEFAULTS["RISK_PER_TRADE"],
            step=0.001, format="%.3f"
        )

# =========================
# 입력 & 실행
# =========================
raw = st.text_area("티커 입력", "005930 SPY QQQ")
run = st.button("분석 실행")

if run:
    tickers = normalize_tickers(raw)
    rows, detail = [], {}

    for t in tickers:
        m = "KR" if is_kr_code(t) else "US"
        df = load_kr(t) if m=="KR" else load_us(t)
        df = add_indicators(df, params)
        last = df.iloc[-1]

        rows.append({
            "market":m,
            "ticker":t,
            "O/X":"O" if buy_signal(last,params) else "X",
            "close":last["Close"]
        })
        detail[t] = df

    st.session_state["analysis_df"] = pd.DataFrame(rows)
    st.session_state["detail"] = detail

    base = pd.DataFrame({"ticker":st.session_state["analysis_df"]["ticker"]})
    st.session_state["positions"] = base.merge(
        st.session_state["positions"], on="ticker", how="left"
    )

# =========================
# 결과 표시
# =========================
if st.session_state["analysis_df"] is not None:
    st.subheader("보유 입력")
    st.session_state["positions"] = st.data_editor(
        st.session_state["positions"],
        use_container_width=True
    )

    st.subheader("결과")
    st.dataframe(st.session_state["analysis_df"], use_container_width=True)
