import re
import io
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock as krx
import streamlit as st
from datetime import datetime, timedelta

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
    STOP_ATR_MULT=1.8,
    HOLD_DAYS=20,
    LOOKBACK_YEARS=2,
)

# =========================
# 유틸
# =========================
def is_kr_code(x):
    return bool(re.fullmatch(r"\d{6}", x.strip()))

def normalize_tickers(raw):
    return [x for x in re.split(r"[,\n\s]+", raw.strip()) if x]

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
    df["VOL_RATIO"] = df["Volume"] / df["VOL_AVG"]
    df["ATR"] = compute_atr(df, p["ATR_PERIOD"])
    df["ATR_PCT"] = df["ATR"] / df["Close"]
    return df

# =========================
# 매수 / 매도
# =========================
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
        except: pass

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
# 데이터
# =========================
def load_us(t):
    df = yf.download(t, period=f"{DEFAULTS['LOOKBACK_YEARS']}y", progress=False)
    if isinstance(df.columns,pd.MultiIndex):
        df = df.xs(t,axis=1,level=1)
    df.columns = [c.title() for c in df.columns]
    return df[["Open","High","Low","Close","Volume"]].dropna()

def load_kr(t):
    e = datetime.now().strftime("%Y-%m-%d")
    s = (datetime.now()-timedelta(days=365*DEFAULTS["LOOKBACK_YEARS"])).strftime("%Y-%m-%d")
    df = krx.get_market_ohlcv_by_date(s,e,t)
    df = df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
    return df[["Open","High","Low","Close","Volume"]].dropna()

# =========================
# UI
# =========================
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='font-size:36px;font-weight:700'>스윙 트레이딩 분석기 (매수·매도·근거)</h1>",
    unsafe_allow_html=True
)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    p = {}
    for k in DEFAULTS:
        p[k] = st.number_input(k, value=DEFAULTS[k])

# -------------------------
# Session State (절대 리셋 방지)
# -------------------------
if "positions" not in st.session_state:
    st.session_state["positions"] = pd.DataFrame(
        columns=["ticker","entry_price","entry_date"]
    )

raw = st.text_area("티커 입력","005930 SPY QQQ")
run = st.button("분석 실행")

if run:
    tickers = normalize_tickers(raw)
    rows, detail = [], {}

    for t in tickers:
        m = "KR" if is_kr_code(t) else "US"
        df = load_kr(t) if m=="KR" else load_us(t)
        df = add_indicators(df,p)
        last = df.iloc[-1]

        rows.append({
            "market":m,
            "ticker":t,
            "O/X":"O" if buy_signal(last,p) else "X",
            "close":last["Close"]
        })
        detail[t]=(df,last)

    df_res = pd.DataFrame(rows)

    # ---- 포지션 동기화 (값 유지)
    base = pd.DataFrame({"ticker":df_res["ticker"]})
    st.session_state["positions"] = base.merge(
        st.session_state["positions"], on="ticker", how="left"
    )

    # ---- 평단 입력
    st.subheader("보유 입력 (절대 초기화 안 됨)")
    pos = st.data_editor(
        st.session_state["positions"],
        key="pos_editor",
        use_container_width=True
    )
    st.session_state["positions"] = pos

    # ---- 결과
    st.subheader("결과 + 매도 추천")

    for _,r in df_res.iterrows():
        t = r["ticker"]
        df,last = detail[t]
        row = pos[pos["ticker"]==t].iloc[0]

        if pd.notna(row["entry_price"]):
            sig,reason,stop,target,hd = sell_signal(
                last,row["entry_price"],p,row["entry_date"]
            )
        else:
            sig,reason,stop,target,hd = "N/A","평단 필요",None,None,None

        color = {"SELL":"red","PARTIAL SELL":"orange","HOLD":"gray"}.get(sig,"black")

        with st.expander(f"{t} | 매도: {sig}"):
            st.markdown(f"<b style='color:{color}'>{sig}</b> - {reason}",unsafe_allow_html=True)

            st.line_chart(
                df[["Close","MA_FAST","MA_SLOW"]]
            )

            if stop and target:
                st.write(f"평단: {row['entry_price']}, 손절: {stop:.2f}, 목표: {target:.2f}")

