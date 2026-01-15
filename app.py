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
    items = re.split(r"[,\n\s]+", raw.strip())
    return [x.strip().upper() for x in items if x.strip()]

def compute_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df, MA_FAST, MA_SLOW, VOL_LOOKBACK, ATR_PERIOD):
    df = df.copy()
    df["MA_FAST"] = df["Close"].rolling(MA_FAST).mean()
    df["MA_SLOW"] = df["Close"].rolling(MA_SLOW).mean()
    df["VOL_AVG"] = df["Volume"].rolling(VOL_LOOKBACK).mean()
    df["VOL_RATIO"] = np.where(df["VOL_AVG"] > 0, df["Volume"] / df["VOL_AVG"], np.nan)
    df["ATR"] = compute_atr(df, ATR_PERIOD)
    df["ATR_PCT"] = df["ATR"] / df["Close"]
    return df

# =========================
# 매수 / 매도 로직
# =========================
def rule_signal(last, params):
    if pd.isna(last["MA_FAST"]) or pd.isna(last["MA_SLOW"]):
        return 0
    trend = last["MA_FAST"] > last["MA_SLOW"] and last["Close"] > last["MA_FAST"]
    volume = last["VOL_RATIO"] >= params["VOL_SPIKE"]
    atr_ok = params["ATR_PCT_MIN"] <= last["ATR_PCT"] <= params["ATR_PCT_MAX"]
    return int(trend and volume and atr_ok)

def sell_recommendation(last, params, entry_price, entry_date):
    if entry_price is None or np.isnan(entry_price) or entry_price <= 0:
        return "N/A", "평단 입력 필요", None, None, None

    close = float(last["Close"])
    atr = float(last["ATR"])
    ma_fast = float(last["MA_FAST"])
    ma_slow = float(last["MA_SLOW"])

    stop = entry_price - params["STOP_ATR_MULT"] * atr
    target = entry_price + 2 * (entry_price - stop)

    hold_days = None
    if entry_date:
        try:
            d0 = datetime.strptime(entry_date, "%Y-%m-%d").date()
            hold_days = (datetime.now().date() - d0).days
        except:
            pass

    if close < stop:
        return "SELL", "손절가 이탈", stop, target, hold_days
    if close >= target:
        return "PARTIAL SELL", "목표가(2R) 도달", stop, target, hold_days
    if close < ma_fast or ma_fast < ma_slow:
        return "PARTIAL SELL", "추세 이탈", stop, target, hold_days
    if hold_days is not None and hold_days >= params["HOLD_DAYS"]:
        return "SELL", "보유 기간 초과", stop, target, hold_days

    return "HOLD", "추세 유지", stop, target, hold_days

# =========================
# 근거 표
# =========================
def build_reason_table(last, params):
    return pd.DataFrame([
        ["MA_FAST > MA_SLOW", last["MA_FAST"] > last["MA_SLOW"]],
        ["Close > MA_FAST", last["Close"] > last["MA_FAST"]],
        ["VOL_RATIO ≥ 기준", last["VOL_RATIO"] >= params["VOL_SPIKE"]],
        ["ATR% 범위", params["ATR_PCT_MIN"] <= last["ATR_PCT"] <= params["ATR_PCT_MAX"]],
    ], columns=["조건", "통과"])

# =========================
# 데이터 로드
# =========================
def load_us(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=f"{DEFAULTS['LOOKBACK_YEARS']}y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )

    # ✅ MultiIndex 방어 (미국주식에서 필수)
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)

        # (ticker, field) 구조
        if ticker in lv0:
            df = df[ticker]

        # (field, ticker) 구조
        elif ticker in lv1:
            df = df.xs(ticker, axis=1, level=1)

        else:
            # 최후의 안전장치: 첫 ticker 블록 선택
            uniq1 = list(pd.unique(lv1))
            if uniq1:
                df = df.xs(uniq1[0], axis=1, level=1)
            else:
                raise ValueError(f"Unexpected MultiIndex columns: {df.columns}")

    # 컬럼명 정규화
    df = df.rename(columns=lambda c: str(c).title())

    # Close 없고 Adj Close만 있는 경우
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    keep = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"US data missing columns: {missing} / columns={list(df.columns)}")

    return df[keep].dropna()


def load_kr(code):
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365*DEFAULTS["LOOKBACK_YEARS"])).strftime("%Y-%m-%d")
    df = krx.get_market_ohlcv_by_date(start, end, code)
    df = df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
    return df[["Open","High","Low","Close","Volume"]].dropna()

# =========================
# 엑셀
# =========================
def apply_currency_formats(ws):
    fmt_krw = u'₩#,##0'
    fmt_usd = u'$#,##0.00'
    header = {ws.cell(1, c).value: c for c in range(1, ws.max_column+1)}
    if "market" not in header:
        return
    for r in range(2, ws.max_row+1):
        mkt = ws.cell(r, header["market"]).value
        fmt = fmt_krw if mkt=="KR" else fmt_usd if mkt=="US" else None
        if not fmt:
            continue
        for col in ["close","stop","target"]:
            if col in header:
                ws.cell(r, header[col]).number_format = fmt

def build_excel(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Signals")
        apply_currency_formats(w.book["Signals"])
    return buf.getvalue()

# =========================
# UI
# =========================
st.set_page_config(page_title="Swing Scanner", layout="wide")

st.markdown(
    "<h1 style='font-size:36px;font-weight:700;'>웹 티커 입력 → 스윙 판단(O/X) + 매도 추천 + 엑셀</h1>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("전략 설정")
    params = {}
    for k in ["MA_FAST","MA_SLOW","VOL_LOOKBACK","ATR_PERIOD","HOLD_DAYS"]:
        params[k] = st.number_input(k, value=DEFAULTS[k])
    for k in ["VOL_SPIKE","ATR_PCT_MIN","ATR_PCT_MAX","STOP_ATR_MULT"]:
        params[k] = st.number_input(k, value=float(DEFAULTS[k]))

raw = st.text_area("티커 입력", "005930\nSPY QQQ")
run = st.button("분석 실행")

if run:
    tickers = normalize_tickers(raw)

    rows, detail = [], {}
    for t in tickers:
        mkt = "KR" if is_kr_code(t) else "US"
        df = load_kr(t) if mkt=="KR" else load_us(t)
        df = add_indicators(df, params["MA_FAST"], params["MA_SLOW"],
                            params["VOL_LOOKBACK"], params["ATR_PERIOD"])
        last = df.iloc[-1]
        rows.append({"market":mkt,"ticker":t,"O/X":"O" if rule_signal(last,params) else "X","close":last["Close"]})
        detail[t] = (df,last)

    df_res = pd.DataFrame(rows)
    st.dataframe(df_res, use_container_width=True)

    # ===============================
    # session_state 기반 보유 입력
    # ===============================
    if "positions" not in st.session_state:
        st.session_state["positions"] = pd.DataFrame({
            "ticker": df_res["ticker"],
            "entry_price": np.nan,
            "entry_date": ""
        })

    # 티커 동기화 옵션
    cur = st.session_state["positions"]
    if set(cur["ticker"]) != set(df_res["ticker"]):
        st.session_state["positions"] = pd.merge(
            pd.DataFrame({"ticker": df_res["ticker"]}),
            cur,
            on="ticker",
            how="left"
        )

    positions = st.data_editor(
        st.session_state["positions"],
        key="positions_editor",
        use_container_width=True
    )
    st.session_state["positions"] = positions

    # 매도 추천
    for i, r in df_res.iterrows():
        t = r["ticker"]
        df,last = detail[t]
        pos = positions[positions["ticker"]==t].iloc[0]
        sig,reason,_,_,_ = sell_recommendation(last, params, pos["entry_price"], pos["entry_date"])
        with st.expander(f"{t} → {sig}"):
            st.write(reason)
            st.dataframe(build_reason_table(last,params))
            st.line_chart(df[["Close","MA_FAST","MA_SLOW"]])
            st.line_chart(df[["Volume","VOL_AVG"]])

    # 엑셀
    st.download_button(
        "엑셀 다운로드",
        data=build_excel(df_res),
        file_name="swing_result.xlsx"
    )
