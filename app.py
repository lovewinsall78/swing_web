import re
import io
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock as krx
import streamlit as st
from datetime import datetime, timedelta

# ---------------- Defaults ----------------
DEFAULTS = dict(
    HOLD_DAYS=20,
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
    TOP_N=10
)

# ---------------- Helpers ----------------
def is_kr_code(x: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", x.strip()))

def normalize_tickers(raw: str):
    items = re.split(r"[,\n\s]+", raw.strip())
    items = [x.strip().upper() for x in items if x.strip()]
    return items

def compute_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df, MA_FAST, MA_SLOW, VOL_LOOKBACK, ATR_PERIOD):
    df = df.copy()
    df["MA_FAST"] = df["Close"].rolling(MA_FAST).mean()
    df["MA_SLOW"] = df["Close"].rolling(MA_SLOW).mean()
    df["VOL_AVG"] = df["Volume"].rolling(VOL_LOOKBACK).mean()
    df["VOL_RATIO"] = np.where(df["VOL_AVG"] > 0, df["Volume"] / df["VOL_AVG"], np.nan)
    df["ATR"] = compute_atr(df, ATR_PERIOD)
    df["ATR_PCT"] = df["ATR"] / df["Close"]
    df["RET_20"] = df["Close"].pct_change(20)
    return df

def rule_signal(last, VOL_SPIKE, ATR_PCT_MIN, ATR_PCT_MAX):
    needed = ["MA_FAST", "MA_SLOW", "VOL_RATIO", "ATR_PCT"]
    if any(pd.isna(last.get(k, np.nan)) for k in needed):
        return 0

    trend_ok = (last["MA_FAST"] > last["MA_SLOW"]) and (last["Close"] > last["MA_FAST"])
    vol_ok   = (last["VOL_RATIO"] >= VOL_SPIKE)
    atr_ok   = (ATR_PCT_MIN <= last["ATR_PCT"] <= ATR_PCT_MAX)
    return 1 if (trend_ok and vol_ok and atr_ok) else 0

def score_row(last):
    score = 0
    if last["MA_FAST"] > last["MA_SLOW"]:
        score += 40

    vr = last.get("VOL_RATIO", np.nan)
    if pd.notna(vr):
        score += int(min(25, max(0, (vr - 1.0) * 20)))

    r20 = last.get("RET_20", np.nan)
    if pd.notna(r20):
        score += int(min(20, max(0, r20 * 200)))

    ap = last.get("ATR_PCT", np.nan)
    if pd.notna(ap):
        if ap > 0.045:
            score -= 10
        elif ap < 0.010:
            score -= 5
        else:
            score += 10

    return int(score)

def position_size(entry, atr, ACCOUNT_SIZE, RISK_PER_TRADE, STOP_ATR_MULT):
    if pd.isna(atr) or atr <= 0:
        return 0, np.nan
    stop_dist = STOP_ATR_MULT * atr
    risk_amount = ACCOUNT_SIZE * RISK_PER_TRADE
    qty = int(risk_amount / stop_dist)
    stop_price = entry - stop_dist
    return max(0, qty), float(stop_price)

# ---------------- Data Loaders ----------------
def load_us(ticker: str):
    df = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )

    # MultiIndex 처리
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)

        if ticker in lv0:          # (티커, 필드)
            df = df[ticker]
        elif ticker in lv1:        # (필드, 티커)
            df = df.xs(ticker, axis=1, level=1)
        else:
            # 최후의 수단: 1개만 있는 경우가 많으므로 첫 번째 블록 선택
            # 하지만 중복 컬럼이 생기면 폭발하므로 "ticker" 축을 골라야 함
            uniq0 = list(pd.unique(lv0))
            uniq1 = list(pd.unique(lv1))
            if len(uniq0) > 1 and len(uniq1) <= 10:
                df = df.xs(uniq1[0], axis=1, level=1)
            elif len(uniq1) > 1 and len(uniq0) <= 10:
                df = df[uniq0[0]]
            else:
                raise ValueError(f"Unexpected MultiIndex columns: {df.columns}")

    df = df.rename(columns=lambda c: str(c).title())

    # Close 없고 Adj Close만 있으면 대체
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    keep = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"US data missing columns: {missing} / columns={list(df.columns)}")

    df = df[keep].dropna()
    return df

def load_kr(code: str):
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
    df = krx.get_market_ohlcv_by_date(start, end, code)
    df = df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    return df

# ---------------- Analysis ----------------
def analyze_one(ticker: str, params: dict):
    try:
        if is_kr_code(ticker):
            market = "KR"
            df = load_kr(ticker)
        else:
            market = "US"
            df = load_us(ticker)

        df = add_indicators(
            df,
            MA_FAST=params["MA_FAST"],
            MA_SLOW=params["MA_SLOW"],
            VOL_LOOKBACK=params["VOL_LOOKBACK"],
            ATR_PERIOD=params["ATR_PERIOD"],
        )

        last = df.iloc[-1]

        cand = rule_signal(
            last,
            VOL_SPIKE=params["VOL_SPIKE"],
            ATR_PCT_MIN=params["ATR_PCT_MIN"],
            ATR_PCT_MAX=params["ATR_PCT_MAX"],
        )

        sc = score_row(last) if cand == 1 else 0

        entry = float(last["Close"])
        qty, stop = position_size(
            entry, float(last["ATR"]),
            ACCOUNT_SIZE=params["ACCOUNT_SIZE"],
            RISK_PER_TRADE=params["RISK_PER_TRADE"],
            STOP_ATR_MULT=params["STOP_ATR_MULT"],
        )

        target = entry + 2 * (entry - stop) if (not np.isnan(stop)) else np.nan

        return {
            "market": market,
            "ticker": ticker,
            "O/X": "O" if cand == 1 else "X",
            "candidate": int(cand),
            "score": int(sc),
            "close": float(entry),
            "stop": (None if np.isnan(stop) else float(stop)),
            "target(2R)": (None if np.isnan(target) else float(target)),
            "qty": int(qty),
            "vol_ratio": (None if pd.isna(last["VOL_RATIO"]) else float(last["VOL_RATIO"])),
            "atr_pct(%)": (None if pd.isna(last["ATR_PCT"]) else float(last["ATR_PCT"])*100),
            "ret_20(%)": (None if pd.isna(last["RET_20"]) else float(last["RET_20"])*100),
            "date": str(df.index[-1].date()),
            "error": ""
        }
    except Exception as e:
        return {
            "market": "?",
            "ticker": ticker,
            "O/X": "X",
            "candidate": 0,
            "score": 0,
            "close": None,
            "stop": None,
            "target(2R)": None,
            "qty": 0,
            "vol_ratio": None,
            "atr_pct(%)": None,
            "ret_20(%)": None,
            "date": "",
            "error": str(e)
        }

# ---------------- Excel formatting (KRW/USD) ----------------
def apply_currency_formats_openpyxl(ws):
    """
    시트에서:
      - market 값이 KR이면 close/stop/target(2R) -> ₩#,##0
      - market 값이 US이면 close/stop/target(2R) -> $#,##0.00
    """
    fmt_krw = u'₩#,##0'
    fmt_usd = u'$#,##0.00'
    price_cols = {"close", "stop", "target(2R)"}

    # 헤더 찾기
    header = {}
    for col in range(1, ws.max_column + 1):
        v = ws.cell(row=1, column=col).value
        if isinstance(v, str):
            header[v.strip()] = col

    if "market" not in header:
        return

    market_col = header["market"]
    price_col_idxs = [header[c] for c in price_cols if c in header]
    if not price_col_idxs:
        return

    for r in range(2, ws.max_row + 1):
        mkt = ws.cell(row=r, column=market_col).value
        if mkt == "KR":
            numfmt = fmt_krw
        elif mkt == "US":
            numfmt = fmt_usd
        else:
            continue

        for c in price_col_idxs:
            cell = ws.cell(row=r, column=c)
            if isinstance(cell.value, (int, float)) and cell.value is not None:
                cell.number_format = numfmt

def build_excel(df_all: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="Signals_All", index=False)

        df_cand = df_all[df_all["candidate"] == 1].copy()
        if df_cand.empty:
            df_cand = pd.DataFrame([{"note": "nottoday"}])
        df_cand.to_excel(writer, sheet_name="Candidates", index=False)

        def order_sheet(df, market):
            d = df[(df["market"] == market) & (df["candidate"] == 1)].copy()
            if d.empty:
                return pd.DataFrame([{
                    "market": market, "ticker": "nottoday", "action": "NONE",
                    "qty": 0, "stop": "", "target(2R)": "", "note": "No candidates"
                }])
            d = d.sort_values("score", ascending=False).head(10)
            d["qty_safe"] = (d["qty"] * 0.5).astype(int)
            return pd.DataFrame({
                "market": d["market"],
                "ticker": d["ticker"],
                "action": "BUY",
                "qty": d["qty_safe"],
                "stop": d["stop"],
                "target(2R)": d["target(2R)"],
                "note": ["Manual entry (M-STOCK)"] * len(d)
            })

        order_sheet(df_all, "KR").to_excel(writer, sheet_name="Order_KR", index=False)
        order_sheet(df_all, "US").to_excel(writer, sheet_name="Order_US", index=False)

        # ✅ 통화 서식 적용 (핵심)
        wb = writer.book
        for name in ["Signals_All", "Candidates", "Order_KR", "Order_US"]:
            if name in wb.sheetnames:
                apply_currency_formats_openpyxl(wb[name])

    return output.getvalue()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Swing Scanner", layout="wide")
st.title("웹 티커 입력 → 스윙 판단(O/X) + 주문표 엑셀")

with st.sidebar:
    st.header("스윙 전략 설정 (초보자 권장값)")

    params = {}

    st.markdown("### 추세 지표 (이동평균)")
    params["MA_FAST"] = st.number_input("MA_FAST (단기 이동평균)", 5, 200, DEFAULTS["MA_FAST"])
    st.write("단기 이동평균 기간(기본 20). 작을수록 신호 빠름/노이즈 증가")

    params["MA_SLOW"] = st.number_input("MA_SLOW (장기 이동평균)", 10, 300, DEFAULTS["MA_SLOW"])
    st.write("장기 이동평균 기간(기본 60). 단기선이 장기선 위면 상승 추세로 판단")

    st.markdown("---")
    st.markdown("### 거래량 / 변동성 조건")

    params["VOL_LOOKBACK"] = st.number_input("VOL_LOOKBACK (거래량 평균 기간)", 5, 200, DEFAULTS["VOL_LOOKBACK"])
    st.write("거래량 평균을 계산하는 기간(기본 20)")

    params["VOL_SPIKE"] = st.number_input("VOL_SPIKE (거래량 급증 기준)", 1.0, 5.0, float(DEFAULTS["VOL_SPIKE"]), step=0.05)
    st.write("현재 거래량/평균 거래량 비율 기준(예: 1.5 = 평균의 150%)")

    params["ATR_PERIOD"] = st.number_input("ATR_PERIOD (ATR 계산 기간)", 5, 100, DEFAULTS["ATR_PERIOD"])
    st.write("ATR(평균 변동폭) 계산 기간(기본 14)")

    params["ATR_PCT_MIN"] = st.number_input("ATR_PCT_MIN (최소 변동성)", 0.0, 0.2, float(DEFAULTS["ATR_PCT_MIN"]), step=0.001, format="%.3f")
    st.write("너무 안 움직이는 종목 제외")

    params["ATR_PCT_MAX"] = st.number_input("ATR_PCT_MAX (최대 변동성)", 0.0, 0.5, float(DEFAULTS["ATR_PCT_MAX"]), step=0.001, format="%.3f")
    st.write("변동성이 너무 큰 종목 제외")

    st.markdown("---")
    st.markdown("### 자금 관리")
    params["ACCOUNT_SIZE"] = st.number_input("ACCOUNT_SIZE (총 투자금)", 100_000, 1_000_000_000, DEFAULTS["ACCOUNT_SIZE"], step=100_000)
    st.write("포지션 크기 계산용 가상 계좌 금액")

    params["RISK_PER_TRADE"] = st.number_input("RISK_PER_TRADE (1회 최대 손실 비율)", 0.001, 0.05, float(DEFAULTS["RISK_PER_TRADE"]), step=0.001, format="%.3f")
    st.write("예: 0.01 = 1% 손실까지만 허용")

    params["STOP_ATR_MULT"] = st.number_input("STOP_ATR_MULT (손절 ATR 배수)", 0.5, 5.0, float(DEFAULTS["STOP_ATR_MULT"]), step=0.1)
    st.write("손절폭 = ATR × 배수 (기본 1.8)")

st.write("**입력 방법**: KR은 6자리 코드(예: `005930`), US는 티커(예: `SPY`). 콤마/줄바꿈/공백 모두 가능.")
raw = st.text_area("티커 입력", value="005930 000660\nSPY QQQ", height=120)

run = st.button("분석 실행")

def format_currency_for_display(row, col):
    v = row.get(col, None)
    mkt = row.get("market", "")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    try:
        if mkt == "KR":
            return f"₩{float(v):,.0f}"
        if mkt == "US":
            return f"${float(v):,.2f}"
        return str(v)
    except Exception:
        return str(v)

if run:
    tickers = normalize_tickers(raw)
    if not tickers:
        st.warning("티커를 1개 이상 입력하세요.")
        st.stop()

    rows = [analyze_one(t, params) for t in tickers]
    df = pd.DataFrame(rows)
    df = df.sort_values(["candidate", "score"], ascending=[False, False]).reset_index(drop=True)

    st.subheader("결과")

    # 화면 표시용(문자열 포맷) DataFrame
    df_view = df.copy()
    for c in ["close", "stop", "target(2R)"]:
        if c in df_view.columns:
            df_view[c] = df_view.apply(lambda r: format_currency_for_display(r, c), axis=1)

    st.dataframe(df_view, use_container_width=True)

    n_cand = int((df["candidate"] == 1).sum())
    if n_cand == 0:
        st.error("NOT TODAY: 조건을 만족하는 후보가 없습니다. (O=0)")
    else:
        st.success(f"후보(O) {n_cand}개")

    xlsx_bytes = build_excel(df)  # ✅ 숫자 원본 df로 엑셀 생성(서식은 openpyxl에서 처리)

    st.download_button(
        label="엑셀 1개로 다운로드 (All-in-One)",
        data=xlsx_bytes,
        file_name="Swing_Output_AllInOne.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
