import re
import io
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock as krx
import streamlit as st
from datetime import datetime, timedelta
from openpyxl.styles import numbers


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

# ---------------- Utils ----------------
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

# ✅ 근거 테이블(조건 통과 여부) - 전역 함수로 두어 IndentationError 방지
def build_reason_table(last, params):
    def safe(v):
        return None if pd.isna(v) else float(v)

    close = safe(last.get("Close", np.nan))
    ma_fast = safe(last.get("MA_FAST", np.nan))
    ma_slow = safe(last.get("MA_SLOW", np.nan))
    vol_ratio = safe(last.get("VOL_RATIO", np.nan))
    atr_pct = safe(last.get("ATR_PCT", np.nan))

    rows = []

    # 1) 추세
    c1 = (ma_fast is not None) and (ma_slow is not None) and (ma_fast > ma_slow)
    c2 = (close is not None) and (ma_fast is not None) and (close > ma_fast)
    rows.append({
        "조건": "추세: MA_FAST > MA_SLOW",
        "현재값": f"{ma_fast:,.2f} > {ma_slow:,.2f}" if (ma_fast is not None and ma_slow is not None) else "데이터 부족",
        "기준": "단기선이 장기선 위",
        "통과": bool(c1)
    })
    rows.append({
        "조건": "추세: Close > MA_FAST",
        "현재값": f"{close:,.2f} > {ma_fast:,.2f}" if (close is not None and ma_fast is not None) else "데이터 부족",
        "기준": "종가가 단기선 위",
        "통과": bool(c2)
    })

    # 2) 거래량
    c3 = (vol_ratio is not None) and (vol_ratio >= float(params["VOL_SPIKE"]))
    rows.append({
        "조건": "거래량: VOL_RATIO >= VOL_SPIKE",
        "현재값": f"{vol_ratio:,.2f}" if vol_ratio is not None else "데이터 부족",
        "기준": f">= {float(params['VOL_SPIKE']):,.2f}",
        "통과": bool(c3)
    })

    # 3) 변동성(ATR%)
    atr_min = float(params["ATR_PCT_MIN"])
    atr_max = float(params["ATR_PCT_MAX"])
    c4 = (atr_pct is not None) and (atr_min <= atr_pct <= atr_max)
    rows.append({
        "조건": "변동성: ATR_PCT_MIN <= ATR_PCT <= ATR_PCT_MAX",
        "현재값": f"{atr_pct*100:,.2f}%" if atr_pct is not None else "데이터 부족",
        "기준": f"{atr_min*100:,.2f}% ~ {atr_max*100:,.2f}%",
        "통과": bool(c4)
    })

    return pd.DataFrame(rows)

# ---------------- Data Load ----------------
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

    # ✅ MultiIndex(필드×티커) 또는 (티커×필드) 처리
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)

        # 케이스 A: (티커, 필드)
        if ticker in lv0:
            df = df[ticker]
        # 케이스 B: (필드, 티커)
        elif ticker in lv1:
            df = df.xs(ticker, axis=1, level=1)
        else:
            # 최후의 수단: 첫 ticker를 골라서 슬라이스
            uniq0 = list(pd.unique(lv0))
            uniq1 = list(pd.unique(lv1))
            if len(uniq1) >= 1 and ("Close" in uniq0 or "close" in [str(x).lower() for x in uniq0]):
                df = df.xs(uniq1[0], axis=1, level=1)
            elif len(uniq0) >= 1:
                df = df[uniq0[0]]
            else:
                raise ValueError(f"Unexpected MultiIndex columns: {df.columns}")

    # 컬럼명 통일
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

# ---------------- Analyzer ----------------
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

        reason_df = build_reason_table(last, params)

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

            "close": float(round(entry, 2)),
            "stop": (np.nan if np.isnan(stop) else float(round(stop, 2))),
            "target(2R)": (np.nan if np.isnan(target) else float(round(target, 2))),
            "qty": int(qty),

            "vol_ratio": (np.nan if pd.isna(last["VOL_RATIO"]) else float(round(float(last["VOL_RATIO"]), 2))),
            "atr_pct(%)": (np.nan if pd.isna(last["ATR_PCT"]) else float(round(float(last["ATR_PCT"]) * 100, 2))),
            "ret_20(%)": (np.nan if pd.isna(last["RET_20"]) else float(round(float(last["RET_20"]) * 100, 2))),
            "date": str(df.index[-1].date()),
            "reason": reason_df.to_dict(orient="records"),
            "error": ""
        }

    except Exception as e:
        return {
            "market": "?",
            "ticker": ticker,
            "O/X": "X",
            "candidate": 0,
            "score": 0,
            "close": np.nan,
            "stop": np.nan,
            "target(2R)": np.nan,
            "qty": 0,
            "vol_ratio": np.nan,
            "atr_pct(%)": np.nan,
            "ret_20(%)": np.nan,
            "date": "",
            "reason": [],
            "error": str(e)
        }

# ---------------- Excel ----------------
def build_excel(df_all: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # reason 컬럼은 엑셀에선 제외(리스트/오브젝트라 보기 불편)
        df_export = df_all.drop(columns=["reason"], errors="ignore")
        df_export.to_excel(writer, sheet_name="Signals_All", index=False)

        df_cand = df_all[df_all["candidate"] == 1].copy()
        if df_cand.empty:
            df_cand = pd.DataFrame([{"note": "nottoday"}])
        else:
            df_cand = df_cand.drop(columns=["reason"], errors="ignore")
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

    return output.getvalue()

# ---------------- UI ----------------
st.set_page_config(page_title="Swing Scanner", layout="wide")
st.title("웹 티커 입력 → 스윙 판단(O/X) + 근거표 + 주문표 엑셀")

with st.sidebar:
    st.header("스윙 전략 설정 (초보자 권장값)")

    params = {}

    st.markdown("### 추세 지표 (이동평균)")
    params["MA_FAST"] = st.number_input("MA_FAST (단기 이동평균)", 5, 200, DEFAULTS["MA_FAST"])
    st.write(
        "최근 주가의 단기 흐름을 보는 이동평균 기간입니다.\n"
        "- 값이 작을수록 신호는 빠르지만 잦은 실패 가능\n"
        "- 값이 클수록 신호는 느리지만 안정적\n"
        "일반적으로 10~30일을 사용하며 기본값 20은 무난합니다."
    )

    params["MA_SLOW"] = st.number_input("MA_SLOW (장기 이동평균)", 10, 300, DEFAULTS["MA_SLOW"])
    st.write(
        "중·장기 추세를 판단하는 기준 이동평균입니다.\n"
        "단기 이동평균(MA_FAST)이 이 값 위에 있으면 상승 추세로 판단합니다.\n"
        "보통 50~120일을 사용하며 기본값은 60입니다."
    )

    st.markdown("---")
    st.markdown("### 거래량 / 변동성 조건")

    params["VOL_LOOKBACK"] = st.number_input("VOL_LOOKBACK (거래량 평균 기간)", 5, 200, DEFAULTS["VOL_LOOKBACK"])
    st.write(
        "평균 거래량을 계산하는 기간입니다.\n"
        "현재 거래량이 이 평균 대비 얼마나 증가했는지(VOL_RATIO) 판단하는 데 사용됩니다."
    )

    params["VOL_SPIKE"] = st.number_input("VOL_SPIKE (거래량 급증 기준)", 1.0, 5.0, float(DEFAULTS["VOL_SPIKE"]), step=0.05)
    st.write(
        "현재 거래량이 평균 대비 몇 배 이상일 때 진입 후보로 볼지 정합니다.\n"
        "예: 1.5는 평균 거래량 대비 150% 이상을 의미합니다."
    )

    params["ATR_PERIOD"] = st.number_input("ATR_PERIOD (ATR 계산 기간)", 5, 100, DEFAULTS["ATR_PERIOD"])
    st.write(
        "ATR은 주가의 평균 변동폭을 나타내는 지표입니다.\n"
        "값이 클수록 주가 변동이 큰 종목입니다."
    )

    params["ATR_PCT_MIN"] = st.number_input("ATR_PCT_MIN (최소 변동성)", 0.0, 0.2, float(DEFAULTS["ATR_PCT_MIN"]), step=0.001, format="%.3f")
    st.write("너무 움직임이 없는(변동성 낮은) 종목을 제외하기 위한 최소 변동성 기준입니다.")

    params["ATR_PCT_MAX"] = st.number_input("ATR_PCT_MAX (최대 변동성)", 0.0, 0.5, float(DEFAULTS["ATR_PCT_MAX"]), step=0.001, format="%.3f")
    st.write("변동성이 지나치게 큰(고위험) 종목을 제외하기 위한 상한선입니다.")

    st.markdown("---")
    st.markdown("### 자금 관리")

    params["ACCOUNT_SIZE"] = st.number_input("ACCOUNT_SIZE (총 투자금)", 100_000, 1_000_000_000, DEFAULTS["ACCOUNT_SIZE"], step=100_000)
    st.write("가상의 전체 계좌 금액입니다. 실제 주문이 아니라 포지션 크기 계산에만 사용됩니다.")

    params["RISK_PER_TRADE"] = st.number_input("RISK_PER_TRADE (1회 최대 손실 비율)", 0.001, 0.05, float(DEFAULTS["RISK_PER_TRADE"]), step=0.001, format="%.3f")
    st.write("한 종목에서 감수할 최대 손실 비율입니다. 예: 0.01은 1% 손실을 의미합니다.")

    params["STOP_ATR_MULT"] = st.number_input("STOP_ATR_MULT (손절 ATR 배수)", 0.5, 5.0, float(DEFAULTS["STOP_ATR_MULT"]), step=0.1)
    st.write("손절 가격을 ATR 기준으로 얼마나 여유 있게 둘지 정합니다. 보통 1.5~2.0 범위를 많이 사용합니다.")

st.write("**입력 방법**: KR은 6자리 코드(예: `005930`), US는 티커(예: `SPY`). 콤마/줄바꿈/공백 모두 가능.")
raw = st.text_area("티커 입력", value="005930 000660\nSPY QQQ", height=120)

run = st.button("분석 실행")

if run:
    tickers = normalize_tickers(raw)
    if not tickers:
        st.warning("티커를 1개 이상 입력하세요.")
        st.stop()

    rows = [analyze_one(t, params) for t in tickers]
    df = pd.DataFrame(rows)

    df = df.sort_values(["candidate", "score"], ascending=[False, False]).reset_index(drop=True)

    st.subheader("결과")

    # ✅ 메인 테이블에는 reason(리스트) 숨기고, 숫자는 천단위 콤마로 표시
    df_main = df.drop(columns=["reason"], errors="ignore")

    st.dataframe(
        df_main,
        use_container_width=True,
        column_config={
            "close": st.column_config.NumberColumn("close", format="%,.2f"),
            "stop": st.column_config.NumberColumn("stop", format="%,.2f"),
            "target(2R)": st.column_config.NumberColumn("target(2R)", format="%,.2f"),
            "qty": st.column_config.NumberColumn("qty", format="%,d"),
            "vol_ratio": st.column_config.NumberColumn("vol_ratio", format="%.2f"),
            "atr_pct(%)": st.column_config.NumberColumn("atr_pct(%)", format="%.2f"),
            "ret_20(%)": st.column_config.NumberColumn("ret_20(%)", format="%.2f"),
            "score": st.column_config.NumberColumn("score", format="%,d"),
        }
    )

    n_cand = int((df["candidate"] == 1).sum())
    if n_cand == 0:
        st.error("NOT TODAY: 조건을 만족하는 후보가 없습니다. (O=0)")
    else:
        st.success(f"후보(O) {n_cand}개")

    st.markdown("---")
    st.subheader("근거(조건 통과 여부) — 종목별로 확인")

    for i, row in df.iterrows():
        title = f"{row['market']} {row['ticker']} | O/X: {row['O/X']} | score: {row['score']}"
        with st.expander(title, expanded=(i == 0)):
            if row.get("error"):
                st.error(row["error"])
            reason_list = row.get("reason", [])
            if reason_list:
                reason_table = pd.DataFrame(reason_list)
                st.dataframe(reason_table, use_container_width=True)
            else:
                st.write("근거 데이터 없음")

   xlsx_bytes = build_excel(df)
def build_excel(df_all: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:

        df_export = df_all.drop(columns=["reason"], errors="ignore")
        df_export.to_excel(writer, sheet_name="Signals_All", index=False)

        ws = writer.book["Signals_All"]

        # 컬럼 위치 찾기 (헤더 기준)
        headers = {cell.value: idx+1 for idx, cell in enumerate(ws[1])}

        col_close  = headers.get("close")
        col_stop   = headers.get("stop")
        col_target = headers.get("target(2R)")

        # 2행부터 데이터 시작
        for r in range(2, ws.max_row + 1):
            market = ws.cell(row=r, column=headers["market"]).value

            for c in [col_close, col_stop, col_target]:
                cell = ws.cell(row=r, column=c)

                if market == "KR":
                    # 한국 주식: 천단위 정수
                    cell.number_format = '#,##0'
                elif market == "US":
                    # 미국 주식: 달러 + 소수점 2자리
                    cell.number_format = '$#,##0.00'

        # ---- 나머지 시트들 ----
        df_cand = df_export[df_export["candidate"] == 1].copy()
        if df_cand.empty:
            df_cand = pd.DataFrame([{"note": "nottoday"}])
        df_cand.to_excel(writer, sheet_name="Candidates", index=False)

        # 주문 시트는 기존 로직 그대로
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

    return output.getvalue()

    st.download_button(
        label="엑셀 1개로 다운로드 (All-in-One)",
        data=xlsx_bytes,
        file_name="Swing_Output_AllInOne.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
