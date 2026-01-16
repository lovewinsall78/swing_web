# ============================================
# Swing Scanner (KR/US) - FINAL (No-Reset Editor + Wide Tables + Entry Calc Fix)
# - TOP10 자동추천(후보풀 스윙전략 스캔) -> 입력칸 자동삽입
# - 분석(O/X) + 점수 + 근거표 + 차트 + 매도추천
# - 보유(평단/진입일) 입력 리셋/롤백 방지(Widget-state -> Store sync)
# - ✅ 평단 입력 후 계산 안됨 문제 수정(시장값 정규화 + KR/US 자동판별)
# - KR/US 통화 입력 분리(₩ / $), ACCOUNT_SIZE 콤마 입력
# - 엑셀 다운로드(KR=₩#,##0 / US=$#,##0.00)
# ============================================

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
# Defaults
# -----------------------------
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
    ACCOUNT_SIZE=10_000_000,
    RISK_PER_TRADE=0.01,
    TOP_N=10,
)

# -----------------------------
# Candidate Universe
# -----------------------------
KR_UNIVERSE = [
    "005930","000660","035420","035720","051910","068270","207940","005380","000270","012330",
    "028260","066570","096770","003550","034020","015760","017670","018260","055550","033780",
    "010130","086790","009150","010950","034730","036570","011170","090430","030200","032830",
]
US_UNIVERSE = [
    "SPY","QQQ","IWM","DIA","XLK","XLF","SMH","SOXX",
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","CRM","ADBE","ORCL","NFLX",
    "V","MA","JPM","BAC","GS","C"
]

# -----------------------------
# Utils
# -----------------------------
def is_kr_code(x: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", x.strip()))

def normalize_tickers(raw: str):
    items = re.split(r"[,\n\s]+", (raw or "").strip())
    return [x.strip().upper() for x in items if x.strip()]

def parse_entry_text(market: str, s: str):
    """
    ✅ 강화판:
    - market 값 정규화(KR/US)
    - market이 비어/이상하면 입력 문자열의 ₩/$로 추정
    - 그래도 모르겠으면 '.' 있으면 US(소수), 없으면 KR(정수)로 추정
    """
    if s is None:
        return np.nan

    t = str(s).strip()
    if not t:
        return np.nan

    # 시장값 정규화
    m = str(market).strip().upper() if market is not None else ""

    # 통화기호/공백/콤마 제거
    raw = t.replace("₩", "").replace("$", "").replace(" ", "").replace(",", "")

    try:
        val = float(raw)

        # 시장값이 비정상일 때: 통화기호로 추정
        if m not in ("KR", "US"):
            if "₩" in t:
                m = "KR"
            elif "$" in t:
                m = "US"

        if m == "KR":
            return int(val)
        if m == "US":
            return round(val, 2)

        # 최후의 추정
        return round(val, 2) if "." in raw else int(val)

    except Exception:
        return np.nan

def format_currency(mkt: str, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    try:
        m = str(mkt).strip().upper()
        if m == "KR":
            return f"₩{int(round(float(v))):,}"
        if m == "US":
            return f"${float(v):,.2f}"
        return str(v)
    except Exception:
        return str(v)

# -----------------------------
# Indicators
# -----------------------------
def compute_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df, p):
    df = df.copy()
    df["MA_FAST"] = df["Close"].rolling(int(p["MA_FAST"])).mean()
    df["MA_SLOW"] = df["Close"].rolling(int(p["MA_SLOW"])).mean()
    df["VOL_AVG"] = df["Volume"].rolling(int(p["VOL_LOOKBACK"])).mean()
    df["VOL_RATIO"] = np.where(df["VOL_AVG"] > 0, df["Volume"] / df["VOL_AVG"], np.nan)
    df["ATR"] = compute_atr(df, int(p["ATR_PERIOD"]))
    df["ATR_PCT"] = df["ATR"] / df["Close"]
    df["RET_20"] = df["Close"].pct_change(20)
    return df

def rule_signal(last, p) -> int:
    needed = ["MA_FAST", "MA_SLOW", "VOL_RATIO", "ATR_PCT", "Close"]
    if any(pd.isna(last.get(k, np.nan)) for k in needed):
        return 0
    trend_ok = (last["MA_FAST"] > last["MA_SLOW"]) and (last["Close"] > last["MA_FAST"])
    vol_ok   = (last["VOL_RATIO"] >= float(p["VOL_SPIKE"]))
    atr_ok   = (float(p["ATR_PCT_MIN"]) <= last["ATR_PCT"] <= float(p["ATR_PCT_MAX"]))
    return int(trend_ok and vol_ok and atr_ok)

def score_row(last):
    score = 0
    if last.get("MA_FAST", np.nan) > last.get("MA_SLOW", np.nan):
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

def position_size(entry, atr, account_size, risk_per_trade, stop_atr_mult):
    if pd.isna(atr) or atr <= 0:
        return 0, np.nan
    stop_dist = stop_atr_mult * atr
    risk_amount = account_size * risk_per_trade
    qty = int(risk_amount / stop_dist) if stop_dist > 0 else 0
    stop_price = entry - stop_dist
    return max(0, qty), float(stop_price)

def build_reason_table(last, p) -> pd.DataFrame:
    def safe(v):
        return None if pd.isna(v) else float(v)

    close = safe(last.get("Close", np.nan))
    ma_fast = safe(last.get("MA_FAST", np.nan))
    ma_slow = safe(last.get("MA_SLOW", np.nan))
    vol_ratio = safe(last.get("VOL_RATIO", np.nan))
    atr_pct = safe(last.get("ATR_PCT", np.nan))

    rows = []
    c1 = (ma_fast is not None) and (ma_slow is not None) and (ma_fast > ma_slow)
    c2 = (close is not None) and (ma_fast is not None) and (close > ma_fast)
    rows.append({"조건": "추세: MA_FAST > MA_SLOW",
                 "현재값": "데이터 부족" if (ma_fast is None or ma_slow is None) else f"{ma_fast:.2f} > {ma_slow:.2f}",
                 "기준": "단기선이 장기선 위", "통과": bool(c1)})
    rows.append({"조건": "추세: Close > MA_FAST",
                 "현재값": "데이터 부족" if (close is None or ma_fast is None) else f"{close:.2f} > {ma_fast:.2f}",
                 "기준": "종가가 단기선 위", "통과": bool(c2)})

    spike = float(p["VOL_SPIKE"])
    c3 = (vol_ratio is not None) and (vol_ratio >= spike)
    rows.append({"조건": "거래량: VOL_RATIO >= VOL_SPIKE",
                 "현재값": "데이터 부족" if vol_ratio is None else f"{vol_ratio:.2f}",
                 "기준": f">= {spike:.2f}", "통과": bool(c3)})

    atr_min = float(p["ATR_PCT_MIN"])
    atr_max = float(p["ATR_PCT_MAX"])
    c4 = (atr_pct is not None) and (atr_min <= atr_pct <= atr_max)
    rows.append({"조건": "변동성: ATR_PCT_MIN <= ATR_PCT <= ATR_PCT_MAX",
                 "현재값": "데이터 부족" if atr_pct is None else f"{atr_pct*100:.2f}%",
                 "기준": f"{atr_min*100:.2f}% ~ {atr_max*100:.2f}%", "통과": bool(c4)})

    return pd.DataFrame(rows)

# -----------------------------
# Data Load (cache)
# -----------------------------
@st.cache_data(ttl=60*20, show_spinner=False)
def _load_us_cached(ticker: str, years: int) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=f"{years}y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)
        if ticker in lv0:
            df = df[ticker]
        elif ticker in lv1:
            df = df.xs(ticker, axis=1, level=1)
        else:
            uniq1 = list(pd.unique(lv1))
            if uniq1:
                df = df.xs(uniq1[0], axis=1, level=1)
            else:
                raise ValueError(f"Unexpected MultiIndex columns: {df.columns}")

    df = df.rename(columns=lambda c: str(c).title())
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    keep = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"US data missing columns: {missing} / columns={list(df.columns)}")

    return df[keep].dropna()

@st.cache_data(ttl=60*20, show_spinner=False)
def _load_kr_cached(code: str, years: int) -> pd.DataFrame:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365*years)).strftime("%Y-%m-%d")
    df = krx.get_market_ohlcv_by_date(start, end, code)
    df = df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
    return df[["Open","High","Low","Close","Volume"]].dropna()

def load_data(ticker: str, lookback_years: int) -> pd.DataFrame:
    if is_kr_code(ticker):
        return _load_kr_cached(ticker, lookback_years)
    return _load_us_cached(ticker, lookback_years)

# -----------------------------
# Sell Recommendation
# -----------------------------
def sell_recommendation(last: pd.Series, p: dict, entry_price: float, entry_date: str):
    if entry_price is None or (isinstance(entry_price, float) and np.isnan(entry_price)) or entry_price <= 0:
        return "N/A", "평단(진입가) 입력 필요", None, None, None

    holding_days = None
    if isinstance(entry_date, str) and entry_date.strip():
        try:
            d0 = datetime.strptime(entry_date.strip(), "%Y-%m-%d").date()
            holding_days = (datetime.now().date() - d0).days
        except Exception:
            holding_days = None

    close = float(last.get("Close", np.nan))
    atr = float(last.get("ATR", np.nan))
    if pd.isna(close):
        return "N/A", "종가 데이터 부족", None, None, holding_days

    stop_price = None
    target_price = None
    if (not pd.isna(atr)) and atr > 0:
        stop_price = float(entry_price - float(p["STOP_ATR_MULT"]) * atr)
        target_price = float(entry_price + 2 * (entry_price - stop_price))  # 2R

    if stop_price is not None and close < stop_price:
        return "SELL", "손절가 이탈(ATR 기준)", stop_price, target_price, holding_days
    if target_price is not None and close >= target_price:
        return "PARTIAL SELL", "목표가(2R) 도달", stop_price, target_price, holding_days

    ma_fast = last.get("MA_FAST", np.nan)
    ma_slow = last.get("MA_SLOW", np.nan)
    if (pd.notna(ma_fast) and close < ma_fast) or (pd.notna(ma_fast) and pd.notna(ma_slow) and ma_fast < ma_slow):
        return "PARTIAL SELL", "추세 이탈(이평선 기준)", stop_price, target_price, holding_days

    if holding_days is not None and holding_days >= int(p["HOLD_DAYS"]):
        return "SELL", "보유 기간 초과(HOLD_DAYS)", stop_price, target_price, holding_days

    return "HOLD", "추세 유지", stop_price, target_price, holding_days

# -----------------------------
# Analyze One
# -----------------------------
def analyze_one_with_detail(ticker: str, p: dict):
    try:
        market = "KR" if is_kr_code(ticker) else "US"
        df = load_data(ticker, int(p["LOOKBACK_YEARS"]))
        df_ind = add_indicators(df, p)
        last = df_ind.iloc[-1]

        cand = rule_signal(last, p)
        sc = score_row(last) if cand == 1 else 0

        entry = float(last["Close"])
        qty, stop_buy = position_size(
            entry=entry,
            atr=float(last.get("ATR", np.nan)),
            account_size=float(p["ACCOUNT_SIZE"]),
            risk_per_trade=float(p["RISK_PER_TRADE"]),
            stop_atr_mult=float(p["STOP_ATR_MULT"]),
        )
        target_buy = entry + 2 * (entry - stop_buy) if (not np.isnan(stop_buy)) else np.nan
        reason_df = build_reason_table(last, p)

        res = dict(
            market=market,
            ticker=ticker,
            OX=("O" if cand == 1 else "X"),
            candidate=int(cand),
            score=int(sc),
            close=float(entry),
            stop=(np.nan if np.isnan(stop_buy) else float(stop_buy)),
            target_2R=(np.nan if np.isnan(target_buy) else float(target_buy)),
            qty=int(qty),
            vol_ratio=(np.nan if pd.isna(last.get("VOL_RATIO", np.nan)) else float(last["VOL_RATIO"])),
            atr_pct=(np.nan if pd.isna(last.get("ATR_PCT", np.nan)) else float(last["ATR_PCT"]) * 100),
            ret_20=(np.nan if pd.isna(last.get("RET_20", np.nan)) else float(last["RET_20"]) * 100),
            date=str(df_ind.index[-1].date()),
            error="",
        )
        return res, df_ind, reason_df

    except Exception as e:
        res = dict(
            market="?",
            ticker=ticker,
            OX="X",
            candidate=0,
            score=0,
            close=np.nan,
            stop=np.nan,
            target_2R=np.nan,
            qty=0,
            vol_ratio=np.nan,
            atr_pct=np.nan,
            ret_20=np.nan,
            date="",
            error=str(e),
        )
        return res, None, None

# -----------------------------
# Recommendation (Universe Scan -> Top10)
# -----------------------------
def recommend_top10(p: dict, top_n: int = 10):
    universe = [(t, "KR") for t in KR_UNIVERSE] + [(t, "US") for t in US_UNIVERSE]
    rows = []
    prog = st.progress(0)
    total = len(universe)

    for i, (t, _m) in enumerate(universe, start=1):
        res, _df, _reason = analyze_one_with_detail(t, p)
        if not res.get("error"):
            rows.append(res)
        prog.progress(int(i * 100 / total))

    prog.empty()

    if not rows:
        return [], pd.DataFrame()

    df = pd.DataFrame(rows)

    cand = df[df["candidate"] == 1].sort_values("score", ascending=False)
    picks = []
    for t in cand["ticker"].astype(str).tolist():
        if t not in picks:
            picks.append(t)
        if len(picks) >= top_n:
            break

    if len(picks) < top_n:
        all_sorted = df.sort_values(["candidate", "score"], ascending=[False, False])
        for t in all_sorted["ticker"].astype(str).tolist():
            if t not in picks:
                picks.append(t)
            if len(picks) >= top_n:
                break

    return picks[:top_n], df

# -----------------------------
# Charts (Altair)
# -----------------------------
def _reset_index_as_date(df_ind: pd.DataFrame) -> pd.DataFrame:
    d = df_ind.reset_index()
    if "Date" not in d.columns:
        d = d.rename(columns={d.columns[0]: "Date"})
    d["Date"] = pd.to_datetime(d["Date"])
    return d

def price_chart(df_ind: pd.DataFrame, entry=None, stop=None, target=None):
    d = _reset_index_as_date(df_ind)
    base = alt.Chart(d).encode(x="Date:T")
    close_line = base.mark_line().encode(y=alt.Y("Close:Q", title="Price"))
    ma_fast = base.mark_line(opacity=0.7).encode(y="MA_FAST:Q")
    ma_slow = base.mark_line(opacity=0.7).encode(y="MA_SLOW:Q")

    layers = [close_line, ma_fast, ma_slow]

    def rule(y):
        return alt.Chart(pd.DataFrame({"y": [float(y)]})).mark_rule().encode(y="y:Q")

    if entry is not None and not (isinstance(entry, float) and np.isnan(entry)):
        layers.append(rule(entry))
    if stop is not None and not (isinstance(stop, float) and np.isnan(stop)):
        layers.append(rule(stop))
    if target is not None and not (isinstance(target, float) and np.isnan(target)):
        layers.append(rule(target))

    return alt.layer(*layers).properties(height=280)

def volume_chart(df_ind: pd.DataFrame):
    d = _reset_index_as_date(df_ind)
    base = alt.Chart(d).encode(x="Date:T")
    vol = base.mark_line().encode(y=alt.Y("Volume:Q", title="Volume"))
    avg = base.mark_line(opacity=0.7).encode(y="VOL_AVG:Q")
    return (vol + avg).properties(height=180)

# -----------------------------
# Excel Download (KR/US format by row)
# -----------------------------
def build_excel_bytes_with_formats(df_all: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="Signals_All", index=False)

        df_cand = df_all[df_all.get("candidate", 0) == 1].copy()
        if df_cand.empty:
            df_cand = pd.DataFrame([{"note": "nottoday"}])
        df_cand.to_excel(writer, sheet_name="Candidates", index=False)

        wb = writer.book

        def apply_formats(ws):
            fmt_krw = u"₩#,##0"
            fmt_usd = u"$#,##0.00"

            header = {}
            for col in range(1, ws.max_column + 1):
                v = ws.cell(row=1, column=col).value
                if isinstance(v, str):
                    header[v.strip()] = col

            if "market" not in header:
                return
            market_col = header["market"]

            price_cols = {
                "close", "stop", "target(2R)", "target_2R",
                "entry_price", "stop_by_entry", "target_by_entry(2R)"
            }
            price_idxs = [header[c] for c in price_cols if c in header]
            if not price_idxs:
                return

            for r in range(2, ws.max_row + 1):
                mkt = ws.cell(row=r, column=market_col).value
                numfmt = fmt_krw if mkt == "KR" else (fmt_usd if mkt == "US" else None)
                if not numfmt:
                    continue
                for c in price_idxs:
                    cell = ws.cell(row=r, column=c)
                    if isinstance(cell.value, (int, float)) and cell.value is not None:
                        cell.number_format = numfmt

        for name in ["Signals_All", "Candidates"]:
            if name in wb.sheetnames:
                apply_formats(wb[name])

    return out.getvalue()

# -----------------------------
# Positions sync (run 버튼에서만!)
# -----------------------------
def sync_positions_on_run(df_analysis: pd.DataFrame):
    base = df_analysis[["ticker", "market"]].copy()
    base["ticker"] = base["ticker"].astype(str).str.upper()
    base["market"] = base["market"].astype(str).str.strip().str.upper()

    ed = st.session_state["positions_editor_df"].copy()
    ed["ticker"] = ed["ticker"].astype(str).str.upper()
    ed["market"] = ed["market"].astype(str).str.strip().str.upper()

    ed = base.merge(ed, on=["ticker", "market"], how="left")
    ed["entry_text"] = ed["entry_text"].fillna("")
    ed["entry_date"] = ed["entry_date"].fillna("")
    st.session_state["positions_editor_df"] = ed

# ============================================
# Streamlit UI
# ============================================
st.set_page_config(page_title="Swing Scanner", layout="wide")

st.markdown(
    "<h1 style='font-size:34px;font-weight:700;margin:0'>웹 티커 입력 → 스윙 판단(O/X) + 근거(표+차트) + 매도 추천 + 엑셀</h1>",
    unsafe_allow_html=True
)

# -----------------------------
# Session init
# -----------------------------
if "analysis_df" not in st.session_state:
    st.session_state["analysis_df"] = None
if "analysis_detail" not in st.session_state:
    st.session_state["analysis_detail"] = {}  # {TICKER: (df_ind, reason_df)}
if "ticker_input_text" not in st.session_state:
    st.session_state["ticker_input_text"] = "005930 000660\nSPY QQQ"

if "positions_editor_df" not in st.session_state:
    st.session_state["positions_editor_df"] = pd.DataFrame(
        columns=["ticker", "market", "entry_text", "entry_date"]
    )

if "ACCOUNT_SIZE" not in st.session_state:
    st.session_state["ACCOUNT_SIZE"] = DEFAULTS["ACCOUNT_SIZE"]

# ⭐ 핵심: data_editor 위젯 상태 -> 저장소 반영(리셋 방지)
if "positions_editor_widget" in st.session_state:
    w = st.session_state["positions_editor_widget"]
    if isinstance(w, pd.DataFrame):
        st.session_state["positions_editor_df"] = w.copy()

# -----------------------------
# Sidebar (종류별 구분 + 설명문구)
# -----------------------------
with st.sidebar:
    st.header("스윙 전략 설정")
    params = {}

    with st.expander("① 추세 판단 (이동평균)", expanded=True):
        params["MA_FAST"] = st.number_input("MA_FAST (단기 이동평균)", 5, 200, DEFAULTS["MA_FAST"], key="MA_FAST")
        st.write(
            "최근 주가의 단기 흐름을 보는 이동평균 기간입니다.\n"
            "- 값이 작을수록 신호는 빠르지만 잦은 실패 가능\n"
            "- 값이 클수록 신호는 느리지만 안정적\n"
            "일반적으로 10~30일, 기본값 20"
        )

        params["MA_SLOW"] = st.number_input("MA_SLOW (장기 이동평균)", 10, 300, DEFAULTS["MA_SLOW"], key="MA_SLOW")
        st.write(
            "중·장기 추세 판단 기준입니다.\n"
            "단기 이동평균(MA_FAST)이 장기 이동평균(MA_SLOW) 위면 상승 추세로 판단.\n"
            "보통 50~120일, 기본값 60"
        )

    with st.expander("② 거래량 · 변동성 조건", expanded=False):
        params["VOL_LOOKBACK"] = st.number_input("VOL_LOOKBACK (거래량 평균 기간)", 5, 200, DEFAULTS["VOL_LOOKBACK"], key="VOL_LOOKBACK")
        st.write("평균 거래량을 계산하는 기간입니다.")

        params["VOL_SPIKE"] = st.number_input("VOL_SPIKE (거래량 급증 기준)", 1.0, 5.0, float(DEFAULTS["VOL_SPIKE"]), step=0.05, key="VOL_SPIKE")
        st.write("예: 1.5는 평균 거래량 대비 150% 이상이면 통과")

        params["ATR_PERIOD"] = st.number_input("ATR_PERIOD (ATR 계산 기간)", 5, 100, DEFAULTS["ATR_PERIOD"], key="ATR_PERIOD")
        st.write("ATR은 평균 변동폭(손절/변동성 필터)에 사용됩니다.")

        params["ATR_PCT_MIN"] = st.number_input("ATR_PCT_MIN (최소 변동성)", 0.0, 0.2, float(DEFAULTS["ATR_PCT_MIN"]), step=0.001, format="%.3f", key="ATR_PCT_MIN")
        st.write("너무 움직임이 없는 종목 제외")

        params["ATR_PCT_MAX"] = st.number_input("ATR_PCT_MAX (최대 변동성)", 0.0, 0.5, float(DEFAULTS["ATR_PCT_MAX"]), step=0.001, format="%.3f", key="ATR_PCT_MAX")
        st.write("변동성이 과도한 종목 제외")

    with st.expander("③ 리스크 · 손절 · 보유 관리", expanded=False):
        params["STOP_ATR_MULT"] = st.number_input("STOP_ATR_MULT (손절 ATR 배수)", 0.5, 5.0, float(DEFAULTS["STOP_ATR_MULT"]), step=0.1, key="STOP_ATR_MULT")
        st.write("손절 가격을 ATR 기준으로 얼마나 여유 있게 둘지. 보통 1.5~2.0")

        params["HOLD_DAYS"] = st.number_input("HOLD_DAYS (최대 보유일)", 1, 200, DEFAULTS["HOLD_DAYS"], key="HOLD_DAYS")
        st.write("보유기간 초과 시 매도 고려")

    with st.expander("④ 계좌 · 포지션 사이징", expanded=False):
        acc_str = st.text_input(
            "ACCOUNT_SIZE (총 투자금, 천단위 콤마 가능)",
            value=f"{int(st.session_state['ACCOUNT_SIZE']):,}",
            key="ACCOUNT_SIZE_TEXT"
        )
        try:
            params["ACCOUNT_SIZE"] = int(str(acc_str).replace(",", "").strip())
            st.session_state["ACCOUNT_SIZE"] = params["ACCOUNT_SIZE"]
        except Exception:
            params["ACCOUNT_SIZE"] = st.session_state["ACCOUNT_SIZE"]

        params["RISK_PER_TRADE"] = st.number_input(
            "RISK_PER_TRADE (1회 최대 손실 비율)",
            0.001, 0.05, float(DEFAULTS["RISK_PER_TRADE"]),
            step=0.001, format="%.3f", key="RISK_PER_TRADE"
        )
        st.write("예: 0.01은 계좌의 1% 손실을 한 종목에서 감수")

    params["LOOKBACK_YEARS"] = DEFAULTS["LOOKBACK_YEARS"]
    params["TOP_N"] = DEFAULTS["TOP_N"]

    st.markdown("---")
    st.session_state["show_debug_entry"] = st.checkbox("디버그(평단 계산표) 표시", value=False)

# -----------------------------
# Input + 추천 버튼
# -----------------------------
st.write("입력: KR은 6자리(예: 005930), US는 티커(예: SPY). 콤마/줄바꿈/공백 가능.")

c1, c2, c3 = st.columns([2, 1, 1])

with c1:
    if st.button("스윙전략 기준 TOP10 추천 → 입력칸 채우기", key="btn_reco10"):
        picks, df_scan = recommend_top10(params, top_n=int(params["TOP_N"]))
        if not picks:
            st.warning("추천 실패: 데이터/네트워크/후보풀 점검이 필요합니다.")
        else:
            kr = [t for t in picks if is_kr_code(t)]
            us = [t for t in picks if not is_kr_code(t)]
            text = ""
            if kr:
                text += " ".join(kr)
            if us:
                text += ("\n" if text else "") + " ".join(us)
            st.session_state["ticker_input_text"] = text
            st.session_state["last_reco_scan_df"] = df_scan

with c2:
    if st.button("입력칸 비우기", key="btn_clear_input"):
        st.session_state["ticker_input_text"] = ""

with c3:
    if st.button("추천 스캔 Top30 보기", key="btn_show_scan"):
        df_scan = st.session_state.get("last_reco_scan_df", None)
        if isinstance(df_scan, pd.DataFrame) and not df_scan.empty:
            wide_cfg = {
                "ticker": st.column_config.TextColumn("ticker", width="small"),
                "market": st.column_config.TextColumn("market", width="small"),
                "OX": st.column_config.TextColumn("O/X", width="small"),
                "candidate": st.column_config.NumberColumn("cand", width="small"),
                "score": st.column_config.NumberColumn("score", width="small"),
                "date": st.column_config.TextColumn("date", width="small"),
                "error": st.column_config.TextColumn("error", width="large"),
            }
            st.dataframe(
                df_scan.sort_values(["candidate","score"], ascending=[False, False]).head(30),
                use_container_width=True,
                hide_index=True,
                column_config=wide_cfg
            )
        else:
            st.info("아직 추천 실행 전입니다.")

raw = st.text_area(
    "티커 입력",
    value=st.session_state["ticker_input_text"],
    height=120,
    key="ticker_input_area"
)
st.session_state["ticker_input_text"] = raw

run = st.button("분석 실행", key="run_button")

# -----------------------------
# Run analysis (버튼 눌렀을 때만)
# -----------------------------
if run:
    tickers = normalize_tickers(raw)
    if not tickers:
        st.warning("티커를 1개 이상 입력하세요.")
    else:
        results = []
        detail_map = {}
        prog = st.progress(0)
        for i, t in enumerate(tickers, start=1):
            res, df_ind, reason_df = analyze_one_with_detail(t, params)
            results.append(res)
            if df_ind is not None and reason_df is not None and not res.get("error"):
                detail_map[t.upper()] = (df_ind, reason_df)
            prog.progress(int(i * 100 / len(tickers)))
        prog.empty()

        df = pd.DataFrame(results)
        df = df.sort_values(["candidate", "score"], ascending=[False, False]).reset_index(drop=True)

        st.session_state["analysis_df"] = df
        st.session_state["analysis_detail"] = detail_map

        # 보유 입력 동기화(run 버튼에서만!)
        sync_positions_on_run(df)

# -----------------------------
# Render results
# -----------------------------
df_saved = st.session_state.get("analysis_df", None)
detail_saved = st.session_state.get("analysis_detail", {})

if df_saved is None or df_saved.empty:
    st.info("분석 실행을 눌러 결과를 생성하세요.")
    st.stop()

# -----------------------------
# Positions (No-reset data_editor)
# -----------------------------
st.markdown("---")
st.subheader("보유 입력 (평단/진입일)")

st.write(
    "- KR 평단 예시: `₩10,000,000` / `10,000,000` / `10000000`\n"
    "- US 평단 예시: `$123.45` / `123.45` / `1,234.56`\n"
    "평단은 `entry_text`에 입력하고, 계산용 평단은 아래에서 자동 계산됩니다."
)

st.data_editor(
    st.session_state["positions_editor_df"],
    key="positions_editor_widget",
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    column_config={
        "ticker": st.column_config.TextColumn("ticker", disabled=True, width="small"),
        "market": st.column_config.TextColumn("market", disabled=True, width="small"),
        "entry_text": st.column_config.TextColumn("평단 입력 (KR: ₩... / US: $...)", width="large"),
        "entry_date": st.column_config.TextColumn("진입일(YYYY-MM-DD)", width="medium"),
    },
)

# ✅ 계산 단계 (시장 정규화 필수!)
edited_store = st.session_state["positions_editor_df"].copy()
edited_store["ticker"] = edited_store["ticker"].astype(str).str.upper()
edited_store["market"] = edited_store["market"].astype(str).str.strip().str.upper()

pos_calc = edited_store.copy()
pos_calc["entry_price"] = [
    parse_entry_text(m, t) for m, t in zip(pos_calc["market"], pos_calc["entry_text"])
]
pos_calc["entry_display"] = [
    format_currency(m, v) for m, v in zip(pos_calc["market"], pos_calc["entry_price"])
]

st.caption("계산된 평단(읽기 전용)")
st.dataframe(
    pos_calc[["ticker","market","entry_display","entry_date"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "ticker": st.column_config.TextColumn("ticker", width="small"),
        "market": st.column_config.TextColumn("market", width="small"),
        "entry_display": st.column_config.TextColumn("평단(표시)", width="large"),
        "entry_date": st.column_config.TextColumn("진입일", width="medium"),
    }
)

# (선택) 디버그: 입력→계산 원본 확인
if st.session_state.get("show_debug_entry", False):
    st.caption("디버그: entry_text → entry_price 원본 확인")
    st.dataframe(
        pos_calc[["ticker","market","entry_text","entry_price"]],
        use_container_width=True,
        hide_index=True
    )

# pos_map
pos_map = {}
for _, r in pos_calc.iterrows():
    pos_map[(str(r["ticker"]).upper(), str(r["market"]).strip().upper())] = dict(
        entry_text=r.get("entry_text",""),
        entry_price=r.get("entry_price", np.nan),
        entry_date=r.get("entry_date",""),
    )

# -----------------------------
# Add sell recommendation columns
# -----------------------------
df_out = df_saved.copy()
df_out["ticker"] = df_out["ticker"].astype(str).str.upper()
df_out["market"] = df_out["market"].astype(str).str.strip().str.upper()

sell_sig, sell_reason = [], []
hold_days_list = []
stop_by_entry_list, target_by_entry_list = [], []
entry_text_list, entry_price_list = [], []

for _, row in df_out.iterrows():
    tkr = str(row["ticker"]).upper()
    mkt = str(row["market"]).strip().upper()
    info = pos_map.get((tkr, mkt), {})
    entry_text = info.get("entry_text", "")
    entry_price = info.get("entry_price", np.nan)
    entry_date = info.get("entry_date", "")

    entry_text_list.append(entry_text)
    entry_price_list.append(entry_price)

    if tkr not in detail_saved:
        sell_sig.append("N/A")
        sell_reason.append("근거 데이터 없음")
        hold_days_list.append(np.nan)
        stop_by_entry_list.append(np.nan)
        target_by_entry_list.append(np.nan)
        continue

    df_ind, _reason_df = detail_saved[tkr]
    last = df_ind.iloc[-1]
    sig, reason, stp, tgt, hd = sell_recommendation(last, params, entry_price, entry_date)

    sell_sig.append(sig)
    sell_reason.append(reason)
    hold_days_list.append(np.nan if hd is None else int(hd))
    stop_by_entry_list.append(np.nan if stp is None else float(stp))
    target_by_entry_list.append(np.nan if tgt is None else float(tgt))

df_out["entry_text"] = entry_text_list
df_out["entry_price"] = entry_price_list
df_out["sell_signal"] = sell_sig
df_out["sell_reason"] = sell_reason
df_out["hold_days"] = hold_days_list
df_out["stop_by_entry"] = stop_by_entry_list
df_out["target_by_entry(2R)"] = target_by_entry_list

# -----------------------------
# Results table (wide)
# -----------------------------
st.markdown("---")
st.subheader("결과 요약 (매수/매도 추천 포함)")

df_view = df_out.copy()
df_view = df_view.rename(columns={"OX":"O/X", "target_2R":"target(2R)"})

for col in ["close", "stop", "target(2R)", "entry_price", "stop_by_entry", "target_by_entry(2R)"]:
    if col in df_view.columns:
        df_view[col] = df_view.apply(lambda r: format_currency(r.get("market",""), r.get(col, np.nan)), axis=1)

wide_cfg = {
    "market": st.column_config.TextColumn("market", width="small"),
    "ticker": st.column_config.TextColumn("ticker", width="small"),
    "O/X": st.column_config.TextColumn("O/X", width="small"),
    "candidate": st.column_config.NumberColumn("cand", width="small"),
    "score": st.column_config.NumberColumn("score", width="small"),
    "date": st.column_config.TextColumn("date", width="small"),
    "close": st.column_config.TextColumn("close", width="medium"),
    "stop": st.column_config.TextColumn("stop", width="medium"),
    "target(2R)": st.column_config.TextColumn("target(2R)", width="medium"),
    "entry_text": st.column_config.TextColumn("평단 입력", width="large"),
    "entry_price": st.column_config.TextColumn("평단(계산)", width="medium"),
    "sell_signal": st.column_config.TextColumn("sell", width="small"),
    "sell_reason": st.column_config.TextColumn("sell_reason", width="large"),
    "hold_days": st.column_config.NumberColumn("hold_days", width="small"),
    "stop_by_entry": st.column_config.TextColumn("stop(entry)", width="medium"),
    "target_by_entry(2R)": st.column_config.TextColumn("target(entry)", width="medium"),
    "qty": st.column_config.NumberColumn("qty", width="small"),
    "vol_ratio": st.column_config.NumberColumn("vol_ratio", width="small"),
    "atr_pct": st.column_config.NumberColumn("atr% ", width="small"),
    "ret_20": st.column_config.NumberColumn("ret20% ", width="small"),
    "error": st.column_config.TextColumn("error", width="large"),
}

show_cols = [
    "market","ticker","O/X","candidate","score","date",
    "close","stop","target(2R)",
    "entry_text","entry_price",
    "sell_signal","sell_reason","hold_days",
    "stop_by_entry","target_by_entry(2R)",
    "qty","vol_ratio","atr_pct","ret_20","error"
]

st.dataframe(
    df_view[show_cols],
    use_container_width=True,
    hide_index=True,
    column_config=wide_cfg
)

n_cand = int((df_out["candidate"] == 1).sum())
if n_cand == 0:
    st.error("NOT TODAY: 조건을 만족하는 후보가 없습니다. (O=0)")
else:
    st.success(f"후보(O) {n_cand}개")

# -----------------------------
# Evidence (table + charts)
# -----------------------------
st.markdown("---")
st.subheader("근거(조건표 + 차트)")

for _, row in df_out.iterrows():
    tkr = str(row["ticker"]).upper()
    mkt = str(row.get("market","")).strip().upper()
    ox = row.get("O/X", row.get("OX",""))
    sig = row.get("sell_signal","")
    err = row.get("error","")

    title = f"{tkr} ({mkt}) | 매수:{ox} | 매도:{sig}"
    with st.expander(title, expanded=False):
        if err:
            st.error(f"데이터 오류: {err}")
            continue
        if tkr not in detail_saved:
            st.warning("근거 데이터 없음")
            continue

        df_ind, reason_df = detail_saved[tkr]
        st.write("매수 조건 근거(통과 여부)")
        st.dataframe(reason_df, use_container_width=True, hide_index=True)

        entry_price = pos_map.get((tkr, mkt), {}).get("entry_price", np.nan)
        entry_date = pos_map.get((tkr, mkt), {}).get("entry_date", "")

        stop_e, target_e = None, None
        if entry_price is not None and not (isinstance(entry_price, float) and np.isnan(entry_price)) and entry_price > 0:
            last = df_ind.iloc[-1]
            _sig, _reason, stop_e, target_e, _hd = sell_recommendation(last, params, float(entry_price), entry_date)

        st.write("가격 차트 (Close + MA + 평단/손절/목표 수평선)")
        st.altair_chart(price_chart(df_ind, entry=entry_price, stop=stop_e, target=target_e), use_container_width=True)

        st.write("거래량 차트 (Volume + 평균)")
        st.altair_chart(volume_chart(df_ind), use_container_width=True)

# -----------------------------
# Excel download
# -----------------------------
st.markdown("---")
st.subheader("엑셀 다운로드")

df_excel = df_out.copy()
df_excel = df_excel.rename(columns={"OX":"O/X", "target_2R":"target(2R)"})
xlsx_bytes = build_excel_bytes_with_formats(df_excel)

st.download_button(
    label="엑셀 다운로드 (KR ₩ / US $ 자동 적용)",
    data=xlsx_bytes,
    file_name="Swing_Scanner_Output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
