# =========================================================
# Swing Scanner FINAL (KR / US) - ALL FEATURES INTEGRATED
# - 균형추천(옵션)
# - 추천 실행 후 티커 입력창 자동 반영(확정)
# - 보유입력(평단/진입일) 2번 입력 문제 해결(form 저장 방식)
# - 회사명(KR: pykrx / US: yfinance)
# - 결과 근거(조건표 + 차트)
# - 매도 추천(HOLD / PARTIAL SELL / SELL)
# - 추천 스캔 Top30 (입력창 아래, 가로 가시성 최대)
# - 엑셀 다운로드(KR ₩ / US $ 행별 서식 적용)
# - 사이드바 카테고리 구분 + 한글 설명 문구
# - 계좌 가정값 콤마 입력 지원
# =========================================================

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

# 추천용 유니버스(원하면 여기 늘리세요)
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
# Page + CSS (회사명 줄바꿈 / 표 가독성)
# -----------------------------
st.set_page_config(page_title="Swing Scanner", layout="wide")

st.markdown(
    """
    <style>
    /* 데이터프레임 셀 줄바꿈(회사명 길어도 보이게) */
    div[data-testid="stDataFrame"] .ag-cell {
        white-space: normal !important;
        line-height: 1.25 !important;
    }
    div[data-testid="stDataFrame"] .ag-row { height: auto !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# 폰트 크기(방법1: HTML)
st.markdown(
    "<h1 style='font-size:34px;font-weight:700;margin:0'>웹 티커 입력 → 스윙 판단(O/X) + 근거(표+차트) + 매도 추천 + 엑셀</h1>",
    unsafe_allow_html=True
)

# =========================================================
# Utils
# =========================================================
def is_kr_code(x: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", str(x).strip()))

def normalize_tickers(raw: str):
    items = re.split(r"[,\n\s]+", (raw or "").strip())
    return [x.strip().upper() for x in items if x.strip()]

def parse_entry_text(market: str, s: str):
    """KR: 정수(원), US: 소수(달러)"""
    if s is None:
        return np.nan
    t = str(s).strip()
    if not t:
        return np.nan

    m = str(market).strip().upper() if market is not None else ""
    raw = t.replace("₩", "").replace("$", "").replace(" ", "").replace(",", "")

    try:
        val = float(raw)

        # market이 이상하면 통화기호로 추정
        if m not in ("KR", "US"):
            if "₩" in t:
                m = "KR"
            elif "$" in t:
                m = "US"

        if m == "KR":
            return int(val)
        if m == "US":
            return round(val, 2)

        # 최후 추정
        return round(val, 2) if "." in raw else int(val)
    except Exception:
        return np.nan

def format_currency(mkt: str, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    m = str(mkt).strip().upper()
    try:
        if m == "KR":
            return f"₩{int(round(float(v))):,}"
        if m == "US":
            return f"${float(v):,.2f}"
        return str(v)
    except Exception:
        return str(v)

# =========================================================
# Company Names
# =========================================================
@st.cache_data(ttl=60*60*12, show_spinner=False)
def get_kr_name(code: str) -> str:
    try:
        return krx.get_market_ticker_name(code) or ""
    except Exception:
        return ""

@st.cache_data(ttl=60*60*12, show_spinner=False)
def get_us_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or info.get("displayName") or ""
    except Exception:
        return ""

def get_company_name(ticker: str) -> str:
    if is_kr_code(ticker):
        return get_kr_name(ticker) or ""
    return get_us_name(ticker) or ""

# =========================================================
# Indicators
# =========================================================
def compute_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
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

# =========================================================
# Data Load (cache)
# =========================================================
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

    # MultiIndex 처리 (yfinance 버전에 따라 다름)
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)
        if ticker in lv0:
            df = df[ticker]
        elif ticker in lv1:
            df = df.xs(ticker, axis=1, level=1)
        else:
            # fallback: 첫 ticker 선택
            uniq1 = list(pd.unique(lv1))
            if uniq1:
                df = df.xs(uniq1[0], axis=1, level=1)
            else:
                raise ValueError(f"Unexpected MultiIndex columns: {df.columns}")

    # 컬럼명 안정화(튜플/기타 타입 대비)
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
    return _load_kr_cached(ticker, lookback_years) if is_kr_code(ticker) else _load_us_cached(ticker, lookback_years)

# =========================================================
# Sell Recommendation
# =========================================================
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
        target_price = float(entry_price + 2 * (entry_price - stop_price))

    # 1) ATR 손절
    if stop_price is not None and close < stop_price:
        return "SELL", "손절가 이탈(ATR 기준)", stop_price, target_price, holding_days

    # 2) 2R 목표 도달
    if target_price is not None and close >= target_price:
        return "PARTIAL SELL", "목표가(2R) 도달", stop_price, target_price, holding_days

    # 3) 추세 약화
    ma_fast = last.get("MA_FAST", np.nan)
    ma_slow = last.get("MA_SLOW", np.nan)
    if (pd.notna(ma_fast) and close < ma_fast) or (pd.notna(ma_fast) and pd.notna(ma_slow) and ma_fast < ma_slow):
        return "PARTIAL SELL", "추세 이탈(이평선 기준)", stop_price, target_price, holding_days

    # 4) 보유기간 초과
    if holding_days is not None and holding_days >= int(p["HOLD_DAYS"]):
        return "SELL", "보유 기간 초과(HOLD_DAYS)", stop_price, target_price, holding_days

    return "HOLD", "추세 유지", stop_price, target_price, holding_days

# =========================================================
# Analyze One
# =========================================================
def analyze_one_with_detail(ticker: str, p: dict):
    try:
        market = "KR" if is_kr_code(ticker) else "US"
        name = get_company_name(ticker)

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
            name=name,
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
        return dict(
            market=("KR" if is_kr_code(ticker) else "US"),
            ticker=ticker,
            name=get_company_name(ticker),
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
        ), None, None

# =========================================================
# Recommendation (균형추천)
# =========================================================
def recommend_top_balanced(p: dict, top_n: int = 10, kr_n: int = 5, us_n: int = 5):
    universe = KR_UNIVERSE + US_UNIVERSE
    rows = []
    prog = st.progress(0)
    total = max(1, len(universe))

    for i, t in enumerate(universe, start=1):
        res, _df, _reason = analyze_one_with_detail(t, p)
        rows.append(res)
        prog.progress(int(i * 100 / total))
    prog.empty()

    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["market"] = df["market"].astype(str).str.strip().str.upper()
    df["name"] = df["ticker"].apply(get_company_name)

    df_ok = df[df["error"].astype(str) == ""].copy()
    df_ok = df_ok.sort_values(["candidate", "score"], ascending=[False, False])

    kr_list = df_ok[df_ok["market"] == "KR"]["ticker"].tolist()
    us_list = df_ok[df_ok["market"] == "US"]["ticker"].tolist()

    picks = []
    picks += kr_list[:max(0, int(kr_n))]
    picks += us_list[:max(0, int(us_n))]

    for t in df_ok["ticker"].tolist():
        if t not in picks:
            picks.append(t)
        if len(picks) >= top_n:
            break

    if len(picks) < top_n:
        df_all_sorted = df.sort_values(["candidate", "score"], ascending=[False, False])
        for t in df_all_sorted["ticker"].tolist():
            if t not in picks:
                picks.append(t)
            if len(picks) >= top_n:
                break

    return picks[:top_n], df

# =========================================================
# Charts (Altair)
# =========================================================
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

# =========================================================
# Excel (KR/US row formats)
# =========================================================
def build_excel_bytes_with_formats(df_all: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="Signals_All", index=False)

        df_cand = df_all[df_all.get("candidate", 0) == 1].copy()
        if df_cand.empty:
            df_cand = pd.DataFrame([{"note": "nottoday"}])
        df_cand.to_excel(writer, sheet_name="Candidates", index=False)

        # 주문표(간단)
        def make_order(df, market):
            d = df[(df["market"] == market) & (df["candidate"] == 1)].copy()
            if d.empty:
                return pd.DataFrame([{
                    "market": market, "ticker": "nottoday", "name": "",
                    "action": "NONE", "qty": 0,
                    "stop": np.nan, "target_2R": np.nan,
                    "note": "No candidates"
                }])
            d = d.sort_values("score", ascending=False).head(10)
            d["qty_safe"] = (d["qty"] * 0.5).astype(int)
            o = pd.DataFrame({
                "market": d["market"],
                "ticker": d["ticker"],
                "name": d["name"],
                "action": "BUY",
                "qty": d["qty_safe"],
                "stop": d["stop"],
                "target_2R": d["target_2R"],
                "note": ["Manual entry (M-STOCK)"] * len(d)
            })
            return o

        make_order(df_all, "KR").to_excel(writer, sheet_name="Order_KR", index=False)
        make_order(df_all, "US").to_excel(writer, sheet_name="Order_US", index=False)

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
                "close", "stop", "target_2R", "target_2r", "target(2R)", "target_2R(2R)",
                "entry_price", "stop_by_entry", "target_by_entry_2r", "target_by_entry(2R)"
            }
            price_idxs = [header[c] for c in list(header.keys()) if c in price_cols]
            # fallback: known cols if present
            for c in ["close","stop","target_2R","entry_price","stop_by_entry","target_by_entry(2R)"]:
                if c in header and header[c] not in price_idxs:
                    price_idxs.append(header[c])

            for r in range(2, ws.max_row + 1):
                mkt = ws.cell(row=r, column=market_col).value
                numfmt = fmt_krw if mkt == "KR" else (fmt_usd if mkt == "US" else None)
                if not numfmt:
                    continue
                for c in price_idxs:
                    cell = ws.cell(row=r, column=c)
                    if isinstance(cell.value, (int, float)) and cell.value is not None:
                        cell.number_format = numfmt

        for name in wb.sheetnames:
            apply_formats(wb[name])

    return out.getvalue()

# =========================================================
# Positions sync (run 버튼에서만) - ticker+market 기준
# =========================================================
def sync_positions_on_run(df_analysis: pd.DataFrame):
    base = df_analysis[["ticker", "market", "name"]].copy()
    base["ticker"] = base["ticker"].astype(str).str.upper()
    base["market"] = base["market"].astype(str).str.strip().str.upper()

    ed = st.session_state["positions_store"].copy()
    if not ed.empty:
        ed["ticker"] = ed["ticker"].astype(str).str.upper()
        ed["market"] = ed["market"].astype(str).str.strip().str.upper()

    ed = base.merge(ed, on=["ticker", "market"], how="left", suffixes=("", "_old"))
    ed["name"] = ed["name"].fillna("")
    ed["entry_text"] = ed["entry_text"].fillna("")
    ed["entry_date"] = ed["entry_date"].fillna("")
    st.session_state["positions_store"] = ed[["ticker","market","name","entry_text","entry_date"]]

# =========================================================
# Session init
# =========================================================
if "analysis_df" not in st.session_state:
    st.session_state["analysis_df"] = None
if "analysis_detail" not in st.session_state:
    st.session_state["analysis_detail"] = {}
if "positions_store" not in st.session_state:
    st.session_state["positions_store"] = pd.DataFrame(columns=["ticker","market","name","entry_text","entry_date"])
if "ACCOUNT_SIZE" not in st.session_state:
    st.session_state["ACCOUNT_SIZE"] = DEFAULTS["ACCOUNT_SIZE"]
if "show_scan_table" not in st.session_state:
    st.session_state["show_scan_table"] = False
if "last_reco_scan_df" not in st.session_state:
    st.session_state["last_reco_scan_df"] = None

# 티커 입력창 key(추천 자동 입력 안정화)
if "ticker_input_area" not in st.session_state:
    st.session_state["ticker_input_area"] = "005930 000660\nSPY QQQ"

# =========================================================
# Sidebar (카테고리 구분 + 설명)
# =========================================================
with st.sidebar:
    st.header("스윙 전략 설정")
    params = {}

    with st.expander("① 추세 판단 (이동평균)", expanded=True):
        params["MA_FAST"] = st.number_input("MA_FAST (단기 이동평균)", 5, 200, DEFAULTS["MA_FAST"], key="MA_FAST")
        st.write("최근 주가의 단기 흐름(10~30일 권장). 값이 작으면 민감, 크면 안정적.")
        params["MA_SLOW"] = st.number_input("MA_SLOW (장기 이동평균)", 10, 300, DEFAULTS["MA_SLOW"], key="MA_SLOW")
        st.write("중·장기 추세 기준선(50~120일 권장). 단기선이 장기선 위면 상승 추세로 판단.")

    with st.expander("② 거래량 · 변동성 조건", expanded=False):
        params["VOL_LOOKBACK"] = st.number_input("VOL_LOOKBACK (거래량 평균 기간)", 5, 200, DEFAULTS["VOL_LOOKBACK"], key="VOL_LOOKBACK")
        st.write("평균 거래량을 계산하는 기간(예: 20일).")
        params["VOL_SPIKE"] = st.number_input("VOL_SPIKE (거래량 급증 기준)", 1.0, 5.0, float(DEFAULTS["VOL_SPIKE"]), step=0.05, key="VOL_SPIKE")
        st.write("현재 거래량이 평균 대비 몇 배 이상이면 통과인지(예: 1.50 = 평균의 150%).")
        params["ATR_PERIOD"] = st.number_input("ATR_PERIOD (ATR 계산 기간)", 5, 100, DEFAULTS["ATR_PERIOD"], key="ATR_PERIOD")
        st.write("ATR(평균 변동폭) 계산 기간(보통 14일).")
        params["ATR_PCT_MIN"] = st.number_input("ATR_PCT_MIN (최소 변동성)", 0.0, 0.2, float(DEFAULTS["ATR_PCT_MIN"]), step=0.001, format="%.3f", key="ATR_PCT_MIN")
        st.write("너무 안 움직이는 종목 제외(최소 변동성).")
        params["ATR_PCT_MAX"] = st.number_input("ATR_PCT_MAX (최대 변동성)", 0.0, 0.5, float(DEFAULTS["ATR_PCT_MAX"]), step=0.001, format="%.3f", key="ATR_PCT_MAX")
        st.write("너무 위험하게 출렁이는 종목 제외(최대 변동성).")

    with st.expander("③ 리스크 · 손절 · 보유 관리", expanded=False):
        params["STOP_ATR_MULT"] = st.number_input("STOP_ATR_MULT (손절 ATR 배수)", 0.5, 5.0, float(DEFAULTS["STOP_ATR_MULT"]), step=0.1, key="STOP_ATR_MULT")
        st.write("손절 폭을 ATR 기준으로 설정(보통 1.5~2.0).")
        params["HOLD_DAYS"] = st.number_input("HOLD_DAYS (최대 보유일)", 1, 200, DEFAULTS["HOLD_DAYS"], key="HOLD_DAYS")
        st.write("오래 들고 가는 것을 제한하는 규칙(예: 20일).")

    with st.expander("④ 계좌 · 포지션 사이징", expanded=False):
        acc_str = st.text_input("ACCOUNT_SIZE (총 투자금, 콤마 가능)", value=f"{int(st.session_state['ACCOUNT_SIZE']):,}", key="ACCOUNT_SIZE_TEXT")
        try:
            params["ACCOUNT_SIZE"] = int(str(acc_str).replace(",", "").strip())
            st.session_state["ACCOUNT_SIZE"] = params["ACCOUNT_SIZE"]
        except Exception:
            params["ACCOUNT_SIZE"] = st.session_state["ACCOUNT_SIZE"]
        st.write("실제 주문이 아니라 수량 계산(포지션 사이징)에만 사용.")

        params["RISK_PER_TRADE"] = st.number_input("RISK_PER_TRADE (1회 최대 손실 비율)", 0.001, 0.05, float(DEFAULTS["RISK_PER_TRADE"]), step=0.001, format="%.3f", key="RISK_PER_TRADE")
        st.write("한 종목에서 감수할 최대 손실 비율(예: 0.010 = 1%).")

    with st.expander("⑤ 추천 옵션 (KR/US 균형)", expanded=True):
        st.checkbox("균형 추천 사용", value=True, key="BALANCED_ON")
        st.number_input("KR 추천 개수", 0, 10, 5, key="KR_PICK")
        st.number_input("US 추천 개수", 0, 10, 5, key="US_PICK")

    params["LOOKBACK_YEARS"] = DEFAULTS["LOOKBACK_YEARS"]
    params["TOP_N"] = DEFAULTS["TOP_N"]

# =========================================================
# Main: Input + 추천/스캔 + 실행
# =========================================================
st.write("입력: KR은 6자리(예: 005930), US는 티커(예: SPY). 콤마/줄바꿈/공백 가능.")

# 버튼 영역
b1, b2, b3 = st.columns([2, 1, 1])

with b1:
    if st.button("TOP10 추천 → 입력칸 채우기", key="btn_reco"):
        top_n = int(params["TOP_N"])
        if st.session_state.get("BALANCED_ON", True):
            picks, df_scan = recommend_top_balanced(
                params,
                top_n=top_n,
                kr_n=int(st.session_state.get("KR_PICK", 5)),
                us_n=int(st.session_state.get("US_PICK", 5)),
            )
        else:
            picks, df_scan = recommend_top_balanced(params, top_n=top_n, kr_n=0, us_n=0)

        st.session_state["last_reco_scan_df"] = df_scan

        if picks:
            kr_list = [t for t in picks if is_kr_code(t)]
            us_list = [t for t in picks if not is_kr_code(t)]
            text = ""
            if kr_list:
                text += " ".join(kr_list)
            if us_list:
                text += ("\n" if text else "") + " ".join(us_list)

            # ✅ 핵심: text_area key 값을 직접 업데이트
            st.session_state["ticker_input_area"] = text
            st.session_state["show_scan_table"] = True
            st.rerun()
        else:
            st.warning("추천 실패: 데이터/네트워크/후보풀 점검 필요")

with b2:
    if st.button("입력칸 비우기", key="btn_clear"):
        st.session_state["ticker_input_area"] = ""
        st.rerun()

with b3:
    if st.button("추천 스캔 보기 ON/OFF", key="btn_scan_toggle"):
        st.session_state["show_scan_table"] = not st.session_state["show_scan_table"]

# 입력창: value 주지 말고 key만 (자동 입력 안정화)
raw = st.text_area("티커 입력", height=120, key="ticker_input_area")

# 추천 스캔표: 입력창 바로 아래 + 넓게
if st.session_state.get("show_scan_table", False):
    df_scan = st.session_state.get("last_reco_scan_df", None)
    if isinstance(df_scan, pd.DataFrame) and not df_scan.empty:
        scan_view = df_scan.copy()
        scan_view["ticker"] = scan_view["ticker"].astype(str).str.upper()
        scan_view["market"] = scan_view["market"].astype(str).str.strip().str.upper()
        scan_view["name"] = scan_view["ticker"].apply(get_company_name)

        st.caption("추천 스캔 Top30 (가로 가시성 최대)")
        st.dataframe(
            scan_view.sort_values(["candidate", "score"], ascending=[False, False]).head(30)[
                ["market","ticker","name","OX","candidate","score","date","error"]
            ],
            use_container_width=True,
            hide_index=True,
            height=280,
            column_config={
                "market": st.column_config.TextColumn("MKT", width="small"),
                "ticker": st.column_config.TextColumn("TICKER", width="small"),
                "name": st.column_config.TextColumn("회사명", width="large"),
                "OX": st.column_config.TextColumn("O/X", width="small"),
                "candidate": st.column_config.NumberColumn("cand", width="small"),
                "score": st.column_config.NumberColumn("score", width="small"),
                "date": st.column_config.TextColumn("date", width="small"),
                "error": st.column_config.TextColumn("error", width="large"),
            }
        )
    else:
        st.info("추천 실행 후 스캔표가 표시됩니다.")

# 분석 실행
run = st.button("분석 실행", key="btn_run")

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
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["market"] = df["market"].astype(str).str.strip().str.upper()
        df["name"] = df["ticker"].apply(get_company_name)
        df = df.sort_values(["candidate","score"], ascending=[False, False]).reset_index(drop=True)

        st.session_state["analysis_df"] = df
        st.session_state["analysis_detail"] = detail_map

        # 보유입력 동기화(run에서만!)
        sync_positions_on_run(df)

# =========================================================
# Render results
# =========================================================
df_saved = st.session_state.get("analysis_df", None)
detail_saved = st.session_state.get("analysis_detail", {})

if df_saved is None or df_saved.empty:
    st.info("분석 실행을 눌러 결과를 생성하세요.")
    st.stop()

# -----------------------------
# Positions (평단 입력 안정화: form 저장 방식)
# -----------------------------
st.markdown("---")
st.subheader("보유 입력 (평단/진입일)")

st.write(
    "- KR 평단 예시: `₩10,000,000` / `10,000,000`\n"
    "- US 평단 예시: `$123.45` / `123.45`\n"
    "입력 후 아래 [보유 입력 저장]을 누르면 확정되며, 저장된 값으로 매도추천/차트 기준선이 계산됩니다."
)

with st.form("positions_form", clear_on_submit=False):
    edited_positions = st.data_editor(
        st.session_state["positions_store"],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="positions_editor",
        column_config={
            "ticker": st.column_config.TextColumn("ticker", disabled=True, width="small"),
            "market": st.column_config.TextColumn("market", disabled=True, width="small"),
            "name": st.column_config.TextColumn("회사명", disabled=True, width="large"),
            "entry_text": st.column_config.TextColumn("평단 입력 (KR: ₩... / US: $...)", width="large"),
            "entry_date": st.column_config.TextColumn("진입일(YYYY-MM-DD)", width="medium"),
        },
    )
    save_positions = st.form_submit_button("보유 입력 저장")

if save_positions:
    st.session_state["positions_store"] = edited_positions.copy()
    st.success("보유 입력이 저장되었습니다.")

pos = st.session_state["positions_store"].copy()
pos["ticker"] = pos["ticker"].astype(str).str.upper()
pos["market"] = pos["market"].astype(str).str.strip().str.upper()
pos["entry_price"] = [parse_entry_text(m, t) for m, t in zip(pos["market"], pos["entry_text"])]
pos["entry_display"] = [format_currency(m, v) for m, v in zip(pos["market"], pos["entry_price"])]

st.caption("계산된 평단(읽기 전용)")
st.dataframe(
    pos[["market","ticker","name","entry_display","entry_date"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "market": st.column_config.TextColumn("MKT", width="small"),
        "ticker": st.column_config.TextColumn("TICKER", width="small"),
        "name": st.column_config.TextColumn("회사명", width="large"),
        "entry_display": st.column_config.TextColumn("계산된 평단", width="medium"),
        "entry_date": st.column_config.TextColumn("진입일", width="medium"),
    }
)

# map for sell signals
pos_map = {}
for _, r in pos.iterrows():
    pos_map[(r["ticker"], r["market"])] = dict(
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
df_out["name"] = df_out["ticker"].apply(get_company_name)

sell_sig, sell_reason, hold_days_list = [], [], []
stop_by_entry_list, target_by_entry_list = [], []
entry_text_list, entry_price_list = [], []

for _, row in df_out.iterrows():
    tkr = row["ticker"]
    mkt = row["market"]
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
st.subheader("결과 요약 (회사명 + 매수/매도 추천 포함)")

# 표시용(문자열 통화 포맷)
df_view = df_out.rename(columns={"OX":"O/X", "target_2R":"target(2R)"}).copy()
for col in ["close", "stop", "target(2R)", "entry_price", "stop_by_entry", "target_by_entry(2R)"]:
    if col in df_view.columns:
        df_view[col] = df_view.apply(lambda r: format_currency(r["market"], r.get(col, np.nan)), axis=1)

show_cols = [
    "market","ticker","name","O/X","candidate","score","date",
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
    height=440,
    column_config={
        "market": st.column_config.TextColumn("MKT", width="small"),
        "ticker": st.column_config.TextColumn("TICKER", width="small"),
        "name": st.column_config.TextColumn("회사명", width="large"),
        "sell_reason": st.column_config.TextColumn("sell_reason", width="large"),
        "entry_text": st.column_config.TextColumn("평단 입력", width="large"),
        "error": st.column_config.TextColumn("error", width="large"),
    }
)

# ✅ 삼항연산 금지(DeltaGenerator 출력 방지)
n_cand = int((df_out["candidate"] == 1).sum())
if n_cand > 0:
    st.success(f"후보(O) {n_cand}개")
else:
    st.error("NOT TODAY: 조건을 만족하는 후보가 없습니다. (O=0)")

# -----------------------------
# Evidence (table + charts)
# -----------------------------
st.markdown("---")
st.subheader("근거(조건표 + 차트)")

for _, row in df_out.iterrows():
    tkr = row["ticker"]
    mkt = row["market"]
    nm  = str(row.get("name",""))
    ox = row.get("O/X", row.get("OX",""))
    sig = row.get("sell_signal","")
    err = row.get("error","")

    title = f"{tkr} | {nm} | ({mkt}) | 매수:{ox} | 매도:{sig}"
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

        st.write("가격 차트 (Close + MA + 평단/손절/목표)")
        st.altair_chart(price_chart(df_ind, entry=entry_price, stop=stop_e, target=target_e), use_container_width=True)

        st.write("거래량 차트 (Volume + 평균)")
        st.altair_chart(volume_chart(df_ind), use_container_width=True)

# -----------------------------
# Excel download (numeric 기반 + 서식 적용)
# -----------------------------
st.markdown("---")
st.subheader("엑셀 다운로드")

# 엑셀은 숫자 컬럼 유지해야 서식 적용됨(표시용 df_view 말고 df_out 사용)
df_excel = df_out.copy()
xlsx_bytes = build_excel_bytes_with_formats(df_excel)

st.download_button(
    label="엑셀 다운로드 (KR ₩ / US $ 자동 적용)",
    data=xlsx_bytes,
    file_name="Swing_Scanner_Output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# -----------------------------
# Next Steps (3가지)
# -----------------------------
with st.expander("다음 단계(3가지)"):
    st.write(
        "1) 유니버스 확장: KR/US 추천 풀을 더 늘려서 후보 품질 개선\n"
        "2) 손절/익절 규칙 고도화: 트레일링 스탑, 분할매도(1R/2R/3R) 옵션\n"
        "3) 리포트 자동화: 후보 리스트/보유현황을 매일 1회 엑셀로 자동 저장(배포 환경 기준)"
    )
