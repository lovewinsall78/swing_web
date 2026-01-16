import re
import io
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock as krx
import streamlit as st
from datetime import datetime, timedelta
import altair as alt

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

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    df["MA_FAST"] = df["Close"].rolling(int(params["MA_FAST"])).mean()
    df["MA_SLOW"] = df["Close"].rolling(int(params["MA_SLOW"])).mean()
    df["VOL_AVG"] = df["Volume"].rolling(int(params["VOL_LOOKBACK"])).mean()
    df["VOL_RATIO"] = np.where(df["VOL_AVG"] > 0, df["Volume"] / df["VOL_AVG"], np.nan)
    df["ATR"] = compute_atr(df, int(params["ATR_PERIOD"]))
    df["ATR_PCT"] = df["ATR"] / df["Close"]
    df["RET_20"] = df["Close"].pct_change(20)
    return df

def rule_signal(last: pd.Series, params: dict) -> int:
    needed = ["MA_FAST", "MA_SLOW", "VOL_RATIO", "ATR_PCT", "Close"]
    if any(pd.isna(last.get(k, np.nan)) for k in needed):
        return 0
    trend_ok = (last["MA_FAST"] > last["MA_SLOW"]) and (last["Close"] > last["MA_FAST"])
    vol_ok   = (last["VOL_RATIO"] >= float(params["VOL_SPIKE"]))
    atr_ok   = (float(params["ATR_PCT_MIN"]) <= last["ATR_PCT"] <= float(params["ATR_PCT_MAX"]))
    return int(trend_ok and vol_ok and atr_ok)

def score_row(last: pd.Series) -> int:
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

def position_size(entry: float, atr: float, account_size: float, risk_per_trade: float, stop_atr_mult: float):
    if pd.isna(atr) or atr <= 0:
        return 0, np.nan
    stop_dist = stop_atr_mult * atr
    risk_amount = account_size * risk_per_trade
    qty = int(risk_amount / stop_dist) if stop_dist > 0 else 0
    stop_price = entry - stop_dist
    return max(0, qty), float(stop_price)

def build_reason_table(last: pd.Series, params: dict) -> pd.DataFrame:
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

    vol_spike = float(params["VOL_SPIKE"])
    c3 = (vol_ratio is not None) and (vol_ratio >= vol_spike)
    rows.append({"조건": "거래량: VOL_RATIO >= VOL_SPIKE",
                 "현재값": "데이터 부족" if vol_ratio is None else f"{vol_ratio:.2f}",
                 "기준": f">= {vol_spike:.2f}", "통과": bool(c3)})

    atr_min = float(params["ATR_PCT_MIN"])
    atr_max = float(params["ATR_PCT_MAX"])
    c4 = (atr_pct is not None) and (atr_min <= atr_pct <= atr_max)
    rows.append({"조건": "변동성: ATR_PCT_MIN <= ATR_PCT <= ATR_PCT_MAX",
                 "현재값": "데이터 부족" if atr_pct is None else f"{atr_pct*100:.2f}%",
                 "기준": f"{atr_min*100:.2f}% ~ {atr_max*100:.2f}%", "통과": bool(c4)})

    return pd.DataFrame(rows)

def format_currency_for_display(market: str, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    try:
        if market == "KR":
            return f"₩{float(v):,.0f}"
        if market == "US":
            return f"${float(v):,.2f}"
        return str(v)
    except Exception:
        return str(v)

# =========================
# KR/US 평단 입력 파싱
# =========================
def parse_entry_text(market: str, s: str):
    """
    KR: ₩10,000,000 / 10,000,000 / 10000000 -> int(원)
    US: $123.45 / 1,234.56 / 123.4         -> float(달러, 2dp)
    """
    if s is None:
        return np.nan
    t = str(s).strip()
    if t == "":
        return np.nan

    t = t.replace("₩", "").replace("$", "").replace(" ", "").replace(",", "")
    try:
        if market == "KR":
            return int(float(t))
        return round(float(t), 2)
    except Exception:
        return np.nan

def compute_entry_price_column(pos_df: pd.DataFrame) -> pd.DataFrame:
    """entry_text를 기반으로 entry_price(계산용)를 매번 재계산. entry_text는 건드리지 않음."""
    df = pos_df.copy()
    prices = []
    for _, r in df.iterrows():
        mkt = str(r.get("market", "")).strip()
        txt = r.get("entry_text", "")
        prices.append(parse_entry_text(mkt, txt))
    df["entry_price"] = prices
    return df

# =========================
# 데이터 로더
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

    # ✅ MultiIndex 방어
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)

        if ticker in lv0:
            df = df[ticker]  # (ticker, field)
        elif ticker in lv1:
            df = df.xs(ticker, axis=1, level=1)  # (field, ticker)
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

def load_kr(code: str) -> pd.DataFrame:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * DEFAULTS["LOOKBACK_YEARS"])).strftime("%Y-%m-%d")
    df = krx.get_market_ohlcv_by_date(start, end, code)
    df = df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

# =========================
# 매도 추천
# =========================
def sell_recommendation(last: pd.Series, params: dict, entry_price: float, entry_date: str):
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
        stop_price = float(entry_price - float(params["STOP_ATR_MULT"]) * atr)
        target_price = float(entry_price + 2 * (entry_price - stop_price))  # 2R

    # 1) 손절
    if stop_price is not None and close < stop_price:
        return "SELL", "손절가 이탈(ATR 기준)", stop_price, target_price, holding_days

    # 2) 목표가
    if target_price is not None and close >= target_price:
        return "PARTIAL SELL", "목표가(2R) 도달", stop_price, target_price, holding_days

    # 3) 추세 이탈
    ma_fast = last.get("MA_FAST", np.nan)
    ma_slow = last.get("MA_SLOW", np.nan)
    trend_exit = False
    if pd.notna(ma_fast) and close < ma_fast:
        trend_exit = True
    if pd.notna(ma_fast) and pd.notna(ma_slow) and ma_fast < ma_slow:
        trend_exit = True
    if trend_exit:
        return "PARTIAL SELL", "추세 이탈(이평선 기준)", stop_price, target_price, holding_days

    # 4) 보유기간 초과
    if holding_days is not None and holding_days >= int(params["HOLD_DAYS"]):
        return "SELL", "보유 기간 초과(HOLD_DAYS)", stop_price, target_price, holding_days

    return "HOLD", "추세 유지", stop_price, target_price, holding_days

# =========================
# 분석(티커 1개)
# =========================
def analyze_one_with_df(ticker: str, params: dict):
    try:
        if is_kr_code(ticker):
            market = "KR"
            df = load_kr(ticker)
        else:
            market = "US"
            df = load_us(ticker)

        df_ind = add_indicators(df, params)
        last = df_ind.iloc[-1]

        cand = rule_signal(last, params)
        sc = score_row(last) if cand == 1 else 0

        entry = float(last["Close"])
        qty, stop_buy = position_size(
            entry=entry,
            atr=float(last.get("ATR", np.nan)),
            account_size=float(params["ACCOUNT_SIZE"]),
            risk_per_trade=float(params["RISK_PER_TRADE"]),
            stop_atr_mult=float(params["STOP_ATR_MULT"]),
        )
        target_buy = entry + 2 * (entry - stop_buy) if (not np.isnan(stop_buy)) else np.nan
        reason_df = build_reason_table(last, params)

        result = {
            "market": market,
            "ticker": ticker,
            "O/X": "O" if cand == 1 else "X",
            "candidate": int(cand),
            "score": int(sc),
            "close": float(entry),
            "stop": (np.nan if np.isnan(stop_buy) else float(stop_buy)),
            "target(2R)": (np.nan if np.isnan(target_buy) else float(target_buy)),
            "qty": int(qty),
            "vol_ratio": (np.nan if pd.isna(last.get("VOL_RATIO", np.nan)) else float(last["VOL_RATIO"])),
            "atr_pct(%)": (np.nan if pd.isna(last.get("ATR_PCT", np.nan)) else float(last["ATR_PCT"]) * 100),
            "ret_20(%)": (np.nan if pd.isna(last.get("RET_20", np.nan)) else float(last["RET_20"]) * 100),
            "date": str(df_ind.index[-1].date()),
            "error": ""
        }
        return result, df_ind, reason_df

    except Exception as e:
        result = {
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
            "error": str(e)
        }
        return result, None, None

# =========================
# positions 동기화 (run 버튼에서만)
# =========================
def sync_positions_with_analysis(df_analysis: pd.DataFrame):
    """
    run 버튼 눌렀을 때만 호출
    - ticker/market 갱신
    - 기존 entry_text/entry_date 유지
    - entry_price는 렌더링 때 entry_text로 재계산하므로 여기서 굳이 덮지 않음
    """
    cur = st.session_state["positions_df"].copy()

    base = df_analysis[["ticker", "market"]].copy()
    base["ticker"] = base["ticker"].astype(str).str.upper()
    base["market"] = base["market"].astype(str)

    merged = base.merge(cur, on=["ticker", "market"], how="left")

    if "entry_text" not in merged.columns:
        merged["entry_text"] = ""
    if "entry_date" not in merged.columns:
        merged["entry_date"] = ""
    if "entry_price" not in merged.columns:
        merged["entry_price"] = np.nan

    st.session_state["positions_df"] = merged

# =========================
# 엑셀 다운로드 (KR/US 표시는 Excel number_format으로)
# =========================
def build_excel_bytes_with_formats(df_all: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="Signals_All", index=False)

        df_cand = df_all[df_all.get("candidate", 0) == 1].copy()
        if df_cand.empty:
            df_cand = pd.DataFrame([{"note": "nottoday"}])
        df_cand.to_excel(writer, sheet_name="Candidates", index=False)

        wb = writer.book

        def apply_formats(ws):
            fmt_krw = u"₩#,##0"
            fmt_usd = u"$#,##0.00"
            price_cols = {
                "close", "stop", "target(2R)",
                "entry_price", "stop_by_entry", "target_by_entry(2R)"
            }

            header = {}
            for col in range(1, ws.max_column + 1):
                v = ws.cell(row=1, column=col).value
                if isinstance(v, str):
                    header[v.strip()] = col

            if "market" not in header:
                return
            market_col = header["market"]

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

    return output.getvalue()

# =========================
# 차트(Altair)
# =========================
def _reset_index_as_date(df_ind: pd.DataFrame) -> pd.DataFrame:
    d = df_ind.reset_index()
    if "Date" not in d.columns:
        d = d.rename(columns={d.columns[0]: "Date"})
    d["Date"] = pd.to_datetime(d["Date"])
    return d

def price_chart_with_lines(df_ind: pd.DataFrame, entry=None, stop=None, target=None):
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

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Swing Scanner", layout="wide")
st.markdown(
    """
    <h1 style="font-size:36px;font-weight:700;margin-bottom:10px;">
        웹 티커 입력 → 스윙 판단(O/X) + 근거(표+차트) + 매도 추천 + 엑셀
    </h1>
    """,
    unsafe_allow_html=True
)

# -------------------------
# session_state init
# -------------------------
if "analysis_df" not in st.session_state:
    st.session_state["analysis_df"] = None
if "analysis_detail" not in st.session_state:
    st.session_state["analysis_detail"] = {}

# ✅ 위젯 키와 분리된 실제 저장 DF
if "positions_df" not in st.session_state:
    st.session_state["positions_df"] = pd.DataFrame(
        columns=["ticker", "market", "entry_text", "entry_price", "entry_date"]
    )

if "ACCOUNT_SIZE" not in st.session_state:
    st.session_state["ACCOUNT_SIZE"] = DEFAULTS["ACCOUNT_SIZE"]

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("스윙 전략 설정")
    params = {}

    with st.expander("① 추세 판단 (이동평균)", expanded=True):
        params["MA_FAST"] = st.number_input("MA_FAST (단기 이동평균)", 5, 200, DEFAULTS["MA_FAST"], key="MA_FAST")
        st.write("단기 주가 흐름 기준. 보통 10~30일(기본 20).")
        params["MA_SLOW"] = st.number_input("MA_SLOW (장기 이동평균)", 10, 300, DEFAULTS["MA_SLOW"], key="MA_SLOW")
        st.write("중·장기 추세 기준. 보통 50~120일(기본 60).")

    with st.expander("② 거래량 · 변동성 조건", expanded=False):
        params["VOL_LOOKBACK"] = st.number_input("VOL_LOOKBACK (거래량 평균 기간)", 5, 200, DEFAULTS["VOL_LOOKBACK"], key="VOL_LOOKBACK")
        st.write("평균 거래량 계산 기간.")
        params["VOL_SPIKE"] = st.number_input("VOL_SPIKE (거래량 급증 기준)", 1.0, 5.0, float(DEFAULTS["VOL_SPIKE"]), step=0.05, key="VOL_SPIKE")
        st.write("예: 1.5 = 평균 대비 150% 이상 거래량이면 통과.")
        params["ATR_PERIOD"] = st.number_input("ATR_PERIOD (ATR 계산 기간)", 5, 100, DEFAULTS["ATR_PERIOD"], key="ATR_PERIOD")
        st.write("ATR은 평균 변동폭(변동성/손절 기준).")
        params["ATR_PCT_MIN"] = st.number_input("ATR_PCT_MIN (최소 변동성)", 0.0, 0.2, float(DEFAULTS["ATR_PCT_MIN"]), step=0.001, format="%.3f", key="ATR_PCT_MIN")
        st.write("너무 안 움직이는 종목 제외.")
        params["ATR_PCT_MAX"] = st.number_input("ATR_PCT_MAX (최대 변동성)", 0.0, 0.5, float(DEFAULTS["ATR_PCT_MAX"]), step=0.001, format="%.3f", key="ATR_PCT_MAX")
        st.write("너무 위험한 종목 제외.")

    with st.expander("③ 리스크 · 손절 · 보유 관리", expanded=False):
        params["STOP_ATR_MULT"] = st.number_input("STOP_ATR_MULT (손절 ATR 배수)", 0.5, 5.0, float(DEFAULTS["STOP_ATR_MULT"]), step=0.1, key="STOP_ATR_MULT")
        st.write("보통 1.5~2.0 많이 사용.")
        params["HOLD_DAYS"] = st.number_input("HOLD_DAYS (최대 보유일)", 1, 200, DEFAULTS["HOLD_DAYS"], key="HOLD_DAYS")
        st.write("보유기간이 길어지면 정리 기준.")

    with st.expander("④ 계좌 가정값 (천단위 콤마 입력)", expanded=False):
        acc_str = st.text_input(
            "ACCOUNT_SIZE (총 투자금)",
            value=f"{int(st.session_state['ACCOUNT_SIZE']):,}",
            key="ACCOUNT_SIZE_TEXT"
        )
        st.write("예: 10,000,000 처럼 콤마 입력 가능")
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
        st.write("예: 0.01 = 계좌의 1% 손실까지 허용.")

# -------------------------
# 입력/실행
# -------------------------
st.write("입력: KR은 6자리(예: 005930), US는 티커(예: SPY). 콤마/줄바꿈/공백 가능.")
raw = st.text_area("티커 입력", value="005930 000660\nSPY QQQ", height=120, key="ticker_input")
run = st.button("분석 실행", key="run_button")

if run:
    tickers = normalize_tickers(raw)
    if not tickers:
        st.warning("티커를 1개 이상 입력하세요.")
    else:
        results = []
        detail_map = {}

        for t in tickers:
            res, df_ind, reason_df = analyze_one_with_df(t, params)
            results.append(res)
            if df_ind is not None and reason_df is not None:
                detail_map[t.upper()] = (df_ind, reason_df)

        df = pd.DataFrame(results)
        df = df.sort_values(["candidate", "score"], ascending=[False, False]).reset_index(drop=True)

        st.session_state["analysis_df"] = df
        st.session_state["analysis_detail"] = detail_map

        # ✅ run 버튼에서만 positions 동기화(기존 입력 유지)
        sync_positions_with_analysis(df)

# -------------------------
# 결과 렌더링
# -------------------------
df_saved = st.session_state.get("analysis_df", None)
detail_saved = st.session_state.get("analysis_detail", {})

if df_saved is None or df_saved.empty:
    st.info("분석 실행을 눌러 결과를 생성하세요.")
    st.stop()

# =========================
# 보유 입력 (위젯 key 분리로 에러 해결)
# =========================
st.markdown("---")
st.subheader("보유 입력 (KR/US 통화 입력 지원, 리셋 방지)")

st.write(
    "- KR 예시: `₩10,000,000` 또는 `10,000,000` 또는 `10000000`\n"
    "- US 예시: `$123.45` 또는 `123.45` 또는 `1,234.56`\n"
    "입력은 `entry_text`에 하고, 계산용 평단은 자동으로 `entry_price`에 반영됩니다."
)

# 화면에 보여줄 때마다 entry_price는 entry_text 기준으로 재계산(텍스트는 건드리지 않음)
pos_for_view = compute_entry_price_column(st.session_state["positions_df"])

edited_positions = st.data_editor(
    pos_for_view,
    key="positions_editor_widget",   # ✅ 위젯 key는 이걸로 고정 (세션 대입 금지)
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "ticker": st.column_config.TextColumn("ticker", disabled=True),
        "market": st.column_config.TextColumn("market", disabled=True),
        "entry_text": st.column_config.TextColumn("평단 입력 (KR: ₩10,000,000 / US: $123.45)"),
        "entry_price": st.column_config.NumberColumn("평단(계산용 자동)", disabled=True, format="%.2f"),
        "entry_date": st.column_config.TextColumn("보유 시작일(YYYY-MM-DD)"),
    },
)

# ✅ 반환값을 positions_df에 저장 (위젯 key와 다른 키라서 안전)
st.session_state["positions_df"] = compute_entry_price_column(edited_positions)

# ticker+market -> 보유정보 매핑
pos_map = {}
for _, r in st.session_state["positions_df"].iterrows():
    pos_map[(str(r["ticker"]).upper(), str(r["market"]))] = {
        "entry_text": r.get("entry_text", ""),
        "entry_price": r.get("entry_price", np.nan),
        "entry_date": r.get("entry_date", ""),
    }

# =========================
# 매도추천 컬럼 추가
# =========================
df_out = df_saved.copy()

sell_signals, sell_reasons = [], []
stop_by_entry_list, target_by_entry_list, hold_days_list = [], [], []
entry_text_list, entry_price_list = [], []

for _, row in df_out.iterrows():
    tkr = str(row["ticker"]).upper()
    mkt = str(row["market"])
    key = (tkr, mkt)

    entry_text = pos_map.get(key, {}).get("entry_text", "")
    entry_price = pos_map.get(key, {}).get("entry_price", np.nan)
    entry_date = pos_map.get(key, {}).get("entry_date", "")

    entry_text_list.append(entry_text)
    entry_price_list.append(entry_price)

    if tkr not in detail_saved:
        sell_signals.append("N/A")
        sell_reasons.append("근거 데이터 없음")
        stop_by_entry_list.append(np.nan)
        target_by_entry_list.append(np.nan)
        hold_days_list.append(np.nan)
        continue

    df_ind, _reason_df = detail_saved[tkr]
    last = df_ind.iloc[-1]
    sig, reason, stp, tgt, hd = sell_recommendation(last, params, entry_price, entry_date)

    sell_signals.append(sig)
    sell_reasons.append(reason)
    stop_by_entry_list.append(np.nan if stp is None else float(stp))
    target_by_entry_list.append(np.nan if tgt is None else float(tgt))
    hold_days_list.append(np.nan if hd is None else int(hd))

df_out["entry_text"] = entry_text_list
df_out["entry_price"] = entry_price_list
df_out["sell_signal"] = sell_signals
df_out["sell_reason"] = sell_reasons
df_out["hold_days"] = hold_days_list
df_out["stop_by_entry"] = stop_by_entry_list
df_out["target_by_entry(2R)"] = target_by_entry_list

# =========================
# 결과 표
# =========================
st.markdown("---")
st.subheader("결과 (매수/매도 추천 포함)")

def highlight_sell_signal(val):
    if val == "SELL":
        return "background-color: rgba(255,0,0,0.15);"
    if val == "PARTIAL SELL":
        return "background-color: rgba(255,165,0,0.15);"
    if val == "HOLD":
        return "background-color: rgba(128,128,128,0.12);"
    return ""

df_view = df_out.copy()
for c in ["close", "stop", "target(2R)", "entry_price", "stop_by_entry", "target_by_entry(2R)"]:
    if c in df_view.columns:
        df_view[c] = df_view.apply(lambda r: format_currency_for_display(r.get("market", ""), r.get(c, None)), axis=1)

styled = df_view.style.applymap(highlight_sell_signal, subset=["sell_signal"])
st.dataframe(styled, use_container_width=True)

n_cand = int((df_out["candidate"] == 1).sum())
if n_cand == 0:
    st.error("NOT TODAY: 조건을 만족하는 후보가 없습니다. (O=0)")
else:
    st.success(f"후보(O) {n_cand}개")

# =========================
# 근거(표 + 차트)
# =========================
st.markdown("---")
st.subheader("근거(조건표 + 차트)")

for _, row in df_out.iterrows():
    tkr = str(row["ticker"]).upper()
    mkt = row.get("market", "")
    ox = row.get("O/X", "")
    sig = row.get("sell_signal", "")
    err = row.get("error", "")

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
        st.dataframe(reason_df, use_container_width=True)

        entry_price = row.get("entry_price", np.nan)
        entry_date = pos_map.get((tkr, str(mkt)), {}).get("entry_date", "")

        if entry_price is not None and not (isinstance(entry_price, float) and np.isnan(entry_price)) and entry_price > 0:
            last = df_ind.iloc[-1]
            _, _, stop_e, target_e, _ = sell_recommendation(last, params, float(entry_price), entry_date)
        else:
            stop_e, target_e = None, None

        st.write("가격 차트 (Close + MA + 평단/손절/목표 수평선)")
        st.altair_chart(price_chart_with_lines(df_ind, entry=entry_price, stop=stop_e, target=target_e), use_container_width=True)

        st.write("거래량 차트 (Volume + 평균)")
        st.altair_chart(volume_chart(df_ind), use_container_width=True)

        st.write(
            "요약\n"
            f"- close: {format_currency_for_display(mkt, row.get('close'))}\n"
            f"- (매수기준) stop: {format_currency_for_display(mkt, row.get('stop'))} / target(2R): {format_currency_for_display(mkt, row.get('target(2R)'))}\n"
            f"- (평단기준) stop: {format_currency_for_display(mkt, row.get('stop_by_entry'))} / target(2R): {format_currency_for_display(mkt, row.get('target_by_entry(2R)'))}\n"
            f"- sell_reason: {row.get('sell_reason','')}"
        )

# =========================
# 엑셀 다운로드
# =========================
st.markdown("---")
st.subheader("엑셀 다운로드")

xlsx_bytes = build_excel_bytes_with_formats(df_out)
st.download_button(
    label="엑셀 다운로드 (KR ₩ / US $ 자동 적용)",
    data=xlsx_bytes,
    file_name="Swing_Scanner_Output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
