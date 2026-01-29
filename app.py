# app.py
import re
import io
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import streamlit as st

import yfinance as yf
from pykrx import stock as krx

# -----------------------------
# Defaults
# -----------------------------
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
    TOP_N=10,
)

KR_SAMPLE = [
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "035420",  # NAVER
    "035720",  # 카카오
    "051910",  # LG화학
    "005380",  # 현대차
    "068270",  # 셀트리온
    "207940",  # 삼성바이오로직스
    "006400",  # 삼성SDI
    "012450",  # 한화에어로스페이스
]

US_SAMPLE = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AMD",
]


# -----------------------------
# Helpers
# -----------------------------
def is_kr_code(x: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", str(x).strip()))


def normalize_tickers(raw: str):
    items = re.split(r"[,\n\s]+", str(raw).strip())
    items = [x.strip().upper() for x in items if x.strip()]
    return items


def safe_float(v):
    try:
        if v is None:
            return None
        if isinstance(v, str):
            v = v.replace(",", "").strip()
            if v == "":
                return None
        return float(v)
    except Exception:
        return None


def to_date(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, date):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                pass
    return None


def format_money(mkt, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    try:
        v = float(v)
    except Exception:
        return ""
    if mkt == "KR":
        return f"{v:,.0f}원"
    if mkt == "US":
        return f"${v:,.2f}"
    return f"{v:,.2f}"


# -----------------------------
# Name lookup (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def get_kr_name(code: str) -> str:
    try:
        return krx.get_market_ticker_name(code)
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def get_us_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).fast_info  # fast_info는 이름이 없을 수 있음
    except Exception:
        info = None

    # 이름은 info에 없는 경우가 많아서 .info를 시도하되 실패하면 빈값
    try:
        inf = yf.Ticker(ticker).info
        name = inf.get("shortName") or inf.get("longName") or ""
        return str(name)[:80]
    except Exception:
        return ""


def market_of_ticker(t: str) -> str:
    return "KR" if is_kr_code(t) else "US"


# -----------------------------
# Indicators
# -----------------------------
def compute_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
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
    df["RET_20"] = df["Close"].pct_change(20)
    return df


def rule_signal(last, VOL_SPIKE, ATR_PCT_MIN, ATR_PCT_MAX):
    needed = ["MA_FAST", "MA_SLOW", "VOL_RATIO", "ATR_PCT"]
    if any(pd.isna(last.get(k, np.nan)) for k in needed):
        return 0

    trend_ok = (last["MA_FAST"] > last["MA_SLOW"]) and (last["Close"] > last["MA_FAST"])
    vol_ok = last["VOL_RATIO"] >= VOL_SPIKE
    atr_ok = (ATR_PCT_MIN <= last["ATR_PCT"] <= ATR_PCT_MAX)
    return 1 if (trend_ok and vol_ok and atr_ok) else 0


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


def position_size(entry, atr, ACCOUNT_SIZE, RISK_PER_TRADE, STOP_ATR_MULT):
    if atr is None or pd.isna(atr) or atr <= 0:
        return 0, np.nan
    stop_dist = STOP_ATR_MULT * atr
    risk_amount = ACCOUNT_SIZE * RISK_PER_TRADE
    qty = int(risk_amount / stop_dist)
    stop_price = entry - stop_dist
    return max(0, qty), float(stop_price)


def build_reason_table(last, params):
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
    rows.append(
        dict(
            조건="추세: MA_FAST > MA_SLOW",
            현재값=f"{ma_fast:.2f} > {ma_slow:.2f}" if (ma_fast is not None and ma_slow is not None) else "데이터 부족",
            기준="단기선이 장기선 위",
            통과=bool(c1),
        )
    )
    rows.append(
        dict(
            조건="추세: Close > MA_FAST",
            현재값=f"{close:.2f} > {ma_fast:.2f}" if (close is not None and ma_fast is not None) else "데이터 부족",
            기준="종가가 단기선 위",
            통과=bool(c2),
        )
    )

    c3 = (vol_ratio is not None) and (vol_ratio >= float(params["VOL_SPIKE"]))
    rows.append(
        dict(
            조건="거래량: VOL_RATIO >= VOL_SPIKE",
            현재값=f"{vol_ratio:.2f}" if vol_ratio is not None else "데이터 부족",
            기준=f">= {float(params['VOL_SPIKE']):.2f}",
            통과=bool(c3),
        )
    )

    atr_min = float(params["ATR_PCT_MIN"])
    atr_max = float(params["ATR_PCT_MAX"])
    c4 = (atr_pct is not None) and (atr_min <= atr_pct <= atr_max)
    rows.append(
        dict(
            조건="변동성: ATR_PCT_MIN <= ATR_PCT <= ATR_PCT_MAX",
            현재값=f"{atr_pct*100:.2f}%" if atr_pct is not None else "데이터 부족",
            기준=f"{atr_min*100:.2f}% ~ {atr_max*100:.2f}%",
            통과=bool(c4),
        )
    )

    return pd.DataFrame(rows)


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_us(ticker: str) -> pd.DataFrame:
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

        if ticker in lv0:
            df = df[ticker]
        elif ticker in lv1:
            df = df.xs(ticker, axis=1, level=1)
        else:
            # 마지막 방어
            uniq0 = list(pd.unique(lv0))
            uniq1 = list(pd.unique(lv1))
            if len(uniq0) > 0 and len(uniq1) > 0:
                # (필드, 티커)로 가정하고 첫 티커
                df = df.xs(uniq1[0], axis=1, level=1)
            else:
                raise ValueError(f"Unexpected MultiIndex columns: {df.columns}")

    # columns가 문자열이 아닐 수 있어 title() 안전 처리
    df.columns = [str(c).title() for c in df.columns]

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    keep = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"US data missing columns: {missing} / columns={list(df.columns)}")

    df = df[keep].dropna()
    return df


@st.cache_data(show_spinner=False)
def load_kr(code: str) -> pd.DataFrame:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
    df = krx.get_market_ohlcv_by_date(start, end, code)
    df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# -----------------------------
# Analysis per ticker
# -----------------------------
def analyze_one(ticker: str, params: dict):
    try:
        ticker = str(ticker).strip().upper()
        if not ticker:
            raise ValueError("Empty ticker")

        if is_kr_code(ticker):
            market = "KR"
            name = get_kr_name(ticker)
            base = load_kr(ticker)
        else:
            market = "US"
            name = get_us_name(ticker)
            base = load_us(ticker)

        df = add_indicators(
            base,
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
        atr = float(last["ATR"]) if pd.notna(last.get("ATR", np.nan)) else np.nan
        qty, stop = position_size(
            entry,
            atr,
            ACCOUNT_SIZE=params["ACCOUNT_SIZE"],
            RISK_PER_TRADE=params["RISK_PER_TRADE"],
            STOP_ATR_MULT=params["STOP_ATR_MULT"],
        )
        target = entry + 2 * (entry - stop) if (not np.isnan(stop)) else np.nan

        return {
            "market": market,
            "ticker": ticker,
            "name": name,
            "O/X": "O" if cand == 1 else "X",
            "candidate": int(cand),
            "score": int(sc),
            "close": entry,
            "stop": (None if np.isnan(stop) else float(stop)),
            "target(2R)": (None if np.isnan(target) else float(target)),
            "qty": int(qty),
            "vol_ratio": (None if pd.isna(last.get("VOL_RATIO", np.nan)) else float(last["VOL_RATIO"])),
            "atr_pct(%)": (None if pd.isna(last.get("ATR_PCT", np.nan)) else float(last["ATR_PCT"]) * 100.0),
            "ret_20(%)": (None if pd.isna(last.get("RET_20", np.nan)) else float(last["RET_20"]) * 100.0),
            "date": str(df.index[-1].date()),
            "error": "",
            "_df": df,  # 내부용
            "_last": last,  # 내부용
        }
    except Exception as e:
        return {
            "market": "?",
            "ticker": str(ticker),
            "name": "",
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
            "error": str(e),
            "_df": None,
            "_last": None,
        }


# -----------------------------
# Sell recommendation (simple, rule-based)
# -----------------------------
def sell_recommend(pos_row: dict, last: pd.Series, params: dict):
    """
    return (action, reason)
    action: HOLD / SELL / WATCH
    """
    mkt = pos_row.get("market", "")
    qty = safe_float(pos_row.get("qty", 0)) or 0
    avg = safe_float(pos_row.get("avg_price", 0)) or 0
    bd = to_date(pos_row.get("buy_date"))
    hold_days = (date.today() - bd).days if bd else None

    close = safe_float(last.get("Close", np.nan))
    ma_fast = safe_float(last.get("MA_FAST", np.nan))
    ma_slow = safe_float(last.get("MA_SLOW", np.nan))
    atr = safe_float(last.get("ATR", np.nan))

    if close is None or ma_fast is None or ma_slow is None:
        return "WATCH", "지표 데이터 부족"

    # 1) 손절: 종가가 (평단 - STOP_ATR_MULT*ATR) 아래
    if avg > 0 and atr is not None and atr > 0:
        stop_calc = avg - float(params["STOP_ATR_MULT"]) * atr
        if close < stop_calc:
            return "SELL", f"손절: 종가<{format_money(mkt, stop_calc)}"

    # 2) 추세 이탈: 종가가 MA_FAST 아래로 내려옴
    if close < ma_fast:
        return "SELL", "추세 이탈: Close < MA_FAST"

    # 3) 보유일 초과
    if hold_days is not None and hold_days >= int(params["HOLD_DAYS"]):
        return "SELL", f"보유일 초과: {hold_days}일"

    return "HOLD", "조건 미충족(유지)"


# -----------------------------
# Excel builder (KR/USD format)
# -----------------------------
def build_excel(df_all: pd.DataFrame, positions_df: pd.DataFrame) -> bytes:
    from openpyxl.styles import Font, Alignment
    from openpyxl.utils import get_column_letter

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # --- Signals_All
        df1 = df_all.copy()
        df1.to_excel(writer, sheet_name="Signals_All", index=False)

        # --- Candidates
        df_cand = df1[df1["candidate"] == 1].copy()
        if df_cand.empty:
            df_cand = pd.DataFrame([{"note": "NOT TODAY"}])
        df_cand.to_excel(writer, sheet_name="Candidates", index=False)

        # --- Order sheets
        def order_sheet(df, market):
            d = df[(df["market"] == market) & (df["candidate"] == 1)].copy()
            if d.empty:
                return pd.DataFrame(
                    [
                        {
                            "market": market,
                            "ticker": "nottoday",
                            "name": "",
                            "action": "NONE",
                            "qty": 0,
                            "stop": "",
                            "target(2R)": "",
                            "note": "No candidates",
                        }
                    ]
                )
            d = d.sort_values("score", ascending=False).head(10)
            d["qty_safe"] = (d["qty"] * 0.5).astype(int)
            return pd.DataFrame(
                {
                    "market": d["market"],
                    "ticker": d["ticker"],
                    "name": d.get("name", ""),
                    "action": "BUY",
                    "qty": d["qty_safe"],
                    "stop": d["stop"],
                    "target(2R)": d["target(2R)"],
                    "note": ["Manual entry (M-STOCK)"] * len(d),
                }
            )

        order_sheet(df1, "KR").to_excel(writer, sheet_name="Order_KR", index=False)
        order_sheet(df1, "US").to_excel(writer, sheet_name="Order_US", index=False)

        # --- Positions (if any)
        if positions_df is None or positions_df.empty:
            pd.DataFrame([{"note": "No positions"}]).to_excel(writer, sheet_name="Positions", index=False)
        else:
            positions_df.to_excel(writer, sheet_name="Positions", index=False)

        # ----------------
        # Formatting
        # ----------------
        wb = writer.book

        def autosize(ws):
            for col in ws.columns:
                max_len = 0
                col_letter = get_column_letter(col[0].column)
                for cell in col:
                    try:
                        v = "" if cell.value is None else str(cell.value)
                        max_len = max(max_len, len(v))
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(max(10, max_len + 2), 45)

            # header style
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center")

        def apply_currency_formats(ws, market_col_name="market"):
            # 숫자 컬럼들 후보
            num_cols = {
                "close": ("KR", "#,##0", "US", "$#,##0.00"),
                "stop": ("KR", "#,##0", "US", "$#,##0.00"),
                "target(2R)": ("KR", "#,##0", "US", "$#,##0.00"),
                "mkt_value": ("KR", "#,##0", "US", "$#,##0.00"),
                "pnl": ("KR", "#,##0", "US", "$#,##0.00"),
                "avg_price": ("KR", "#,##0", "US", "$#,##0.00"),
                "last_close": ("KR", "#,##0", "US", "$#,##0.00"),
                "stop_calc": ("KR", "#,##0", "US", "$#,##0.00"),
            }

            headers = [c.value for c in ws[1]]
            if market_col_name not in headers:
                return

            mkt_idx = headers.index(market_col_name) + 1
            col_idx = {h: i + 1 for i, h in enumerate(headers)}

            for r in range(2, ws.max_row + 1):
                mkt = ws.cell(row=r, column=mkt_idx).value
                mkt = str(mkt).strip().upper()

                for k, (m1, f1, m2, f2) in num_cols.items():
                    if k in col_idx:
                        c = ws.cell(row=r, column=col_idx[k])
                        if isinstance(c.value, (int, float)) and c.value is not None:
                            c.number_format = f1 if mkt == "KR" else f2

        for name in wb.sheetnames:
            ws = wb[name]
            autosize(ws)
            if name in ("Signals_All", "Candidates", "Order_KR", "Order_US", "Positions"):
                apply_currency_formats(ws)

    return output.getvalue()


# -----------------------------
# UI Styling (font sizes)
# -----------------------------
def apply_css():
    st.markdown(
        """
<style>
/* 전체 기본 폰트 크기 */
html, body, [class*="css"]  { font-size: 15px; }

/* 메인 타이틀 크기 */
h1 { font-size: 30px !important; line-height: 1.2 !important; }

/* 서브헤더 */
h2 { font-size: 22px !important; }
h3 { font-size: 18px !important; }

/* 표 가로 시인성 */
[data-testid="stDataFrame"] div { font-size: 14px; }
</style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Session State init
# -----------------------------
def init_state():
    if "ticker_input" not in st.session_state:
        st.session_state["ticker_input"] = " ".join(KR_SAMPLE[:2] + US_SAMPLE[:2])

    if "last_scan_df" not in st.session_state:
        st.session_state["last_scan_df"] = pd.DataFrame()

    if "scan_detail" not in st.session_state:
        st.session_state["scan_detail"] = {}  # ticker -> {"df":..., "last":...}

    if "positions_df" not in st.session_state:
        st.session_state["positions_df"] = pd.DataFrame(
            [
                {
                    "market": "KR",
                    "ticker": "",
                    "name": "",
                    "qty": 0,
                    "avg_price": "",
                    "buy_date": "",
                }
            ]
        )


# -----------------------------
# Main
# -----------------------------
st.set_page_config(page_title="Swing Scanner", layout="wide")
apply_css()
init_state()

st.title("웹 티커 입력 → 스윙 판단(O/X) + 근거(표+차트) + 주문표 엑셀")

# -----------------------------
# Sidebar: grouped categories + explanations
# -----------------------------
with st.sidebar:
    st.header("스윙 전략 설정")

    # 계좌값: 천단위 콤마 입력을 위해 text_input 사용
    st.markdown("### 자금 관리")
    acct_default = f"{DEFAULTS['ACCOUNT_SIZE']:,}"
    acct_raw = st.text_input("ACCOUNT_SIZE (총 투자금) - 콤마 입력 가능", value=st.session_state.get("ACCOUNT_SIZE_RAW", acct_default), key="ACCOUNT_SIZE_RAW")
    acct_val = safe_float(acct_raw)
    if acct_val is None or acct_val <= 0:
        acct_val = float(DEFAULTS["ACCOUNT_SIZE"])
        st.caption("계좌 금액이 올바르지 않아 기본값으로 적용됩니다.")
    st.write(f"적용값: {acct_val:,.0f}원")

    risk = st.number_input("RISK_PER_TRADE (1회 최대 손실 비율)", 0.001, 0.05, float(DEFAULTS["RISK_PER_TRADE"]), step=0.001, format="%.3f", key="RISK_PER_TRADE")
    stop_mult = st.number_input("STOP_ATR_MULT (손절 ATR 배수)", 0.5, 5.0, float(DEFAULTS["STOP_ATR_MULT"]), step=0.1, key="STOP_ATR_MULT")
    hold_days = st.number_input("HOLD_DAYS (최대 보유일)", 1, 200, DEFAULTS["HOLD_DAYS"], key="HOLD_DAYS")

    st.markdown("---")
    st.markdown("### 추세 지표 (이동평균)")
    ma_fast = st.number_input("MA_FAST (단기 이동평균)", 5, 200, DEFAULTS["MA_FAST"], key="MA_FAST")
    st.write(
        "최근 주가의 단기 흐름을 보는 이동평균 기간입니다.\n"
        "- 작을수록 신호가 빠르지만 흔들림(휩쏘) 증가\n"
        "- 클수록 신호가 느리지만 안정적\n"
        "기본값 20은 무난한 설정입니다."
    )
    ma_slow = st.number_input("MA_SLOW (장기 이동평균)", 10, 300, DEFAULTS["MA_SLOW"], key="MA_SLOW")
    st.write(
        "중·장기 추세 기준 이동평균입니다.\n"
        "MA_FAST가 MA_SLOW 위면 상승 추세로 봅니다.\n"
        "기본값 60은 완만한 추세 기준입니다."
    )

    st.markdown("---")
    st.markdown("### 거래량 / 변동성 조건")
    vol_lb = st.number_input("VOL_LOOKBACK (거래량 평균 기간)", 5, 200, DEFAULTS["VOL_LOOKBACK"], key="VOL_LOOKBACK")
    st.write("현재 거래량이 최근 평균 대비 얼마나 커졌는지(VOL_RATIO) 계산하는 기간입니다.")

    vol_spike = st.number_input("VOL_SPIKE (거래량 급증 기준)", 1.0, 5.0, float(DEFAULTS["VOL_SPIKE"]), step=0.05, key="VOL_SPIKE")
    st.write("예: 1.5는 평균 대비 150% 이상 거래량이면 통과.")

    atr_period = st.number_input("ATR_PERIOD (ATR 계산 기간)", 5, 100, DEFAULTS["ATR_PERIOD"], key="ATR_PERIOD")
    st.write("ATR은 평균 변동폭입니다. 값이 클수록 변동이 큰 종목입니다.")

    atr_min = st.number_input("ATR_PCT_MIN (최소 변동성)", 0.0, 0.2, float(DEFAULTS["ATR_PCT_MIN"]), step=0.001, format="%.3f", key="ATR_PCT_MIN")
    atr_max = st.number_input("ATR_PCT_MAX (최대 변동성)", 0.0, 0.5, float(DEFAULTS["ATR_PCT_MAX"]), step=0.001, format="%.3f", key="ATR_PCT_MAX")
    st.write("너무 안 움직이거나(최소 미달), 너무 위험하게 크면(최대 초과) 제외합니다.")

params = dict(
    HOLD_DAYS=int(hold_days),
    MA_FAST=int(ma_fast),
    MA_SLOW=int(ma_slow),
    ATR_PERIOD=int(atr_period),
    VOL_LOOKBACK=int(vol_lb),
    VOL_SPIKE=float(vol_spike),
    ATR_PCT_MIN=float(atr_min),
    ATR_PCT_MAX=float(atr_max),
    ACCOUNT_SIZE=float(acct_val),
    RISK_PER_TRADE=float(risk),
    STOP_ATR_MULT=float(stop_mult),
    TOP_N=int(DEFAULTS["TOP_N"]),
)

# -----------------------------
# Recommendation controls
# -----------------------------
st.subheader("추천 티커 자동 입력")
colA, colB, colC, colD = st.columns([1.2, 1, 1, 2])

with colA:
    rec_mode = st.selectbox("추천 옵션", ["KR(한국) 10개", "US(미국) 10개", "균형(KR5+US5)"], index=2)

with colB:
    if st.button("추천 실행"):
        if rec_mode.startswith("KR"):
            rec = KR_SAMPLE[:10]
        elif rec_mode.startswith("US"):
            rec = US_SAMPLE[:10]
        else:
            rec = KR_SAMPLE[:5] + US_SAMPLE[:5]
        st.session_state["ticker_input"] = " ".join(rec)
        st.rerun()

with colC:
    if st.button("입력 지우기"):
        st.session_state["ticker_input"] = ""
        st.rerun()

with colD:
    st.write("추천 실행 후 **아래 티커 입력창에 자동 반영**됩니다.")

# -----------------------------
# Ticker input + run
# -----------------------------
st.write("입력: KR은 6자리(예: 005930), US는 티커(예: SPY). 공백/콤마/줄바꿈 모두 가능.")
raw = st.text_area("티커 입력", value=st.session_state["ticker_input"], height=110, key="ticker_input_area")

col_run, col_opt = st.columns([1, 2])
with col_run:
    run_scan = st.button("분석 실행", use_container_width=True)
with col_opt:
    st.caption("팁: 추천 실행 → 자동입력 → 분석 실행 흐름으로 쓰면 편합니다.")

# -----------------------------
# Scan
# -----------------------------
df_map = {}  # ticker -> df_ind
last_map = {}  # ticker -> last row

if run_scan:
    tickers = normalize_tickers(raw)
    if not tickers:
        st.warning("티커를 1개 이상 입력하세요.")
        st.stop()

    rows = []
    with st.spinner("데이터 로딩/계산 중..."):
        for t in tickers:
            r = analyze_one(t, params)
            rows.append({k: v for k, v in r.items() if not k.startswith("_")})
            if r.get("_df") is not None and r.get("_last") is not None:
                df_map[r["ticker"]] = r["_df"]
                last_map[r["ticker"]] = r["_last"]

    df_all = pd.DataFrame(rows)
    df_all = df_all.sort_values(["candidate", "score"], ascending=[False, False]).reset_index(drop=True)

    st.session_state["last_scan_df"] = df_all
    st.session_state["scan_detail"] = {"df_map": df_map, "last_map": last_map}

# Reuse last scan
df_all = st.session_state.get("last_scan_df", pd.DataFrame())
detail = st.session_state.get("scan_detail", {})
df_map = detail.get("df_map", {}) if isinstance(detail, dict) else {}
last_map = detail.get("last_map", {}) if isinstance(detail, dict) else {}

# -----------------------------
# Results table (wide, company name included)
# -----------------------------
st.subheader("추천 스캔 보기 (가로 가시성 강화)")

if df_all is None or df_all.empty:
    st.info("아직 분석 결과가 없습니다. 위에서 티커를 넣고 '분석 실행'을 눌러주세요.")
else:
    show = df_all.copy()

    # 표시용 통화(표시만)
    show["close_disp"] = show.apply(lambda r: format_money(r["market"], r["close"]), axis=1)
    show["stop_disp"] = show.apply(lambda r: format_money(r["market"], r["stop"]), axis=1)
    show["target_disp"] = show.apply(lambda r: format_money(r["market"], r["target(2R)"]), axis=1)

    cols = [
        "market",
        "ticker",
        "name",
        "O/X",
        "score",
        "close_disp",
        "stop_disp",
        "target_disp",
        "qty",
        "vol_ratio",
        "atr_pct(%)",
        "ret_20(%)",
        "date",
        "error",
    ]
    cols = [c for c in cols if c in show.columns]

    st.dataframe(
        show[cols],
        use_container_width=True,
        height=320,
        column_config={
            "name": st.column_config.TextColumn("회사명", width="large"),
            "ticker": st.column_config.TextColumn("티커", width="small"),
            "error": st.column_config.TextColumn("오류", width="large"),
        },
    )

    n_cand = int((show["candidate"] == 1).sum()) if "candidate" in show.columns else 0
    if n_cand > 0:
        st.success(f"후보(O) {n_cand}개")
    else:
        st.error("NOT TODAY: 조건을 만족하는 후보가 없습니다. (O=0)")

# -----------------------------
# Evidence (reason table + chart)
# -----------------------------
st.subheader("근거 보기 (표 + 차트)")
if df_all is None or df_all.empty:
    st.info("분석 결과가 있어야 근거를 볼 수 있습니다.")
else:
    cand_tickers = df_all["ticker"].tolist()
    pick = st.selectbox("티커 선택", cand_tickers, index=0)

    df_ind = df_map.get(pick)
    last = last_map.get(pick)

    if df_ind is None or last is None:
        st.warning("해당 티커의 근거 데이터를 찾지 못했습니다. (캐시/로드 문제일 수 있음)")
    else:
        reason_df = build_reason_table(last, params)
        st.dataframe(reason_df, use_container_width=True)

        # 차트: Close + MA
        chart = df_ind[["Close", "MA_FAST", "MA_SLOW"]].dropna().tail(140)
        st.line_chart(chart)

# -----------------------------
# Positions editor (no reset, no double input) + valuation fills into table
# -----------------------------
st.subheader("보유 포지션 입력 (리셋 방지)")

# 편집은 form 안에서 수행 -> 입력 중 리런으로 튀는 현상 완화
with st.form("positions_form", clear_on_submit=False):
    st.write("시장(KR/US), 티커, 수량, 평단, 매수일을 입력하세요. (매수일: YYYY-MM-DD 권장)")
    edited = st.data_editor(
        st.session_state["positions_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="positions_editor",  # returned df로 저장
        column_config={
            "market": st.column_config.SelectboxColumn("시장", options=["KR", "US"], required=True, width="small"),
            "ticker": st.column_config.TextColumn("티커", width="small"),
            "name": st.column_config.TextColumn("회사명(자동)", width="large"),
            "qty": st.column_config.NumberColumn("수량", min_value=0, step=1, width="small"),
            "avg_price": st.column_config.TextColumn("평단(콤마 가능)", width="small"),
            "buy_date": st.column_config.TextColumn("매수일", width="small"),
        },
    )
    saved = st.form_submit_button("보유 저장/반영")

if saved:
    # 저장 시에만 세션에 반영 -> 입력 중 2번쳐야 들어가는 현상 줄임
    dfp = edited.copy()

    # 정규화/회사명 채우기
    dfp["market"] = dfp.get("market", "").astype(str).str.strip().str.upper()
    dfp["ticker"] = dfp.get("ticker", "").astype(str).str.strip().str.upper()

    def fill_auto_name(row):
        m = row.get("market", "")
        t = str(row.get("ticker", "")).strip().upper()
        if not t:
            return ""
        if m == "KR" and is_kr_code(t):
            return get_kr_name(t)
        if m == "US":
            return get_us_name(t)
        return row.get("name", "")

    dfp["name"] = dfp.apply(fill_auto_name, axis=1)

    # avg_price를 숫자로 파싱해 별도 컬럼 저장(원본 문자열은 유지)
    dfp["avg_price_num"] = dfp["avg_price"].apply(safe_float)

    st.session_state["positions_df"] = dfp
    st.success("보유 포지션이 저장되었습니다.")

# -----------------------------
# Position valuation + sell recommendation (✅ 계산값을 표에 '기입')
# -----------------------------
st.subheader("보유 포지션 평가/매도 추천(자동 계산)")
pos_df = st.session_state["positions_df"].copy()

if pos_df is None or pos_df.empty:
    st.info("보유 포지션이 없습니다.")
else:
    # 스캔 상세 맵
    df_map = detail.get("df_map", {}) if isinstance(detail, dict) else {}
    last_map = detail.get("last_map", {}) if isinstance(detail, dict) else {}

    last_close = []
    stop_calc = []
    mkt_value = []
    pnl = []
    pnl_pct = []
    hold_days_list = []
    action_list = []
    reason_list = []

    for _, pr in pos_df.iterrows():
        m = str(pr.get("market", "")).strip().upper()
        t = str(pr.get("ticker", "")).strip().upper()
        qty = safe_float(pr.get("qty", 0)) or 0
        avgp = safe_float(pr.get("avg_price_num", pr.get("avg_price", 0))) or 0
        bd = to_date(pr.get("buy_date"))

        if not t:
            last_close.append(np.nan)
            stop_calc.append(np.nan)
            mkt_value.append(np.nan)
            pnl.append(np.nan)
            pnl_pct.append(np.nan)
            hold_days_list.append("")
            action_list.append("")
            reason_list.append("")
            continue

        df_ind = df_map.get(t)
        last = last_map.get(t)

        if df_ind is None or last is None:
            # 스캔에 없으면 개별 로드
            try:
                base = load_kr(t) if (m == "KR" and is_kr_code(t)) else load_us(t)
                df_ind = add_indicators(
                    base,
                    MA_FAST=params["MA_FAST"],
                    MA_SLOW=params["MA_SLOW"],
                    VOL_LOOKBACK=params["VOL_LOOKBACK"],
                    ATR_PERIOD=params["ATR_PERIOD"],
                )
                last = df_ind.iloc[-1]
            except Exception:
                df_ind = None
                last = None

        if df_ind is None or last is None or df_ind.empty:
            last_close.append(np.nan)
            stop_calc.append(np.nan)
            mkt_value.append(np.nan)
            pnl.append(np.nan)
            pnl_pct.append(np.nan)
            hold_days_list.append("")
            action_list.append("WATCH")
            reason_list.append("데이터 로드 실패")
            continue

        c = safe_float(last.get("Close", np.nan))
        atr = safe_float(last.get("ATR", np.nan))

        # 계산 손절(평단 기준)
        if avgp > 0 and atr is not None and atr > 0:
            stop_p = avgp - float(params["STOP_ATR_MULT"]) * atr
        else:
            stop_p = np.nan

        if c is not None:
            mv = qty * c
            p = qty * (c - avgp) if avgp > 0 else np.nan
            pp = ((c - avgp) / avgp * 100.0) if avgp > 0 else np.nan
        else:
            mv = np.nan
            p = np.nan
            pp = np.nan

        hd = (date.today() - bd).days if bd else None

        tmp_row = dict(pr)
        tmp_row["stop_calc"] = stop_p
        act, rsn = sell_recommend(tmp_row, last, params)

        last_close.append(c if c is not None else np.nan)
        stop_calc.append(stop_p)
        mkt_value.append(mv)
        pnl.append(p)
        pnl_pct.append(pp)
        hold_days_list.append(hd if hd is not None else "")
        action_list.append(act)
        reason_list.append(rsn)

    # ✅ 계산값을 표에 '기입' + 세션에 반영
    pos_df["last_close"] = last_close
    pos_df["stop_calc"] = stop_calc
    pos_df["mkt_value"] = mkt_value
    pos_df["pnl"] = pnl
    pos_df["pnl_pct(%)"] = pnl_pct
    pos_df["hold_days"] = hold_days_list
    pos_df["action"] = action_list
    pos_df["action_reason"] = reason_list
    st.session_state["positions_df"] = pos_df.copy()

    showp = pos_df.copy()
    showp["last_close_disp"] = showp.apply(lambda r: format_money(r["market"], r["last_close"]), axis=1)
    showp["stop_calc_disp"] = showp.apply(lambda r: format_money(r["market"], r["stop_calc"]), axis=1)
    showp["mkt_value_disp"] = showp.apply(lambda r: format_money(r["market"], r["mkt_value"]), axis=1)
    showp["pnl_disp"] = showp.apply(lambda r: format_money(r["market"], r["pnl"]), axis=1)

    view_cols = [
        "market",
        "ticker",
        "name",
        "qty",
        "avg_price",
        "buy_date",
        "last_close_disp",
        "stop_calc_disp",
        "mkt_value_disp",
        "pnl_disp",
        "pnl_pct(%)",
        "hold_days",
        "action",
        "action_reason",
    ]
    view_cols = [c for c in view_cols if c in showp.columns]
    st.dataframe(
        showp[view_cols],
        use_container_width=True,
        column_config={"name": st.column_config.TextColumn("회사명", width="large")},
        height=280,
    )

# -----------------------------
# Excel download (includes currency formats + all sheets)
# -----------------------------
st.subheader("엑셀 다운로드 (통화/천단위 포함)")
if df_all is None or df_all.empty:
    st.info("분석 결과가 있어야 엑셀을 생성할 수 있습니다.")
else:
    xlsx_bytes = build_excel(df_all, st.session_state["positions_df"])
    st.download_button(
        label="엑셀 1개로 다운로드 (All-in-One)",
        data=xlsx_bytes,
        file_name="Swing_Output_AllInOne.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# -----------------------------
# Next steps (3)
# -----------------------------
st.subheader("다음 단계 (3가지)")
st.write(
    "1) **티커 추천 → 분석 실행**으로 후보를 뽑고, 후보의 근거(표+차트)를 확인하세요.\n"
    "2) **보유 포지션**에 수량/평단/매수일을 입력해 자동 평가/매도 추천을 확인하세요.\n"
    "3) 결과가 마음에 들면 **엑셀 다운로드**로 주문표/후보/전체/포지션까지 한 번에 저장하세요."
)
