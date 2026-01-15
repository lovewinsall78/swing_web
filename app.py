import re
import io
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock as krx
import streamlit as st
from datetime import datetime, timedelta

# =========================
# Defaults
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
# Utils
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
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df: pd.DataFrame, MA_FAST: int, MA_SLOW: int, VOL_LOOKBACK: int, ATR_PERIOD: int) -> pd.DataFrame:
    df = df.copy()
    df["MA_FAST"] = df["Close"].rolling(MA_FAST).mean()
    df["MA_SLOW"] = df["Close"].rolling(MA_SLOW).mean()
    df["VOL_AVG"] = df["Volume"].rolling(VOL_LOOKBACK).mean()
    df["VOL_RATIO"] = np.where(df["VOL_AVG"] > 0, df["Volume"] / df["VOL_AVG"], np.nan)
    df["ATR"] = compute_atr(df, ATR_PERIOD)
    df["ATR_PCT"] = df["ATR"] / df["Close"]
    df["RET_20"] = df["Close"].pct_change(20)
    return df

def rule_signal(last: pd.Series, params: dict) -> int:
    needed = ["MA_FAST", "MA_SLOW", "VOL_RATIO", "ATR_PCT", "Close"]
    if any(pd.isna(last.get(k, np.nan)) for k in needed):
        return 0
    trend_ok = (last["MA_FAST"] > last["MA_SLOW"]) and (last["Close"] > last["MA_FAST"])
    vol_ok = (last["VOL_RATIO"] >= float(params["VOL_SPIKE"]))
    atr_ok = (float(params["ATR_PCT_MIN"]) <= last["ATR_PCT"] <= float(params["ATR_PCT_MAX"]))
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

def position_size(entry: float, atr: float, ACCOUNT_SIZE: float, RISK_PER_TRADE: float, STOP_ATR_MULT: float):
    if pd.isna(atr) or atr <= 0:
        return 0, np.nan
    stop_dist = STOP_ATR_MULT * atr
    risk_amount = ACCOUNT_SIZE * RISK_PER_TRADE
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

    vol_spike = float(params["VOL_SPIKE"])
    c3 = (vol_ratio is not None) and (vol_ratio >= vol_spike)
    rows.append({
        "조건": "거래량: VOL_RATIO >= VOL_SPIKE",
        "현재값": f"{vol_ratio:,.2f}" if vol_ratio is not None else "데이터 부족",
        "기준": f">= {vol_spike:,.2f}",
        "통과": bool(c3)
    })

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
# Data loaders
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

    # yfinance가 MultiIndex로 오는 케이스 방어
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)

        if ticker in lv0:          # (ticker, field)
            df = df[ticker]
        elif ticker in lv1:        # (field, ticker)
            df = df.xs(ticker, axis=1, level=1)
        else:
            # 최후수단: 첫 블록 선택(중복 컬럼 방지 위해 ticker 축을 하나 선택)
            uniq0 = list(pd.unique(lv0))
            uniq1 = list(pd.unique(lv1))
            if len(uniq1) >= 1:
                df = df.xs(uniq1[0], axis=1, level=1)
            elif len(uniq0) >= 1:
                df = df[uniq0[0]]
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
    return df[["Open","High","Low","Close","Volume"]].dropna()

# =========================
# Sell recommendation
# =========================
def sell_recommendation(last: pd.Series, params: dict, entry_price: float, entry_date: str):
    """
    entry_price: 보유 평단(진입가)
    entry_date:  'YYYY-MM-DD' 형식 (없어도 됨)
    반환: (sell_signal, sell_reason, stop_price, target_price, holding_days)
    """
    if entry_price is None or (isinstance(entry_price, float) and np.isnan(entry_price)) or entry_price <= 0:
        return "N/A", "평단(진입가) 입력 필요", None, None, None

    # 보유일수 계산(입력 시)
    holding_days = None
    if isinstance(entry_date, str) and entry_date.strip():
        try:
            d0 = datetime.strptime(entry_date.strip(), "%Y-%m-%d").date()
            holding_days = (datetime.now().date() - d0).days
        except Exception:
            holding_days = None

    close = float(last.get("Close", np.nan))
    atr = float(last.get("ATR", np.nan))
    ma_fast = float(last.get("MA_FAST", np.nan))
    ma_slow = float(last.get("MA_SLOW", np.nan))

    if pd.isna(close):
        return "N/A", "종가 데이터 부족", None, None, holding_days

    # ATR 없으면 stop/target 계산 불가
    if pd.isna(atr) or atr <= 0:
        stop_price = None
        target_price = None
    else:
        stop_price = float(entry_price - float(params["STOP_ATR_MULT"]) * atr)
        target_price = float(entry_price + 2 * (entry_price - stop_price))  # 2R

    # 1) 손절
    if stop_price is not None and close < stop_price:
        return "SELL", "손절가 이탈(ATR 기준)", stop_price, target_price, holding_days

    # 2) 목표가
    if target_price is not None and close >= target_price:
        return "PARTIAL SELL", "목표가(2R) 도달", stop_price, target_price, holding_days

    # 3) 추세 이탈(비중축소)
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
# Analyzer (keeps df_ind for evidence charts)
# =========================
def analyze_one_with_df(ticker: str, params: dict):
    try:
        if is_kr_code(ticker):
            market = "KR"
            df = load_kr(ticker)
        else:
            market = "US"
            df = load_us(ticker)

        df_ind = add_indicators(
            df,
            MA_FAST=int(params["MA_FAST"]),
            MA_SLOW=int(params["MA_SLOW"]),
            VOL_LOOKBACK=int(params["VOL_LOOKBACK"]),
            ATR_PERIOD=int(params["ATR_PERIOD"]),
        )

        last = df_ind.iloc[-1]

        cand = rule_signal(last, params)
        sc = score_row(last) if cand == 1 else 0

        entry = float(last["Close"])
        qty, stop_buy = position_size(
            entry=entry,
            atr=float(last.get("ATR", np.nan)),
            ACCOUNT_SIZE=float(params["ACCOUNT_SIZE"]),
            RISK_PER_TRADE=float(params["RISK_PER_TRADE"]),
            STOP_ATR_MULT=float(params["STOP_ATR_MULT"]),
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
# Excel export (KRW/USD formats)
# =========================
def apply_currency_formats_openpyxl(ws):
    """
    market 기준:
      KR -> ₩#,##0
      US -> $#,##0.00
    적용 컬럼:
      close, stop, target(2R), stop_by_entry, target_by_entry(2R)
    """
    fmt_krw = u'₩#,##0'
    fmt_usd = u'$#,##0.00'
    price_cols = {"close", "stop", "target(2R)", "stop_by_entry", "target_by_entry(2R)"}

    # header map
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
        if mkt == "KR":
            numfmt = fmt_krw
        elif mkt == "US":
            numfmt = fmt_usd
        else:
            continue

        for c in price_idxs:
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

        wb = writer.book
        for name in ["Signals_All", "Candidates"]:
            if name in wb.sheetnames:
                apply_currency_formats_openpyxl(wb[name])

    return output.getvalue()

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Swing Scanner", layout="wide")

# ✅ 폰트 크기(방법 1)
st.markdown(
    """
    <h1 style="font-size:36px;font-weight:700;margin-bottom:10px;">
        웹 티커 입력 → 스윙 판단(O/X) + 근거(표+차트) + 매도 추천 + 엑셀
    </h1>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("스윙 전략 설정 (초보자 권장값)")
    params = {}

    st.markdown("### 추세 지표 (이동평균)")
    params["MA_FAST"] = st.number_input("MA_FAST (단기 이동평균)", 5, 200, DEFAULTS["MA_FAST"])
    params["MA_SLOW"] = st.number_input("MA_SLOW (장기 이동평균)", 10, 300, DEFAULTS["MA_SLOW"])

    st.markdown("---")
    st.markdown("### 거래량 / 변동성 조건")
    params["VOL_LOOKBACK"] = st.number_input("VOL_LOOKBACK (거래량 평균 기간)", 5, 200, DEFAULTS["VOL_LOOKBACK"])
    params["VOL_SPIKE"] = st.number_input("VOL_SPIKE (거래량 급증 기준)", 1.0, 5.0, float(DEFAULTS["VOL_SPIKE"]), step=0.05)
    params["ATR_PERIOD"] = st.number_input("ATR_PERIOD (ATR 계산 기간)", 5, 100, DEFAULTS["ATR_PERIOD"])
    params["ATR_PCT_MIN"] = st.number_input("ATR_PCT_MIN (최소 변동성)", 0.0, 0.2, float(DEFAULTS["ATR_PCT_MIN"]), step=0.001, format="%.3f")
    params["ATR_PCT_MAX"] = st.number_input("ATR_PCT_MAX (최대 변동성)", 0.0, 0.5, float(DEFAULTS["ATR_PCT_MAX"]), step=0.001, format="%.3f")

    st.markdown("---")
    st.markdown("### 자금 관리 / 매도 설정")
    params["ACCOUNT_SIZE"] = st.number_input("ACCOUNT_SIZE (총 투자금)", 100_000, 1_000_000_000, DEFAULTS["ACCOUNT_SIZE"], step=100_000)
    params["RISK_PER_TRADE"] = st.number_input("RISK_PER_TRADE (1회 최대 손실 비율)", 0.001, 0.05, float(DEFAULTS["RISK_PER_TRADE"]), step=0.001, format="%.3f")
    params["STOP_ATR_MULT"] = st.number_input("STOP_ATR_MULT (손절 ATR 배수)", 0.5, 5.0, float(DEFAULTS["STOP_ATR_MULT"]), step=0.1)
    params["HOLD_DAYS"] = st.number_input("HOLD_DAYS (최대 보유일)", 1, 200, DEFAULTS["HOLD_DAYS"])

st.write("**입력 방법**: KR은 6자리 코드(예: `005930`), US는 티커(예: `SPY`). 콤마/줄바꿈/공백 모두 가능.")
raw = st.text_area("티커 입력", value="005930 000660\nSPY QQQ", height=120)
run = st.button("분석 실행")

if run:
    tickers = normalize_tickers(raw)
    if not tickers:
        st.warning("티커를 1개 이상 입력하세요.")
        st.stop()

    results = []
    detail_map = {}  # ticker -> (df_ind, reason_df)

    for t in tickers:
        res, df_ind, reason_df = analyze_one_with_df(t, params)
        results.append(res)
        if df_ind is not None and reason_df is not None:
            detail_map[t] = (df_ind, reason_df)

    df = pd.DataFrame(results)
    df = df.sort_values(["candidate", "score"], ascending=[False, False]).reset_index(drop=True)

    # ---- 보유 입력(매도 추천용) ----
    st.markdown("---")
    st.subheader("보유 종목 입력 (매도 추천용)")

    pos_base = pd.DataFrame({
        "ticker": df["ticker"].tolist(),
        "entry_price": [np.nan] * len(df),
        "entry_date": [""] * len(df)  # YYYY-MM-DD
    })

    positions = st.data_editor(
        pos_base,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "ticker": st.column_config.TextColumn("ticker", disabled=True),
            "entry_price": st.column_config.NumberColumn("평단(진입가)", format="%.4f"),
            "entry_date": st.column_config.TextColumn("보유 시작일(YYYY-MM-DD)")
        }
    )

    pos_map = {}
    for _, r in positions.iterrows():
        pos_map[str(r["ticker"]).upper()] = {
            "entry_price": r["entry_price"],
            "entry_date": r["entry_date"]
        }

    # ---- 매도 추천 계산 ----
    sell_signals, sell_reasons = [], []
    stop_by_entry_list, target_by_entry_list, hold_days_list = [], [], []

    for _, row in df.iterrows():
        tkr = str(row["ticker"]).upper()
        if tkr not in detail_map:
            sell_signals.append("N/A")
            sell_reasons.append("데이터 없음")
            stop_by_entry_list.append(np.nan)
            target_by_entry_list.append(np.nan)
            hold_days_list.append(np.nan)
            continue

        df_ind, _reason_df = detail_map[tkr]
        last = df_ind.iloc[-1]

        entry_price = pos_map.get(tkr, {}).get("entry_price", np.nan)
        entry_date = pos_map.get(tkr, {}).get("entry_date", "")

        sig, reason, stp, tgt, hd = sell_recommendation(last, params, entry_price, entry_date)

        sell_signals.append(sig)
        sell_reasons.append(reason)
        stop_by_entry_list.append(np.nan if stp is None else float(stp))
        target_by_entry_list.append(np.nan if tgt is None else float(tgt))
        hold_days_list.append(np.nan if hd is None else int(hd))

    df["sell_signal"] = sell_signals
    df["sell_reason"] = sell_reasons
    df["hold_days"] = hold_days_list
    df["stop_by_entry"] = stop_by_entry_list
    df["target_by_entry(2R)"] = target_by_entry_list

    # ---- 화면 결과표(통화 문자열 표시) ----
    st.markdown("---")
    st.subheader("결과 (매수/매도 추천 포함)")

    df_view = df.copy()
    for c in ["close", "stop", "target(2R)", "stop_by_entry", "target_by_entry(2R)"]:
        if c in df_view.columns:
            df_view[c] = df_view.apply(
                lambda r: format_currency_for_display(r.get("market", ""), r.get(c, None)),
                axis=1
            )

    st.dataframe(df_view, use_container_width=True)

    n_cand = int((df["candidate"] == 1).sum())
    if n_cand == 0:
        st.error("NOT TODAY: 조건을 만족하는 후보가 없습니다. (O=0)")
    else:
        st.success(f"후보(O) {n_cand}개")

    # ---- 근거(표 + 차트) ----
    st.markdown("---")
    st.subheader("근거(조건표 + 차트)")

    for _, row in df.iterrows():
        tkr = row["ticker"]
        mkt = row["market"]
        ox = row["O/X"]
        sig = row.get("sell_signal", "")
        err = row.get("error", "")

        title = f"{tkr} ({mkt}) | 매수:{ox} | 매도:{sig}"
        with st.expander(title, expanded=False):
            if err:
                st.error(f"데이터 오류: {err}")
                continue

            if tkr not in detail_map:
                st.warning("근거 데이터 없음")
                continue

            df_ind, reason_df = detail_map[tkr]

            st.write("조건별 통과/실패 근거")
            st.dataframe(reason_df, use_container_width=True)

            st.write("종가 + 이동평균")
            chart_df1 = df_ind[["Close", "MA_FAST", "MA_SLOW"]].dropna()
            st.line_chart(chart_df1)

            st.write("거래량 + 평균 거래량")
            chart_df2 = df_ind[["Volume", "VOL_AVG"]].dropna()
            st.line_chart(chart_df2)

            # 매도/보유 정보 요약
            close = row.get("close", np.nan)
            stop_buy = row.get("stop", np.nan)
            tgt_buy = row.get("target(2R)", np.nan)
            entry_stop = row.get("stop_by_entry", np.nan)
            entry_tgt = row.get("target_by_entry(2R)", np.nan)

            st.write(
                f"- close: {format_currency_for_display(mkt, close)}\n"
                f"- (매수기준) stop: {format_currency_for_display(mkt, stop_buy)} / target(2R): {format_currency_for_display(mkt, tgt_buy)}\n"
                f"- (평단기준) stop: {format_currency_for_display(mkt, entry_stop)} / target(2R): {format_currency_for_display(mkt, entry_tgt)}\n"
                f"- sell_reason: {row.get('sell_reason','')}"
            )

    # ---- Excel download (KRW/USD formats) ----
    st.markdown("---")
    st.subheader("엑셀 다운로드")

    xlsx_bytes = build_excel(df)
    st.download_button(
        label="엑셀 다운로드 (KR ₩ / US $ 자동 적용)",
        data=xlsx_bytes,
        file_name="Swing_Scanner_Output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
