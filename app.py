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
# 1. Defaults & Config
# -----------------------------
DEFAULTS = dict(
    MA_FAST=20, MA_SLOW=60, ATR_PERIOD=14, VOL_LOOKBACK=20,
    VOL_SPIKE=1.5, ATR_PCT_MIN=0.008, ATR_PCT_MAX=0.060,
    STOP_ATR_MULT=1.8, ACCOUNT_SIZE=10_000_000, RISK_PER_TRADE=0.01,
    LOOKBACK_YEARS=2
)

KR_UNIVERSE = ["005930","000660","035420","035720","051910","068270","207940","005380","000270","012330","066570","003550"]
US_UNIVERSE = ["SPY","QQQ","NVDA","AAPL","MSFT","TSLA","AMZN","GOOGL","META","AMD","AVGO","NFLX"]

st.set_page_config(page_title="Swing Scanner Final Pro", layout="wide")

# -----------------------------
# 2. Utility Functions
# -----------------------------
def is_kr_code(x: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", str(x).strip()))

def normalize_tickers(raw: str):
    items = re.split(r"[,\n\s]+", (raw or "").strip())
    return [x.strip().upper() for x in items if x.strip()]

@st.cache_data(ttl=3600)
def get_company_name(t):
    if t == "BTC-USD": return "Bitcoin"
    try:
        if is_kr_code(t): return krx.get_market_ticker_name(t) or t
        return yf.Ticker(t).info.get("shortName", t)
    except: return t

def parse_entry_val(market, text):
    if not text or pd.isna(text) or str(text).strip() == "": return 0.0
    raw = str(text).replace("₩","").replace("$","").replace(",","").strip()
    try:
        val = float(raw)
        return float(int(val)) if market == "KR" else round(val, 2)
    except: return 0.0

def format_curr(mkt, v):
    if not v or v == 0 or pd.isna(v): return ""
    try:
        return f"₩{int(round(float(v))):,}" if mkt == "KR" else f"${float(v):,.2f}"
    except: return str(v)

# -----------------------------
# 3. Session State
# -----------------------------
if "pos_df" not in st.session_state:
    st.session_state.pos_df = pd.DataFrame(columns=["market","ticker","name","entry_text","entry_price","entry_display","entry_date"])
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "all_data" not in st.session_state:
    st.session_state.all_data = {}
if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = "BTC-USD 005930 NVDA"
if "msg" not in st.session_state:
    st.session_state.msg = ""

def on_pos_edit():
    if "pos_editor" in st.session_state:
        ed = st.session_state["pos_editor"]["edited_rows"]
        for idx, changes in ed.items():
            idx_int = int(idx)
            mkt = st.session_state.pos_df.at[idx_int, "market"]
            if "entry_text" in changes:
                new_text = changes["entry_text"]
                new_val = parse_entry_val(mkt, new_text)
                st.session_state.pos_df.at[idx_int, "entry_text"] = str(new_text)
                st.session_state.pos_df.at[idx_int, "entry_price"] = float(new_val)
                st.session_state.pos_df.at[idx_int, "entry_display"] = format_curr(mkt, new_val)

# -----------------------------
# 4. Analysis Engine
# -----------------------------
@st.cache_data(ttl=1200)
def load_data(ticker, years):
    try:
        if is_kr_code(ticker):
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=365*years)).strftime("%Y%m%d")
            df = krx.get_market_ohlcv_by_date(start, end, ticker)
            df = df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
        else:
            df = yf.download(ticker, period=f"{years}y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=lambda x: x.title())
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except: return pd.DataFrame()

def analyze_one(ticker, p):
    df = load_data(ticker, p["LOOKBACK_YEARS"])
    if df.empty: return {"candidate": 0, "ticker": ticker, "score": 0, "error": "Data Error"}, None
    
    df["MA_FAST"] = df["Close"].rolling(int(p["MA_FAST"])).mean()
    df["MA_SLOW"] = df["Close"].rolling(int(p["MA_SLOW"])).mean()
    df["VOL_AVG"] = df["Volume"].rolling(int(p["VOL_LOOKBACK"])).mean()
    tr = pd.concat([(df["High"]-df["Low"]), (df["High"]-df["Close"].shift()).abs(), (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(int(p["ATR_PERIOD"])).mean()
    
    last = df.iloc[-1]
    vol_ratio = last["Volume"] / last["VOL_AVG"] if last["VOL_AVG"] > 0 else 0
    atr_pct = last["ATR"] / last["Close"]
    
    c1 = (last["MA_FAST"] > last["MA_SLOW"]) and (last["Close"] > last["MA_FAST"])
    c2 = vol_ratio >= p["VOL_SPIKE"]
    c3 = p["ATR_PCT_MIN"] <= atr_pct <= p["ATR_PCT_MAX"]
    
    score = (40 if c1 else 0) + int(min(30, vol_ratio * 10)) + (30 if c3 else 0)
    cand = 1 if (c1 and c2 and c3) else 0
    stop_dist = p["STOP_ATR_MULT"] * last["ATR"]
    
    res = {
        "market": "KR" if is_kr_code(ticker) else "US",
        "ticker": ticker,
        "name": get_company_name(ticker),
        "OX": "O" if cand else "X",
        "candidate": cand,
        "score": score,
        "close": float(last["Close"]),
        "stop": float(last["Close"] - stop_dist),
        "target": float(last["Close"] + stop_dist * 2),
        "vol_ratio": float(vol_ratio),
        "atr_pct": float(atr_pct * 100),
        "error": ""
    }
    return res, df

# -----------------------------
# 5. Sidebar (Params)
# -----------------------------
with st.sidebar:
    st.header("⚙️ 전략 파라미터 설정")
    p = {k: st.number_input(k, value=v) for k, v in DEFAULTS.items() if k != "LOOKBACK_YEARS"}
    p["LOOKBACK_YEARS"] = DEFAULTS["LOOKBACK_YEARS"]
    params = p

# -----------------------------
# 6. Main UI
# -----------------------------
st.title("⚖️ Swing Scanner Final Pro")

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("🌟 국장5+미장5 추천"):
        with st.spinner("최적 종목 스캔 중..."):
            kr_list = [analyze_one(t, params)[0] for t in KR_UNIVERSE]
            kr_top = pd.DataFrame([x for x in kr_list if x["candidate"]]).sort_values("score", ascending=False).head(5)["ticker"].tolist()
            us_list = [analyze_one(t, params)[0] for t in US_UNIVERSE]
            us_top = pd.DataFrame([x for x in us_list if x["candidate"]]).sort_values("score", ascending=False).head(5)["ticker"].tolist()
            st.session_state.ticker_input = " ".join(["BTC-USD"] + kr_top + us_top)
            st.session_state.msg = f"✅ 추천 완료: 국장 {len(kr_top)}개, 미장 {len(us_top)}개 (비트코인 포함)"
            st.rerun()

if st.session_state.msg:
    st.success(st.session_state.msg)
    st.session_state.msg = ""

ticker_area = st.text_area("분석 티커 입력", value=st.session_state.ticker_input, height=100)

if st.button("🚀 분석 실행", type="primary"):
    tickers = normalize_tickers(ticker_area)
    results, data_map = [], {}
    for t in tickers:
        res, df = analyze_one(t, params)
        results.append(res)
        if df is not None: data_map[t] = df
    st.session_state.analysis_df = pd.DataFrame(results)
    st.session_state.all_data = data_map
    
    new_rows = []
    for _, row in st.session_state.analysis_df.iterrows():
        exist = st.session_state.pos_df[st.session_state.pos_df["ticker"] == row["ticker"]]
        new_rows.append(exist.iloc[0].to_dict() if not exist.empty else {
            "market": row["market"], "ticker": row["ticker"], "name": row["name"],
            "entry_text": "", "entry_price": 0.0, "entry_display": "", "entry_date": None
        })
    st.session_state.pos_df = pd.DataFrame(new_rows)
    st.session_state.pos_df["entry_date"] = pd.to_datetime(st.session_state.pos_df["entry_date"])

# -----------------------------
# 7. 결과 및 차트 표시
# -----------------------------
if st.session_state.analysis_df is not None:
    # 7-1. 평단 관리
    st.subheader("📥 보유 종목 평단 관리")
    st.data_editor(st.session_state.pos_df, key="pos_editor", on_change=on_pos_edit,
        column_config={"entry_text": "평단가 입력", "entry_display": "✅ 계산된 평단", "entry_date": st.column_config.DateColumn("진입일"), "name": "종목명"},
        hide_index=True, use_container_width=True)

    # 7-2. 분석표
    st.subheader("🔍 분석 결과 및 매도 추천")
    df_view = st.session_state.analysis_df.copy()
    def get_sig(r):
        pos = st.session_state.pos_df[st.session_state.pos_df["ticker"] == r["ticker"]]
        if pos.empty or not pos.iloc[0]["entry_price"]: return "HOLD", "-", 0.0
        e, c = pos.iloc[0]["entry_price"], r["close"]
        p = (c - e) / e * 100
        return ("🔴 SELL", "손절", p) if c < e*0.95 else ("🟢 TAKE", "익절", p) if c > e*1.15 else ("⚪ HOLD", "유지", p)
    df_view[["Signal", "Reason", "Profit%"]] = df_view.apply(lambda r: pd.Series(get_sig(r)), axis=1)
    
    disp = df_view.copy()
    for c in ["close", "stop", "target"]: disp[c] = disp.apply(lambda r: format_curr(r["market"], r[c]), axis=1)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # 7-3. 차트 섹션 (신규 추가)
    st.subheader("📊 종목별 추세 차트 (최근 120일)")
    tabs = st.tabs([f"{row['name']} ({row['ticker']})" for _, row in df_view.iterrows()])
    for i, (_, row) in enumerate(df_view.iterrows()):
        with tabs[i]:
            df_chart = st.session_state.all_data.get(row["ticker"])
            if df_chart is not None:
                c_data = df_chart.tail(120).reset_index()
                base = alt.Chart(c_data).encode(x=alt.X('Date:T', title='날짜'))
                line = base.mark_line(color='#1f77b4').encode(y=alt.Y('Close:Q', title='가격', scale=alt.Scale(zero=False)))
                ma20 = base.mark_line(color='orange', strokeDash=[5,5]).encode(y='MA_FAST:Q')
                ma60 = base.mark_line(color='red', strokeDash=[2,2]).encode(y='MA_SLOW:Q')
                st.altair_chart((line + ma20 + ma60).properties(height=300), use_container_width=True)
            else: st.warning("차트 데이터를 불러올 수 없습니다.")

    # 7-4. 엑셀 다운로드
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_view.to_excel(writer, index=False, sheet_name='분석_결과')
        st.session_state.pos_df.merge(df_view[["ticker", "close", "Profit%"]], on="ticker", how="left").to_excel(writer, index=False, sheet_name='나의_포트폴리오')
    st.download_button("📂 분석 결과 + 포트폴리오 엑셀 다운로드", output.getvalue(), f"Swing_Report_{datetime.now().strftime('%Y%m%d')}.xlsx")

st.markdown("---")
st.caption("Swing Scanner Final Pro | 초기 기능 검수 완료 + 탭 형태의 종목별 차트 기능 추가")
