import sys
sys.path.insert(0, ".")
import streamlit as st
import pandas as pd

from src.data.collector import DataCollector
from src.features.technical import TechnicalFeatures
from src.models.xgboost_model import XGBoostTrader

st.set_page_config(page_title="Swing Scanner", layout="wide", page_icon="🔍")


@st.cache_resource
def load_model():
    model = XGBoostTrader(target_col="target_5d")
    model.load("models/xgb_walk_forward.pkl")
    return model


@st.cache_data(ttl=1800)
def scan_ticker(ticker):
    collector = DataCollector()
    feat = TechnicalFeatures()
    model = load_model()
    try:
        df_raw = collector.fetch_by_dates(ticker, start="2024-01-01")
        if len(df_raw) < 250:
            return None
        df = feat.add_all_features(df_raw)
        df = model.predict(df)
        latest = df.iloc[-1]
        return {
            "Ticker": ticker,
            "Signal": latest["signal"],
            "Confidence": round(latest["confidence"] * 100, 1),
            "Price": round(latest["Close"], 2),
            "RSI": round(latest["rsi_14"], 1),
            "Regime": ("Trending" if latest["strong_trend"] == 1
                       else "Choppy" if latest["choppy"] == 1 else "Neutral"),
            "AboveEMA200": "Yes" if latest["above_ema200"] == 1 else "No",
            "ATR": round(latest["atr_14"], 2),
        }
    except Exception:
        return None


st.title("🔍 Multi-Stock Swing Scanner")
st.caption("Scans your whole watchlist at once · shows today's BUY/HOLD/SELL for each")

with st.sidebar:
    st.header("⚙️ Scanner Settings")
    default_list = ("AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, TSLA, "
                    "JPM, BAC, V, MA, JNJ, UNH, WMT, COST, HD, MCD, "
                    "NKE, DIS, NFLX, SBUX, KO, PEP, XOM, CVX, BA, CAT, "
                    "AVGO, QCOM, ORCL, CRM, SPY, QQQ, AMC")
    tickers_input = st.text_area("Tickers (comma-separated)", default_list, height=120)
    show_only = st.radio("Show", ["All", "BUY only", "BUY + SELL"], index=0)
    scan_btn = st.button("🚀 Scan All", type="primary", use_container_width=True)

if scan_btn:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    results = []
    progress = st.progress(0, text="Scanning...")
    for i, ticker in enumerate(tickers):
        row = scan_ticker(ticker)
        if row:
            results.append(row)
        progress.progress((i + 1) / len(tickers), text=f"Scanned {ticker}")
    progress.empty()

    if not results:
        st.warning("No results — check your ticker symbols.")
    else:
        all_df = pd.DataFrame(results)
        buys = (all_df["Signal"] == "BUY").sum()
        sells = (all_df["Signal"] == "SELL").sum()
        holds = (all_df["Signal"] == "HOLD").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 BUY signals", buys)
        c2.metric("🟡 HOLD", holds)
        c3.metric("🔴 SELL signals", sells)
        st.divider()

        df = all_df.copy()
        if show_only == "BUY only":
            df = df[df["Signal"] == "BUY"]
        elif show_only == "BUY + SELL":
            df = df[df["Signal"].isin(["BUY", "SELL"])]
        df = df.sort_values("Confidence", ascending=False).reset_index(drop=True)

        def highlight(row):
            style = {"BUY":  "background-color: #1e7e4f; color: white; font-weight: bold",
                     "SELL": "background-color: #b03a3a; color: white; font-weight: bold",
                     "HOLD": ""}.get(row["Signal"], "")
            return [style] * len(row)

        if len(df) == 0:
            st.info("No stocks match that filter right now.")
        else:
            st.dataframe(df.style.apply(highlight, axis=1),
                         use_container_width=True, height=600)
        st.caption("Sorted by confidence. Green rows = BUY signals worth investigating.")
else:
    st.info("👈 Set your tickers and click **Scan All** to see today's signals.")

st.divider()
st.caption("⚠️ Research tool, not financial advice.")