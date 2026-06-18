import sys
sys.path.insert(0, ".")
import streamlit as st
import pandas as pd

from src.data.collector import DataCollector
from src.features.technical import TechnicalFeatures
from src.models.day_trade_model import DayTradeModel

st.set_page_config(page_title="Day-Trade Scanner", layout="wide", page_icon="⚡")


@st.cache_resource
def load_model():
    m = DayTradeModel(target_col="target_12bar")
    m.load("models/day_trade_model.pkl")
    return m


@st.cache_data(ttl=300)  # cache 5 min only — intraday changes fast
def scan(ticker):
    collector = DataCollector()
    feat = TechnicalFeatures()
    model = load_model()
    try:
        df_raw = collector.fetch_intraday(ticker, interval="5m", days=60)
        if len(df_raw) < 500:
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
            "Vol vs avg": round(latest["volume_ratio"], 2),
            "ATR": round(latest["atr_14"], 2),
        }
    except Exception:
        return None


st.title("⚡ Day-Trade Scanner (5-min bars)")
st.caption("Intraday signals · predicts ~1hr ahead · ⚠️ thinner & riskier than swing signals")

st.warning("⚠️ **Day-trade signals are experimental and high-risk.** The intraday model "
           "trains on only ~60 days of data with an unstable edge (AUC ~0.53). "
           "Transaction costs hit intraday trades hardest. Treat with extra caution.")

with st.sidebar:
    st.header("⚙️ Settings")
    default = "AAPL, MSFT, NVDA, AMD, TSLA, META, AMZN, GOOGL, SPY, QQQ, AMC, INTC, BAC, F"
    tickers_input = st.text_area("Tickers (liquid stocks only)", default, height=100)
    show_only = st.radio("Show", ["All", "BUY only", "BUY + SELL"], index=0)
    scan_btn = st.button("⚡ Scan Now", type="primary", use_container_width=True)
    st.caption("Only run during market hours for live signals.")

if scan_btn:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    results = []
    prog = st.progress(0, text="Scanning intraday...")
    for i, t in enumerate(tickers):
        row = scan(t)
        if row:
            results.append(row)
        prog.progress((i + 1) / len(tickers), text=f"Scanned {t}")
    prog.empty()

    if not results:
        st.warning("No results.")
    else:
        all_df = pd.DataFrame(results)
        buys = (all_df["Signal"] == "BUY").sum()
        sells = (all_df["Signal"] == "SELL").sum()
        holds = (all_df["Signal"] == "HOLD").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 BUY", buys)
        c2.metric("🟡 HOLD", holds)
        c3.metric("🔴 SELL", sells)

        df = all_df.copy()
        if show_only == "BUY only":
            df = df[df["Signal"] == "BUY"]
        elif show_only == "BUY + SELL":
            df = df[df["Signal"].isin(["BUY", "SELL"])]
        df = df.sort_values("Confidence", ascending=False).reset_index(drop=True)

        def highlight(row):
            style = {"BUY": "background-color: #1e7e4f; color: white; font-weight: bold",
                     "SELL": "background-color: #b03a3a; color: white; font-weight: bold",
                     "HOLD": ""}.get(row["Signal"], "")
            return [style] * len(row)

        if len(df) == 0:
            st.info("No stocks match that filter.")
        else:
            st.dataframe(df.style.apply(highlight, axis=1), use_container_width=True, height=500)
else:
    st.info("👈 Click **Scan Now** to check intraday signals.")