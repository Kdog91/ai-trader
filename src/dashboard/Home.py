import sys
sys.path.insert(0, ".")
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.collector import DataCollector
from src.features.technical import TechnicalFeatures
from src.models.xgboost_model import XGBoostTrader
from src.backtesting.engine import BacktestEngine

st.set_page_config(page_title="AI Trading System", layout="wide", page_icon="📈")


@st.cache_resource
def load_model():
    model = XGBoostTrader(target_col="target_5d")
    model.load("models/xgb_walk_forward.pkl")
    return model


@st.cache_data(ttl=3600)
def get_data(ticker, start="2023-01-01"):
    collector = DataCollector()
    feat = TechnicalFeatures()
    df_raw = collector.fetch_by_dates(ticker, start=start)
    return feat.add_all_features(df_raw)


st.title("🤖 AI Stock Trading System")
st.caption("Swing model trained on 2011-2022 · validated out-of-sample · research & education only")

with st.expander("ℹ️ About this system", expanded=False):
    st.markdown("""
    **What this is:** an ML system that predicts short-term stock direction.
    - **Home** (this page): analyze a single stock in depth
    - **Swing Scanner**: scan your whole watchlist for daily signals
    - **Day-Trade Scanner**: intraday 5-minute signals (higher risk)

    **Honest limits:** the edge is thin (~0.53 AUC, ~1.9 profit factor out-of-sample),
    regime-dependent, and unproven in live trading. This is a research tool,
    **not financial advice.**
    """)

with st.sidebar:
    st.header("⚙️ Single-Stock Analysis")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    start_date = st.selectbox("History", ["2023-01-01", "2020-01-01", "2015-01-01"], index=0)
    buy_threshold = st.slider("Buy confidence threshold", 0.50, 0.75, 0.55, 0.01)
    run = st.button("🚀 Analyze", type="primary", use_container_width=True)

if run or ticker:
    try:
        model = load_model()
        df = get_data(ticker, start_date)
        df = model.predict(df)
        latest = df.iloc[-1]

        sig = latest["signal"]
        conf = latest["confidence"] * 100
        sig_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig, "⚪")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal", f"{sig_color} {sig}")
        c2.metric("Confidence", f"{conf:.1f}%")
        c3.metric("Price", f"${latest['Close']:.2f}")
        c4.metric("ATR (volatility)", f"${latest['atr_14']:.2f}")

        tab1, tab2, tab3 = st.tabs(["📊 Chart", "📈 Backtest", "🔍 Why this signal"])

        # --- Entry / Exit Plan (only for BUY signals) ---
        if sig == "BUY":
            entry = latest["Close"]
            atr = latest["atr_14"]
            stop = entry - 2 * atr
            target = entry + 4 * atr
            risk_pct = (entry - stop) / entry * 100
            reward_pct = (target - entry) / entry * 100

            st.markdown("### 📋 Trade Plan (if you took this signal)")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Entry", f"${entry:.2f}")
            p2.metric("Stop-loss", f"${stop:.2f}", f"-{risk_pct:.1f}%")
            p3.metric("Target", f"${target:.2f}", f"+{reward_pct:.1f}%")
            p4.metric("Max hold", "10 days")
            st.caption(
                f"Plan uses the same rules as the backtest: stop at 2×ATR below entry, "
                f"target at 4×ATR above, exit after 10 trading days. "
                f"Risk/reward ≈ 1:{reward_pct/risk_pct:.1f}. "
                f"⚠️ This is the model's mechanical plan, not advice — and the signal is unproven live."
            )
        else:
            st.info(f"No trade plan shown — current signal is **{sig}**, not BUY. "
                    f"A plan only makes sense when the model says BUY.")

        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.7, 0.3], vertical_spacing=0.05,
                                subplot_titles=(f"{ticker} Price + Signals", "RSI"))
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
            buys = df[df["signal"] == "BUY"]
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys["Low"] * 0.98, mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="lime"),
                name="BUY"), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df["ema_200"], line=dict(color="orange", width=1),
                name="EMA200"), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df["rsi_14"], line=dict(color="cyan", width=1),
                name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark",
                              xaxis_rangeslider_visible=False, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            engine = BacktestEngine(starting_capital=100_000, position_size=0.25)
            results = engine.run(df, buy_threshold=buy_threshold)
            if "error" in results:
                st.warning(results["error"])
            else:
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Total Return", f"{results['total_return_pct']}%")
                b2.metric("Win Rate", f"{results['win_rate_pct']}%")
                b3.metric("Profit Factor", f"{results['profit_factor']}")
                b4.metric("Max Drawdown", f"{results['max_drawdown_pct']}%")
                b5, b6, b7, b8 = st.columns(4)
                b5.metric("Total Trades", results["total_trades"])
                b6.metric("Sharpe", results["sharpe_ratio"])
                b7.metric("Avg Win", f"${results['avg_win']:,.0f}")
                b8.metric("Avg Loss", f"${results['avg_loss']:,.0f}")
                eq = pd.Series(engine.equity_curve, index=df.index[:len(engine.equity_curve)])
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=eq.index, y=eq, line=dict(color="lime"), name="Equity"))
                fig2.add_hline(y=100_000, line_dash="dash", line_color="gray")
                fig2.update_layout(title="Equity Curve (starting $100k)",
                                   height=350, template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)
                if engine.trades:
                    trades_df = pd.DataFrame([{
                        "Entry": t.entry_date[:10], "Exit": t.exit_date[:10] if t.exit_date else "-",
                        "Entry $": t.entry_price, "Exit $": t.exit_price,
                        "P&L $": t.pnl, "P&L %": t.pnl_pct, "Reason": t.exit_reason
                    } for t in engine.trades])
                    st.dataframe(trades_df, use_container_width=True, height=250)

        with tab3:
            st.subheader("What the model is seeing right now")
            sig_data = {
                "RSI (14)": f"{latest['rsi_14']:.1f}",
                "MACD cross": "Bullish" if latest['macd_cross'] == 1 else "Bearish",
                "Above EMA200": "Yes" if latest['above_ema200'] == 1 else "No",
                "Bollinger position": f"{latest['bb_pct']:.2f} (0=low, 1=high)",
                "Volume vs avg": f"{latest['volume_ratio']:.2f}x",
                "ADX (trend strength)": f"{latest['adx_14']:.1f}",
                "Market regime": "Strong trend" if latest['strong_trend'] == 1
                                 else "Choppy" if latest['choppy'] == 1 else "Neutral",
                "Distance from 50-day avg": f"{latest['dist_from_sma50']*100:.1f}%",
            }
            for k, v in sig_data.items():
                col_a, col_b = st.columns([1, 2])
                col_a.write(f"**{k}**")
                col_b.write(v)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check the ticker symbol is valid and the model file exists.")

st.divider()
st.caption("⚠️ Experimental research tool, not financial advice. "
           "Never trade money you can't afford to lose.")