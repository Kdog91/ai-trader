import sys
sys.path.insert(0, ".")
import pandas as pd
from datetime import datetime, timedelta

from src.data.collector import DataCollector
from src.data.database import Database
from src.features.technical import TechnicalFeatures
from src.models.xgboost_model import XGBoostTrader


def log_todays_signals(tickers):
    """Save today's swing signals to the database."""
    collector = DataCollector()
    feat = TechnicalFeatures()
    db = Database()
    model = XGBoostTrader(target_col="target_5d")
    model.load("models/xgb_walk_forward.pkl")

    print(f"\n{'='*55}")
    print(f"LOGGING SIGNALS — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*55}")

    logged = 0
    for ticker in tickers:
        try:
            df = feat.add_all_features(collector.fetch_by_dates(ticker, start="2024-01-01"))
            df = model.predict(df)
            latest = df.iloc[-1]
            db.log_prediction(ticker, latest["signal"],
                              round(latest["confidence"] * 100, 1),
                              round(latest["Close"], 2), "swing")
            print(f"  {ticker:5s}  {latest['signal']:5s}  "
                  f"{latest['confidence']*100:.0f}%  ${latest['Close']:.2f}")
            logged += 1
        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")
    print(f"\n  Logged {logged} signals to database.")


def check_old_predictions():
    """Score predictions that are 5+ days old against what actually happened."""
    collector = DataCollector()
    db = Database()

    pending = db.get_unchecked_predictions()
    if len(pending) == 0:
        print("\n  No predictions to check yet.")
        return

    print(f"\n{'='*55}")
    print("CHECKING OLD PREDICTIONS")
    print(f"{'='*55}")

    today = datetime.now().date()
    checked = 0
    for _, row in pending.iterrows():
        sig_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
        days_elapsed = (today - sig_date).days
        if days_elapsed < 5:
            continue  # not old enough to judge a 5-day prediction

        try:
            df = collector.fetch_by_dates(row["ticker"],
                                          start=row["date"])
            if len(df) < 5:
                continue
            price_then = row["price_at_signal"]
            price_now = df["Close"].iloc[min(5, len(df) - 1)]  # ~5 days later
            went_up = price_now > price_then

            # Was the signal right?
            if row["signal"] == "BUY":
                correct = 1 if went_up else 0
            elif row["signal"] == "SELL":
                correct = 1 if not went_up else 0
            else:
                correct = -1  # HOLD, not scored

            if correct != -1:
                db.mark_checked(row["id"], round(price_now, 2), correct)
                result = "✓ RIGHT" if correct == 1 else "✗ WRONG"
                print(f"  {row['ticker']:5s} {row['signal']:5s} "
                      f"${price_then:.2f}→${price_now:.2f}  {result}")
                checked += 1
        except Exception as e:
            print(f"  {row['ticker']}: ERROR - {e}")
    print(f"\n  Scored {checked} predictions.")


def show_scorecard():
    """Show the running track record."""
    db = Database()
    df = db.get_scorecard()
    if len(df) == 0:
        print("\n  No scored predictions yet. Come back in 5+ days.")
        return

    actionable = df[df["signal"].isin(["BUY", "SELL"])]
    if len(actionable) == 0:
        print("\n  No BUY/SELL predictions scored yet.")
        return

    print(f"\n{'='*55}")
    print("FORWARD-TEST SCORECARD (real unseen data)")
    print(f"{'='*55}")
    total = len(actionable)
    correct = actionable["correct"].sum()
    print(f"  Total scored signals: {total}")
    print(f"  Correct:              {correct}")
    print(f"  Accuracy:             {correct/total*100:.1f}%")
    print(f"\n  (50% = no edge. Above 53-55% = the backtest edge is holding up live.)")


if __name__ == "__main__":
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSLA",
                 "JPM", "V", "WMT", "DIS", "NFLX", "CRM", "AMC", "SPY", "QQQ"]

    # Every run does all three: log today, check old, show scorecard
    log_todays_signals(watchlist)
    check_old_predictions()
    show_scorecard()