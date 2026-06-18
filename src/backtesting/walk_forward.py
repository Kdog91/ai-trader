import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from src.data.collector import DataCollector
from src.features.technical import TechnicalFeatures
from src.models.xgboost_model import XGBoostTrader
from src.backtesting.engine import BacktestEngine


def build_dataset(tickers, collector, feat, start, end):
    """Fetch + feature-engineer a set of stocks for a date range."""
    all_data = []
    for ticker in tickers:
        try:
            df_raw = collector.fetch_by_dates(ticker, start=start, end=end)
            if len(df_raw) < 250:
                continue
            df_feat = feat.add_all_features(df_raw)
            all_data.append(df_feat)
            print(f"    {ticker}: {len(df_feat)} rows")
        except Exception as e:
            print(f"    {ticker}: ERROR - {e}")
    return pd.concat(all_data, ignore_index=True)


if __name__ == "__main__":
    collector = DataCollector()
    feat      = TechnicalFeatures()

    # TRAINING STOCKS (model learns ONLY these)
    train_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "TSLA",
        "INTC", "CSCO", "ORCL", "CRM", "ADBE", "QCOM", "TXN",
        "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP",
        "JNJ", "PFE", "UNH", "MRK", "TMO",
        "XOM", "CVX", "COP",
        "WMT", "COST", "HD", "MCD", "NKE", "PG", "KO", "PEP",
        "BA", "CAT", "GE", "DIS",
    ]

    # TEST STOCKS (model has NEVER seen these)
    test_tickers = ["NVDA", "V", "MA", "ABT", "SBUX", "NFLX", "AVGO", "ABBV"]

    TRAIN_START, TRAIN_END = "2011-01-01", "2022-12-31"
    TEST_START,  TEST_END  = "2023-01-01", "2026-06-17"

    print("=" * 60)
    print("OUT-OF-SAMPLE TEST  (the honest experiment)")
    print("=" * 60)
    print(f"  Train: {len(train_tickers)} stocks, {TRAIN_START} to {TRAIN_END}")
    print(f"  Test:  {len(test_tickers)} UNSEEN stocks, {TEST_START} to {TEST_END}")

    print("\n" + "=" * 60)
    print("STEP 1: Training on 2011-2022 data only")
    print("=" * 60)
    train_df = build_dataset(train_tickers, collector, feat, TRAIN_START, TRAIN_END)
    print(f"\n  Training rows: {len(train_df):,}\n")

    model = XGBoostTrader(target_col="target_5d")
    metrics = model.train(train_df)
    print(f"\n  Training AUC: {metrics['cv_auc_mean']}")
    model.save("models/xgb_walk_forward.pkl")

    print("\n" + "=" * 60)
    print("STEP 2: Backtesting on UNSEEN stocks (2023-2026)")
    print("=" * 60)

    results_summary = []
    for ticker in test_tickers:
        df_raw = collector.fetch_by_dates(ticker, start=TEST_START, end=TEST_END)
        if len(df_raw) < 60:
            print(f"\n  {ticker}: not enough data, skipping")
            continue
        df = feat.add_all_features(df_raw)
        df = model.predict(df)

        engine = BacktestEngine(starting_capital=100_000, position_size=0.25)
        res = engine.run(df, buy_threshold=0.55)

        if "error" in res:
            print(f"\n  {ticker}: {res['error']}")
            continue

        results_summary.append((ticker, res))
        print(f"\n  {ticker}:")
        print(f"    Return: {res['total_return_pct']}%   "
              f"Trades: {res['total_trades']}   "
              f"Win rate: {res['win_rate_pct']}%   "
              f"Profit factor: {res['profit_factor']}   "
              f"Max DD: {res['max_drawdown_pct']}%")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if results_summary:
        pfs     = [r["profit_factor"] for _, r in results_summary if r["profit_factor"] < 900]
        returns = [r["total_return_pct"] for _, r in results_summary]
        wins    = [r["win_rate_pct"] for _, r in results_summary]
        avg_pf  = np.mean(pfs) if pfs else 0
        print(f"  Avg profit factor (unseen): {avg_pf:.2f}")
        print(f"  Avg return:                 {np.mean(returns):.2f}%")
        print(f"  Avg win rate:               {np.mean(wins):.2f}%")
        print(f"  Profitable stocks:          "
              f"{sum(1 for r in returns if r > 0)}/{len(returns)}")
        print()
        if avg_pf >= 1.3:
            print("  >>> REAL EDGE: profit factor holds up on unseen data.")
        elif avg_pf >= 1.0:
            print("  >>> MARGINAL: barely profitable. Edge is weak/uncertain.")
        else:
            print("  >>> NO EDGE: the earlier results were memorization.")
    print("=" * 60)