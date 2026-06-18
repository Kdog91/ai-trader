import sys
sys.path.insert(0, ".")
import numpy as np
from src.data.collector import DataCollector
from src.features.technical import TechnicalFeatures
from src.models.xgboost_model import XGBoostTrader
from src.backtesting.engine import BacktestEngine


def run_period(model, collector, feat, tickers, start, end, label):
    """Backtest the model across several stocks for one time period."""
    print("\n" + "=" * 60)
    print(f"  {label}  ({start} to {end})")
    print("=" * 60)

    results = []
    for ticker in tickers:
        try:
            df_raw = collector.fetch_by_dates(ticker, start=start, end=end)
            if len(df_raw) < 60:
                continue
            df = feat.add_all_features(df_raw)
            if len(df) < 30:
                continue
            df = model.predict(df)

            engine = BacktestEngine(starting_capital=100_000, position_size=0.25)
            res = engine.run(df, buy_threshold=0.55)
            if "error" in res:
                continue
            results.append((ticker, res))
            print(f"    {ticker:5s}  return={res['total_return_pct']:>7}%  "
                  f"trades={res['total_trades']:>3}  "
                  f"PF={res['profit_factor']:>5}  "
                  f"maxDD={res['max_drawdown_pct']:>7}%")
        except Exception as e:
            print(f"    {ticker}: ERROR - {e}")

    if results:
        pfs     = [r["profit_factor"] for _, r in results if r["profit_factor"] < 900]
        returns = [r["total_return_pct"] for _, r in results]
        avg_pf  = np.mean(pfs) if pfs else 0
        print(f"\n    SUMMARY: avg PF={avg_pf:.2f}   "
              f"avg return={np.mean(returns):.2f}%   "
              f"profitable={sum(1 for r in returns if r > 0)}/{len(returns)}")
    return results


if __name__ == "__main__":
    collector = DataCollector()
    feat      = TechnicalFeatures()
    model     = XGBoostTrader(target_col="target_5d")

    # Use the walk-forward model (trained ONLY on 2011-2022)
    model.load("models/xgb_walk_forward.pkl")
    print("Loaded model trained on 2011-2022 only.")

    # Stocks to test across all periods (mix that existed back then)
    tickers = ["AAPL", "MSFT", "JPM", "WMT", "XOM", "DIS", "BA", "KO"]

    print("\n" + "#" * 60)
    print("#  STRESS TEST: does the edge survive market crashes?")
    print("#" * 60)

    # 2020 COVID crash + recovery
    run_period(model, collector, feat, tickers,
               "2020-01-01", "2020-12-31", "2020 COVID CRASH + RECOVERY")

    # 2022 bear market (rates, inflation)
    run_period(model, collector, feat, tickers,
               "2022-01-01", "2022-12-31", "2022 BEAR MARKET")

    # 2018 Q4 selloff
    run_period(model, collector, feat, tickers,
               "2018-01-01", "2018-12-31", "2018 SELLOFF YEAR")

    # A calm bull year for comparison
    run_period(model, collector, feat, tickers,
               "2021-01-01", "2021-12-31", "2021 CALM BULL (comparison)")

    print("\n" + "#" * 60)
    print("#  HOW TO READ THIS")
    print("#" * 60)
    print("  If PF stays above ~1.0 in crash years -> edge is robust.")
    print("  If PF collapses below 1.0 in crashes  -> edge only works in bull markets.")
    print("  The 2021 bull year is your baseline for comparison.")