import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Trade:
    entry_date: str
    entry_price: float
    exit_date: str = None
    exit_price: float = None
    shares: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


class BacktestEngine:
    """
    Simulates trading the model's signals over historical data.
    Accounts for transaction costs and uses ATR-based stops/targets.
    Tells you the REAL bottom line: does this make money?
    """

    def __init__(self, starting_capital: float = 100_000,
                 cost_per_trade: float = 0.001,   # 0.1% per trade (commission + slippage)
                 position_size: float = 0.95,     # use 95% of capital per trade
                 stop_atr_mult: float = 2.0,      # stop loss = 2x ATR below entry
                 target_atr_mult: float = 4.0,    # target = 4x ATR above entry (2:1 reward)
                 max_hold_days: int = 10):        # exit after 10 days regardless
        self.starting_capital = starting_capital
        self.cost_per_trade   = cost_per_trade
        self.position_size    = position_size
        self.stop_atr_mult    = stop_atr_mult
        self.target_atr_mult  = target_atr_mult
        self.max_hold_days    = max_hold_days
        self.trades: List[Trade] = []
        self.equity_curve = []

    def run(self, df: pd.DataFrame, buy_threshold: float = 0.55) -> dict:
        """
        df must have: Close, High, Low, atr_14, confidence (from model.predict)
        Enters long when confidence > buy_threshold.
        Exits on stop loss, target, or max hold days.
        """
        capital = self.starting_capital
        self.trades = []
        self.equity_curve = []

        in_position = False
        entry_price = stop_price = target_price = 0.0
        entry_idx = 0
        shares = 0
        current_trade = None

        rows = df.reset_index()

        for i in range(len(rows)):
            row   = rows.iloc[i]
            price = row["Close"]
            high  = row["High"]
            low   = row["Low"]

            if in_position:
                exit_now = False
                exit_price = price
                reason = ""

                # Check stop loss (intraday low hit it)
                if low <= stop_price:
                    exit_price = stop_price
                    exit_now = True
                    reason = "stop_loss"
                # Check target (intraday high hit it)
                elif high >= target_price:
                    exit_price = target_price
                    exit_now = True
                    reason = "target"
                # Check max hold time
                elif (i - entry_idx) >= self.max_hold_days:
                    exit_now = True
                    reason = "time_exit"

                if exit_now:
                    proceeds = shares * exit_price * (1 - self.cost_per_trade)
                    cost_basis = shares * entry_price
                    pnl = proceeds - cost_basis
                    capital += proceeds

                    current_trade.exit_date  = str(row.iloc[0])
                    current_trade.exit_price = round(exit_price, 2)
                    current_trade.pnl        = round(pnl, 2)
                    current_trade.pnl_pct    = round(pnl / cost_basis * 100, 2)
                    current_trade.exit_reason = reason
                    self.trades.append(current_trade)

                    in_position = False
                    shares = 0

            # Entry signal
            elif row["confidence"] > buy_threshold:
                atr = row["atr_14"]
                entry_price  = price * (1 + self.cost_per_trade)  # pay slippage on entry
                stop_price   = price - (atr * self.stop_atr_mult)
                target_price = price + (atr * self.target_atr_mult)
                shares = int((capital * self.position_size) / entry_price)

                if shares > 0:
                    capital -= shares * entry_price
                    entry_idx = i
                    in_position = True
                    current_trade = Trade(
                        entry_date=str(row.iloc[0]),
                        entry_price=round(entry_price, 2),
                        shares=shares,
                    )

            # Track equity (cash + current position value)
            position_value = shares * price if in_position else 0
            self.equity_curve.append(capital + position_value)

        # Close any open position at the end
        if in_position:
            final_price = rows.iloc[-1]["Close"]
            proceeds = shares * final_price * (1 - self.cost_per_trade)
            capital += proceeds
            cost_basis = shares * entry_price
            current_trade.exit_date  = str(rows.iloc[-1].iloc[0])
            current_trade.exit_price = round(final_price, 2)
            current_trade.pnl        = round(proceeds - cost_basis, 2)
            current_trade.pnl_pct    = round((proceeds - cost_basis) / cost_basis * 100, 2)
            current_trade.exit_reason = "end_of_data"
            self.trades.append(current_trade)

        return self._metrics(capital)

    def _metrics(self, final_capital: float) -> dict:
        if not self.trades:
            return {"error": "No trades were triggered. Try lowering buy_threshold."}

        pnls     = [t.pnl for t in self.trades]
        winners  = [p for p in pnls if p > 0]
        losers   = [p for p in pnls if p < 0]

        equity = pd.Series(self.equity_curve)
        peak   = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min() * 100

        daily_ret = equity.pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                  if daily_ret.std() > 0 else 0)

        total_return = (final_capital - self.starting_capital) / self.starting_capital * 100

        # Count exit reasons
        reasons = {}
        for t in self.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        return {
            "total_return_pct": round(total_return, 2),
            "final_capital":    round(final_capital, 2),
            "total_trades":     len(self.trades),
            "win_rate_pct":     round(len(winners) / len(pnls) * 100, 2),
            "avg_win":          round(np.mean(winners), 2) if winners else 0,
            "avg_loss":         round(np.mean(losers), 2) if losers else 0,
            "profit_factor":    round(sum(winners) / abs(sum(losers)), 2) if losers else 999,
            "sharpe_ratio":     round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "exit_reasons":     reasons,
        }


# --- TEST IT ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.collector import DataCollector
    from src.features.technical import TechnicalFeatures
    from src.models.xgboost_model import XGBoostTrader

    collector = DataCollector()
    feat      = TechnicalFeatures()
    model     = XGBoostTrader(target_col="target_5d")

    print("=" * 60)
    print("BACKTESTING THE MULTI-STOCK MODEL")
    print("=" * 60)

    # Load the trained multi-stock model
    model.load("models/xgb_multi_stock.pkl")
    print("  Loaded multi-stock model\n")

    # Backtest on a few individual stocks the model has seen
    test_tickers = ["AAPL", "MSFT", "NVDA", "JPM", "WMT"]

    for ticker in test_tickers:
        df_raw = collector.fetch_by_dates(ticker, start="2011-01-01")
        df     = feat.add_all_features(df_raw)
        df     = model.predict(df)   # adds 'confidence' column

        engine = BacktestEngine(starting_capital=100_000)
        results = engine.run(df, buy_threshold=0.55)

        print(f"\n{'=' * 60}")
        print(f"  {ticker}  (15-year backtest)")
        print(f"{'=' * 60}")
        if "error" in results:
            print(f"    {results['error']}")
            continue
        print(f"    Total return:    {results['total_return_pct']}%")
        print(f"    Final capital:   ${results['final_capital']:,}")
        print(f"    Total trades:    {results['total_trades']}")
        print(f"    Win rate:        {results['win_rate_pct']}%")
        print(f"    Avg win:         ${results['avg_win']:,}")
        print(f"    Avg loss:        ${results['avg_loss']:,}")
        print(f"    Profit factor:   {results['profit_factor']}")
        print(f"    Sharpe ratio:    {results['sharpe_ratio']}")
        print(f"    Max drawdown:    {results['max_drawdown_pct']}%")
        print(f"    Exits:           {results['exit_reasons']}")

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)