import yfinance as yf
import pandas as pd
from datetime import datetime

class DataCollector:
    """Fetches live and historical stock data from Yahoo Finance."""

    def __init__(self):
        self.watchlist = [
            "AAPL", "TSLA", "NVDA", "AMD", "SPY",
            "QQQ", "MSFT", "AMZN", "META", "GOOGL", "AMC"
        ]

    def fetch_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Fetch daily OHLCV data for a ticker."""
        print(f"  Fetching {ticker} daily data...")
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df["ticker"] = ticker
        df.dropna(inplace=True)
        return df

    def fetch_by_dates(self, ticker: str, start: str, end: str = None) -> pd.DataFrame:
        """
        Fetch data between exact dates. Format: 'YYYY-MM-DD'.
        Example: fetch_by_dates('AAPL', '2011-01-01') pulls ~15 years.
        """
        print(f"  Fetching {ticker} from {start} to {end or 'today'}...")
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df["ticker"] = ticker
        df.dropna(inplace=True)
        return df

    def fetch_intraday(self, ticker: str, interval: str = "5m",
                       days: int = 60) -> pd.DataFrame:
        """
        Fetch intraday data for day trading.
        interval options: '1m','2m','5m','15m','30m','60m','90m'
        NOTE: Yahoo limits free intraday data to ~60 days max.
              1m data is limited to ~7 days.
        """
        period = f"{min(days, 60)}d"
        print(f"  Fetching {ticker} intraday ({interval}, {period})...")
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df["ticker"] = ticker
        df.dropna(inplace=True)
        return df

    def fetch_multiple(self, tickers: list = None, period: str = "2y") -> dict:
        """Fetch multiple tickers at once."""
        tickers = tickers or self.watchlist
        data = {}
        print(f"\nFetching data for {len(tickers)} stocks...")
        for ticker in tickers:
            try:
                data[ticker] = self.fetch_daily(ticker, period)
                print(f"  {ticker}: {len(data[ticker])} rows fetched")
            except Exception as e:
                print(f"  {ticker}: ERROR - {e}")
        return data

    def get_latest_price(self, ticker: str) -> dict:
        """Get the most recent price info for a ticker."""
        stock = yf.Ticker(ticker)
        info  = stock.info
        return {
            "ticker":        ticker,
            "price":         info.get("currentPrice") or info.get("regularMarketPrice"),
            "change_pct":    info.get("regularMarketChangePercent"),
            "volume":        info.get("regularMarketVolume"),
            "avg_volume":    info.get("averageVolume"),
            "52w_high":      info.get("fiftyTwoWeekHigh"),
            "52w_low":       info.get("fiftyTwoWeekLow"),
            "market_cap":    info.get("marketCap"),
            "fetched_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


# --- TEST IT ---
if __name__ == "__main__":
    collector = DataCollector()

    # Test 1: Fetch ~15 years of AAPL daily data
    print("=" * 50)
    print("TEST 1: Fetching AAPL daily data (15 years)")
    print("=" * 50)
    df = collector.fetch_by_dates("AAPL", start="2011-01-01")
    print(df.tail())
    print(f"\nTotal rows: {len(df)}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Columns: {list(df.columns)}")

    # Test 2: Latest price
    print("\n" + "=" * 50)
    print("TEST 2: Latest AAPL price info")
    print("=" * 50)
    price = collector.get_latest_price("AAPL")
    for k, v in price.items():
        print(f"  {k}: {v}")

    # Test 3: Intraday 5-minute data for day trading
    print("\n" + "=" * 50)
    print("TEST 3: Fetching AAPL intraday (5-min bars)")
    print("=" * 50)
    df_intra = collector.fetch_intraday("AAPL", interval="5m", days=60)
    print(df_intra.tail())
    print(f"\nTotal intraday rows: {len(df_intra)}")
    if len(df_intra) > 0:
        print(f"Date range: {df_intra.index[0]} to {df_intra.index[-1]}")