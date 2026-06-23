import sys
sys.path.insert(0, ".")
import talib
import yfinance as yf
import pandas as pd

# All 61 TA-Lib candlestick pattern functions
PATTERNS = [f for f in dir(talib) if f.startswith("CDL")]

# Friendly names for the common ones (rest show their TA-Lib name)
NICE = {
    "CDLDOJI": "Doji (indecision)",
    "CDLHAMMER": "Hammer (bullish reversal)",
    "CDLSHOOTINGSTAR": "Shooting Star (bearish reversal)",
    "CDLENGULFING": "Engulfing",
    "CDLMORNINGSTAR": "Morning Star (bullish reversal)",
    "CDLEVENINGSTAR": "Evening Star (bearish reversal)",
    "CDL3WHITESOLDIERS": "Three White Soldiers (bullish)",
    "CDL3BLACKCROWS": "Three Black Crows (bearish)",
    "CDLHARAMI": "Harami",
    "CDLPIERCING": "Piercing Line (bullish)",
    "CDLDARKCLOUDCOVER": "Dark Cloud Cover (bearish)",
    "CDLMARUBOZU": "Marubozu (strong move)",
}


def explore(ticker, days=180):
    print(f"\n{'='*55}")
    print(f"  CANDLESTICK PATTERNS for {ticker}")
    print(f"{'='*55}")
    df = yf.download(ticker, period=f"{days}d", progress=False)
    if df.empty:
        print("  No data - check the ticker symbol.")
        return
    # Flatten columns if yfinance returns multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)

    # --- Patterns on the most recent candle ---
    print(f"\n  TODAY'S CANDLE ({df.index[-1].date()}):")
    found_today = False
    for p in PATTERNS:
        result = getattr(talib, p)(o, h, l, c)
        val = result[-1]
        if val != 0:
            name = NICE.get(p, p.replace("CDL", ""))
            direction = "bullish" if val > 0 else "bearish"
            print(f"    - {name}  ({direction})")
            found_today = True
    if not found_today:
        print("    None of the 61 patterns detected on today's candle.")

    # --- Patterns over the last 10 days ---
    print(f"\n  LAST 10 DAYS (any pattern that fired):")
    found_recent = False
    for i in range(max(0, len(c) - 10), len(c)):
        date = df.index[i].date()
        for p in PATTERNS:
            result = getattr(talib, p)(o, h, l, c)
            if result[i] != 0:
                name = NICE.get(p, p.replace("CDL", ""))
                direction = "bullish" if result[i] > 0 else "bearish"
                print(f"    {date}: {name} ({direction})")
                found_recent = True
    if not found_recent:
        print("    No patterns in the last 10 days.")
    print()


if __name__ == "__main__":
    ticker = input("Enter a ticker (e.g. AAPL): ").upper().strip() or "AAPL"
    explore(ticker)