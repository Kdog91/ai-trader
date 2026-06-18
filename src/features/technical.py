import pandas as pd
import numpy as np
import ta

class TechnicalFeatures:
    """
    Transforms raw OHLCV data into AI-readable features.
    Includes technical indicators, candlestick patterns, and
    market regime features (trending vs choppy).
    """

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        # ── TREND INDICATORS ──────────────────────────────
        df["ema_9"]   = ta.trend.ema_indicator(close, window=9)
        df["ema_21"]  = ta.trend.ema_indicator(close, window=21)
        df["ema_50"]  = ta.trend.ema_indicator(close, window=50)
        df["ema_200"] = ta.trend.ema_indicator(close, window=200)
        df["sma_20"]  = ta.trend.sma_indicator(close, window=20)

        df["above_ema200"]  = (close > df["ema_200"]).astype(int)
        df["ema9_cross_21"] = (df["ema_9"] > df["ema_21"]).astype(int)

        # ── MACD ──────────────────────────────────────────
        macd_ind          = ta.trend.MACD(close)
        df["macd"]        = macd_ind.macd()
        df["macd_signal"] = macd_ind.macd_signal()
        df["macd_hist"]   = macd_ind.macd_diff()
        df["macd_cross"]  = (df["macd"] > df["macd_signal"]).astype(int)

        # ── RSI ───────────────────────────────────────────
        df["rsi_14"]      = ta.momentum.rsi(close, window=14)
        df["rsi_7"]       = ta.momentum.rsi(close, window=7)
        df["rsi_overbought"]  = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"]    = (df["rsi_14"] < 30).astype(int)

        # ── BOLLINGER BANDS ───────────────────────────────
        bb                = ta.volatility.BollingerBands(close, window=20)
        df["bb_upper"]    = bb.bollinger_hband()
        df["bb_lower"]    = bb.bollinger_lband()
        df["bb_mid"]      = bb.bollinger_mavg()
        df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"]      = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ── VOLUME INDICATORS ─────────────────────────────
        df["volume_sma"]   = ta.trend.sma_indicator(volume, window=20)
        df["volume_ratio"] = volume / df["volume_sma"]
        df["obv"]          = ta.volume.on_balance_volume(close, volume)

        # ── VOLATILITY ────────────────────────────────────
        df["atr_14"]   = ta.volatility.average_true_range(high, low, close, window=14)
        df["atr_pct"]  = df["atr_14"] / close

        # ── STOCHASTIC ────────────────────────────────────
        stoch          = ta.momentum.StochasticOscillator(high, low, close)
        df["stoch_k"]  = stoch.stoch()
        df["stoch_d"]  = stoch.stoch_signal()

        # ── PRICE ACTION ──────────────────────────────────
        df["daily_return"]  = close.pct_change()
        df["hl_range"]      = (high - low) / close
        df["gap_pct"]       = (df["Open"] - close.shift(1)) / close.shift(1)

        # ── LAG FEATURES (memory for the AI) ─────────────
        for lag in [1, 2, 3, 5]:
            df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)
            df[f"rsi_lag_{lag}"]    = df["rsi_14"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume_ratio"].shift(lag)

        # ── CANDLESTICK PATTERN FEATURES ──────────────────
        body       = (close - df["Open"]).abs()
        range_hl   = (high - low).replace(0, np.nan)
        upper_wick = high - df[["Close", "Open"]].max(axis=1)
        lower_wick = df[["Close", "Open"]].min(axis=1) - low

        df["body_pct"]       = body / range_hl
        df["upper_wick_pct"] = upper_wick / range_hl
        df["lower_wick_pct"] = lower_wick / range_hl
        df["is_green"]       = (close > df["Open"]).astype(int)

        # Doji: tiny body (indecision)
        df["doji"] = (body / range_hl < 0.1).astype(int)

        # Hammer: small body up top, long lower wick (bullish reversal)
        df["hammer"] = (
            (lower_wick > body * 2) & (upper_wick < body)
        ).astype(int)

        # Shooting star: small body at bottom, long upper wick (bearish reversal)
        df["shooting_star"] = (
            (upper_wick > body * 2) & (lower_wick < body)
        ).astype(int)

        # Bullish engulfing
        prev_red    = close.shift(1) < df["Open"].shift(1)
        today_green = close > df["Open"]
        df["bull_engulf"] = (
            prev_red & today_green &
            (close > df["Open"].shift(1)) &
            (df["Open"] < close.shift(1))
        ).astype(int)

        # Bearish engulfing
        prev_green = close.shift(1) > df["Open"].shift(1)
        today_red  = close < df["Open"]
        df["bear_engulf"] = (
            prev_green & today_red &
            (df["Open"] > close.shift(1)) &
            (close < df["Open"].shift(1))
        ).astype(int)

        # ── REGIME FEATURES (trending vs choppy) ──────────
        df["adx_14"] = ta.trend.adx(high, low, close, window=14)
        df["strong_trend"] = (df["adx_14"] > 25).astype(int)
        df["choppy"]       = (df["adx_14"] < 20).astype(int)

        sma_50 = ta.trend.sma_indicator(close, window=50)
        df["dist_from_sma50"] = (close - sma_50) / sma_50

        df["vol_20"] = df["daily_return"].rolling(20).std()
        df["vol_regime"] = (df["vol_20"] > df["vol_20"].rolling(60).mean()).astype(int)

        df["trend_10"] = (close > close.shift(10)).astype(int)
        df["trend_20"] = (close > close.shift(20)).astype(int)

        # ── TARGET LABELS (what we're predicting) ─────────
        for horizon in [1, 3, 5, 10]:
            future = close.shift(-horizon) / close - 1
            df[f"target_{horizon}d"] = (future > 0.005).astype(int)

       # ── INTRADAY TARGETS (for day trading on 5-min bars) ──
        # 3 bars=15min, 6 bars=30min, 12 bars=1hr ahead
        for bars in [3, 6, 12]:
            future = close.shift(-bars) / close - 1
            df[f"target_{bars}bar"] = (future > 0.002).astype(int)  # >0.2% move     

        df.dropna(inplace=True)
        return df

    def get_feature_columns(self) -> list:
        """Returns the exact feature list the ML model trains on."""
        return [
            "rsi_14", "rsi_7", "rsi_overbought", "rsi_oversold",
            "macd", "macd_signal", "macd_hist", "macd_cross",
            "bb_width", "bb_pct",
            "volume_ratio", "obv",
            "atr_14", "atr_pct",
            "stoch_k", "stoch_d",
            "daily_return", "hl_range", "gap_pct",
            "above_ema200", "ema9_cross_21",
            "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
            "rsi_lag_1", "rsi_lag_2",
            "volume_lag_1", "volume_lag_2",
            "body_pct", "upper_wick_pct", "lower_wick_pct", "is_green",
            "doji", "hammer", "shooting_star", "bull_engulf", "bear_engulf",
            "adx_14", "strong_trend", "choppy", "dist_from_sma50",
            "vol_20", "vol_regime", "trend_10", "trend_20",
        ]


# --- TEST IT ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.collector import DataCollector

    collector = DataCollector()
    feat      = TechnicalFeatures()

    print("=" * 50)
    print("TEST: Adding features to AAPL data")
    print("=" * 50)
    df_raw = collector.fetch_daily("AAPL", period="2y")
    df     = feat.add_all_features(df_raw)

    print(f"  Raw rows:      {len(df_raw)}")
    print(f"  After features: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Model features: {len(feat.get_feature_columns())}")

    print("\n  New candlestick + regime features (latest values):")
    for col in ["body_pct", "doji", "hammer", "bull_engulf",
                "adx_14", "strong_trend", "choppy", "dist_from_sma50"]:
        print(f"    {col:18s}: {df[col].iloc[-1]:.4f}")

    print("\nTECHNICAL FEATURES FULLY WORKING")