import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler


class XGBoostTrader:
    """
    Gradient-boosted decision trees that predict price direction.
    Outputs a confidence score (0-100%) that becomes your BUY/SELL signal.
    """

    def __init__(self, target_col: str = "target_5d"):
        self.target_col = target_col
        self.model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            eval_metric="logloss",
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_cols = [
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
        ]

    def train(self, df: pd.DataFrame) -> dict:
        """Train with time-series cross-validation (NO data leakage)."""
        X = df[self.feature_cols]
        y = df[self.target_col]

        tscv = TimeSeriesSplit(n_splits=5)
        aucs, accs = [], []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            X_tr_s  = self.scaler.fit_transform(X_tr)
            X_val_s = self.scaler.transform(X_val)

            self.model.fit(X_tr_s, y_tr)
            prob = self.model.predict_proba(X_val_s)[:, 1]
            pred = (prob > 0.5).astype(int)

            auc = roc_auc_score(y_val, prob)
            acc = accuracy_score(y_val, pred)
            aucs.append(auc)
            accs.append(acc)
            print(f"  Fold {fold}: AUC={auc:.3f}  Accuracy={acc:.3f}")

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        return {
            "cv_auc_mean":  round(np.mean(aucs), 4),
            "cv_auc_std":   round(np.std(aucs), 4),
            "cv_acc_mean":  round(np.mean(accs), 4),
            "n_samples":    len(X),
            "n_features":   len(self.feature_cols),
        }

    def train_multi_stock(self, tickers: list, collector, feat,
                          start: str = "2011-01-01") -> dict:
        """
        Train on many stocks at once. Stacks all their data into
        one big dataset so the model learns general patterns
        instead of memorizing one stock's quirks.
        """
        all_data = []
        print(f"  Building dataset from {len(tickers)} stocks...\n")

        for ticker in tickers:
            try:
                df_raw = collector.fetch_by_dates(ticker, start=start)
                if len(df_raw) < 250:
                    print(f"    {ticker}: skipped (only {len(df_raw)} rows)")
                    continue
                df_feat = feat.add_all_features(df_raw)
                all_data.append(df_feat)
                print(f"    {ticker}: {len(df_feat)} usable rows")
            except Exception as e:
                print(f"    {ticker}: ERROR - {e}")

        combined = pd.concat(all_data, ignore_index=True)
        print(f"\n  Combined dataset: {len(combined):,} total rows")

        print("\n  Training on combined data...\n")
        return self.train(combined)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate BUY/SELL/HOLD signals with confidence scores."""
        X = df[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        df = df.copy()
        df["confidence"] = self.model.predict_proba(X_scaled)[:, 1]
        df["signal"] = np.where(df["confidence"] > 0.60, "BUY",
                       np.where(df["confidence"] < 0.40, "SELL", "HOLD"))
        return df

    def predict_latest(self, df: pd.DataFrame) -> dict:
        """Get the signal for the most recent day."""
        result = self.predict(df)
        latest = result.iloc[-1]
        conf = latest["confidence"]
        return {
            "signal":     latest["signal"],
            "confidence": round(conf * 100, 1),
            "price":      round(latest["Close"], 2),
            "atr":        round(latest["atr_14"], 2),
        }

    def feature_importance(self, top_n: int = 10) -> list:
        """Which features matter most to the model's decisions."""
        importances = self.model.feature_importances_
        pairs = sorted(zip(self.feature_cols, importances),
                       key=lambda x: x[1], reverse=True)
        return pairs[:top_n]

    def save(self, path: str = "models/xgb_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "features": self.feature_cols,
                     "target": self.target_col}, path)
        print(f"  Model saved to {path}")

    def load(self, path: str = "models/xgb_model.pkl"):
        data = joblib.load(path)
        self.model        = data["model"]
        self.scaler       = data["scaler"]
        self.feature_cols = data["features"]
        self.target_col   = data["target"]


# --- TEST IT ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.collector import DataCollector
    from src.features.technical import TechnicalFeatures

    collector = DataCollector()
    feat      = TechnicalFeatures()
    model     = XGBoostTrader(target_col="target_5d")

    tickers = [
        # Big Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSLA",
        "INTC", "CSCO", "ORCL", "CRM", "ADBE", "QCOM", "TXN", "AVGO",
        # Finance
        "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "V", "MA",
        # Healthcare
        "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT",
        # Energy
        "XOM", "CVX", "COP",
        # Consumer
        "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "PG", "KO", "PEP",
        # Industrial / Other
        "BA", "CAT", "GE", "DIS", "NFLX",
        # --- MARKET DIRECTION / LEADING INDICATORS ---
        "SPY",   # S&P 500 - overall market
        "QQQ",   # Nasdaq 100 - tech-heavy
        "DIA",   # Dow Jones
        "IWM",   # Russell 2000 - small caps (risk appetite)
        "VXX",   # Volatility (fear gauge proxy)
        "TLT",   # 20yr Treasury bonds (flight to safety)
        "HYG",   # High-yield bonds (credit risk appetite)
        "GLD",   # Gold (safe haven)
        "XLF",   # Financial sector
        "XLK",   # Tech sector
        "XLE",   # Energy sector
        "SMH",   # Semiconductors (economic leading indicator)
    ]

    print("=" * 60)
    print(f"MULTI-STOCK TRAINING ({len(tickers)} stocks, 15 years each)")
    print("=" * 60)

    metrics = model.train_multi_stock(tickers, collector, feat)

    print(f"\n  RESULTS:")
    print(f"    CV AUC:      {metrics['cv_auc_mean']} (+/- {metrics['cv_auc_std']})")
    print(f"    CV Accuracy: {metrics['cv_acc_mean']}")
    print(f"    Samples:     {metrics['n_samples']:,}")
    print(f"    Features:    {metrics['n_features']}")

    print("\n" + "=" * 60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 60)
    for feature, importance in model.feature_importance(10):
        bar = "#" * int(importance * 100)
        print(f"  {feature:18s} {importance:.3f} {bar}")

    print("\n" + "=" * 60)
    print("SAVING MULTI-STOCK MODEL")
    print("=" * 60)
    model.save("models/xgb_multi_stock.pkl")

    print("\nMULTI-STOCK MODEL FULLY WORKING")