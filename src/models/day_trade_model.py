import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from src.data.collector import DataCollector
from src.features.technical import TechnicalFeatures


class DayTradeModel:
    """
    XGBoost model for INTRADAY day trading on 5-minute bars.
    Predicts whether price rises >0.2% in the next ~1 hour (12 bars).
    NOTE: Trained on only ~60 days of data (Yahoo's intraday limit),
    so expect a THINNER, less reliable edge than the swing model.
    """

    def __init__(self, target_col: str = "target_12bar"):
        self.target_col = target_col
        self.model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            eval_metric="logloss",
            random_state=42,
        )
        self.scaler = StandardScaler()
        # Same features as swing model, minus the daily-specific ones
        self.feature_cols = [
            "rsi_14", "rsi_7", "rsi_overbought", "rsi_oversold",
            "macd", "macd_signal", "macd_hist", "macd_cross",
            "bb_width", "bb_pct",
            "volume_ratio",
            "atr_14", "atr_pct",
            "stoch_k", "stoch_d",
            "daily_return", "hl_range",
            "return_lag_1", "return_lag_2", "return_lag_3",
            "rsi_lag_1", "rsi_lag_2",
            "body_pct", "upper_wick_pct", "lower_wick_pct", "is_green",
            "doji", "hammer", "shooting_star", "bull_engulf", "bear_engulf",
        ]

    def train(self, df: pd.DataFrame) -> dict:
        X = df[self.feature_cols]
        y = df[self.target_col]

        tscv = TimeSeriesSplit(n_splits=5)
        aucs, accs = [], []
        for fold, (tr, va) in enumerate(tscv.split(X), 1):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y.iloc[tr], y.iloc[va]
            X_tr_s = self.scaler.fit_transform(X_tr)
            X_va_s = self.scaler.transform(X_va)
            self.model.fit(X_tr_s, y_tr)
            prob = self.model.predict_proba(X_va_s)[:, 1]
            aucs.append(roc_auc_score(y_va, prob))
            accs.append(accuracy_score(y_va, (prob > 0.5).astype(int)))
            print(f"  Fold {fold}: AUC={aucs[-1]:.3f}  Acc={accs[-1]:.3f}")

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return {
            "cv_auc_mean": round(np.mean(aucs), 4),
            "cv_auc_std":  round(np.std(aucs), 4),
            "n_samples":   len(X),
        }

    def train_multi_stock(self, tickers, collector, feat, interval="5m") -> dict:
        all_data = []
        print(f"  Building intraday dataset from {len(tickers)} stocks...\n")
        for ticker in tickers:
            try:
                df_raw = collector.fetch_intraday(ticker, interval=interval, days=60)
                if len(df_raw) < 500:
                    print(f"    {ticker}: skipped ({len(df_raw)} bars)")
                    continue
                df_feat = feat.add_all_features(df_raw)
                all_data.append(df_feat)
                print(f"    {ticker}: {len(df_feat)} bars")
            except Exception as e:
                print(f"    {ticker}: ERROR - {e}")
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\n  Combined: {len(combined):,} bars")
        return self.train(combined)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        df = df.copy()
        df["confidence"] = self.model.predict_proba(X_scaled)[:, 1]
        df["signal"] = np.where(df["confidence"] > 0.58, "BUY",
                       np.where(df["confidence"] < 0.42, "SELL", "HOLD"))
        return df

    def save(self, path: str = "models/day_trade_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "features": self.feature_cols, "target": self.target_col}, path)
        print(f"  Saved to {path}")

    def load(self, path: str = "models/day_trade_model.pkl"):
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_cols = data["features"]
        self.target_col = data["target"]


# --- TRAIN IT ---
if __name__ == "__main__":
    collector = DataCollector()
    feat = TechnicalFeatures()
    model = DayTradeModel(target_col="target_12bar")

    # Liquid, high-volume stocks are best for day trading
    tickers = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN",
               "GOOGL", "SPY", "QQQ", "AMC", "INTC", "BAC", "F"]

    print("=" * 60)
    print("DAY-TRADE MODEL TRAINING (5-min bars, ~1hr ahead)")
    print("=" * 60)
    metrics = model.train_multi_stock(tickers, collector, feat)

    print(f"\n  RESULTS:")
    print(f"    CV AUC:    {metrics['cv_auc_mean']} (+/- {metrics['cv_auc_std']})")
    print(f"    Samples:   {metrics['n_samples']:,}")

    model.save()
    print("\n  Reminder: intraday edge is thin (only ~60 days data).")
    print("  Expect AUC near 0.50-0.54. Treat day-trade signals with extra caution.")