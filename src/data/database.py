import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/database/trading.db")

class Database:
    """Saves and loads stock data from local SQLite database."""

    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(DB_PATH)
        self._init_tables()

    def _init_tables(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker    TEXT    NOT NULL,
                date      TEXT    NOT NULL,
                open      REAL,
                high      REAL,
                low       REAL,
                close     REAL,
                volume    INTEGER,
                UNIQUE(ticker, date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker     TEXT,
                date       TEXT,
                signal     TEXT,
                confidence REAL,
                model      TEXT,
                entry_price REAL,
                stop_loss   REAL,
                target      REAL,
                created_at  TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT,
                entry_date  TEXT,
                exit_date   TEXT,
                entry_price REAL,
                exit_price  REAL,
                shares      INTEGER,
                pnl         REAL,
                pnl_pct     REAL,
                status      TEXT DEFAULT 'open'
            )
        """)
        conn.commit()
        conn.close()

    def save_ohlcv(self, df: pd.DataFrame, ticker: str):
        """Save OHLCV dataframe to database."""
        conn = sqlite3.connect(self.db_path)
        rows_saved = 0
        for date, row in df.iterrows():
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO ohlcv
                    (ticker, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    str(date.date()),
                    round(float(row["Open"]),  4),
                    round(float(row["High"]),  4),
                    round(float(row["Low"]),   4),
                    round(float(row["Close"]), 4),
                    int(row["Volume"]),
                ))
                rows_saved += 1
            except Exception as e:
                print(f"  Row error {date}: {e}")
        conn.commit()
        conn.close()
        return rows_saved

    def load_ohlcv(self, ticker: str, limit: int = None) -> pd.DataFrame:
        """Load OHLCV data for a ticker from database."""
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT date, open, high, low, close, volume
            FROM ohlcv WHERE ticker = '{ticker}'
            ORDER BY date ASC
            {f'LIMIT {limit}' if limit else ''}
        """
        df = pd.read_sql(query, conn, parse_dates=["date"], index_col="date")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        conn.close()
        return df

    def get_all_tickers(self) -> list:
        """Get list of all tickers in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT DISTINCT ticker FROM ohlcv ORDER BY ticker")
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tickers

    def get_row_count(self, ticker: str = None) -> int:
        """Get total rows in database."""
        conn = sqlite3.connect(self.db_path)
        if ticker:
            count = conn.execute(
                "SELECT COUNT(*) FROM ohlcv WHERE ticker=?", (ticker,)
            ).fetchone()[0]
        else:
            count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        conn.close()
        return count

    def save_signal(self, ticker: str, signal: str, confidence: float,
                    model: str, entry_price: float,
                    stop_loss: float, target: float):
        """Save a BUY/SELL signal to database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO signals
            (ticker, date, signal, confidence, model, entry_price, stop_loss, target)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker,
            datetime.now().strftime("%Y-%m-%d"),
            signal, confidence, model,
            entry_price, stop_loss, target
        ))
        conn.commit()
        conn.close()

    def log_prediction(self, ticker, signal, confidence, price, model_type):
        """Save today's prediction to check later."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                ticker TEXT,
                signal TEXT,
                confidence REAL,
                price_at_signal REAL,
                model_type TEXT,
                checked INTEGER DEFAULT 0,
                price_after REAL,
                correct INTEGER,
                UNIQUE(date, ticker, model_type)
            )
        """)
        conn.execute("""
            INSERT OR IGNORE INTO predictions
            (date, ticker, signal, confidence, price_at_signal, model_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now().strftime("%Y-%m-%d"), ticker, signal,
              confidence, price, model_type))
        conn.commit()
        conn.close()

    def get_unchecked_predictions(self, days_old: int = 5):
        """Get predictions waiting to be verified."""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql("SELECT * FROM predictions WHERE checked = 0", conn)
        except Exception:
            df = pd.DataFrame()
        conn.close()
        return df

    def mark_checked(self, pred_id, price_after, correct):
        """Mark a prediction as checked with the result."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE predictions SET checked=1, price_after=?, correct=?
            WHERE id=?
        """, (price_after, correct, pred_id))
        conn.commit()
        conn.close()

    def get_scorecard(self, model_type: str = None):
        """Get the track record of all checked predictions."""
        conn = sqlite3.connect(self.db_path)
        q = "SELECT * FROM predictions WHERE checked = 1"
        if model_type:
            q += f" AND model_type = '{model_type}'"
        try:
            df = pd.read_sql(q, conn)
        except Exception:
            df = pd.DataFrame()
        conn.close()
        return df


# --- TEST IT ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.collector import DataCollector

    db        = Database()
    collector = DataCollector()

    print("=" * 50)
    print("TEST 1: Saving AAPL to database")
    print("=" * 50)
    df = collector.fetch_daily("AAPL", period="2y")
    saved = db.save_ohlcv(df, "AAPL")
    print(f"  Saved {saved} rows for AAPL")

    print("\n" + "=" * 50)
    print("TEST 2: Logging a test prediction")
    print("=" * 50)
    db.log_prediction("AAPL", "BUY", 62.5, 295.95, "swing")
    print("  Prediction logged successfully")

    print("\n  DATABASE FULLY WORKING (with prediction logging)")