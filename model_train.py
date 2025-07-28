# model_train.py

import os
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from ml_utils import generate_features, label_data

# --- Config ---
symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "INFY.NS", "TCS.NS"]  # Add more if needed
interval = "1d"
period = "6mo"
model_dir = "models"
global_model_path = os.path.join(model_dir, "global_model.pkl")

from sklearn.ensemble import RandomForestClassifier

# --- Data Fetcher ---
def fetch_stock_data(symbol, interval="1d", period="6mo"):
    data = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    data.dropna(inplace=True)
    return data

# --- Combine Data ---
all_data = []

print("Fetching and preparing data...")

for symbol in symbols:
    try:
        df = fetch_stock_data(symbol, interval=interval, period=period)
        df = generate_features(df)
        df = label_data(df)
        df["symbol"] = symbol  # Optional: Track source symbol
        all_data.append(df)
        print(f"✅ Processed: {symbol}")
    except Exception as e:
        print(f"❌ Failed: {symbol} | {e}")

# --- Concatenate and Train ---
combined_df = pd.concat(all_data)
X = combined_df[['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower', 'ema_9', 'ema_21', 'sma_20', 'volume_change', 'returns']]
y = combined_df['signal']
X = X.astype(np.float64)  # ✅ convert to float64 here
# Remove rows with inf or NaN in X or y
mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[mask]
y = y[mask]
print("Training global model...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

# --- Save Model ---
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, global_model_path)
print(f"🎯 Global model saved at {global_model_path}")