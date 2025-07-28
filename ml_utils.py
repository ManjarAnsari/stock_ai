import pandas as pd
import numpy as np
import joblib
import os

# --- Constants ---
DEFAULT_MODEL_DIR = "models"

# ✅ Shared feature list (Used in both training and prediction)
features = [
    'rsi', 'macd', 'macd_signal',
    'bb_upper', 'bb_middle', 'bb_lower',
    'ema_9', 'ema_21', 'sma_20',
    'volume_change', 'returns'
]

# --- Model Loader (Global Model Support) ---
def load_global_model(model_path=os.path.join(DEFAULT_MODEL_DIR, "global_model.pkl")):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"⚠️ Global model not found at {model_path}. Train and save the model first!")

# --- Feature Engineering ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_sma(series, period):
    return series.rolling(window=period).mean()

def generate_features(df):
    df = df.copy()
    df['rsi'] = compute_rsi(df['Close'])
    df['macd'], df['macd_signal'] = compute_macd(df['Close'])
    df['ema_9'] = compute_ema(df['Close'], 9)
    df['ema_21'] = compute_ema(df['Close'], 21)
    df['sma_20'] = compute_sma(df['Close'], 20)  # Ensure 'Close' is a Series
    df['stddev_20'] = df['Close'].rolling(window=20).std()  # Assign std to its own column
    df['bb_upper'] = df['sma_20'] + (2 * df['stddev_20'])  # Use column-wise operations
    df['bb_middle'] = df['sma_20']
    df['bb_lower'] = df['sma_20'] - (2 * df['stddev_20'])

    df['volume_change'] = df['Volume'].pct_change()
    df['returns'] = df['Close'].pct_change()

    df.dropna(inplace=True)
    return df

# --- Label Generation for Training ---
def label_data(df):
    df = df.copy()
    close = df['Close'].astype(float)
    future_close = close.shift(-1)
    price_diff = future_close - close
    df['future_close'] = future_close
    df['price_diff'] = price_diff
    df['signal'] = df['price_diff'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df.dropna(inplace=True)
    return df

# --- Signal Prediction ---
def generate_signal(X, model):
    if model is None:
        raise ValueError("Model not loaded. Train and save the model first.")
    preds = model.predict(X)
    reasons = ["📈 Buy" if p == 1 else "📉 Sell" if p == -1 else "⏸️ Hold" for p in preds]
    return preds, reasons

def predict_signals(df, model=None):
    if model is None:
        model = load_global_model()
    X = df[features]
    df['signal'], df['reason'] = generate_signal(X, model)
    return df

# --- Entry/Exit Strategy ---
def get_entry_exit(preds, prices):
    entries, exits = [], []
    position = False
    for i, signal in enumerate(preds):
        if signal == 1 and not position:
            entries.append(prices[i])
            exits.append(None)
            position = True
        elif signal == -1 and position:
            entries.append(None)
            exits.append(prices[i])
            position = False
        else:
            entries.append(None)
            exits.append(None)
    return entries, exits