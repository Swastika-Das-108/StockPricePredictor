import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# ---------------- DOWNLOAD STOCK DATA ----------------
ticker = "AAPL"

df = yf.download(
    ticker,
    period="10y",
    auto_adjust=True
)

# ---------------- TECHNICAL INDICATORS ----------------

# Moving averages
df["MA10"] = df["Close"].rolling(10).mean()
df["MA50"] = df["Close"].rolling(50).mean()

# Exponential moving averages
df["EMA20"] = df["Close"].ewm(span=20).mean()
df["EMA50"] = df["Close"].ewm(span=50).mean()

# Daily return
df["Daily_Return"] = df["Close"].pct_change()

# Volatility
df["Volatility"] = (
    df["Close"]
    .rolling(10)
    .std()
)

# RSI
delta = df["Close"].diff()

gain = (
    delta.where(delta > 0, 0)
    .rolling(14)
    .mean()
)

loss = (
    -delta.where(delta < 0, 0)
    .rolling(14)
    .mean()
)

rs = gain / loss

df["RSI"] = (
    100 - (100 / (1 + rs))
)

# MACD
ema12 = df["Close"].ewm(span=12).mean()
ema26 = df["Close"].ewm(span=26).mean()

df["MACD"] = ema12 - ema26

# Trend strength
df["Trend"] = (
    df["MA10"] - df["MA50"]
)

# Remove empty rows
df.dropna(inplace=True)

# ---------------- TARGETS ----------------
df["Tomorrow"] = (
    df["Close"].shift(-1)
)

df["7_Days"] = (
    df["Close"].shift(-7)
)

df["30_Days"] = (
    df["Close"].shift(-30)
)

df.dropna(inplace=True)

# ---------------- FEATURES ----------------
features = [
    "Close",
    "Volume",
    "MA10",
    "MA50",
    "EMA20",
    "EMA50",
    "Daily_Return",
    "Volatility",
    "RSI",
    "MACD",
    "Trend"
]

X = df[features]

y_tomorrow = df["Tomorrow"]
y_7 = df["7_Days"]
y_30 = df["30_Days"]

# ---------------- TRAIN FUNCTION ----------------
def train_model(y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(X_test)

    accuracy = r2_score(
        y_test,
        pred
    )

    return model, accuracy

# ---------------- TRAIN ----------------
model_tomorrow, acc_t = train_model(
    y_tomorrow
)

model_7, acc_7 = train_model(
    y_7
)

model_30, acc_30 = train_model(
    y_30
)

# ---------------- SAVE ----------------
joblib.dump(
    model_tomorrow,
    "model_tomorrow.pkl"
)

joblib.dump(
    model_7,
    "model_7.pkl"
)

joblib.dump(
    model_30,
    "model_30.pkl"
)

joblib.dump(
    {
        "Tomorrow": acc_t,
        "7 Days": acc_7,
        "30 Days": acc_30
    },
    "accuracy.pkl"
)

print("✅ Advanced AI models trained")

print(
    f"Tomorrow Accuracy: {acc_t:.2f}"
)

print(
    f"7 Days Accuracy: {acc_7:.2f}"
)

print(
    f"30 Days Accuracy: {acc_30:.2f}"
)