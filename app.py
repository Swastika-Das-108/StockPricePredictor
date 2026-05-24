import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from textblob import TextBlob

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Trading Assistant",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Predictor")

# =========================
# SESSION STATE
# =========================
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

# =========================
# DATA LOADER (FIXED)
# =========================
def get_data(symbol, period):
    try:
        df = yf.download(symbol, period=period, auto_adjust=False)

        if df is None or df.empty:
            return None

        # Fix MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Reset index → creates Date column
        df = df.reset_index()

        # Force correct Date column
        if "Date" not in df.columns:
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)

        # Convert to datetime (IMPORTANT)
        df["Date"] = pd.to_datetime(df["Date"])

        return df

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


# =========================
# RSI
# =========================
def rsi(df, period=14):
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()

    rs = gain / loss
    rsi_value = 100 - (100 / (1 + rs))

    return rsi_value.fillna(50)


# =========================
# SENTIMENT
# =========================
def sentiment(symbol):
    texts = [
        f"{symbol} shows strong performance",
        f"{symbol} has volatility risk",
        f"{symbol} remains stable"
    ]

    scores = [TextBlob(t).sentiment.polarity for t in texts]
    return sum(scores) / len(scores)


# =========================
# AI SIGNAL ENGINE
# =========================
def ai_signal(df, symbol):
    df = df.copy()
    df["RSI"] = rsi(df)
    df = df.dropna()

    if len(df) < 50:
        return "🟡 HOLD", 50

    latest_rsi = float(df["RSI"].iloc[-1])

    ma20 = float(df["Close"].rolling(20).mean().dropna().iloc[-1])
    ma50 = float(df["Close"].rolling(50).mean().dropna().iloc[-1])

    sent = sentiment(symbol)

    score = 50

    # RSI logic
    if latest_rsi < 30:
        score += 20
    elif latest_rsi > 70:
        score -= 20

    # Moving averages
    if ma20 > ma50:
        score += 15
    else:
        score -= 15

    # Sentiment
    if sent > 0:
        score += 15
    else:
        score -= 15

    score = max(0, min(100, score))

    if score >= 70:
        return "🟢 BUY", score
    elif score <= 40:
        return "🔴 SELL", score
    else:
        return "🟡 HOLD", score


# =========================
# CHART (FIXED)
# =========================
def plot_chart(df, symbol):

    required_cols = ["Date", "Open", "High", "Low", "Close"]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.write("Available columns:", df.columns.tolist())
        return

    df = df.sort_values("Date")

    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )])

    fig.update_layout(
        template="plotly_dark",
        title=f"{symbol} Stock Chart",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# PORTFOLIO FUNCTIONS
# =========================
def add_stock(symbol, shares, buy_price):

    for stock in st.session_state.portfolio:
        if stock["symbol"] == symbol:
            st.warning("Stock already exists in portfolio")
            return

    st.session_state.portfolio.append({
        "symbol": symbol,
        "shares": shares,
        "buy_price": buy_price
    })


def show_portfolio():

    st.subheader("💼 Portfolio Tracker")

    if not st.session_state.portfolio:
        st.info("No stocks added yet")
        return

    rows = []

    for stock in st.session_state.portfolio:

        data = yf.Ticker(stock["symbol"]).history(period="1d")

        if data.empty:
            continue

        current_price = float(data["Close"].dropna().iloc[-1])

        profit = (current_price - stock["buy_price"]) * stock["shares"]

        rows.append([
            stock["symbol"],
            stock["shares"],
            stock["buy_price"],
            round(current_price, 2),
            round(profit, 2)
        ])

    if rows:
        df_port = pd.DataFrame(
            rows,
            columns=["Symbol", "Shares", "Buy Price", "Current Price", "P/L"]
        )

        st.dataframe(df_port, use_container_width=True)


# =========================
# INPUTS
# =========================
symbol = st.text_input("Stock Symbol", "AAPL")

period = st.selectbox(
    "Timeframe",
    ["1mo", "3mo", "6mo", "1y"],
    index=3
)

# =========================
# MAIN ANALYSIS
# =========================
if st.button("Run Analysis"):

    df = get_data(symbol, period)

    if df is None:
        st.error("No data found")

    else:
        st.success("Analysis Complete")

        signal, score = ai_signal(df, symbol)

        st.subheader("📊 AI Signal")
        st.write(f"Signal: {signal}")
        st.write(f"Confidence Score: {score}/100")

        # Price movement
        if len(df) >= 2:
            change = ((df["Close"].iloc[-1] - df["Close"].iloc[-2]) /
                      df["Close"].iloc[-2]) * 100

            st.subheader("🔔 Price Alert")

            if change > 3:
                st.success(f"🚀 Price Jump +{change:.2f}%")
            elif change < -3:
                st.error(f"🔻 Price Drop {change:.2f}%")
            else:
                st.info(f"Normal movement {change:.2f}%")

        st.subheader("📉 Chart")
        plot_chart(df, symbol)


# =========================
# PORTFOLIO UI
# =========================
st.divider()

st.subheader("➕ Add to Portfolio")

col1, col2, col3 = st.columns(3)

with col1:
    p_symbol = st.text_input("Symbol", key="p_symbol")

with col2:
    shares = st.number_input("Shares", min_value=1, value=1)

with col3:
    buy_price = st.number_input("Buy Price", min_value=0.0, value=100.0)

if st.button("Add Stock"):
    if p_symbol.strip():
        add_stock(p_symbol.upper(), shares, buy_price)
        st.success("Stock added!")

show_portfolio()
