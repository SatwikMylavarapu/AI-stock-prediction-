import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px

# Download NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# FIX: Fetch stock list with Headers to avoid HTTP 403 error
@st.cache_data
def load_stock_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        sp500 = pd.read_html(response.text)[0]
        return sp500["Symbol"].tolist()
    except Exception as e:
        st.error(f"Error fetching stock list: {e}")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="5y") # Reduced from 30y to 5y for faster LSTM training
    return data

# Fetch stock fundamentals
def get_stock_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    pe_ratio = info.get("trailingPE", None)
    # Using forwardPE as a proxy for industry average if not available
    industry_pe = info.get("forwardPE", None)  
    return pe_ratio, industry_pe

# Check if stock is overpriced
def is_stock_overpriced(pe_ratio, industry_avg):
    if pe_ratio is None or industry_avg is None:
        return "Unknown (No Data)"
    return "✅ Fair" if pe_ratio < industry_avg else "🚨 Overpriced"

# Stock volatility calculation
def calculate_volatility(data):
    if data.empty: return 0
    return np.std(np.log(data["Close"] / data["Close"].shift(1))) * 100

# Scrape news headlines
def scrape_google_news(query="Stock Market"):
    url = f"https://www.google.com/search?q={query}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")

        news_results = []
        for article in articles[:5]:
            title = article.text
            parent = article.find_parent("a")
            link = f"https://www.google.com{parent['href']}" if parent and "href" in parent.attrs else "#"
            news_results.append((title, link))
        return news_results
    except:
        return []

# Sentiment analysis
def analyze_sentiment(news_list):
    if not news_list: return 0
    sentiment_scores = [sia.polarity_scores(news[0])["compound"] for news in news_list]
    return np.mean(sentiment_scores)

# OPTIMIZED: Cache the model training so it doesn't run on every UI interaction
@st.cache_resource
def train_lstm_model(data_close_values):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close_values.reshape(-1, 1))

    X_train, y_train = [], []
    # Using 60 days of lookback
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0) # Reduced epochs for speed
    
    return model, scaler

# --- Streamlit UI ---
st.set_page_config(page_title="📈 AI Stock Market Dashboard", layout="wide")
st.title("📈 AI-Powered Stock Market Analysis")

# Sidebar
st.sidebar.header("🔎 Select Stock(s)")
all_stocks = load_stock_list()
selected_stocks = st.sidebar.multiselect("Choose stocks", all_stocks, default=["AAPL", "MSFT"])
show_predictions = st.sidebar.checkbox("Show AI Predictions", value=True)

if not selected_stocks:
    st.warning("Please select at least one stock from the sidebar.")
else:
    # Graph Visualization
    st.subheader(f"Analysis for: {', '.join(selected_stocks)}")
    fig = go.Figure()

    for stock in selected_stocks:
        with st.spinner(f"Processing {stock}..."):
            stock_data = get_stock_data(stock)
            if stock_data.empty:
                st.error(f"No data found for {stock}")
                continue

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=stock_data.index, open=stock_data['Open'],
                high=stock_data['High'], low=stock_data['Low'],
                close=stock_data['Close'], name=f"{stock} Hist"
            ))

            # MAs
            stock_data["MA_50"] = stock_data["Close"].rolling(window=50).mean()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["MA_50"], name=f"{stock} 50-Day MA", line=dict(width=1)))

            # Predictions
            if show_predictions and len(stock_data) > 60:
                model, scaler = train_lstm_model(stock_data["Close"].values)
                
                # Get the last 60 days to predict the next day
                last_60_days = stock_data["Close"].values[-60:].reshape(-1, 1)
                last_60_days_scaled = scaler.transform(last_60_days)
                
                X_test = np.array([last_60_days_scaled])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                
                pred_price_scaled = model.predict(X_test)
                pred_price = scaler.inverse_transform(pred_price_scaled)

                # Add a point for tomorrow's forecast
                tomorrow = stock_data.index[-1] + pd.Timedelta(days=1)
                fig.add_trace(go.Scatter(
                    x=[stock_data.index[-1], tomorrow], 
                    y=[stock_data["Close"].iloc[-1], pred_price[0][0]],
                    mode="lines+markers", name=f"{stock} Forecast",
                    line=dict(dash="dash", color="cyan")
                ))

    st.plotly_chart(fig, use_container_width=True)

    # Volatility Table
    st.subheader("📊 Volatility & Valuation")
    valuation_rows = []
    for stock in selected_stocks:
        data = get_stock_data(stock)
        vol = calculate_volatility(data)
        pe, ind_pe = get_stock_fundamentals(stock)
        status = is_stock_overpriced(pe, ind_pe)
        valuation_rows.append([stock, f"{vol:.2f}%", pe, ind_pe, status])

    df_val = pd.DataFrame(valuation_rows, columns=["Stock", "Volatility", "P/E Ratio", "Industry P/E", "Status"])
    st.table(df_val)

    # News & Sentiment
    col1, col2 = st.columns(2)
    with col1:
        st.header("📰 Market News")
        m_news = scrape_google_news("Stock Market Today")
        for txt, link in m_news:
            st.markdown(f"📌 [{txt}]({link})")
    
    with col2:
        st.header("📢 Sentiment")
        sent = analyze_sentiment(m_news)
        emoji = "🟢" if sent > 0.05 else "🔴" if sent < -0.05 else "🟡"
        st.metric("AI Market Sentiment", f"{emoji} {sent:.2f}")

    st.subheader("🔥 Trending")
    t_news = scrape_google_news("Trending Stocks")
    for txt, link in t_news:
        st.caption(f"🔥 [{txt}]({link})")
