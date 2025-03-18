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
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from tensorflow.keras import Input
import plotly.express as px

nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Fetch stock list from Wikipedia
@st.cache_data
def load_stock_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500 = pd.read_html(url)[0]
    return sp500["Symbol"].tolist()

# Fetch stock data
def get_stock_data(ticker="AAPL"):
    stock = yf.Ticker(ticker)
    data = stock.history(period="30y")
    return data

# Fetch stock fundamentals
def get_stock_fundamentals(ticker="AAPL"):
    stock = yf.Ticker(ticker)
    info = stock.info
    pe_ratio = info.get("trailingPE", None)
    industry_pe = info.get("forwardPE", None)  # Fetch industry P/E dynamically
    return pe_ratio, industry_pe

# Check if stock is overpriced
def is_stock_overpriced(pe_ratio, industry_avg):
    if pe_ratio is None or industry_avg is None:
        return "Unknown (No Data)"
    return "✅ Fair" if pe_ratio < industry_avg else "🚨 Overpriced"

# Stock volatility calculation (log returns)
def calculate_volatility(data):
    return np.std(np.log(data["Close"] / data["Close"].shift(1))) * 100

# Scrape news headlines for sentiment analysis
def scrape_google_news(query="Stock Market"):
    url = f"https://www.google.com/search?q={query}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")

    news_results = []
    for article in articles[:5]:
        title = article.text
        parent = article.find_parent("a")
        link = f"https://www.google.com{parent['href']}" if parent and "href" in parent.attrs else "No Link Available"
        news_results.append((title, link))
    
    return news_results if news_results else []

# Sentiment analysis of financial news
def analyze_sentiment(news_list):
    sentiment_scores = [sia.polarity_scores(news[0])["compound"] for news in news_list]
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    return avg_sentiment

# Train LSTM model for stock price prediction
def train_lstm_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(60, len(scaled_data) - 1):
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
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    return model, scaler

# Streamlit UI
st.set_page_config(page_title="📈 AI Stock Market Dashboard", layout="wide")
st.title("📈 AI-Powered Stock Market Analysis")

# Sidebar Selection
st.sidebar.header("🔎 Select Stock(s)")
all_stocks = load_stock_list()
selected_stocks = st.sidebar.multiselect("Choose stocks", all_stocks, default=["AAPL", "MSFT", "GOOGL"])
show_predictions = st.sidebar.checkbox("Show AI Predictions", value=True)

# Graph Visualization
st.subheader(f"AI-Powered Stock Analysis for {', '.join(selected_stocks)}")
fig = go.Figure()

for stock in selected_stocks:
    stock_data = get_stock_data(stock)
    
    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name=f"{stock} Historical"
    ))

    # Moving Averages for trend clarity
    stock_data["MA_50"] = stock_data["Close"].rolling(window=50).mean()
    stock_data["MA_200"] = stock_data["Close"].rolling(window=200).mean()

    fig.add_trace(go.Scatter(
        x=stock_data.index, y=stock_data["MA_50"],
        mode="lines", name=f"{stock} 50-Day MA",
        line=dict(color="orange", dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=stock_data.index, y=stock_data["MA_200"],
        mode="lines", name=f"{stock} 200-Day MA",
        line=dict(color="purple", dash="dot")
    ))

    # LSTM Predictions
    model, scaler = train_lstm_model(stock_data)
    scaled_data = scaler.transform(stock_data["Close"].values.reshape(-1, 1))
    X_test = np.array([scaled_data[-60:].flatten()])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predictions = scaler.inverse_transform(model.predict(X_test))

    future_dates = pd.date_range(stock_data.index[-1], periods=8)[1:]
    fig.add_trace(go.Scatter(
        x=future_dates, y=predictions.flatten(),
        mode="lines", name=f"{stock} Forecast",
        line=dict(color="blue", dash="dash")
    ))

st.plotly_chart(fig, use_container_width=True, height=700)

# Stock Valuation Table
st.subheader("📊 Stock Volatility & Overpricing Analysis")
valuation_data = []

for stock in selected_stocks:
    stock_data = get_stock_data(stock)
    volatility = calculate_volatility(stock_data)
    pe_ratio, industry_avg = get_stock_fundamentals(stock)
    pricing_status = is_stock_overpriced(pe_ratio, industry_avg)

    valuation_data.append([stock, f"{volatility:.2f}%", pe_ratio, industry_avg, pricing_status])

df_valuation = pd.DataFrame(valuation_data, columns=["Stock", "Volatility", "P/E Ratio", "Industry P/E", "Status"])
st.dataframe(df_valuation)

# Market Sentiment
st.header("📰 Market-Wide Financial News")
market_news = scrape_google_news("Stock Market Today")
for news, url in market_news:
    st.markdown(f"📌 [{news}]({url})")

market_sentiment = analyze_sentiment(market_news)
sentiment_emoji = "🟢" if market_sentiment > 0 else "🔴" if market_sentiment < 0 else "🟡"
st.subheader(f"📢 AI Market Sentiment: {sentiment_emoji} {market_sentiment:.2f}")

# Trending Stocks
st.subheader("Live Trending Stocks")
trending_stocks = scrape_google_news("Trending Stocks")
for news, url in trending_stocks:
    st.markdown(f"📌 [{news}]({url})")
