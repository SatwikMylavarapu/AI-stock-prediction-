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
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# --- Configuration & Helpers ---

# Download NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def get_session():
    """Creates a session with retries and a browser-like User-Agent to avoid rate limits."""
    session = Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    })
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

# --- Data Functions ---

@st.cache_data
def load_stock_list():
    """Fetches S&P 500 list from Wikipedia with headers to avoid 403 Forbidden error."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        sp500 = pd.read_html(response.text)[0]
        return sp500["Symbol"].tolist()
    except Exception as e:
        st.error(f"Error fetching stock list: {e}")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

def get_stock_data(ticker):
    """Fetches historical data using a custom session to bypass YF rate limits."""
    session = get_session()
    stock = yf.Ticker(ticker, session=session)
    data = stock.history(period="5y")
    return data

def get_stock_fundamentals(ticker):
    """Fetches PE ratios using a custom session."""
    session = get_session()
    stock = yf.Ticker(ticker, session=session)
    info = stock.info
    pe_ratio = info.get("trailingPE", None)
    industry_pe = info.get("forwardPE", None)  
    return pe_ratio, industry_pe

def calculate_volatility(data):
    if data.empty: return 0
    return np.std(np.log(data["Close"] / data["Close"].shift(1))) * 100

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

def analyze_sentiment(news_list):
    if not news_list: return 0
    sentiment_scores = [sia.polarity_scores(news[0])["compound"] for news in news_list]
    return np.mean(sentiment_scores)

# --- AI Model ---

@st.cache_resource
def train_lstm_model(data_close_values):
    """Trains LSTM model; cached so it only runs once per stock data update."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close_values.reshape(-1, 1))

    X_train, y_train = [], []
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
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
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
    # Visualization
    fig = go.Figure()
    for stock in selected_stocks:
        with st.spinner(f"Fetching {stock}..."):
            stock_data = get_stock_data(stock)
            if stock_data.empty:
                st.error(f"Could not retrieve data for {stock}. You may be rate limited.")
                continue

            fig.add_trace(go.Candlestick(
                x=stock_data.index, open=stock_data['Open'],
                high=stock_data['High'], low=stock_data['Low'],
                close=stock_data['Close'], name=f"{stock} Historical"
            ))

            if show_predictions and len(stock_data) > 60:
                model, scaler = train_lstm_model(stock_data["Close"].values)
                last_60 = scaler.transform(stock_data["Close"].values[-60:].reshape(-1, 1))
                X_test = np.reshape(np.array([last_60]), (1, 60, 1))
                pred = scaler.inverse_transform(model.predict(X_test))

                tomorrow = stock_data.index[-1] + pd.Timedelta(days=1)
                fig.add_trace(go.Scatter(
                    x=[stock_data.index[-1], tomorrow], 
                    y=[stock_data["Close"].iloc[-1], pred[0][0]],
                    mode="lines+markers", name=f"{stock} Prediction",
                    line=dict(dash="dash", color="cyan")
                ))

    st.plotly_chart(fig, use_container_width=True)

    # Fundamentals Table
    st.subheader("📊 Volatility & Valuation")
    valuation_rows = []
    for stock in selected_stocks:
        data = get_stock_data(stock)
        vol = calculate_volatility(data)
        pe, ind_pe = get_stock_fundamentals(stock)
        status = "✅ Fair" if (pe and ind_pe and pe < ind_pe) else "🚨 Overpriced" if (pe and ind_pe) else "Unknown"
        valuation_rows.append([stock, f"{vol:.2f}%", pe, ind_pe, status])

    st.table(pd.DataFrame(valuation_rows, columns=["Stock", "Volatility", "P/E Ratio", "Industry P/E", "Status"]))

    # News Section
    st.header("📰 Market News & Sentiment")
    m_news = scrape_google_news("Stock Market Today")
    sent = analyze_sentiment(m_news)
    emoji = "🟢" if sent > 0.05 else "🔴" if sent < -0.05 else "🟡"
    st.metric("Overall Market Sentiment", f"{emoji} {sent:.2f}")

    for txt, link in m_news:
        st.markdown(f"📌 [{txt}]({link})")
