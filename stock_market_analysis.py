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

# --- Setup ---
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# CUSTOM SESSION: This is the critical fix for the 429 error
@st.cache_resource
def get_yf_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    })
    return session

@st.cache_data
def load_stock_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        sp500 = pd.read_html(response.text)[0]
        return sp500["Symbol"].tolist()
    except:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# --- Data Fetching ---
def get_stock_data(ticker):
    session = get_yf_session()
    # Passing the session directly into the Ticker object
    stock = yf.Ticker(ticker, session=session)
    data = stock.history(period="5y")
    return data

def get_stock_fundamentals(ticker):
    session = get_yf_session()
    stock = yf.Ticker(ticker, session=session)
    info = stock.info
    return info.get("trailingPE"), info.get("forwardPE")

# --- Logic ---
def calculate_volatility(data):
    if data.empty: return 0
    return np.std(np.log(data["Close"] / data["Close"].shift(1))) * 100

def scrape_news(query):
    url = f"https://www.google.com/search?q={query}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")
        return [(a.text, f"https://www.google.com{a.find_parent('a')['href']}") for a in articles[:5]]
    except: return []

# --- AI Model ---
@st.cache_resource
def train_model(data_values):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data_values.reshape(-1, 1))
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model, scaler

# --- UI ---
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")
st.title("📈 AI-Powered Stock Market Analysis")

stocks = load_stock_list()
selected = st.sidebar.multiselect("Stocks", stocks, default=["AAPL", "MSFT"])
show_pred = st.sidebar.checkbox("Show AI Forecast", value=True)

if selected:
    fig = go.Figure()
    for s in selected:
        with st.spinner(f"Loading {s}..."):
            df = get_stock_data(s)
            if df.empty:
                st.error(f"Error: Yahoo Finance is blocking the request for {s}. Try again in 10 mins.")
                continue
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=s))
            
            if show_pred and len(df) > 60:
                m, sc = train_model(df["Close"].values)
                last_60 = sc.transform(df["Close"].values[-60:].reshape(-1, 1))
                p = sc.inverse_transform(m.predict(np.array([last_60])))
                tmrw = df.index[-1] + pd.Timedelta(days=1)
                fig.add_trace(go.Scatter(x=[df.index[-1], tmrw], y=[df["Close"].iloc[-1], p[0][0]], mode="lines+markers", name=f"{s} Forecast"))
    
    st.plotly_chart(fig, use_container_width=True)

    # Fundamentals
    v_data = []
    for s in selected:
        df = get_stock_data(s)
        pe, ipe = get_stock_fundamentals(s)
        v_data.append([s, f"{calculate_volatility(df):.2f}%", pe, ipe])
    st.table(pd.DataFrame(v_data, columns=["Stock", "Volatility", "P/E", "Forward P/E"]))

    # News
    st.subheader("📰 Market News")
    for txt, url in scrape_news("Stock Market"):
        st.markdown(f"📌 [{txt}]({url})")
