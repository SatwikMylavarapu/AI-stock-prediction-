import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from curl_cffi import requests as curlr
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

@st.cache_resource
def get_yf_session():
    return curlr.Session(impersonate="chrome")

@st.cache_data
def load_stock_list():
    # Scrapes the S&P 500 (which contains all Dow 30 companies)
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        response = curlr.get(url, impersonate="chrome")
        sp500 = pd.read_html(response.text)[0]
        return sorted(sp500["Symbol"].tolist())
    except:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "UNH", "V", "JPM", "HD"]

def get_stock_data(ticker):
    session = get_yf_session()
    stock = yf.Ticker(ticker, session=session)
    return stock.history(period="5y")

def get_stock_fundamentals(ticker):
    session = get_yf_session()
    stock = yf.Ticker(ticker, session=session)
    info = stock.info
    return info.get("trailingPE"), info.get("forwardPE")

# --- UI Layout ---
st.set_page_config(page_title="AI Market Analyzer", layout="wide")
st.title("📈 AI-Powered Stock Market Analysis")

# --- NEW: Hidden Explanation Section ---
with st.expander("ℹ️ How to understand this data (P/E Ratios Explained)"):
    st.markdown("""
    ### What is a P/E Ratio?
    The **Price-to-Earnings (P/E)** ratio tells you how much investors are willing to pay for every $1 of company profit.
    
    * **Trailing P/E (The Past):** Calculated using earnings from the **last 12 months**. It's based on hard facts but doesn't show where the company is going.
    * **Forward P/E (The Future):** Based on **predicted earnings** for the next 12 months. It's an estimate of growth.
    
    **Cheat Sheet:**
    * **Low (0-15):** Often "Value" stocks or companies in slow-growth industries. Could be a bargain or a sign of trouble.
    * **Average (16-25):** The standard range for healthy, established companies.
    * **High (25+):** "Growth" stocks. Investors expect big things in the future. High risk, but high potential.
    """)

# --- Sidebar ---
all_symbols = load_stock_list()
selected = st.sidebar.multiselect(
    "Select Stocks (S&P 500 & Dow Jones)", 
    all_symbols, 
    default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
)
show_pred = st.sidebar.checkbox("Show AI Forecast", value=True)

if selected:
    fig = go.Figure()
    metrics = []
    
    for s in selected:
        with st.spinner(f"Fetching {s}..."):
            df = get_stock_data(s)
            if df.empty:
                st.error(f"Could not retrieve data for {s}.")
                continue
            
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=s))
            
            pe, fpe = get_stock_fundamentals(s)
            metrics.append({
                "Ticker": s, 
                "Price": round(df['Close'].iloc[-1], 2),
                "Trailing P/E": pe, 
                "Forward P/E": fpe,
                "Status": "Growth" if (pe and pe > 25) else "Value"
            })
            
            if show_pred and len(df) > 60:
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
                last_60 = scaled[-60:].reshape(1, 60, 1)
                
                model = Sequential([Input(shape=(60, 1)), LSTM(50), Dense(1)])
                model.compile(optimizer='adam', loss='mse')
                pred_scaled = model.predict(last_60, verbose=0)
                p = scaler.inverse_transform(pred_scaled)
                
                tmrw = df.index[-1] + pd.Timedelta(days=1)
                fig.add_trace(go.Scatter(x=[df.index[-1], tmrw], y=[df["Close"].iloc[-1], p[0][0]], mode="lines+markers", name=f"{s} Forecast", line=dict(dash='dash')))
    
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("📊 Market Metrics Comparison")
    st.table(pd.DataFrame(metrics))
