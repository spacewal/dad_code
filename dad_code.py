# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/162DI105K4RNb4KiKz_Q-qRUd1lUjoVtF
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit page configuration
st.set_page_config(page_title="S&P 500 Stock Analysis", layout="wide")

# Fetch the S&P 500 tickers and sectors
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_table = pd.read_html(url)
sp500_df = sp500_table[0]
sp500_df['Symbol'] = sp500_df['Symbol'].str.replace('.', '-')

# Select only the necessary columns
sp500_df = sp500_df[['Symbol', 'GICS Sector']]

# Define functions for financial indicators here

def calculate_ichimoku_cloud(df):
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    df['conversion_line'] = (df['High'].rolling(window=tenkan_period).max() + df['Low'].rolling(window=tenkan_period).min()) / 2
    df['base_line'] = (df['High'].rolling(window=kijun_period).max() + df['Low'].rolling(window=kijun_period).min()) / 2
    df['senkou_span_a'] = ((df['conversion_line'] + df['base_line']) / 2).shift(kijun_period)
    df['senkou_span_b'] = ((df['High'].rolling(window=senkou_span_b_period).max() + df['Low'].rolling(window=senkou_span_b_period).min()) / 2).shift(kijun_period)
    last_price = df['Close'].iloc[-1]
    span_a = df['senkou_span_a'].iloc[-1]
    span_b = df['senkou_span_b'].iloc[-1]
    cloud_status = "ABOVE CLOUD" if last_price >= span_a and last_price >= span_b else "NOT ABOVE CLOUD"
    return cloud_status

def calculate_awesome_oscillator(df):
    df['Midpoint'] = (df['High'] + df['Low']) / 2
    df['SMA_Short'] = df['Midpoint'].rolling(window=5).mean()
    df['SMA_Long'] = df['Midpoint'].rolling(window=34).mean()
    df['AO'] = df['SMA_Short'] - df['SMA_Long']
    return df['AO'].iloc[-1]

def calculate_vwap(df):
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap.iloc[-1]

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def calculate_macd(df):
    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

# Function to load and process data
st.cache_resource
def load_data(tickers):
    data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period="1y")
        if not hist_data.empty:
            stock_dict = {
                'Date': hist_data.index[-1],
                'Ticker': ticker,
                'Previous_Close': hist_data['Close'].iloc[-1],
                'Volume': hist_data['Volume'].iloc[-1],
                'Cloud_Status': calculate_ichimoku_cloud(hist_data),
                'AO': calculate_awesome_oscillator(hist_data),
                'VWAP': calculate_vwap(hist_data)
            }
            data.append(stock_dict)
    return pd.DataFrame(data)

# Load data
df_stocks = load_data(sp500_df['Symbol'])
df_stocks = df_stocks.merge(sp500_df, on='Symbol', how='left')

# Streamlit UI
st.title('S&P 500 Stock Analysis')
st.sidebar.header('Filter Options')

# Filters for sector and volume
sector_filter = st.sidebar.multiselect('Select Sector', options=sp500_df['GICS Sector'].unique())
if sector_filter:
    df_stocks = df_stocks[df_stocks['GICS Sector'].isin(sector_filter)]

volume_filter = st.sidebar.slider('Minimum Volume', int(df_stocks['Volume'].min()), int(df_stocks['Volume'].max()), int(df_stocks['Volume'].min()))
df_stocks = df_stocks[df_stocks['Volume'] >= volume_filter]

# Displaying the DataFrame
st.write(df_stocks)

# Optionally add plots
if st.sidebar.checkbox("Show Plots"):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, ax = plt.subplots()
    sns.countplot(data=df_stocks, x='GICS Sector')
    plt.xticks(rotation=45)
    st.pyplot(fig)

