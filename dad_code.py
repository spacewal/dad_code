import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit page setup
st.title('S&P 500 Stock Analysis')

# Fetch S&P 500 tickers and sectors from Wikipedia
st.cache_resource
def load_sp500_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)
    sp500_df = sp500_table[0]
    sp500_df['Symbol'] = sp500_df['Symbol'].str.replace('.', '-')
    sp500_df = sp500_df[['Symbol', 'GICS Sector']]
    return sp500_df

sp500_df = load_sp500_data()
tickers = sp500_df['Symbol'].tolist()

# Select ticker input
selected_ticker = st.selectbox('Select a ticker:', tickers)

# Functions

# Define the Ichimoku Cloud calculation function
def calculate_ichimoku_cloud(df):
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52

    df['conversion_line'] = (df['High'].rolling(window=tenkan_period).max() + df['Low'].rolling(window=tenkan_period).min()) / 2
    df['base_line'] = (df['High'].rolling(window=kijun_period).max() + df['Low'].rolling(window=kijun_period).min()) / 2
    df['senkou_span_a'] = ((df['conversion_line'] + df['base_line']) / 2).shift(kijun_period)
    df['senkou_span_b'] = ((df['High'].rolling(window=senkou_span_b_period).max() + df['Low'].rolling(window=senkou_span_b_period).min()) / 2).shift(kijun_period)

    # Assuming 'last_price' is the last 'Close' price from the historical data
    last_price = df['Close'].iloc[-1]
    span_a = df['senkou_span_a'].iloc[-1]
    span_b = df['senkou_span_b'].iloc[-1]

    # Check if the last price is above the Ichimoku Cloud
    cloud_status = "ABOVE CLOUD" if last_price >= span_a and last_price >= span_b else "NOT ABOVE CLOUD"
    return cloud_status

def calculate_awesome_oscillator(df, short_period=5, long_period=34):
    # Calculate the midpoint ((High + Low) / 2) of each bar
    df['Midpoint'] = (df['High'] + df['Low']) / 2

    # Calculate the short and long period simple moving averages (SMAs) of the midpoints
    df['SMA_Short'] = df['Midpoint'].rolling(window=short_period).mean()
    df['SMA_Long'] = df['Midpoint'].rolling(window=long_period).mean()

    # Calculate the Awesome Oscillator as the difference between the short and long period SMAs
    df['AO'] = df['SMA_Short'] - df['SMA_Long']

    # Return the last value of the Awesome Oscillator series
    return df['AO'].iloc[-1]


# Define the interpretation functions
def interpret_ao(ao_value):
    return "BULLISH" if ao_value >= 0 else "BEARISH"

def interpret_ao_movement(current_ao, previous_ao):
    if current_ao >= 0 and previous_ao < current_ao:
        return "BULLISH_INCREASING"
    elif current_ao >= 0 and previous_ao > current_ao:
        return "BULLISH_DECREASING"
    elif current_ao < 0 and previous_ao < current_ao:
        return "BEARISH_INCREASING"
    elif current_ao < 0 and previous_ao > current_ao:
        return "BEARISH_DECREASING"
    return "STABLE"  # If current and previous AO values are the same

# Define the VWAP calculation function
def calculate_vwap(df):
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap.iloc[-1]  # Return only the last value


# Define the function to calculate EMA using pandas 'ewm' method
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# Define a function to evaluate conditions for each EMA and assign labels
def evaluate_ema_conditions(row):
    labels = {}
    # Check conditions for each EMA
    for ema in ['EMA_21', 'EMA_36', 'EMA_50', 'EMA_95', 'EMA_200']:
        if row[ema] >= max([row[e] for e in ['EMA_50', 'EMA_95', 'EMA_200'] if e != ema]):
            labels[ema] = "BULL"
        elif row[ema] < row['EMA_36'] and row[ema] > max([row[e] for e in ['EMA_50', 'EMA_200'] if e != ema]):
            labels[ema] = "BULL"
        elif row[ema] < row['EMA_36'] and row[ema] < row['EMA_21'] and row[ema] > max([row[e] for e in ['EMA_95', 'EMA_200'] if e != ema]):
            labels[ema] = "BULL"
        elif row[ema] < row['EMA_21'] and row[ema] < row['EMA_36'] and row[ema] < row['EMA_50'] and row[ema] > row['EMA_200']:
            labels[ema] = "BULL"
        elif row[ema] < row['EMA_21'] and row[ema] < row['EMA_36'] and row[ema] < row['EMA_50'] and row[ema] < row['EMA_95']:
            labels[ema] = "BULL"
        else:
            labels[ema] = "BEAR"
    return labels

# Define the function to calculate smoothed RSI
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))

    return RSI

# Define the function to calculate traditional RSI
def calculate_rsi_trad(data, period=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    average_gain = gain.rolling(window=period).mean()
    average_loss = loss.rolling(window=period).mean()

    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Define the cahold function
def cahold(previous_close, latest_price):
    return "BULLISH" if latest_price >= previous_close else "BEARISH"

# Define the function to calculate MACD
def calculate_macd(df, slow_period=26, fast_period=12, signal_period=9):
    # Calculate the short-term EMA (fast period)
    ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()

    # Calculate the long-term EMA (slow period)
    ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()

    # Calculate the MACD line
    macd_line = ema_fast - ema_slow

    # Calculate the Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    return macd_line, signal_line

# Define the function to calculate returns
def calculate_returns(df):
    return df['Close'].pct_change().dropna()

# Create dataframe

# Load ticker data from yfinance
st.cache_resource
def load_ticker_data(ticker):
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period="1y")
    return hist_data

hist_data = load_ticker_data(selected_ticker)

# Calculate indicators
if not hist_data.empty:
    ichimoku_status = calculate_ichimoku_cloud(hist_data)
    ao_status = calculate_awesome_oscillator(hist_data)
    # (calculate other indicators here)
    st.write("Ichimoku Cloud Status:", ichimoku_status)
    st.write("Awesome Oscillator Status:", ao_status)
    # (display other indicators)

# Display historical data chart
st.line_chart(hist_data[['Close']])

# Additional settings and configurations
# (any additional Streamlit components you want to add)

# Run this with: streamlit run your_script_name.py

# Initialize an empty list to store the data
data = []

# Correct the loop to iterate over ticker symbols from the 'Symbol' column of sp500_df
for ticker in sp500_df['Symbol']:  # Make sure to access the 'Symbol' column
    # Fetch the ticker data
    stock = yf.Ticker(ticker)

    # Get the historical data for the ticker
    hist_data = stock.history(period="1y")

    # Check if there's sufficient data to process
    if not hist_data.empty and len(hist_data) > 1:
        # Calculate returns
        hist_data['Returns'] = calculate_returns(hist_data)

        # Calculate other indicators and append results
        cloud_status = calculate_ichimoku_cloud(hist_data)
        ao_value = calculate_awesome_oscillator(hist_data)
        ao_movement = interpret_ao_movement(ao_value, hist_data['AO'].iloc[-2] if len(hist_data['AO']) >= 2 else None)
        vwap_value = calculate_vwap(hist_data)

        # Additional financial calculations
        for window in [21, 36, 50, 95, 200]:
            hist_data[f'EMA_{window}'] = calculate_ema(hist_data['Close'], span=window)

        macd_line, signal_line = calculate_macd(hist_data)
        rsi_smoothed = calculate_rsi(hist_data['Close'])
        rsi_trad = calculate_rsi_trad(hist_data['Close'])
        cahold_value = cahold(hist_data['Close'].iloc[-2], hist_data['Close'].iloc[-1]) if len(hist_data) >= 2 else None

        # Create a dictionary for the current ticker's data
        stock_dict = {
            'Date': hist_data.index[-1],
            'Returns': hist_data['Returns'].iloc[-1],
            'Ticker': ticker,
            'Previous_Close': hist_data['Close'].iloc[-1],
            'Volume': hist_data['Volume'].iloc[-1],
            'Cloud_Status': cloud_status,
            'Awesome_Oscillator': ao_value,
            'AO_Interpretation': interpret_ao(ao_value),
            'AO_Movement': ao_movement,
            'VWAP': vwap_value,
            'RSI_Smoothed': rsi_smoothed.iloc[-1],
            'RSI_Trad': rsi_trad.iloc[-1],
            'Cahold_Status': cahold_value,
            'MACD': macd_line.iloc[-1],
            'Signal_Line': signal_line.iloc[-1],
            'EMA_Labels': evaluate_ema_conditions(hist_data.iloc[-1])  # Ensure this method fits your DataFrame structure
        }

        # Append the dictionary to the data list
        data.append(stock_dict)

# Convert the list of dictionaries into a DataFrame
df_stocks = pd.DataFrame(data)

# Merge your existing stock DataFrame with the sector data
df_stocks = df_stocks.merge(sp500_df, on='Symbol', how='left')

# Filter stocks with volume greater than 1 million
df_filtered = df_stocks[df_stocks['Volume'] > 1000000]

# Creating the Streamlit application
st.title('S&P 500 Stock Data Viewer')

# Adding a sidebar for configuration
st.sidebar.header('Filter Settings')

# Adding filters to the sidebar
selected_ticker = st.sidebar.multiselect('Select Ticker', options=df_filtered['Ticker'].unique(), default=df_filtered['Ticker'].unique())
selected_sector = st.sidebar.multiselect('Select Sector', options=df_filtered['GICS Sector'].unique(), default=df_filtered['GICS Sector'].unique())

# Numeric filters for Volume and Returns
min_volume = st.sidebar.slider('Minimum Volume', int(df_filtered['Volume'].min()), int(df_filtered['Volume'].max()), int(df_filtered['Volume'].min()))
max_volume = st.sidebar.slider('Maximum Volume', int(df_filtered['Volume'].min()), int(df_filtered['Volume'].max()), int(df_filtered['Volume'].max()))

min_returns = st.sidebar.slider('Minimum Returns', float(df_filtered['Returns'].min()), float(df_filtered['Returns'].max()), float(df_filtered['Returns'].min()))
max_returns = st.sidebar.slider('Maximum Returns', float(df_filtered['Returns'].min()), float(df_filtered['Returns'].max()), float(df_filtered['Returns'].max()))

# Filters for Awesome Oscillator interpretation and Cloud Status
ao_options = st.sidebar.multiselect('Awesome Oscillator Interpretation', options=df_filtered['AO_Interpretation'].unique(), default=df_filtered['AO_Interpretation'].unique())
cloud_status_options = st.sidebar.multiselect('Cloud Status', options=df_filtered['Cloud_Status'].unique(), default=df_filtered['Cloud_Status'].unique())

# Filter data based on selection
query = (df_filtered['Ticker'].isin(selected_ticker) & 
         df_filtered['GICS Sector'].isin(selected_sector) & 
         (df_filtered['Volume'] >= min_volume) & (df_filtered['Volume'] <= max_volume) & 
         (df_filtered['Returns'] >= min_returns) & (df_filtered['Returns'] <= max_returns) & 
         df_filtered['AO_Interpretation'].isin(ao_options) & 
         df_filtered['Cloud_Status'].isin(cloud_status_options))

filtered_data = df_filtered[query]

# Displaying the DataFrame
st.write(filtered_data)
