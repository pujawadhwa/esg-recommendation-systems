import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from time import sleep

# Manually defined ESG scores for demonstration
esg_scores = {
    'AAPL': {'environment': 70, 'social': 60, 'governance': 80},
    'MSFT': {'environment': 75, 'social': 65, 'governance': 85},
    'GOOGL': {'environment': 65, 'social': 70, 'governance': 75},
    'AMZN': {'environment': 60, 'social': 55, 'governance': 70},
    'TSLA': {'environment': 80, 'social': 75, 'governance': 65}
}

# Function to get ESG data (using manual data for demonstration)
def get_esg_data(ticker):
    return esg_scores.get(ticker, {'environment': 0, 'social': 0, 'governance': 0})

# Function to get stock price data with error handling and retries
def get_stock_data(ticker, start, end, retries=3, delay=5):
    for _ in range(retries):
        try:
            stock = yf.download(ticker, start=start, end=end)
            if not stock.empty:
                return stock
        except Exception as e:
            st.warning(f"Failed to download data for {ticker}: {e}")
            sleep(delay)
    return None

# Weighting ESG scores
def calculate_weighted_esg(esg_scores, weights):
    weighted_score = (esg_scores['environment'] * weights['environment'] +
                      esg_scores['social'] * weights['social'] +
                      esg_scores['governance'] * weights['governance'])
    return weighted_score

# LSTM-based stock price prediction
def create_lstm_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(tf.keras.layers.LSTM(50, return_sequences=False))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_stock_prices(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    train_data_len = int(np.ceil(len(scaled_data) * 0.8))

    train_data = scaled_data[0:int(train_data_len), :]

    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = create_lstm_model()
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[train_data_len - 60:, :]
    x_test, y_test = [], scaled_data[train_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions

# Define the tickers and date range
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2018-01-01'
end_date = '2023-12-31'

# Collect stock price data with error handling
stock_data = {ticker: get_stock_data(ticker, start_date, end_date) for ticker in tickers}
stock_data = {k: v for k, v in stock_data.items() if v is not None}

if not stock_data:
    st.error("Failed to download stock data for all tickers. Please try again later.")
else:
    # User inputs for weighting
    st.title("ESG-Based Stock Recommendation System")

    env_weight = st.slider('Environmental Weight', 0.0, 1.0, 0.5)
    soc_weight = st.slider('Social Weight', 0.0, 1.0, 0.3)
    gov_weight = st.slider('Governance Weight', 0.0, 1.0, 0.2)

    # Update weights
    weights = {'environment': env_weight, 'social': soc_weight, 'governance': gov_weight}

    # Recalculate weighted ESG scores
    weighted_esg_scores = {ticker: calculate_weighted_esg(get_esg_data(ticker), weights) for ticker in tickers}

    # Display recommendations
    st.write("Stock Recommendations based on Weighted ESG Scores:")
    for ticker, score in weighted_esg_scores.items():
        st.write(f"{ticker}: {score}")

    # Visualize stock trends
    st.write("Stock Price Trends:")
    fig, ax = plt.subplots(figsize=(14, 7))
    for ticker in stock_data:
        sns.lineplot(ax=ax, x=stock_data[ticker].index, y=stock_data[ticker]['Close'], label=ticker)
    st.pyplot(fig)

    # Visualize LSTM predictions
    st.write("LSTM Stock Price Predictions:")
    for ticker in stock_data:
        predictions = predict_stock_prices(stock_data[ticker])
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(stock_data[ticker].index, stock_data[ticker]['Close'], label='Actual Prices')
        ax.plot(stock_data[ticker].index[-len(predictions):], predictions, label='Predicted Prices', color='red')
        ax.set_title(f'LSTM Stock Price Prediction: {ticker}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)
