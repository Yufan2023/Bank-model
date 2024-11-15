import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def preprocess_data():
    # Define tickers
    tickers = {
        "TD_Close": "TD.TO",
        "BMO_Close": "BMO.TO",
        "RBC_Close": "RY.TO",
        "BNS_Close": "BNS.TO",
        "Financials_Close": "XLF"
    }
    economic_tickers = {
        "Interest_Rates": "^IRX",
        "Volatility_Index": "^VIX",
        "Bank_Index": "^BKX",
        "SP500": "^GSPC"
    }

    start_date = "2015-01-01"
    end_date = "2024-01-01"

    # Download stock and economic data
    data = pd.DataFrame()
    for column_name, ticker in tickers.items():
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        data[column_name] = stock_data['Close']

    for column_name, ticker in economic_tickers.items():
        econ_data = yf.download(ticker, start=start_date, end=end_date)
        data[column_name] = econ_data['Close']

    # Add technical indicators
    data['TD_20_MA'] = data['TD_Close'].rolling(window=20).mean()
    data['TD_200_MA'] = data['TD_Close'].rolling(window=200).mean()

    # Sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [0.5] * len(data)  # Placeholder if real sentiment data isn't available
    data['Sentiment_Score'] = sentiment_scores

    # Fill missing values and normalize
    data.dropna(inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Prepare data for LSTM
    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i, 1:])
        y.append(data_scaled[i, 0])  # Target: TD_Close

    return np.array(X), np.array(y), scaler, data
