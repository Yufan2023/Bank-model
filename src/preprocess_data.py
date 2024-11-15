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

    # Add sentiment analysis for political news
    political_news = [
        "New policy to boost economic growth signed into law",
        "Political unrest causes market uncertainty",
        "Election results indicate stable government",
        "Trade agreement reached between major countries"
    ]
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(news)['compound'] for news in political_news]
    sentiment_dates = pd.date_range(start='2023-01-01', periods=len(sentiment_scores), freq='ME')

    sentiment_df = pd.DataFrame({
        'Date': sentiment_dates,
        'Sentiment_Score': sentiment_scores
    })

    # Add political event indicators
    political_events = pd.DataFrame({
        'Date': pd.to_datetime([
            '2020-11-03',  # US Presidential Election
            '2021-01-20',  # Inauguration Day
            '2022-11-08',  # US Midterm Elections
            '2016-06-23',  # Brexit Referendum
            '2017-01-20',  # Inauguration Day
            '2019-12-12',  # UK General Election
            '2020-03-11',  # WHO declares COVID-19 a pandemic
            '2021-01-06',  # US Capitol Riot
            '2022-02-24'   # Start of Russia-Ukraine conflict
        ]),
        'Event_Indicator': 1
    })

    # Merge sentiment and event indicators with main data
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Date'}, inplace=True)
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)

    sentiment_df['Date'] = sentiment_df['Date'].dt.tz_localize(None)
    data = pd.merge(data, sentiment_df, on='Date', how='left')
    data['Sentiment_Score'].fillna(0, inplace=True)

    political_events['Date'] = political_events['Date'].dt.tz_localize(None)
    data = pd.merge(data, political_events, on='Date', how='left')
    data['Event_Indicator'].fillna(0, inplace=True)

    # Normalize all data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.drop(columns=['Date']))
    sequence_length = 60

    # Prepare data for LSTM
    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i, 1:])
        y.append(data_scaled[i, 0])  # Target: TD_Close

    return np.array(X), np.array(y), scaler, data
