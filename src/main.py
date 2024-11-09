import pandas as pd
from src.data_loader import download_stock_data, download_economic_data
from src.preprocessing import add_technical_indicators, preprocess_data, reshape_for_lstm
from src.model import build_lstm_model, train_model, save_model
from src.sentiment_analysis import analyze_sentiment, merge_sentiment_with_data
from src.utils import plot_actual_vs_predicted, plot_future_predictions

# Define constants
start_date = "2015-01-01"
end_date = "2024-01-01"
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

# Load and preprocess data
data = download_stock_data(tickers, start_date, end_date)
data = download_economic_data(data, economic_tickers, start_date, end_date)
data = add_technical_indicators(data)

# Integrate sentiment analysis
news_headlines = [
    "New policy to boost economic growth signed into law",
    "Political unrest causes market uncertainty",
    "Election results indicate stable government",
    "Trade agreement reached between major countries"
]
sentiment_df = analyze_sentiment(news_headlines)
data = merge_sentiment_with_data(data, sentiment_df)

# Preprocess data and reshape
data_scaled, scaler = preprocess_data(data)
X_lstm, y_lstm = reshape_for_lstm(data_scaled)

# Split data
train_size = int(0.8 * len(X_lstm))
X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]

# Build, train, and save the model
model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
history = train_model(model, X_train, y_train, X_test, y_test)
save_model(model, 'models/lstm_model.h5')
