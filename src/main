import pandas as pd
from src.data_loader import download_data
from src.preprocessing import preprocess_data
from src.model import build_model
from src.sentiment_analysis import analyze_sentiment

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

# Load and preprocess data
data = download_data(tickers, start_date, end_date)
X_lstm, y_lstm, scaler = preprocess_data(data)

# Split data into training and testing sets
train_size = int(0.8 * len(X_lstm))
X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]

# Build and train the model
model = build_model((X_train.shape[1], X_train.shape[2]))
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Extend with prediction and plotting logic as needed
