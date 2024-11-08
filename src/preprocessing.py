from sklearn.preprocessing import MinMaxScaler
import numpy as np

def add_technical_indicators(data):
    data['TD_20_MA'] = data['TD_Close'].rolling(window=20).mean()
    data['TD_200_MA'] = data['TD_Close'].rolling(window=200).mean()
    data.dropna(inplace=True)
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def reshape_for_lstm(data_scaled, sequence_length=60):
    X = data_scaled[:, 1:]
    y = data_scaled[:, 0]
    X_lstm, y_lstm = [], []
    for i in range(sequence_length, len(X)):
        X_lstm.append(X[i-sequence_length:i])
        y_lstm.append(y[i])
    return np.array(X_lstm), np.array(y_lstm)
