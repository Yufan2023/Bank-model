import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X = data_scaled[:, 1:]  # All columns except the target
    y = data_scaled[:, 0]   # The target column
    X_lstm, y_lstm = [], []

    for i in range(sequence_length, len(X)):
        X_lstm.append(X[i-sequence_length:i])
        y_lstm.append(y[i])

    return np.array(X_lstm), np.array(y_lstm), scaler
