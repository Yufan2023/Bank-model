import numpy as np

def predict_future_prices(model, X_test, y_test, scaler):
    # Predict historical prices
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(
        np.concatenate((y_pred_scaled, np.zeros((y_pred_scaled.shape[0], X_test.shape[2]))), axis=1)
    )[:, 0]

    y_test_actual = scaler.inverse_transform(
        np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X_test.shape[2]))), axis=1)
    )[:, 0]

    # Predict future prices
    future_days = 30
    last_sequence = X_test[-1]
    predicted_future = []

    for _ in range(future_days):
        next_pred = model.predict(last_sequence.reshape(1, X_test.shape[1], X_test.shape[2]))
        predicted_future.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], [next_pred], axis=0)

    predicted_future = scaler.inverse_transform(
        np.concatenate((np.array(predicted_future).reshape(-1, 1), np.zeros((len(predicted_future), X_test.shape[2]))), axis=1)
    )[:, 0]

    return y_pred, y_test_actual, predicted_future
