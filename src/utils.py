import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_test_actual, y_pred):
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_actual, color='blue', label='Actual TD Close Price')
    plt.plot(y_pred, color='red', label='Predicted TD Close Price (LSTM)')
    plt.xlabel('Days')
    plt.ylabel('TD Bank Close Price')
    plt.title('TD Bank Stock Price Prediction with Political Dynamics')
    plt.legend()
    plt.show()

def plot_future_predictions(y_test_actual, future_days, predicted_future_scaled):
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(y_test_actual)), y_test_actual, color='blue', label='Actual TD Close Price')
    plt.plot(range(len(y_test_actual), len(y_test_actual) + future_days), predicted_future_scaled, color='green', label='Predicted Future TD Close Price')
    plt.xlabel('Days')
    plt.ylabel('TD Bank Close Price')
    plt.title('TD Bank Future Stock Price Prediction')
    plt.legend()
    plt.show()
