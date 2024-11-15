from src.preprocess_data import preprocess_data
from src.train_model import train_lstm_model
from src.predict_future import predict_future_prices
import matplotlib.pyplot as plt

# Step 1: Preprocess the data
X, y, scaler, data = preprocess_data()

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 2: Train the model
model, history = train_lstm_model(X_train, y_train, X_test, y_test)

# Step 3: Predict prices
y_pred, y_test_actual, predicted_future = predict_future_prices(model, X_test, y_test, scaler)

# Step 4: Visualize results
plt.figure(figsize=(14, 7))
plt.plot(y_test_actual, color='blue', label='Actual TD Close Price')
plt.plot(y_pred, color='red', label='Predicted TD Close Price')
plt.legend()
plt.title("TD Bank Stock Price Prediction with Political Dynamics")
plt.show()

# Future Predictions Plot
plt.figure(figsize=(14, 7))
plt.plot(range(len(y_test_actual)), y_test_actual, label="Actual Prices")
plt.plot(range(len(y_test_actual), len(y_test_actual) + len(predicted_future)), predicted_future, label="Future Predictions")
plt.legend()
plt.title("Future TD Bank Stock Price Prediction with Political Dynamics")
plt.show()

