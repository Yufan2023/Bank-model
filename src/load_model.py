from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model('models/lstm_model.h5')
print("Model loaded successfully")

# Replace with real data input
example_input = np.random.random((1, 60, X_train.shape[2]))  

# Make a prediction
prediction = model.predict(example_input)
print("Predicted value:", prediction)
