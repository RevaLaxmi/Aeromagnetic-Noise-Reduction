import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# Load test data
test_X = np.load("C:/Users/Reva Laxmi Chauhan/Desktop/Aeromagnetic-Noise-Reduction/src3/data/test_X.npy")
test_y = np.load("C:/Users/Reva Laxmi Chauhan/Desktop/Aeromagnetic-Noise-Reduction/src3/data/test_y.npy")

# Reshape for LSTM
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

# Load trained model with custom objects
model = tf.keras.models.load_model(
    "C:/Users/Reva Laxmi Chauhan/Desktop/Aeromagnetic-Noise-Reduction/src3/models/lstm_model.h5",
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()}  # Explicitly register MSE loss
)

# Make predictions
pred_y = model.predict(test_X)

# Compute metrics
mae = mean_absolute_error(test_y, pred_y)
rmse = mean_squared_error(test_y, pred_y, squared=False)
correlation = np.corrcoef(test_y.flatten(), pred_y.flatten())[0, 1]

# Print results
print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, Correlation: {correlation:.3f}")
