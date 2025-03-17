import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load test data
test_X = np.load("C:/Users/Reva Laxmi Chauhan/Desktop/Aeromagnetic-Noise-Reduction/src3/data/test_X.npy")
test_y = np.load("C:/Users/Reva Laxmi Chauhan/Desktop/Aeromagnetic-Noise-Reduction/src3/data/test_y.npy")

# Reshape for LSTM
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

# Load trained model with explicit loss function
model = tf.keras.models.load_model(
    "C:/Users/Reva Laxmi Chauhan/Desktop/Aeromagnetic-Noise-Reduction/src3/models/lstm_model.h5",
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()}  # Fix for 'mse' error
)

# Make predictions
pred_y = model.predict(test_X)

# Plot actual vs. predicted values for Flux B (x-component)
plt.figure(figsize=(10, 5))
plt.plot(test_y[:500, 0], label="True Flux B_x", color="blue")
plt.plot(pred_y[:500, 0], label="Predicted Flux B_x", color="red", linestyle="dashed")
plt.legend()
plt.xlabel("Sample")
plt.ylabel("Flux B_x")
plt.title("Flux B_x: True vs. Predicted")
plt.show()
