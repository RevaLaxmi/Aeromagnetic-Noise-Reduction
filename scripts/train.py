import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

# Load preprocessed training data
train_X = np.load("C:/Users/Reva Laxmi Chauhan/Desktop/Aeromagnetic-Noise-Reduction/src3/data/train_X.npy")
train_y = np.load("C:/Users/Reva Laxmi Chauhan/Desktop/Aeromagnetic-Noise-Reduction/src3/data/train_y.npy")

# Reshape for LSTM (samples, timesteps, features)
train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, 3)),  # LSTM layer
    LSTM(32, return_sequences=False),  # Second LSTM layer
    Dense(16, activation='relu'),  # Fully connected layer
    Dense(3)  # Output: flux_b_x, flux_b_y, flux_b_z
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(train_X, train_y, epochs=50, batch_size=64, validation_split=0.1)

# Save trained model
model.save("C:/Users/Reva Laxmi Chauhan/Desktop/Aeromagnetic-Noise-Reduction/src3/models/lstm_model.h5")

print("Model training complete. Model saved.")
