import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm(seq_length=10):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, 3)),
        LSTM(64, return_sequences=False),
        Dense(32, activation="relu"),
        
        Dense(3)  # Output: flux_b_x, flux_b_y, flux_b_z
    ])

    model.compile(optimizer="adam", loss="mse")
    return model

'''
working with 32 neurons so we're having non linearity.. 
"While the Kalman Filter provides a straightforward way to filter noise based on state-space models, 
it does not learn from historical patterns. LSTMs, being deep learning models specialized for time-series data, can identify and correct for complex patterns in noisy aeromagnetic data, 
leading to a significant improvement in denoising performance
'''