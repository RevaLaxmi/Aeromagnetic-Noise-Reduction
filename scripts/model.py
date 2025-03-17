from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(30, return_sequences=False),
        Dense(10, activation='tanh'),
        Dense(1)  # Output layer
    ])
    model.compile(optimizer="adam", loss="mse")
    return model
