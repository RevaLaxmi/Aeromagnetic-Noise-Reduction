import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from preprocess import load_and_prepare

# Load data
file_path = "data/Flt1003_train.h5"
X_train, X_test, y_train, y_test, feature_names = load_and_prepare(file_path)

# Build MLP model
def build_mlp(input_dim):
    model = Sequential([
        Dense(50, activation='tanh', input_shape=(input_dim,)),
        Dense(30, activation='tanh'),
        Dense(10, activation='tanh'),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

model = build_mlp(X_train.shape[1])
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=64,
    verbose=1
)

# Evaluate on test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"🧪 Test Loss (MSE): {test_loss:.4f}")
print(f"📏 Test MAE: {test_mae:.4f}")

# Predictions
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"📉 RMSE: {rmse:.4f}")

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(y_test[:300].values, label="True Signal", alpha=0.8)
plt.plot(y_pred[:300], label="MLP Prediction", alpha=0.8)
plt.legend()
plt.title("True vs Predicted Magnetic Signal (first 300 samples)")
plt.xlabel("Time step")
plt.ylabel("mag_1_c")
plt.show()
