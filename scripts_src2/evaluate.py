import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_loader import load_data, preprocess_data
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tensorflow.keras.losses import MeanSquaredError



# Load trained model
# model = load_model("lstm_flux_model.h5", compile=False)
model = load_model("model.keras", compile=False)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Load saved scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load and preprocess test data (1005)
df_test = load_data("Flt1005_train.h5")
X_test, y_test, _ = preprocess_data(df_test)

# Make predictions
predicted = model.predict(X_test)

# Convert back to original scale
predicted = scaler.inverse_transform(
    np.hstack((predicted, np.zeros((predicted.shape[0], 3)))))[:, :3]

y_test_original = scaler.inverse_transform(
    np.hstack((y_test, np.zeros((y_test.shape[0], 3)))))[:, :3]

# Compute evaluation metrics
mae = mean_absolute_error(y_test_original[:, 0], predicted[:, 0])
rmse = np.sqrt(mean_squared_error(y_test_original[:, 0], predicted[:, 0]))
correlation, _ = pearsonr(y_test_original[:, 0], predicted[:, 0])

print(f"Flux C LSTM Filtered vs Truth Flux B: MAE={mae:.3f}, RMSE={rmse:.3f}, Correlation={correlation:.3f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test_original[:, 0], label="True Flux B X", alpha=0.6)
plt.plot(predicted[:, 0], label="LSTM Denoised Flux C X", linestyle="dashed")
plt.legend()
plt.show()
