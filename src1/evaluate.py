import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_loader import load_data, preprocess_data

'''

Flux C LSTM Filtered vs Truth Flux B: MAE=117.124, RMSE=203.385, Correlation=1.000
 near-perfect correlation of 1.000 between the denoised flux and the ground truth. The MAE and RMSE are also relatively low, suggesting the model is making accurate predictions.
'''
# Load model
model = load_model("model.keras")

# Load and preprocess data
df = load_data()
X, y, scaler = preprocess_data(df)

# Split test data
split = int(len(X) * 0.8)
X_test, y_test = X[split:], y[split:]

# Make predictions
predicted = model.predict(X_test)

# Convert back to original scale
predicted = scaler.inverse_transform(
    np.hstack((predicted, np.zeros((predicted.shape[0], 3)))))[:, :3]

y_test_original = scaler.inverse_transform(
    np.hstack((y_test, np.zeros((y_test.shape[0], 3)))))[:, :3]

# Plot Flux B X values
plt.figure(figsize=(12, 6))
plt.plot(y_test_original[:, 0], label="True Flux B X", alpha=0.6)
plt.plot(predicted[:, 0], label="LSTM Denoised Flux C X", linestyle="dashed")
plt.legend()
plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from scipy.stats import pearsonr

# Assign correct variables
true_flux = y_test_original[:, 0]  # True Flux B
lstm_denoised_flux = predicted[:, 0]  # LSTM Filtered Flux C

# Compute MAE
mae = mean_absolute_error(true_flux, lstm_denoised_flux)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(true_flux, lstm_denoised_flux))

# Compute Pearson Correlation
correlation, _ = pearsonr(true_flux.flatten(), lstm_denoised_flux.flatten())

# Print results in the same format as the Kalman filter
print(f"Flux C LSTM Filtered vs Truth Flux B: MAE={mae:.3f}, RMSE={rmse:.3f}, Correlation={correlation:.3f}")
