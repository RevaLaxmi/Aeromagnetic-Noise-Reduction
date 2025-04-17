'''
testing
'''

import h5py
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from step4 import preprocess

# Load new test data
file_path = "data\Flt1005_train.h5"

with h5py.File(file_path, 'r') as f:
    data_dict = {}
    for key in f.keys():
        try:
            data = f[key][:]
            if isinstance(data, np.ndarray):
                data_dict[key] = data
        except Exception as e:
            print(f"Could not load {key}: {e}")

    df_test = pd.DataFrame(data_dict)

# Preprocess the test data (apply same preprocessing as training data)
df_test = preprocess(df_test)

# Extract features and labels from the test data
from utils import extract_features_and_labels
X_test, y_test = extract_features_and_labels(df_test, target='mag_1_c')

# Load the trained model and scaler
mlp = joblib.load("models/mlp_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Scale the test data using the saved scaler
X_test_scaled = scaler.transform(X_test)

# Make predictions
y_test_pred = mlp.predict(X_test_scaled)

# Evaluate the model on the test data
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Test R^2:", test_r2)
print("Test MSE:", test_mse)
