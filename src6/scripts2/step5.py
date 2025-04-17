import h5py
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from step4 import preprocess

# Load new test data
file_path = "data/Flt1005_train.h5"

with h5py.File(file_path, 'r') as f:
    print("Available keys in the HDF5 file:")
    # Print the available keys to understand the structure of the file
    for key in f.keys():
        print("-", key)
    
    # Read data from the file (checking if it's an array or scalar)
    data_dict = {}
    for key in f.keys():
        try:
            data = f[key][:]
            if isinstance(data, np.ndarray):
                data_dict[key] = data
            else:
                print(f"Skipping scalar value for {key}")
        except Exception as e:
            print(f"Could not load {key}: {e}")
    
    # Convert the loaded data to a pandas DataFrame
    df_test = pd.DataFrame(data_dict)

# Preprocess the test data (apply same preprocessing as training data)
df_test = preprocess(df_test)

# Print columns after preprocessing to check if anything changed
print("Columns after preprocessing:", df_test.columns)

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
