'''
Step 1 — loading the data, selecting your input and target columns, and preparing everything for modeling.
'''

'''

Concept	-> What It Means	-> How It Applies
Features (X) ->	Input data the model learns from	Sensor signals other than mag_1_c
Target (y) ->	What you want to predict	Clean signal: mag_1_c
Normalization (StandardScaler)	->  Adjust feature values to have mean 0, std 1	Makes training stable and fast
Train/test split	-> Split data into parts for training and evaluation	Helps test generalization of the model
Regression	-> Predicting a continuous value (not classification)	Your case: continuous magnetic signal

'''

import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

file_path = "data/Flt1003_train.h5"

# Read data using h5py
print("Reading datasets...")
data_dict = {}
with h5py.File(file_path, "r") as f:
    for key in f.keys():
        try:
            data = f[key][()]  # Extract as NumPy array
            # Convert to 1D if it's a single column (avoid scalars like dt, N)
            if np.ndim(data) == 1 or data.shape[1:] == ():
                data_dict[key] = data
        except Exception as e:
            print(f"Could not load {key}: {e}")

# Create DataFrame
df = pd.DataFrame(data_dict)
print("DataFrame preview:")
print(df.head())
print("Shape:", df.shape)

# === Preprocessing ===
target_col = 'mag_1_c'
exclude_keywords = ['mag_1_c', 'mag_2_c', 'mag_3_c', 'dt', 'N']
input_cols = [col for col in df.columns if not any(ex in col for ex in exclude_keywords)]

X = df[input_cols]
y = df[target_col]

# Normalize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("✅ Input features:", len(input_cols))
print("🧠 Training set:", X_train.shape, y_train.shape)
print("🧪 Testing set:", X_test.shape, y_test.shape)
