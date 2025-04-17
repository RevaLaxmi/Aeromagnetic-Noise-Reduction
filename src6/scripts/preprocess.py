'''
STEP 2
'''

'''
Loads the .h5 file,

Filters valid flight segments,

Selects and normalizes the input features,

Returns train-test split data for modeling.
'''

# preprocess_model.py

import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Example model, replace with your choice
from sklearn.metrics import mean_squared_error

def load_h5_to_dataframe(file_path):
    """Loads HDF5 flight data and returns a Pandas DataFrame."""
    data_dict = {}
    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            try:
                data = f[key][()]
                if np.ndim(data) == 1 or data.shape[1:] == ():
                    data_dict[key] = data
            except Exception as e:
                print(f"⚠️ Could not load {key}: {e}")
    df = pd.DataFrame(data_dict)
    return df

def clean_data(df):
    """Handles missing values, types, and converts columns."""
    # Handling missing values (fill with zeros or drop based on use case)
    df = df.fillna(0)  # You can also choose df.dropna(axis=0) if you prefer to drop rows with NaNs
    
    # Convert columns to the appropriate type if necessary (example)
    # df['column_name'] = df['column_name'].astype(float)
    
    print("\nData cleaned. Missing values filled.")
    return df

def preprocess_data(df, target_col='mag_1_c'):
    """Selects features, scales them, and returns train/test splits."""
    # Exclude target column and potentially irrelevant columns
    exclude_keywords = ['mag_1_c', 'mag_2_c', 'mag_3_c', 'dt', 'N']
    input_cols = [col for col in df.columns if not any(ex in col for ex in exclude_keywords)]

    # Features and target variable
    X = df[input_cols]
    y = df[target_col]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining data shape: {X_train.shape}, {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, input_cols

def build_model(X_train, y_train):
    """Build a machine learning model and fit to the data."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Example model
    model.fit(X_train, y_train)
    
    print("\nModel trained.")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using MSE (Mean Squared Error)."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nMean Squared Error: {mse}")
    
    return mse

if __name__ == "__main__":
    file_path = "data/Flt1003_train.h5"   # Replace with the correct file path
    df = load_h5_to_dataframe(file_path)
    
    print(f"Loaded DataFrame with shape: {df.shape}")
    
    # Clean the data (handle missing values)
    df_cleaned = clean_data(df)
    
    # Preprocess the data for training
    X_train, X_test, y_train, y_test, input_cols = preprocess_data(df_cleaned)
    
    # Train the model
    model = build_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
