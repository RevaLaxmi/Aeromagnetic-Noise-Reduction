'''
processing pt2 - tensor flow model 
'''

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import h5py

# Magnetometer Cleaning
def clean_magnetometer(df):
    if 'mag_1_uc' in df and 'mag_1_lag' in df:
        df['mag_1_uc_smooth'] = savgol_filter(df['mag_1_uc'], window_length=11, polyorder=2, mode='nearest')
        df['mag_1_lag_smooth'] = df['mag_1_lag'].rolling(window=5, min_periods=1).mean()
    return df

# Geolocation Cleaning
def clean_geolocation(df):
    if 'lat' in df and 'lon' in df:
        df['lat'] = df['lat'].interpolate(method='linear')
        df['lon'] = df['lon'].interpolate(method='linear')
        df['lat_diff'] = df['lat'].diff().fillna(0)
        df['lon_diff'] = df['lon'].diff().fillna(0)
    return df

# INS Cleaning
def clean_ins(df):
    if 'ins_acc_x' in df:
        df['ins_acc_x_smooth'] = df['ins_acc_x'].rolling(window=5, min_periods=1).mean()
    return df

# Environmental Cleaning
def clean_environment(df):
    if 'dem' in df:
        df['dem_z'] = (df['dem'] - df['dem'].mean()) / df['dem'].std()
    if 'diurnal' in df:
        df['diurnal_sin'] = np.sin(2 * np.pi * df['diurnal'] / 24)
        df['diurnal_cos'] = np.cos(2 * np.pi * df['diurnal'] / 24)
    return df

# Apply All Cleaning Steps
def preprocess(df):
    df = clean_magnetometer(df)
    df = clean_geolocation(df)
    df = clean_ins(df)
    df = clean_environment(df)
    return df

# Extract Features and Labels
def extract_features_and_labels(df, target='mag_1_c'):
    feature_columns = ['mag_1_uc_smooth', 'mag_1_lag_smooth', 'lat', 'lon', 'lat_diff', 'lon_diff',
                       'ins_acc_x_smooth', 'dem_z', 'diurnal_sin', 'diurnal_cos']
    X = df[feature_columns].fillna(0)
    y = df[target].fillna(0)
    return X, y

# Load the .h5 file using h5py
def load_and_preprocess_data(file_path):
    # List of features to load
    keys_to_load = ['mag_1_uc', 'mag_1_lag', 'mag_1_c', 'lat', 'lon', 'ins_acc_x', 'dem', 'diurnal']
    
    with h5py.File(file_path, 'r') as f:
        data_dict = {}
        for key in keys_to_load:
            if key in f:
                data_dict[key] = np.array(f[key])
            else:
                print(f"Warning: Key {key} not found in file.")
    
    # Convert to DataFrame
    df = pd.DataFrame(data_dict)
    df_cleaned = preprocess(df)
    return df_cleaned

# Main
if __name__ == "__main__":
    file_path = 'data/Flt1003_train.h5'

    # Load and preprocess
    df_cleaned = load_and_preprocess_data(file_path)

    # Extract features/labels
    X, y = extract_features_and_labels(df_cleaned, target='mag_1_c')

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build MLP
    model = Sequential([
        Dense(50, input_dim=X_train_scaled.shape[1], activation='tanh'),
        Dense(30, activation='tanh'),
        Dense(10, activation='tanh'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train
    history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

    # Evaluate
    test_loss = model.evaluate(X_test_scaled, y_test)
    print(f'Test Loss (MSE): {test_loss:.4f}')

    # Save model
    model.save('mlp_model.h5')
