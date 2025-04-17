import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import h5py
import joblib

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
    keys_to_load = ['mag_1_uc', 'mag_1_lag', 'mag_1_c', 'lat', 'lon', 'ins_acc_x', 'dem', 'diurnal']
    with h5py.File(file_path, 'r') as f:
        data_dict = {key: np.array(f[key]) for key in keys_to_load if key in f}
    df = pd.DataFrame(data_dict)
    df_cleaned = preprocess(df)
    return df_cleaned

# Main
if __name__ == "__main__":
    file_path = 'data/Flt1003_train.h5'

    # Load and preprocess
    df_cleaned = load_and_preprocess_data(file_path)

    # Extract features and labels
    X, y = extract_features_and_labels(df_cleaned, target='mag_1_c')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Normalize labels
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    # Build MLP model
    model = Sequential([
        Dense(50, input_dim=X_train_scaled.shape[1], activation='tanh'),
        Dense(30, activation='tanh'),
        Dense(10, activation='tanh'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32,
                        validation_data=(X_test_scaled, y_test_scaled))

    # Evaluate and inverse transform predictions
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_scaled)
    
    print("Min of true target:", y_true.min())
    print("Max of true target:", y_true.max())
    print("Mean of true target:", y_true.mean())
    print("First 10 true target values:\n", y_true[:10].flatten())

    

    mse = mean_squared_error(y_true, y_pred)
    print(f'Inverse Transformed Test MSE: {mse:.4f}')

    # Save model and scalers
    model.save('mlp_model3.h5')
    joblib.dump(scaler_X, 'scaler_X3.pkl')
    joblib.dump(scaler_y, 'scaler_y3.pkl')
