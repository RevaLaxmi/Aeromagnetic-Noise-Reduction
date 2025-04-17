'''
testing on the other flight data
'''

import numpy as np
import pandas as pd
import h5py
import joblib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

# --- Cleaning Functions ---
def clean_magnetometer(df):
    if 'mag_1_uc' in df and 'mag_1_lag' in df:
        df['mag_1_uc_smooth'] = savgol_filter(df['mag_1_uc'], window_length=11, polyorder=2, mode='nearest')
        df['mag_1_lag_smooth'] = df['mag_1_lag'].rolling(window=5, min_periods=1).mean()
    return df

def clean_geolocation(df):
    if 'lat' in df and 'lon' in df:
        df['lat'] = df['lat'].interpolate(method='linear')
        df['lon'] = df['lon'].interpolate(method='linear')
        df['lat_diff'] = df['lat'].diff().fillna(0)
        df['lon_diff'] = df['lon'].diff().fillna(0)
    return df

def clean_ins(df):
    if 'ins_acc_x' in df:
        df['ins_acc_x_smooth'] = df['ins_acc_x'].rolling(window=5, min_periods=1).mean()
    return df

def clean_environment(df):
    if 'dem' in df:
        df['dem_z'] = (df['dem'] - df['dem'].mean()) / df['dem'].std()
    if 'diurnal' in df:
        df['diurnal_sin'] = np.sin(2 * np.pi * df['diurnal'] / 24)
        df['diurnal_cos'] = np.cos(2 * np.pi * df['diurnal'] / 24)
    return df

def preprocess(df):
    df = clean_magnetometer(df)
    df = clean_geolocation(df)
    df = clean_ins(df)
    df = clean_environment(df)
    return df

def extract_features_and_labels(df, target='mag_1_c'):
    feature_columns = ['mag_1_uc_smooth', 'mag_1_lag_smooth', 'lat', 'lon', 'lat_diff', 'lon_diff',
                       'ins_acc_x_smooth', 'dem_z', 'diurnal_sin', 'diurnal_cos']
    X = df[feature_columns].fillna(0)
    y = df[target].fillna(0)
    return X, y

def load_and_preprocess_data(file_path):
    keys_to_load = ['mag_1_uc', 'mag_1_lag', 'mag_1_c', 'lat', 'lon', 'ins_acc_x', 'dem', 'diurnal']
    with h5py.File(file_path, 'r') as f:
        data_dict = {key: np.array(f[key]) for key in keys_to_load if key in f}
    df = pd.DataFrame(data_dict)
    df_cleaned = preprocess(df)
    return df_cleaned

# --- Main Script ---
if __name__ == "__main__":
    # Load new flight file
    file_path = 'data/Flt1005_train.h5'
    df_cleaned = load_and_preprocess_data(file_path)

    # Extract features and labels
    X_new, y_new = extract_features_and_labels(df_cleaned, target='mag_1_c')

    # Load scalers
    scaler_X = joblib.load('scaler_X3.pkl')
    scaler_y = joblib.load('scaler_y3.pkl')

    # Scale input and output
    X_new_scaled = scaler_X.transform(X_new)
    y_new_scaled = scaler_y.transform(y_new.values.reshape(-1, 1))

    # Load trained model
    model = load_model('mlp_model3.h5', compile=False)

    y_pred_scaled = model.predict(X_new_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_new_scaled)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f'Flt1005 - Inverse Transformed Test MSE: {mse:.4f}, RMSE: {rmse:.4f}')

    print("\nSample true values (first 10):", y_true[:10].flatten())
    print("Sample predicted values (first 10):", y_pred[:10].flatten())

    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:100], label='True', linewidth=2)
    plt.plot(y_pred[:100], label='Predicted', linewidth=2)
    plt.title("Flt1005 - True vs Predicted (First 100 samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("mag_1_c")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
