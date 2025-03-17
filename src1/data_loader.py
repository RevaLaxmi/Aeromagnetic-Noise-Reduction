import numpy as np
import h5py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

'''
testing training on flight - data file 1003 
'''

def load_data(file_path="data\Flt1003_train.h5"):
    with h5py.File(file_path, "r") as f:
        flux_b_x = np.array(f["flux_b_x"])
        flux_b_y = np.array(f["flux_b_y"])
        flux_b_z = np.array(f["flux_b_z"])
        flux_c_x = np.array(f["flux_c_x"])
        flux_c_y = np.array(f["flux_c_y"])
        flux_c_z = np.array(f["flux_c_z"])
    
    df = pd.DataFrame({
        "flux_b_x": flux_b_x, "flux_b_y": flux_b_y, "flux_b_z": flux_b_z,
        "flux_c_x": flux_c_x, "flux_c_y": flux_c_y, "flux_c_z": flux_c_z
    })
    return df

def preprocess_data(df, seq_length=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    def create_sequences(data, target, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(target[i + seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(
        df_scaled[["flux_c_x", "flux_c_y", "flux_c_z"]].values,
        df_scaled[["flux_b_x", "flux_b_y", "flux_b_z"]].values,
        seq_length
    )

    return X.reshape((X.shape[0], seq_length, 3)), y, scaler
