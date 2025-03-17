import numpy as np
import h5py
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
import os

def load_data(file_name="Flt1003_train.h5", data_dir="data"):
    """
    Loads aeromagnetic noise data from the given .h5 file.
    
    :param file_name: Name of the .h5 file (e.g., "Flt1003_train.h5").
    :param data_dir: Directory where the data files are stored.
    :return: DataFrame containing flux_b (ground truth) and flux_c (noisy signals).
    """
    
    with h5py.File(file_name, "r") as f:
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
    """
    Normalizes and processes the data into sequences for LSTM training.
    
    :param df: Input DataFrame containing flux_b and flux_c values.
    :param seq_length: Length of input sequences for LSTM.
    :return: Processed input sequences (X), target values (y), and fitted scaler.
    """
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

