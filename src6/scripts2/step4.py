'''
preprocessing  - handlig the data cleaning and feature engineering aspect
looking at all the priprity sections on their own and applying what logistic works best 


Primary Features to keep:


mag_1_c, mag_1_dc, mag_1_lag, mag_1_uc (High R² values)


lat, lon, utm_x, utm_y (Geospatial data with strong correlation)


ins_lat, ins_lon, ins_acc_x (Flight dynamics with moderate to high R² values)


dem, diurnal, flux_a_t, flux_b_t, flux_a_x (Environmental or temporal data)
'''

# preprocessing.py

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

### Magnetometer Cleaning ###
def clean_magnetometer(df):
    if 'mag_1_uc' in df and 'mag_1_lag' in df:
        df['mag_1_uc_smooth'] = savgol_filter(df['mag_1_uc'], window_length=11, polyorder=2, mode='nearest')
        df['mag_1_lag_smooth'] = df['mag_1_lag'].rolling(window=5, min_periods=1).mean()
    return df

### Geolocation Cleaning ###
def clean_geolocation(df):
    if 'lat' in df and 'lon' in df:
        df['lat'] = df['lat'].interpolate(method='linear')
        df['lon'] = df['lon'].interpolate(method='linear')
        df['lat_diff'] = df['lat'].diff().fillna(0)
        df['lon_diff'] = df['lon'].diff().fillna(0)
    return df

### INS Cleaning ###
def clean_ins(df):
    if 'ins_acc_x' in df:
        df['ins_acc_x_smooth'] = df['ins_acc_x'].rolling(window=5, min_periods=1).mean()
    return df

### Environmental Cleaning ###
def clean_environment(df):
    if 'dem' in df:
        df['dem_z'] = (df['dem'] - df['dem'].mean()) / df['dem'].std()
    if 'diurnal' in df:
        df['diurnal_sin'] = np.sin(2 * np.pi * df['diurnal'] / 24)
        df['diurnal_cos'] = np.cos(2 * np.pi * df['diurnal'] / 24)
    return df

### Apply All Cleaning Steps ###
def preprocess(df):
    df = clean_magnetometer(df)
    df = clean_geolocation(df)
    df = clean_ins(df)
    df = clean_environment(df)
    return df
