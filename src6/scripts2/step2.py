'''
creating a de noising model - according the the data that we have4

tie line - line number
flight - flight number
year - year
doy - day of year
tt s fiducial seconds past midnight UTC
utmX m x-coordinate, WGS-84 UTM zone 18N
utmY m y-coordinate, WGS-84 UTM zone 18N
utmZ m z-coordinate, GPS altitude (WGS-84)
msl m z-coordinate, GPS altitude above EGM2008 Geoid
lat deg latitude, WGS-84
lon deg longitude, WGS-84
baro m barometric altimeter
radar m filtered radar altimeter
topo m radar topography (WGS-84)
dem m digital elevation model from SRTM (WGS-84)
drape m planned survey drape (WGS-84)
ins pitch deg INS-computed aircraft pitch
ins roll deg INS-computed aircraft roll
ins yaw deg INS-computed aircraft yaw
diurnal nT measured diurnal
mag 1 c nT Mag 1: compensated magnetic field
mag 1 lag nT Mag 1: lag-corrected magnetic field
mag 1 dc nT Mag 1: diurnal-corrected magnetic field
mag 1 igrf nT Mag 1: IGRF & diurnal-corrected magnetic field
mag 1 uc nT Mag 1: uncompensated magnetic field
mag 2 uc nT Mag 2: uncompensated magnetic field
mag 3 uc nT Mag 3: uncompensated magnetic field
mag 4 uc nT Mag 4: uncompensated magnetic field
mag 5 uc nT Mag 5: uncompensated magnetic field
mag 6 uc nT Mag 6: uncompensated magnetic field
flux a x nT Flux A: fluxgate x-axis
flux a y nT Flux A: fluxgate y-axis
flux a z nT Flux A: fluxgate z-axis
flux a t nT Flux A: fluxgate total
flux b x nT Flux B: fluxgate x-axis
flux b y nT Flux B: fluxgate y-axis
flux b z nT Flux B: fluxgate z-axis
flux b t nT Flux B: fluxgate total
8
flux c x nT Flux C: fluxgate x-axis
flux c y nT Flux C: fluxgate y-axis
flux c z nT Flux C: fluxgate z-axis
flux c t nT Flux C: fluxgate total
flux d x nT Flux D: fluxgate x-axis
flux d y nT Flux D: fluxgate y-axis
flux d z nT Flux D: fluxgate z-axis
flux d t nT Flux D: fluxgate total
ogs mag nT OGS survey diurnal-corrected, levelled, magnetic field
ogs alt m OGS survey, GPS altitude (WGS-84)
ins acc x m/s2
INS x-acceleration
ins acc y m/s2
INS y-acceleration
ins acc z m/s2
INS z-acceleration
ins wander rad INS-computed wander angle (ccw from north)
ins lat rad INS-computed latitude
ins lon rad INS-computed longitude
ins alt m INS-computed altitude (WG... and so on
'''

import h5py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# File path
file_path = 'data/Flt1003_train.h5'  # Adjust the path as necessary

# Read the file and dynamically load all the datasets
print("Reading datasets...")

# Initialize an empty dictionary to hold the datasets
data_dict = {}

with h5py.File(file_path, 'r') as f:
    # Loop over all top-level keys in the HDF5 file
    for key in f.keys():
        try:
            # Load data for the key
            data_dict[key] = f[key][:]
            print(f"Successfully loaded {key}!")
        except KeyError as e:
            print(f"Error loading {key}: {e}")
        except Exception as e:
            print(f"Unexpected error loading {key}: {e}")

# Check the content of 'mag_6_uc'
if 'mag_6_uc' in data_dict:
    print("Content of mag_6_uc:")
    print(f"Shape of 'mag_6_uc': {data_dict['mag_6_uc'].shape}")
    print(f"First few values: {data_dict['mag_6_uc'][:5]}")
else:
    print("Key 'mag_6_uc' not found in the dataset!")

# Convert the data into a DataFrame
df = pd.DataFrame(data_dict)

# Check if the target column 'mag_1_c' exists
target = 'mag_1_c'

if target not in df.columns:
    raise KeyError(f"Target column '{target}' not found in the dataset!")

# Clean the dataset and prepare for training
features = [
    'mag_1_uc', 'mag_2_uc', 'mag_3_uc', 'mag_4_uc', 'mag_5_uc',
    # 'mag_6_uc',   # REMOVE THIS LINE
    'flux_a_x', 'flux_a_y', 'flux_a_z', 'flux_a_t',
    'flux_b_x', 'flux_b_y', 'flux_b_z', 'flux_b_t',
    'flux_c_x', 'flux_c_y', 'flux_c_z', 'flux_c_t',
    'flux_d_x', 'flux_d_y', 'flux_d_z', 'flux_d_t',
    'baro', 'radar', 'topo', 'dem', 'drape',
    'ins_acc_x', 'ins_acc_y', 'ins_acc_z', 'ins_pitch', 'ins_roll',
    'ins_vn', 'ins_vw', 'ins_vu', 'ins_lat', 'ins_lon', 'ins_alt',
    'diurnal', 'lat', 'lon', 'utm_x', 'utm_y', 'utm_z'
]


# Ensure all features exist in the dataset
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"Missing the following features: {', '.join(missing_features)}")
    raise KeyError(f"Missing the following features: {', '.join(missing_features)}")

# Clean the dataset by dropping rows with missing values in features or target
df_clean = df[features + [target]].dropna()

# Prepare the input (X) and target (y)
X = df_clean[features]
y = df_clean[target]

# Split into training and test sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate the performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Model R^2 score on test data: {r2:.3f}")
print(f"Model Mean Squared Error on test data: {mse:.3f}")

# Optionally, save the model for later use
joblib.dump(model, 'magnetic_signal_denoising_model.pkl')
