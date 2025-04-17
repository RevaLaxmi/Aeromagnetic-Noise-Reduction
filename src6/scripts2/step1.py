import h5py
import pandas as pd
import numpy as np

file_path = "data/Flt1003_train.h5"  # adjust path if needed

# Open the file and read all datasets
with h5py.File(file_path, "r") as f:
    print("Reading datasets...")
    data_dict = {}
    for key in f.keys():
        try:
            data = f[key][:]
            data_dict[key] = data
        except Exception as e:
            print(f"Could not load {key}: {e}")

# Convert to pandas DataFrame
df = pd.DataFrame(data_dict)

# Print preview
print("DataFrame preview:")
print(df.head())
print("Shape:", df.shape)




with h5py.File(file_path, "r") as f:
    print("Available keys in the H5 file:")
    for key in f.keys():
        print("-", key)




with h5py.File(file_path, "r") as f:
    print("Available keys in the H5 file:")
    for key in f.keys():
        print("-", key)  # This will list all top-level keys
    
    # Check the contents of a specific key (e.g., mag_1_uc)
    try:
        mag_1_uc = f['mag_1_uc'][:]
        print("Successfully loaded mag_1_uc!")
    except KeyError as e:
        print(f"Error loading 'mag_1_uc': {e}")
