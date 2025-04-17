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
