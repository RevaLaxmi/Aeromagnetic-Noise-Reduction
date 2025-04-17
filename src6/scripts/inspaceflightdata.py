import h5py
import pandas as pd
import numpy as np

def load_h5_to_dataframe(file_path):
    """Loads HDF5 file and returns a DataFrame of 1D-compatible keys."""
    data_dict = {}
    with h5py.File(file_path, "r") as f:
        print("📁 Keys in the HDF5 file:")
        for key in f.keys():
            print("🔑", key)
            try:
                data = f[key][()]
                if np.ndim(data) == 1 or data.shape[1:] == ():
                    data_dict[key] = data
            except Exception as e:
                print(f"⚠️ Could not load {key}: {e}")
    df = pd.DataFrame(data_dict)
    return df

def inspect_flight_data(df):
    """Prints unique tt line values and previews structure."""
    print("\n🧠 DataFrame shape:", df.shape)
    print("\n📋 Available columns:", list(df.columns))

    if 'tt' in df.columns:
        print("\n✈️ Unique 'tt' values:")
        print(df['tt'].unique())
        print("📏 Total flights:", df['tt'].nunique())
        print("🔍 'tt' column dtype:", df['tt'].dtype)

        print("\n📌 First 10 rows with 'tt':")
        print(df[['tt']].drop_duplicates().head(10))
    else:
        print("❌ 'tt' column not found!")

if __name__ == "__main__":
    file_path =  "data/Flt1003_train.h5"  # 🔁 Replace this with your actual file path
    print("🚀 Loading HDF5 tt data...")
    df = load_h5_to_dataframe(file_path)
    inspect_flight_data(df)
