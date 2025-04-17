'''
loading cloumn na,es

'''

import h5py

file_path = "data/Flt1003_train.h5"

with h5py.File(file_path, "r") as f:
    print("Available keys in the H5 file:")
    for key in f.keys():
        print("-", key)
