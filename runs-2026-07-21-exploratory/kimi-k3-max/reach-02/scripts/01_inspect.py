"""Inspect the MC_Maze NWB file structure via remfile streaming."""
import h5py
import remfile
from pynwb import NWBHDF5IO

S3_MCMAZE = "https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a"

disk_cache = remfile.DiskCache("cache/remfile_cache")
rem_file = remfile.File(S3_MCMAZE, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")


def show(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  DATASET {name}  shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"  GROUP   {name}")


print("== top level ==")
h5py_file.visititems(show)
