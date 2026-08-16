"""Inspect the structure of one Grosmark & Buzsaki (DANDI:000044) session."""
import h5py
import remfile
import numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap

ASSET = "5349c68b-c0a7-46c0-9900-cda050722fa4"  # Achilles 10252013
URL = f"https://api.dandiarchive.org/api/dandisets/000044/versions/0.250624.0426/assets/{ASSET}/download/"

cache = remfile.DiskCache("/tmp/remfile_cache")
f = remfile.File(URL, disk_cache=cache)
h5 = h5py.File(f, "r")

print("--- top level ---")
def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}  shape={obj.shape} dtype={obj.dtype} chunks={obj.chunks} compression={obj.compression}")
h5.visititems(walk)
