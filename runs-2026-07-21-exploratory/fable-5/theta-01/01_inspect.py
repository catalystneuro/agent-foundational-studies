"""Inspect the structure of the Achilles 10252013 session from DANDI:000044."""
import h5py
import remfile
import numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap

S3 = "https://api.dandiarchive.org/api/assets/5349c68b-c0a7-46c0-9900-cda050722fa4/download/"

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")

print("=== top level ===")
def show(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  {name}  shape={obj.shape} dtype={obj.dtype} chunks={obj.chunks} compression={obj.compression}")
h5f.visititems(show)

io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()
print(nwbfile)
