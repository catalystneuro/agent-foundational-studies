"""Explore structure of one session of DANDI 000056 (Peyrache 2015 HD dataset)."""
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np

ASSET_ID = "4cc64fe0-7b1e-404c-8b86-fb5659292830"  # Mouse17-130128
URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"

disk_cache = remfile.DiskCache("cache")
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)
print()
print("keys:", list(nwb.keys()))

units = nwb["units"]
print("\nunits:", type(units), "n =", len(units))
print("metadata columns:", units.metadata_columns)
print("rates:", np.round(units["rate"], 2))

states = nwb["states"]
print("\nstates:")
print(states)

red = nwb["SubjectPosition"]["RedLED"]
blue = nwb["SubjectPosition"]["BlueLED"]
print("\nRedLED:", type(red), red.shape if hasattr(red, "shape") else "")
print("red time support:", red.time_support)
print("red first values:\n", red.values[:5])
print("red dt median:", np.median(np.diff(red.t[:1000])))
print("n sentinel rows red:", np.sum(np.any(red.values < 0, axis=1)), "/", len(red.values))
