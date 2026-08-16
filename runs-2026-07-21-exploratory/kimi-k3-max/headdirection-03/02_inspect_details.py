"""Detailed inspection of one session: units metadata, states, LED position data."""
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np

ASSET_ID = "4cc64fe0-7b1e-404c-8b86-fb5659292830"  # sub-Mouse17_ses-Mouse17-130128
URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"

disk_cache = remfile.DiskCache("/tmp/remfile_cache_hd")
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

units = nwb["units"]
print("=== UNITS ===")
print("n units:", len(units))
print("metadata columns:", units.metadata_columns)
print(units.metadata.head(20) if hasattr(units.metadata, "head") else units.metadata)
print()

print("=== STATES ===")
states = nwb["states"]
print(states)
print()

print("=== LED position ===")
red = nwb["SubjectPosition/RedLED"]
blue = nwb["SubjectPosition/BlueLED"]
print("RedLED:", type(red), red.shape if hasattr(red, "shape") else "")
print("RedLED rate:", red.rate if hasattr(red, "rate") else "n/a")
print("RedLED time support:", red.time_support)
print("RedLED first values:\n", red.values[:5])
print("RedLED NaN frac:", np.isnan(red.values[:, 0]).mean())
print("BlueLED NaN frac:", np.isnan(blue.values[:, 0]).mean())
print("t range:", red.t[0], red.t[-1])
print()

print("=== units time support ===")
print(units.time_support)
