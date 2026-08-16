"""Load MC_Maze with pynapple, inspect structure, and validate data streams."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

S3_MCMAZE = "https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a"

disk_cache = remfile.DiskCache("cache/remfile_cache")
rem_file = remfile.File(S3_MCMAZE, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

print("\n== keys ==")
print(list(nwb.keys()))

print("\n== trials table ==")
trials = nwb["trials"]
print(type(trials))
print(trials)

print("\n== units ==")
units = nwb["units"]
print(units)

print("\n== behavior ==")
print(nwb["behavior"])
