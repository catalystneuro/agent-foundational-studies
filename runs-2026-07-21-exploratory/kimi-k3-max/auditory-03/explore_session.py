"""Explore one session of DANDI 000986 (Jaramillo auditory cortex Neuropixels)."""
import h5py
import remfile
import numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/cac/52e/cac52ee7-20d9-4f7d-a234-a04c22a94083"

disk_cache = remfile.DiskCache('/tmp/remfile_cache_auditory03')
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)
print("\n--- keys ---")
print(list(nwb.keys()))

print("\n--- trials ---")
trials = nwb["trials"]
print(type(trials))
print(trials)

print("\n--- units ---")
units = nwb["units"]
print(type(units))
print(unts_shape := getattr(units, "shape", None))
print("n units:", len(units))
