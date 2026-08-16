"""Inspect one session of DANDI 000056 (Peyrache et al. head direction dataset)."""
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap

ASSET_ID = "4cc64fe0-7b1e-404c-8b86-fb5659292830"  # sub-Mouse17_ses-Mouse17-130128
URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"

disk_cache = remfile.DiskCache("/tmp/remfile_cache_hd")
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)
print()
print("keys:", list(nwb.keys()))
