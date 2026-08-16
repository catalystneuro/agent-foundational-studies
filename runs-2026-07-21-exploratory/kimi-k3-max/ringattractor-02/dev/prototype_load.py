"""Prototype: load Mouse17-130128 from DANDI 000056, inspect, build HD signal."""
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

ASSETS = {
    "Mouse17-130128": "4cc64fe0-7b1e-404c-8b86-fb5659292830",
    "Mouse20-130514": "748aa5de-c0de-4aa7-a7ef-2aad2f87a7eb",
    "Mouse24-131213": "ada02790-6eb6-48ee-902d-9ba017303586",
    "Mouse25-140123": "bdb30f7d-ba69-4d2e-8504-2efab69cd8d7",
    "Mouse28-140310": "656704ea-a4cd-40f4-8158-a6533ebf2eee",
}

def load_session(session):
    url = f"https://api.dandiarchive.org/api/assets/{ASSETS[session]}/download/"
    disk_cache = remfile.DiskCache("/tmp/remfile_cache_hd")
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb

nwb = load_session("Mouse17-130128")
print(nwb)
print("\n--- keys ---")
for k in nwb.keys():
    print(k)
