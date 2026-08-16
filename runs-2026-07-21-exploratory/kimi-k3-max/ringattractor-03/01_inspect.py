"""Prototype: stream Mouse28-140310 from DANDI 000056 and inspect structure."""
import h5py
import remfile
import numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap

ASSETS = {
    "Mouse17-130128": "4cc64fe0-7b1e-404c-8b86-fb5659292830",
    "Mouse20-130514": "748aa5de-c0de-4aa7-a7ef-2aad2f87a7eb",
    "Mouse24-131213": "ada02790-6eb6-48ee-902d-9ba017303586",
    "Mouse25-140123": "bdb30f7d-ba69-4d2e-8504-2efab69cd8d7",
    "Mouse28-140310": "656704ea-a4cd-40f4-8158-a6533ebf2eee",
}

def load_session(asset_id, cache_dir="/tmp/remfile_cache_hd"):
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile)

nwb = load_session(ASSETS["Mouse28-140310"])
print(nwb)
print("\n--- units ---")
units = nwb["units"]
print(units)
print("metadata columns:", units.metadata_columns)
print("rates:", np.round(units["rate"][:], 2))
print("\n--- states ---")
states = nwb["states"]
print(states)
print("labels:", np.unique(states["label"]))
for lab in np.unique(states["label"]):
    ep = states[states["label"] == lab]
    print(lab, "total duration [s]:", round(float((ep.end - ep.start).sum()), 1), " n epochs:", len(ep))
print("\n--- LEDs ---")
red = nwb["SubjectPosition/RedLED"]
blue = nwb["SubjectPosition/BlueLED"]
print(red)
print("red shape:", red.values.shape, "rate:", red.rate)
