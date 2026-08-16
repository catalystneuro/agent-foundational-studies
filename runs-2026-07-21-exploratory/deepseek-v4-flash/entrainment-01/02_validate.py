# %% [markdown]
# # 02 — Validate data streams
# Check LFP sampling rate / channels, position timestamps (known rate-field
# quirk in this file), unit metadata, epoch labels.

# %%
import h5py
import numpy as np
import remfile
import requests
from pynwb import NWBHDF5IO
import pynapple as nap

ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"

def resolve_s3_url(asset_id):
    dl = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    r = requests.get(dl, allow_redirects=False, timeout=120)
    return r.headers["Location"]

s3_url = resolve_s3_url(ASSET_ID)
disk_cache = remfile.DiskCache("/tmp/remfile_cache_entrainment01")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print("=== UNITS ===")
print("n units:", len(nwb["units"]))
print("keys sample:", list(nwb["units"].keys())[:5])
ct = nwb["units"].get_info("cell_type")
if ct is None:
    # fallback
    ct_series = None
    print("cell_type info: None")
else:
    ct_series = ct
    import pandas as pd
    print("cell_type counts:", ct.value_counts().to_dict())
print("location:", nwb["units"].get_info("location"))
print("n_shanks:", nwb["units"].get_info("shank_id"))

print("\n=== EPOCHS ===")
ep = nwb["epochs"]
print(ep)
print("labels:", ep.label)

print("\n=== STATES ===")
st = nwb["states"]
print(st.label)
print("start:", st.start, "end:", st.end)

print("\n=== POSITION (1.6mLinearMazeSpatialSeries) ===")
pos = nwb["1.6mLinearMazeSpatialSeries"]
print(pos)
t = np.asarray(pos.t)
print("n:", len(t), "t[0]:", t[0], "t[-1]:", t[-1])
dt = np.diff(t)
print("dt median:", np.median(dt), "dt[0:5]:", dt[:5])

print("\n=== LFP ===")
lfp = nwb["LFP"]
print(lfp)
print("lfp shape:", lfp.shape)
print("elastic:", getattr(lfp, "elastic", None), "rate:", getattr(lfp, "rate", None))