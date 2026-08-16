# %% [markdown]
# # 01 — Load NWB (Achilles, DANDI 000044) and inspect structure
# Streams the classic Buzsáki lab CA1 linear-maze session via remfile.

# %%
import h5py
import numpy as np
import remfile
import requests
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET_ID = "000044"
ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"


def resolve_s3_url(asset_id):
    """Resolve the GET-only presigned S3 URL for an asset."""
    dl = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    r = requests.get(dl, allow_redirects=False, timeout=120)
    return r.headers["Location"]


s3_url = resolve_s3_url(ASSET_ID)
print("S3 URL resolved, length", len(s3_url))

disk_cache = remfile.DiskCache("/tmp/remfile_cache_entrainment01")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)