"""01_load_data.py — open the Achilles session from DANDI 000044 via remfile and inspect."""
import json
import os

import h5py
import numpy as np
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO

CACHE_DIR = "/tmp/remfile_cache_ripples02"
os.makedirs(CACHE_DIR, exist_ok=True)

ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"
DANDI_URL = f"https://api.dandiarchive.org/api/dandisets/000044/versions/draft/assets/{ASSET_ID}/download/"

# The /download/ endpoint 302-redirects to a presigned S3 URL; grab Location without following.
r = requests.get(DANDI_URL, allow_redirects=False)
s3_url = r.headers["Location"]
print("S3 URL resolved:", s3_url[:100], "...")

disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# Save the resolved URL (without signature expiry concerns for this session) for later scripts
with open("scripts/s3_url.txt", "w") as f:
    f.write(s3_url)
