# %% [markdown]
# # Load MC_Maze (DANDI 000128) - macaque M1/PMd reaching dataset
# This script resolves the asset via the DANDI REST API (the `+` in the asset path
# breaks the dandi client's `get_asset_by_path`) and streams the NWB file with remfile.

# %%
import requests
import remfile
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np

DANDISET_ID = "000128"
ASSET_PATH = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

# Resolve the asset ID from the DANDI REST API
r = requests.get(
    f"https://api.dandiarchive.org/api/dandisets/{DANDISET_ID}/versions/draft/assets/",
    params={"path": ASSET_PATH},
    timeout=60,
)
r.raise_for_status()
results = r.json()["results"]
print(f"Found {len(results)} matching assets")
asset = results[0]
asset_id = asset["asset_id"]
print(f"Asset: {asset['path']}")
print(f"Asset size: {asset['size'] / 1e6:.1f} MB")

s3_url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
print(f"Download URL: {s3_url}")

# Streaming access with remfile + disk cache
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)