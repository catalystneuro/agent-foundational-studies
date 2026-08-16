"""Step 1: resolve the S3 URL for the Achilles session and verify data streams."""
import requests
import h5py
import remfile
import numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap

DANDI_SET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"

# Resolve the asset's download URL via the DANDI API.
api_url = f"https://api.dandiarchive.org/api/dandisets/{DANDI_SET}/versions/draft/assets/"
r = requests.get(api_url, params={"path": ASSET_PATH})
r.raise_for_status()
results = r.json()["results"]
assert len(results) == 1, f"expected 1 asset, got {len(results)}"
asset_id = results[0]["asset_id"]
print("asset_id:", asset_id, "size GB:", results[0]["size"] / 1e9)

download_url = f"https://api.dandiarchive.org/api/dandisets/{DANDI_SET}/versions/draft/assets/{asset_id}/download/"
# GET-only presigned redirect: HEAD 403s, so grab Location from a GET without following.
r = requests.get(download_url, allow_redirects=False, stream=True)
s3_url = r.headers["Location"]
r.close()
print("S3 URL resolved:", s3_url[:80], "...")

disk_cache = remfile.DiskCache("/tmp/remfile_cache_replay02")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# --- Verify the key streams -------------------------------------------------
units = nwb["units"]
print("\nunits:", type(units), "n =", len(units))
print("metadata columns:", units.metadata_columns)
print("cell_type values:", np.unique(units.get_info("cell_type").values, return_counts=True))

pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print("\npos2d:", pos2d.shape, "t range:", pos2d.t[0], pos2d.t[-1], "median dt:", np.median(np.diff(pos2d.t[:1000])))
print("lin:", lin.shape, "t range:", lin.t[0], lin.t[-1])
print("lin valid fraction:", np.mean(~np.isnan(lin.values)))

epochs = nwb["epochs"]
print("\nepochs:\n", epochs)

states = nwb["states"]
print("\nstates labels:", np.unique(states["label"]))

# LFP structure (do not load data yet)
lfp = h5py_file["processing/ecephys/LFP/LFP"]
print("\nLFP data shape:", lfp["data"].shape, "dtype:", lfp["data"].dtype)
print("LFP conversion:", lfp["data"].attrs.get("conversion"))
print("LFP rate:", lfp["starting_time"].attrs.get("rate"), "start:", lfp["starting_time"][()])
