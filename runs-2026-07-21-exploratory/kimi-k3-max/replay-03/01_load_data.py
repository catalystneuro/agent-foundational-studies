"""Step 1: Resolve and load the Achilles session from DANDI 000044, validate streams."""
import requests
import h5py
import remfile
import numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000044"
SESSION_SUBSTR = "sub-Achilles_ses-Achilles-10252013_behavior+ecephys"

# --- Resolve the asset URL via the DANDI API ---
api = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/"
assets = []
url = api
while url:
    r = requests.get(url).json()
    assets.extend(r["results"])
    url = r["next"]
print(f"{len(assets)} assets in dandiset {DANDISET}")

match = [a for a in assets if SESSION_SUBSTR in a["path"]]
assert len(match) == 1, f"expected 1 match, got {len(match)}"
asset_id = match[0]["asset_id"]  # NOTE: key is asset_id, not identifier
print("asset:", match[0]["path"], asset_id)

# The /download/ endpoint redirects to a presigned S3 URL; HEAD 403s, so grab
# the Location header from a GET without following redirects.
dl = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
r = requests.get(dl, allow_redirects=False)
s3_url = r.headers["Location"]
print("S3 URL resolved:", s3_url[:100], "...")

# --- Stream with remfile + local disk cache ---
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# --- Validate key streams ---
units = nwb["units"]
print("\nunits:", type(units), len(units), "cells")
print("metadata columns:", units.metadata_columns)
print("cell_type counts:", units.get_info("cell_type").value_counts().to_dict())

pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print("\npos2d:", pos2d.shape, "t range", pos2d.t[0], pos2d.t[-1])
dt = np.diff(pos2d.t)
print("pos2d dt: median", np.median(dt), "min", dt.min(), "max", dt.max())
print("lin valid fraction:", np.mean(~np.isnan(lin.values)))

epochs = nwb["epochs"]
print("\nepochs:\n", epochs.as_dataframe())

states = nwb["states"]
print("\nstates label counts:", states["label"].value_counts().to_dict())

# LFP structure (do not load the full array)
lfp = h5py_file["processing/ecephys/LFP/LFP"]
print("\nLFP data:", lfp["data"].shape, lfp["data"].dtype, "chunks", lfp["data"].chunks)
print("LFP conversion:", lfp["conversion"] if "conversion" in lfp else lfp["data"].attrs.get("conversion"))
sr = lfp["starting_time"].attrs["rate"]
print("LFP rate:", sr)

np.savez_compressed(
    "session_meta.npz",
    s3_url=np.array([s3_url]),
)
print("\nOK")
