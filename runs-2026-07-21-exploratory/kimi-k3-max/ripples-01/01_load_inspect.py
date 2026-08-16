# 01: Resolve DANDI S3 URL, stream the Achilles NWB, inspect structure
import requests
import h5py
import remfile
import numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"

# --- resolve the presigned S3 URL (GET without following redirects) ---
r = requests.get(
    f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/",
    params={"path": ASSET_PATH},
)
r.raise_for_status()
results = r.json()["results"]
assert len(results) == 1, results
asset_id = results[0]["asset_id"]
print("asset_id:", asset_id)

r2 = requests.get(
    f"https://api.dandiarchive.org/api/assets/{asset_id}/download/",
    allow_redirects=False,
)
s3_url = r2.headers["Location"]
print("S3 URL resolved:", s3_url[:100], "...")

with open("s3_url.txt", "w") as f:
    f.write(s3_url)

# --- stream with remfile + disk cache ---
disk_cache = remfile.DiskCache("/tmp/remfile_cache_ripples")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# --- inspect key streams ---
print("\n=== epochs ===")
epochs = nwb["epochs"]
print(epochs)
print("labels:", np.unique(epochs.label) if hasattr(epochs, "label") else "n/a")

print("\n=== states ===")
states = nwb["states"]
print(type(states))
print(states)

print("\n=== units ===")
units = nwb["units"]
print(units)
print("cell_type counts:", units.get_info("cell_type").value_counts())
print("location counts:", units.get_info("location").value_counts())

print("\n=== position ===")
pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print("pos2d:", pos2d.shape, "t range:", pos2d.t[0], pos2d.t[-1], "dt median:", np.median(np.diff(pos2d.t)))
print("lin:", lin.shape, "t range:", lin.t[0], lin.t[-1])

# --- LFP structure ---
lfp = h5py_file["processing/ecephys/LFP/LFP"]
print("\n=== LFP ===")
print("data shape:", lfp["data"].shape, "dtype:", lfp["data"].dtype)
print("conversion:", lfp["data"].attrs.get("conversion"))
print("starting_time attrs:", dict(lfp["starting_time"].attrs))
print("starting_time:", lfp["starting_time"][()])
