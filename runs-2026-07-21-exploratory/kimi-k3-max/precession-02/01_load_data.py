# 01_load_data.py — stream the Achilles session from DANDI 000044 and validate data streams
import numpy as np
import requests
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"

# Resolve the asset's download URL via the DANDI API, then grab the presigned S3 URL
# from the redirect Location header (HEAD requests 403 on this endpoint; use GET).
api_url = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/"
r = requests.get(api_url, params={"path": ASSET_PATH})
r.raise_for_status()
asset = r.json()["results"][0]
download_url = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/{asset['asset_id']}/download/"
r = requests.get(download_url, allow_redirects=False)
s3_url = r.headers["Location"]
print("S3 URL resolved:", s3_url[:100], "...")

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# Inspect the pieces we need
units = nwb["units"]
print("\nunits:", type(units), "n =", len(units))
print("cell_type counts:", units.get_info("cell_type").value_counts().to_dict())

pos = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print("\nposition:", pos.shape, "rate attr check -> timestamps span",
      pos.t[0], "to", pos.t[-1], "s (BUG: rate field holds the period)")
print("linearized:", lin.shape)

lfp = nwbfile.processing["ecephys"]["LFP"]["LFP"]
print("\nLFP:", lfp.data.shape, lfp.data.dtype, "conversion", lfp.conversion)
print("LFP rate:", lfp.starting_time.attrs["rate"], "Hz, t0 =", lfp.starting_time.value)

epochs = nwb["epochs"]
print("\nepochs labels:", np.unique(epochs.label.values) if hasattr(epochs, "label") else epochs)
print(epochs)
