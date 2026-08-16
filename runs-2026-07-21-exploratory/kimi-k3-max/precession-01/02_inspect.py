"""Step 2: inspect units, epochs, position, and LFP structure."""
import numpy as np
import requests
import remfile
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap

ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"
dl = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
r = requests.get(dl, allow_redirects=False)
s3_url = r.headers["Location"]

disk_cache = remfile.DiskCache("/tmp/remfile_cache_precession")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

units = nwb["units"]
print("n units:", len(units))
print("unit keys (first 10):", list(units.keys())[:10])
print("cell_type values:", np.unique(units.get_info("cell_type").values))
print("location values:", np.unique(units.get_info("location").values))

epochs = nwb["epochs"]
print("\nepochs:")
print(epochs)

pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print("\npos2d:", pos2d.shape, pos2d.rate, "Hz, t range", pos2d.index[0], pos2d.index[-1])
print("lin:", lin.shape, "t range", lin.index[0], lin.index[-1])

lfp = nwb["LFP"]
print("\nLFP:", type(lfp), lfp.shape, "rate:", lfp.rate)
print("LFP t range:", lfp.index[0], lfp.index[-1])

# check LFP conversion factor in the raw h5
es = h5py_file["processing/ecephys/LFP/LFP"]
print("LFP conversion:", es.attrs.get("conversion"), "unit:", es.attrs.get("unit"))
print("starting_time attrs:", dict(es["starting_time"].attrs))
