"""Load the Achilles-10252013 session from DANDI 000044 and validate streams."""
import numpy as np
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/7632d81b-2819-473d-8946-34dc939e6028"
CACHE = "/tmp/remfile_cache_placefields02"

disk_cache = remfile.DiskCache(CACHE)
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# --- units ---
units = nwb["units"]
print("\nunits:", type(units), len(units))
print("unit keys (first 10):", list(units.keys())[:10])
ct = units.get_info("cell_type")
print("cell_type counts:\n", ct.value_counts())

# --- position ---
pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print("\npos2d:", pos2d.shape, "t range", pos2d.t[0], pos2d.t[-1])
print("lin:", lin.shape, "t range", lin.t[0], lin.t[-1])
dt = np.diff(pos2d.t[:1000])
print("pos dt: median", np.median(dt), "min", dt.min(), "max", dt.max())

# --- epochs ---
epochs = nwb["epochs"]
print("\nepochs:\n", epochs.as_dataframe() if hasattr(epochs, "as_dataframe") else epochs)

np.savez(
    "cache/session_meta.npz",
    pos_t=pos2d.t, pos_xy=np.asarray(pos2d.values), lin_t=lin.t, lin=np.asarray(lin.values),
)
print("\nsaved cache/session_meta.npz")
