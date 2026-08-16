import h5py, remfile, numpy as np
from pynwb import NWBHDF5IO
import pynapple as nap

s3_url = open("s3_url.txt").read().strip()
disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
print("remfile opened")
h5py_file = h5py.File(rem_file, "r")
print("h5py opened")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print("NWB loaded")

units = nwb["units"]
print("units keys:", list(units.keys())[:10], "... n =", len(units.keys()))
info = units.get_info("cell_type")
print("cell_type:", info.unique())
ct = np.asarray(info.values)
print("n exc:", (ct == "Excitatory").sum(), "inh:", (ct=="Inhibitory").sum())

# positions
pos2d = nwb["1.6mLinearMazeSpatialSeries"]
print("pos2d shape:", pos2d.shape, "rate:", getattr(pos2d, "rate", None))
print("pos2d t first:", pos2d.t[0], "last:", pos2d.t[-1], "n:", len(pos2d.t))
dt = np.diff(pos2d.t)
print("dt median:", np.median(dt), "min:", dt.min(), "max:", dt.max())

lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print("lin shape:", lin.shape, "rate:", getattr(lin, "rate", None))
lv = np.asarray(lin.values).ravel()
print("lin valid frac:", np.isfinite(lv).mean(), "range:", np.nanmin(lv[lv>-1e6]), np.nanmax(lv))

epochs = nwb["epochs"]
print("epochs:\n", epochs.as_dataframe().head())
maze = epochs[epochs.label == "MazeEpoch"]
print("maze epoch:", maze)
print("LFP:", nwb["LFP"])
