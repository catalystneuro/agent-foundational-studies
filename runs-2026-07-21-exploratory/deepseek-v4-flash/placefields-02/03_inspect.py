"""Inspect units, position, epochs of the Achilles session."""
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

s3_url = open("s3_url.txt").read().strip()
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# ---- units ----
u = nwb["units"]
print("n units:", len(u))
print("keys (first 15):", list(u.keys())[:15])
print("cell_type counts:", u.get_info("cell_type").value_counts().to_dict())
print("location counts:", u.get_info("location").value_counts().to_dict())
print("metadata columns: rate, location, shank_id, cell_type")
try:
    print("group counts:", u.get_info("group").value_counts().to_dict() if len(u.get_info("group")) else "n/a")
except Exception as e:
    print("group info err:", e)

# ---- rate per unit
print("\nrate stats:", u.rate.min(), u.rate.max(), u.rate.median())

# ---- epochs
ep = nwb["epochs"]
print("\nepochs:")
print(ep.as_dataframe())
df = ep.as_dataframe()
print("\nlabels:", list(ep.label)) if hasattr(ep, "label") else print("labels:", ep["label"])

# ---- position
pos2d = nwb["1.6mLinearMazeSpatialSeries"]
poslin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print("\npos2d rate attr:", pos2d.rate, "n:", len(pos2d), "t0:", pos2d.time_support.start[0] if len(pos2d.time_support) else None)
print("poslin rate attr:", poslin.rate, "n:", len(poslin))

# verify the timestamps are sane (rate attr should be Hz, 39.06)
dt = np.diff(np.asarray(pos2d.t))
print("median dt:", np.median(dt), "min:", dt.min(), "max:", dt.max())
print("pos2d time span:", pos2d.t[0], pos2d.t[-1], "n:", len(pos2d.t))

# fraction of valid linearized positions during the maze epoch
maze = ep.as_dataframe()
maze_ep = ep[ep.label == "MazeEpoch"] if hasattr(ep, "label") else None
print("\nmaze epoch:", maze[maze["label"] == "MazeEpoch"][["start", "end"]].values if "label" in maze.columns else maze)

# NaN fraction of linearized data overall
lin_raw = np.asarray(poslin.values).ravel()
print("lin NaN fraction overall:", np.isnan(lin_raw).mean())

io.close()