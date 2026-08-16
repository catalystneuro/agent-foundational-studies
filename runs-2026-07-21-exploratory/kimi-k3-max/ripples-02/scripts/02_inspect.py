"""02_inspect.py — validate each data stream: units, epochs, states, position, LFP."""
import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

s3_url = open("scripts/s3_url.txt").read().strip()
disk_cache = remfile.DiskCache("/tmp/remfile_cache_ripples02")
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=disk_cache), "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwb = nap.NWBFile(io.read())

print("=== epochs ===")
print(nwb["epochs"])
print("\n=== states ===")
print(nwb["states"])

units = nwb["units"]
print("\n=== units ===")
print(units)
print("metadata columns:", units.metadata_columns)
ct = units.get_info("cell_type")
print("cell_type counts:\n", ct.value_counts())

pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print("\n=== position ===")
print("pos2d:", pos2d.shape, "t range", pos2d.t[0], pos2d.t[-1])
print("pos2d dt median:", np.median(np.diff(pos2d.t[:1000])))
print("lin:", lin.shape, "t range", lin.t[0], lin.t[-1])

lfp = nwb["LFP"]
print("\n=== LFP ===")
print(lfp)
print("LFP t range:", lfp.t[0], lfp.t[-1], "n samples:", len(lfp.t))
dt = np.median(np.diff(lfp.t[:2000]))
print("LFP dt:", dt, "-> fs =", 1.0 / dt)

# check the raw h5 for LFP conversion factor
dset = h5py_file["processing/ecephys/LFP/LFP/data"]
print("LFP dataset shape:", dset.shape, "dtype:", dset.dtype)
print("conversion:", dset.attrs.get("conversion"))
st = h5py_file["processing/ecephys/LFP/LFP/starting_time"]
print("starting_time attrs:", dict(st.attrs))
