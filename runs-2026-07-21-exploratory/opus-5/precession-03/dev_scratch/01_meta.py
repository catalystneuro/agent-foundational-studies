"""Inspect metadata: epochs, states, electrodes, units, sampling rates."""
import h5py
import remfile
import numpy as np
from collections import Counter

ASSET = "5349c68b-c0a7-46c0-9900-cda050722fa4"
URL = f"https://api.dandiarchive.org/api/dandisets/000044/versions/0.250624.0426/assets/{ASSET}/download/"
h5 = h5py.File(remfile.File(URL, disk_cache=remfile.DiskCache("/tmp/remfile_cache")), "r")

print("session_description:", h5["session_description"][()])
print("related_publications:", h5["general/related_publications"][:])
print("subject:", {k: h5[f"general/subject/{k}"][()] for k in h5["general/subject"]})

print("\n--- epochs ---")
for lbl, a, b in zip(h5["intervals/epochs/label"][:], h5["intervals/epochs/start_time"][:], h5["intervals/epochs/stop_time"][:]):
    print(f"{lbl!r:30s} {a:10.1f} {b:10.1f}  dur={b-a:8.1f}s")

print("\n--- behavior/states labels ---")
sl = h5["processing/behavior/states/label"][:]
print(Counter([s.decode() if isinstance(s, bytes) else s for s in sl]))
print("first 6:", list(zip(sl[:6], h5["processing/behavior/states/start_time"][:6], h5["processing/behavior/states/stop_time"][:6])))

print("\n--- LFP ---")
lfp = h5["processing/ecephys/LFP/LFP"]
print("attrs:", dict(lfp.attrs))
print("data attrs:", dict(lfp["data"].attrs))
print("starting_time attrs:", dict(lfp["starting_time"].attrs))

print("\n--- position ---")
for grp in ["1.6mLinearMazePosition/1.6mLinearMazeSpatialSeries", "1.6mLinearMazeLinearizedPosition/1.6mLinearMazeLinearizedTimeSeries"]:
    g = h5[f"processing/behavior/{grp}"]
    print(grp, "starting_time attrs:", dict(g["starting_time"].attrs), "data attrs:", dict(g["data"].attrs))
    print("  reference_frame:", g["reference_frame"][()])
    print("  data[:5]:", g["data"][:5].ravel())

print("\n--- electrodes ---")
loc = h5["general/extracellular_ephys/electrodes/location"][:]
print(Counter([l.decode() if isinstance(l, bytes) else l for l in loc]))
print("group_name:", Counter([x.decode() if isinstance(x,bytes) else x for x in h5["general/extracellular_ephys/electrodes/group_name"][:]]))
print("LFP electrodes index:", h5["processing/ecephys/LFP/LFP/electrodes"][:8], "...")

print("\n--- units ---")
ct = h5["units/cell_type"][:]
ul = h5["units/location"][:]
print("cell_type:", Counter([x.decode() if isinstance(x,bytes) else x for x in ct]))
print("location:", Counter([x.decode() if isinstance(x,bytes) else x for x in ul]))
print("unit cols:", list(h5["units"]))
