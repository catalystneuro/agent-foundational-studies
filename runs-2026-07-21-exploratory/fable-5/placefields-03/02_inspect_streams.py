"""Inspect the individual data streams of the Achilles session before analysis."""

import numpy as np
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

LINDI_URL = (
    "https://lindi.neurosift.org/dandi/dandisets/000044/assets/"
    "5349c68b-c0a7-46c0-9900-cda050722fa4/nwb.lindi.json"
)

local_cache = lindi.LocalCache(cache_dir="./lindi_cache")
f = lindi.LindiH5pyFile.from_lindi_file(LINDI_URL, local_cache=local_cache)
io = NWBHDF5IO(file=f, mode="r")
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print("### EPOCHS ###")
epochs = nwb["epochs"]
print(epochs)
print()

print("### STATES ###")
states = nwb["states"]
print(states)
print()

print("### UNITS ###")
units = nwb["units"]
print(units)
print()
print("cell_type counts:")
ct = units.get_info("cell_type")
vals, counts = np.unique(np.asarray(ct, dtype=object).astype(str), return_counts=True)
for v, c in zip(vals, counts):
    print(f"   {v}: {c}")
print("location counts:")
loc = np.asarray(units.get_info("location"), dtype=object).astype(str)
vals, counts = np.unique(loc, return_counts=True)
for v, c in zip(vals, counts):
    print(f"   {v}: {c}")
print()

print("### POSITION (2D) ###")
pos2d = nwb["1.6mLinearMazeSpatialSeries"]
print(pos2d)
print("time range:", pos2d.index[0], "->", pos2d.index[-1])
print("dt median:", np.median(np.diff(pos2d.index)))
d = np.asarray(pos2d.values)
print("x range:", np.nanmin(d[:, 0]), np.nanmax(d[:, 0]))
print("y range:", np.nanmin(d[:, 1]), np.nanmax(d[:, 1]))
print("n NaN rows:", int(np.isnan(d).any(axis=1).sum()))
print()

print("### LINEARIZED POSITION ###")
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print(lin)
dl = np.asarray(lin.values).ravel()
print("range:", np.nanmin(dl), np.nanmax(dl), "n NaN:", int(np.isnan(dl).sum()))
print("time range:", lin.index[0], "->", lin.index[-1])
print()

print("### LFP ###")
lfp = nwb["LFP"]
print(lfp)
print("lfp rate approx:", 1.0 / np.median(np.diff(lfp.index[:1000])))
print()

print("### coverage of position within epochs ###")
for i in range(len(epochs)):
    ep = epochs[i]
    lab = epochs.get_info("label")[i] if "label" in epochs.metadata_columns else "?"
    n = len(lin.restrict(ep))
    print(f"  epoch {i} [{lab}] {ep.start[0]:.1f}-{ep.end[0]:.1f}s "
          f"({ep.end[0]-ep.start[0]:.0f}s), n position samples = {n}")
