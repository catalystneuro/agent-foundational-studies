"""Inspect drifting gratings table + units/electrodes structure."""
import numpy as np
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

ASSET_ID = "58703c97-c0a9-4736-b684-73c85c1a444a"
LINDI_URL = f"https://lindi.neurosift.org/dandi/dandisets/000021/assets/{ASSET_ID}/nwb.lindi.json"

local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(LINDI_URL, local_cache=local_cache)
io = NWBHDF5IO(file=f)
nwbfile = io.read()

# --- drifting gratings table (read columns directly; compound timeseries col breaks to_dataframe under LINDI)
tab = nwbfile.intervals["drifting_gratings_presentations"]
print("columns:", list(tab.colnames))
starts = tab["start_time"].data[:]
stops = tab["stop_time"].data[:]
ori = tab["orientation"].data[:]
tf = tab["temporal_frequency"].data[:]
print("n sweeps:", len(starts))
print("orientations:", np.unique(ori[~np.isnan(ori)]))
print("temporal freqs:", np.unique(tf[~np.isnan(tf)]))
print("n blank (NaN ori):", np.sum(np.isnan(ori)))
print("sweep durations (s):", np.unique(np.round(stops - starts, 3))[:10])

# --- units
units = nwbfile.units
print("\nunits columns:", list(units.colnames))
print("n units:", len(units["id"].data[:]) if hasattr(units["id"], 'data') else len(units.id[:]))
quality = units["quality"].data[:]
print("quality values:", np.unique(quality, return_counts=True))

# --- electrodes for location mapping
elec = nwbfile.electrodes
print("\nelectrodes columns:", list(elec.colnames))
loc = elec["location"].data[:]
print("electrode locations:", np.unique(loc))
