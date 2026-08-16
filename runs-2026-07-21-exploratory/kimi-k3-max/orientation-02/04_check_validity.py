"""Check overlap of drifting grating presentations with invalid times; inspect blanks."""
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import pandas as pd
import numpy as np

DANDISET = "000021"
VERSION = "0.251116.2246"
ASSET_ID = "58703c97-c0a9-4736-b684-73c85c1a444a"
URL = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/{ASSET_ID}/download/"

disk_cache = remfile.DiskCache('/tmp/remfile_cache_orientation')
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

dg = nwbfile.intervals['drifting_gratings_presentations'].to_dataframe()
inv = nwb['invalid_times'].values  # (n, 2) array of start/end

print("Invalid intervals:\n", inv)

# Overlap test: presentation [s, e) overlaps invalid [a, b) if s < b and e > a
starts = dg['start_time'].values
stops = dg['stop_time'].values
overlap = np.zeros(len(dg), dtype=bool)
for a, b in inv:
    overlap |= (starts < b) & (stops > a)
print("\nPresentations overlapping invalid times:", overlap.sum(), "of", len(dg))
print("Orientation distribution of excluded rows:")
print(dg.loc[overlap, 'orientation'].value_counts(dropna=False))

# Blank sweeps
blank = dg[dg['orientation'].isna()]
print("\nBlank sweep stimulus_name:", blank['stimulus_name'].unique())
print("Blank sweep durations:", (blank['stop_time'] - blank['start_time']).describe())
print("Blank rows per block:\n", blank['stimulus_block'].value_counts())

# valid dg after exclusion
valid = dg[~overlap]
print("\nValid presentations:", len(valid))
print("Valid per orientation (excluding blanks):")
print(valid['orientation'].value_counts(dropna=False).sort_index())
print("\nValid blank sweeps:", valid['orientation'].isna().sum())

# Check inter-presentation gap (is there a gap between stop and next start?)
valid_sorted = valid.sort_values('start_time')
gaps = valid_sorted['start_time'].values[1:] - valid_sorted['stop_time'].values[:-1]
print("\nInter-presentation gaps (s): min={:.3f} median={:.3f} max={:.3f}".format(
    gaps.min(), np.median(gaps), gaps.max()))
