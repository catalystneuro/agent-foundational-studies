import remfile, h5py
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import pynapple as nap

url = "https://api.dandiarchive.org/api/dandisets/000128/versions/draft/assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
disk_cache = remfile.DiskCache('/tmp/remfile_cache_mcmaze')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

# Trials table as pandas
trials = nwbfile.intervals['trials'].to_dataframe()
print("trials columns:", list(trials.columns))
print(trials.head(8).to_string())
print("\nn trials:", len(trials), " success frac:", trials['success'].mean())
print("delay range (ms):", trials['delay'].min(), trials['delay'].max())
print("rt range (ms):", trials['rt'].min(), trials['rt'].max())

# target positions: ragged array via index
tp = nwbfile.intervals['trials']['target_pos'].data[:]   # (3807,2)
tpi = nwbfile.intervals['trials']['target_pos_index'].data[:]
# first target of each trial = target_pos[target_pos_index[i]] ... check ragged layout
print("\ntarget_pos_index head:", tpi[:8])
# active_target tells which of the targets is the real one
at = trials['active_target'].values
print("active_target unique:", np.unique(at))
# reconstruct per-trial target: rows between index[i] and index[i+1]
targets = []
for i in range(len(trials)):
    s = tpi[i]
    e = tpi[i+1] if i+1 < len(tpi) else len(tp)
    tgt_set = tp[s:e]
    targets.append(tgt_set[at[i]] if at[i] < len(tgt_set) else tgt_set[-1])
targets = np.array(targets)
print("unique targets (first 20):", np.unique(targets, axis=0)[:20])
print("n unique targets:", len(np.unique(targets, axis=0)))
ang = np.arctan2(targets[:,1], targets[:,0])
print("target radius (mm?) unique:", np.unique(np.round(np.hypot(targets[:,0], targets[:,1]),1)))
print("target angle (deg) unique:", np.sort(np.unique(np.round(np.degrees(ang),1))))

# Units
units = nwbfile.units.to_dataframe()
print("\nunits columns:", list(units.columns))
print(units.head(5).to_string())
print("heldout frac:", units['heldout'].mean() if 'heldout' in units else 'n/a')

# Behavior sampling
hv = nwbfile.processing['behavior']['hand_vel']
ts = hv.timestamps[:1000]
print("\nhand_vel dt median (s):", np.median(np.diff(ts)), " -> rate ~", 1/np.median(np.diff(ts)), "Hz")
print("session duration (h):", hv.timestamps[-1]/3600)
