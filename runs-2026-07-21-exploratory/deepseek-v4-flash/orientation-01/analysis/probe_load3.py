"""Check static gratings parameter values and quality info."""
import lindi
from pynwb import NWBHDF5IO
import numpy as np
import pandas as pd

local_cache = lindi.LocalCache(cache_dir='/tmp/lindi_cache_orient')
url = 'https://lindi.neurosift.org/dandi/dandisets/000021/assets/58703c97-c0a9-4736-b684-73c85c1a444a/nwb.lindi.json'
f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
io = NWBHDF5IO(file=f)
nwbfile = io.read()

sg = nwbfile.intervals['static_gratings_presentations'].to_dataframe(exclude={'timeseries'})
print('static_gratings:')
print('  orientations:', sorted(sg['orientation'].dropna().unique()))
print('  spatial_freqs:', sorted(sg['spatial_frequency'].dropna().unique()))
print('  sizes:', sorted(sg['size'].dropna().unique()))
print('  durations (s):', np.round((sg.stop_time - sg.start_time).describe()).to_dict())
print('  time range:', sg.start_time.min(), sg.stop_time.max())

inv = nwbfile.intervals['invalid_times'].to_dataframe()
print('\ninvalid_times:')
print(inv[['start_time', 'stop_time']])

# overlap check
ov = 0
for _, r in inv.iterrows():
    ov += ((sg.start_time >= r.start_time) & (sg.start_time < r.stop_time)).sum()
# also overlaps where presentation straddles
for _, r in inv.iterrows():
    ov += ((sg.start_time < r.start_time) & (sg.stop_time > r.start_time)).sum()
print('\nstatic presentations overlapping invalid_times:', ov)

# units quality
ut = nwbfile.units
q = ut['quality'][:]
print('\nquality values:', np.unique(q, return_counts=True))
fr = ut['firing_rate'][:]
print('firing_rate stats:', np.nanpercentile(fr, [0, 25, 50, 75, 100]))

# electrodes locations full list and per-probe
loc = nwbfile.electrodes['location'][:]
print('\nlocations:', np.unique(loc, return_counts=True))
pid = nwbfile.electrodes['probe_id'][:]
print('\nprobe_ids:', np.unique(pid, return_counts=True))