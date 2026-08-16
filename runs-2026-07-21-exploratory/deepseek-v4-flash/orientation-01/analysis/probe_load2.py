"""Probe session structure without materializing full data."""
import lindi
from pynwb import NWBHDF5IO
import numpy as np

local_cache = lindi.LocalCache(cache_dir='/tmp/lindi_cache_orient')
url = 'https://lindi.neurosift.org/dandi/dandisets/000021/assets/58703c97-c0a9-4736-b684-73c85c1a444a/nwb.lindi.json'
f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
io = NWBHDF5IO(file=f)
nwbfile = io.read()

print('=== INTERVAL TABLES (names + colnames) ===')
for k in nwbfile.intervals.keys():
    t = nwbfile.intervals[k]
    print(f'{k}: nrows={len(t)}, cols={list(t.colnames)}')

print('\n=== UNITS TABLE ===')
ut = nwbfile.units
print(f'n_units={len(ut)}, cols={list(ut.colnames)}')

# Small columns only
for c in ['peak_channel', 'electrode_group']:
    if c in ut.colnames:
        data = ut[c][:]
        print(f'  {c}: dtype={data.dtype}, first 5={data[:5]}')

print()
print('=== ELECTRODES ===')
et = nwbfile.electrodes
print(f'n=rows={len(et)}, cols={list(et.colnames)}')
for c in ['location']:
    if c in et.colnames:
        data = et[c][:]
        print(f'  first 5 locations: {data[:5]}')

print()
print('session:', nwbfile.session_description[:100])
# Session type / stimulus blocks
print('stimulus:', [s.name for s in nwbfile.stimulus][:20])