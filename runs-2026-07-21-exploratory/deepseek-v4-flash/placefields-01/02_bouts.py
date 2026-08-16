# %% [markdown]
# # Preprocessing: run bouts, direction, occupancy, tuning curves
# The linearized position is NaN except during on-track runs. We detect run
# bouts as contiguous valid stretches, merge over short gaps (<0.3 s), and
# filter by duration (>=1 s), path span (>=0.3 m), and speed (>=0.15 m/s).

# %%
import numpy as np
import h5py
import pynapple as nap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import remfile
from pynwb import NWBHDF5IO

s3_url = open('/tmp/achilles_url.txt').read().strip()
rem_file = remfile.File(s3_url, disk_cache=remfile.DiskCache('/tmp/remfile_cache'))
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

units = nwb['units']
lin = nwb['1.6mLinearMazeLinearizedTimeSeries']
lin_abs = np.asarray(lin.values)[:, 0]   # linearized position (m)
t_abs = np.asarray(lin.t)                # absolute session time (s)

# %% [markdown]
# ## Detect on-track run bouts
# %%
valid = ~np.isnan(lin_abs)

edge = np.diff(valid.astype(np.int8))
starts = np.flatnonzero(edge == 1) + 1
if valid[0]:
    starts = np.concatenate([[0], starts])
ends = np.flatnonzero(edge == -1) + 1
if valid[-1]:
    ends = np.concatenate([ends, [len(valid)]])

# merge runs separated by < ~0.3 s (12 samples at 39.06 Hz)
merged_s, merged_e = [], []
cur_s, cur_e = starts[0], ends[0]
for s, e in zip(starts[1:], ends[1:]):
    if s - cur_e < 12:
        cur_e = e
    else:
        merged_s.append(cur_s); merged_e.append(cur_e)
        cur_s, cur_e = s, e
merged_s.append(cur_s); merged_e.append(cur_e)

# per-bout stats pass filters: duration >= 1 s, span >= 0.3 m, median |v| >= 0.15 m/s
bout_dict = {}
for s_idx, e_idx in zip(merged_s, merged_e):
    x = lin_abs[s_idx:e_idx]
    t = t_abs[s_idx:e_idx]
    dur = t[-1] - t[0]
    if dur < 1.0:
        continue
    span = np.nanmax(x) - np.nanmin(x)
    if span < 0.3:
        continue
    dx = np.diff(x)
    dt = np.diff(t)
    v = dx / dt
    med_v = np.median(np.abs(v)) if len(v) else 0.0
    if med_v < 0.15:
        continue
    direction = int(np.sign(np.nanmedian(dx)))
    if direction == 0:
        continue
    bout_dict[len(bout_dict)] = {'start': t[0], 'end': t[-1], 'direction': direction,
                                 'duration': dur, 'span': span, 'speed': med_v}

nb = len(bout_dict)
bouts_iv = nap.IntervalSet(start=[b['start'] for b in bout_dict.values()],
                            end=[b['end'] for b in bout_dict.values()])
print('run bouts:', nb)
print('total run time (s):', float(np.sum(bouts_iv['end'] - bouts_iv['start'])))
n_pos = sum(b['direction'] == 1 for b in bout_dict.values())
n_neg = sum(b['direction'] == -1 for b in bout_dict.values())
print(f'direction counts: +dir {n_pos}, -dir {n_neg}')

# ---- occupancy across the track: histogram of valid linearized samples ----
NBIN = 50
edges = np.linspace(0.0, 1.6, NBIN + 1)
in_bouts = np.zeros(len(lin_abs), dtype=bool)
for b in bout_dict.values():
    in_bouts |= (t_abs >= b['start']) & (t_abs < b['end'])
occ_idx = np.flatnonzero(in_bouts & valid)
occ_hist, _ = np.histogram(lin_abs[occ_idx], bins=edges)
occ_sec = occ_hist / 39.0625  # seconds of occupancy per bin
print('occupancy per bin (s), mean:', occ_sec.mean().round(2), 'min:', occ_sec.min().round(2), 'max:', occ_sec.max().round(2))