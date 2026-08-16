"""Place field analysis on the Achilles linear track (DANDI 000044).

Pipeline:
  1. Reconstruct position timestamps (rate field bug: period stored as rate).
  2. Build run bouts from the linearized position series.
  3. Per-direction 1D tuning curves on the linearized axis.
  4. Skaggs spatial information + circular time-shift shuffle null.
  5. Place-cell identification (excitatory, rate/peak/SI thresholds).
  6. Directional selectivity (correlation of pos/neg maps).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py, remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import scipy.stats as st

RNG = np.random.default_rng(7)
NSHUF = 500
NBINS = 50

# ---------- load ----------
with open("_asset.json") as f:
    info = json.load(f)
rem_file = remfile.File(info["s3_url"], disk_cache=remfile.DiskCache("/tmp/remfile_cache_000044"))
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f)
nwb = nap.NWBFile(io.read())

units = nwb["units"]
ct = units.get_info("cell_type")
exc = ct[ct == "excitatory"].index.tolist()
inh = ct[ct == "inhibitory"].index.tolist()
print(f"units: {len(units)} = {len(exc)} exc + {len(inh)} inh")

# ---------- position: repair timestamps ----------
pos_raw = np.asarray(nwb["1.6mLinearMazeSpatialSeries"].values)          # (N,2) x,y
lin_raw = np.asarray(nwb["1.6mLinearMazeLinearizedTimeSeries"].values).ravel()  # (N,)
n = len(pos_raw)
t0 = 18079.5  # MazeEpoch start
FS = 39.0625
t = t0 + np.arange(n) / FS
print(f"pos n={n}, span={t[-1]-t[0]:.1f} s, expected maze epoch 2067.5 s")

pos = nap.TsdFrame(t=t, d=pos_raw)
lin = nap.Tsd(t=t, d=lin_raw)

valid_lin = np.isfinite(lin_raw)
print(f"lin valid: {valid_lin.mean()*100:.1f}% ({valid_lin.sum()} samples)")
pos_valid = np.all(np.isfinite(pos_raw), axis=1)
print(f"2D pos valid: {pos_valid.mean()*100:.1f}% ({(~np.all(np.isfinite(pos_raw),axis=1)&valid_lin).sum()} lin-good/pos-bad)")

# ---------- run bouts from linearized series ----------
good = np.flatnonzero(valid_lin)
gaps = np.diff(good)
bout_edges = np.where(gaps > 0.3 * FS)[0]  # split where gap > 0.3 s
starts = np.concatenate([[good[0]], good[bout_edges + 1]])
ends = np.concatenate([good[bout_edges], [good[-1]]])

bouts = []
for s, e in zip(starts, ends):
    dur = (t[e] - t[s])
    seg = lin_raw[s:e + 1]
    span = np.abs(seg[-1] - seg[0])
    speed = np.median(np.abs(np.diff(seg))) * FS
    if dur >= 1.0 and span > 0.3 and speed > 0.15:
        bouts.append((s, e, "pos" if seg[-1] > seg[0] else "neg"))
bouts = np.array(bouts, dtype=object)
n_pos = (bouts[:, 2] == "pos").sum()
n_neg = (bouts[:, 2] == "neg").sum()
run_sec = sum((e - s) / FS for s, e, _ in bouts)
print(f"bouts: {len(bouts)} ({n_pos} pos, {n_neg} neg), run time {run_sec:.1f} s")

# IntervalSet of run bouts (per direction too)
run_ep = nap.IntervalSet(start=[t[b[0]] for b in bouts], end=[t[b[1]] for b in bouts])
run_ep_pos = nap.IntervalSet(start=[t[b[0]] for b in bouts if b[2]=="pos"],
                             end=[t[b[1]] for b in bouts if b[2] == "pos"])
run_ep_neg = nap.IntervalSet(start=[t[b[0]] for b in bouts if b[2] == "neg"],
                             end=[t[b[1]] for b in bouts if b[2] == "neg"])

# run-time (tau) coordinate: concatenated along-bout time axis
tau = np.full(n, np.nan)
cur = 0.0
for s, e, d in bouts:
    tau[s:e + 1] = cur + np.arange(e - s + 1) / FS
    cur = tau[e]
run_total = cur
print(f"concatenated run time: {run_total:.1f} s")