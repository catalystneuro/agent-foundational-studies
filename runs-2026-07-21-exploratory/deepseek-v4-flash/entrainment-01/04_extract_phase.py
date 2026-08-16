# %% [markdown]
# # 04 — Theta phase extraction and run epochs
# Bandpass-filter the reference LFP around the theta peak (6–11 Hz), compute
# instantaneous phase with the Hilbert transform, and define run bouts from
# 2-D speed during the maze epoch. Saves intermediates for the stats step.

# %%
import numpy as np
import remfile
import requests
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal

ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"
FS = 1250.0
rng = np.random.default_rng(0)

def resolve_s3_url(asset_id):
    dl = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    r = requests.get(dl, allow_redirects=False, timeout=120)
    return r.headers["Location"]

s3_url = resolve_s3_url(ASSET_ID)
disk_cache = remfile.DiskCache("/tmp/remfile_cache_entrainment01")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

d = np.load("theta_channel.npz", allow_pickle=False)
best_ch = int(d["best_ch"])
conversion = float(d["conversion"])
print("theta channel:", best_ch)

# %% [markdown]
# ## Theta-band LFP over the maze epoch
# %%
t_maze0, t_maze1 = 18079.5, 20147.0
pad_s = 30.0
i0 = int((t_maze0 - pad_s) * FS)
i1 = int((t_maze1 + pad_s) * FS)
raw = np.asarray(h5py_file["processing/ecephys/LFP/LFP/data"][i0:i1, best_ch], dtype=np.float64)
lfp_uV = raw * conversion * 1e6
lfp_t = np.arange(i0, i1) / FS
print("LFP segment:", lfp_t[0], "-", lfp_t[-1], "n =", len(lfp_uV))

# %% Hilbert phase
band = (6.0, 11.0)
sos = signal.butter(4, band, btype="band", fs=FS, output="sos")
theta_f = signal.sosfiltfilt(sos, lfp_uV)
phase = np.angle(signal.hilbert(theta_f))          # 0 = positive peak
print("phase n =", len(phase))

# %% Run bouts
pos = nwb["1.6mLinearMazeSpatialSeries"]
pt = np.asarray(pos.t)
px = np.asarray(pos["x"], dtype=float)
py = np.asarray(pos["y"], dtype=float)
valid = np.isfinite(px) & np.isfinite(py)
n = len(pt)
sp = np.zeros(n)
sp[1:-1] = np.hypot(px[2:] - px[:-2], py[2:] - py[:-2]) / (pt[2:] - pt[:-2])
center_ok = np.zeros(n, dtype=bool)
center_ok[1:-1] = valid[2:] & valid[:-2]
sp = sp * center_ok

# interpolate across short invalid gaps (< 0.5 s), long gaps -> 0
bad = ~center_ok
gap_idx = np.where(bad)[0]
if len(gap_idx):
    # find contiguous bad stretches
    splits = np.where(np.diff(gap_idx) > 1)[0] + 1
    chunks = np.split(gap_idx, splits)
    for c in chunks:
        dt_gap = pt[c[-1]] - pt[c[0]]
        if dt_gap < 0.5:
            lo_i = c[0] - 1 if c[0] > 0 else 0
            hi_i = c[-1] + 1 if c[-1] < n - 1 else n - 1
            if valid[lo_i] and valid[hi_i] and hi_i > lo_i:
                t_lo, t_hi = pt[lo_i], pt[hi_i]
                sp[c] = np.interp(pt[c], [t_lo, t_hi], [sp[lo_i], sp[hi_i]])

run_mask = sp >= 0.10

# convert run mask to Intervalset: runs with gaps < 0.3 s merged, min 1 s
dm = np.diff(run_mask.astype(np.int8))
starts = np.where(dm == 1)[0] + 1
ends = np.where(dm == -1)[0] + 1
if run_mask[0]:
    starts = np.concatenate([[0], starts])
if run_mask[-1]:
    ends = np.concatenate([ends, [n - 1]])
bouts = []
for s, e in zip(starts, ends):
    if len(bouts) and pt[s] - pt[bouts[-1][1]] < 0.3:
        bouts[-1][1] = e
    else:
        bouts.append([s, e])
bouts = [b for b in bouts if pt[b[1]] - pt[b[0]] >= 1.0]
run_start = np.array([pt[b[0]] for b in bouts])
run_end = np.array([pt[b[1]] for b in bouts])
run_epoch = nap.IntervalSet(start=run_start, end=run_end)
print("run bouts:", len(run_epoch), "total run time: %.1f s" % run_epoch.tot_length())

speed_tsd = nap.Tsd(t=pt, d=sp)

np.savez("phase_data.npz",
         lfp_t=lfp_t, lfp_uV=lfp_uV, theta_f=theta_f,
         phase=phase, run_start=run_start, run_end=run_end)
print("saved phase_data.npz")