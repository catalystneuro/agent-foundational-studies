"""Theta phase entrainment analysis on the Achilles-10252013 maze epoch.

Steps:
1. Load maze-epoch LFP (reference channel) with buffer, bandpass 4-12 Hz, Hilbert phase.
2. Compute running speed from 2D position; define running bouts (speed > 10 cm/s).
3. Restrict unit spike trains to running bouts; interpolate theta phase at spike times.
4. Per-unit MRL, Rayleigh p, preferred phase; random-time control.
5. Save results to npz/csv for the figure script.
"""
import remfile, h5py
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

rng = np.random.default_rng(42)

# ---------------- load ----------------
s3_url = open("s3_url.txt").read().strip()
disk_cache = remfile.DiskCache("/tmp/remfile_cache_entrainment03")
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=disk_cache), "r")
nwbfile = NWBHDF5IO(file=h5py_file).read()
nwb = nap.NWBFile(nwbfile)

es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
fs = es.rate
conv = es.conversion
best_ch = int(np.load("theta_channel_selection.npz")["best_ch"])
print("reference channel:", best_ch)

epochs = nwb["epochs"]
maze = epochs[epochs.label == "MazeEpoch"]
maze_start, maze_end = float(maze.start[0]), float(maze.end[0])
print(f"maze epoch: {maze_start}-{maze_end} s")

# ---------------- LFP: load with 10 s buffer, filter, phase ----------------
buf = 10.0
i0 = int((maze_start - buf) * fs)
i1 = int((maze_end + buf) * fs)
lfp = es.data[i0:i1, best_ch].astype(np.float64) * conv  # volts
t_lfp = np.arange(i0, i1) / fs
print("LFP samples loaded:", len(lfp))

b, a = signal.butter(4, [4, 12], btype="bandpass", fs=fs)
lfp_filt = signal.filtfilt(b, a, lfp)
analytic = signal.hilbert(lfp_filt)
phase = np.angle(analytic)  # 0 = peak of filtered LFP, +/-pi = trough

# trim buffer
keep = (t_lfp >= maze_start) & (t_lfp <= maze_end)
t_lfp, lfp, lfp_filt, phase = t_lfp[keep], lfp[keep], lfp_filt[keep], phase[keep]

# ---------------- speed and running bouts ----------------
pos = nwb["1.6mLinearMazeSpatialSeries"]
xy = pos.values  # (n, 2), meters
print("position range:", np.nanmin(xy, axis=0), np.nanmax(xy, axis=0))

# DATA BUG: the SpatialSeries stores the sampling *period* (0.0256 s) in the
# `rate` field, so pynapple's timestamps are spaced 39 s apart. The true
# sampling rate is 1/0.0256 = 39.06 Hz and the series spans the maze epoch.
ss = nwbfile.processing["behavior"]["1.6mLinearMazePosition"]["1.6mLinearMazeSpatialSeries"]
fs_pos = 1.0 / float(ss.rate)  # stored 'rate' is actually the period in seconds
print(f"position sampling rate: {fs_pos:.2f} Hz")
pt = maze_start + np.arange(xy.shape[0]) / fs_pos
pxy = xy
valid = np.isfinite(pxy).all(axis=1)
print(f"valid position samples in maze: {valid.mean()*100:.1f}%")

# speed from consecutive valid samples
dt = np.diff(pt)
dxy = np.diff(pxy, axis=0)
step = np.linalg.norm(dxy, axis=1)
ok = valid[:-1] & valid[1:] & (dt > 0)
spd_t = (pt[:-1] + pt[1:]) / 2
spd = np.full_like(spd_t, np.nan, dtype=np.float64)
spd[ok] = step[ok] / dt[ok]  # m/s

# interpolate over short tracking gaps only (< 0.5 s); treat long dropouts as
# stationary (speed 0) rather than carrying speed across them
okv = np.isfinite(spd)
spd_i = np.interp(spd_t, spd_t[okv], spd[okv])
inv = ~valid
edges_inv = np.diff(inv.astype(int))
g_starts = np.where(edges_inv == 1)[0] + 1
g_ends = np.where(edges_inv == -1)[0] + 1
if inv[0]:
    g_starts = np.r_[0, g_starts]
if inv[-1]:
    g_ends = np.r_[g_ends, len(inv)]
for gs, ge in zip(g_starts, g_ends):
    if (ge - gs) / fs_pos >= 0.5:
        in_gap = (spd_t >= pt[gs]) & (spd_t <= pt[min(ge, len(pt) - 1)])
        spd_i[in_gap] = 0.0
sig = 0.25 * fs_pos
w = signal.windows.gaussian(int(sig * 8) | 1, std=sig)
w /= w.sum()
spd_s = np.convolve(spd_i, w, mode="same")
print(f"position sampling: {fs_pos:.1f} Hz; median smoothed speed (maze): {np.median(spd_s)*100:.1f} cm/s")

# running bouts: smoothed speed > 10 cm/s, min bout 0.5 s, merge gaps < 0.5 s
thr = 0.10
above = spd_s > thr
edges = np.diff(above.astype(int))
starts = spd_t[:-1][edges == 1]
ends = spd_t[:-1][edges == -1]
if above[0]:
    starts = np.r_[spd_t[0], starts]
if above[-1]:
    ends = np.r_[ends, spd_t[-1]]
bouts = np.column_stack([starts, ends])
# merge bouts separated by < 0.5 s
merged = []
for s, e in bouts:
    if merged and s - merged[-1][1] < 0.5:
        merged[-1][1] = e
    else:
        merged.append([s, e])
bouts = np.array([b for b in merged if b[1] - b[0] >= 0.5])
run_ep = nap.IntervalSet(start=bouts[:, 0], end=bouts[:, 1])
run_total = float(np.sum(bouts[:, 1] - bouts[:, 0]))
print(f"running bouts: {len(bouts)}, total {run_total:.0f} s of {maze_end-maze_start:.0f} s maze epoch")

# ---------------- spike phases ----------------
units = nwb["units"]
unit_ids = list(units.keys())
cell_type = units.get_info("cell_type")
location = units.get_info("location")

# unit-circle interpolation of phase (avoids wraparound artifacts)
cos_p, sin_p = np.cos(phase), np.sin(phase)

def phases_at(times):
    c = np.interp(times, t_lfp, cos_p)
    s = np.interp(times, t_lfp, sin_p)
    return np.arctan2(s, c)

def rayleigh_p(n, R):
    """Zar's approximation for the Rayleigh test p-value."""
    z = n * R**2
    p = np.exp(-z)
    if n < 50:
        p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - (n * R) ** 2)) - (1 + 2 * n))
    return min(p, 1.0)

rows = []
spike_phases_store = {}
for uid in tqdm(unit_ids, desc="units"):
    spk = units[uid].restrict(run_ep)
    n = len(spk)
    if n < 50:
        continue
    ph = phases_at(spk.t)
    R = np.abs(np.mean(np.exp(1j * ph)))
    p = rayleigh_p(n, R)
    pref = np.angle(np.mean(np.exp(1j * ph)))
    dur = run_total
    rows.append(dict(unit=uid, cell_type=cell_type[uid], location=location[uid],
                     n_spikes=n, rate=n / dur, mrl=R, rayleigh_p=p, pref_phase=pref))
    spike_phases_store[uid] = ph

res = pd.DataFrame(rows)
print(res.groupby("cell_type")["mrl"].describe())

# ---------------- random-time control ----------------
# For each included unit, draw the same number of random times inside running bouts
# (preserves spike count and the common theta-phase distribution).
bout_lens = bouts[:, 1] - bouts[:, 0]
cum = np.cumsum(bout_lens)

def random_run_times(n):
    u = rng.random(n) * cum[-1]
    idx = np.searchsorted(cum, u)
    prev = np.r_[0, cum[:-1]][idx]
    return bouts[idx, 0] + (u - prev)

ctrl_mrl = []
for _, row in tqdm(res.iterrows(), total=len(res), desc="control"):
    n = int(row["n_spikes"])
    ph = phases_at(random_run_times(n))
    ctrl_mrl.append(np.abs(np.mean(np.exp(1j * ph))))
res["mrl_control"] = ctrl_mrl

res.to_csv("unit_phase_locking.csv", index=False)
np.savez("theta_analysis.npz",
             bouts=bouts, spd_t=spd_t, spd_s=spd_s,
             maze_start=maze_start, maze_end=maze_end, best_ch=best_ch, fs=fs)
import pickle
with open("spike_phases.pkl", "wb") as f:
    pickle.dump({int(k): v for k, v in spike_phases_store.items()}, f)

sig = res["rayleigh_p"] < 0.01
print(f"\nunits included: {len(res)}; Rayleigh p<0.01: {sig.sum()} ({sig.mean()*100:.0f}%)")
for ct in ["excitatory", "inhibitory"]:
    sub = res[res["cell_type"] == ct]
    s = sub["rayleigh_p"] < 0.01
    print(f"{ct}: {s.sum()}/{len(sub)} locked, median MRL {sub['mrl'].median():.3f}, "
          f"control median {sub['mrl_control'].median():.3f}")
sig_exc = res[(res["cell_type"] == "excitatory") & sig]
pooled = np.angle(np.mean(np.exp(1j * sig_exc["pref_phase"].values)))
print(f"pooled preferred phase (sig exc): {np.degrees(pooled):.0f} deg (0 = LFP peak)")
print("saved unit_phase_locking.csv, theta_analysis.npz, spike_phases.pkl")
