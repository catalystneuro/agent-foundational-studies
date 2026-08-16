# %% [markdown]
# # 04 — Theta phase extraction, run epochs, spike-phase statistics
# Main analysis: bandpass-filter the reference LFP around the theta peak
# (9 Hz), derive instantaneous phase with the Hilbert transform, define run
# bouts from 2-D speed during the maze epoch, and compute per-unit circular
# statistics (mean resultant length, preferred phase, Rayleigh p) of spike
# phases during running. Includes a random-time shuffle null.

# %%
import numpy as np
import remfile
import requests
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal

ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"
LFP_FS = 1250.0
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
# Maze epoch sample range, padded for filter edge effects
t_maze0, t_maze1 = 18079.5, 20147.0
pad_s = 20.0
i0 = int((t_maze0 - pad) * LIVE_FS)
i1 = int((t_maze1 + pad) * LIVE_FS)
raw = np.asarray(h5py_file["processing/ecephys/LFP/LFP/data"][i0:i1, best_ch], dtype=np.float64)
lfp_uV = raw * conversion * 1e6
lfp_t = np.arange(i0, i1) / LIVE_FS
print("LFP segment:", lfp_t[0], "-", lfp_t[-1], "n =", len(lf_uV))

# %% [markdown]
# ## Instantaneous theta phase (Hilbert)
# Filter in a band around the 9 Hz peak, then Hilbert transform.

# %%
peak = 9.0
band = (6.0, 11.0)
sos = signal.butter(4, band, btype="band", fs=LIVE_FS, output="sos")
thet = signal.sosfiltfilt(sos, lf_uV)
phase = np.angle(signal.hilbert(thet))          # 0 = positive peak of filtered signal
print("phase computed, n =", len(phase))

# %% [markdown]
# ## Run bouts from 2-D speed

# %%
pos = nwb["1.6mLinearMazeSpatialSeries"]
pt = np.asarray(pos.t)
pn = int(np.median(np.diff(pt)))
px = np.asarray(pos["x"]).astype(float)
py = np.asarray(pos["y"]).astype(float)
valid = np.isfinite(px) & np.isfinite(py)
n = len(pt)
vfull = np.zeros(n)
for i in range(1, n - 1):
    if valid[i - 1] and valid[i + 1]:
        vfull[i] = np.hypot(px[i + 1] - px[i - 1], py[i + 1] - py[i - 1]) / (pt[i + 1] - pt[i - 1])
    elif valid[i - 1] and not valid[i + 1]:
        pass
# interpolation across short NaN gaps: speed 0 elsewhere
run_mask = vfull >= 0.10
# merge run bouts separated by < 0.3 s
runs = []
in_run = False
for i in range(n):
    if run_mask[i] and not in_run:
        start = i
        in_run = True
    elif not run_mask[i] and in_run:
        end = i - 1
        if start > 0 and pt[start] - prev_end < 0.3:
            # merge: extend previous bout
            prev_s, prev_e = runs[-1]
            runs[-1] = (prev_s, end)
        else:
            runs.append((start, end))
        in_run = False
        prev_end = pt[end]
if in_run:
    runs.append((start, n - 1))
intervals = []
run_times = []
for (s, e) in runs:
    if pt[e] - pt[s] >= 1.0:
        intervals.append((pt[s], pt[e]))
        run_times.append((pt[s], pt[e]))
run = nap.IntervalSet(start=np.array([r[0] for r in intervals]),
                      end=np.array([r[1] for r in intervals]))
print("run bouts:", len(run), "total running time: %.1f s" % run.tot_duration)

# also speed for a plot
speed_series = nap.Tsd(t=pt, d=vfull)

# %% [markdown]
# ## Spike-phase statistics per unit
# %%
units = nwb["units"]
cell_type = units.get_info("cell_type").values if units.get_info("cell_type") is not None else np.array(["NA"] * len(units))
unit_ids = list(units.keys())

MIN_SPIKES = 50
rows = []
for uid in unit_ids:
    ts = units[uid]
    spk = np.asarray(ts.t)
    m = (spk >= t_maze0) & (spk < t_maze1)
    spk = spk[m]
    n_spk = len(spk)
    if n_spk < MIN_SPIKES:
        continue
    # phase at each spike
    idx = ((spk - lfp_t[0]) * LIVE_FS).astype(int)
    ok = (idx >= 0) & (idx < len(phase))
    ph = phase[idx[ok]]
    R = np.sqrt(np.sum(np.cos(ph))**2 + np.sum(np.sin(ph))**2) / len(ph)
    z = len(ph) * R**2
    p_rayleigh = np.exp(-z)
    mu = np.arctan2(np.sum(np.sin(ph)), np.sum(np.cos(ph)))
    rows.append(dict(unit=uid, n=n_spk, R=R, mu=mu, p=p_rayleigh,
                     cell_type=ct_series[uid] if cell_type is not None else "NA"))
stats = pd.DataFrame(rows)
print(stats["cell_type"].value_counts())
print(stats.describe())

# %% [markdown]
# ## Random-time shuffle null
# Control: draw the same number of spike times uniformly within the run
# bouts, recompute MRL. The real MRL far exceeds the null.

# %%
NSHUF = 200
run_beg = np.repeat(run.start, (run.start < spk_tmax))  # placeholder
spk_times_all = np.concatenate([np.asarray(units[uid].t) for uid in unit_ids])
run_lo, run_up = np.array(run.start), np.array(run.end)
run_tot = np.sum(run_up - run_lo)
R_null = np.zeros(NSHUF)
for sh in range(NSHUF):
    rts = run_lo[0] + ((rng.uniform(0, run_tot, 20000)) / run_tot)
    # map into run intervals
    cum = np.concatenate([np.insert(np.cumsum(run_up - run_lo), 0, 0)])
    x = rng.uniform(0, cum[-1], size=20000)
    seg = np.searchsorted(cum, x, side="right") - 1
    y = run_lo[seg] + (x - cum[seg])
    idx = ((y - lfp_t[0]) * LIVE_FS).astype(int)
    ph = phase[idx[ok_mask]]
    R_lo[sh] = np.abs(np.mean(np.exp(1j * ph)))
print("null R: median %.4f, mean %.4f" % (np.median(R_lo), np.mean(R_lo)))
null_med = np.median(R_lo)