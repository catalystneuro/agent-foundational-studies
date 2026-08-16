# %% [markdown]
# # Theta Phase Precession in Hippocampal Place Cells
#
# DANDI dataset **000044** ("Diversity in neural firing dynamics within the
# hippocampal formation", Buzsaki lab, Rutgers): a rat CA1 tetrode recording on
# a 1.6 m linear maze. This notebook demonstrates **theta phase precession**
# (O'Keefe & Recce, 1993): a hippocampal place cell fires at progressively
# earlier phases of the ~8 Hz theta rhythm as the rat traverses the cell's
# place field.
#
# * Session: `sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`
# * Theta is recovered from LFP channel 117 (picked data-driven by theta/delta PSD).
# * Pipeline: streaming NWB access (remfile) -> run-bout detection on a
#   linearized track -> per-direction tuning curves -> place-cell detection
#   (Skaggs spatial information, time-shift shuffle null) -> circular-linear
#   regression of spike theta phase vs. normalized position inside each field,
#   with a phase-permutation significance test and a per-lap check.

# %% [markdown]
# ## 1. Setup and streaming data loading

# %%
import os
import json
import requests
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import matplotlib.pyplot as plt
from scipy import signal
from tqdm.auto import tqdm

RNG = np.random.default_rng(7)
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# Resolve the presigned S3 blob URL for the asset (cached in s3_url.txt).
if os.path.exists("s3_url.txt"):
    s3_url = open("s3_url.txt").read().strip()
else:
    asset_id = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"
    r = requests.get(
        f"https://api.dandiarchive.org/api/assets/{asset_id}/download/",
        allow_redirects=False, timeout=30)
    s3_url = r.headers["Location"]
    open("s3_url.txt", "w").write(s3_url)

print("Streaming NWB from S3 blob ...")
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print("NWB loaded")

# %%
units = nwb["units"]
unit_ids = list(units.keys())
print("total units:", len(unit_ids))
info = units.get_info("cell_type")
ct_map = dict(zip(info.index, np.asarray(info.values)))
exc_ids = [u for u in unit_ids if ct_map[u] == "excitatory"]
inh_ids = [u for u in unit_ids if ct_map[u] == "inhibitory"]
print(f"excitatory: {len(exc_ids)}, inhibitory: {len(inh_ids)}")

# Position (meters) and linearized position along the 1.6-m track, ~39 Hz.
pos2d = nwb["1.6mLinearMazeSpatialSeries"]
t_pos = np.asarray(pos2d.t)
pos_xy = np.asarray(pos2d.values)
lin_nwb = nwb["1.6mLinearMazeLinearizedTimeSeries"]
t_lin = np.asarray(lin_nwb.t)
lin = np.asarray(lin_nwb.values).ravel()
print("n position samples:", len(t_pos),
      "| dt median: %.5f s" % np.median(np.diff(t_pos)))
print("linearized range: %.2f - %.2f m | valid fraction: %.3f"
      % (np.nanmin(lin), np.nanmax(lin), np.isfinite(lin).mean()))

epochs = nwb["epochs"]
maze = epochs[epochs.label == "MazeEpoch"]
maze_start = float(maze.start[0])
maze_end = float(maze.end[0])
print(f"MazeEpoch: {maze_start:.1f} - {maze_end:.1f} s")

# %% [markdown]
# ### Figure 1. Raw overview: track geometry, position validity, spike rasters
# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
vm = np.isfinite(pos_xy[:, 0]) & np.isfinite(pos_xy[:, 1])
sc = ax.scatter(pos_xy[vm, 0], pos_xy[vm, 1], c=t_pos[vm], s=2, cmap="viridis")
ax.set_title("2-D position trajectory (maze epoch)")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
plt.colorbar(sc, ax=ax, label="time (s)")

ax = axes[0, 1]
ax.plot(t_pos, np.isfinite(lin) * 1.0, lw=0.5)
ax.set_xlim(maze_start, maze_end)
ax.set_xlabel("time (s)"); ax.set_ylabel("on-track = 1")
ax.set_title("Linearized-position validity across the maze epoch")

ax = axes[1, 0]
t0 = maze_start + 200.0
t1 = maze_start + 400.0
for u, cid in enumerate(unit_ids):
    if u > 25:
        break
    st = np.asarray(units[cid].t)
    st = st[(st >= t0) & (st <= t1)]
    ax.plot(st, np.full(len(st), u) + 0.2, "k|", ms=2, mew=0.7)
ax.set_ylim(-0.5, 11)
ax.set_xlabel("time (s)"); ax.set_ylabel("unit")
ax.set_title("Example spike trains (200 s of maze epoch)")

ax = axes[1, 1]
tchunk = (t_pos >= t0) & (t_pos <= t1)
ax.plot(t_pos[tchunk], lin[tchunk], lw=0.8)
ax.set_xlabel("time (s)"); ax.set_ylabel("linearized position (m)")
ax.set_title("Linearized position over the same 200 s")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig01_raw_overview.png", dpi=150)
plt.close()

# %% [markdown]
# ## 2. Theta LFP: channel selection and instantaneous phase
#
# The 128-channel LFP (1250 Hz, 3.815e-7 V/ADU) is read straight from the HDF5
# dataset. We first scan a 200-s chunk of the maze epoch across the full
# electrode array, score each channel by the ratio of theta-band (6-10 Hz) to
# delta-band (1-4 Hz) PSD, and keep the strongest-theta channel. We then
# band-pass that channel (6-10 Hz, Butterworth, sosfiltfilt) and take the
# Hilbert transform. The instantaneous phase is defined so that phase 0 is the
# positive peak of the theta oscillation.

# %%
FS = 1250.0
THETA_LO, THETA_HI = 6.0, 10.0
DELTA_LO, DELTA_HI = 3.0, 5.0
LFP_CONV = 3.815e-7  # volts per ADU

dset = h5py_file["processing/ecephys/LFP/LFP/data"]
print("LFP dataset:", dset.shape)

scan_start = maze_start + 360.0
scan_end = scan_start + 200.0
si0 = int(scan_start * FS)
si1 = int(scan_end * FS)

scratch = np.asarray(dset[si0:si1, :], dtype=np.float64) * 1e6  # uV (full 128 ch)
ratios = {}
for ch in range(scratch.shape[1]):
    seg = scratch[:, ch]
    seg = (seg - seg.mean()) / seg.std()
    freqs, pxx = signal.welch(seg, FS, nperseg=1024)
    bth = (freqs >= THETA_LO) & (freqs <= THETA_HI)
    bdl = (freqs >= DELTA_LO) & (freqs <= DELTA_HI)
    ratios[ch] = pxx[bth].mean() / pxx[bdl].mean()
theta_ch = max(ratios, key=ratios.get)
print("theta/delta ratio per channel (top 5):")
for ch in sorted(ratios, key=ratios.get, reverse=True)[:5]:
    print(f"  ch {ch}: {ratios[ch]:.2f}")
print("selected theta channel:", theta_ch)

# %%
# Read the full maze-epoch LFP on the selected channel and compute theta phase.
i0m = int(maze_start * FS)
i1m = int(maze_end * FS)
lfp_maze = np.asarray(dset[i0m:i1m, theta_ch], dtype=np.float64) * LFP_CONV
t_lfp = (np.arange(i0m, i1m) + 0.5) / FS  # sample-centered absolute time
sos = signal.butter(3, [THETA_LO, THETA_HI], btype="bandpass", fs=FS, output="sos")
lfp_filt = signal.sosfiltfilt(sos, lfp_maze)
theta_phase = np.angle(signal.hilbert(lfp_filt))
print("theta samples:", len(lfp_maze), "| range (uV): %.1f .. %.1f"
      % (lfp_maze.min() * 1e6, lfp_maze.max() * 1e6))

# Sanity check figure: PSD of the chosen channel and a zoomed raw+filtered trace.
freqs, pxx = signal.welch(lfp_maze, FS, nperseg=FS * 4)
peak_f = freqs[np.argmax(pxx[(freqs >= 3) & (freqs <= 30)])]

fig, ax = plt.subplots(2, 1, figsize=(12, 7))
ax[0].loglog(freqs, pxx)
ax[0].axvspan(THETA_LO, THETA_HI, color="C1", alpha=0.3)
ax[0].set_xlim(0.5, 60)
ax[0].set_title(f"LFP PSD (ch {theta_ch}), theta-band shaded; peak {peak_f:.2f} Hz")
ax[0].set_xlabel("Hz"); ax[0].set_ylabel("PSD")

zs = int((scan_start + 30) * FS)
ze = zs + 4000  # ~3.2 s
tt = t_lfp[zs:ze]
ax[1].plot(tt, lfp_maze[zs:ze] * 1e6, lw=0.7, label="raw LFP (uV)")
ax[1].plot(tt, lfp_filt[zs:ze] * 4e7, lw=0.9, label="theta filter (scaled)")
ax[1].set_title("Raw and theta-filtered LFP (3.2 s)")
ax[1].set_xlabel("time (s)"); ax[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig02_theta_lfp.png", dpi=150)
plt.close()

# %% [markdown]
# ## 3. Run bout detection
#
# The linearized position is NaN whenever the rat is off the linear arm of the
# maze (sitting on the reward platforms). We define a "run bout" as a
# contiguous set of valid linearized samples, merging gaps shorter than 0.3 s,
# and keeping only bouts that:
# * last at least 1.0 s,
# * span at least 0.3 m along the track,
# * have median |speed| > 0.15 m/s (median |d(lin)/dt|).
#
# Each bout is labelled with its direction of travel (+: increasing linearized
# coordinate, -: decreasing), because place fields must be analyzed separately
# per direction.

# %%
valid = np.isfinite(lin)
bout_idx = []
in_bout = False
for i in range(len(t_pos)):
    if valid[i] and not in_bout:
        start = i
        in_bout = True
    elif not valid[i] and in_bout:
        bout_idx_ = (start, i)
        # merge with previous bout if gap <= 0.3 s
        if bout_idx and t_pos[start] - t_pos[bout_idx[-1][1] - 1] <= 0.3:
            bout_idx[-1] = (bout_idx[-1][0], i)
        else:
            bout_idx.append(bout_idx_)
        in_bout = False
if in_bout:
    bout_idx.append((start, len(t_pos)))

# Filter by duration, span and speed
bouts = []   # dicts with t, x, direction
for (s, e) in bout_idx:
    dur = t_pos[e - 1] - t_pos[s]
    xseg = lin[s:e]
    tseg = t_pos[s:e]
    m = np.isfinite(xseg)
    xv = xseg[m]; tv = tseg[m]
    if len(xv) < 5:
        continue
    span = xv.max() - xv.min()
    speed = np.median(np.abs(np.diff(xv) / np.diff(tv)))
    if dur < 1.0 or span < 0.3 or speed < 0.15:
        continue
    dirn = 1 if np.median(np.diff(xv) / np.diff(tv)) > 0 else -1
    bouts.append(dict(t=tv, x=xv, direction=dirn, start=tv[0], end=tv[-1]))
print("run bouts:", len(bouts), "| +dir:", sum(b["direction"] == 1 for b in bouts),
      "| -dir:", sum(b["direction"] == -1 for b in bouts))
tot = sum(b["end"] - b["start"] for b in bouts)
print("total run time: %.1f s (%.1f%% of maze epoch)"
      % (tot, 100 * tot / (maze_end - maze_start)))
print("median bout duration: %.2f s | median speed: %.2f m/s"
      % (np.median([b['end'] - b['start'] for b in bouts]),
         np.median([np.median(np.abs(np.diff(b['x']) / np.diff(b['t']))) for b in bouts])))

# %%
# Figure 3: bout structure overview
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax = axes[0]
for b in bouts:
    c = "C0" if b["direction"] == 1 else "C3"
    ax.plot(b["t"], b["x"], lw=1.0, color=c)
ax.set_xlabel("time (s)"); ax.set_ylabel("linearized position (m)")
ax.set_title("Run bouts (blue = +dir, red = -dir)")
ax.set_xlim(maze_start, maze_end)

ax = axes[1]
durs = [b["end"] - b["start"] for b in bouts]
ax.hist(durs, bins=30, color="0.5", edgecolor="k")
ax.set_xlabel("bout duration (s)"); ax.set_ylabel("count")
ax.set_title("Run bout durations")

ax = axes[2]
spds = [np.median(np.abs(np.diff(b["x"]) / np.diff(b["t"]))) for b in bouts]
ax.hist(spds, bins=30, color="0.5", edgecolor="k")
ax.set_xlabel("median speed (m/s)"); ax.set_ylabel("count")
ax.set_title("Run bout speeds")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig03_run_bouts.png", dpi=150)
plt.close()

# %% [markdown]
# ## 4. Per-direction tuning curves and place cell detection
#
# For each travel direction we pool the position samples of all bouts of that
# direction and build a tuning curve per cell, binning the track into 50 bins
# (3.2 cm each) and smoothing with a 1.5-bin Gaussian kernel. Directional
# tuning is essential here: a cell's field can be visited in either direction,
# and precession must be measured separately for each direction.
#
# **Place cell criterion.** A cell-direction is a place cell if its tuning
# curve peak rate is >= 1 Hz and its Skaggs spatial information (SI) is
# significant (p < 0.05) with respect to a circular time-shift shuffle: we
# repeatedly shift that cell's spike train by a random time offset *inside the
# pooled run time*, recompute the tuning curve and SI, and compare. Shifting
# spike times (not spike positions) preserves the occupancy/rate structure of
# the run and produces a valid null for SI.

# %%
N_BINS = 50
bin_edges = np.linspace(0.0, 1.6, N_BINS + 1)
bin_c = 0.5 * (bin_edges[:-1] + bin_edges[1:])
dt_pos = np.median(np.diff(t_pos))          # sampling interval of position
smooth_k = np.exp(-0.5 * (np.arange(-6, 7) / 1.5) ** 2)
smooth_k /= smooth_k.sum()

def smooth1d(v):
    return np.convolve(v, smooth_k, mode="same")

# Per direction: pooled position samples (time, linearized position)
run_by_dir = {1: [], -1: []}
for b in bouts:
    run_by_dir[b["direction"]].append((b["t"], b["x"]))

def tuning_curve_dir(x_samples, dt=dt_pos, edges=bin_edges):
    """Occupancy (s) per bin from pooled position samples."""
    occ, _ = np.histogram(x_samples, edges)
    return occ * dt

# Pooled position arrays per direction (for occupancy / tuning)
occ_dir = {}
pos_dir = {}
for d in (1, -1):
    xs = np.concatenate([x for (t, x) in run_by_dir[d]]) if run_by_dir[d] else np.array([])
    pos_dir[d] = xs
    occ_dir[d] = tuning_curve_dir(xs)
    occ_dir[d][occ_dir[d] == 0] = np.nan

print("per-direction occupancy (seconds, bins with zero occupancy -> NaN):")
for d in (1, -1):
    print("  dir", d, "samples:", len(pos_dir[d]), "| occupied bins:",
          np.isfinite(occ_dir[d]).sum(), "/", N_BINS)

# %%
print("Computing tuning curves for all cells in both directions ...")
tuning = {}   # (direction, unit) -> rate array
spike_count_in_bins = {}
for d in (1, -1):
    for uid in tqdm(unit_ids, desc=f"direction {d:+.0f} tuning"):
        st = np.asarray(units[uid].t)
        xs_list = []
        for (tb, xv) in run_by_dir[d]:
            m = (st >= tb[0]) & (st <= tb[-1])
            if m.any():
                xs_list.append(np.interp(st[m], tb, xv))
        if not xs_list:
            tuning[(d, uid)] = None
            spike_count_in_bins[(d, uid)] = None
            continue
        xs = np.concatenate(xs_list)
        cnt, _ = np.histogram(xs, bin_edges)
        spike_count_in_bins[(d, uid)] = cnt
        tuning[(d, uid)] = cnt / occ_dir[d]

# %% [markdown]
# ## 5. Place cell detection: Skaggs spatial information with a time-shift null
#
# For each cell-direction we compute the Skaggs spatial information
# SI = sum_i p_i (r_i / R) log2(r_i / R) (bits/spike), where p_i is the
# fractional occupancy of bin i and r_i the smoothed firing rate. A cell is a
# place cell only if its SI exceeds the 95th percentile of a **time-shift
# shuffle** null: we shift the cell's full spike train by a random time offset
# on the concatenated run-time axis (per direction), which randomizes the
# spike-position association while keeping the exact firing statistics, and
# recompute SI. The p-value is the fraction of 300 shuffles with SI >= the
# observed SI.

# %%
def skaggs_si(rate, occ):
    """Skaggs spatial information in bits/spike."""
    mask = np.isfinite(occ) & (occ > 0) & np.isfinite(rate) & (rate > 0)
    if mask.sum() < 3:
        return np.nan
    p = occ[mask] / occ[mask].sum()
    r = rate[mask]
    R = (p * r).sum()
    if R <= 0:
        return np.nan
    return float((p * (r / R) * np.log2(r / R)).sum())

def smooth_rate(cnt, occ):
    """Smooth spike counts and occupancy with the same kernel, then divide."""
    c = smooth1d(np.nan_to_num(cnt, nan=0.0))
    o = smooth1d(np.nan_to_num(occ, nan=0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(o > 0, c / o, np.nan)
    return r

# %%
NSHUF_SI = 300
# Build one concatenated run-time axis per direction for the spike-time shifts.
run_tau_dir = {}      # direction -> (tau, total_time)
pos_all_dir = {}      # direction -> positions aligned to tau
bout_off_dir = {}     # direction -> cumulative offsets per bout
for d in (1, -1):
    ts, ps, offs = [], [], []
    c = 0.0
    for (tb, xv) in run_by_dir[d]:
        ts.append(c + (tb - tb[0]))
        ps.append(xv)
        offs.append(c)
        c += tb[-1] - tb[0]
    run_tau_dir[d] = (np.concatenate(ts), c)
    pos_all_dir[d] = np.concatenate(ps)
    bout_off_dir[d] = offs

si_sig = {}   # (d, uid) -> dict(si, p, nspk, peak)
rng_shuf = np.random.default_rng(123)
for d in (1, -1):
    tau_all, T = run_tau_dir[d]
    pos_all = pos_all_dir[d]
    offs = bout_off_dir[d]
    for uid in tqdm(unit_ids, desc=f"SI shuffle dir {d:+.0f}"):
        st = np.asarray(units[uid].t)
        if len(st) == 0:
            si_sig[(d, uid)] = {"si": np.nan, "p": 1.0, "nspk": 0, "peak": 0.0}
            continue
        tau_spk = []
        for k, (tb, xv) in enumerate(run_by_dir[d]):
            m = (st >= tb[0]) & (st <= tb[-1])
            if m.any():
                tau_spk.append((st[m] - tb[0]) + offs[k])
        if not tau_spk:
            si_sig[(d, uid)] = {"si": np.nan, "p": 1.0, "nspk": 0, "peak": 0.0}
            continue
        tau_spk = np.concatenate(tau_spk)
        xs_obs = np.interp(tau_spk, tau_all, pos_all)
        cnt_obs, _ = np.histogram(xs_obs, bin_edges)
        rate_obs = smooth_rate(cnt_obs, occ_dir[d])
        si_obs = skaggs_si(rate_obs, occ_dir[d])
        if not np.isfinite(si_obs):
            si_sig[(d, uid)] = {"si": np.nan, "p": 1.0, "nspk": len(tau_spk),
                                "peak": np.nanmax(rate_obs)}
            continue
        counts_sh = np.zeros(NSHUF_SI)
        for h in range(NSHUF_SI):
            tau_sh = (tau_spk + rng_shuf.uniform(0, T)) % T
            xs_sh = np.interp(tau_sh, tau_all, pos_all)
            cnt_sh, _ = np.histogram(xs_sh, bin_edges)
            rate_sh = smooth_rate(cnt_sh, occ_dir[d])
            counts_sh[h] = skaggs_si(rate_sh, occ_dir[d])
        p = float((np.sum(np.nan_to_num(counts_sh, nan=-np.inf) >= si_obs) + 1)
                 / (NSHUF_SI + 1))
        peak = float(np.nanmax(rate_obs))
        si_sig[(d, uid)] = {"si": float(si_obs), "p": p, "nspk": len(tau_spk),
                            "peak": peak}

# Summary of the place-cell readout
n_place = sum(
    1 for (d, uid), v in si_sig.items()
    if uid in exc_ids and v["p"] < 0.05 and v["peak"] >= 1.0 and v["si"] > 0)
print(f"excitatory cell-directions passing p<0.05 with peak>=1 Hz: {n_place}")
print("cells with >=1 qualifying direction:",
      len({uid for (d, uid), v in si_sig.items()
           if uid in exc_ids and v["p"] < 0.05 and v["peak"] >= 1.0 and v["si"] > 0}))

# %% [markdown]
# ## 6. Place field extraction and theta phase at each spike
#
# For every cell-direction that passed the SI test (p < 0.05, peak >= 1 Hz) we
# extract a single contiguous field: starting from the maximal bin we grow the
# field outward while the smoothed rate stays above 20% of the peak, with a
# minimum width of 0.2 m and a maximum of 1.2 m. Only cell-directions with at
# least 30 in-field spikes are kept.
#
# The **field progress coordinate** must be oriented by travel direction:
# progress = (x - lo) / (hi - lo) for + direction and (hi - x) / (hi - lo) for
# - direction, so that progress runs 0 -> 1 as the animal crosses the field in
# the direction of travel. Mixing directions without this re-orientation
# cancels precession at the population level.
#
# We also interpolate the instantaneous theta phase onto every in-field spike
# from the precomputed Hilbert phase of the selected LFP channel.

# %%
FIELD_FRAC = 0.2        # fall to 20% of peak to delimit the field
MIN_WIDTH, MAX_WIDTH = 0.2, 1.2   # meters
MIN_FIELD_SPIKES = 30

def find_field(rate, edges=bin_edges):
    """Return (lo, hi) meters of the contiguous field about the peak bin."""
    r = np.nan_to_num(rate, nan=0.0)
    pk = int(np.nanargmax(rate))
    thr = FIELD_FRAC * r[pk]
    lo = pk; hi = pk
    while lo > 0 and r[lo - 1] > thr:
        lo -= 1
    while hi < len(r) - 1 and r[hi + 1] > thr:
        hi += 1
    return float(edges[lo]), float(edges[hi + 1])

def theta_phase_at(times):
    """Theta phase at arbitrary times (interp on the maze-epoch phase array)."""
    t0 = t_lfp[0]
    idx = np.clip(((times - t0) * FS).astype(int), 0, len(theta_phase) - 1)
    return theta_phase[idx]

fields = []   # each: dict(direction, uid, lo, hi, rate, spike_x, spike_phase, ...)
for d in (1, -1):
    for uid in exc_ids:
        v = si_sig.get((d, uid))
        if v is None or not (v["p"] < 0.05 and v["peak"] >= 1.0):
            continue
        cnt = spike_count_in_bins[(d, uid)]
        if cnt is None:
            continue
        rate = smooth_rate(cnt, occ_dir[d])   # smoothed rate for field finding
        lo, hi = find_field(rate)
        w = hi - lo
        if w < MIN_WIDTH or w > MAX_WIDTH:
            continue
        # collect in-field spikes with their linear position
        st = np.asarray(units[uid].t)
        xs = []
        for (tb, xv) in run_by_dir[d]:
            m = (st >= tb[0]) & (st <= tb[-1])
            if m.any():
                xq = np.interp(st[m], tb, xv)
                k = (xq >= lo) & (xq <= hi)
                xs.append((st[m][k], xq[k]))
        if not xs:
            continue
        st_sp, x_sp = map(np.concatenate, zip(*xs))
        if len(st_sp) < MIN_FIELD_SPIKES:
            continue
        # direction-oriented progress
        if d == 1:
            prog = (x_sp - lo) / (hi - lo)
        else:
            prog = (hi - x_sp) / (hi - lo)
        ph = theta_phase_at(st_sp)
        fields.append(dict(uid=uid, direction=d, lo=lo, hi=hi, width=w,
                           t=st_sp, x=x_sp, progress=prog, phase=ph,
                           rate=rate, n=len(st_sp)))

print("place fields (cell-directions):", len(fields),
      "| distinct cells:", len({f["uid"] for f in fields}))
print("fields with >=%d spikes: %d" % (MIN_FIELD_SPIKES,
      sum(f["n"] >= MIN_FIELD_SPIKES for f in fields)))
if fields:
    print("width distribution: median %.2f m, range %.2f-%.2f m"
          % (np.median([f["width"] for f in fields]),
             min(f["width"] for f in fields), max(f["width"] for f in fields)))

# %% [markdown]
# ## 7. Circular-linear regression of theta phase on field progress
#
# Phase precession means the firing phase advances (earlier in the theta
# cycle) as the animal crosses the field. We model the phase as
#   phase = phi0 - 2*pi*slope * progress  (mod 2 pi),
# i.e. a circular variable regressed on a linear one. For each place field we
# find the slope by maximizing the length of the mean resultant vector
#   R(k) = |mean(exp(1j * (phase - 2*pi*k*progress)))|
# over a grid of k in [-6, 6] cycles per field width, then compute:
# * the Kempter circular-linear correlation coefficient r (signed),
# * a p-value from a **phase-permutation shuffle**: 500 shuffles of the phase
#   values across the spikes, keeping the progress values fixed, destroying
#   only the phase-progress relationship.
#
# A cell shows precession if the fitted slope is significantly negative and
# rho < 0 (phase advances to earlier theta phases along the field).

# %%
K_GRID = np.linspace(-6.0, 6.0, 1201)
NSHUF_PREC = 500

def circular_linear_stats(progress, phase, rng):
    """Grid-search circular-linear regression + Kempter-style correlation.

    Returns dict with slope (cycles/field), rho (signed circular-linear
    correlation), Rmax, and p (phase-permutation test).
    """
    n = len(progress)
    p1 = phase[:, None] - 2 * np.pi * K_GRID[None, :] * progress[:, None]
    z = np.exp(1j * p1).mean(axis=0)
    R = np.abs(z)
    kidx = int(np.argmax(R))
    slope = K_GRID[kidx]           # cycles per field width
    Rmax = float(R[kidx])
    # scale progress to the circle for the correlation coefficient
    alpha = 2 * np.pi * K_GRID[kidx] * progress
    R_l = np.abs(np.mean(np.exp(1j * alpha)))
    R_c = np.abs(np.mean(np.exp(1j * phase)))
    R_lc = np.abs(np.mean(np.exp(1j * (phase - alpha))))
    denom = np.sqrt((1 - R_l ** 2) * (1 - R_c ** 2))
    rho = float(R_lc - R_l * R_c) / denom if denom > 1e-6 else np.nan
    shuf = np.zeros(NSHUF_PREC)
    k_shuf = np.linspace(-6.0, 6.0, 301)
    for h in range(NSHUF_PREC):
        ph_sh = rng.permutation(phase)
        p2 = ph_sh[:, None] - 2 * np.pi * k_shuf[None, :] * progress[:, None]
        R_sh = np.abs(np.exp(1j * p2).mean(axis=0)).max()
        shuf[h] = R_sh
    p = float((np.sum(shuf >= Rmax) + 1) / (NSHUF_PREC + 1))
    return dict(slope_cyc_per_field=slope, rho=rho, Rmax=Rmax, p=p, n=n)

# Compute for all fields (vectorized over fields where possible)
prec = {}
rng_prec = np.random.default_rng(2024)
for i, f in enumerate(tqdm(fields, desc="precession fit")):
    if f["n"] < 30:
        continue
    pr = np.clip(f["progress"], 0.0, 1.0)   # sanitize
    prec[i] = circular_linear_stats(pr, f["phase"], rng_prec)

rows = [dict(uid=f["uid"], direction=f["direction"], lo=f["lo"], hi=f["hi"],
             width=f["width"], n=f["n"], **prec[i])
        for i, f in enumerate(fields) if i in prec]
print("cell-directions with >=30 in-field spikes:", len(rows))

sig_neg = [r for r in rows if r["p"] < 0.05 and r["rho"] < 0]
sig_pos = [r for r in rows if r["p"] < 0.05 and r["rho"] > 0]
print("significant negative (precession):", len(sig_neg))
print("significant positive (anti-precession):", len(sig_pos))
if rows:
    med_slope = np.median([r["slope_cyc_per_field"] for r in sig_neg])
    med_rho = np.median([r["rho"] for r in sig_neg])
    print("median slope (sig. neg.): %.2f cyc/field | median rho: %.2f" % (med_slope, med_rho))

# %% [markdown]
# ## 8. Per-lap (per-traversal) check
#
# A strong demonstration of precession is that it also appears within single
# traversals of the field. For each place field we find every run bout that
# crosses the field, take the spikes of that passage, and (if there are at
# least 4 spikes spanning at least 25% of the field) compute the same
# circular-linear correlation with the cell's fitted slope held fixed. The
# per-lap rho values are pooled across all fields and laps; precession
# predicts a strong negative median and a large fraction of negative laps.

# %%
MIN_LAP_SPIKES = 4
MIN_LAP_SPAN = 0.25

lap_results = []   # (uid, direction, lap_rho, lap_span)
for i, f in enumerate(fields):
    if i not in prec:
        continue
    slope = prec[i]["slope_cyc_per_field"]
    uid = f["uid"]; d = f["direction"]
    lo, hi = f["lo"], f["hi"]
    st = np.asarray(units[uid].t)
    for (tb, xv) in run_by_dir[d]:
        m = (st >= tb[0]) & (st <= tb[-1])
        if m.sum() < MIN_LAP_SPIKES:
            continue
        xq = np.interp(st[m], tb, xv)
        k = (xq >= lo) & (xq <= hi)
        if k.sum() < MIN_LAP_SPIKES:
            continue
        st_k = st[m][k]; x_k = xq[k]
        # progress oriented by direction
        if d == 1:
            prog = (x_k - lo) / (hi - lo)
        else:
            prog = (hi - x_k) / (hi - lo)
        prog = np.clip(prog, 0, 1)
        span = prog.max() - prog.min()
        if span < MIN_LAP_SPAN:
            continue
        ph = theta_phase_at(st_k)
        alpha = 2 * np.pi * slope * prog
        R_l = np.abs(np.mean(np.exp(1j * alpha)))
        R_c = np.abs(np.mean(np.exp(1j * ph)))
        R_lc = np.abs(np.mean(np.exp(1j * (ph - alpha))))
        denom = np.sqrt((1 - R_l ** 2) * (1 - R_c ** 2))
        if denom < 1e-6:
            continue
        rho_lap = float(R_lc - R_l * R_c) / denom
        lap_results.append((uid, d, rho_lap, span))

laps = np.array([r[2] for r in lap_results])
print("laps analyzed:", len(laps))
if len(laps):
    print("median per-lap rho: %.3f | negative fraction: %.2f"
          % (np.median(laps), (laps < 0).mean()))
    print("laps with |rho|>0.3 (decent fit):", ((np.abs(laps) > 0.3)).mean().round(3))

# %% [markdown]
# ## 9. Example cells
#
# Figure 4 shows three of the strongest precession cells. For each, the top
# panel is the smoothed per-direction tuning curve with the detected field
# shaded; the bottom panel is a scatter of theta phase vs. field progress with
# spikes colored by lap (each traversal of the field in a different color).
# Precession is the steady downward ramp of phase across the field, i.e. each
# subsequent lap the cell fires earlier in the theta cycle.

# %%
sig_cells = sorted(sig_neg, key=lambda r: -abs(r["rho"]))[:3]
n_ex = len(sig_cells)
fig, axes = plt.subplots(n_ex, 2, figsize=(13, 3.2 * n_ex))
if n_ex == 0:
    print("no significant precession cells to plot")
else:
    axes = np.atleast_2d(axes)
    for j, row in enumerate(sig_cells):
        f = next(f for f in fields if f["uid"] == row["uid"]
                 and f["direction"] == row["direction"])
        ax = axes[j, 0]
        mid_bin = np.clip(int(np.argmax(f["rate"])), 0, N_BINS - 1)
        rate_sm = smooth1d(np.nan_to_num(f["rate"], nan=0.0))
        ax.plot(bin_c, rate_sm, "k", lw=1.5)
        ax.axvspan(row["lo"], row["hi"], color="C1", alpha=0.25)
        ax.set_title(f"unit {row['uid']}, dir {row['direction']:+.0f}, "
                     f"rho={row['rho']:.2f}, p={row['p']:.3f}")
        ax.set_xlabel("position (m)"); ax.set_ylabel("rate (Hz)")

        ax = axes[j, 1]
        # color each lap by run bout order
        uid = row["uid"]; d = row["direction"]
        lo, hi = row["lo"], row["hi"]
        st = np.asarray(units[uid].t)
        laps_plot = []
        for bi, (tb, xv) in enumerate(run_by_dir[d]):
            m = (st >= tb[0]) & (st <= tb[-1])
            xq = np.interp(st[m], tb, xv)
            k = (xq >= lo) & (xq <= hi)
            if k.sum() >= 3:
                if d == 1:
                    pr = np.clip((xq[k] - lo) / (hi - lo), 0, 1)
                else:
                    pr = np.clip((hi - xq[k]) / (hi - lo), 0, 1)
                ph = theta_phase_at(st[m][k])
                ax.scatter(pr, np.degrees(ph), s=9, alpha=0.75,
                           c=[bi] * len(pr), cmap="rainbow")
        ax.set_xlabel("progress through field"); ax.set_ylabel("theta phase (deg)")
        ax.set_ylim(-180, 180)
        ax.set_title("Phase vs. progress, lap-colored")
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig04_example_cells.png", dpi=150)
    plt.close()

# %% [markdown]
# ## 10. Population summary figures
#
# Figure 5 shows the population readout: the fitted slope (cycles per field
# width) for all cell-directions with >= 30 in-field spikes, the signed
# circular-linear correlation rho, and the counts of significant negative
# (precession) vs significant positive (anti-precession) cell-directions.

# %%
all_rows = rows
slopes = np.array([r["slope_cyc_per_field"] for r in all_rows])
rhos = np.array([r["rho"] for r in all_rows])
ps = np.array([r["p"] for r in all_rows])

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax = axes[0, 0]
ax.hist(slopes, bins=40, color="0.6", edgecolor="k")
ax.set_xlabel("slope (cycles / field)"); ax.set_ylabel("count")
ax.set_title(f"Fitted precession slope (n={len(slopes)})")

ax = axes[0, 1]
ax.hist(rhos, bins=40, color="0.6", edgecolor="k")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("circular-linear rho"); ax.set_ylabel("count")
ax.set_title("Signed circular-linear correlation")

ax = axes[1, 0]
ax.scatter(slopes, rhos, s=14, c="0.3", alpha=0.8)
ax.axhline(0, color="k", lw=0.7); ax.axvline(0, color="k", lw=0.7)
ax.set_xlabel("slope (cyc/field)"); ax.set_ylabel("rho")
ax.set_title("rho vs slope")

ax = axes[1, 1]
labels = ["sig neg\n(precession)", "sig pos", "non-sig"]
counts = [len(sig_neg), len(sig_pos),
          len(all_rows) - len(sig_neg) - len(sig_pos)]
bars = ax.bar(labels, counts, color=["C0", "C3", "0.7"])
ax.bar_label(bars)
ax.set_ylabel("cell-directions")
ax.set_title("Significance (phase-permutation, p<0.05)")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig05_population.png", dpi=150)
plt.close()

# %% [markdown]
# ## 11. Pooled phase-progress relationship
#
# To visualize the population-level precession we pool all significant
# precession cells. Because each cell has its own phase offset (the phase at
# the start of its field), we subtract each cell's fitted phase offset
# phi0 = angle(mean(exp(i*phase - i*2*pi*slope*progress))) before pooling,
# otherwise the random offsets smear out the wedge. We then bin progress into
# 20 bins and show the circular mean phase in each bin plus the fitted slope.

# %%
xs_all = []; ys_all = []
for r in sig_neg:
    f = next(f for f in fields if f["uid"] == r["uid"]
             and f["direction"] == r["direction"])
    pr = np.clip(f["progress"], 0, 1)
    ph = f["phase"]
    sl = r["slope_cyc_per_field"]
    phi0 = np.angle(np.mean(np.exp(1j * (ph - 2 * np.pi * sl * pr))))
    ycorr = np.mod(ph - phi0 + np.pi, 2 * np.pi) - np.pi
    xs_all.append(pr); ys_all.append(ycorr)
xs_all = np.concatenate(xs_all)
ys_all = np.concatenate(ys_all)

# bin means (circular mean of corrected phase per progress bin)
edges = np.linspace(0, 1, 21)
bin_idx = np.clip(np.digitize(xs_all, edges) - 1, 0, 19)
means = np.full(20, np.nan)
errs = np.full(20, np.nan)
for b in range(20):
    yb = ys_all[bin_idx == b]
    if len(yb) < 5:
        continue
    rl = np.abs(np.mean(np.exp(1j * yb)))
    mb = np.angle(np.mean(np.exp(1j * yb)))
    means[b] = mb
    errs[b] = np.sqrt(-2 * np.log(rl)) if rl < 1 else 0.0

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter(xs_all, np.degrees(ys_all), s=4, alpha=0.25, color="0.4")
bc = 0.5 * (edges[:-1] + edges[1:])
for b in range(20):
    if np.isfinite(means[b]):
        ax.errorbar(bc[b], np.degrees(means[b]),
                    yerr=np.degrees(errs[b]), fmt="o", ms=5,
                    color="C0", ecolor="C0", capsize=2)
sl_pop = np.median([r["slope_cyc_per_field"] for r in sig_neg])
xline = np.linspace(0.02, 0.98, 50)
ph0_pop = np.angle(np.mean(np.exp(1j * (ys_all - 2 * np.pi * sl_pop * xs_all))))
yline = np.degrees(np.mod(ph0_pop - 2 * np.pi * sl_pop * xline + np.pi, 2 * np.pi) - np.pi)
ax.plot(xline, yline, "C1", lw=2, label=f"median slope {sl_pop:.2f} cyc/field")
ax.set_xlabel("progress through field"); ax.set_ylabel("theta phase (deg, offset-corrected)")
ax.set_ylim(-180, 180); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.legend()
ax.set_title(f"Pooled phase vs. progress ({len(sig_neg)} significant cells)")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig06_pooled_progress.png", dpi=150)
plt.close()

# %% [markdown]
# ## 12. Per-lap precession
#
# The per-lap circular-linear correlation distribution confirms that
# precession is visible within individual traversals of the field, not just
# in the pooled scatter.

# %%
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.hist(laps, bins=40, color="0.6", edgecolor="k")
ax.axvline(0, color="k", lw=1)
ax.axvline(np.median(laps), color="C1", lw=2,
           label=f"median {np.median(laps):.2f}")
ax.set_xlabel("per-lap circular-linear rho")
ax.set_ylabel("count")
ax.set_title(f"Per-lap correlation ({len(laps)} laps)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig07_perlap.png", dpi=150)
plt.close()

# %% [markdown]
# ## 13. Summary and saving results

# %%
# Save the core results as JSON for downstream reporting.
summary = {
    "dandiset": "000044",
    "session": "sub-Achilles_ses-Achilles-10252013_behavior+ecephys",
    "maze_epoch_s": [maze_start, maze_end],
    "n_units_total": len(unit_ids),
    "n_excitatory": len(exc_ids),
    "n_run_bouts": len(bouts),
    "run_time_s": round(tot, 1),
    "theta_channel": theta_ch,
    "theta_peak_hz": round(peak_f, 2),
    "n_place_fields": len(fields),
    "n_cells_with_field": len({f["uid"] for f in fields}),
    "n_fitted": len(rows),
    "n_sig_neg": len(sig_neg),
    "n_sig_pos": len(sig_pos),
    "median_slope_sig_neg": (round(float(np.median([r["slope_cyc_per_field"]
        for r in sig_neg])), 3) if sig_neg else None),
    "median_rho_sig_neg": (round(float(np.median([r["rho"] for r in sig_neg])), 3)
        if sig_neg else None),
    "n_laps": len(laps),
    "median_lap_rho": (round(float(np.median(laps)), 3) if len(laps) else None),
    "lap_negative_fraction": (round(float((laps < 0).mean()), 3) if len(laps) else None),
    "figures": sorted(os.listdir(FIGDIR)),
}
with open("results_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print("\n===== FINAL SUMMARY =====")
for k, v in summary.items():
    print(f"  {k}: {v}")
