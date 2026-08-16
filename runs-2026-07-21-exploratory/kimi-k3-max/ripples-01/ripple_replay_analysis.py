# %% [markdown]
# # Sharp-Wave Ripples and Replay in Rat Hippocampus (DANDI 000044)
#
# This notebook demonstrates the two hallmark signatures of hippocampal
# memory reactivation using a public dataset from the DANDI Archive:
#
# 1. **Sharp-wave ripples (SWRs)**: brief (~50-100 ms), high-frequency
#    (100-250 Hz) oscillations of the CA1 local field potential that occur
#    during non-REM sleep and quiet wakefulness, riding on top of slow
#    sharp waves.
# 2. **Replay**: during SWRs after spatial experience, place cells
#    reactivate in rapid sequences that recapitulate trajectories through
#    the environment, compressed ~10-20x relative to behavior.
#
# **Dataset**: DANDI dandiset
# [000044](https://dandiarchive.org/dandiset/000044) ("Diversity in neural
# firing dynamics supports both rigid and learned hippocampal sequences",
# Buzsaki lab), session `sub-Achilles_ses-Achilles-10252013`. A rat ran
# ~42 laps on a 1.6 m linear track (MazeEpoch) between ~5 h of pre-sleep
# (PREEpoch) and ~4 h of post-sleep (POSTEpoch). The NWB file contains
# 137 sorted CA1 units (120 excitatory, 17 inhibitory), 128-channel LFP
# at 1250 Hz, 2D + linearized position at 39 Hz, and sleep-state labels
# (Awake / Non-REM / REM).
#
# **Approach**:
# - Stream the 8.7 GB NWB file with `remfile` (HTTP range requests + disk
#   cache); no full download.
# - Pick the LFP channel with the strongest ripple content data-driven.
# - Detect SWRs in POST Non-REM (bandpass 100-250 Hz, Hilbert envelope,
#   mean+4SD peak threshold, mean+1SD boundaries, 30-500 ms duration).
# - Build place-field rate maps from run bouts on the track
#   (Skaggs spatial information, circular time-shift shuffle).
# - Bayesian-decode position in 20 ms bins during each SWR and score
#   sequential structure with the posterior-weighted correlation between
#   time and position; test significance against cell-identity and
#   within-event time-bin shuffles.
# - Control: decode PRE-sleep SWRs with the same maze template
#   (experience has not happened yet in this session).

# %% [markdown]
# ## Setup and Streaming Data Access

# %%
import os
import requests
import h5py
import remfile
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from scipy import signal, ndimage, stats
from tqdm import tqdm
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
LFP_PATH = "processing/ecephys/LFP/LFP"
LFP_FS = 1250.0
LFP_CONVERSION = 3.815e-7  # V per count

# Resolve the presigned S3 URL through the DANDI API (URLs expire, so this
# must be done fresh on each run).
r = requests.get(
    f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/",
    params={"path": ASSET_PATH})
r.raise_for_status()
asset_id = r.json()["results"][0]["asset_id"]
r2 = requests.get(f"https://api.dandiarchive.org/api/assets/{asset_id}/download/",
                  allow_redirects=False)
s3_url = r2.headers["Location"]

disk_cache = remfile.DiskCache("/tmp/remfile_cache_ripples")
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())
print(nwb)


def load_lfp_channel(ch, t0, t1, fs=LFP_FS):
    """Load one LFP channel over [t0, t1) seconds -> (t, microvolts)."""
    data = h5py_file[f"{LFP_PATH}/data"]
    i0, i1 = int(t0 * fs), int(min(t1 * fs, data.shape[0]))
    x = data[i0:i1, ch].astype(np.float64) * LFP_CONVERSION * 1e6
    return np.arange(i0, i1) / fs, x


def load_lfp_block(chs, t0, t1, fs=LFP_FS):
    """Load several LFP channels -> (t, array[n_samples, n_ch]) in uV."""
    data = h5py_file[f"{LFP_PATH}/data"]
    i0, i1 = int(t0 * fs), int(min(t1 * fs, data.shape[0]))
    x = data[i0:i1, list(chs)].astype(np.float64) * LFP_CONVERSION * 1e6
    return np.arange(i0, i1) / fs, x


# %% [markdown]
# ## Dataset Inspection
#
# The session has three epochs (PRE sleep, Maze, POST sleep) and a
# sleep-state table. Units carry `cell_type` and `location` metadata.

# %%
epochs_raw = nwb["epochs"]
epochs = {lab: epochs_raw[epochs_raw.label == lab]
          for lab in np.unique(epochs_raw.label)}
states_raw = nwb["states"]
states = {lab: states_raw[states_raw.label == lab]
          for lab in np.unique(states_raw.label)}
units = nwb["units"]

print("=== epochs ===")
print(epochs_raw)
print("=== states ===")
print(states_raw)
print("=== units ===")
print(units.get_info("cell_type").value_counts())
print(units.get_info("location").value_counts())

pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print(f"position: {pos2d.shape[0]} samples at "
      f"{1/np.median(np.diff(pos2d.t)):.1f} Hz over the maze epoch")
print(f"LFP: {h5py_file[f'{LFP_PATH}/data'].shape} int16 at {LFP_FS} Hz")

# %% [markdown]
# ## Ripple Channel Selection
#
# The electrode table has no anatomical coordinates, so we pick the ripple
# channel data-driven: load a 120 s Non-REM window from POST sleep on all
# 128 channels and score each by ripple-band (100-250 Hz) power, envelope
# "peakiness" (99th percentile / median, which favors transient ripple
# events over steadily noisy channels), and ripple-to-delta ratio.

# %%
post = epochs["POSTEpoch"]
post_start, post_end = float(post.start[0]), float(post.end[0])
nrem_post = states["Non-REM"]
nrem_post = nrem_post[(nrem_post.end > post_start) & (nrem_post.start < post_end)]
print(f"POST Non-REM: {np.sum(nrem_post.end - nrem_post.start):.0f} s "
      f"in {len(nrem_post)} bouts")

durs = nrem_post.end - nrem_post.start
t0 = nrem_post.start[int(np.argmax(durs))] + 5
t_blk, X = load_lfp_block(range(128), t0, t0 + 120)

sos_rip = signal.butter(4, [100, 250], btype="band", fs=LFP_FS, output="sos")
sos_del = signal.butter(4, [1, 4], btype="band", fs=LFP_FS, output="sos")
Xr = signal.sosfiltfilt(sos_rip, X, axis=0)
Xd = signal.sosfiltfilt(sos_del, X, axis=0)
rip_rms = np.sqrt(np.mean(Xr**2, axis=0))
del_rms = np.sqrt(np.mean(Xd**2, axis=0))
env_blk = np.abs(signal.hilbert(Xr, axis=0))
peakiness = np.percentile(env_blk, 99, axis=0) / (np.median(env_blk, axis=0) + 1e-9)
score = peakiness * rip_rms / del_rms
best_ch = int(np.argmax(score))
print(f"selected channel {best_ch}: ripple RMS {rip_rms[best_ch]:.1f} uV, "
      f"peakiness {peakiness[best_ch]:.1f}, ripple/delta {rip_rms[best_ch]/del_rms[best_ch]:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(range(128), rip_rms, color="steelblue")
axes[0].axvline(best_ch, color="red", ls="--")
axes[0].set_xlabel("LFP channel"); axes[0].set_ylabel("ripple-band RMS (uV)")
axes[0].set_title("Ripple-band (100-250 Hz) power")
axes[1].bar(range(128), peakiness, color="seagreen")
axes[1].axvline(best_ch, color="red", ls="--")
axes[1].set_xlabel("LFP channel"); axes[1].set_ylabel("envelope p99 / median")
axes[1].set_title("Ripple envelope peakiness")
axes[2].bar(range(128), score, color="darkorange")
axes[2].axvline(best_ch, color="red", ls="--", label=f"ch {best_ch}")
axes[2].set_xlabel("LFP channel"); axes[2].set_ylabel("peakiness x ripple/delta")
axes[2].set_title("Combined channel score"); axes[2].legend()
plt.tight_layout()
plt.savefig("fig_channel_selection.png", dpi=150)
plt.close()

# visual check: raw + ripple-filtered traces for the top 4 channels
top4 = np.argsort(score)[-4:][::-1]
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
seg = (t_blk >= t0 + 30) & (t_blk < t0 + 31.5)
for ax, ch in zip(axes, top4):
    ax.plot(t_blk[seg], X[seg, ch], color="gray", lw=0.5, label="raw")
    ax.plot(t_blk[seg], Xr[seg, ch] * 3, color="navy", lw=0.7, label="ripple band x3")
    ax.set_ylabel(f"ch {ch}\n(uV)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"channel {ch}: peakiness {peakiness[ch]:.2f}, "
                 f"ripple/delta {rip_rms[ch]/del_rms[ch]:.3f}", fontsize=10)
axes[-1].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig("fig_channel_traces.png", dpi=150)
plt.close()

# %% [markdown]
# ## Sharp-Wave Ripple Detection in POST Sleep
#
# Standard SWR detection: bandpass the channel at 100-250 Hz, take the
# Hilbert envelope, smooth (Gaussian, sigma 4 ms), and threshold. Peaks
# must exceed mean+4SD of the Non-REM envelope; event boundaries are at
# mean+1SD; events closer than 30 ms are merged; accepted durations are
# 30-500 ms. Only events fully inside Non-REM bouts are kept.

# %%
PEAK_Z, EDGE_Z = 4.0, 1.0
MERGE_GAP, MIN_DUR, MAX_DUR = 0.030, 0.030, 0.500

chunk_s = 200.0
edges_ld = np.arange(post_start, post_end + chunk_s, chunk_s)
parts = []
for a_, b_ in tqdm(list(zip(edges_ld[:-1], edges_ld[1:])), desc="loading POST LFP"):
    _, x = load_lfp_channel(best_ch, a_, b_)
    parts.append(x)
lfp_post = np.concatenate(parts)
t_post = np.arange(len(lfp_post)) / LFP_FS + post_start

lfp_filt = signal.sosfiltfilt(sos_rip, lfp_post)
env_s = ndimage.gaussian_filter1d(np.abs(signal.hilbert(lfp_filt)), 0.004 * LFP_FS)

nrem_mask = np.zeros(len(lfp_post), dtype=bool)
for s, e in zip(nrem_post.start, nrem_post.end):
    nrem_mask[max(0, int((s - post_start) * LFP_FS)):
              min(len(lfp_post), int((e - post_start) * LFP_FS))] = True

mu, sd = env_s[nrem_mask].mean(), env_s[nrem_mask].std()
thr_peak, thr_edge = mu + PEAK_Z * sd, mu + EDGE_Z * sd
print(f"envelope mean {mu:.1f} uV, sd {sd:.1f}; "
      f"peak thr {thr_peak:.1f}, edge thr {thr_edge:.1f}")


def detect_events(env_s, nrem_mask, thr_peak, thr_edge, fs=LFP_FS):
    above = env_s > thr_peak
    d = np.diff(above.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [len(above)]
    events = []
    for s, e in zip(starts, ends):
        while s > 0 and env_s[s] > thr_edge:
            s -= 1
        while e < len(env_s) and env_s[e] > thr_edge:
            e += 1
        events.append([s, e])
    merged = []
    for ev in events:
        if merged and (ev[0] - merged[-1][1]) < MERGE_GAP * fs:
            merged[-1][1] = ev[1]
        else:
            merged.append(list(ev))
    return [(s, e) for s, e in merged
            if MIN_DUR <= (e - s) / fs <= MAX_DUR and nrem_mask[s:e].all()]


keep = detect_events(env_s, nrem_mask, thr_peak, thr_edge)
ev_start = np.array([t_post[s] for s, e in keep])
ev_end = np.array([t_post[e] for s, e in keep])
ev_peak_t = np.array([t_post[s + np.argmax(env_s[s:e])] for s, e in keep])
ev_peak_amp = np.array([env_s[s:e].max() for s, e in keep])
ev_dur = ev_end - ev_start
print(f"detected {len(keep)} SWRs in POST Non-REM "
      f"({len(keep)/(nrem_mask.sum()/LFP_FS)*60:.1f} per min)")

# %% [markdown]
# ### Ripple Detection Validation

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
seg_t0 = 25010.0
seg = (t_post >= seg_t0) & (t_post < seg_t0 + 4)
axes[0].plot(t_post[seg], lfp_post[seg], color="gray", lw=0.5)
axes[0].set_ylabel("raw LFP (uV)")
axes[0].set_title(f"Ripple detection on ch {best_ch}, POST Non-REM example")
axes[1].plot(t_post[seg], lfp_filt[seg], color="navy", lw=0.7)
axes[1].set_ylabel("100-250 Hz (uV)")
axes[2].plot(t_post[seg], env_s[seg], color="darkred", lw=0.8)
axes[2].axhline(thr_peak, color="red", ls="--", lw=1, label=f"peak thr (mean+{PEAK_Z}SD)")
axes[2].axhline(thr_edge, color="orange", ls="--", lw=1, label=f"edge thr (mean+{EDGE_Z}SD)")
axes[2].set_ylabel("envelope (uV)"); axes[2].set_xlabel("time (s)")
axes[2].legend(loc="upper right", fontsize=8)
for ax in axes:
    for s, e in zip(ev_start, ev_end):
        if e > seg_t0 and s < seg_t0 + 4:
            ax.axvspan(s, e, color="red", alpha=0.15)
plt.tight_layout()
plt.savefig("fig_ripple_detection_example.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
vals = env_s[nrem_mask][::20]
ax.hist(vals, bins=200, range=(0, np.percentile(vals, 99.9)),
        color="steelblue", density=True)
ax.axvline(thr_peak, color="red", ls="--", label=f"peak thr = {thr_peak:.1f} uV")
ax.axvline(thr_edge, color="orange", ls="--", label=f"edge thr = {thr_edge:.1f} uV")
ax.set_xlabel("ripple-band envelope (uV)"); ax.set_ylabel("density")
ax.set_title("Envelope distribution over POST Non-REM"); ax.legend()
plt.tight_layout()
plt.savefig("fig_envelope_distribution.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(ev_dur * 1000, bins=50, color="steelblue")
axes[0].set_xlabel("duration (ms)"); axes[0].set_ylabel("count")
axes[0].set_title(f"Ripple durations (median {np.median(ev_dur)*1000:.0f} ms)")
axes[1].hist(ev_peak_amp, bins=50, color="darkorange")
axes[1].set_xlabel("peak envelope (uV)")
axes[1].set_title(f"Peak amplitudes (median {np.median(ev_peak_amp):.0f} uV)")
bins = np.arange(post_start, post_end, 600)
counts, _ = np.histogram(ev_start, bins=bins)
axes[2].plot((bins[:-1] - post_start) / 3600, counts / 10, color="seagreen")
axes[2].set_xlabel("hours into POST"); axes[2].set_ylabel("ripples per min")
axes[2].set_title("Ripple rate across POST epoch")
plt.tight_layout()
plt.savefig("fig_ripple_properties.png", dpi=150)
plt.close()

# peri-event averages aligned to the envelope peak: the sharp wave and the
# ~150 Hz oscillation are both visible in the event-locked average
win = 0.4
nw = int(win * LFP_FS)
segs_raw, segs_filt, segs_env = [], [], []
for pt in ev_peak_t:
    c = int((pt - post_start) * LFP_FS)
    if c - nw >= 0 and c + nw < len(lfp_post):
        segs_raw.append(lfp_post[c - nw:c + nw])
        segs_filt.append(lfp_filt[c - nw:c + nw])
        segs_env.append(env_s[c - nw:c + nw])
segs_raw, segs_filt, segs_env = map(np.array, (segs_raw, segs_filt, segs_env))
taxis = (np.arange(2 * nw) - nw) / LFP_FS * 1000
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, segs_, col, title in zip(
        axes, (segs_raw, segs_filt, segs_env), ("gray", "navy", "darkred"),
        ("Mean raw LFP (sharp wave)", "Mean ripple-band LFP", "Mean ripple envelope")):
    m = segs_.mean(axis=0)
    s_ = segs_.std(axis=0) / np.sqrt(len(segs_))
    ax.plot(taxis, m, color=col)
    ax.fill_between(taxis, m - s_, m + s_, alpha=0.3, color=col)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_title(title); ax.set_xlabel("ms from peak")
axes[0].set_ylabel("uV")
plt.tight_layout()
plt.savefig("fig_ripple_perievent.png", dpi=150)
plt.close()

# %% [markdown]
# ## Place Fields on the Linear Track
#
# Run bouts are contiguous stretches with valid linearized position (the
# series is NaN off-track), merged over gaps < 0.3 s, requiring >= 1 s
# duration, > 0.3 m span, and median speed > 0.15 m/s. Rate maps (50 bins,
# 1.5-bin Gaussian smoothing) are computed with pynapple, pooled over
# running direction. Place cells are excitatory units with Skaggs spatial
# information above a circular time-shift shuffle (spike times are shifted
# on a concatenated run-bout time axis so spike counts stay fixed), peak
# rate >= 1 Hz, and mean run rate >= 0.1 Hz.

# %%
N_BINS, TRACK_LEN, SMOOTH_BINS, MIN_OCC_S = 50, 1.6, 1.5, 0.1
N_SHUFFLE_SI = 200
RNG_SI = np.random.default_rng(42)

maze = epochs["MazeEpoch"]
t_pos_all = lin.t
x_all = np.asarray(lin.values).ravel()
valid = ~np.isnan(x_all)
d = np.diff(valid.astype(int))
starts = list(np.where(d == 1)[0] + 1)
ends = list(np.where(d == -1)[0] + 1)
if valid[0]:
    starts = [0] + starts
if valid[-1]:
    ends = ends + [len(valid)]
bouts = [[t_pos_all[s], t_pos_all[e - 1]] for s, e in zip(starts, ends)]
merged = []
for b in bouts:
    if merged and b[0] - merged[-1][1] < 0.3:
        merged[-1][1] = b[1]
    else:
        merged.append(list(b))
run_bouts = []
for a, b in merged:
    if b - a < 1.0:
        continue
    m = (t_pos_all >= a) & (t_pos_all <= b) & valid
    if m.sum() < 5:
        continue
    xs = x_all[m]
    if xs.max() - xs.min() < 0.3:
        continue
    if np.median(np.abs(np.diff(xs)) / np.diff(t_pos_all[m])) < 0.15:
        continue
    run_bouts.append((a, b))
run_iset = nap.IntervalSet(start=[a for a, b in run_bouts],
                           end=[b for a, b in run_bouts])
total_run = float(np.sum([b - a for a, b in run_bouts]))
print(f"{len(run_bouts)} run bouts, {total_run:.0f} s")

cell_type = units.get_info("cell_type")
exc_keys = cell_type[cell_type == "excitatory"].index.to_numpy()
exc_units = units[list(exc_keys)]

lin_maze = lin.restrict(maze)
lin_frame = nap.TsdFrame(t=lin_maze.t, d=np.asarray(lin_maze.values),
                         columns=["lin"], time_support=lin_maze.time_support)
edges = np.linspace(0, TRACK_LEN, N_BINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
tc = nap.compute_tuning_curves(exc_units, lin_frame, bins=[edges], epochs=run_iset)
fs_pos = 1.0 / np.median(np.diff(t_pos_all))
lin_run = lin_frame.restrict(run_iset)
x_run_all = np.asarray(lin_run.values).ravel()
occ_s = np.histogram(x_run_all[~np.isnan(x_run_all)], bins=edges)[0] / fs_pos
rate = np.asarray(tc).copy()
low_occ = occ_s < MIN_OCC_S
rate[:, low_occ] = np.nan
rate_sm = np.array([ndimage.gaussian_filter1d(np.nan_to_num(r), SMOOTH_BINS)
                    for r in rate])
rate_sm[:, low_occ] = np.nan


def skaggs_si(rate_map, occ_map):
    r = np.nan_to_num(rate_map)
    o = occ_map / np.nansum(occ_map)
    mean_r = np.nansum(o * r)
    if mean_r <= 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(o * r * np.log2(r / mean_r)) / mean_r)


# concatenated run-bout time axis for circular shifts
bout_starts = np.array([a for a, b in run_bouts])
bout_ends = np.array([b for a, b in run_bouts])
bout_durs = bout_ends - bout_starts
cum_dur = np.concatenate([[0], np.cumsum(bout_durs)])[:-1]
T_run = bout_durs.sum()


def wall_to_tau(ts):
    i = np.searchsorted(bout_starts, ts, side="right") - 1
    i = np.clip(i, 0, len(bout_starts) - 1)
    return ts - bout_starts[i] + cum_dur[i]


pos_run = lin_maze.restrict(run_iset)
t_run, x_run = pos_run.t, np.asarray(pos_run.values).ravel()
valid_run = ~np.isnan(x_run)
tau_pos = wall_to_tau(t_run[valid_run])
x_run_v = x_run[valid_run]
order_tau = np.argsort(tau_pos)
tau_pos, x_run_v = tau_pos[order_tau], x_run_v[order_tau]


def si_from_tau(sp_tau):
    idx = np.clip(np.searchsorted(tau_pos, sp_tau), 0, len(x_run_v) - 1)
    h, _ = np.histogram(x_run_v[idx], bins=edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        rm = h / occ_s
    return skaggs_si(ndimage.gaussian_filter1d(np.nan_to_num(rm), SMOOTH_BINS), occ_s)


spk_tau_all = [wall_to_tau(exc_units[k].restrict(run_iset).t) for k in exc_keys]
si_real = np.array([si_from_tau(st) for st in spk_tau_all])
si_null = np.zeros((len(exc_keys), N_SHUFFLE_SI))
for j in tqdm(range(N_SHUFFLE_SI), desc="SI shuffle"):
    shift = RNG_SI.uniform(0, T_run)
    for i in range(len(exc_keys)):
        si_null[i, j] = si_from_tau(np.sort((spk_tau_all[i] + shift) % T_run))
si_p = (np.sum(si_null >= si_real[:, None], axis=1) + 1) / (N_SHUFFLE_SI + 1)

peak_rate = np.nanmax(rate_sm, axis=1)
n_spikes = np.array([len(exc_units[k].restrict(run_iset)) for k in exc_keys])
is_place = (si_p < 0.05) & (peak_rate >= 1.0) & (n_spikes / total_run >= 0.1)
print(f"place cells: {is_place.sum()}/{len(exc_keys)}; "
      f"median SI {np.median(si_real):.2f} vs null {np.median(si_null):.2f} bits/spike")

# %% [markdown]
# ### Place Field Figures

# %%
order = np.argsort(si_real)[::-1]
fig, axes = plt.subplots(3, 4, figsize=(14, 8))
for ax, i in zip(axes.ravel(), order[:12]):
    ax.plot(centers, rate_sm[i], color="navy")
    ax.fill_between(centers, 0, rate_sm[i], color="navy", alpha=0.2)
    ax.set_title(f"unit {exc_keys[i]}: SI {si_real[i]:.2f}, p={si_p[i]:.3f}", fontsize=9)
    ax.set_ylim(bottom=0); ax.set_xticks([0, 0.8, 1.6])
for ax in axes[-1]:
    ax.set_xlabel("track position (m)")
for ax in axes[:, 0]:
    ax.set_ylabel("rate (Hz)")
plt.suptitle("Example place fields (top 12 by spatial information)")
plt.tight_layout()
plt.savefig("fig_placefields_examples.png", dpi=150)
plt.close()

pc_idx = np.where(is_place)[0]
pk = np.array([np.nanargmax(rate_sm[i]) for i in pc_idx])
pc_order = pc_idx[np.argsort(pk)]
norm = np.array([r / np.nanmax(r) if np.nanmax(r) > 0 else r
                 for r in rate_sm[pc_order]])
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(norm, aspect="auto", extent=[0, TRACK_LEN, len(pc_order), 0],
          cmap="viridis", interpolation="nearest")
ax.set_xlabel("track position (m)"); ax.set_ylabel("place cell (sorted by peak)")
ax.set_title(f"{len(pc_order)} place cells, direction-pooled rate maps")
plt.tight_layout()
plt.savefig("fig_placefields_population.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(si_null.ravel(), bins=60, density=True, alpha=0.6, color="gray",
        label="time-shift null")
ax.hist(si_real, bins=60, density=True, alpha=0.6, color="seagreen",
        label="excitatory units")
ax.set_xlabel("Skaggs spatial information (bits/spike)")
ax.set_ylabel("density")
ax.set_title(f"Spatial information: {is_place.sum()} place cells")
ax.legend()
plt.tight_layout()
plt.savefig("fig_placefields_si.png", dpi=150)
plt.close()

# %% [markdown]
# ## Replay: Bayesian Decoding of Ripple Events
#
# Each SWR is divided into 20 ms bins. Under a Poisson encoding model with
# the place-field rate maps f_i(x), the posterior over track position is
#
#     P(x | n)  proportional to  prod_i f_i(x)^n_i * exp(-tau * sum_i f_i(x))
#
# Sequential structure is scored with the **weighted correlation**: the
# correlation between bin time and position, with each (time, position)
# pair weighted by its posterior mass. Events need >= 5 bins (>= 100 ms)
# and >= 5 active place cells. Significance per event comes from two
# shuffles (500 each): (A) permuting rate maps across cells (destroys
# spatial coding, keeps spike statistics) and (B) circularly shifting each
# cell's spike counts across time bins within the event (destroys temporal
# order, keeps within-event co-firing statistics).

# %%
BIN, MIN_BINS, MIN_CELLS = 0.020, 5, 5
N_SHUFFLE = 500
RATE_FLOOR = 0.01
RNG = np.random.default_rng(7)

place_keys = exc_keys[is_place]
template = np.clip(np.nan_to_num(rate_sm[is_place]), RATE_FLOOR, None)
C, P = template.shape
logf = np.log(template)
sumf = template.sum(axis=0)
spikes = [units[k].t for k in place_keys]
print(f"template: {C} place cells x {P} position bins")


def collect_qualified(ev_starts, ev_ends):
    all_counts, qual_idx = [], []
    for ei_, (a, b) in enumerate(zip(ev_starts, ev_ends)):
        ed = np.arange(a, b + 1e-9, BIN)
        if len(ed) < 2:
            continue
        counts = np.zeros((len(ed) - 1, C))
        for c, sp in enumerate(spikes):
            i0, i1 = np.searchsorted(sp, a), np.searchsorted(sp, b)
            if i1 > i0:
                counts[:, c] = np.histogram(sp[i0:i1], bins=ed)[0]
        if counts.shape[0] >= MIN_BINS and (counts.sum(axis=0) > 0).sum() >= MIN_CELLS:
            all_counts.append(counts)
            qual_idx.append(ei_)
    return all_counts, qual_idx


def build_padded(all_counts):
    E = len(all_counts)
    Bmax = max(c.shape[0] for c in all_counts)
    counts_pad = np.zeros((E, Bmax, C))
    bin_mask = np.zeros((E, Bmax), dtype=bool)
    t_rel = np.zeros((E, Bmax))
    for e, c in enumerate(all_counts):
        nb = c.shape[0]
        counts_pad[e, :nb] = c
        bin_mask[e, :nb] = True
        t_rel[e, :nb] = (np.arange(nb) + 0.5) * BIN
    return counts_pad, bin_mask, t_rel, np.array([c.shape[0] for c in all_counts])


def decode_events(counts_pad, bin_mask, t_rel, logf_mat):
    lp = counts_pad.reshape(-1, C) @ logf_mat - BIN * sumf[None, :]
    lp -= lp.max(axis=1, keepdims=True)
    post = np.exp(lp)
    post /= post.sum(axis=1, keepdims=True)
    post = post.reshape(*counts_pad.shape[:2], P)
    W = post * bin_mask[:, :, None]
    T = t_rel[:, :, None]
    X = centers[None, None, :]
    ws = W.sum(axis=(1, 2), keepdims=True)
    mt = (W * T).sum(axis=(1, 2), keepdims=True) / ws
    mx = (W * X).sum(axis=(1, 2), keepdims=True) / ws
    cov = (W * (T - mt) * (X - mx)).sum(axis=(1, 2))
    vt = (W * (T - mt) ** 2).sum(axis=(1, 2))
    vx = (W * (X - mx) ** 2).sum(axis=(1, 2))
    wc = cov / np.sqrt(vt * vx)
    return post, wc


post_counts, post_qual_idx = collect_qualified(ev_start, ev_end)
counts_pad, bin_mask, t_rel, n_bins_arr = build_padded(post_counts)
E = len(post_counts)
print(f"qualified POST events (>= {MIN_BINS} bins, >= {MIN_CELLS} cells): {E}")

post_real, wc_real = decode_events(counts_pad, bin_mask, t_rel, logf)
print(f"median |weighted corr|: {np.median(np.abs(wc_real)):.3f}")

wc_null_cellid = np.zeros((E, N_SHUFFLE))
for j in tqdm(range(N_SHUFFLE), desc="cell-ID shuffle"):
    _, wc_null_cellid[:, j] = decode_events(counts_pad, bin_mask, t_rel,
                                            logf[RNG.permutation(C)])

wc_null_binshift = np.zeros((E, N_SHUFFLE))
bidx = np.arange(counts_pad.shape[1])[None, :, None]
for j in tqdm(range(N_SHUFFLE), desc="bin-shift shuffle"):
    shifts = RNG.integers(0, np.maximum(n_bins_arr, 1)[:, None], size=(E, C))
    src = (bidx - shifts[:, None, :]) % n_bins_arr[:, None, None]
    rolled = np.take_along_axis(counts_pad, src, axis=1)
    rolled[~bin_mask] = 0.0
    _, wc_null_binshift[:, j] = decode_events(rolled, bin_mask, t_rel, logf)

p_cellid = (np.sum(np.abs(wc_null_cellid) >= np.abs(wc_real[:, None]), axis=1) + 1) / (N_SHUFFLE + 1)
p_binshift = (np.sum(np.abs(wc_null_binshift) >= np.abs(wc_real[:, None]), axis=1) + 1) / (N_SHUFFLE + 1)
sig = p_cellid < 0.05
sig2 = p_binshift < 0.05
print(f"significant (cell-ID): {sig.sum()}/{E} ({100*sig.mean():.1f}%)")
print(f"significant (bin-shift): {sig2.sum()}/{E} ({100*sig2.mean():.1f}%)")
print(f"significant under both: {(sig & sig2).sum()}/{E}")
fwd = int(np.sum((wc_real > 0) & sig))
rev = int(np.sum((wc_real < 0) & sig))
print(f"forward {fwd}, reverse {rev} among cell-ID-significant events")

# %% [markdown]
# ### Replay Figures

# %%
sig_idx = np.where(sig)[0]
top = sig_idx[np.argsort(np.abs(wc_real[sig_idx]))[::-1][:6]]
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, e in zip(axes.ravel(), top):
    nb = n_bins_arr[e]
    ax.imshow(post_real[e, :nb].T, aspect="auto", origin="lower",
              extent=[0, nb * BIN * 1000, 0, 1.6], cmap="hot")
    wm = (post_real[e, :nb] * centers[None, :]).sum(axis=1)
    ax.plot((np.arange(nb) + 0.5) * BIN * 1000, wm, color="cyan", lw=1.2)
    direction = "forward" if wc_real[e] > 0 else "reverse"
    ax.set_title(f"wc={wc_real[e]:.2f}, p={p_cellid[e]:.3f} ({direction})", fontsize=10)
    ax.set_ylabel("track position (m)"); ax.set_xlabel("time in event (ms)")
plt.suptitle("Example replay events: Bayesian position decoding during ripples")
plt.tight_layout()
plt.savefig("fig_replay_examples.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].hist(np.abs(wc_null_cellid).ravel(), bins=80, density=True, alpha=0.6,
             color="gray", label="cell-ID null")
axes[0].hist(np.abs(wc_real), bins=30, density=True, alpha=0.7,
             color="crimson", label="real events")
axes[0].set_xlabel("|weighted correlation|"); axes[0].set_ylabel("density")
axes[0].set_title("Sequence strength: real vs shuffle"); axes[0].legend()
axes[1].hist(p_cellid, bins=25, color="steelblue", alpha=0.8)
axes[1].axvline(0.05, color="red", ls="--")
axes[1].set_xlabel("p value (cell-ID shuffle)"); axes[1].set_ylabel("events")
axes[1].set_title(f"Per-event p values: {sig.sum()}/{E} significant")
axes[2].hist(np.abs(wc_real[sig]), bins=20, color="crimson", alpha=0.8,
             label=f"significant (n={sig.sum()})")
axes[2].hist(np.abs(wc_real[~sig]), bins=20, color="gray", alpha=0.6,
             label=f"not significant (n={(~sig).sum()})")
axes[2].set_xlabel("|weighted correlation|"); axes[2].set_ylabel("events")
axes[2].set_title("Sequence strength by significance"); axes[2].legend()
plt.tight_layout()
plt.savefig("fig_replay_stats.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].bar(["forward", "reverse"], [fwd, rev], color=["seagreen", "darkorange"])
axes[0].set_ylabel("significant events")
axes[0].set_title(f"Direction of significant replay (p<0.05, n={sig.sum()})")
null_q = np.percentile(np.abs(wc_null_cellid), 95, axis=1)
axes[1].scatter(np.abs(wc_real), null_q, c=sig, cmap="bwr", alpha=0.5, s=12)
lims = [0, max(np.abs(wc_real).max(), null_q.max()) * 1.05]
axes[1].plot(lims, lims, "k--", lw=1)
axes[1].set_xlabel("|wc| real"); axes[1].set_ylabel("|wc| null 95th percentile")
axes[1].set_title("Real vs per-event null (red = significant)")
axes[1].set_xlim(lims); axes[1].set_ylim(lims)
plt.tight_layout()
plt.savefig("fig_replay_direction.png", dpi=150)
plt.close()

# %% [markdown]
# ## Peri-Ripple Firing Rates and a Showcase Event
#
# Both pyramidal cells and interneurons increase firing during SWRs, with
# interneurons peaking slightly earlier, a classic physiological signature.
# The showcase figure combines the LFP, the spike raster (cells ordered by
# place-field position), and the decoded posterior for one strong event.

# %%
inh_keys = cell_type[cell_type == "inhibitory"].index.to_numpy()
WIN, PBIN = 0.5, 0.010
pedges = np.arange(-WIN, WIN + 1e-9, PBIN)
pc = 0.5 * (pedges[:-1] + pedges[1:])


def peri_rates(keys):
    rates = np.zeros((len(keys), len(pedges) - 1))
    for i, k in enumerate(keys):
        sp = units[k].t
        i0_all = np.searchsorted(sp, ev_peak_t - WIN)
        i1_all = np.searchsorted(sp, ev_peak_t + WIN)
        rel = [sp[i0:i1] - pk for i0, i1, pk in zip(i0_all, i1_all, ev_peak_t)
               if i1 > i0]
        if rel:
            rates[i], _ = np.histogram(np.concatenate(rel), bins=pedges)
    return rates / (len(ev_peak_t) * PBIN)


def zscore(r, base_mask):
    mu_ = r[:, base_mask].mean(axis=1, keepdims=True)
    sd_ = r[:, base_mask].std(axis=1, keepdims=True)
    sd_[sd_ == 0] = 1.0
    return (r - mu_) / sd_


base_mask = (pc >= -0.5) & (pc <= -0.3)
z_exc = zscore(peri_rates(list(exc_keys)), base_mask)
z_inh = zscore(peri_rates(list(inh_keys)), base_mask)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(pc * 1000, z_exc.mean(axis=0), color="seagreen",
        label=f"excitatory (n={len(exc_keys)})")
ax.plot(pc * 1000, z_inh.mean(axis=0), color="crimson",
        label=f"inhibitory (n={len(inh_keys)})")
ax.axvline(0, color="k", ls=":", lw=0.8)
ax.set_xlabel("time from ripple peak (ms)"); ax.set_ylabel("firing rate (z-scored)")
ax.set_title("Peri-ripple firing: CA1 pyramidal cells and interneurons (POST Non-REM)")
ax.legend()
plt.tight_layout()
plt.savefig("fig_peri_ripple_psth.png", dpi=150)
plt.close()

# showcase event: long, significant, forward, high |wc|
cand = np.where(sig & (n_bins_arr >= 6))[0]
ranked = cand[np.argsort(np.abs(wc_real[cand]))[::-1]]
fwd_cand = [e for e in ranked if wc_real[e] > 0]
show = fwd_cand[3] if len(fwd_cand) > 3 else ranked[0]
show_ripple = post_qual_idx[show]
a, b = ev_start[show_ripple], ev_end[show_ripple]
print(f"showcase ripple {show_ripple}: {a:.3f}-{b:.3f} s, "
      f"wc={wc_real[show]:.2f}, p={p_cellid[show]:.3f}")

pad = 0.08
t_lfp, x_lfp = load_lfp_channel(best_ch, a - pad, b + pad)
x_filt = signal.sosfiltfilt(sos_rip, x_lfp)
peak_pos = np.array([centers[np.argmax(r_)] if r_.max() > 0 else np.nan
                     for r_ in np.nan_to_num(rate_sm[is_place])])
cell_order = np.argsort(peak_pos)
raster = []
for k in np.asarray(place_keys)[cell_order]:
    sp = units[k].t
    raster.append(sp[np.searchsorted(sp, a - pad):np.searchsorted(sp, b + pad)])

nb = n_bins_arr[show]
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.4, 1.4], hspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax1.plot((t_lfp - a) * 1000, x_lfp, color="gray", lw=0.6, label="raw LFP")
ax1.plot((t_lfp - a) * 1000, x_filt * 4, color="navy", lw=0.8, label="ripple band x4")
ax1.axvspan(0, (b - a) * 1000, color="red", alpha=0.08)
ax1.set_ylabel("uV"); ax1.legend(loc="upper right", fontsize=8)
ax1.set_title(f"Ripple event at {a:.2f} s (POST Non-REM): "
              f"wc={wc_real[show]:.2f}, p={p_cellid[show]:.3f}")
ax1.set_xlim(-pad * 1000, (b - a + pad) * 1000)
ax2 = fig.add_subplot(gs[1], sharex=ax1)
for row, spk in enumerate(raster):
    if len(spk):
        ax2.scatter((spk - a) * 1000, np.full(len(spk), row), s=4, color="black")
ax2.axvspan(0, (b - a) * 1000, color="red", alpha=0.08)
ax2.set_ylabel("place cell (sorted by field position)")
ax2.set_ylim(-1, len(raster))
ax3 = fig.add_subplot(gs[2])
ax3.imshow(post_real[show, :nb].T, aspect="auto", origin="lower",
           extent=[0, nb * BIN * 1000, 0, 1.6], cmap="hot")
wm = (post_real[show, :nb] * centers[None, :]).sum(axis=1)
ax3.plot((np.arange(nb) + 0.5) * BIN * 1000, wm, color="cyan", lw=1.5)
ax3.set_xlabel("time from event start (ms)"); ax3.set_ylabel("track position (m)")
ax3.set_title("Bayesian decoded position")
plt.savefig("fig_replay_showcase.png", dpi=150)
plt.close()

# %% [markdown]
# ## Control: PRE-Sleep Ripples Do Not Replay the Maze
#
# If the sequences reflect experience on the track, SWRs during the
# pre-maze sleep should not contain track-like sequences when decoded with
# the maze template. We detect PRE Non-REM ripples with identical criteria
# (PRE-specific envelope statistics) and decode them the same way.

# %%
pre = epochs["PREEpoch"]
pre_start, pre_end = float(pre.start[0]), float(pre.end[0])
nrem_pre = states["Non-REM"]
nrem_pre = nrem_pre[(nrem_pre.end > pre_start) & (nrem_pre.start < pre_end)]
print(f"PRE Non-REM: {np.sum(nrem_pre.end - nrem_pre.start):.0f} s")

edges_ld = np.arange(pre_start, pre_end + chunk_s, chunk_s)
parts = []
for a_, b_ in tqdm(list(zip(edges_ld[:-1], edges_ld[1:])), desc="loading PRE LFP"):
    _, x = load_lfp_channel(best_ch, a_, b_)
    parts.append(x)
lfp_pre = np.concatenate(parts)
t_pre = np.arange(len(lfp_pre)) / LFP_FS + pre_start
env_pre = ndimage.gaussian_filter1d(
    np.abs(signal.hilbert(signal.sosfiltfilt(sos_rip, lfp_pre))), 0.004 * LFP_FS)

nrem_mask_pre = np.zeros(len(lfp_pre), dtype=bool)
for s, e in zip(nrem_pre.start, nrem_pre.end):
    nrem_mask_pre[max(0, int((s - pre_start) * LFP_FS)):
                  min(len(lfp_pre), int((e - pre_start) * LFP_FS))] = True
mu_p, sd_p = env_pre[nrem_mask_pre].mean(), env_pre[nrem_mask_pre].std()
keep_pre = detect_events(env_pre, nrem_mask_pre,
                         mu_p + PEAK_Z * sd_p, mu_p + EDGE_Z * sd_p)
pre_ev_start = np.array([t_pre[s] for s, e in keep_pre])
pre_ev_end = np.array([t_pre[e] for s, e in keep_pre])
print(f"PRE ripples: {len(keep_pre)} "
      f"({len(keep_pre)/(nrem_mask_pre.sum()/LFP_FS)*60:.1f} per min Non-REM)")

pre_counts, _ = collect_qualified(pre_ev_start, pre_ev_end)
counts_pre, mask_pre, t_rel_pre, n_bins_pre = build_padded(pre_counts)
E_pre = len(pre_counts)
print(f"qualified PRE events: {E_pre}")

_, wc_pre = decode_events(counts_pre, mask_pre, t_rel_pre, logf)
wc_pre_null = np.zeros((E_pre, N_SHUFFLE))
for j in tqdm(range(N_SHUFFLE), desc="PRE cell-ID shuffle"):
    _, wc_pre_null[:, j] = decode_events(counts_pre, mask_pre, t_rel_pre,
                                         logf[RNG.permutation(C)])
p_pre = (np.sum(np.abs(wc_pre_null) >= np.abs(wc_pre[:, None]), axis=1) + 1) / (N_SHUFFLE + 1)
sig_pre = p_pre < 0.05

print(f"PRE: median |wc| {np.median(np.abs(wc_pre)):.3f}, "
      f"significant {sig_pre.sum()}/{E_pre} ({100*sig_pre.mean():.1f}%)")
print(f"POST: median |wc| {np.median(np.abs(wc_real)):.3f}, "
      f"significant {sig.sum()}/{E} ({100*sig.mean():.1f}%)")
u = stats.mannwhitneyu(np.abs(wc_real), np.abs(wc_pre), alternative="greater")
chi2 = stats.chi2_contingency([[sig.sum(), (~sig).sum()],
                               [sig_pre.sum(), (~sig_pre).sum()]])
print(f"Mann-Whitney POST > PRE: p={u.pvalue:.2e}; chi-square p={chi2.pvalue:.2e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
bins = np.linspace(0, 1, 50)
axes[0].hist(np.abs(wc_pre), bins=bins, density=True, alpha=0.6,
             color="steelblue", label=f"PRE (n={E_pre})")
axes[0].hist(np.abs(wc_real), bins=bins, density=True, alpha=0.6,
             color="crimson", label=f"POST (n={E})")
axes[0].hist(np.abs(wc_pre_null).ravel(), bins=bins, density=True, alpha=0.35,
             color="gray", label="cell-ID null")
axes[0].set_xlabel("|weighted correlation|"); axes[0].set_ylabel("density")
axes[0].set_title("Sequence strength: PRE vs POST sleep"); axes[0].legend()
axes[1].bar(["PRE", "POST"], [100 * sig_pre.mean(), 100 * sig.mean()],
            color=["steelblue", "crimson"])
axes[1].axhline(5, color="k", ls="--", lw=1, label="chance (5%)")
axes[1].set_ylabel("% significant events (p<0.05)")
axes[1].set_title(f"Replay fraction (chi2 p={chi2.pvalue:.1e})"); axes[1].legend()
for vals_, col, lab in [(np.abs(wc_pre), "steelblue", "PRE"),
                        (np.abs(wc_real), "crimson", "POST")]:
    xs = np.sort(vals_)
    axes[2].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=col, label=lab)
xs = np.sort(np.abs(wc_pre_null).ravel())
xs = xs[:: max(1, len(xs) // 2000)]
axes[2].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color="gray", label="null")
axes[2].set_xlabel("|weighted correlation|"); axes[2].set_ylabel("cumulative fraction")
axes[2].set_title(f"CDF (MW p={u.pvalue:.1e})"); axes[2].legend()
plt.tight_layout()
plt.savefig("fig_pre_vs_post.png", dpi=150)
plt.close()

# %% [markdown]
# ## Summary
#
# - **Ripples**: 3083 sharp-wave ripples were detected in POST Non-REM
#   sleep on LFP channel 117 (0.58 per min; median duration 48 ms). The
#   event-locked average shows the defining SWR anatomy: a slow sharp wave
#   with a ~150 Hz oscillation locked to the envelope peak. Pyramidal
#   cells and interneurons both increase firing ~20-25x (z-scored) around
#   the ripple peak, interneurons slightly earlier.
# - **Place fields**: 88/120 excitatory units have significant spatial
#   tuning on the 1.6 m track (Skaggs SI, circular time-shift shuffle
#   p<0.05, peak >= 1 Hz), and their rate maps tile the track.
# - **Replay**: of 328 POST ripples long and active enough to decode
#   (>= 100 ms, >= 5 place cells), 17.4% show significant sequential
#   structure (cell-identity shuffle p<0.05; 18.6% under the within-event
#   bin-shift shuffle; 15.5% under both), far above the 5% chance level.
#   Median |weighted correlation| is 0.39 vs 0.28 in the null. Significant
#   events split roughly evenly between forward (30) and reverse (27)
#   trajectories, as expected for sleep replay.
# - **Specificity**: PRE-sleep ripples decoded with the same maze template
#   sit at chance (5.6% significant; median |wc| 0.25), and the POST
#   distribution is strongly shifted relative to PRE (Mann-Whitney
#   p = 2.9e-12; chi-square p = 8.1e-10), showing the sequences are
#   experience-dependent.
