# %% [markdown]
# # Hippocampal replay during sharp-wave ripples
#
# **Dataset:** DANDI Archive dandiset
# [000044](https://dandiarchive.org/dandiset/000044) (Grosmark & Buzsaki 2016,
# *"Diversity in neural firing dynamics supports both rigid and learned
# hippocampal sequences"*), session `sub-Achilles_ses-Achilles-10252013`:
# rat CA1 tetrode recording (137 units) with 128-ch LFP at 1250 Hz during a
# PRE sleep epoch, ~35 min of running on a 1.6 m linear track, and a POST
# sleep epoch.
#
# **Question.** During sharp-wave ripple (SWR) events in sleep, does the CA1
# population reactivate spatial trajectories experienced on the track, and is
# this reactivation stronger after the maze experience than before it?
#
# **Approach.**
# 1. Build a spatial template: place fields of excitatory CA1 units from run
#    bouts on the track (Skaggs information with a circular time-shift shuffle).
# 2. Detect SWRs in Non-REM sleep from the ripple-band (100-250 Hz) LFP
#    envelope on the best CA1 channel.
# 3. Bayesian-decode each SWR into a position trajectory (20 ms bins) and
#    score trajectory structure with the posterior-mass-weighted correlation
#    between time and position (Davidson et al. 2009 style), against 500
#    cell-ID shuffles per event.
# 4. Compare decodable SWRs in PRE vs POST sleep.
#
# **Result.** 19.4% of POST-sleep SWRs encode a significant trajectory vs
# 4.6% in PRE sleep (chance is 5% by construction; Fisher p = 3.2e-10), with
# both forward and reverse replays. Sleep after the maze replays the track.
#
# Runs end-to-end in ~4-5 min with a warm remfile cache (first run streams
# ~1 GB of LFP and takes longer). Requires: pynapple, pynwb, remfile, h5py,
# scipy, matplotlib, tqdm.

# %% [markdown]
# ## 1. Setup and streaming access
#
# The NWB file (8.7 GB) is streamed with remfile + a local disk cache, so only
# the chunks we read are fetched. The download URL is resolved through the
# DANDI API (the `/download/` endpoint 403s on HEAD, so we grab the redirect
# `Location` from a GET without following it).

# %%
import numpy as np
import h5py
import remfile
import requests
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.stats import fisher_exact, binomtest, mannwhitneyu
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
RNG = np.random.default_rng(7)

ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"  # sub-Achilles ses-10252013 behavior+ecephys
dl = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
s3_url = requests.get(dl, allow_redirects=False).headers["Location"]
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())
print(nwb)

units = nwb["units"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
states = nwb["states"]
print("cell types:", units.get_info("cell_type").value_counts().to_dict())
print("states:", states["label"].value_counts().to_dict())

# %% [markdown]
# The session has 137 sorted units (120 excitatory, 17 inhibitory), a
# linearized position series sampled at 39.06 Hz during the maze epoch
# (18079.5-20147 s), sleep/wake state labels, and 128-ch LFP at 1250 Hz.

# %%
FS = 1250.0
CONV = 3.815e-7  # LFP int16 -> V
MAZE_END = 20147.0
lfp_ds = h5py_file["processing/ecephys/LFP/LFP/data"]
t_pos = lin.t
x = np.asarray(lin.values, dtype=float).ravel()  # (N,1) -> (N,)
dt = np.median(np.diff(t_pos))
print(f"position: {len(t_pos)} samples @ {1/dt:.2f} Hz, "
      f"valid fraction {np.mean(~np.isnan(x)):.2f}")
print(f"LFP: {lfp_ds.shape[0]} samples x {lfp_ds.shape[1]} ch, "
      f"{lfp_ds.shape[0]/FS/3600:.2f} h")

# %% [markdown]
# ## 2. Run bouts and place fields
#
# The linearized position is NaN except during on-track runs (the animal sits
# at the reward platforms between laps). Run bouts are contiguous valid
# stretches merged over gaps <0.3 s, kept if they last >=1 s, span >0.3 m, and
# have median speed >0.15 m/s. Rate maps use 50 position bins smoothed with a
# 1.5-bin Gaussian. Place cells are excitatory units whose Skaggs spatial
# information exceeds a 500-draw circular time-shift shuffle (shifts applied
# on a concatenated run-bout time axis, so shuffled spikes stay on real
# trajectories), with mean rate >0.1 Hz and peak >1 Hz.

# %%
# ---- run bouts ----
valid = ~np.isnan(x)
d = np.diff(valid.astype(int))
starts = list(np.where(d == 1)[0] + 1)
ends = list(np.where(d == -1)[0] + 1)
if valid[0]:
    starts = [0] + starts
if valid[-1]:
    ends = ends + [len(valid)]
bouts = [[t_pos[s], t_pos[e - 1]] for s, e in zip(starts, ends)]
merged = [bouts[0]]
for b in bouts[1:]:
    if b[0] - merged[-1][1] < 0.3:
        merged[-1][1] = b[1]
    else:
        merged.append(list(b))
run_bouts = []
for s, e in merged:
    i0, i1 = np.searchsorted(t_pos, s), np.searchsorted(t_pos, e)
    seg = x[i0:i1 + 1]
    seg = seg[~np.isnan(seg)]
    if len(seg) < 3:
        continue
    if (e - s) >= 1.0 and (seg.max() - seg.min()) > 0.3 \
            and np.median(np.abs(np.diff(seg))) / dt > 0.15:
        run_bouts.append((s, e))
print(f"run bouts: {len(run_bouts)}, total {sum(e - s for s, e in run_bouts):.1f} s")

# ---- valid position samples on a concatenated bout time axis ----
samp_t, samp_x, samp_tau = [], [], []
tau = 0.0
for s, e in run_bouts:
    i0, i1 = np.searchsorted(t_pos, s), np.searchsorted(t_pos, e)
    tt, xx = t_pos[i0:i1 + 1], x[i0:i1 + 1]
    m = ~np.isnan(xx)  # gap samples inside merged bouts are still NaN -> drop
    samp_t.append(tt[m])
    samp_x.append(xx[m])
    samp_tau.append(tau + (tt[m] - tt[m][0]))
    tau += tt[-1] - tt[0]
samp_t = np.concatenate(samp_t)
samp_x = np.concatenate(samp_x)
samp_tau = np.concatenate(samp_tau)
total_tau = tau

# ---- rate maps + Skaggs SI with circular time-shift shuffle ----
NBINS, TRACK_LEN, NSHUF = 50, 1.6, 500
edges = np.linspace(0, TRACK_LEN, NBINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
occ_t = np.histogram(samp_x, bins=edges)[0] * dt

bout_starts = np.array([s for s, e in run_bouts])
bout_ends = np.array([e for s, e in run_bouts])
bout_tau0 = np.concatenate([[0], np.cumsum(bout_ends - bout_starts)[:-1]])


def wall_to_tau(spk_t):
    bi = np.searchsorted(bout_ends, spk_t)
    ok = (bi < len(run_bouts)) & (spk_t >= bout_starts[np.clip(bi, 0, len(run_bouts) - 1)])
    tau_out = np.full(len(spk_t), np.nan)
    tau_out[ok] = bout_tau0[bi[ok]] + (spk_t[ok] - bout_starts[bi[ok]])
    return tau_out, ok


def nearest_positions(query_tau):
    idx = np.clip(np.searchsorted(samp_tau, query_tau), 0, len(samp_tau) - 1)
    left = np.clip(idx - 1, 0, len(samp_tau) - 1)
    idx = np.where(np.abs(samp_tau[left] - query_tau) < np.abs(samp_tau[idx] - query_tau),
                   left, idx)
    near = np.abs(samp_tau[idx] - query_tau) < 0.05
    return samp_x[idx], near


def rate_map(px):
    cnt = np.histogram(px, bins=edges)[0]
    with np.errstate(all="ignore"):
        rm = cnt / np.maximum(occ_t, 1e-9)
    rm[occ_t < 0.05] = 0.0
    return gaussian_filter1d(rm, 1.5)


def skaggs_si(rm):
    p = occ_t / occ_t.sum()
    R = np.sum(p * rm)
    if R <= 0:
        return 0.0, R
    m = rm > 0  # 0 * log2(0) -> NaN; zero-rate bins contribute nothing
    return max(np.sum(p[m] * (rm[m] / R) * np.log2(rm[m] / R)), 0.0), R


cell_type = units.get_info("cell_type")
exc_keys = [k for k in units.keys() if cell_type[k] == "excitatory"]
results = {}
for k in tqdm(exc_keys, desc="rate maps + SI shuffle"):
    spk = units[k].t
    spk = spk[(spk >= bout_starts[0]) & (spk <= bout_ends[-1])]
    tau_spk, ok = wall_to_tau(spk)
    tau_spk, spk = tau_spk[ok], spk[ok]
    px, near = nearest_positions(tau_spk)
    rm = rate_map(px[near])
    si, mean_rate = skaggs_si(rm)
    si_null = np.empty(NSHUF)
    for i, shift in enumerate(RNG.uniform(0, total_tau, NSHUF)):
        sx, snear = nearest_positions((tau_spk + shift) % total_tau)
        si_null[i], _ = skaggs_si(rate_map(sx[snear]))
    results[k] = dict(rate_map=rm, si=si, si_null=si_null,
                      p=(np.sum(si_null >= si) + 1) / (NSHUF + 1),
                      mean_rate=mean_rate, peak=rm.max())

place_keys = [k for k, v in results.items()
              if v["p"] < 0.05 and v["mean_rate"] > 0.1 and v["peak"] > 1.0]
print(f"place cells: {len(place_keys)} / {len(exc_keys)} excitatory, "
      f"median SI {np.median([results[k]['si'] for k in place_keys]):.2f} bits/spike")

# %% [markdown]
# 87 of 120 excitatory units are significant place cells; their fields tile
# the track (fig 1-2).

# %%
# ---- fig 1: raw data overview ----
peak_pos = {k: centers[np.argmax(results[k]["rate_map"])] for k in place_keys}
sorted_keys = sorted(place_keys, key=lambda k: peak_pos[k])
s0, e0 = run_bouts[len(run_bouts) // 2]

fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), height_ratios=[1.2, 1, 1])
ax = axes[0]
ax.plot(t_pos - t_pos[0], x, lw=0.3, color="0.6", rasterized=True)
for s, e in run_bouts:
    ax.axvspan(s - t_pos[0], e - t_pos[0], color="tab:red", alpha=0.15, lw=0)
ax.set_xlim(0, t_pos[-1] - t_pos[0])
ax.set_xlabel("time in maze epoch (s)")
ax.set_ylabel("linearized position (m)")
ax.set_title("Linearized position on the 1.6 m track (red = run bouts used for place fields)")

ax = axes[1]
for i, k in enumerate(sorted_keys):
    spk = units[k].t
    spk = spk[(spk >= s0) & (spk <= e0)]
    ax.plot(spk - s0, np.full_like(spk, i), "|", ms=2, color="k", rasterized=True)
ax.set_xlim(0, e0 - s0)
ax.set_xlabel("time in example run bout (s)")
ax.set_ylabel("place cell (sorted by field)")
ax.set_title(f"Spike raster of {len(place_keys)} place cells during one example run bout")

ax = axes[2]
i0, i1 = np.searchsorted(t_pos, s0), np.searchsorted(t_pos, e0)
ax.plot(t_pos[i0:i1] - s0, x[i0:i1], color="0.6", lw=1)
for k in sorted_keys:
    spk = units[k].t
    spk = spk[(spk >= s0) & (spk <= e0)]
    if len(spk):
        idx = np.clip(np.searchsorted(t_pos, spk), 0, len(x) - 1)
        ax.plot(spk - s0, x[idx], ".", ms=1.5, rasterized=True)
ax.set_xlim(0, e0 - s0)
ax.set_xlabel("time in example run bout (s)")
ax.set_ylabel("position (m)")
ax.set_title("Trajectory with spikes (all place cells)")
fig.tight_layout()
fig.savefig("fig1_raw_data_overview.png", dpi=150)
plt.close(fig)

# ---- fig 2: place fields ----
fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.35, wspace=0.25)
ax = fig.add_subplot(gs[0, :])
maps = np.array([results[k]["rate_map"] for k in sorted_keys])
ax.imshow(maps / maps.max(axis=1, keepdims=True), aspect="auto", cmap="viridis",
          extent=[0, 1.6, 0, len(sorted_keys)], origin="lower")
ax.set_xlabel("position on track (m)")
ax.set_ylabel("place cell (sorted by peak)")
ax.set_title(f"Place fields of {len(place_keys)} place cells (direction-pooled, normalized)")

ax = fig.add_subplot(gs[1, 0])
ax.hist([results[k]["si"] for k in results], bins=30, alpha=0.6,
        label="all excitatory", color="0.5")
ax.hist([results[k]["si"] for k in place_keys], bins=30, alpha=0.8,
        label="place cells", color="tab:blue")
ax.set_xlabel("Skaggs spatial information (bits/spike)")
ax.set_ylabel("count")
ax.legend(frameon=False)
ax.set_title("Spatial information")

ax = fig.add_subplot(gs[1, 1])
ex = sorted_keys[len(sorted_keys) // 4::len(sorted_keys) // 3][:3]
for color, k in zip(["tab:blue", "tab:orange", "tab:green"], ex):
    ax.hist(results[k]["si_null"], bins=40, histtype="step", density=True,
            label=f"cell {k} null", color=color)
    ax.axvline(results[k]["si"], color=color, ls="--", lw=1)
ax.set_xlabel("Skaggs SI (bits/spike)")
ax.set_ylabel("density")
ax.set_title("Example SI shuffle nulls (dashed = observed)")
ax.legend(frameon=False, fontsize=7)
fig.savefig("fig2_place_fields.png", dpi=150)
plt.close(fig)
print("fig1, fig2 saved")

# %% [markdown]
# ## 3. Sharp-wave ripple detection
#
# The best ripple channel is chosen data-driven: on a 200 s POST Non-REM
# block, each channel gets a score = envelope peakiness (p99/median of the
# smoothed 100-250 Hz envelope) times the ripple/delta PSD ratio. Channel 117
# wins, consistent with a CA1 pyramidal-layer site. SWRs are then detected on
# the full-session envelope of that channel: bandpass 100-250 Hz (4th-order
# Butterworth, SOS form for numerical stability), Hilbert envelope, 4 ms
# Gaussian smoothing. Thresholds come from Non-REM samples only: candidate
# edges where the envelope exceeds mean+1 SD, events kept when the peak
# exceeds mean+4 SD; events <30 ms apart are merged and those lasting
# 30-500 ms fully inside a Non-REM block are kept.

# %%
nonrem = states[states.label == "Non-REM"]
nr_start = np.asarray(nonrem.start)
nr_end = np.asarray(nonrem.end)
post_mask = nr_start > MAZE_END
print(f"Non-REM: {len(nr_start)} blocks, "
      f"PRE {np.sum(nr_end[~post_mask] - nr_start[~post_mask]):.0f} s, "
      f"POST {np.sum(nr_end[post_mask] - nr_start[post_mask]):.0f} s")

sos = signal.butter(4, [100, 250], btype="bandpass", fs=FS, output="sos")
win = signal.windows.gaussian(int(FS * 0.008), int(FS * 0.004))


def ripple_env(trace):
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, trace)))
    return np.convolve(env, win / win.sum(), mode="same")


# ---- channel selection on a POST Non-REM block ----
blk_s = nr_start[post_mask][0]
blk = (int(blk_s * FS), int((blk_s + 200) * FS))
scores = np.zeros(128)
for ch in tqdm(range(128), desc="channel scan"):
    tr = lfp_ds[blk[0]:blk[1], ch].astype(float) * CONV
    env_s = ripple_env(tr)
    f, psd = signal.welch(tr, FS, nperseg=int(FS * 4))
    rd = psd[(f >= 100) & (f <= 250)].mean() / psd[(f >= 1) & (f <= 4)].mean()
    scores[ch] = np.percentile(env_s, 99) / np.median(env_s) * rd
best_ch = int(np.argmax(scores))
print(f"best ripple channel: {best_ch}")

# ---- full-session envelope on the best channel ----
N_SAMP = lfp_ds.shape[0]
full = np.empty(N_SAMP, dtype=np.float32)
CHUNK = int(FS * 600)
for i0 in tqdm(range(0, N_SAMP, CHUNK), desc="LFP stream"):
    i1 = min(i0 + CHUNK, N_SAMP)
    full[i0:i1] = lfp_ds[i0:i1, best_ch].astype(np.float32) * CONV
env_s = ripple_env(full.astype(float)).astype(np.float32)
del full

# ---- thresholds from Non-REM samples, event detection ----
nr_mask = np.zeros(N_SAMP, dtype=bool)
for s, e in zip(nr_start, nr_end):
    nr_mask[int(s * FS):int(e * FS)] = True
mu, sd = env_s[nr_mask].mean(), env_s[nr_mask].std()
thr_peak, thr_edge = mu + 4 * sd, mu + 1 * sd
print(f"envelope {mu * 1e6:.1f}+-{sd * 1e6:.1f} uV; "
      f"peak thr {thr_peak * 1e6:.1f} uV, edge thr {thr_edge * 1e6:.1f} uV")

above = env_s > thr_edge
d = np.diff(above.astype(int))
ev_s = list(np.where(d == 1)[0] + 1)
ev_e = list(np.where(d == -1)[0] + 1)
if above[0]:
    ev_s = [0] + ev_s
if above[-1]:
    ev_e = ev_e + [N_SAMP]
events = [[s, e] for s, e in zip(ev_s, ev_e) if env_s[s:e].max() > thr_peak]
merged = [events[0]]
for ev in events[1:]:
    if ev[0] - merged[-1][1] < int(0.030 * FS):
        merged[-1][1] = ev[1]
    else:
        merged.append(list(ev))
kept = []
for s, e in merged:
    dur = (e - s) / FS
    if 0.030 <= dur <= 0.500:
        bi = np.searchsorted(nr_end, s / FS)
        if bi < len(nr_end) and s / FS >= nr_start[bi] and e / FS <= nr_end[bi]:
            kept.append((s / FS, e / FS))
events = np.array(kept)
pre_ev = events[events[:, 0] < MAZE_END]
post_ev = events[events[:, 0] > MAZE_END]
print(f"SWRs: {len(pre_ev)} PRE "
      f"({len(pre_ev) / (np.sum(nr_end[~post_mask] - nr_start[~post_mask]) / 60):.1f}/min), "
      f"{len(post_ev)} POST "
      f"({len(post_ev) / (np.sum(nr_end[post_mask] - nr_start[post_mask]) / 60):.1f}/min), "
      f"median {np.median(events[:, 1] - events[:, 0]) * 1000:.0f} ms")

# %% [markdown]
# ~8300 SWRs at ~31/min of Non-REM in both sleep epochs, median duration
# ~48 ms. The peri-SWR raster (fig 3, bottom) shows the expected strong
# co-activation of place cells around ripple centers.

# %%
# ---- fig 3: ripple detection ----
durs = post_ev[:, 1] - post_ev[:, 0]
ex = post_ev[np.argsort(np.abs(durs - np.median(durs)))[0]]
c = (ex[0] + ex[1]) / 2
w = 0.35
i0, i1 = int((c - w) * FS), int((c + w) * FS)
trace = lfp_ds[i0:i1, best_ch].astype(float) * CONV * 1e6
tt = np.arange(i0, i1) / FS - c
filt = signal.sosfiltfilt(sos, trace)
env_ex = np.convolve(np.abs(signal.hilbert(filt)), win / win.sum(), mode="same")

fig = plt.figure(figsize=(10, 7.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.45, wspace=0.3)
ax = fig.add_subplot(gs[0, 0])
ax.plot(np.arange(128), scores, ".", ms=3, color="0.5")
ax.plot(best_ch, scores[best_ch], "o", color="tab:red",
        label=f"ch {best_ch} (selected)")
ax.set_xlabel("LFP channel")
ax.set_ylabel("peakiness x ripple/delta")
ax.legend(frameon=False)
ax.set_title("Ripple-channel selection metric")

ax = fig.add_subplot(gs[0, 1])
ax.hist((events[:, 1] - events[:, 0]) * 1000, bins=np.arange(20, 300, 5),
        color="0.4")
ax.set_xlabel("SWR duration (ms)")
ax.set_ylabel("count")
ax.set_title(f"SWR duration distribution (n={len(events)})")

ax = fig.add_subplot(gs[1, :])
ax.plot(tt * 1000, trace, color="0.6", lw=0.5, label="wideband LFP")
ax.plot(tt * 1000, filt + trace.mean(), color="tab:blue", lw=0.7,
        label="ripple band (100-250 Hz)")
ax.plot(tt * 1000, env_ex + trace.mean(), color="tab:red", lw=1.2, label="envelope")
ax.axhline(thr_peak * 1e6 + trace.mean(), color="tab:red", ls="--", lw=0.8)
ax.axhline(thr_edge * 1e6 + trace.mean(), color="tab:red", ls=":", lw=0.8)
ax.axvspan((ex[0] - c) * 1000, (ex[1] - c) * 1000, color="tab:red", alpha=0.15, lw=0)
ax.set_xlabel("time from SWR center (ms)")
ax.set_ylabel("LFP (uV)")
ax.legend(frameon=False, ncol=3, fontsize=8)
ax.set_title(f"Example SWR on ch {best_ch} (dashed = peak threshold, dotted = edge threshold)")

ax = fig.add_subplot(gs[2, :])
for i, ev in enumerate(post_ev[:200]):
    ec = (ev[0] + ev[1]) / 2
    for k in place_keys:
        spk = units[k].t
        j0, j1 = np.searchsorted(spk, ec - 0.15), np.searchsorted(spk, ec + 0.15)
        ax.plot((spk[j0:j1] - ec) * 1000, np.full(j1 - j0, i), "|", ms=1.5,
                color="k", alpha=0.5, rasterized=True)
ax.set_xlim(-150, 150)
ax.set_xlabel("time from SWR center (ms)")
ax.set_ylabel("SWR event")
ax.set_title("Peri-SWR spikes of place cells (first 200 POST events)")
fig.savefig("fig3_ripple_detection.png", dpi=150)
plt.close(fig)
print("fig3 saved")

# %% [markdown]
# ## 4. Bayesian decoding of SWRs and replay statistics
#
# Each SWR is binned at 20 ms and decoded with a Poisson Bayesian decoder over
# the direction-pooled place-field template:
# `P(x|n) ∝ exp(N @ log F - dt * sum_x F)`. Events need >=5 spiked bins and
# >=5 active place cells. Trajectory structure is scored with the
# posterior-mass-weighted correlation between time-bin index and position;
# the sign reads forward (+) vs reverse (-). Each event is compared against
# 500 cell-ID shuffles (rate maps permuted across cells), which destroys
# place-field order while keeping spike counts and event structure.

# %%
BIN = 0.020
MIN_BINS, MIN_CELLS = 5, 5
F = np.array([results[k]["rate_map"] for k in place_keys])
C, X = F.shape
logF = np.log(F + 1e-3)
Fsum = F.sum(axis=0)
spikes = {k: units[k].t for k in place_keys}
perms = RNG.permuted(np.tile(np.arange(C), (NSHUF, 1)), axis=1)


def weighted_corr(post):
    T, Xx = post.shape[-2], post.shape[-1]
    tv = np.arange(T, dtype=float)
    xv = centers[:Xx]
    W = post.sum(axis=(-2, -1), keepdims=True)
    mt = (post * tv[None, :, None]).sum(axis=(-2, -1), keepdims=True) / W
    mx = (post * xv[None, None, :]).sum(axis=(-2, -1), keepdims=True) / W
    dtv = tv[None, :, None] - mt
    dxv = xv[None, None, :] - mx
    cov = (post * dtv * dxv).sum(axis=(-2, -1)) / W.squeeze()
    vt = (post * dtv ** 2).sum(axis=(-2, -1)) / W.squeeze()
    vx = (post * dxv ** 2).sum(axis=(-2, -1)) / W.squeeze()
    with np.errstate(all="ignore"):
        return cov / np.sqrt(vt * vx)


def decode_event(s, e):
    edges_ = np.arange(s, e + 1e-9, BIN)
    T = len(edges_) - 1
    if T < MIN_BINS:
        return None
    N = np.zeros((T, C))
    for ci, k in enumerate(place_keys):
        spk = spikes[k]
        j0, j1 = np.searchsorted(spk, s), np.searchsorted(spk, e)
        if j1 > j0:
            N[:, ci] = np.histogram(spk[j0:j1], bins=edges_)[0]
    if (N.sum(1) > 0).sum() < MIN_BINS or (N.sum(0) > 0).sum() < MIN_CELLS:
        return None
    # np.errstate: macOS Accelerate BLAS emits spurious matmul overflow/invalid
    # RuntimeWarnings on these shapes; outputs verified NaN-free.
    with np.errstate(all="ignore"):
        lp = N @ logF - BIN * Fsum[None, :]
    lp -= lp.max(axis=1, keepdims=True)
    post = np.exp(lp)
    post /= post.sum(axis=1, keepdims=True)
    score = float(weighted_corr(post)[0])
    # cell-ID shuffles; gotcha: N[:, perms] -> (T, NSHUF, C), so transpose
    Np = N[:, perms].transpose(1, 0, 2)
    with np.errstate(all="ignore"):
        lps = Np @ logF - BIN * Fsum[None, None, :]
    lps -= lps.max(axis=2, keepdims=True)
    ps = np.exp(lps)
    ps /= ps.sum(axis=2, keepdims=True)
    null = weighted_corr(ps)
    null = null[np.isfinite(null)]
    pval = (np.sum(np.abs(null) >= abs(score)) + 1) / (len(null) + 1)
    return score, pval, post


decoded = []
for s, e in tqdm(events, desc="decoding SWRs"):
    r = decode_event(s, e)
    if r is None:
        continue
    score, pval, post = r
    decoded.append(dict(start=s, end=e, epoch="PRE" if s < MAZE_END else "POST",
                        score=score, pval=pval, post_prob=post))

pre = [o for o in decoded if o["epoch"] == "PRE"]
post = [o for o in decoded if o["epoch"] == "POST"]
sig_pre = [o for o in pre if o["pval"] < 0.05]
sig_post = [o for o in post if o["pval"] < 0.05]
p_fisher = fisher_exact([[len(sig_post), len(post) - len(sig_post)],
                         [len(sig_pre), len(pre) - len(sig_pre)]])[1]
n_fwd = sum(1 for o in sig_post if o["score"] > 0)
n_rev = sum(1 for o in sig_post if o["score"] < 0)
p_binom = binomtest(n_fwd, n_fwd + n_rev, 0.5).pvalue
p_mw = mannwhitneyu(np.abs([o["score"] for o in post]),
                    np.abs([o["score"] for o in pre]))[1]
print(f"\ndecodable: {len(pre)} PRE, {len(post)} POST")
print(f"PRE:  {len(sig_pre)}/{len(pre)} significant "
      f"({100 * len(sig_pre) / len(pre):.1f}%)")
print(f"POST: {len(sig_post)}/{len(post)} significant "
      f"({100 * len(sig_post) / len(post):.1f}%), {n_fwd} fwd / {n_rev} rev")
print(f"Fisher POST vs PRE p={p_fisher:.2e}; binomial fwd/rev p={p_binom:.2f}; "
      f"MW |score| p={p_mw:.2e}")

# %% [markdown]
# ## 5. Results
#
# POST-sleep SWRs decode to significant spatial trajectories far above chance
# (19.4% vs the 5% false-positive rate and 4.6% in PRE sleep; Fisher
# p = 3.2e-10), and their weighted-correlation magnitudes are larger than PRE
# (Mann-Whitney p = 1.9e-10). Both forward and reverse trajectories occur
# (23 vs 18 of the significant POST events; not significantly biased in this
# session). Fig 4 shows examples: the decoded position sweeps across a large
# fraction of the 1.6 m track within a ~100 ms ripple, i.e. at ~20x real-time
# speed, the signature of hippocampal replay.

# %%
# ---- fig 4: replay ----
sig_post_sorted = sorted(sig_post, key=lambda o: -abs(o["score"]))
fwd_ex = [o for o in sig_post_sorted if o["score"] > 0][:3]
rev_ex = [o for o in sig_post_sorted if o["score"] < 0][:3]

fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.1, 1], hspace=0.5, wspace=0.35)
for row, grp, name in [(0, fwd_ex, "forward"), (1, rev_ex, "reverse")]:
    for col, o in enumerate(grp):
        ax = fig.add_subplot(gs[row, col])
        P = o["post_prob"]
        T = P.shape[0]
        ax.imshow(P.T, aspect="auto", origin="lower", cmap="hot",
                  extent=[0, T * 20, 0, 1.6])
        wmean = (P * centers[None, :]).sum(1) / P.sum(1)
        ax.plot(np.arange(T) * 20 + 10, wmean, color="cyan", lw=1.2)
        ax.set_title(f"{name}, score={o['score']:.2f}, p={o['pval']:.3f}", fontsize=8)
        if col == 0:
            ax.set_ylabel("decoded position (m)")
        ax.set_xlabel("time in SWR (ms)")

ax = fig.add_subplot(gs[2, 0])
bins = np.linspace(0, 1, 31)
ax.hist(np.abs([o["score"] for o in pre]), bins=bins, alpha=0.7, density=True,
        label=f"PRE (n={len(pre)})", color="0.5")
ax.hist(np.abs([o["score"] for o in post]), bins=bins, alpha=0.7, density=True,
        label=f"POST (n={len(post)})", color="tab:red")
ax.set_xlabel("|weighted correlation|")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"|score| distributions (MW p={p_mw:.1e})")

ax = fig.add_subplot(gs[2, 1])
fracs = [100 * len(sig_pre) / len(pre), 100 * len(sig_post) / len(post)]
bars = ax.bar(["PRE", "POST"], fracs, color=["0.5", "tab:red"], width=0.55)
ax.axhline(5, color="k", ls="--", lw=0.8, label="chance (5%)")
for b, f, n_s, n in zip(bars, fracs, [len(sig_pre), len(sig_post)], [len(pre), len(post)]):
    ax.text(b.get_x() + b.get_width() / 2, f + 0.4, f"{n_s}/{n}", ha="center", fontsize=8)
ax.set_ylabel("% events significant (p<0.05)")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"Replay above chance after maze (Fisher p={p_fisher:.1e})", fontsize=8)
ax.set_ylim(0, max(fracs) * 1.25)

ax = fig.add_subplot(gs[2, 2])
ax.bar(["forward", "reverse"], [n_fwd, n_rev],
       color=["tab:blue", "tab:orange"], width=0.55)
ax.set_ylabel("significant POST events")
ax.set_title(f"Trajectory direction (binomial p={p_binom:.2f})", fontsize=8)
fig.savefig("fig4_replay.png", dpi=150)
plt.close(fig)
print("fig4 saved")
print("\nDone.")
