# %% [markdown]
# # Sharp-wave ripples and replay in rat hippocampal CA1
#
# This notebook demonstrates **sharp-wave ripple (SWR) events** and **hippocampal
# replay** using a single recording session from the
# [DANDI Archive](https://dandiarchive.org).
#
# **Dataset**: Dandiset
# [000044](https://dandiarchive.org/dandiset/000044) ("Diversity in neural firing
# dynamics supports sparsity and specificity in memory", Buzsaki lab), session
# `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`: 137 CA1 units
# (120 excitatory, 17 inhibitory), 128-channel LFP at 1250 Hz, sleep-state
# labels, and position on a 1.6 m linear maze. The session has three epochs:
# PRE sleep, Maze (~34 min of running), and POST sleep.
#
# **Analysis outline**:
# 1. Stream the NWB file with remfile (no full download).
# 2. Pick the best ripple channel data-driven from POST Non-REM LFP.
# 3. Detect SWRs (100-250 Hz band, Hilbert envelope, threshold crossing).
# 4. Build place fields from maze running (per-direction tuning curves,
#    Skaggs spatial information with circular time-shift shuffles).
# 5. Decode position from place-cell spikes during each SWR (Bayesian
#    decoding, 20 ms bins) and quantify replay with the posterior-mass-weighted
#    correlation between time and position, against cell-ID shuffles.
# 6. Compare POST-sleep replay to PRE-sleep (before the animal ever ran the
#    track) as the control.
#
# Expected runtime: ~10 minutes on a laptop (dominated by LFP filtering and
# the shuffle statistics).

# %% [markdown]
# ## Setup and streaming data access
#
# The file is 8.7 GB, so we stream it with `remfile` plus a local disk cache:
# only the byte ranges we actually read are fetched and cached.

# %%
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO
from scipy import signal, stats as sstats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

os.makedirs("figures", exist_ok=True)
CACHE_DIR = "/tmp/remfile_cache_swr_replay"
os.makedirs(CACHE_DIR, exist_ok=True)

# Resolve the presigned S3 URL through the DANDI API (the /download/ endpoint
# 302-redirects; we take the Location header without following it).
ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"  # sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb
DANDI_URL = f"https://api.dandiarchive.org/api/dandisets/000044/versions/draft/assets/{ASSET_ID}/download/"
r = requests.get(DANDI_URL, allow_redirects=False)
s3_url = r.headers["Location"]

h5py_file = h5py.File(remfile.File(s3_url, disk_cache=remfile.DiskCache(CACHE_DIR)), "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwb = nap.NWBFile(io.read())
print(nwb)

# %% [markdown]
# The file contains spike times (`units`), session `epochs` (PRE/Maze/POST),
# sleep `states` (Awake/Non-REM/REM), 128-channel `LFP`, and 2D + linearized
# position for the maze epoch.

# %%
units = nwb["units"]
epochs = nwb["epochs"]
states = nwb["states"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]

print(epochs)
ct = units.get_info("cell_type")
print(ct.value_counts())

FS = 1250.0
lfp_dset = h5py_file["processing/ecephys/LFP/LFP/data"]
LFP_CONV = lfp_dset.attrs["conversion"]  # V per count
print("LFP:", lfp_dset.shape, lfp_dset.dtype, "conversion", LFP_CONV)

# %% [markdown]
# ## Choosing the ripple channel
#
# Ripple amplitude varies strongly across the 128 LFP channels (the probes
# span CA1 at different depths). We score every channel on a 200 s POST
# Non-REM block by the *peakiness* of its ripple-band envelope (99th
# percentile / median) times its ripple/delta power ratio, and keep the
# winner. This is the same kind of data-driven choice one would make by eye
# from a raw trace.

# %%
post = epochs[epochs.label == "POSTEpoch"]
pre = epochs[epochs.label == "PREEpoch"]
nonrem = states[states.label == "Non-REM"]
post_nonrem = nonrem.intersect(post)

durs = post_nonrem.end - post_nonrem.start
i0 = int(np.argmax(durs))
t0 = float(post_nonrem.start[i0])
t1 = min(float(post_nonrem.end[i0]), t0 + 200.0)
print(f"channel scoring on {t0:.0f}-{t1:.0f} s")

block = lfp_dset[int(t0 * FS):int(t1 * FS), :].astype(np.float64) * LFP_CONV * 1e6  # uV

sos_ripple = signal.butter(4, [100, 250], btype="bandpass", fs=FS, output="sos")
scores = np.zeros(block.shape[1])
for ch in range(block.shape[1]):
    x = block[:, ch]
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos_ripple, x)))
    peakiness = np.percentile(env, 99) / np.median(env)
    f, pxx = signal.welch(x, fs=FS, nperseg=8192)
    scores[ch] = peakiness * pxx[(f >= 100) & (f <= 250)].mean() / pxx[(f >= 1) & (f <= 4)].mean()

CH = int(np.argmax(scores))
print(f"best ripple channel: {CH} (score {scores[CH]:.4f}; runner-up {np.sort(scores)[-2]:.4f})")

# %% [markdown]
# ## Detecting sharp-wave ripples
#
# Standard SWR detection: bandpass the channel at 100-250 Hz, take the
# Hilbert envelope, smooth with a 4 ms Gaussian, and threshold. Event peaks
# must exceed mean+4SD of the Non-REM envelope; event edges are where the
# envelope falls back below mean+1SD. Events closer than 30 ms are merged,
# and we keep events of 30-500 ms that lie fully inside Non-REM sleep.

# %%
print("reading full LFP channel (43.6M samples)...")
lfp = lfp_dset[:, CH].astype(np.float64) * LFP_CONV * 1e6
t_lfp = np.arange(len(lfp)) / FS

filt = signal.sosfiltfilt(sos_ripple, lfp)
env = gaussian_filter1d(np.abs(signal.hilbert(filt)), 0.004 * FS)

nr_mask = np.zeros(len(lfp), dtype=bool)
for s, e in zip(nonrem.start, nonrem.end):
    nr_mask[int(s * FS):int(e * FS)] = True
mu, sd = env[nr_mask].mean(), env[nr_mask].std()
peak_thr, edge_thr = mu + 4 * sd, mu + 1 * sd
print(f"Non-REM envelope: mean {mu:.1f} uV, SD {sd:.1f} uV")

above = env > edge_thr
d = np.diff(above.astype(int))
starts, ends = np.where(d == 1)[0] + 1, np.where(d == -1)[0] + 1
if above[0]:
    starts = np.r_[0, starts]
if above[-1]:
    ends = np.r_[ends, len(above)]

merged = []
for s, e in zip(starts, ends):
    if merged and (s - merged[-1][1]) < 0.030 * FS:
        merged[-1][1] = e
    else:
        merged.append([s, e])

keep = []
for s, e in merged:
    dur = (e - s) / FS
    if 0.030 <= dur <= 0.500 and env[s:e].max() >= peak_thr:
        keep.append((s / FS, e / FS))
rip_t = np.array(keep)

in_nr = np.zeros(len(rip_t), dtype=bool)
for s, e in zip(nonrem.start, nonrem.end):
    in_nr |= (rip_t[:, 0] >= s) & (rip_t[:, 1] <= e)
rip_t = rip_t[in_nr]

ripples = nap.IntervalSet(start=rip_t[:, 0], end=rip_t[:, 1])
rip_pre = ripples.intersect(pre)
rip_post = ripples.intersect(post)

pre_nr_s = float((nonrem.intersect(pre).end - nonrem.intersect(pre).start).sum())
post_nr_s = float((post_nonrem.end - post_nonrem.start).sum())
print(f"SWRs: {len(rip_pre)} PRE ({len(rip_pre) / (pre_nr_s / 60):.1f}/min of Non-REM), "
      f"{len(rip_post)} POST ({len(rip_post) / (post_nr_s / 60):.1f}/min)")
print(f"median duration {np.median(ripples.end - ripples.start) * 1e3:.0f} ms")

# %% [markdown]
# ### Figure 1: SWR detection and event statistics

# %%
# example POST ripple: 80-150 ms long, highest envelope peak
pdur = rip_post.end - rip_post.start
cand = np.where((pdur > 0.08) & (pdur < 0.15))[0]
peaks = [env[int(rip_post.start[i] * FS):int(rip_post.end[i] * FS)].max() for i in cand]
ex = (float(rip_post.start[cand[np.argmax(peaks)]]), float(rip_post.end[cand[np.argmax(peaks)]]))

fig, axes = plt.subplots(3, 2, figsize=(13, 10), gridspec_kw={"width_ratios": [1.6, 1]})

ax = axes[0, 0]
i0, i1 = int((ex[0] - 1.0) * FS), int((ex[1] + 1.0) * FS)
tt = t_lfp[i0:i1] - (ex[0] - 1.0)
ax.plot(tt, lfp[i0:i1], color="0.6", lw=0.4, label="wideband LFP")
ax.plot(tt, filt[i0:i1] * 3 + 1200, color="k", lw=0.6, label="ripple band (100-250 Hz, x3, offset)")
ax.plot(tt, env[i0:i1] * 3 + 1200, color="crimson", lw=1.2, label="envelope (x3, offset)")
ax.axhline(edge_thr * 3 + 1200, color="crimson", ls=":", lw=0.8, label="edge thr (mean+1SD)")
ax.axhline(peak_thr * 3 + 1200, color="crimson", ls="--", lw=0.8, label="peak thr (mean+4SD)")
ax.axvspan(1.0, 1.0 + ex[1] - ex[0], color="crimson", alpha=0.15)
ax.set_xlim(0, tt[-1]); ax.set_xlabel("time (s)"); ax.set_ylabel("LFP (uV)")
ax.set_title(f"A  SWR detection on CA1 LFP (ch {CH})", loc="left")
ax.legend(loc="upper right", fontsize=7.5, frameon=False)

ax = axes[0, 1]
i0, i1 = int((ex[0] - 0.08) * FS), int((ex[1] + 0.08) * FS)
tt = (t_lfp[i0:i1] - ex[0]) * 1e3
ax.plot(tt, lfp[i0:i1], color="0.6", lw=0.5, label="wideband")
ax.plot(tt, filt[i0:i1] * 3 + 900, color="k", lw=0.7, label="ripple band x3")
ax.plot(tt, env[i0:i1] * 3 + 900, color="crimson", lw=1.2, label="envelope x3")
ax.axvspan(0, (ex[1] - ex[0]) * 1e3, color="crimson", alpha=0.15)
ax.set_xlim(tt[0], tt[-1]); ax.set_xlabel("time from ripple start (ms)")
ax.set_title("B  example ripple (zoom)", loc="left")
ax.legend(loc="upper right", fontsize=7.5, frameon=False)

ax = axes[1, 0]
d_all = (ripples.end - ripples.start) * 1e3
ax.hist(d_all, bins=np.arange(30, 300, 5), color="steelblue", edgecolor="none")
ax.set_xlabel("ripple duration (ms)"); ax.set_ylabel("count")
ax.set_title(f"C  duration distribution (n={len(d_all)}, median {np.median(d_all):.0f} ms)", loc="left")

ax = axes[1, 1]
bins = np.arange(0, t_lfp[-1], 300)
cnt, _ = np.histogram(ripples.start, bins=bins)
ax.plot((bins[:-1] + 150) / 3600, cnt / 5, color="k", lw=1)
ax.axvspan(18079.5 / 3600, 20147 / 3600, color="orange", alpha=0.2, label="maze epoch")
ax.set_xlabel("session time (h)"); ax.set_ylabel("SWRs / min")
ax.set_title("D  SWR rate across session (Non-REM only)", loc="left")
ax.legend(frameon=False, fontsize=8)

ax = axes[2, 0]
win = int(0.15 * FS)
rng = np.random.default_rng(0)
sel = rng.choice(len(rip_post), size=min(400, len(rip_post)), replace=False)
trigs = []
for i in sel:
    ic = int((rip_post.start[i] + rip_post.end[i]) / 2 * FS)
    if ic - win >= 0 and ic + win < len(filt):
        trigs.append(filt[ic - win:ic + win])
tt = (np.arange(2 * win) - win) / FS * 1e3
ax.plot(tt, np.array(trigs).mean(axis=0), color="k", lw=1.2)
ax.set_xlabel("time from ripple center (ms)"); ax.set_ylabel("ripple-band LFP (uV)")
ax.set_title(f"E  mean ripple waveform (n={len(trigs)} POST)", loc="left")

ax = axes[2, 1]
iri = np.diff(ripples.start)
ax.hist(iri[iri < 60], bins=100, color="steelblue")
ax.set_yscale("log")
ax.set_xlabel("inter-ripple interval (s)"); ax.set_ylabel("count (log)")
ax.set_title("F  inter-ripple intervals", loc="left")

fig.tight_layout()
fig.savefig("figures/fig1_swr_detection.png", dpi=150)
plt.close(fig)

# %% [markdown]
# The detected events have the hallmarks of SWRs: a large-amplitude ~150 Hz
# oscillation riding on a sharp wave, 30-300 ms durations (median ~60 ms),
# and occurrence almost exclusively within Non-REM sleep at ~30/min.

# %% [markdown]
# ## Place fields on the linear maze
#
# To ask *what* is replayed we need a spatial template. The linearized
# position is NaN except during on-track runs, so we define run bouts as
# contiguous valid stretches (merged over <0.3 s gaps, >=1 s long, >0.3 m
# span, median speed >0.15 m/s) and compute per-direction ratemaps
# (50 bins over 0-1.6 m, 1.5-bin Gaussian smoothing).
#
# A cell is a **place cell** if its Skaggs spatial information is significant
# against circular time-shift shuffles of the spike train along the
# concatenated run-bout time axis (p<0.05) in at least one direction, with
# peak rate >=1 Hz and >=30 spikes. Shifting in time (not space) is the
# correct null here: spatial information is translation-invariant under
# uniform occupancy, so a spatial circular shift would not break the
# spike-position relationship at all.

# %%
N_BINS, TRACK_LEN, N_SHUF_PLACE = 50, 1.6, 200
rng = np.random.default_rng(42)

pos_t, pos_x = lin.t, np.asarray(lin.values).ravel()
valid = ~np.isnan(pos_x)
d = np.diff(valid.astype(int))
starts, ends = np.where(d == 1)[0] + 1, np.where(d == -1)[0] + 1
if valid[0]:
    starts = np.r_[0, starts]
if valid[-1]:
    ends = np.r_[ends, len(valid)]
raw_bouts = []
for s, e in zip(starts, ends):
    if raw_bouts and pos_t[s] - pos_t[raw_bouts[-1][1] - 1] < 0.3:
        raw_bouts[-1][1] = e
    else:
        raw_bouts.append([s, e])

bout_list = []
for s, e in raw_bouts:
    t0b, t1b = pos_t[s], pos_t[e - 1]
    if t1b - t0b < 1.0:
        continue
    x = pos_x[s:e]
    if x.max() - x.min() < 0.3:
        continue
    v = np.diff(x) / np.diff(pos_t[s:e])
    if np.median(np.abs(v)) < 0.15:
        continue
    bout_list.append((t0b, t1b, 1 if np.median(v) > 0 else -1))
print(f"run bouts: {len(bout_list)} "
      f"({sum(b[2] > 0 for b in bout_list)} pos-dir, {sum(b[2] < 0 for b in bout_list)} neg-dir)")

exc_ids = ct[ct == "excitatory"].index.to_numpy()
edges = np.linspace(0, TRACK_LEN, N_BINS + 1)
centers = (edges[:-1] + edges[1:]) / 2
binw = edges[1] - edges[0]


def skaggs_si(ratemap, occ_s):
    p = occ_s / occ_s.sum()
    r = np.nansum(p * ratemap)
    if r <= 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nansum(p * ratemap * np.log2(ratemap / r)) / r


results = {}
for direction, label in [(1, "pos"), (-1, "neg")]:
    bts = [b for b in bout_list if b[2] == direction]
    ep = nap.IntervalSet(start=[b[0] for b in bts], end=[b[1] for b in bts])

    # occupancy and a concatenated run-bout time axis ("tau") for shuffles
    occ = np.zeros(N_BINS)
    pos_in, tau_pos, bout_bounds = [], [], []
    tau = 0.0
    for t0b, t1b, _ in bts:
        m = (pos_t >= t0b) & (pos_t <= t1b)
        x, tp = pos_x[m], pos_t[m]
        ok = ~np.isnan(x)  # merged gaps still contain NaN samples
        pos_in.append(x[ok])
        tau_pos.append(tau + (tp[ok] - t0b))
        bout_bounds.append((tau, t0b, t1b))
        tau += t1b - t0b
        occ += np.bincount(np.clip((x[ok] / binw).astype(int), 0, N_BINS - 1),
                           minlength=N_BINS) / 39.0625
    pos_in, tau_pos = np.concatenate(pos_in), np.concatenate(tau_pos)

    spk_tau = {}
    for uid in exc_ids:
        st = units[uid].restrict(ep).t
        taus = [tau0 + (st[(st >= w0) & (st <= w1)] - w0) for tau0, w0, w1 in bout_bounds]
        spk_tau[uid] = np.sort(np.concatenate(taus)) if any(len(a) for a in taus) else np.array([])

    def ratemap_from_tau(sp_tau):
        if len(sp_tau) == 0:
            return np.full(N_BINS, np.nan), 0
        x = np.interp(sp_tau, tau_pos, pos_in)
        cnt = np.bincount(np.clip((x / binw).astype(int), 0, N_BINS - 1), minlength=N_BINS)
        with np.errstate(divide="ignore", invalid="ignore"):
            rm = cnt / occ
        rm[occ < 0.05] = np.nan
        return rm, len(sp_tau)

    rm_real, si_real, si_null, nspk = {}, {}, {}, {}
    for uid in tqdm(exc_ids, desc=f"place fields {label}-dir"):
        rm, n = ratemap_from_tau(spk_tau[uid])
        rm_s = gaussian_filter1d(np.nan_to_num(rm), 1.5)
        rm_s[np.isnan(rm)] = np.nan
        rm_real[uid] = rm_s
        si_real[uid] = skaggs_si(np.nan_to_num(rm), occ)
        nspk[uid] = n
        if n < 10:
            si_null[uid] = np.full(N_SHUF_PLACE, np.nan)
            continue
        null = np.empty(N_SHUF_PLACE)
        for k, sh in enumerate(rng.uniform(1, tau - 1, N_SHUF_PLACE)):
            xs = np.interp((spk_tau[uid] + sh) % tau, tau_pos, pos_in)
            cnt = np.bincount(np.clip((xs / binw).astype(int), 0, N_BINS - 1), minlength=N_BINS)
            with np.errstate(divide="ignore", invalid="ignore"):
                null[k] = skaggs_si(np.nan_to_num(cnt / occ), occ)
        si_null[uid] = null
    results[label] = dict(rm=rm_real, si=si_real, si_null=si_null, nspk=nspk)

place_cells = []
for uid in exc_ids:
    for label in ["pos", "neg"]:
        r = results[label]
        null = r["si_null"][uid]
        if np.all(np.isnan(null)):
            continue
        p = (np.sum(null >= r["si"][uid]) + 1) / (len(null) + 1)
        if p < 0.05 and np.nanmax(r["rm"][uid]) >= 1.0 and r["nspk"][uid] >= 30:
            place_cells.append(uid)
            break
place_cells = np.array(place_cells)
print(f"place cells: {len(place_cells)} / {len(exc_ids)} excitatory")

# direction-pooled template, cells ordered by peak position
template = {u: np.nanmean(np.stack([results["pos"]["rm"][u], results["neg"]["rm"][u]]), axis=0)
            for u in place_cells}
order = np.array(sorted(place_cells, key=lambda u: centers[np.nanargmax(template[u])]))
TEMPLATE = np.stack([template[u] for u in order])

# %% [markdown]
# ### Figure 2: the place-field template

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1, 1]})

ax = axes[0]
z = (TEMPLATE - np.nanmean(TEMPLATE, axis=1, keepdims=True)) / np.nanstd(TEMPLATE, axis=1, keepdims=True)
im = ax.imshow(z, aspect="auto", cmap="viridis", extent=[centers[0], centers[-1], len(z), 0])
ax.set_xlabel("linearized position (m)"); ax.set_ylabel("place cell (sorted by peak)")
ax.set_title(f"A  place-field template (n={len(z)} cells)", loc="left")
fig.colorbar(im, ax=ax, label="z-scored rate", shrink=0.8)

picks = np.linspace(4, len(order) - 4, 6).astype(int)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(picks)))
uid_to_row = {u: i for i, u in enumerate(order)}
for ax, label, ttl in [(axes[1], "pos", "B  example ratemaps (+dir)"),
                       (axes[2], "neg", "C  same cells (-dir)")]:
    for c, pi in zip(colors, picks):
        ax.plot(centers, results[label]["rm"][order[pi]], color=c, lw=1.2)
    ax.set_xlabel("linearized position (m)"); ax.set_ylabel("firing rate (Hz)")
    ax.set_title(ttl, loc="left")

fig.tight_layout()
fig.savefig("figures/fig2_place_fields.png", dpi=150)
plt.close(fig)

# %% [markdown]
# 88 of 120 excitatory cells are significant place cells, and their fields
# tile the full 1.6 m track: exactly the substrate needed for sequence
# detection during SWRs.

# %% [markdown]
# ## Replay during SWRs: Bayesian decoding
#
# For every SWR we bin place-cell spikes into 20 ms bins and compute the
# Bayesian posterior over track position under a Poisson model with the
# direction-pooled template as ratemaps. Replay strength is the
# **posterior-mass-weighted correlation** between time and position
# (Diba & Buzsaki, 2007): near +1 means the decoded position sweeps forward
# along the track as the event unfolds, near -1 means reverse. Significance
# comes from 500 cell-ID shuffles per event (the template rows are randomly
# reassigned across cells, destroying sequence structure while preserving
# firing rates and spatial coverage). Events need >=5 active place cells and
# >=5 non-empty bins. PRE-sleep SWRs, recorded before the animal ever ran
# this track, are the control.

# %%
BIN, N_SHUF, MIN_CELLS, MIN_BINS = 0.020, 500, 5, 5
rng = np.random.default_rng(7)

R = np.nan_to_num(TEMPLATE).copy()
R[R < 0.01] = 0.01
logR = np.log(R)
rate_sum = R.sum(axis=0)  # invariant under cell permutation

spike_trains = {u: units[u].t for u in order}


def decode_event(t0, t1):
    nbins = int(np.ceil((t1 - t0) / BIN))
    if nbins < MIN_BINS:
        return None
    C = np.zeros((nbins, len(order)))
    for j, uid in enumerate(order):
        st = spike_trains[uid]
        sp = st[(st >= t0) & (st < t1)]
        if len(sp):
            C[:, j] = np.bincount(((sp - t0) / BIN).astype(int), minlength=nbins)
    if (C.sum(axis=0) > 0).sum() < MIN_CELLS or (C.sum(axis=1) > 0).sum() < MIN_BINS:
        return None
    return C


def posterior_from_counts(C, logR_):
    logpost = C @ logR_ - BIN * rate_sum[None, :]
    logpost -= logpost.max(axis=1, keepdims=True)
    post_ = np.exp(logpost)
    return post_ / post_.sum(axis=1, keepdims=True)


def weighted_corr(post_):
    T = post_.shape[0]
    tt, xx = np.arange(T) * BIN, centers
    W = post_
    wsum = W.sum()
    mt = (W * tt[:, None]).sum() / wsum
    mx = (W * xx[None, :]).sum() / wsum
    cov = (W * (tt[:, None] - mt) * (xx[None, :] - mx)).sum() / wsum
    vt = (W * (tt[:, None] - mt) ** 2).sum() / wsum
    vx = (W * (xx[None, :] - mx) ** 2).sum() / wsum
    return 0.0 if vt <= 0 or vx <= 0 else cov / np.sqrt(vt * vx)


def analyze(ev_starts, ev_ends, label):
    rows = []
    for t0, t1 in tqdm(list(zip(ev_starts, ev_ends)), desc=f"decode {label}"):
        C = decode_event(t0, t1)
        if C is None:
            continue
        wc = weighted_corr(posterior_from_counts(C, logR))
        null = np.empty(N_SHUF)
        for k in range(N_SHUF):
            null[k] = weighted_corr(posterior_from_counts(C, logR[rng.permutation(len(order))]))
        p = (np.sum(np.abs(null) >= abs(wc)) + 1) / (N_SHUF + 1)
        rows.append((t0, t1, wc, p, int((C.sum(axis=0) > 0).sum()), C.shape[0]))
    return np.array(rows)


pre_rows = analyze(rip_pre.start, rip_pre.end, "PRE")
post_rows = analyze(rip_post.start, rip_post.end, "POST")

for rows, label in [(pre_rows, "PRE"), (post_rows, "POST")]:
    sig = rows[:, 3] < 0.05
    print(f"{label}: {len(rows)} decodable SWRs, {sig.sum()} significant ({100 * sig.mean():.1f}%), "
          f"forward {(sig & (rows[:, 2] > 0)).sum()}, reverse {(sig & (rows[:, 2] < 0)).sum()}")
print("Mann-Whitney |wc| POST vs PRE: p =",
      sstats.mannwhitneyu(np.abs(post_rows[:, 2]), np.abs(pre_rows[:, 2])).pvalue)

# %% [markdown]
# ### Figure 3: replay examples and population statistics

# %%
def pick_example(rows, sign):
    cand = rows[(rows[:, 3] < 0.01) & (np.sign(rows[:, 2]) == sign)]
    dur = cand[:, 1] - cand[:, 0]
    cand = cand[(dur > 0.06) & (dur < 0.2)]
    return cand[np.argmax(np.abs(cand[:, 2]))]


ex_fwd, ex_rev = pick_example(post_rows, 1), pick_example(post_rows, -1)
peakpos = centers[np.nanargmax(np.nan_to_num(TEMPLATE), axis=1)]

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1], hspace=0.32, wspace=0.3)

for k, (ex, name) in enumerate([(ex_fwd, "forward"), (ex_rev, "reverse")]):
    t0, t1, wc, p = ex[:4]
    C = decode_event(t0, t1)
    post_ = posterior_from_counts(C, logR)
    T = post_.shape[0]
    ax = fig.add_subplot(gs[k, 0])
    ax.imshow(post_.T, aspect="auto", origin="lower", cmap="bone_r",
              extent=[0, T * BIN * 1e3, centers[0], centers[-1]])
    for j, uid in enumerate(order):
        st = spike_trains[uid]
        sp = st[(st >= t0) & (st < t1)]
        if len(sp):
            ax.plot((sp - t0) * 1e3, np.full(len(sp), peakpos[j]), "|",
                    color="crimson", ms=4, mew=0.8)
    ttC = np.repeat((np.arange(T) + 0.5) * BIN * 1e3, len(centers))
    xxC = np.tile(centers, T)
    WW = post_.ravel()
    mt, mx = np.average(ttC, weights=WW), np.average(xxC, weights=WW)
    slope = (np.average((ttC - mt) * (xxC - mx), weights=WW)
             / np.average((ttC - mt) ** 2, weights=WW))
    ax.plot([0, T * BIN * 1e3], [mx - slope * mt, mx + slope * (T * BIN * 1e3 - mt)],
            color="orange", lw=1.5)
    ax.set_xlabel("time from SWR start (ms)"); ax.set_ylabel("position (m)")
    ax.set_title(f"{'A' if k == 0 else 'B'}  example {name} replay (wc={wc:.2f}, p={p:.3f})",
                 loc="left")

ax = fig.add_subplot(gs[0, 1])
bins = np.linspace(0, 1, 41)
ax.hist(np.abs(post_rows[:, 2]), bins=bins, alpha=0.7, density=True,
        color="crimson", label=f"POST (n={len(post_rows)})")
ax.hist(np.abs(pre_rows[:, 2]), bins=bins, alpha=0.7, density=True,
        color="steelblue", label=f"PRE (n={len(pre_rows)})")
ax.set_xlabel("|weighted correlation|"); ax.set_ylabel("density")
ax.set_title("C  replay strength, POST vs PRE", loc="left")
ax.legend(frameon=False, fontsize=9)

ax = fig.add_subplot(gs[0, 2])
fracs = [(pre_rows[:, 3] < 0.05).mean(), (post_rows[:, 3] < 0.05).mean()]
ax.bar([0, 1], [f * 100 for f in fracs], color=["steelblue", "crimson"], width=0.6)
ax.axhline(5, color="k", ls="--", lw=1, label="chance (5%)")
for i, (f, rows) in enumerate(zip(fracs, [pre_rows, post_rows])):
    ax.text(i, f * 100 + 0.4, f"{f * 100:.1f}%\n({(rows[:, 3] < 0.05).sum()})",
            ha="center", fontsize=10)
ax.set_xticks([0, 1], ["PRE sleep", "POST sleep"])
ax.set_ylabel("% SWRs with significant replay")
ax.set_title("D  significant replay events", loc="left")
ax.legend(frameon=False, fontsize=9)
ax.set_ylim(0, max(fracs[1] * 135, 12))

ax = fig.add_subplot(gs[1, 1])
for rows, c, lab in [(post_rows, "crimson", "POST"), (pre_rows, "steelblue", "PRE")]:
    v = np.sort(np.abs(rows[:, 2]))
    ax.plot(v, 1 - np.arange(len(v)) / len(v), color=c, label=lab)
ax.set_xlabel("|weighted correlation|"); ax.set_ylabel("1 - CDF")
ax.set_title("E  cumulative distributions", loc="left")
ax.legend(frameon=False, fontsize=9)

ax = fig.add_subplot(gs[1, 2])
sig_post = post_rows[post_rows[:, 3] < 0.05]
maze_end = float(epochs[epochs.label == "MazeEpoch"].end[0])
bins_t = np.arange(maze_end, t_lfp[-1], 1800)
mids = (bins_t[:-1] + 900 - maze_end) / 60
fwd = [np.sum((sig_post[:, 0] >= b0) & (sig_post[:, 0] < b1) & (sig_post[:, 2] > 0))
       for b0, b1 in zip(bins_t[:-1], bins_t[1:])]
rev = [np.sum((sig_post[:, 0] >= b0) & (sig_post[:, 0] < b1) & (sig_post[:, 2] < 0))
       for b0, b1 in zip(bins_t[:-1], bins_t[1:])]
ax.bar(mids, fwd, width=25, color="darkorange", label="forward")
ax.bar(mids, -np.array(rev), width=25, color="purple", label="reverse")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("time into POST (min)"); ax.set_ylabel("significant SWRs")
ax.set_title("F  replay direction over POST", loc="left")
ax.legend(frameon=False, fontsize=9)

fig.savefig("figures/fig3_replay.png", dpi=150)
plt.close(fig)

# %% [markdown]
# POST-sleep SWRs show replay at more than twice the chance rate while
# PRE-sleep SWRs sit at chance, and the whole |weighted correlation|
# distribution is shifted upward after maze experience (Mann-Whitney
# p ~ 1e-9). Both forward and reverse events occur, as expected for
# rest-box replay of a linear track.

# %% [markdown]
# ## Peri-SWR firing modulation
#
# A final physiological check: both excitatory and inhibitory CA1 populations
# should increase firing sharply around SWR centers, and SWRs with more
# participating place cells should tend to show stronger replay.

# %%
post_centers_t = (rip_post.start + rip_post.end) / 2
WIN, BINW = 0.5, 0.01
pedges = np.arange(-WIN, WIN + BINW, BINW)
pmids = (pedges[:-1] + pedges[1:]) / 2


def peri_hist(uid):
    st = units[uid].t
    counts = np.zeros(len(pedges) - 1)
    for c in post_centers_t:
        lo, hi = np.searchsorted(st, c - WIN), np.searchsorted(st, c + WIN)
        counts += np.histogram(st[lo:hi] - c, bins=pedges)[0]
    return counts / (len(post_centers_t) * BINW)


inh_ids = ct[ct == "inhibitory"].index.to_numpy()
exc_h = np.stack([peri_hist(u) for u in tqdm(exc_ids, desc="peri-SWR exc")])
inh_h = np.stack([peri_hist(u) for u in tqdm(inh_ids, desc="peri-SWR inh")])
place_h = np.stack([peri_hist(u) for u in order])


def zscore(H):
    base = H[:, (pmids >= -0.5) & (pmids <= -0.25)]
    mu_, sd_ = base.mean(axis=1), base.std(axis=1)
    sd_[sd_ == 0] = 1
    return (H - mu_[:, None]) / sd_[:, None]


fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

ax = axes[0]
ax.plot(pmids * 1e3, zscore(place_h).mean(axis=0), color="crimson",
        label=f"place cells (n={len(order)})")
ax.plot(pmids * 1e3, zscore(exc_h).mean(axis=0), color="steelblue",
        label=f"all excitatory (n={len(exc_ids)})")
ax.plot(pmids * 1e3, zscore(inh_h).mean(axis=0), color="k",
        label=f"inhibitory (n={len(inh_ids)})")
ax.axvspan(-30, 30, color="crimson", alpha=0.1)
ax.axvline(0, color="0.5", ls=":", lw=0.8)
ax.set_xlabel("time from SWR center (ms)"); ax.set_ylabel("firing rate (z)")
ax.set_title("A  peri-SWR firing modulation (POST)", loc="left")
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
cand = post_rows[post_rows[:, 4] >= 15]
ex = cand[np.argsort(cand[:, 1] - cand[:, 0])[len(cand) // 2]]
t0, t1 = ex[0], ex[1]
for j, uid in enumerate(order):
    st = spike_trains[uid]
    sp = st[(st >= t0 - 0.15) & (st <= t1 + 0.15)]
    ax.plot((sp - t0) * 1e3, np.full(len(sp), peakpos[j]), "|", color="k", ms=3, mew=0.7)
ax.axvspan(0, (t1 - t0) * 1e3, color="crimson", alpha=0.12)
ax.set_xlabel("time from SWR start (ms)"); ax.set_ylabel("place field peak (m)")
ax.set_title("B  place-cell raster around one SWR", loc="left")
ax.set_ylim(-0.05, 1.65)

ax = axes[2]
part = post_rows[:, 4] / len(order)
sig = post_rows[:, 3] < 0.05
ax.scatter(part[~sig], np.abs(post_rows[~sig, 2]), s=4, color="0.6", alpha=0.4, label="not sig.")
ax.scatter(part[sig], np.abs(post_rows[sig, 2]), s=6, color="crimson", alpha=0.7, label="p<0.05")
ax.set_xlabel("fraction of place cells active"); ax.set_ylabel("|weighted correlation|")
ax.set_title("C  participation vs replay strength (POST)", loc="left")
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig("figures/fig4_periswr.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Summary
#
# - **SWRs detected**: several thousand Non-REM events on a single CA1
#   channel (median duration ~60 ms, ~30/min of Non-REM), with the canonical
#   ~150 Hz oscillation on a sharp wave.
# - **Place template**: 88/120 excitatory cells are shuffle-significant place
#   cells whose fields tile the 1.6 m track.
# - **Replay**: ~11% of decodable POST-sleep SWRs show significant sequential
#   replay (both forward and reverse), versus ~5% (chance) in PRE sleep; the
#   replay-strength distributions differ at p ~ 1e-9 (Mann-Whitney).
# - **Physiology**: CA1 units increase firing ~20-25 z around SWR centers,
#   and higher place-cell participation accompanies stronger replay.
#
# This reproduces the classic Foster & Wilson (2006) / Diba & Buzsaki (2007)
# result: after spatial experience, hippocampal sharp-wave ripples during
# sleep reactivate compressed sequences of place-cell activity that sweep
# along the experienced trajectory.
