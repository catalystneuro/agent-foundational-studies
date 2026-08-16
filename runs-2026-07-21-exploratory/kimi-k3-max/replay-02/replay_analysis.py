# %% [markdown]
# # Hippocampal replay: decoding spatial trajectories during sharp-wave ripples
#
# This script demonstrates hippocampal replay using a single classic recording
# session from the DANDI Archive: `sub-Achilles_ses-Achilles-10252013` from
# dandiset **000044** (Buzsáki lab, "Diversity in neural firing dynamics...").
# A rat ran ~42 laps on a 1.6 m linear track (MazeEpoch) between a PRE and a
# POST sleep epoch, while 137 CA1 units and 128 channels of LFP were recorded.
#
# The analysis has four stages:
#
# 1. **Place-field template.** Run bouts on the track are extracted from the
#    linearized position signal, and firing-rate maps are built for all
#    excitatory units. Place cells are selected with a Skaggs spatial
#    information criterion against circular time-shift shuffles.
# 2. **Sharp-wave ripple (SWR) detection.** The best ripple channel is chosen
#    data-driven from the LFP (it lands on channel 117, the CA1 pyramidal
#    layer). SWRs are detected from the 100-250 Hz envelope during Non-REM
#    sleep in the PRE and POST epochs.
# 3. **Bayesian decoding.** Each SWR is binned into 20 ms windows and the
#    population spike counts are decoded against the place-field template with
#    a Poisson Bayesian decoder, giving a position probability distribution at
#    each moment of the ripple.
# 4. **Replay scoring.** Each event is scored with the posterior-mass-weighted
#    correlation between time and position. Significance is assessed against
#    500 cell-ID shuffles per event. SWRs during POST sleep (after track
#    experience) are expected to replay trajectories above chance; PRE sleep
#    (before the animal ever ran on the track) is the control.
#
# Data is streamed from the archive with remfile + a local disk cache; nothing
# is downloaded wholesale. Runtime is roughly 5-10 minutes.

# %% [markdown]
# ## Setup and streaming access

# %%
import os
import requests
import h5py
import remfile
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal, stats
from tqdm import tqdm

os.makedirs("figures", exist_ok=True)

DANDI_SET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
FS_LFP = 1250.0
FS_POS = 1 / 0.0256  # 39.0625 Hz

# Resolve the asset download URL. The /download/ endpoint redirects to a
# GET-only presigned S3 URL, so grab the Location header without following.
api_url = f"https://api.dandiarchive.org/api/dandisets/{DANDI_SET}/versions/draft/assets/"
r = requests.get(api_url, params={"path": ASSET_PATH})
asset_id = r.json()["results"][0]["asset_id"]
r = requests.get(
    f"https://api.dandiarchive.org/api/dandisets/{DANDI_SET}/versions/draft/assets/{asset_id}/download/",
    allow_redirects=False, stream=True)
s3_url = r.headers["Location"]
r.close()

disk_cache = remfile.DiskCache("/tmp/remfile_cache_replay02")
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())
print(nwb)

# %% [markdown]
# ## Data streams
#
# The NWB file contains `units` (137 sorted units with `cell_type` metadata),
# the 2D and linearized position during the maze epoch, a 128-channel LFP at
# 1250 Hz, an epoch table (PRE / Maze / POST), and a sleep-state table
# (Awake / Non-REM / REM).

# %%
units = nwb["units"]
unit_ids = np.array(list(units.keys()))
cell_type = np.array(units.get_info("cell_type").values, dtype=str)
spikes = {k: np.asarray(units[k].t) for k in unit_ids}
print(f"{len(unit_ids)} units: {(cell_type == 'excitatory').sum()} excitatory, "
      f"{(cell_type == 'inhibitory').sum()} inhibitory")

pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin_tsd = nwb["1.6mLinearMazeLinearizedTimeSeries"]
pos_t = pos2d.t
pos_xy = np.asarray(pos2d.values)
lin = np.asarray(lin_tsd.values).ravel()

epochs = nwb["epochs"]
epoch_start = np.asarray(epochs.start)
epoch_end = np.asarray(epochs.end)
epoch_label = np.array(epochs["label"], dtype=str)
states = nwb["states"]
state_start = np.asarray(states.start)
state_end = np.asarray(states.end)
state_label = np.array(states["label"], dtype=str)
for s0, e0, lab in zip(epoch_start, epoch_end, epoch_label):
    print(f"{lab}: {s0:.1f}-{e0:.1f} s")

# %% [markdown]
# ## Run bouts and the place-field template
#
# The linearized position is NaN except during fast on-track runs. Run bouts
# are contiguous valid stretches, merged over gaps shorter than 0.3 s, at
# least 1 s long, spanning more than 0.3 m, with median speed above 0.15 m/s.
# Rate maps use 50 bins over the 1.6 m track. Place cells are excitatory units
# with in-run mean rate >= 0.1 Hz, peak >= 1 Hz, and Skaggs spatial
# information significant at p < 0.05 against 500 circular time shifts of the
# spike train on the concatenated run-bout time axis (shifting over wall-clock
# time would pile shuffled spikes onto the reward platforms and inflate the
# null).

# %%
TRACK_LEN, N_BINS = 1.6, 50
DT_POS = 1 / FS_POS

valid = ~np.isnan(lin)
d = np.diff(valid.astype(int))
starts, ends = np.where(d == 1)[0] + 1, np.where(d == -1)[0]
if valid[0]:
    starts = np.r_[0, starts]
if valid[-1]:
    ends = np.r_[ends, len(valid) - 1]
merge = (pos_t[starts[1:]] - pos_t[ends[:-1]]) < 0.3
bouts, i = [], 0
while i < len(starts):
    j = i
    while j < len(starts) - 1 and merge[j]:
        j += 1
    bouts.append((pos_t[starts[i]], pos_t[ends[j]]))
    i = j + 1

kept = []
for t0, t1 in bouts:
    if t1 - t0 < 1.0:
        continue
    m = (pos_t >= t0) & (pos_t <= t1) & valid
    x = lin[m]
    if len(x) < 5 or (x.max() - x.min()) < 0.3:
        continue
    if np.median(np.abs(np.diff(x))) / DT_POS < 0.15:
        continue
    kept.append((t0, t1))
bout_starts = np.array([k[0] for k in kept])
bout_ends = np.array([k[1] for k in kept])
print(f"run bouts: {len(kept)}, total {bout_ends.sum() - bout_starts.sum():.1f} s "
      f"of {epoch_end[1] - epoch_start[1]:.0f} s maze epoch")

# concatenated run-bout time axis (tau) for the SI shuffle
bout_durs = bout_ends - bout_starts
tau_offset = np.r_[0, np.cumsum(bout_durs)[:-1]]
T_total = bout_durs.sum()
in_bout = np.zeros(len(pos_t), bool)
for t0, t1 in kept:
    in_bout |= (pos_t >= t0) & (pos_t <= t1)
samp = in_bout & valid  # mask NaN samples inside merged gaps
x_samp, t_samp = lin[samp], pos_t[samp]
bidx_s = np.searchsorted(bout_starts, t_samp, side="right") - 1
tau_samp = tau_offset[bidx_s] + (t_samp - bout_starts[bidx_s])

edges = np.linspace(0, TRACK_LEN, N_BINS + 1)
bin_centers = (edges[:-1] + edges[1:]) / 2
occ_sec = np.bincount(np.clip((x_samp / TRACK_LEN * N_BINS).astype(int), 0, N_BINS - 1),
                      minlength=N_BINS) * DT_POS


def spike_positions(st):
    """Track positions of spikes inside run bouts (nearest position sample)."""
    if len(st) == 0:
        return np.array([])
    bidx = np.searchsorted(bout_starts, st, side="right") - 1
    ok = (bidx >= 0) & (bidx < len(kept))
    ok[ok] &= st[ok] <= bout_ends[bidx[ok]]
    st = st[ok]
    if len(st) == 0:
        return np.array([])
    idx = np.clip(np.searchsorted(pos_t, st), 1, len(pos_t) - 1)
    closer_right = (pos_t[idx] - st) < (st - pos_t[idx - 1])
    x = lin[np.where(closer_right, idx, idx - 1)]
    return x[~np.isnan(x)]


def rate_map_from_x(x):
    counts = np.bincount(np.clip((x / TRACK_LEN * N_BINS).astype(int), 0, N_BINS - 1),
                         minlength=N_BINS)
    with np.errstate(divide="ignore", invalid="ignore"):
        rm = counts / occ_sec
    rm[occ_sec < 0.1] = 0.0
    return rm, counts


def skaggs_si(rm):
    p = occ_sec / occ_sec.sum()
    R = (rm * p).sum()
    if R <= 0:
        return 0.0
    m = (rm > 0) & (p > 0)
    return float((p[m] * rm[m] / R * np.log2(rm[m] / R)).sum())


exc_ids = unit_ids[cell_type == "excitatory"]
rate_maps, spike_counts, si_val = {}, {}, {}
spike_tau = {}
for k in exc_ids:
    x = spike_positions(spikes[k])
    rate_maps[k], spike_counts[k] = rate_map_from_x(x)
    si_val[k] = skaggs_si(rate_maps[k])
    st = spikes[k]
    bidx = np.searchsorted(bout_starts, st, side="right") - 1
    ok = (bidx >= 0) & (bidx < len(kept))
    ok[ok] &= st[ok] <= bout_ends[bidx[ok]]
    spike_tau[k] = tau_offset[bidx[ok]] + (st[ok] - bout_starts[bidx[ok]])

NSHUF_SI = 500
rng_si = np.random.default_rng(42)
si_null = {k: np.zeros(NSHUF_SI) for k in exc_ids}
for k in tqdm(list(exc_ids), desc="SI shuffle"):
    tau = spike_tau[k]
    if len(tau) < 10:
        continue
    for i, sh in enumerate(rng_si.uniform(0, T_total, NSHUF_SI)):
        x_sh = np.interp((tau + sh) % T_total, tau_samp, x_samp)
        si_null[k][i] = skaggs_si(rate_map_from_x(x_sh)[0])

si_arr = np.array([si_val[k] for k in exc_ids])
si_p = np.array([(np.sum(si_null[k] >= si_val[k]) + 1) / (NSHUF_SI + 1) for k in exc_ids])
mean_rate = np.array([spike_counts[k].sum() / occ_sec.sum() for k in exc_ids])
peak_rate = np.array([rate_maps[k].max() for k in exc_ids])
is_place = (si_p < 0.05) & (mean_rate >= 0.1) & (peak_rate >= 1.0)
place_ids = exc_ids[is_place]
R = np.array([rate_maps[k] for k in place_ids])  # template: (n_cells, N_BINS), Hz
print(f"place cells: {is_place.sum()}/{len(exc_ids)} excitatory units")

# %% [markdown]
# ### Figure: behavior and place fields

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
ax = axes[0]
ax.plot(pos_t[::10], pos_xy[::10, 0], lw=0.3)
ax.set_ylabel("x (m)")
ax.set_title("2D position (x) over the maze epoch")
ax = axes[1]
ax.plot(pos_t[::10], lin[::10], lw=0.3)
for t0, t1 in kept:
    ax.axvspan(t0, t1, color="orange", alpha=0.15, lw=0)
ax.set_ylabel("linearized (m)")
ax.set_title("Linearized position with run bouts highlighted")
ax = axes[2]
example = place_ids[np.argmax(R.max(axis=1))]
st = spikes[example]
m = (st > pos_t[0]) & (st < pos_t[0] + 120)
ax.eventplot([st[m] - pos_t[0]], lineoffsets=0, color="k")
ax.set_xlim(0, 120)
ax.set_yticks([])
ax.set_xlabel("time in maze epoch (s)")
ax.set_title(f"Example place cell {example} spikes, first 120 s of maze")
fig.tight_layout()
fig.savefig("figures/fig01_behavior.png", dpi=150)
plt.close(fig)

order = np.argsort(np.argmax(R, axis=1))
rm_norm = R[order] / R[order].max(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
ax = axes[0]
im = ax.imshow(rm_norm, aspect="auto", cmap="viridis",
               extent=[0, TRACK_LEN, len(place_ids), 0])
ax.set_xlabel("track position (m)")
ax.set_ylabel("place cell (sorted by peak)")
ax.set_title(f"Place-field template ({len(place_ids)} cells)")
fig.colorbar(im, ax=ax, label="normalized rate")
ax = axes[1]
for row in R[:6]:
    ax.plot(bin_centers, row, lw=1)
ax.set_xlabel("track position (m)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("Example rate maps (6 cells)")
ax = axes[2]
ax.hist(si_arr[is_place], bins=30, alpha=0.7, label=f"place (n={is_place.sum()})")
ax.hist(si_arr[~is_place], bins=30, alpha=0.7, label=f"non-place (n={(~is_place).sum()})")
ax.set_xlabel("Skaggs SI (bits/spike)")
ax.set_ylabel("count")
ax.set_title("Spatial information")
ax.legend()
fig.tight_layout()
fig.savefig("figures/fig02_place_fields.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Ripple-channel selection and SWR detection
#
# The electrodes table carries no anatomy, so the ripple channel is chosen
# data-driven: on a 200 s POST Non-REM chunk, each channel is scored by the
# peakiness of its 100-250 Hz envelope (99th percentile / median) times its
# ripple/delta power ratio. The winner is channel 117, the CA1 pyramidal-layer
# channel (independently the strongest theta channel in this session).
#
# SWR detection on that channel: bandpass 100-250 Hz (4th-order Butterworth in
# SOS form), Hilbert envelope, Gaussian smoothing with sigma 4 ms. Thresholds
# come from Non-REM samples only: event edges at mean+1SD, peaks must exceed
# mean+4SD. Events closer than 30 ms are merged; events must last 30-500 ms
# and lie fully inside a Non-REM interval.

# %%
lfp_ds = h5py_file["processing/ecephys/LFP/LFP"]
conv = lfp_ds["data"].attrs["conversion"]

post_start = epoch_start[epoch_label == "POSTEpoch"][0]
post_nr = [(s0, e0) for s0, e0, lab in zip(state_start, state_end, state_label)
           if lab == "Non-REM" and s0 >= post_start]
s0, e0 = max(post_nr, key=lambda se: se[1] - se[0])
chunk_t0, chunk_dur = s0 + 10, 200.0
i0, i1 = int(chunk_t0 * FS_LFP), int((chunk_t0 + chunk_dur) * FS_LFP)
chunk = np.asarray(lfp_ds["data"][i0:i1, :], dtype=np.float32) * conv

sos_ripple = signal.butter(4, [100, 250], btype="bandpass", fs=FS_LFP, output="sos")
metric = np.zeros(chunk.shape[1])
for ch in tqdm(range(chunk.shape[1]), desc="scoring channels"):
    x = chunk[:, ch]
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos_ripple, x)))
    peakiness = np.percentile(env, 99) / np.median(env)
    f, Pxx = signal.welch(x, fs=FS_LFP, nperseg=8192)
    metric[ch] = (peakiness * Pxx[(f >= 100) & (f <= 250)].mean()
                  / Pxx[(f >= 1) & (f <= 4)].mean())
best_ch = int(np.argmax(metric))
print(f"best ripple channel: {best_ch}")

print("reading full-session LFP for the ripple channel...")
lfp = np.asarray(lfp_ds["data"][:, best_ch], dtype=np.float32) * conv
t_lfp = np.arange(len(lfp)) / FS_LFP

nonrem = np.zeros(len(lfp), bool)
for s0, e0 in zip(state_start[state_label == "Non-REM"], state_end[state_label == "Non-REM"]):
    nonrem[int(s0 * FS_LFP):int(e0 * FS_LFP)] = True
print(f"Non-REM total: {nonrem.sum() / FS_LFP:.0f} s")

filt = signal.sosfiltfilt(sos_ripple, lfp)
env = np.abs(signal.hilbert(filt))
sigma_samp = 0.004 * FS_LFP
env = signal.convolve(
    env,
    signal.windows.gaussian(int(8 * sigma_samp) + 1, sigma_samp) / (np.sqrt(2 * np.pi) * sigma_samp),
    mode="same")
mu, sd = env[nonrem].mean(), env[nonrem].std()
lo, hi = mu + sd, mu + 4 * sd
print(f"envelope thresholds: edge {lo * 1e6:.1f} uV, peak {hi * 1e6:.1f} uV")

above = env > lo
d = np.diff(above.astype(int))
starts, ends = np.where(d == 1)[0] + 1, np.where(d == -1)[0]
if above[0]:
    starts = np.r_[0, starts]
if above[-1]:
    ends = np.r_[ends, len(above) - 1]
merged, i = [], 0
while i < len(starts):
    j = i
    while j < len(starts) - 1 and (starts[j + 1] - ends[j]) / FS_LFP < 0.03:
        j += 1
    merged.append((starts[i], ends[j]))
    i = j + 1
events = []
for si_, ei_ in merged:
    dur = (ei_ - si_) / FS_LFP
    if 0.03 <= dur <= 0.5 and env[si_:ei_].max() >= hi and nonrem[si_:ei_].all():
        events.append((si_ / FS_LFP, ei_ / FS_LFP))
events = np.array(events)

pre_events = events[events[:, 1] < epoch_end[epoch_label == "PREEpoch"][0]]
post_events = events[events[:, 0] >= post_start]
durs_ms = (events[:, 1] - events[:, 0]) * 1000
nr_pre = sum(e - s for s, e, l in zip(state_start, state_end, state_label)
             if l == "Non-REM" and e < epoch_end[0])
nr_post = sum(e - s for s, e, l in zip(state_start, state_end, state_label)
              if l == "Non-REM" and s >= post_start)
print(f"SWRs: {len(pre_events)} PRE ({len(pre_events) / nr_pre * 60:.1f}/min Non-REM), "
      f"{len(post_events)} POST ({len(post_events) / nr_post * 60:.1f}/min), "
      f"median duration {np.median(durs_ms):.0f} ms")

# %% [markdown]
# ### Figure: SWR detection

# %%
fig, axes = plt.subplots(3, 1, figsize=(11, 9))
ex_s, ex_e = post_events[len(post_events) // 2, 0] - 2, post_events[len(post_events) // 2, 0] + 3
m = (t_lfp >= ex_s) & (t_lfp <= ex_e)
tt = t_lfp[m]
ax = axes[0]
ax.plot(tt, lfp[m] * 1e6, lw=0.4, color="gray", label="wideband")
ax.plot(tt, filt[m] * 1e6, lw=0.6, color="tab:blue", label="100-250 Hz")
for s0, e0 in post_events:
    if e0 < ex_s or s0 > ex_e:
        continue
    ax.axvspan(s0, e0, color="red", alpha=0.25, lw=0)
ax.set_ylabel("uV")
ax.set_title("Example POST Non-REM segment with detected SWRs (red)")
ax.legend(loc="upper right", fontsize=8)
ax = axes[1]
ax.plot(tt, env[m] * 1e6, lw=0.8, color="k")
ax.axhline(lo * 1e6, color="tab:blue", ls="--", lw=1, label="edge (mean+1SD)")
ax.axhline(hi * 1e6, color="tab:red", ls="--", lw=1, label="peak (mean+4SD)")
for s0, e0 in post_events:
    if e0 < ex_s or s0 > ex_e:
        continue
    ax.axvspan(s0, e0, color="red", alpha=0.25, lw=0)
ax.set_ylabel("envelope (uV)")
ax.set_title("Ripple-band envelope and thresholds")
ax.legend(loc="upper right", fontsize=8)
ax = axes[2]
ax.hist(durs_ms, bins=np.arange(30, 300, 5), color="tab:blue", alpha=0.8)
ax.set_xlabel("SWR duration (ms)")
ax.set_ylabel("count")
ax.set_title(f"SWR duration distribution (n={len(events)}, median {np.median(durs_ms):.0f} ms)")
fig.tight_layout()
fig.savefig("figures/fig03_ripple_detection.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Bayesian decoding of SWRs
#
# Each SWR is divided into 20 ms bins. Under a Poisson model the position
# posterior for a bin with spike counts n_i is proportional to
# exp(-tau * sum_i r_i(x)) * prod_i r_i(x)^n_i, where r_i is the rate map of
# cell i and tau = 20 ms. Rate maps are floored at 0.01 Hz so a single spike
# does not zero out a bin everywhere. Events enter decoding only if at least 5
# template cells fire and at least 5 bins are non-empty.
#
# The replay score is the correlation between time and position across all
# (time bin, position bin) pairs weighted by posterior mass: +1 is a perfect
# forward trajectory, -1 a perfect reverse one. Significance uses 500 cell-ID
# shuffles per event (rate maps permuted across cells), which destroys the
# place-field-to-cell assignment while preserving each event's spike
# structure.

# %%
BIN, MIN_BINS, MIN_CELLS, NSHUF, RATE_FLOOR = 0.020, 5, 5, 500, 1e-2
rng = np.random.default_rng(7)
C, P = R.shape
Rf = np.maximum(R, RATE_FLOOR)
logR = np.log(Rf)
sumR = Rf.sum(axis=0)
place_spikes = {k: spikes[k] for k in place_ids}


def event_counts(ev):
    t0, t1 = ev
    n_bins = int((t1 - t0) // BIN)
    if n_bins < MIN_BINS:
        return None
    counts = np.zeros((n_bins, C))
    for ci, k in enumerate(place_ids):
        st = place_spikes[k]
        i0, i1 = np.searchsorted(st, [t0, t0 + n_bins * BIN])
        if i1 > i0:
            counts[:, ci] = np.bincount(((st[i0:i1] - t0) / BIN).astype(int),
                                        minlength=n_bins)
    return counts


def posterior_from_counts(counts):
    with np.errstate(all="ignore"):  # cosmetic Accelerate BLAS FP warnings
        LL = counts @ logR - BIN * sumR[None, :]
    LL -= LL.max(axis=1, keepdims=True)
    Ppost = np.exp(LL)
    return Ppost / Ppost.sum(axis=1, keepdims=True)


def weighted_corr(Ppost, t, x):
    """Posterior-mass-weighted time-position correlation. Ppost: (S, T, P) -> (S,)."""
    S, T, P = Ppost.shape
    W = float(T)  # posterior rows sum to 1
    ex = np.einsum("stp,p->st", Ppost, x)
    mx = ex.sum(axis=1) / W
    mt = t.mean()
    cov = (ex * t[None, :]).sum(axis=1) / W - mt * mx
    var_t = ((t - mt) ** 2).mean()
    dx2 = (x[None, :] - mx[:, None]) ** 2
    var_x = np.einsum("stp,sp->s", Ppost, dx2) / W
    return np.nan_to_num(cov / np.sqrt(var_t * var_x))


def decode_event(counts):
    t = (np.arange(counts.shape[0]) + 0.5) * BIN
    Ppost = posterior_from_counts(counts)
    real = weighted_corr(Ppost[None], t, bin_centers)[0]
    # batched cell-ID shuffles: independent permutations of the cell axis
    perms = rng.permuted(np.tile(np.arange(C), (NSHUF, 1)), axis=1)
    cperm = counts[:, perms].transpose(1, 0, 2)  # (NSHUF, T, C)
    with np.errstate(all="ignore"):  # cosmetic Accelerate BLAS FP warnings;
        # outputs verified NaN-free and bit-identical to a brute-force loop
        LL = cperm @ logR - BIN * sumR[None, None, :]
        LL -= LL.max(axis=2, keepdims=True)
        Pn = np.exp(LL)
        Pn /= Pn.sum(axis=2, keepdims=True)
        null = weighted_corr(Pn, t, bin_centers)
    p = (np.sum(np.abs(null) >= abs(real)) + 1) / (NSHUF + 1)
    return real, p, Ppost


def run_events(ev_list, label):
    rows = []
    for ev in tqdm(ev_list, desc=f"decoding {label}"):
        counts = event_counts(ev)
        if counts is None:
            continue
        if (counts.sum(axis=0) > 0).sum() < MIN_CELLS or \
           (counts.sum(axis=1) > 0).sum() < MIN_BINS:
            continue
        real, p, Ppost = decode_event(counts)
        rows.append({"start": ev[0], "end": ev[1], "score": real, "p": p,
                     "posterior": Ppost})
    print(f"{label}: {len(rows)} decodable events")
    return rows


pre_rows = run_events(pre_events, "PRE")
post_rows = run_events(post_events, "POST")

# %% [markdown]
# ## Results: POST-sleep SWRs replay trajectories, PRE-sleep SWRs do not

# %%
pre_scores = np.array([r["score"] for r in pre_rows])
pre_ps = np.array([r["p"] for r in pre_rows])
post_scores = np.array([r["score"] for r in post_rows])
post_ps = np.array([r["p"] for r in post_rows])
pre_sig, post_sig = pre_ps < 0.05, post_ps < 0.05

for label, sc, sg in [("PRE", pre_scores, pre_sig), ("POST", post_scores, post_sig)]:
    print(f"{label}: {sg.sum()}/{len(sc)} significant ({100 * sg.mean():.1f}%), "
          f"fwd {(sg & (sc > 0)).sum()}, rev {(sg & (sc < 0)).sum()}, "
          f"median |score| {np.median(np.abs(sc)):.3f}")

table = [[pre_sig.sum(), len(pre_rows) - pre_sig.sum()],
         [post_sig.sum(), len(post_rows) - post_sig.sum()]]
_, fisher_p = stats.fisher_exact(table)
mw = stats.mannwhitneyu(np.abs(pre_scores), np.abs(post_scores))
n_fwd = int((post_sig & (post_scores > 0)).sum())
binom_p = stats.binomtest(n_fwd, int(post_sig.sum()), 0.5).pvalue
print(f"Fisher exact PRE vs POST significant fraction: p = {fisher_p:.2e}")
print(f"Mann-Whitney |score| PRE vs POST: p = {mw.pvalue:.2e}")
print(f"POST forward/reverse binomial: p = {binom_p:.2e}")

# %% [markdown]
# ### Figures: example trajectories and population statistics

# %%
fwd_ex = sorted([r for r in post_rows if r["p"] < 0.05 and r["score"] > 0],
                key=lambda r: -r["score"])[:3]
rev_ex = sorted([r for r in post_rows if r["p"] < 0.05 and r["score"] < 0],
                key=lambda r: r["score"])[:3]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, r in zip(axes.ravel(), fwd_ex + rev_ex):
    Ppost = r["posterior"]
    ax.imshow(Ppost.T, aspect="auto", origin="lower", cmap="hot",
              extent=[0, Ppost.shape[0] * BIN * 1000, 0, TRACK_LEN])
    direction = "forward" if r["score"] > 0 else "reverse"
    ax.set_title(f"{direction}, score={r['score']:.2f}, p={r['p']:.3f}", fontsize=10)
    ax.set_xlabel("time in SWR (ms)")
    ax.set_ylabel("position (m)")
fig.suptitle("Example decoded POST-sleep SWR events (Bayesian posterior)")
fig.tight_layout()
fig.savefig("figures/fig04_example_replays.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ax = axes[0]
bins = np.linspace(0, 1, 41)
ax.hist(np.abs(pre_scores), bins=bins, alpha=0.6, density=True,
        label=f"PRE (n={len(pre_scores)})")
ax.hist(np.abs(post_scores), bins=bins, alpha=0.6, density=True,
        label=f"POST (n={len(post_scores)})")
ax.set_xlabel("|replay score| (weighted correlation)")
ax.set_ylabel("density")
ax.set_title(f"|score| distributions (MW p={mw.pvalue:.1e})")
ax.legend()
ax = axes[1]
ax.hist(pre_scores, bins=np.linspace(-1, 1, 61), alpha=0.6, density=True, label="PRE")
ax.hist(post_scores, bins=np.linspace(-1, 1, 61), alpha=0.6, density=True, label="POST")
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("replay score (signed)")
ax.set_ylabel("density")
ax.set_title("Signed score distributions")
ax.legend()
ax = axes[2]
bars = ax.bar(["PRE", "POST"], [100 * pre_sig.mean(), 100 * post_sig.mean()],
              color=["tab:gray", "tab:red"])
ax.set_ylabel("% events significant (p<0.05)")
ax.set_title(f"Significant replay fraction (Fisher p={fisher_p:.1e})")
for bar, rows_, sig_ in zip(bars, [pre_rows, post_rows], [pre_sig, post_sig]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{sig_.sum()}/{len(rows_)}", ha="center", fontsize=10)
ax.set_ylim(0, 100 * post_sig.mean() * 1.35 + 1)
fig.tight_layout()
fig.savefig("figures/fig05_replay_stats.png", dpi=150)
plt.close(fig)
print("DONE")
