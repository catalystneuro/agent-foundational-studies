# %% [markdown]
# # Sharp-Wave Ripples and Hippocampal Replay
#
# This notebook demonstrates two classic hippocampal electrophysiology phenomena using a
# real dataset streamed from the DANDI Archive:
#
# 1. **Sharp-wave ripples (SWRs)** — brief (~30-100 ms), high-frequency (150-250 Hz)
#    oscillatory events in the CA1 local field potential (LFP) that occur during
#    immobility and slow-wave sleep.
# 2. **Replay** — during SWRs, place cells that were active during recent spatial
#    behavior tend to reactivate in a temporally compressed version of the order in
#    which their place fields were visited on the track.
#
# ## Dataset
#
# We use **DANDI:000044** (Grosmark & Buzsaki, *Science* 2016, "Diversity in neural
# firing dynamics supports both rigid and learned hippocampal sequences"). Session
# `sub-Achilles_ses-Achilles-10252013` contains bilateral CA1 silicon-probe recordings
# (128 LFP channels, 137 spike-sorted units) from a rat running a 1.6 m linear track,
# bracketed by a PRE-run rest/sleep epoch and a POST-run rest/sleep epoch:
#
# - **PRE** epoch (0 - 18079.5 s): rest/sleep *before* track running (no learned
#   sequence for this track should be replayed here — negative control).
# - **MazeEpoch** (18079.5 - 20147 s): the rat runs back and forth on the linear track.
# - **POST** epoch (20147 - 34861 s): rest/sleep *after* track running, where we expect
#   replay of the track sequence learned during MazeEpoch.
#
# Data are streamed directly from the DANDI S3 bucket with `remfile` (chunked/cached,
# no full download) and analyzed with `pynapple`.

# %%
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import mannwhitneyu, spearmanr

RNG = np.random.default_rng(42)

# %% [markdown]
# ## 1. Load the NWB file (streaming)

# %%
S3_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/"
    "7632d81b-2819-473d-8946-34dc939e6028"
)  # DANDI:000044, sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %%
epochs_df = nwbfile.epochs.to_dataframe()
print(epochs_df)

units_meta = nwb["units"].metadata
print(units_meta["cell_type"].value_counts())
print(units_meta["location"].value_counts())

# %% [markdown]
# ## 2. Selecting a CA1 ripple-band LFP channel
#
# The 128-channel LFP has no anatomical annotation identifying which channel sits in
# the CA1 pyramidal layer. We select the channel with the highest ripple-band
# (150-250 Hz) to broadband (1-300 Hz) power ratio during a representative bout of
# Non-REM sleep in the POST epoch — a standard, purely data-driven heuristic for
# finding the sharp-wave ripple channel.

# %%
lfp_es = nwbfile.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"]
FS_LFP = lfp_es.rate
N_CHANNELS = lfp_es.data.shape[1]
print("LFP: ", lfp_es.data.shape, "@", FS_LFP, "Hz,", N_CHANNELS, "channels")

states = nwb["states"]
epochs = nwb["epochs"]
pre_start, pre_stop = epochs.loc[0, "start"], epochs.loc[0, "end"]
maze_start, maze_stop = epochs.loc[1, "start"], epochs.loc[1, "end"]
post_start, post_stop = epochs.loc[2, "start"], epochs.loc[2, "end"]

nrem_all = states[states.label == "Non-REM"]
nrem_post = nrem_all[nrem_all.start >= post_start]
durations = nrem_post.end - nrem_post.start
longest_bout = nrem_post[np.argmax(durations)]
t0 = float(longest_bout.start[0])
t1 = min(float(longest_bout.end[0]), t0 + 120.0)
print(f"Channel-selection window: {t0:.1f}-{t1:.1f} s (Non-REM, POST epoch)")

i0, i1 = int(t0 * FS_LFP), int(t1 * FS_LFP)
chan_select_chunk = lfp_es.data[i0:i1, :].astype(np.float32)


def bandpass(x, lo, hi, fs, order=4, axis=0):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=axis)


ripple_band = bandpass(chan_select_chunk, 150, 250, FS_LFP)
broadband = bandpass(chan_select_chunk, 1, 300, FS_LFP)
ripple_power = np.mean(ripple_band**2, axis=0)
broad_power = np.mean(broadband**2, axis=0)
ripple_ratio = ripple_power / broad_power
RIPPLE_CHANNEL = int(np.argmax(ripple_ratio))
print(f"Selected LFP channel {RIPPLE_CHANNEL} (ripple/broadband power ratio = "
      f"{ripple_ratio[RIPPLE_CHANNEL]:.4f}, next best = "
      f"{np.sort(ripple_ratio)[-2]:.4f})")

# %% [markdown]
# ## 3. Sharp-wave ripple detection
#
# We load the full-session LFP trace for the selected channel during the PRE and POST
# epochs (single channel, so this is cheap: ~40 MB each), band-pass filter 150-250 Hz,
# take the Hilbert envelope, z-score it, and detect events where the envelope exceeds
# 2 SD with a peak exceeding 4 SD, lasting 20-400 ms, merging events within 20 ms of
# each other (standard sharp-wave ripple detection algorithm, e.g. Csicsvari et al.
# 1999).

# %%
def load_channel(t_start, t_stop, channel, fs=FS_LFP):
    i0, i1 = int(t_start * fs), int(t_stop * fs)
    t0 = time.time()
    trace = lfp_es.data[i0:i1, channel]
    print(f"  loaded {i1 - i0} samples in {time.time() - t0:.1f}s")
    return trace


print("Loading PRE-epoch LFP channel...")
lfp_pre = load_channel(pre_start, pre_stop, RIPPLE_CHANNEL)
print("Loading POST-epoch LFP channel...")
lfp_post = load_channel(post_start, post_stop, RIPPLE_CHANNEL)


def detect_ripples(
    trace,
    fs,
    t_offset=0.0,
    lo=150,
    hi=250,
    low_thresh=2.0,
    high_thresh=4.0,
    min_dur=0.02,
    max_dur=0.4,
    merge_gap=0.02,
):
    """Detect sharp-wave ripples from a single LFP channel.

    Returns an (n_events, 4) array of [start, end, peak_time, peak_zscore], all in
    absolute time (t_offset added).
    """
    trace = trace.astype(np.float64)
    filt = bandpass(trace, lo, hi, fs)
    env = np.abs(hilbert(filt))
    z = (env - env.mean()) / env.std()

    above_low = z > low_thresh
    d = np.diff(above_low.astype(int))
    starts = np.where(d == 1)[0] + 1
    stops = np.where(d == -1)[0] + 1
    if above_low[0]:
        starts = np.r_[0, starts]
    if above_low[-1]:
        stops = np.r_[stops, len(above_low)]

    events = []
    for s, e in zip(starts, stops):
        if z[s:e].max() < high_thresh:
            continue
        dur = (e - s) / fs
        if dur < min_dur or dur > max_dur:
            continue
        peak_idx = s + np.argmax(z[s:e])
        events.append((s / fs, e / fs, peak_idx / fs, z[peak_idx]))

    events = np.array(events)
    if len(events) == 0:
        return events

    merged = [events[0]]
    for ev in events[1:]:
        if ev[0] - merged[-1][1] < merge_gap:
            s = merged[-1][0]
            e = max(merged[-1][1], ev[1])
            zmax = max(merged[-1][3], ev[3])
            pk = merged[-1][2] if merged[-1][3] >= ev[3] else ev[2]
            merged[-1] = (s, e, pk, zmax)
        else:
            merged.append(ev)
    merged = np.array(merged)
    merged[:, 0:3] += t_offset
    return merged


ripples_pre = detect_ripples(lfp_pre, FS_LFP, t_offset=pre_start)
ripples_post = detect_ripples(lfp_post, FS_LFP, t_offset=post_start)
print(f"Detected {len(ripples_pre)} candidate SWRs in PRE "
      f"({len(ripples_pre) / (pre_stop - pre_start):.3f} Hz)")
print(f"Detected {len(ripples_post)} candidate SWRs in POST "
      f"({len(ripples_post) / (post_stop - post_start):.3f} Hz)")

# %% [markdown]
# Restrict ripples to Non-REM sleep bouts, where SWRs are most reliably associated
# with offline memory consolidation (as opposed to brief immobility during quiet
# wakefulness).

# %%
def filter_events_by_intervalset(events, intervalset):
    starts = intervalset.start
    ends = intervalset.end
    keep = np.zeros(len(events), dtype=bool)
    for i, ev in enumerate(events):
        center = ev[2]
        idx = np.searchsorted(starts, center) - 1
        if 0 <= idx < len(ends) and starts[idx] <= center <= ends[idx]:
            keep[i] = True
    return events[keep]


nrem_pre = nrem_all[nrem_all.end <= maze_start]
nrem_post = nrem_all[nrem_all.start >= post_start]
rip_pre_nrem = filter_events_by_intervalset(ripples_pre, nrem_pre)
rip_post_nrem = filter_events_by_intervalset(ripples_post, nrem_post)
print(f"PRE  Non-REM SWRs: {len(rip_pre_nrem)}/{len(ripples_pre)}")
print(f"POST Non-REM SWRs: {len(rip_post_nrem)}/{len(ripples_post)}")

# %% [markdown]
# ### Figure 1: raw LFP, ripple-band filtered trace, and detected events

# %%
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)

# Panel A: several seconds of raw LFP with detected ripple events marked
window_center = float(nrem_post.start[0]) + 30
w0, w1 = window_center - 3, window_center + 3
seg = lfp_es.data[int(w0 * FS_LFP): int(w1 * FS_LFP), RIPPLE_CHANNEL].astype(float)
t_seg = np.arange(len(seg)) / FS_LFP + w0
axes[0].plot(t_seg, seg, color="k", lw=0.6)
for ev in rip_post_nrem:
    if w0 <= ev[2] <= w1:
        axes[0].axvspan(ev[0], ev[1], color="orange", alpha=0.35)
axes[0].set_ylabel("Raw LFP (a.u.)")
axes[0].set_title(f"CA1 LFP (channel {RIPPLE_CHANNEL}) during Non-REM sleep, POST epoch")

# Panel B & C: zoom on a single high-amplitude example ripple
order = np.argsort(rip_post_nrem[:, 3])[::-1]
example_ev = rip_post_nrem[order[3]]
center = example_ev[2]
i0, i1 = int((center - 0.3) * FS_LFP), int((center + 0.3) * FS_LFP)
raw_zoom = lfp_es.data[i0:i1, RIPPLE_CHANNEL].astype(float)
t_zoom = np.arange(i0, i1) / FS_LFP
filt_zoom = bandpass(raw_zoom, 150, 250, FS_LFP)

axes[1].plot(t_zoom, raw_zoom, color="k", lw=0.8)
axes[1].axvspan(example_ev[0], example_ev[1], color="orange", alpha=0.35)
axes[1].set_ylabel("Raw LFP (a.u.)")
axes[1].set_title(f"Example ripple at t={center:.2f}s (z={example_ev[3]:.1f})")

axes[2].plot(t_zoom, filt_zoom, color="darkred", lw=0.8)
axes[2].axvspan(example_ev[0], example_ev[1], color="orange", alpha=0.35)
axes[2].set_ylabel("150-250 Hz band")
axes[2].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("fig1_ripple_detection.png", dpi=150)
plt.close()
print("saved fig1_ripple_detection.png")

# %% [markdown]
# ### Figure 2: ripple summary statistics

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

durs_pre = (rip_pre_nrem[:, 1] - rip_pre_nrem[:, 0]) * 1000
durs_post = (rip_post_nrem[:, 1] - rip_post_nrem[:, 0]) * 1000
axes[0].hist(durs_pre, bins=40, alpha=0.6, label="PRE", color="steelblue", density=True)
axes[0].hist(durs_post, bins=40, alpha=0.6, label="POST", color="firebrick", density=True)
axes[0].set_xlabel("Ripple duration (ms)")
axes[0].set_ylabel("Density")
axes[0].set_title("Ripple duration")
axes[0].legend()

axes[1].hist(rip_pre_nrem[:, 3], bins=40, alpha=0.6, label="PRE", color="steelblue", density=True)
axes[1].hist(rip_post_nrem[:, 3], bins=40, alpha=0.6, label="POST", color="firebrick", density=True)
axes[1].set_xlabel("Peak envelope z-score")
axes[1].set_title("Ripple amplitude")
axes[1].legend()

nrem_pre_total = float(np.sum(nrem_pre.end - nrem_pre.start))
nrem_post_total = float(np.sum(nrem_post.end - nrem_post.start))
rates = [len(rip_pre_nrem) / nrem_pre_total, len(rip_post_nrem) / nrem_post_total]
axes[2].bar(["PRE", "POST"], rates, color=["steelblue", "firebrick"])
axes[2].set_ylabel("SWR rate during Non-REM (Hz)")
axes[2].set_title("SWR rate")

plt.tight_layout()
plt.savefig("fig2_ripple_stats.png", dpi=150)
plt.close()
print("saved fig2_ripple_stats.png")
print(f"PRE  Non-REM SWR rate: {rates[0]:.3f} Hz over {nrem_pre_total:.0f} s")
print(f"POST Non-REM SWR rate: {rates[1]:.3f} Hz over {nrem_post_total:.0f} s")

# %% [markdown]
# ## 4. Place fields from track running
#
# The animal runs back and forth on the linear track, so place fields are direction-
# dependent. We split running periods into rightward/leftward bouts by velocity sign
# and compute separate 1D tuning curves (occupancy-normalized firing rate vs.
# linearized position) for each direction, using excitatory (putative pyramidal)
# CA1 units. "Place cells" are defined as cells with peak rate > 1 Hz and spatial
# information > 0.2 bits/spike (Skaggs et al. 1993) in at least one direction.

# %%
maze_ep = nap.IntervalSet(start=maze_start, end=maze_stop)
linpos = nwb["1.6mLinearMazeLinearizedTimeSeries"]
pos_tsd = linpos.restrict(maze_ep)[:, 0].dropna()

t_pos = pos_tsd.index.values
x_pos = pos_tsd.values
dt_pos = np.diff(t_pos)
vel = np.diff(x_pos) / dt_pos
vel_t = t_pos[:-1] + dt_pos / 2

SPEED_THRESH = 0.1  # m/s


def mask_to_intervalset(mask, tvec, gap_merge=0.2):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return nap.IntervalSet(start=[], end=[])
    starts = [tvec[idx[0]]]
    ends = []
    for i in range(1, len(idx)):
        if tvec[idx[i]] - tvec[idx[i - 1]] > gap_merge:
            ends.append(tvec[idx[i - 1]])
            starts.append(tvec[idx[i]])
    ends.append(tvec[idx[-1]])
    return nap.IntervalSet(start=starts, end=ends)


right_ep = mask_to_intervalset(vel > SPEED_THRESH, vel_t)
left_ep = mask_to_intervalset(vel < -SPEED_THRESH, vel_t)
print(f"Rightward running: {len(right_ep)} bouts, "
      f"{float(np.sum(right_ep.end - right_ep.start)):.0f} s total")
print(f"Leftward running:  {len(left_ep)} bouts, "
      f"{float(np.sum(left_ep.end - left_ep.start)):.0f} s total")

exc_ca1 = nwb["units"][units_meta.cell_type == "excitatory"]

TRACK_LENGTH = 1.6
N_BINS = 40
tc_right = nap.compute_tuning_curves(
    data=exc_ca1, features=pos_tsd, bins=N_BINS, range=[(0, TRACK_LENGTH)], epochs=right_ep
)
tc_left = nap.compute_tuning_curves(
    data=exc_ca1, features=pos_tsd, bins=N_BINS, range=[(0, TRACK_LENGTH)], epochs=left_ep
)
mi_right = nap.compute_mutual_information(tc_right)
mi_left = nap.compute_mutual_information(tc_left)

unit_ids = tc_right.coords["unit"].values
PEAK_RATE_THRESH = 1.0
SPATIAL_INFO_THRESH = 0.2

place_right_mask = (tc_right.max(dim="0").values > PEAK_RATE_THRESH) & (
    mi_right["bits/spike"].values > SPATIAL_INFO_THRESH
)
place_left_mask = (tc_left.max(dim="0").values > PEAK_RATE_THRESH) & (
    mi_left["bits/spike"].values > SPATIAL_INFO_THRESH
)
place_right_ids = unit_ids[place_right_mask]
place_left_ids = unit_ids[place_left_mask]

peak_pos_right = tc_right.sel(unit=place_right_ids).argmax(dim="0").values
peak_pos_left = tc_left.sel(unit=place_left_ids).argmax(dim="0").values
sorted_right = place_right_ids[np.argsort(peak_pos_right)]
sorted_left = place_left_ids[np.argsort(peak_pos_left)]

print(f"Place cells (rightward): {len(sorted_right)} / {len(unit_ids)}")
print(f"Place cells (leftward):  {len(sorted_left)} / {len(unit_ids)}")

# %% [markdown]
# ### Figure 3: place fields on the linear track

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 7))
for ax, tc, sorted_ids, title in [
    (axes[0], tc_right, sorted_right, "Rightward runs"),
    (axes[1], tc_left, sorted_left, "Leftward runs"),
]:
    tc_pc = tc.sel(unit=sorted_ids)
    tc_pc_norm = (tc_pc / tc_pc.max(dim="0")).values
    bin_centers = tc.coords["0"].values
    im = ax.imshow(
        tc_pc_norm,
        aspect="auto",
        cmap="viridis",
        extent=[bin_centers[0], bin_centers[-1], len(sorted_ids), 0],
    )
    ax.set_xlabel("Position on track (m)")
    ax.set_title(f"{title} (n={len(sorted_ids)} place cells)")
axes[0].set_ylabel("Place cell # (sorted by field location)")
fig.colorbar(im, ax=axes, label="Normalized firing rate", shrink=0.7)
plt.savefig("fig3_place_fields.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig3_place_fields.png")

# %% [markdown]
# ## 5. Replay analysis
#
# For each detected SWR, we take every place cell that fired at least once during the
# event window, rank them by the time of their first spike ("temporal order"), and
# compute the Spearman rank correlation with their rank order in the rightward and
# leftward place-field templates. We keep the better (larger |r|) of the two direction
# templates per event (since we don't know a priori which trajectory, or its forward /
# reverse direction, might be replayed) and require at least 6 participating place
# cells for the rank correlation to be estimated reliably. If SWRs during POST sleep
# reflect replay of the track sequence just learned, |r| should be systematically
# larger in POST than in PRE, where the track sequence was never experienced.

# %%
all_place_ids = np.unique(np.concatenate([sorted_right, sorted_left]))
place_tsgroup = nap.TsGroup({uid: exc_ca1[uid] for uid in all_place_ids})
template_rank_right = {uid: i for i, uid in enumerate(sorted_right)}
template_rank_left = {uid: i for i, uid in enumerate(sorted_left)}

MIN_PARTICIPATING = 6


def spearman_fast(rank_x, rank_y):
    n = len(rank_x)
    d = rank_x - rank_y
    return 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))


def precompute_event_ranks(events, min_cells=MIN_PARTICIPATING):
    """For each event, precompute participating units + temporal rank per direction
    template. This is independent of any template-label shuffling."""
    out = []
    for ev in events:
        spk = place_tsgroup.restrict(nap.IntervalSet(ev[0], ev[1]))
        entry = {}
        for template_rank, name in [(template_rank_right, "R"), (template_rank_left, "L")]:
            participating = [uid for uid in template_rank if len(spk[uid]) > 0]
            if len(participating) < min_cells:
                continue
            first_spike_t = np.array([spk[uid].index.values.min() for uid in participating])
            temporal_rank = np.argsort(np.argsort(first_spike_t))
            entry[name] = (np.array(participating), temporal_rank)
        if entry:
            out.append((ev[2], entry))
    return out


def score_with_template(precomp, tr_right, tr_left):
    rs = np.empty(len(precomp))
    ns = np.empty(len(precomp), dtype=int)
    for i, (_, entry) in enumerate(precomp):
        best_r, best_n = 0.0, 0
        for name, tr in [("R", tr_right), ("L", tr_left)]:
            if name not in entry:
                continue
            uids, temporal_rank = entry[name]
            template_vals = np.array([tr[u] for u in uids])
            template_rank = np.argsort(np.argsort(template_vals))
            r = spearman_fast(temporal_rank, template_rank)
            if best_n == 0 or abs(r) > abs(best_r):
                best_r, best_n = r, len(uids)
        rs[i] = best_r
        ns[i] = best_n
    return rs, ns


precomp_post = precompute_event_ranks(rip_post_nrem)
precomp_pre = precompute_event_ranks(rip_pre_nrem)
print(f"POST SWRs with >= {MIN_PARTICIPATING} participating place cells: "
      f"{len(precomp_post)}/{len(rip_post_nrem)}")
print(f"PRE  SWRs with >= {MIN_PARTICIPATING} participating place cells: "
      f"{len(precomp_pre)}/{len(rip_pre_nrem)}")

r_post, n_post = score_with_template(precomp_post, template_rank_right, template_rank_left)
r_pre, n_pre = score_with_template(precomp_pre, template_rank_right, template_rank_left)

print(f"|r| POST: mean={np.mean(np.abs(r_post)):.3f}, median={np.median(np.abs(r_post)):.3f}")
print(f"|r| PRE:  mean={np.mean(np.abs(r_pre)):.3f}, median={np.median(np.abs(r_pre)):.3f}")

U, p_mwu = mannwhitneyu(np.abs(r_post), np.abs(r_pre), alternative="greater")
print(f"Mann-Whitney U test (|r|_POST > |r|_PRE, one-sided): U={U:.0f}, p={p_mwu:.4g}")

# %% [markdown]
# ### Shuffle control
#
# As a control, we repeatedly shuffle the assignment of place-field rank to cell
# identity (breaking the correspondence between spike order and the physical track
# order while preserving which cells participated in each event) and recompute the
# population-mean |r|. This tells us how much of the observed rank correlation could
# arise merely from event structure (multiple cells co-firing within a short window)
# rather than genuine spatial sequence information.

# %%
N_SHUFFLES = 300


def shuffled_template(template_rank, rng):
    uids = np.array(list(template_rank.keys()))
    ranks = np.array(list(template_rank.values()))
    perm = rng.permutation(len(ranks))
    return dict(zip(uids, ranks[perm]))


null_means_post = np.empty(N_SHUFFLES)
null_means_pre = np.empty(N_SHUFFLES)
for i in range(N_SHUFFLES):
    tr_r = shuffled_template(template_rank_right, RNG)
    tr_l = shuffled_template(template_rank_left, RNG)
    rp, _ = score_with_template(precomp_post, tr_r, tr_l)
    rq, _ = score_with_template(precomp_pre, tr_r, tr_l)
    null_means_post[i] = np.mean(np.abs(rp))
    null_means_pre[i] = np.mean(np.abs(rq))

obs_post = np.mean(np.abs(r_post))
obs_pre = np.mean(np.abs(r_pre))
p_post_shuffle = (np.sum(null_means_post >= obs_post) + 1) / (N_SHUFFLES + 1)
p_pre_shuffle = (np.sum(null_means_pre >= obs_pre) + 1) / (N_SHUFFLES + 1)
print(f"POST: observed mean|r|={obs_post:.4f} vs shuffled null "
      f"{null_means_post.mean():.4f}+-{null_means_post.std():.4f}, p={p_post_shuffle:.3f}")
print(f"PRE:  observed mean|r|={obs_pre:.4f} vs shuffled null "
      f"{null_means_pre.mean():.4f}+-{null_means_pre.std():.4f}, p={p_pre_shuffle:.3f}")

# %% [markdown]
# ### Figure 4: replay score distributions (PRE vs POST)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

bins = np.linspace(0, 1, 30)
axes[0].hist(np.abs(r_pre), bins=bins, density=True, alpha=0.6, label=f"PRE (n={len(r_pre)})",
             color="steelblue")
axes[0].hist(np.abs(r_post), bins=bins, density=True, alpha=0.6, label=f"POST (n={len(r_post)})",
             color="firebrick")
axes[0].axvline(obs_pre, color="steelblue", ls="--", lw=2)
axes[0].axvline(obs_post, color="firebrick", ls="--", lw=2)
axes[0].set_xlabel("|rank-order correlation| with place-field template")
axes[0].set_ylabel("Density")
axes[0].set_title(f"Replay score distribution\nMann-Whitney p={p_mwu:.3g}")
axes[0].legend()

axes[1].bar(
    ["PRE\n(shuffle null)", "PRE\n(observed)", "POST\n(shuffle null)", "POST\n(observed)"],
    [null_means_pre.mean(), obs_pre, null_means_post.mean(), obs_post],
    yerr=[null_means_pre.std(), 0, null_means_post.std(), 0],
    color=["lightsteelblue", "steelblue", "lightcoral", "firebrick"],
)
axes[1].set_ylabel("Mean |r| across SWR events")
axes[1].set_title(f"Observed vs. label-shuffled null\n(POST p={p_post_shuffle:.3f}, "
                   f"PRE p={p_pre_shuffle:.3f})")

plt.tight_layout()
plt.savefig("fig4_replay_scores.png", dpi=150)
plt.close()
print("saved fig4_replay_scores.png")

# %% [markdown]
# ### Figure 5: example replay event
#
# One high-scoring POST-sleep SWR, showing spikes from all rightward-template place
# cells (sorted by their field location on the track) in a window around the event.
# A clean replay event shows a monotonic progression of active cells from low to high
# (or high to low) rank as the ripple unfolds.

# %%
best_idx = np.argmax(np.abs(r_post))
best_center = precomp_post[best_idx][0]
best_event = rip_post_nrem[np.argmin(np.abs(rip_post_nrem[:, 2] - best_center))]
s_win, e_win = best_event[0] - 0.05, best_event[1] + 0.05

fig, ax = plt.subplots(figsize=(8, 5.5))
participating_uids, _ = precomp_post[best_idx][1].get("R", (np.array([]), None))
for row, uid in enumerate(sorted_right):
    sp = exc_ca1[uid].restrict(nap.IntervalSet(s_win, e_win))
    color = "crimson" if uid in participating_uids else "lightgray"
    ax.vlines(sp.index.values, row - 0.4, row + 0.4, color=color, lw=1.6)
ax.axvspan(best_event[0], best_event[1], color="orange", alpha=0.2, label="Detected SWR window")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Place cell # (sorted by field location, rightward template)")
ax.set_title(f"Example POST-sleep replay event (rank-order r={r_post[best_idx]:.2f}, "
             f"t={best_center:.2f}s)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("fig5_example_replay_event.png", dpi=150)
plt.close()
print("saved fig5_example_replay_event.png")

# %% [markdown]
# ## Summary
#
# - We detected thousands of putative sharp-wave ripples on a data-driven CA1 LFP
#   channel in both PRE- and POST-run sleep, with the expected ~30-100 ms duration and
#   sparse (<1 Hz) occurrence rate during Non-REM sleep (Figures 1-2).
# - Track running produced clean, well-tiled place fields for ~90 CA1 pyramidal cells
#   in each running direction (Figure 3).
# - Rank-order correlation between the spiking sequence within each SWR and the
#   place-field template was modestly but significantly higher for POST-sleep SWRs
#   than for PRE-sleep SWRs (Mann-Whitney p reported above), consistent with
#   experience-dependent replay of the just-learned track sequence. The effect size is
#   modest and the population-level rank correlation is only slightly above a
#   cell-identity-shuffled null, which is expected: rank-order correlation on raw spike
#   times is a coarser readout of replay content than full Bayesian place-field
#   decoding, and only a minority of SWRs are expected to reflect a clean, complete
#   replay of a single trajectory. Figure 5 shows an example POST-sleep event with a
#   visually clear, ordered sequence across place cells.
print("Analysis complete.")
