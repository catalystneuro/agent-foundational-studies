# %% [markdown]
# # Hippocampal Replay: Decoding Spatial Trajectories During Sharp-Wave Ripples
#
# This notebook demonstrates hippocampal replay by decoding the spatial trajectory
# represented in CA1 population activity during sharp-wave ripples (SWRs) recorded
# after track running.
#
# **Dataset**: DANDI Dandiset [000044](https://dandiarchive.org/dandiset/000044)
# ("Diversity in neural firing dynamics supports both rigid and learned hippocampal
# sequences", Grosmark & Buzsaki, Science 2016). Session `sub-Buddy_ses-Buddy-06272013`:
# a Long-Evans rat runs on a 1.6 m linear track for reward at both ends (`MazeEpoch`),
# bracketed by a long home-cage sleep/rest recording before (`PREEpoch`) and after
# (`POSTEpoch`) the run. Bilateral CA1 silicon-probe recordings (128 channels, 16
# shanks) yield 68 sorted units, and the LFP (1250 Hz) plus a sleep-scoring table
# (`Awake` / `Non-REM`) are provided.
#
# **Approach**:
# 1. Compute CA1 place fields from spiking activity during track running.
# 2. Detect sharp-wave ripples from CA1 LFP during Non-REM sleep after (`POST`) and
#    before (`PRE`) the run.
# 3. Bayesian-decode the position represented by the spiking activity inside each
#    ripple, using the place fields learned during running.
# 4. Score each ripple for sequential structure (a "replay" of a spatial trajectory)
#    with a weighted correlation between decoded position and time, tested against
#    a shuffle null. If CA1 replays the track, POST ripples should show far more
#    significant sequential decoding than PRE ripples (no track experience yet).

# %%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
from tqdm import tqdm

np.random.seed(0)  # only used for the shuffle null distribution

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10

# %% [markdown]
# ## 1. Load the NWB file (streaming from DANDI S3)

# %%
S3_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/49f/95f/"
    "49f95f2a-1ae4-4720-85a3-899847616078"
)  # sub-Buddy_ses-Buddy-06272013_behavior+ecephys.nwb (dandiset 000044)

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %%
units = nwb["units"]
epochs = nwb["epochs"]
states = nwb["states"]
pos2d = nwb["1.6mLinearMazeSpatialSeries"]

maze_ep = epochs[[l == "MazeEpoch" for l in epochs.label]]
pre_ep = epochs[[l == "PREEpoch" for l in epochs.label]]
post_ep = epochs[[l == "POSTEpoch" for l in epochs.label]]
print("MAZE epoch:", maze_ep)
print("PRE epoch duration (s):", pre_ep.tot_length())
print("POST epoch duration (s):", post_ep.tot_length())

nrem = states[[l == "Non-REM" for l in states.label]]
nrem_pre = nrem.intersect(pre_ep)
nrem_post = nrem.intersect(post_ep)
print(f"Non-REM in PRE:  {nrem_pre.tot_length():.0f} s over {len(nrem_pre)} bouts")
print(f"Non-REM in POST: {nrem_post.tot_length():.0f} s over {len(nrem_post)} bouts")

# %% [markdown]
# ## 2. Behavior: position, speed, and running epochs
#
# The dataset's own `LinearizedTimeSeries` is valid for only ~7% of samples, but the
# raw x-coordinate of the tracked position correlates with it at r > 0.9999 and is
# valid for 97% of samples (the track is a straight line, so x alone is an accurate
# linear position). We use x directly, restricted to the physical track extent
# (0 to 1.6 m) to exclude the animal lingering at the reward wells beyond the ends.
#
# Restricting to the track extent leaves temporal gaps every time the animal visits
# a reward well (up to ~50 s). Naively differencing position/time across those gaps
# creates spurious "teleport" position/speed samples right at the track ends, which
# badly contaminates place fields (verified during prototyping: it produced a single
# artifactual peak at x=0 shared by nearly every cell). We therefore explicitly
# split the on-track samples into contiguous bouts (breaking wherever the time gap
# between consecutive on-track samples exceeds 0.1 s) and compute speed and the
# `Tsd` time support per bout, so no computation ever bridges a reward-well visit.

# %%
x = pos2d.loc["x"]
y = pos2d.loc["y"]
valid = ~np.isnan(x.values)
t_pos = np.asarray(x.index.values)[valid]
x_pos = np.asarray(x.values)[valid]
y_pos = np.asarray(y.values)[valid]

on_track = (x_pos >= 0.0) & (x_pos <= 1.6)
t_track, x_track, y_track = t_pos[on_track], x_pos[on_track], y_pos[on_track]

GAP_THRESH = 0.1  # s; breaks between contiguous on-track bouts
dt_all = np.diff(t_track)
gap_idx = np.where(dt_all > GAP_THRESH)[0]
bout_start_idx = np.concatenate(([0], gap_idx + 1))
bout_end_idx = np.concatenate((gap_idx, [len(t_track) - 1]))
track_support = nap.IntervalSet(start=t_track[bout_start_idx], end=t_track[bout_end_idx])
linpos = nap.Tsd(t=t_track, d=x_track, time_support=track_support)
print(f"On-track bouts: {len(track_support)}, total {track_support.tot_length():.0f} s")

speed_t_list, speed_v_list = [], []
for i0, i1 in zip(bout_start_idx, bout_end_idx):
    if i1 <= i0:
        continue
    tt, xx, yy = t_track[i0 : i1 + 1], x_track[i0 : i1 + 1], y_track[i0 : i1 + 1]
    ddt = np.diff(tt)
    v = np.sqrt(np.diff(xx) ** 2 + np.diff(yy) ** 2) / ddt
    speed_t_list.append(tt[:-1] + ddt / 2)
    speed_v_list.append(v)
speed = nap.Tsd(
    t=np.concatenate(speed_t_list), d=np.concatenate(speed_v_list), time_support=track_support
)
speed_smooth = speed.smooth(std=0.25, windowsize=1.0, time_units="s")

SPEED_THRESH = 0.05  # m/s
run_ep = speed_smooth.threshold(SPEED_THRESH, method="above").time_support.intersect(
    maze_ep
)
print(
    f"Running epochs: {run_ep.tot_length():.0f} s of {maze_ep.tot_length():.0f} s "
    f"MAZE epoch ({100 * run_ep.tot_length() / maze_ep.tot_length():.0f}%)"
)

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
axes[0].plot(t_pos, x_pos, lw=0.4, color="0.6", label="raw x (incl. reward zones)")
axes[0].plot(t_track, x_track, lw=0.5, color="tab:blue", label="on-track x")
axes[0].set_ylabel("Linear position (m)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title("Behavior during MazeEpoch")

axes[1].plot(speed_smooth.index.values, speed_smooth.values, lw=0.4, color="0.3")
axes[1].axhline(SPEED_THRESH, color="tab:red", ls="--", lw=1, label="run threshold")
axes[1].set_ylabel("Speed (m/s)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylim(0, 1.0)
axes[1].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("fig01_behavior_trajectory.png")
plt.close(fig)

# %% [markdown]
# ## 3. Place fields from CA1 pyramidal cells during running
#
# Tuning curves (occupancy-normalized firing rate vs. linear position) are computed
# for left-hemisphere CA1 units classified as excitatory (putative pyramidal cells),
# restricted to running epochs. Spatial information (Skaggs et al., 1993) is used to
# select cells with reliable place fields.

# %%
units_df = units.metadata
ca1_exc_ids = units_df[
    (units_df.location == "lCA1") & (units_df.cell_type == "excitatory")
].index.values
ca1_exc = units[ca1_exc_ids]
print(f"Candidate lCA1 excitatory units: {len(ca1_exc_ids)}")

N_POS_BINS = 40
tc_all = nap.compute_tuning_curves(
    ca1_exc, linpos, bins=N_POS_BINS, range=[(0.0, 1.6)], epochs=run_ep
)
info = nap.compute_mutual_information(tc_all)

INFO_THRESH = 0.3  # bits/spike
RATE_THRESH = 0.1  # Hz mean rate over running epochs, avoid near-silent units
place_cell_ids = info[
    (info["bits/spike"] > INFO_THRESH) & (tc_all.attrs["rates"] > RATE_THRESH)
].index.values
print(f"Place cells selected (info > {INFO_THRESH} bits/spike): {len(place_cell_ids)}")

place_group = ca1_exc[place_cell_ids]
tuning_curves = tc_all.sel(unit=place_cell_ids)
pos_bin_centers = tuning_curves.coords["0"].values

# %%
# Sort place fields by peak location for the population map
tc_vals = tuning_curves.values  # (unit, pos_bin)
peak_bin = np.argmax(tc_vals, axis=1)
order = np.argsort(peak_bin)
tc_norm = tc_vals / (tc_vals.max(axis=1, keepdims=True) + 1e-12)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.3]})
im = axes[0].imshow(
    tc_norm[order],
    aspect="auto",
    cmap="viridis",
    extent=[pos_bin_centers[0], pos_bin_centers[-1], len(order), 0],
)
axes[0].set_xlabel("Position on track (m)")
axes[0].set_ylabel("Place cell (sorted by field peak)")
axes[0].set_title(f"Place field map (n={len(order)} cells)")
plt.colorbar(im, ax=axes[0], label="Normalized rate", fraction=0.046)

example_ids = place_cell_ids[order[:: max(1, len(order) // 6)][:6]]
for uid in example_ids:
    row = np.where(place_cell_ids == uid)[0][0]
    axes[1].plot(
        pos_bin_centers, tc_vals[row], lw=1.5, label=f"unit {uid}"
    )
axes[1].set_xlabel("Position on track (m)")
axes[1].set_ylabel("Firing rate (Hz)")
axes[1].set_title("Example place fields")
axes[1].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig("fig02_place_fields.png")
plt.close(fig)

# %% [markdown]
# ## 4. Sharp-wave ripple detection
#
# The reference LFP channel was chosen by comparing ripple-band (150-250 Hz) envelope
# power across one channel per shank (16 candidates) on a Non-REM snippet; channel 20
# (shank 3, left hemisphere probe) had the largest and most consistent ripple power
# across independent Non-REM bouts, and is used as the ripple-detection channel for
# the whole session. Events are detected on the z-scored ripple-band envelope
# (`pynapple.detect_oscillatory_events`), separately within Non-REM bouts of the PRE
# and POST epochs.

# %%
RIPPLE_CHANNEL = 20
FS_LFP = 1250.0
lfp_dset = h5_file["processing"]["ecephys"]["LFP"]["LFP"]["data"]


def load_lfp_channel(epoch_set, channel, fs=FS_LFP, dset=lfp_dset):
    """Load one LFP channel for each interval in epoch_set and concatenate as a Tsd."""
    ts, ds = [], []
    for s, e in epoch_set.values:
        i0, i1 = int(s * fs), int(e * fs)
        seg = dset[i0:i1, channel].astype(np.float64)
        ts.append(i0 / fs + np.arange(len(seg)) / fs)
        ds.append(seg)
    return nap.Tsd(t=np.concatenate(ts), d=np.concatenate(ds))


def detect_ripples(epoch_set, channel, fs=FS_LFP):
    lfp_sig = load_lfp_channel(epoch_set, channel, fs)
    events = []
    for s, e in tqdm(epoch_set.values, desc="ripple detection per bout"):
        bout_ep = nap.IntervalSet(s, e)
        sig = lfp_sig.restrict(bout_ep)
        if len(sig) < 200:
            continue
        ev = nap.detect_oscillatory_events(
            sig,
            epochs=bout_ep,
            frequency_band=(150.0, 250.0),
            threshold_band=(3.0, 10.0),
            duration_band=(0.02, 0.3),
            min_interval=0.03,
            fs=fs,
        )
        if len(ev) > 0:
            events.append(ev)
    if len(events) == 0:
        return nap.IntervalSet(start=[], end=[])
    starts = np.concatenate([ev.start for ev in events])
    ends = np.concatenate([ev.end for ev in events])
    power = np.concatenate([ev.power for ev in events])
    amplitude = np.concatenate([ev.amplitude for ev in events])
    peak_time = np.concatenate([ev.peak_time for ev in events])
    order = np.argsort(starts)
    return nap.IntervalSet(
        start=starts[order],
        end=ends[order],
        metadata={
            "power": power[order],
            "amplitude": amplitude[order],
            "peak_time": peak_time[order],
        },
    )


ripples_post = detect_ripples(nrem_post, RIPPLE_CHANNEL)
ripples_pre = detect_ripples(nrem_pre, RIPPLE_CHANNEL)
print(f"POST: {len(ripples_post)} candidate ripples over {nrem_post.tot_length():.0f} s NREM")
print(f"PRE:  {len(ripples_pre)} candidate ripples over {nrem_pre.tot_length():.0f} s NREM")
print(f"POST ripple rate: {len(ripples_post) / nrem_post.tot_length():.3f} Hz")
print(f"PRE  ripple rate: {len(ripples_pre) / nrem_pre.tot_length():.3f} Hz")

# %%
# Example raw LFP trace with a detected ripple highlighted
example_ripple = ripples_post.values[np.argmax(ripples_post.amplitude)]
win_s, win_e = example_ripple[0] - 0.25, example_ripple[1] + 0.25
example_ep = nap.IntervalSet(win_s, win_e)
raw_sig = load_lfp_channel(example_ep, RIPPLE_CHANNEL)
filt_sig = nap.apply_bandpass_filter(raw_sig, (150.0, 250.0), fs=FS_LFP)
env_sig = nap.compute_hilbert_envelope(filt_sig)
# For display only: remove slow (<1 Hz) drift so the ripple is visible on the raw trace
raw_display = nap.apply_highpass_filter(raw_sig, 1.0, fs=FS_LFP)

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
axes[0].plot(raw_display.index.values, raw_display.values, color="0.2", lw=0.8)
axes[0].set_ylabel("Raw LFP,\n>1 Hz (a.u.)")
axes[0].set_title(f"Example detected sharp-wave ripple (channel {RIPPLE_CHANNEL})")
axes[1].plot(filt_sig.index.values, filt_sig.values, color="tab:purple", lw=0.8)
axes[1].set_ylabel("150-250 Hz")
axes[2].plot(env_sig.index.values, env_sig.values, color="tab:orange", lw=1)
axes[2].set_ylabel("Ripple envelope")
axes[2].set_xlabel("Time (s)")
for ax in axes:
    ax.axvspan(example_ripple[0], example_ripple[1], color="tab:red", alpha=0.15)
fig.tight_layout()
fig.savefig("fig03_example_ripple_trace.png")
plt.close(fig)

# %%
durations_post = (ripples_post.end - ripples_post.start) * 1000
durations_pre = (ripples_pre.end - ripples_pre.start) * 1000

fig, ax = plt.subplots(figsize=(6, 4))
bins = np.linspace(0, 300, 31)
ax.hist(durations_pre, bins=bins, alpha=0.5, label=f"PRE (n={len(durations_pre)})", color="0.5")
ax.hist(durations_post, bins=bins, alpha=0.6, label=f"POST (n={len(durations_post)})", color="tab:red")
ax.set_xlabel("Ripple duration (ms)")
ax.set_ylabel("Count")
ax.set_title("Detected ripple durations")
ax.legend()
fig.tight_layout()
fig.savefig("fig04_ripple_duration_distribution.png")
plt.close(fig)

# %% [markdown]
# ## 5. Bayesian decoding of position within ripple events
#
# For each detected ripple, CA1 place-cell spike counts in 20 ms bins are decoded
# into a posterior probability distribution over track position, using the place
# fields learned during running (memoryless Bayesian decoder,
# `pynapple.decode_bayes`, uniform prior). Events with too few active place cells or
# too short a duration to contain more than one time bin are excluded, since a
# sequential trajectory cannot be assessed from a single bin.

# %%
DECODE_BIN = 0.02  # s
MIN_ACTIVE_CELLS = 2
MIN_TOTAL_SPIKES = 5
MIN_BINS = 4  # >= 80 ms of ripple, standard for sequence scoring (Karlsson & Frank, 2009)


def filter_events_by_spiking(events, spike_group, min_active=MIN_ACTIVE_CELLS, min_spikes=MIN_TOTAL_SPIKES):
    keep = np.zeros(len(events), dtype=bool)
    n_active = np.zeros(len(events), dtype=int)
    n_spikes = np.zeros(len(events), dtype=int)
    for i, (s, e) in enumerate(events.values):
        ev_ep = nap.IntervalSet(s, e)
        counts = np.array([len(ts.restrict(ev_ep)) for ts in spike_group.values()])
        n_active[i] = np.sum(counts > 0)
        n_spikes[i] = counts.sum()
        keep[i] = (
            (n_active[i] >= min_active)
            and (n_spikes[i] >= min_spikes)
            and (e - s) >= MIN_BINS * DECODE_BIN
        )
    return keep, n_active, n_spikes


keep_post, nact_post, nspk_post = filter_events_by_spiking(ripples_post, place_group)
keep_pre, nact_pre, nspk_pre = filter_events_by_spiking(ripples_pre, place_group)

ripples_post_f = ripples_post[keep_post]
ripples_pre_f = ripples_pre[keep_pre]
print(f"POST ripples retained for decoding: {keep_post.sum()} / {len(ripples_post)}")
print(f"PRE  ripples retained for decoding: {keep_pre.sum()} / {len(ripples_pre)}")

# %%
def weighted_correlation(P, pos_centers):
    """Weighted Pearson correlation between decoded position and time-bin index."""
    T, n_bins = P.shape
    if T < 2:
        return np.nan
    tt = np.arange(T)
    wsum = P.sum()
    if wsum <= 0:
        return np.nan
    row_w = P.sum(axis=1)
    col_w = P.sum(axis=0)
    t_mean = (row_w * tt).sum() / wsum
    x_mean = (col_w * pos_centers).sum() / wsum
    cov = (P * (tt[:, None] - t_mean) * (pos_centers[None, :] - x_mean)).sum() / wsum
    var_t = (row_w * (tt - t_mean) ** 2).sum() / wsum
    var_x = (col_w * (pos_centers - x_mean) ** 2).sum() / wsum
    if var_t <= 0 or var_x <= 0:
        return np.nan
    return cov / np.sqrt(var_t * var_x)


def column_cycle_shuffle_scores(P, pos_centers, n_shuffles=500, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    T, n_bins = P.shape
    scores = np.empty(n_shuffles)
    for k in range(n_shuffles):
        shifts = rng.integers(0, n_bins, size=T)
        P_shuf = np.array([np.roll(P[r], shifts[r]) for r in range(T)])
        scores[k] = weighted_correlation(P_shuf, pos_centers)
    return scores


def decode_and_score(events, place_group, tuning_curves, pos_centers, bin_size=DECODE_BIN, n_shuffles=500, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    if len(events) == 0:
        return pd.DataFrame(results)
    _, posterior = nap.decode_bayes(
        tuning_curves, place_group, events, bin_size=bin_size, time_units="s", uniform_prior=True
    )
    for s, e in tqdm(events.values, desc="decoding + shuffle test"):
        ev_ep = nap.IntervalSet(s, e)
        post_sub = posterior.restrict(ev_ep)
        P = np.asarray(post_sub.values)
        if P.shape[0] < 2:
            continue
        obs_score = weighted_correlation(P, pos_centers)
        if np.isnan(obs_score):
            continue
        shuf_scores = column_cycle_shuffle_scores(P, pos_centers, n_shuffles=n_shuffles, rng=rng)
        pval = (np.sum(np.abs(shuf_scores) >= np.abs(obs_score)) + 1) / (n_shuffles + 1)
        results.append(
            dict(
                start=s, end=e, duration=e - s, n_bins=P.shape[0],
                score=obs_score, pval=pval, significant=pval < 0.05,
            )
        )
    return pd.DataFrame(results)


results_post = decode_and_score(ripples_post_f, place_group, tuning_curves, pos_bin_centers)
results_pre = decode_and_score(ripples_pre_f, place_group, tuning_curves, pos_bin_centers)

frac_sig_post = results_post["significant"].mean() if len(results_post) else np.nan
frac_sig_pre = results_pre["significant"].mean() if len(results_pre) else np.nan
print(f"POST: {len(results_post)} events decoded, {frac_sig_post * 100:.1f}% significant replay (p<0.05)")
print(f"PRE:  {len(results_pre)} events decoded, {frac_sig_pre * 100:.1f}% significant replay (p<0.05)")

results_post.to_csv("replay_scores_post.csv", index=False)
results_pre.to_csv("replay_scores_pre.csv", index=False)

# %% [markdown]
# ## 6. Visualizing decoded replay trajectories

# %%
def plot_event_posterior(ax, event_row, place_group, tuning_curves, pos_centers, bin_size=DECODE_BIN):
    s, e = event_row["start"], event_row["end"]
    ev_ep = nap.IntervalSet(s, e)
    _, posterior = nap.decode_bayes(
        tuning_curves, place_group, ev_ep, bin_size=bin_size, time_units="s", uniform_prior=True
    )
    P = np.asarray(posterior.values)
    t_rel = (np.asarray(posterior.index.values) - s) * 1000
    im = ax.imshow(
        P.T, aspect="auto", origin="lower", cmap="hot",
        extent=[0, (e - s) * 1000, pos_centers[0], pos_centers[-1]],
    )
    ax.set_title(f"score={event_row['score']:.2f}, p={event_row['pval']:.3f}", fontsize=9)
    ax.set_xlabel("Time in event (ms)")
    return im


sig_events = results_post[results_post["significant"]].copy()
sig_events["abs_score"] = sig_events["score"].abs()
sig_events = sig_events.sort_values("abs_score", ascending=False)

n_examples = min(6, len(sig_events))
if n_examples > 0:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes = axes.flatten()
    for i in range(6):
        if i < n_examples:
            im = plot_event_posterior(axes[i], sig_events.iloc[i], place_group, tuning_curves, pos_bin_centers)
            if i % 3 == 0:
                axes[i].set_ylabel("Decoded position (m)")
        else:
            axes[i].axis("off")
    fig.suptitle("Example significant replay events (POST, Non-REM sharp-wave ripples)")
    fig.tight_layout()
    fig.savefig("fig05_example_replay_events.png")
    plt.close(fig)
else:
    print("No significant replay events found to plot examples for.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
bins = np.linspace(-1, 1, 41)
if len(results_post):
    axes[0].hist(results_post["score"], bins=bins, color="tab:red", alpha=0.7, label="POST observed", density=True)
if len(results_pre):
    axes[0].hist(results_pre["score"], bins=bins, color="0.5", alpha=0.6, label="PRE observed", density=True)
axes[0].set_xlabel("Weighted correlation (replay score)")
axes[0].set_ylabel("Density")
axes[0].set_title("Observed replay scores: POST vs PRE")
axes[0].legend(fontsize=8)

labels = ["PRE\n(no track exp.)", "POST\n(after running)"]
fracs = [frac_sig_pre * 100 if not np.isnan(frac_sig_pre) else 0, frac_sig_post * 100 if not np.isnan(frac_sig_post) else 0]
ns = [len(results_pre), len(results_post)]
bars = axes[1].bar(labels, fracs, color=["0.5", "tab:red"])
axes[1].set_ylim(0, max(fracs) * 1.35 + 1)
for b, n in zip(bars, ns):
    axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"n={n}", ha="center", fontsize=9)
axes[1].axhline(5, color="k", ls="--", lw=1, label="chance (5%)")
axes[1].set_ylabel("% events with significant\nsequential replay (p<0.05)")
axes[1].set_title("Fraction of significant replay events")
axes[1].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("fig06_replay_score_summary.png")
plt.close(fig)

# %% [markdown]
# ## 7. Summary
#
# Place fields tiling the 1.6 m track were identified in CA1 pyramidal cells during
# running. During Non-REM sleep after running (POST), a larger fraction of
# sharp-wave ripples show a significant sequential relationship between decoded
# position and time within the event (compared to a within-event shuffle null) than
# ripples detected during Non-REM sleep before the animal had ever run the track
# (PRE): both rates lie above the 5% expected by chance, but POST is elevated over
# PRE. A non-zero PRE rate is expected for this dataset: Grosmark & Buzsaki (2016),
# who collected this recording, report that a subset of CA1 sequences are "rigid"
# and pre-exist experience, while a second, "flexible" subset only appears after
# the animal has run the track. The POST-over-PRE elevation we recover here is
# consistent with that experience-dependent (flexible) component being layered on
# top of pre-existing sequence structure, and is the classic signature of
# hippocampal replay: CA1 population activity during SWRs reconstructs spatial
# trajectories related to the track that was just explored.

# %%
print("=== Summary ===")
print(f"Place cells used for decoding: {len(place_cell_ids)}")
print(f"POST candidate ripples: {len(ripples_post)}, decoded: {len(results_post)}, "
      f"significant: {int(results_post['significant'].sum()) if len(results_post) else 0} "
      f"({frac_sig_post * 100:.1f}%)")
print(f"PRE candidate ripples: {len(ripples_pre)}, decoded: {len(results_pre)}, "
      f"significant: {int(results_pre['significant'].sum()) if len(results_pre) else 0} "
      f"({frac_sig_pre * 100:.1f}%)")

io.close()
