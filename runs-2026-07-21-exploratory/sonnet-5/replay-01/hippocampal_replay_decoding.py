# %% [markdown]
# # Decoding Spatial Trajectories During Sharp-Wave Ripples: A Demonstration of Hippocampal Replay
#
# During quiet rest and sleep, hippocampal place cells that fired sequentially while an animal
# ran along a track are spontaneously reactivated in the same (or reversed) order during brief
# ~50-150 ms population bursts called sharp-wave ripples (SWRs). This "replay" of experience is
# widely thought to support memory consolidation. Here we demonstrate replay directly: we build
# place-field models from a rat running back and forth on a linear track, and use them to
# Bayesian-decode the position represented by the hippocampal population during each detected
# SWR occurring in the subsequent rest session. If replay is present, decoded position should
# trace out a coherent spatial trajectory within single ripples, more often than expected by
# chance.
#
# **Dataset**: DANDI Archive Dandiset
# [000044](https://dandiarchive.org/dandiset/000044) (Grosmark & Buzsaki, "Diversity in neural
# firing dynamics supports both rigid and learned hippocampal sequences"). We use session
# `sub-Achilles/sub-Achilles_ses-Achilles-10252013`, which contains bilateral CA1 silicon-probe
# recordings (128 LFP channels, 137 sorted units) together with linearized position on a 1.6 m
# linear track and a PRE-sleep / MAZE-run / POST-sleep epoch structure with brain-state scoring
# (Awake / Non-REM / REM).
#
# The file is streamed directly from S3 with `remfile` (no full download) and cached locally on
# disk so repeated reads are fast.

# %%
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy.signal import hilbert
from scipy.stats import binomtest
from tqdm import tqdm

np.random.seed(0)
FIGDIR = "figures"

# %% [markdown]
# ## 1. Load the NWB file (streaming)

# %%
S3_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/"
    "7632d81b-2819-473d-8946-34dc939e6028"
)

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# The session has three behavioral epochs (PRE-sleep, linear-track running, POST-sleep) and a
# brain-state scoring (`states`) covering the whole recording.

# %%
epochs = nwb["epochs"]
states = nwb["states"]
print(epochs)

pre_ep = epochs[epochs.label == "PREEpoch"]
maze_ep = epochs[epochs.label == "MazeEpoch"]
post_ep = epochs[epochs.label == "POSTEpoch"]
print(f"MAZE epoch duration: {maze_ep.tot_length():.1f} s")
print(f"POST epoch duration: {post_ep.tot_length():.1f} s")

# %% [markdown]
# ## 2. Inspect and visualize the raw behavioral and spiking data
#
# Position on the linear track is stored both as raw (x, y) and as a linearized coordinate
# (0-1.6 m). The linearized signal is only defined while the animal is actively traversing the
# track (it is NaN elsewhere, e.g. at the reward ports), so dropping NaNs directly recovers the
# set of running bouts.

# %%
lin_pos = nwb["1.6mLinearMazeLinearizedTimeSeries"][:, 0]
lin_pos_maze = lin_pos.restrict(maze_ep).dropna()
xy_pos = nwb["1.6mLinearMazeSpatialSeries"]

units_all = nwb["units"]
units_maze = units_all.restrict(maze_ep)
exc_maze = units_maze[units_maze.cell_type == "excitatory"]
print(f"{len(units_all)} total units ({(units_all.cell_type == 'excitatory').sum()} excitatory,"
      f" {(units_all.cell_type == 'inhibitory').sum()} interneurons)")
print(f"{len(lin_pos_maze)} valid linearized-position samples "
      f"({lin_pos_maze.time_support.tot_length():.0f} s of running, "
      f"{len(lin_pos_maze.time_support)} bouts) out of {maze_ep.tot_length():.0f} s MAZE epoch")

# %%
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

axes[0].plot(xy_pos.restrict(maze_ep).index.values, xy_pos.restrict(maze_ep).values[:, 0],
             lw=0.4, color="0.5")
axes[0].set_ylabel("raw x (m)")
axes[0].set_title("Raw and linearized position during the MAZE running epoch")

axes[1].plot(lin_pos_maze.index.values, lin_pos_maze.values, ".", ms=1.5, color="darkorange")
axes[1].set_ylabel("linearized\nposition (m)")

spk_t = exc_maze.to_tsd()
axes[2].plot(spk_t.index.values, spk_t.values, "|", color="k", ms=2, alpha=0.4)
axes[2].set_ylabel("excitatory\nunit #")
axes[2].set_xlabel("time (s)")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/01_raw_behavior_and_spikes.png", dpi=150)
plt.close()

# %% [markdown]
# ## 3. Build place fields from the MAZE running epoch
#
# We compute 1D tuning curves (occupancy-normalized firing rate vs. linearized position) for
# every excitatory unit using only the running bouts (the intervals recovered above), then
# select place cells using Skaggs spatial information (bits/spike) and a minimum spike-count
# criterion so that tuning curves used for decoding are well estimated.

# %%
N_POS_BINS = 40
tuning_curves = nap.compute_tuning_curves(
    exc_maze, lin_pos_maze, bins=N_POS_BINS, epochs=lin_pos_maze.time_support
)
mutual_info = nap.compute_mutual_information(tuning_curves)

spike_counts = np.array([len(exc_maze[i]) for i in tuning_curves.unit.values])
is_place_cell = (mutual_info["bits/spike"].values > 0.5) & (spike_counts >= 50)
place_cell_ids = tuning_curves.unit.values[is_place_cell]

print(f"{len(place_cell_ids)} / {len(exc_maze)} excitatory units pass the place-cell criteria "
      "(spatial info > 0.5 bits/spike, >= 50 spikes on the track)")

# spikes for these units across the FULL session (needed later for decoding during sleep)
place_cells = units_all[list(place_cell_ids)]
place_field_tc = tuning_curves.sel(unit=place_cell_ids)
pos_bin_centers = place_field_tc.coords[place_field_tc.dims[1]].values.astype(float)

# %%
pf = place_field_tc.values  # (n_cells, n_pos_bins)
pf_norm = pf / (pf.max(axis=1, keepdims=True) + 1e-12)
peak_bin = np.argmax(pf, axis=1)
order = np.argsort(peak_bin)

fig, ax = plt.subplots(figsize=(7, 8))
im = ax.imshow(pf_norm[order], aspect="auto", cmap="viridis",
                extent=[pos_bin_centers[0], pos_bin_centers[-1], len(order), 0])
ax.set_xlabel("linearized position (m)")
ax.set_ylabel("place cell # (sorted by field peak)")
ax.set_title(f"Place fields on the linear track (n={len(order)} cells)")
fig.colorbar(im, ax=ax, label="normalized firing rate")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/02_place_fields_sorted.png", dpi=150)
plt.close()

# %% [markdown]
# ## 4. Detect sharp-wave ripples during POST-sleep Non-REM epochs
#
# SWRs are transient (~150-250 Hz) oscillatory bursts in the LFP that occur mainly during
# Non-REM sleep and quiet wakefulness. We restrict detection to Non-REM periods within the
# POST-sleep epoch (the classic paradigm for testing "offline" replay of a just-run trajectory).
#
# The 128 LFP channels span 14 shanks. To find a channel near the CA1 pyramidal layer (which
# shows the strongest ripple-band power), we scan one representative channel per shank across a
# few short samples of Non-REM POST data, then refine within the winning shank.

# %%
nrem_post = states[states.label == "Non-REM"].intersect(post_ep)
print(f"{len(nrem_post)} Non-REM bouts in POST epoch, {nrem_post.tot_length():.0f} s total")

lfp = nwb["LFP"]
n_channels = lfp.shape[1]
n_shanks = 14
channels_per_shank = n_channels // n_shanks
shank_first_channel = [s * channels_per_shank for s in range(n_shanks)]

RIPPLE_BAND = (150.0, 250.0)


def ripple_band_power(tsd_channel):
    filt = nap.apply_bandpass_filter(tsd_channel, RIPPLE_BAND, fs=1250.0, mode="butter", order=4)
    env = np.abs(hilbert(filt.values))
    return np.mean(env ** 2)


sample_windows = [nap.IntervalSet(start=s, end=s + 60.0) for s in [20200.0, 25000.0, 30000.0]]
shank_power = np.zeros(n_shanks)
for win in tqdm(sample_windows, desc="scanning shanks for ripple power"):
    chunk = lfp[:, shank_first_channel].restrict(win.intersect(nrem_post) if
                                                  win.intersect(nrem_post).tot_length() > 0 else win)
    for s in range(n_shanks):
        shank_power[s] += ripple_band_power(chunk[:, s])
best_shank = int(np.argmax(shank_power))
print(f"strongest ripple power on shank {best_shank + 1}")

shank_channels = list(range(best_shank * channels_per_shank, (best_shank + 1) * channels_per_shank))
chan_power = np.zeros(len(shank_channels))
for win in sample_windows:
    chunk = lfp[:, shank_channels].restrict(win.intersect(nrem_post) if
                                             win.intersect(nrem_post).tot_length() > 0 else win)
    for i in range(len(shank_channels)):
        chan_power[i] += ripple_band_power(chunk[:, i])
RIPPLE_CHANNEL = shank_channels[int(np.argmax(chan_power))]
print(f"selected LFP channel {RIPPLE_CHANNEL} for ripple detection")

# %% [markdown]
# With a ripple channel selected, we band-pass filter (150-250 Hz), take the Hilbert envelope,
# z-score it within the Non-REM POST period, and detect events crossing a low threshold that
# also contain a high peak (a standard two-threshold SWR detector). Events are then merged,
# and filtered by duration.

# %%
t0 = time.time()
ripple_chan = lfp[:, RIPPLE_CHANNEL].restrict(nrem_post)
ripple_filt = nap.apply_bandpass_filter(ripple_chan, RIPPLE_BAND, fs=1250.0, mode="butter", order=4)
ripple_env = np.abs(hilbert(ripple_filt.values))
ripple_env_z = (ripple_env - ripple_env.mean()) / ripple_env.std()
ripple_env_tsd = nap.Tsd(t=ripple_filt.index.values, d=ripple_env_z,
                          time_support=ripple_filt.time_support)
print(f"loaded and filtered {ripple_chan.time_support.tot_length():.0f} s of LFP in "
      f"{time.time() - t0:.1f} s")

LOW_THRESH, HIGH_THRESH = 2.0, 5.0
low_ep = ripple_env_tsd.threshold(LOW_THRESH, method="above").time_support
high_times = ripple_env_tsd.threshold(HIGH_THRESH, method="above").index.values

keep = [(s, e) for s, e in zip(low_ep.start, low_ep.end)
        if np.any((high_times >= s) & (high_times <= e))]
ripple_ep = nap.IntervalSet(start=[k[0] for k in keep], end=[k[1] for k in keep])
ripple_ep = (ripple_ep
             .merge_close_intervals(0.02)
             .drop_short_intervals(0.02)
             .drop_long_intervals(0.4))

ripple_durations_ms = (ripple_ep.end - ripple_ep.start) * 1000
print(f"{len(ripple_ep)} candidate SWR events detected "
      f"(median duration {np.median(ripple_durations_ms):.1f} ms, "
      f"rate {len(ripple_ep) / nrem_post.tot_length() * 60:.2f} events/min of Non-REM sleep)")

# %%
example_win = nap.IntervalSet(start=ripple_ep.start[10] - 1.0, end=ripple_ep.start[10] + 2.0)
raw_snip = lfp[:, RIPPLE_CHANNEL].restrict(example_win)
filt_snip = nap.apply_bandpass_filter(raw_snip, RIPPLE_BAND, fs=1250.0, mode="butter", order=4)
env_snip = np.abs(hilbert(filt_snip.values))
ripples_in_win = ripple_ep.intersect(example_win)

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
axes[0].plot(raw_snip.index.values, raw_snip.values, color="0.3", lw=0.6)
axes[0].set_ylabel("raw LFP\n(a.u.)")
axes[0].set_title(f"Example ripple detection window, channel {RIPPLE_CHANNEL}")
axes[1].plot(filt_snip.index.values, filt_snip.values, color="steelblue", lw=0.6)
axes[1].set_ylabel("150-250 Hz\nfiltered")
axes[2].plot(filt_snip.index.values, env_snip, color="darkred", lw=0.8)
axes[2].axhline(ripple_env_tsd.values.std() * HIGH_THRESH + ripple_env_tsd.values.mean(),
                 color="gray", ls="--", lw=0.8)
for s, e in zip(ripples_in_win.start, ripples_in_win.end):
    for ax in axes:
        ax.axvspan(s, e, color="gold", alpha=0.4)
axes[2].set_ylabel("envelope")
axes[2].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/03_ripple_detection_example.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.hist(ripple_durations_ms, bins=40, color="steelblue", edgecolor="white")
ax.set_xlabel("ripple duration (ms)")
ax.set_ylabel("count")
ax.set_title(f"Detected SWR durations (n={len(ripple_ep)})")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/04_ripple_duration_histogram.png", dpi=150)
plt.close()

# %% [markdown]
# ## 5. Bayesian decoding of position during each ripple
#
# For every SWR we count place-cell spikes in 10 ms bins and use the MAZE-derived place fields
# to compute the posterior probability of position at each bin (`pynapple.decode_bayes`,
# memoryless Bayesian decoder, uniform prior). We restrict the *quantitative replay analysis* to
# events with (a) enough population activity to be decodable and (b) enough 10 ms bins (>=5, i.e.
# >=50 ms duration) for a meaningful within-event trajectory; shorter/quieter events are decoded
# too for visualization but excluded from the significance test because a handful of bins gives
# an unstable trajectory estimate.

# %%
BIN_SIZE = 0.01  # s
MIN_BINS_FOR_SCORING = 5
MIN_TOTAL_SPIKES = 5
MIN_ACTIVE_CELLS = 3

spike_counts_per_ripple = place_cells.restrict(ripple_ep).count(ep=ripple_ep)
n_active_cells = (spike_counts_per_ripple > 0).sum(axis=1).values
total_spikes = spike_counts_per_ripple.values.sum(axis=1)
durations = ripple_ep.end - ripple_ep.start

decodable = (total_spikes >= MIN_TOTAL_SPIKES) & (n_active_cells >= MIN_ACTIVE_CELLS)
scored = decodable & (durations >= BIN_SIZE * MIN_BINS_FOR_SCORING)
print(f"{decodable.sum()} / {len(ripple_ep)} ripples have enough spikes to decode")
print(f"{scored.sum()} / {len(ripple_ep)} ripples also have >= {MIN_BINS_FOR_SCORING} bins "
      "and are used for the replay-significance test")

scored_ep = nap.IntervalSet(start=ripple_ep.start[scored], end=ripple_ep.end[scored])
decoded_pos, posterior = nap.decode_bayes(
    place_field_tc, place_cells, scored_ep, bin_size=BIN_SIZE, time_units="s"
)

t_bins = decoded_pos.index.values
event_idx = np.clip(np.searchsorted(scored_ep.start, t_bins, side="right") - 1,
                     0, len(scored_ep) - 1)
posterior_vals = posterior.values
assert np.all((t_bins >= scored_ep.start[event_idx]) & (t_bins <= scored_ep.end[event_idx]))

# %% [markdown]
# ## 6. Quantifying replay: weighted correlation vs. a time-bin shuffle null
#
# For each event we compute the **weighted correlation** between time and decoded position,
# using the full posterior probability matrix as weights (Davidson et al. 2009; this is the
# standard "replay score" in the literature). To test significance, for each event we build a
# null distribution by randomly permuting the order of that event's own posterior time bins
# (500 permutations) and ask how often the shuffled |correlation| matches or exceeds the observed
# value. This directly tests whether the decoded positions unfold in a coherent temporal sequence,
# independent of how spatially precise any single bin's decode is.

# %%
def weighted_corr(posterior_event, t_idx, pos_bins):
    w = posterior_event
    wsum = w.sum()
    if wsum <= 0:
        return np.nan
    t_mean = (w.sum(axis=1) * t_idx).sum() / wsum
    x_mean = (w.sum(axis=0) * pos_bins).sum() / wsum
    dt = t_idx[:, None] - t_mean
    dx = pos_bins[None, :] - x_mean
    cov = (w * dt * dx).sum() / wsum
    var_t = (w * dt ** 2).sum() / wsum
    var_x = (w * dx ** 2).sum() / wsum
    if var_t <= 0 or var_x <= 0:
        return np.nan
    return cov / np.sqrt(var_t * var_x)


rng = np.random.default_rng(0)
N_PERM = 500
n_events = len(scored_ep)
obs_r = np.full(n_events, np.nan)
pvals = np.full(n_events, np.nan)
n_bins_per_event = np.zeros(n_events, dtype=int)

for ev in tqdm(range(n_events), desc="scoring replay events"):
    m = event_idx == ev
    n = int(m.sum())
    n_bins_per_event[ev] = n
    if n < MIN_BINS_FOR_SCORING:
        continue
    Pe = posterior_vals[m]
    t_idx = np.arange(n)
    r_obs = weighted_corr(Pe, t_idx, pos_bin_centers)
    obs_r[ev] = r_obs
    null_r = np.empty(N_PERM)
    for p in range(N_PERM):
        null_r[p] = weighted_corr(Pe[rng.permutation(n)], t_idx, pos_bin_centers)
    pvals[ev] = np.mean(np.abs(null_r) >= np.abs(r_obs))

valid = ~np.isnan(obs_r)
n_valid = int(valid.sum())
n_sig = int((pvals[valid] < 0.05).sum())
btest = binomtest(n_sig, n_valid, 0.05, alternative="greater")

print(f"{n_valid} scored ripple events")
print(f"median |weighted correlation| = {np.nanmedian(np.abs(obs_r[valid])):.3f}")
print(f"{n_sig} / {n_valid} events ({100 * n_sig / n_valid:.1f}%) have p < 0.05 vs. their own "
      f"time-bin shuffle (5% expected by chance)")
print(f"binomial test for excess of significant events: p = {btest.pvalue:.2e}")

# %% [markdown]
# ## 7. Example decoded replay trajectories
#
# We visualize the most significant events: raw + ripple-band-filtered LFP, a spike raster of
# the place cells (sorted by their track position, matching the panel in Section 3), and the
# decoded posterior probability over position with the maximum-a-posteriori trajectory overlaid.

# %%
order_by_sig = np.where(valid)[0][np.argsort(pvals[valid])]
example_events = order_by_sig[:4]

place_cell_order = place_cell_ids[order]  # sorted by field peak, same order as fig 2

fig, axes = plt.subplots(4, 3, figsize=(13, 12))
for row, ev in enumerate(example_events):
    ev_start, ev_end = scored_ep.start[ev], scored_ep.end[ev]
    pad = 0.05
    win = nap.IntervalSet(start=ev_start - pad, end=ev_end + pad)

    raw = lfp[:, RIPPLE_CHANNEL].restrict(win)
    filt = nap.apply_bandpass_filter(raw, RIPPLE_BAND, fs=1250.0, mode="butter", order=4)
    ax = axes[row, 0]
    ax.plot(raw.index.values - ev_start, raw.values, color="0.3", lw=0.7)
    ax.plot(filt.index.values - ev_start, filt.values, color="steelblue", lw=0.7)
    ax.axvspan(0, ev_end - ev_start, color="gold", alpha=0.3)
    ax.set_ylabel(f"event {ev}\nLFP")
    if row == 0:
        ax.set_title("raw + ripple-band LFP")
    if row == 3:
        ax.set_xlabel("time from ripple onset (s)")

    ax = axes[row, 1]
    for i, uid in enumerate(place_cell_order):
        sp = place_cells[uid].restrict(win).index.values
        ax.plot(sp - ev_start, np.full_like(sp, i), "|", color="k", ms=6)
    ax.axvspan(0, ev_end - ev_start, color="gold", alpha=0.3)
    ax.set_ylabel("place cell\n(track order)")
    if row == 0:
        ax.set_title("spike raster")
    if row == 3:
        ax.set_xlabel("time from ripple onset (s)")

    m = event_idx == ev
    Pe = posterior_vals[m]
    te = t_bins[m] - ev_start
    ax = axes[row, 2]
    im = ax.pcolormesh(np.append(te, te[-1] + BIN_SIZE) - BIN_SIZE / 2, pos_bin_centers, Pe.T,
                        shading="flat", cmap="magma")
    map_pos = pos_bin_centers[np.argmax(Pe, axis=1)]
    ax.plot(te, map_pos, color="cyan", lw=1.2, marker="o", ms=3)
    r_txt = f"r={obs_r[ev]:.2f}, p={pvals[ev]:.3f}"
    ax.set_ylabel("decoded\nposition (m)")
    ax.set_title(r_txt if row > 0 else f"posterior + MAP trajectory\n{r_txt}")
    if row == 3:
        ax.set_xlabel("time from ripple onset (s)")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/05_example_replay_trajectories.png", dpi=150)
plt.close()

# %% [markdown]
# ## 8. Population summary: replay is more sequential than chance

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].hist(np.abs(obs_r[valid]), bins=25, color="steelblue", edgecolor="white")
axes[0].axvline(np.nanmedian(np.abs(obs_r[valid])), color="darkred", ls="--",
                 label="median")
axes[0].set_xlabel("|weighted correlation|")
axes[0].set_ylabel("# ripple events")
axes[0].set_title("Observed replay scores")
axes[0].legend()

axes[1].hist(pvals[valid], bins=20, range=(0, 1), color="seagreen", edgecolor="white")
axes[1].axhline(n_valid / 20, color="gray", ls="--", label="uniform (chance) level")
axes[1].set_xlabel("event p-value (vs. own time-bin shuffle)")
axes[1].set_ylabel("# ripple events")
axes[1].set_title("P-value distribution")
axes[1].legend()

bars = axes[2].bar(["expected\n(chance)", "observed"],
                    [5.0, 100 * n_sig / n_valid],
                    color=["gray", "crimson"])
axes[2].set_ylabel("% events with p < 0.05")
axes[2].set_title(f"Excess of significant replay\nbinomial p = {btest.pvalue:.1e}")
for b in bars:
    axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                 f"{b.get_height():.1f}%", ha="center")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/06_replay_significance_summary.png", dpi=150)
plt.close()

print("Done. Figures written to", FIGDIR)
