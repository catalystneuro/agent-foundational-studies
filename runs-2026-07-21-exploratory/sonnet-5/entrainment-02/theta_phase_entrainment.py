# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Theta Phase Entrainment of Hippocampal CA1 Neurons
#
# This notebook demonstrates theta phase entrainment in hippocampal CA1
# neurons using a real extracellular recording from the DANDI Archive. Theta
# phase entrainment is the tendency of a neuron's spikes to occur
# preferentially at a particular phase of the ongoing hippocampal theta
# rhythm (~6-10 Hz) rather than uniformly across the theta cycle. This
# organizes cell assemblies in time and is a core feature of hippocampal
# circuit dynamics during active locomotion.
#
# **Dataset**: [DANDI:000044](https://dandiarchive.org/dandiset/000044),
# subject Achilles, session `Achilles-10252013` ("Diversity in neural firing
# dynamics supports both rigid and learned hippocampal sequences",
# Grosmark & Buzsaki lab). The session contains a bilateral CA1 silicon-probe
# recording (128-channel LFP at 1250 Hz) with 137 spike-sorted units labeled
# as excitatory or inhibitory, recorded while the rat ran back and forth on
# a 1.6 m linear track, flanked by pre- and post-run sleep.
#
# **Approach**:
# 1. Stream the NWB file directly from S3 (no full download) using `remfile`.
# 2. Select the LFP channel with the strongest theta rhythm during running.
# 3. Restrict to running periods (speed-thresholded) within the maze epoch.
# 4. Band-pass filter (6-10 Hz) and Hilbert-transform the LFP to get the
#    instantaneous theta phase.
# 5. Assign each spike its theta phase and compute circular statistics
#    (mean resultant length, Rayleigh test) per unit.
# 6. Compare phase-locking strength between excitatory and inhibitory cells.

# %%
import sys

import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.signal import hilbert

# %% [markdown]
# ## 1. Load the NWB File (Streamed from DANDI)
#
# The file is streamed with `remfile` (chunked, disk-cached reads directly
# from the S3 blob) rather than downloaded in full. The theta LFP data is
# stored one channel per HDF5 chunk, so requesting a single channel over a
# time window only pulls the bytes for that channel.

# %%
S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/4f5/a84/4f5a84aa-a6e4-496a-9b23-535fa6fbd3ae"
CACHE_DIR = "cache/remfile_cache"

disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %%
units = nwb["units"]
epochs = nwb["epochs"]
lfp = nwb["LFP"]
maze = epochs[epochs.label == "MazeEpoch"]

print(f"n units: {len(units)}")
print(units.metadata.head())
print("\nEpochs:")
print(epochs)
print(f"\nLFP shape: {lfp.shape}, rate: {lfp.rate:.2f} Hz")


# %% [markdown]
# ### Fixing a units bug in the source file
#
# The `1.6mLinearMazeLinearizedPosition` and `1.6mLinearMazePosition`
# TimeSeries in this NWB file store the *reciprocal* of the sampling rate in
# their `rate` field (0.0256 instead of ~39.06 Hz), which corrupts
# timestamp reconstruction if used as-is. We rebuild the timestamps directly
# from `1 / stored_rate`; this does not affect the LFP or spike times, whose
# rate/timestamps are stored correctly.

# %%
def load_position_2d(nwbfile):
    ts = nwbfile.processing["behavior"]["1.6mLinearMazePosition"][
        "1.6mLinearMazeSpatialSeries"
    ]
    true_rate = 1.0 / ts.rate
    t = ts.starting_time + np.arange(ts.data.shape[0]) / true_rate
    return nap.TsdFrame(t=t, d=ts.data[:])


pos2d = load_position_2d(nwbfile).restrict(maze)

# %% [markdown]
# ## 2. Validate the Raw Data Streams
#
# Before any analysis, we look at short snippets of the raw LFP, the spike
# raster, and the animal's position to confirm the streamed data decodes
# sensibly and is temporally aligned.

# %%
t0 = maze.start[0] + 5.0
snippet = lfp.get(t0, t0 + 2.0)
fig, ax = plt.subplots(figsize=(10, 4))
channels_to_show = [0, 32, 64, 96]
offset = 800
for i, ch in enumerate(channels_to_show):
    ax.plot(snippet.t - t0, snippet[:, ch].d + i * offset, lw=0.6)
ax.set_yticks([i * offset for i in range(len(channels_to_show))])
ax.set_yticklabels([f"ch {c}" for c in channels_to_show])
ax.set_xlabel("Time (s)")
ax.set_title("Raw LFP snippet during MazeEpoch (4 example channels)")
fig.tight_layout()
fig.savefig("figures/01_raw_lfp_snippet.png", dpi=150)
plt.show()

# %%
sub_units = units.restrict(nap.IntervalSet(t0, t0 + 2.0))
fig, ax = plt.subplots(figsize=(10, 4))
for i, (uid, spk) in enumerate(sub_units.items()):
    ax.vlines(spk.t - t0, i, i + 0.8, color="k", lw=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Unit index")
ax.set_title("Spike raster, all units, 2s snippet during MazeEpoch")
fig.tight_layout()
fig.savefig("figures/01_spike_raster_snippet.png", dpi=150)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pos2d.t, pos2d[:, 0].d, lw=0.5, label="x")
ax.plot(pos2d.t, pos2d[:, 1].d, lw=0.5, label="y")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (m)")
ax.set_title("2D position during MazeEpoch (linear-track running laps)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/01_position_maze.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Select the LFP Channel with the Strongest Theta Rhythm
#
# We compute the power spectral density (1-40 Hz) for a sparse sample of
# channels (every 8th of the 128) over a running window, and pick the
# channel with the highest ratio of theta-band (6-10 Hz) power to broadband
# (1-40 Hz) power.

# %%
THETA_BAND = (6.0, 10.0)
BROAD_BAND = (1.0, 40.0)
CANDIDATE_CHANNELS = list(range(0, 128, 8))

t0 = maze.start[0] + 100.0
snippet = lfp.get(t0, t0 + 200.0)[:, CANDIDATE_CHANNELS]

ratios, psds = {}, {}
for i, ch in enumerate(CANDIDATE_CHANNELS):
    sig = nap.Tsd(t=snippet.t, d=snippet[:, i].d)
    psd = nap.compute_power_spectral_density(sig, fs=lfp.rate)
    f = psd.index.values
    p = np.abs(psd.iloc[:, 0].values.real)
    theta_mask = (f >= THETA_BAND[0]) & (f <= THETA_BAND[1])
    broad_mask = (f >= BROAD_BAND[0]) & (f <= BROAD_BAND[1])
    ratios[ch] = p[theta_mask].sum() / p[broad_mask].sum()
    psds[ch] = (f, p)

theta_channel = max(ratios, key=ratios.get)
print("Theta/broadband power ratio per candidate channel:")
for ch in CANDIDATE_CHANNELS:
    marker = "  <-- selected" if ch == theta_channel else ""
    print(f"  channel {ch:3d}: {ratios[ch]:.4f}{marker}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
for ch in CANDIDATE_CHANNELS:
    f, p = psds[ch]
    mask = (f >= 0.5) & (f <= BROAD_BAND[1])
    color = "crimson" if ch == theta_channel else "gray"
    lw = 2.0 if ch == theta_channel else 0.8
    alpha = 1.0 if ch == theta_channel else 0.5
    ax.semilogy(f[mask], p[mask], color=color, lw=lw, alpha=alpha,
                label=f"ch {ch} (selected)" if ch == theta_channel else None)
ax.axvspan(*THETA_BAND, color="gold", alpha=0.2, label="theta band")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (log scale)")
ax.set_title("PSD across candidate channels (running epoch)")
ax.legend()

ax = axes[1]
chs = list(ratios.keys())
vals = [ratios[c] for c in chs]
colors = ["crimson" if c == theta_channel else "steelblue" for c in chs]
ax.bar([str(c) for c in chs], vals, color=colors)
ax.set_xlabel("Channel")
ax.set_ylabel("Theta / broadband power ratio")
ax.set_title("Channel ranking")
ax.tick_params(axis="x", rotation=90)
fig.tight_layout()
fig.savefig("figures/02_theta_channel_selection.png", dpi=150)
plt.show()

print(f"\nSelected theta channel: {theta_channel}")

# %% [markdown]
# The clear peak at ~8 Hz superimposed on the 1/f background confirms this
# is genuine theta oscillation, not broadband noise.
#
# ## 4. Restrict to Running Periods
#
# Theta is most prominent and stable during active locomotion. We compute
# running speed from the 2D position and keep only periods where the animal
# moves faster than 5 cm/s, intersected with the maze epoch.

# %%
MIN_SPEED = 0.05  # m/s


def compute_speed(position_2d):
    d = np.diff(position_2d.d, axis=0)
    dt = np.diff(position_2d.t)
    speed = np.sqrt((d ** 2).sum(axis=1)) / dt
    speed = np.concatenate([[speed[0]], speed])
    speed_tsd = nap.Tsd(t=position_2d.t, d=speed)
    return speed_tsd.smooth(std=0.25, windowsize=1.0)


speed = compute_speed(pos2d)
running = speed.threshold(MIN_SPEED, method="above").time_support
running = running.intersect(maze)
print(f"Running epochs: {len(running)} intervals, "
      f"{running.tot_length():.1f} s total out of "
      f"{float(maze.tot_length()):.1f} s maze epoch")

# %% [markdown]
# ## 5. Extract Instantaneous Theta Phase
#
# Load the selected channel over the full maze epoch (streamed one channel
# at a time), band-pass filter it to the theta band, and take the Hilbert
# transform to get the instantaneous phase and amplitude envelope.

# %%
print("Loading full theta channel over the maze epoch (streamed)...")
lfp_channel = lfp[:, theta_channel].restrict(maze)
print(f"Loaded {len(lfp_channel)} samples")

filtered = nap.apply_bandpass_filter(
    lfp_channel, THETA_BAND, fs=lfp.rate, mode="butter", order=4
)
analytic = hilbert(filtered.d)
phase = np.mod(np.angle(analytic), 2 * np.pi)
phase_tsd = nap.Tsd(t=filtered.t, d=phase, time_support=filtered.time_support)
amplitude = nap.Tsd(t=filtered.t, d=np.abs(analytic), time_support=filtered.time_support)

phase_run = phase_tsd.restrict(running)
units_run = units.restrict(running)

# %% [markdown]
# ## 6. Per-Unit Circular Statistics
#
# For each unit with at least 50 spikes during running, we assign each spike
# the instantaneous theta phase of the nearest LFP sample, then compute:
# - **Mean resultant length (MRL)**: 0 = spikes uniformly distributed across
#   the theta cycle, 1 = all spikes at exactly the same phase.
# - **Rayleigh test p-value**: tests the null hypothesis that spike phases
#   are uniformly distributed.

# %%
MIN_SPIKES = 50


def rayleigh_test(phases):
    n = len(phases)
    C, S = np.sum(np.cos(phases)), np.sum(np.sin(phases))
    R = np.sqrt(C ** 2 + S ** 2) / n
    z = n * R ** 2
    p = np.exp(-z) * (
        1
        + (2 * z - z ** 2) / (4 * n)
        - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2)
    )
    return R, p


rows = []
for uid in units.index:
    spk = units_run[uid]
    if len(spk) < MIN_SPIKES:
        continue
    spk_phase = spk.value_from(phase_run)
    R, p = rayleigh_test(spk_phase.d)
    mean_phase = np.angle(np.mean(np.exp(1j * spk_phase.d))) % (2 * np.pi)
    rows.append(dict(
        unit_id=uid,
        cell_type=units.get_info("cell_type")[uid],
        location=units.get_info("location")[uid],
        n_spikes=len(spk),
        mean_phase_rad=mean_phase,
        mrl=R,
        rayleigh_p=p,
        significant=p < 0.05,
    ))

results = pd.DataFrame(rows)
print(f"Analyzed {len(results)} units (of {len(units)} total, "
      f"{MIN_SPIKES}+ spikes during running)")
print(f"Significantly phase-locked (Rayleigh p<0.05): "
      f"{results['significant'].sum()} / {len(results)} "
      f"({100 * results['significant'].mean():.1f}%)")
print(results.groupby("cell_type")["mrl"].describe())

# %% [markdown]
# ## 7. Visualize an Example Running Snippet
#
# Raw LFP, theta band-pass filtered LFP, and a spike raster for one
# strongly phase-locked and one weakly phase-locked unit, both with enough
# spikes to be visible in a short window. Dashed lines mark theta troughs.

# %%
well_sampled = results[results["n_spikes"] >= 500]
strongest = well_sampled.sort_values("mrl", ascending=False).iloc[0]
weakest = well_sampled.sort_values("mrl", ascending=True).iloc[0]
example_units = [int(strongest.unit_id), int(weakest.unit_id)]

window = 6.0
durations = np.asarray(running.end) - np.asarray(running.start)
i_longest = int(np.argmax(durations))
mid = (running.start[i_longest] + running.end[i_longest]) / 2
t0, t1 = mid - window / 2, mid + window / 2

raw_win = lfp_channel.restrict(nap.IntervalSet(t0, t1))
filt_win = filtered.restrict(nap.IntervalSet(t0, t1))
phase_win = phase_tsd.restrict(nap.IntervalSet(t0, t1))
phase_unwrapped = np.unwrap(phase_win.d)
cycle_crossings = phase_win.t[:-1][
    (np.mod(phase_unwrapped[:-1], 2 * np.pi) < np.pi)
    & (np.mod(phase_unwrapped[1:], 2 * np.pi) >= np.pi)
] - t0

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                          gridspec_kw={"height_ratios": [1.3, 1.3, 1.5]})

ax = axes[0]
ax.plot(raw_win.t - t0, raw_win.d, color="steelblue", lw=0.8)
for c in cycle_crossings:
    ax.axvline(c, color="gray", lw=0.5, alpha=0.5, ls="--")
ax.set_ylabel("Raw LFP (a.u.)")
ax.set_title(f"Theta channel — example {window:.0f} s running snippet, "
             "dashed lines = theta troughs")

ax = axes[1]
ax.plot(filt_win.t - t0, filt_win.d, color="darkorange", lw=1.2)
for c in cycle_crossings:
    ax.axvline(c, color="gray", lw=0.5, alpha=0.5, ls="--")
ax.set_ylabel(f"{THETA_BAND[0]:.0f}-{THETA_BAND[1]:.0f} Hz\nfiltered LFP")

ax = axes[2]
for i, uid in enumerate(example_units):
    spk = units[uid].restrict(nap.IntervalSet(t0, t1))
    ax.vlines(spk.t - t0, i, i + 0.8, color="k", lw=1.2)
for c in cycle_crossings:
    ax.axvline(c, color="gray", lw=0.5, alpha=0.5, ls="--")
ax.set_yticks([0.4, 1.4])
ax.set_yticklabels(
    [f"unit {example_units[0]} ({strongest.cell_type})\nMRL={strongest.mrl:.2f}",
     f"unit {example_units[1]} ({weakest.cell_type})\nMRL={weakest.mrl:.2f}"],
    fontsize=8,
)
ax.set_xlabel("Time (s)")
ax.set_xlim(0, window)
fig.tight_layout()
fig.savefig("figures/03_theta_lfp_phase_raster.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Example Polar Phase Histograms
#
# Spike-phase distributions for the strongest-locked pyramidal cell, the
# strongest-locked interneuron, and a non-significantly-locked unit.

# %%
exc = results[results.cell_type == "excitatory"].sort_values("mrl", ascending=False)
inh = results[results.cell_type == "inhibitory"].sort_values("mrl", ascending=False)
examples = [
    ("Strongest pyramidal cell", exc.iloc[0]),
    ("Strongest interneuron", inh.iloc[0]),
    ("Weakest (non-significant) unit", results.sort_values("mrl").iloc[0]),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), subplot_kw={"projection": "polar"})
for ax, (title, row) in zip(axes, examples):
    uid = int(row.unit_id)
    spk = units[uid].restrict(running)
    spk_phase = spk.value_from(phase_tsd).d
    n_bins = 24
    counts, bin_edges = np.histogram(spk_phase, bins=n_bins, range=(0, 2 * np.pi))
    width = 2 * np.pi / n_bins
    ax.bar(bin_edges[:-1], counts, width=width, align="edge",
           color="steelblue", edgecolor="white", alpha=0.85)
    ax.plot([row.mean_phase_rad, row.mean_phase_rad], [0, counts.max()],
            color="crimson", lw=2)
    ax.set_title(
        f"{title}\nunit {uid} ({row.cell_type})\n"
        f"MRL={row.mrl:.2f}, p={row.rayleigh_p:.1e}, n={int(row.n_spikes)}",
        fontsize=9,
    )
    ax.set_theta_zero_location("N")
fig.tight_layout()
fig.savefig("figures/04_example_polar_phase_histograms.png", dpi=150)
plt.show()

# %% [markdown]
# ## 9. Population Summary
#
# Across the population, most CA1 units are significantly phase-locked to
# theta, and interneurons show stronger phase-locking (higher MRL) than
# pyramidal cells, consistent with the literature: interneurons receive
# strong rhythmic inhibitory-network drive at theta frequency, whereas
# pyramidal cells' spike timing is also shaped by spatial (place-field) and
# rate-coding demands that compete with strict phase-locking.

# %%
fig = plt.figure(figsize=(13, 4.5))

ax1 = fig.add_subplot(1, 3, 1)
for ct, color in [("excitatory", "steelblue"), ("inhibitory", "darkorange")]:
    sub = results[results.cell_type == ct]
    ax1.hist(sub.mrl, bins=20, range=(0, 0.6), alpha=0.6, color=color, label=ct)
ax1.set_xlabel("Mean resultant length (MRL)")
ax1.set_ylabel("Number of units")
ax1.set_title("Theta phase-locking strength by cell type")
ax1.legend()

ax2 = fig.add_subplot(1, 3, 2)
order = ["excitatory", "inhibitory"]
data = [results[results.cell_type == ct].mrl.values for ct in order]
ax2.boxplot(data, tick_labels=order, showmeans=True)
rng = np.random.default_rng(0)
for i, d in enumerate(data):
    jitter = rng.normal(0, 0.04, size=len(d))
    ax2.scatter(np.full(len(d), i + 1) + jitter, d, s=10, alpha=0.5, color="gray")
ax2.set_ylabel("MRL")
ax2.set_title("Excitatory vs inhibitory phase locking")

ax3 = fig.add_subplot(1, 3, 3, projection="polar")
for ct, color in [("excitatory", "steelblue"), ("inhibitory", "darkorange")]:
    sub = results[(results.cell_type == ct) & results.significant]
    ax3.scatter(sub.mean_phase_rad, sub.mrl, s=25, alpha=0.7, color=color, label=ct)
ax3.set_theta_zero_location("N")
ax3.set_title("Preferred theta phase\n(significant units)", fontsize=9, pad=20)
ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

fig.tight_layout()
fig.savefig("figures/05_population_summary.png", dpi=150)
plt.show()

print("\nSummary statistics:")
print(results.groupby("cell_type")[["mrl", "significant"]].agg(["mean", "count"]))

# %% [markdown]
# ## Conclusion
#
# The large majority of CA1 units in this recording (~84%) show
# statistically significant theta phase-locking (Rayleigh p < 0.05) during
# running on the linear track, and inhibitory interneurons show
# substantially stronger phase-locking (higher MRL) than excitatory
# pyramidal cells. This directly demonstrates theta phase entrainment of
# hippocampal neurons: spike timing across the population is organized
# relative to the ongoing theta rhythm rather than occurring uniformly
# across the theta cycle.
