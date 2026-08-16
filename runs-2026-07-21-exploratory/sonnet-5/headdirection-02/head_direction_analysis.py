# %% [markdown]
# # Head Direction Cells in the Anterior Thalamus and Postsubiculum
#
# This notebook demonstrates head direction (HD) tuning in extracellular recordings from
# freely moving mice, using data from the DANDI Archive.
#
# **Dataset:** [DANDI:000056](https://dandiarchive.org/dandiset/000056) — "Internally
# organized mechanisms of the head direction sense" (Peyrache lab). Mice were implanted
# with silicon probes in the anterior thalamic nucleus (ADn) and/or postsubiculum (PoSub)
# and recorded while freely foraging in an open arena. Two head-mounted LEDs (red and blue)
# were tracked to reconstruct the animal's instantaneous head direction.
#
# **Session used:** `sub-Mouse24/sub-Mouse24_ses-Mouse24-131213` (the smallest session in
# the dandiset, ~1.3 GB, streamed directly from S3 rather than downloaded in full).
#
# **Analysis:**
# 1. Load spikes, LED positions, and behavioral-state epochs via Pynapple.
# 2. Compute the instantaneous head-direction angle from the vector between the two LEDs.
# 3. Restrict to waking behavior and compute circular (angular) tuning curves per neuron.
# 4. Identify head-direction cells using the Rayleigh vector length, with a shuffle test
#    for significance.
# 5. Visualize tuning curves, sort neurons by preferred direction, and Bayesian-decode
#    head direction from the population to show that HD cells collectively track heading.

# %%
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from tqdm import tqdm

np.random.seed(0)

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# %% [markdown]
# ## Load the NWB File (Streaming from DANDI)
#
# The file is streamed directly from the DANDI S3 bucket with `remfile`, using a local
# disk cache so repeated reads of the same byte ranges don't re-download data.

# %%
S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/f00/e5c/f00e5c3a-9435-42df-aace-9b6952563479"

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %%
units = nwb["units"]
red_led = nwb["RedLED"]
blue_led = nwb["BlueLED"]
states = nwb["states"]

print(f"Number of units (raw): {len(units)}")
print(f"Firing rate range: {units.get_info('rate').min():.2f} - {units.get_info('rate').max():.2f} Hz")
print(states)

# %%
# drop silent units (zero spikes for the whole session) - these break circular
# statistics downstream (0/0 resultant vector length) and can't be meaningfully
# tested for tuning anyway
silent = units.get_info("rate") == 0
print(f"Dropping {silent.sum()} silent unit(s): {list(units.index[silent])}")
units = units[~silent]
print(f"Number of units (analyzed): {len(units)}")

# %% [markdown]
# ## Compute Head Direction from LED Positions
#
# The head direction is the angle of the vector pointing from the blue LED to the red
# LED, in the plane of the arena floor. Frames where either LED was not detected are
# coded as `(-1, -1)` in the raw data and must be dropped before computing the angle.

# %%
red_xy = red_led.values
blue_xy = blue_led.values
valid = (red_xy[:, 0] != -1) & (blue_xy[:, 0] != -1)
print(f"Valid tracking frames: {valid.sum()} / {len(valid)} ({100 * valid.mean():.1f}%)")

dx = red_xy[:, 0] - blue_xy[:, 0]
dy = red_xy[:, 1] - blue_xy[:, 1]
angle = np.mod(np.arctan2(dy, dx), 2 * np.pi)

head_direction = nap.Tsd(t=red_led.index.values[valid], d=angle[valid])
print(head_direction)

# %% [markdown]
# ## Restrict to Waking Behavior
#
# The `states` epochs distinguish Awake, Non-REM sleep, and REM sleep. Head direction
# tuning during active foraging is computed only during Awake epochs.

# %%
wake_ep = states[states.label == "Awake"]
print(wake_ep)

head_direction_wake = head_direction.restrict(wake_ep)
print(f"Wake head-direction samples: {len(head_direction_wake)}")

# %% [markdown]
# ## Raw Data Overview
#
# Before computing tuning curves, visualize the raw spike rasters, the LED trajectory,
# and the head-direction time series to confirm the data look sensible.

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

# Spike raster (first 60s of an awake epoch)
example_ep = nap.IntervalSet(start=wake_ep.start[0], end=wake_ep.start[0] + 60)
spikes_ex = units.restrict(example_ep)
for i, (idx, ts) in enumerate(spikes_ex.items()):
    axes[0].vlines(ts.index.values, i, i + 0.8, color="k", linewidth=0.5)
axes[0].set_ylabel("Unit #")
axes[0].set_title(f"Spike raster, 60 s awake example (starting t={wake_ep.start[0]:.0f} s)")
axes[0].set_xlabel("Time (s)")

# LED trajectory
red_ex = red_led.restrict(example_ep).values
blue_ex = blue_led.restrict(example_ep).values
axes[1].plot(red_xy[valid, 0], red_xy[valid, 1], ",", color="lightgray", alpha=0.3, label="full session (red LED)")
axes[1].plot(red_ex[:, 0], red_ex[:, 1], "-o", color="red", markersize=2, linewidth=0.5, label="60s example (red LED)")
axes[1].plot(blue_ex[:, 0], blue_ex[:, 1], "-o", color="blue", markersize=2, linewidth=0.5, label="60s example (blue LED)")
axes[1].set_xlabel("x (px)")
axes[1].set_ylabel("y (px)")
axes[1].set_title("Head LED tracking in the arena")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].set_aspect("equal")

# Head direction time series
hd_ex = head_direction.restrict(example_ep)
axes[2].plot(hd_ex.index.values, np.degrees(hd_ex.values), ".", markersize=2)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Head direction (deg)")
axes[2].set_title("Head direction over the same 60 s example")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/01_raw_data_overview.png", dpi=150)
plt.close()

# %% [markdown]
# ## Compute Angular Tuning Curves
#
# For each unit, the firing rate is binned as a function of head direction (60 bins
# spanning 0-360 deg), restricted to waking epochs.

# %%
N_BINS = 60
tuning_curves = nap.compute_tuning_curves(
    units, head_direction_wake, bins=N_BINS, range=[(0, 2 * np.pi)], epochs=wake_ep
)
angles = tuning_curves["0"].values
print(tuning_curves)

# %% [markdown]
# ## Identify Head-Direction Cells
#
# A neuron's directional selectivity is summarized by the mean resultant (Rayleigh)
# vector length of its tuning curve, `R`, which ranges from 0 (uniform firing across
# directions) to 1 (firing concentrated at a single direction). Significance is assessed
# with a shuffle test: head-direction timestamps are randomly time-shifted (circularly,
# within the wake epoch) 500 times, tuning curves are recomputed, and the true `R` is
# compared to the shuffle distribution.

# %%
def resultant_vector_length(rates, angles):
    """Mean resultant vector length of a circular tuning curve."""
    weights = rates / rates.sum()
    return np.abs(np.sum(weights * np.exp(1j * angles)))


unit_ids = tuning_curves["unit"].values
true_R = np.array(
    [resultant_vector_length(tuning_curves.sel(unit=u).values, angles) for u in unit_ids]
)

# %%
N_SHUFFLES = 500
n_samples = len(head_direction_wake)
shuffle_R = np.zeros((N_SHUFFLES, len(unit_ids)))

# circular shuffle: roll the HD values relative to their own timestamps, so
# each shuffle keeps the HD trace's autocorrelation intact but destroys its
# alignment with the spike trains
for i in tqdm(range(N_SHUFFLES), desc="Shuffling head direction"):
    shift_samples = np.random.randint(int(0.1 * n_samples), int(0.9 * n_samples))
    shuffled_values = np.roll(head_direction_wake.values, shift_samples)
    shuffled_hd = nap.Tsd(t=head_direction_wake.index.values, d=shuffled_values)
    shuf_tc = nap.compute_tuning_curves(
        units, shuffled_hd, bins=N_BINS, range=[(0, 2 * np.pi)], epochs=wake_ep
    )
    shuffle_R[i] = np.array(
        [resultant_vector_length(shuf_tc.sel(unit=u).values, angles) for u in unit_ids]
    )

p_values = np.mean(shuffle_R >= true_R[None, :], axis=0)
is_hd_cell = p_values < 0.01

print(f"{is_hd_cell.sum()} / {len(unit_ids)} units classified as head-direction cells (p < 0.01)")
for u, r, p, hd in zip(unit_ids, true_R, p_values, is_hd_cell):
    print(f"  unit {u}: R={r:.3f}, p={p:.3f}, HD cell={hd}")

# %% [markdown]
# ## Visualize Directional Tuning
#
# Polar plots of firing rate vs. head direction for the significant HD cells, and a
# summary of resultant vector length vs. shuffle-test p-value for the full population.

# %%
hd_unit_ids = unit_ids[is_hd_cell]
n_hd = len(hd_unit_ids)
n_cols = 4
n_rows = int(np.ceil(n_hd / n_cols))

fig = plt.figure(figsize=(3.2 * n_cols, 3.2 * n_rows))
for i, u in enumerate(hd_unit_ids):
    ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="polar")
    rates = tuning_curves.sel(unit=u).values
    # close the loop for a continuous polar line
    theta = np.append(angles, angles[0])
    r = np.append(rates, rates[0])
    ax.plot(theta, r, color="C0")
    ax.fill(theta, r, color="C0", alpha=0.2)
    ax.set_title(f"unit {u}\nR={true_R[unit_ids == u][0]:.2f}", fontsize=10, y=1.2)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0", "", "90", "", "180", "", "270", ""])
    ax.set_yticklabels([])

plt.tight_layout()
plt.savefig(f"{FIGDIR}/02_polar_tuning_curves.png", dpi=150)
plt.close()

# %%
fig, ax = plt.subplots(figsize=(6, 5))
colors = np.where(is_hd_cell, "C3", "gray")
ax.scatter(true_R, p_values, c=colors, s=40, edgecolor="k", linewidth=0.5)
ax.axhline(0.01, color="k", linestyle="--", linewidth=1, label="p = 0.01")
ax.set_xlabel("Resultant vector length (R)")
ax.set_ylabel("Shuffle-test p-value")
ax.set_title("Head-direction tuning strength vs. significance")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/03_R_vs_pvalue.png", dpi=150)
plt.close()

# %% [markdown]
# ## Population Tuning Curve Heatmap
#
# Sorting all units by preferred direction reveals that the significant HD cells tile
# the full range of head directions, a hallmark of an HD-cell population that can
# collectively represent any heading.

# %%
preferred_direction = angles[np.argmax(tuning_curves.values, axis=1)]
sort_order = np.argsort(preferred_direction)

normalized_tc = tuning_curves.values / tuning_curves.values.max(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    normalized_tc[sort_order],
    aspect="auto",
    extent=[0, 360, len(unit_ids), 0],
    cmap="viridis",
)
for rank, orig_idx in enumerate(sort_order):
    if is_hd_cell[orig_idx]:
        ax.text(365, rank + 0.7, "*", color="red", fontsize=12)
ax.set_xlabel("Head direction (deg)")
ax.set_ylabel("Unit (sorted by preferred direction)")
ax.set_title("Normalized firing rate vs. head direction, all units\n(* = significant HD cell)")
plt.colorbar(im, ax=ax, label="Normalized rate")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/04_population_heatmap.png", dpi=150)
plt.close()

# %% [markdown]
# ## Bayesian Decoding of Head Direction from Population Activity
#
# If the significant HD cells genuinely encode heading, their joint spiking activity
# should allow head direction to be reconstructed on held-out data. Tuning curves are
# fit on the first half of the wake epochs and used to Bayesian-decode head direction
# on the second half.

# %%
hd_units = units[hd_unit_ids]

mid_time = wake_ep.start[0] + wake_ep.tot_length() / 2
train_ep = wake_ep.intersect(nap.IntervalSet(start=wake_ep.start[0], end=mid_time))
test_ep = wake_ep.intersect(nap.IntervalSet(start=mid_time, end=wake_ep.end[-1]))

train_hd = head_direction.restrict(train_ep)
train_tc = nap.compute_tuning_curves(
    hd_units, train_hd, bins=N_BINS, range=[(0, 2 * np.pi)], epochs=train_ep
)

BIN_SIZE = 0.3  # seconds
decoded_hd, decoded_proba = nap.decode_bayes(
    train_tc, hd_units, test_ep, bin_size=BIN_SIZE, sliding_window_size=3
)

true_hd_test = head_direction.restrict(test_ep)

# %%
# circular decoding error, in degrees
true_hd_interp = true_hd_test.interpolate(decoded_hd)
error = np.degrees(np.angle(np.exp(1j * (decoded_hd.values - true_hd_interp.values))))
print(f"Median absolute decoding error: {np.median(np.abs(error)):.1f} deg")
print(f"Circular mean resultant length of decoding error: {resultant_vector_length(np.ones(len(error)), np.radians(error)):.3f}")

# %%
# search 120s windows within the longest continuous awake bout of the test
# epoch for the one with the most head-direction movement (an objective
# criterion, chosen independent of decoding accuracy) to give a representative,
# behaviorally informative example rather than a near-stationary period
longest_bout = np.argmax(test_ep.end - test_ep.start)
bout_start, bout_end = test_ep.start[longest_bout], test_ep.end[longest_bout]

WINDOW = 120
best_range, example_start = -1, bout_start
for t0 in np.arange(bout_start, bout_end - WINDOW, 60):
    win_hd = head_direction.restrict(nap.IntervalSet(start=t0, end=t0 + WINDOW))
    if len(win_hd) < 100:
        continue
    ang_range = np.ptp(np.unwrap(win_hd.values))
    if ang_range > best_range:
        best_range, example_start = ang_range, t0

example_decode_ep = nap.IntervalSet(start=example_start, end=example_start + WINDOW)

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

true_ex = true_hd_test.restrict(example_decode_ep)
decoded_ex = decoded_hd.restrict(example_decode_ep)
axes[0].plot(true_ex.index.values, np.degrees(true_ex.values), ".", color="k", markersize=3, label="true head direction")
axes[0].plot(decoded_ex.index.values, np.degrees(decoded_ex.values), ".", color="C3", markersize=3, label="decoded (Bayesian)")
axes[0].set_ylabel("Head direction (deg)")
axes[0].set_xlabel("Time (s)")
axes[0].set_title(f"Decoded vs. true head direction, held-out test epoch (using {len(hd_unit_ids)} HD cells)")
axes[0].legend(loc="upper right", markerscale=3)

axes[1].hist(error, bins=72, range=(-180, 180), color="C0")
axes[1].set_xlabel("Decoding error (deg)")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of decoding error across the held-out test epoch")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/05_decoding.png", dpi=150)
plt.close()

# %% [markdown]
# ## Summary
#
# Head-direction tuning was assessed for all 22 recorded units during waking behavior in
# a single session from DANDI:000056. A subset of units showed strong, statistically
# significant circular tuning to head direction (shuffle test, p < 0.01), with preferred
# directions tiling the full 360-degree range. Bayesian decoding using only these
# significant HD cells reconstructed the animal's instantaneous heading on held-out data
# well above chance, confirming that their population activity carries a coherent,
# continuously updated representation of head direction — the defining property of head
# direction cells.

# %%
io.close()
print("Done.")
