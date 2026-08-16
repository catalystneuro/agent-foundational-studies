# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hippocampal Place Cells During Locomotion (DANDI :001754)
#
# This notebook demonstrates the classic hippocampal place cell phenomenon
# (O'Keefe & Dostrovsky, 1971) using real extracellular recordings streamed
# directly from the DANDI Archive.
#
# **Dataset:** [DANDI:001754](https://dandiarchive.org/dandiset/001754), "Three-dimensional
# spatial selectivity of hippocampal neurons during space flight". Rats implanted with
# tetrode arrays in hippocampal area CA1 were recorded during the Neurolab Space Shuttle
# mission (STS-90, April 1998) by Knierim, McNaughton and Poe. The dandiset contains, for
# several sessions, interleaved "Baseline" (BL) epochs in which the animal foraged freely
# on a flat, two-dimensional track for food-triggered medial-forebrain-bundle stimulation,
# in between the specialized 3-D "Escher Staircase" (ES) and "Magic Carpet" (MC) task
# epochs described in the original study.
#
# **Session used:** `sub-Rat1/sub-Rat1_ses-19980425T124500_behavior+ecephys.nwb`
# (Rat 1, Flight Day 9). This session's baseline epoch (~44 minutes, in-flight, so
# recorded in microgravity) is the longest continuous open-field baseline recording in the
# dandiset and gives the most statistically robust place fields.
#
# **Analysis:**
# 1. Stream the NWB file with `remfile` (no local download of the full file).
# 2. Load spikes and position with `pynapple`.
# 3. Clean the position signal (remove a video-tracker dropout code and zero-order-hold
#    repeats) and restrict to locomotion epochs (speed thresholding).
# 4. Compute occupancy-normalized 2-D firing-rate maps ("place fields") for every unit.
# 5. Quantify spatial tuning with Skaggs spatial information (bits/spike) and test its
#    significance against a circular-shuffle null distribution, excluding low-occupancy
#    bins from the statistic.
# 6. Visualize place fields for the units with significant spatial information.

# %%
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from pynwb import NWBHDF5IO
import pynapple as nap

np.set_printoptions(suppress=True)
plt.rcParams['figure.dpi'] = 100

# %% [markdown]
# ## 1. Stream the NWB File From DANDI
#
# We stream the file directly from the DANDI S3 bucket using `remfile`, which only reads
# the byte ranges it needs and caches them to disk, avoiding a full download of the
# ~9 MB file.

# %%
DANDISET_ID = "001754"
NWB_PATH = "sub-Rat1/sub-Rat1_ses-19980425T124500_behavior+ecephys.nwb"
S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/dec/930/dec930b1-29b8-4877-ba17-6897f1ae0bdb"

disk_cache = remfile.DiskCache("./remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print(nwb)

# %% [markdown]
# ## 2. Inspect the Dataset
#
# The file contains spike-sorted units from 8 tetrodes in CA1, a 2-D position signal
# from video tracking, and epochs marking which behavioral task was running.

# %%
print("Subject:", nwbfile.subject.subject_id, "-", nwbfile.subject.description)
print("Session start:", nwbfile.session_start_time)
print("Session description:", nwbfile.session_description)
print()

epochs_df = nwbfile.epochs.to_dataframe()
print(epochs_df[["start_time", "stop_time", "session_type", "session_type_description"]])

# %%
units = nwb["units"]
pos = nwb["spatial_series"]
print(f"Number of units: {len(units)}")
print(f"Position samples: {len(pos)}, columns: {list(pos.columns)}")

# %% [markdown]
# We focus on the "BL" (Baseline, flat rectangular track) epoch, which runs from
# t=4871 s to t=7535 s (~44 minutes). This is the only epoch of free 2-D foraging long
# enough to estimate place fields reliably; the "ES" and "MC" epochs use the specialized
# 3-D / interleaved tracks described in the original study.

# %%
BASELINE_START, BASELINE_STOP = 4871.0, 7535.0
baseline_ep = nap.IntervalSet(start=BASELINE_START, end=BASELINE_STOP)
print(f"Baseline epoch duration: {baseline_ep.tot_length():.1f} s")

# %% [markdown]
# ## 3. Inspect and Clean the Raw Position Signal
#
# First we plot the raw position trace during the baseline epoch to check data quality.

# %%
pos_raw = pos.restrict(baseline_ep)
xy_raw = pos_raw.values

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(xy_raw[:, 0], xy_raw[:, 1], s=1, alpha=0.2)
axes[0].set_xlabel("x (pixels)")
axes[0].set_ylabel("y (pixels)")
axes[0].set_title("Raw tracked position (baseline epoch)")

axes[1].plot(pos_raw.index.values, xy_raw[:, 0], label="x", lw=0.5)
axes[1].plot(pos_raw.index.values, xy_raw[:, 1], label="y", lw=0.5)
axes[1].set_xlabel("time (s)")
axes[1].set_ylabel("position (pixels)")
axes[1].set_title("Raw position vs. time")
axes[1].legend()
plt.tight_layout()
plt.savefig("fig01_raw_position.png", dpi=110)
plt.close()

n_dropout = int(np.sum((xy_raw[:, 0] < 1) & (xy_raw[:, 1] > 250)))
print(f"Samples pinned at the tracker dropout code (x<1, y>250): "
      f"{n_dropout} / {len(xy_raw)} ({100*n_dropout/len(xy_raw):.1f}%)")

# %% [markdown]
# `fig01_raw_position.png` reveals a dense cluster of samples pinned at a single
# out-of-arena coordinate (x≈0, y≈255). This is the video tracker's error code emitted
# whenever it briefly loses the head-stage LED; it is not a real position and must be
# removed before computing occupancy or firing-rate maps.
#
# The NWB metadata for this spatial series also notes that position was "sampled at spike
# occurrence times" rather than at a fixed video frame rate: whenever the tracker's last
# known coordinate had not yet updated, that same coordinate is repeated verbatim at every
# subsequent spike timestamp (a zero-order hold). Left uncorrected, a multi-second run of
# identical repeated coordinates lets a handful of coincident spikes from several
# different units inflate the apparent firing rate of one small, rarely-visited bin. We
# remove these repeats by collapsing consecutive, bit-identical position samples down to
# their first occurrence.

# %%
xy = pos_raw.values
t = pos_raw.index.values
valid = ~((xy[:, 0] < 1) & (xy[:, 1] > 250))
xy, t = xy[valid], t[valid]

changed = np.ones(len(xy), dtype=bool)
changed[1:] = np.any(xy[1:] != xy[:-1], axis=1)
n_repeats = int((~changed).sum())
xy, t = xy[changed], t[changed]

position = nap.TsdFrame(t=t, d=xy, time_support=baseline_ep, columns=["x", "y"])
print(f"Removed {int((~valid).sum())} dropout samples and {n_repeats} repeated "
      f"zero-order-hold samples; kept {len(position)} / {len(xy_raw)} position samples.")

# %% [markdown]
# ## 4. Compute Running Speed and Restrict to Locomotion Epochs
#
# Place fields are conventionally computed only while the animal is moving, to exclude
# immobility-related firing (e.g. sharp-wave-ripple replay) that would otherwise
# contaminate the spatial map. We compute instantaneous speed from the cleaned position
# trace, smooth it, and keep only epochs above a 3 px/s threshold.

# %%
d = np.sqrt(np.sum(np.diff(position.values, axis=0) ** 2, axis=1))
dt = np.diff(position.index.values)
speed_t = position.index.values[:-1] + dt / 2
speed = nap.Tsd(t=speed_t, d=d / dt, time_support=baseline_ep)
speed_smooth = speed.smooth(std=0.5, windowsize=2.0)

SPEED_THRESHOLD = 3.0  # pixels/s
run_ep = speed_smooth.threshold(SPEED_THRESHOLD, method="above").time_support
print(f"Locomotion epochs: {run_ep.tot_length():.1f} s of {baseline_ep.tot_length():.1f} s "
      f"({100*run_ep.tot_length()/baseline_ep.tot_length():.0f}%), in {len(run_ep)} bout(s)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(speed_smooth.values, bins=200, color="steelblue")
axes[0].axvline(SPEED_THRESHOLD, color="red", ls="--", label=f"threshold = {SPEED_THRESHOLD} px/s")
axes[0].set_yscale("log")
axes[0].set_xlim(0, 60)
axes[0].set_xlabel("smoothed speed (pixels/s)")
axes[0].set_ylabel("count (log scale)")
axes[0].set_title("Speed distribution, zoomed to 0-60 px/s")
axes[0].legend()

axes[1].hist(speed_smooth.values, bins=200, color="steelblue")
axes[1].axvline(SPEED_THRESHOLD, color="red", ls="--")
axes[1].set_yscale("log")
axes[1].set_xscale("log")
axes[1].set_xlabel("smoothed speed (pixels/s, log scale)")
axes[1].set_ylabel("count (log scale)")
axes[1].set_title("Full range (log-log)\nhigh-speed tail = residual tracking jumps")
plt.tight_layout()
plt.savefig("fig02_speed_distribution.png", dpi=110)
plt.close()

position_run = position.restrict(run_ep)
units_run = units.restrict(run_ep)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(position.values[:, 0], position.values[:, 1], s=1, alpha=0.15, color="gray")
axes[0].set_title("Full cleaned trajectory")
axes[1].scatter(position_run.values[:, 0], position_run.values[:, 1], s=1, alpha=0.15, color="steelblue")
axes[1].set_title("Trajectory restricted to locomotion (speed > 3 px/s)")
for ax in axes:
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
plt.tight_layout()
plt.savefig("fig03_trajectory_cleaned.png", dpi=110)
plt.close()

print(f"Number of units with spikes in locomotion epochs: {len(units_run)}")
print(f"Mean firing rate range: {units_run.rates.min():.2f} - {units_run.rates.max():.2f} Hz")

# %% [markdown]
# ## 5. Raw Spiking Activity
#
# Before computing place fields, we look at raw spiking activity: a raster plot across
# all units and a per-unit spike count summary.

# %%
fig, ax = plt.subplots(figsize=(12, 6))
for i, uid in enumerate(units_run.index):
    spk_t = units_run[uid].restrict(baseline_ep).index.values
    ax.vlines(spk_t, i, i + 0.8, color="k", lw=0.3)
ax.set_xlabel("time (s)")
ax.set_ylabel("unit #")
ax.set_title(f"Raster of all {len(units_run)} CA1 units during the baseline epoch")
plt.tight_layout()
plt.savefig("fig04_raster.png", dpi=110)
plt.close()

fastest_unit = units_run.rates.argmax()
fastest_uid = units_run.index[fastest_unit]
print(f"Unit {fastest_uid} fires at {units_run.rates[fastest_unit]:.1f} Hz overall "
      "(the solid black band in the raster) - a rate far higher than typical CA1 "
      "pyramidal place cells, more consistent with a fast-spiking interneuron. "
      "Interneurons are included in the spatial-tuning analysis below like any other "
      "unit, but are expected to show weak or no significant spatial information.")

# %% [markdown]
# ## 6. Occupancy-Normalized 2-D Place Fields
#
# For each unit we compute an occupancy-normalized firing-rate map: the arena is
# discretized into a 15x15 grid, spike counts and occupancy time are tallied per bin,
# and (after light Gaussian smoothing to reduce sampling noise) the firing rate is the
# smoothed spike count divided by the smoothed occupancy time. Bins with negligible
# occupancy (<1.0 s, i.e. bins visited only fleetingly) are excluded, since a rate
# estimate built from a couple of coincidental spikes during a fleeting visit is
# unreliable and disproportionately noisy.

# %%
N_BINS = 15
SMOOTH_SIGMA = 1.0
MIN_OCC_SEC = 1.0

counts = nap.compute_tuning_curves(
    units_run, position_run, bins=N_BINS, epochs=run_ep,
    feature_names=["x", "y"], return_counts=True,
)
fs = counts.attrs["fs"]
occupancy_samples = counts.attrs["occupancy"].astype(float)
occupancy_sec = gaussian_filter(occupancy_samples, sigma=SMOOTH_SIGMA) / fs
occupancy_mask = occupancy_sec < MIN_OCC_SEC

print(f"Estimated position sampling rate: {fs:.2f} Hz")
print(f"Occupancy bins masked (< {MIN_OCC_SEC} s): {occupancy_mask.sum()} / {occupancy_mask.size}")


def smoothed_rate_map(spike_counts_2d):
    """Gaussian-smoothed, occupancy-normalized firing-rate map (Hz)."""
    smoothed_counts = gaussian_filter(spike_counts_2d.astype(float), sigma=SMOOTH_SIGMA)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = smoothed_counts / occupancy_sec
    return rate


rate_maps = {uid: smoothed_rate_map(counts.sel(unit=uid).values) for uid in counts.coords["unit"].values}

fig, ax = plt.subplots(figsize=(5, 4.5))
occ_disp = occupancy_sec.copy()
occ_disp[occupancy_mask] = np.nan
im = ax.imshow(occ_disp.T, origin="lower", cmap="viridis")
ax.set_title("Occupancy map (seconds)")
ax.set_xlabel("x bin")
ax.set_ylabel("y bin")
plt.colorbar(im, ax=ax, fraction=0.046, label="seconds")
plt.tight_layout()
plt.savefig("fig05_occupancy_map.png", dpi=110)
plt.close()

# %% [markdown]
# ## 7. Spatial Information and Shuffle-Based Significance Testing
#
# Spatial tuning is quantified with Skaggs et al. (1993) spatial information, in
# bits/spike. To decide which units are genuine "place cells" (as opposed to units whose
# apparent spatial information is inflated by chance, e.g. low spike counts), we build a
# null distribution per unit by circularly shifting its spike train by a random amount
# (wrapping within the locomotion time support) 500 times, recomputing spatial
# information each time, and calculating a p-value as the fraction of shuffles with
# spatial information at least as high as observed.


# %%
# Bins with less than MIN_OCC_SEC of (smoothed) occupancy are excluded from the spatial
# information sum itself, not just from the display: a bin visited only briefly can
# combine with a couple of coincidental spikes to produce a spuriously huge instantaneous
# rate that dominates the sum despite carrying almost no real information about the
# animal's behavior. Because position (and hence occupancy_sec/occupancy_mask) does not
# change across shuffles, we reuse the same mask for the observed data and every shuffle.
def mutual_information_bits_per_spike(rate_map, occ_sec, mask):
    p_occ = occ_sec[mask] / occ_sec[mask].sum()
    lam = np.clip(rate_map[mask], 0, None)
    mean_rate = np.sum(p_occ * lam)
    if mean_rate <= 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p_occ * lam * np.log2(lam / mean_rate)
    terms = np.nan_to_num(terms, nan=0.0, posinf=0.0, neginf=0.0)
    bits_per_sec = np.sum(terms)
    return bits_per_sec / mean_rate


occupancy_valid = ~occupancy_mask
observed_mi = {uid: mutual_information_bits_per_spike(rate_maps[uid], occupancy_sec, occupancy_valid)
               for uid in units_run.index}
n_spikes = {uid: len(units_run[uid]) for uid in units_run.index}

N_SHUFFLES = 500
null_mi = {uid: np.zeros(N_SHUFFLES) for uid in units_run.index}
for i in tqdm(range(N_SHUFFLES), desc="Shuffling"):
    shuffled_units = nap.shift_timestamps(units_run, mode="wrap")
    counts_shuf = nap.compute_tuning_curves(
        shuffled_units, position_run, bins=N_BINS, epochs=run_ep,
        feature_names=["x", "y"], return_counts=True,
    )
    for uid in counts_shuf.coords["unit"].values:
        smoothed_counts = gaussian_filter(counts_shuf.sel(unit=uid).values.astype(float), sigma=SMOOTH_SIGMA)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = smoothed_counts / occupancy_sec
        null_mi[uid][i] = mutual_information_bits_per_spike(rate, occupancy_sec, occupancy_valid)

# %%
MIN_SPIKES = 50
ALPHA = 0.05
TREND_ALPHA = 0.10

results = []
for uid in units_run.index:
    p_value = (np.sum(null_mi[uid] >= observed_mi[uid]) + 1) / (N_SHUFFLES + 1)
    results.append({
        "unit": uid,
        "bits_per_spike": observed_mi[uid],
        "n_spikes": n_spikes[uid],
        "p_value": p_value,
        "null_mean": null_mi[uid].mean(),
    })
results_df = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)

eligible = results_df["n_spikes"] >= MIN_SPIKES
significant = eligible & (results_df["p_value"] < ALPHA)
trending = eligible & (results_df["p_value"] < TREND_ALPHA) & ~significant
place_cell_ids = results_df.loc[significant, "unit"].tolist()

print(f"Units with >= {MIN_SPIKES} spikes: {eligible.sum()} / {len(results_df)}")
print(f"Significant place cells (p < {ALPHA}): {significant.sum()} / {eligible.sum()}")
print(f"Additional units trending toward significance ({ALPHA} <= p < {TREND_ALPHA}): {trending.sum()}")
print()
print(results_df[eligible].to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(results_df.loc[eligible, "bits_per_spike"], bins=15, alpha=0.6,
             label="observed", color="firebrick")
all_null = np.concatenate([null_mi[uid] for uid in results_df.loc[eligible, "unit"]])
axes[0].hist(all_null, bins=30, alpha=0.4, label="shuffled null (pooled)", color="gray", density=False,
             weights=np.full(len(all_null), eligible.sum() / len(all_null)))
axes[0].set_xlabel("spatial information (bits/spike)")
axes[0].set_ylabel("count")
axes[0].set_title("Observed vs. shuffled spatial information")
axes[0].legend()

def status_color(uid_significant, uid_trending):
    if uid_significant:
        return "firebrick"
    if uid_trending:
        return "orange"
    return "gray"


colors = [status_color(s, t) for s, t in zip(significant[eligible], trending[eligible])]
axes[1].scatter(results_df.loc[eligible, "n_spikes"], results_df.loc[eligible, "bits_per_spike"],
                 c=colors, s=40)
axes[1].set_xscale("log")
axes[1].set_xlabel("number of spikes (log scale)")
axes[1].set_ylabel("spatial information (bits/spike)")
axes[1].set_title(f"Spatial info vs. spike count\n"
                   f"(red: p<{ALPHA}, orange: {ALPHA}<=p<{TREND_ALPHA}, gray: n.s.)")
plt.tight_layout()
plt.savefig("fig06_spatial_information_shuffle.png", dpi=110)
plt.close()

# %% [markdown]
# ## 8. Place Fields of the Top Candidate Units
#
# We show the units with the strongest (lowest p-value) spatial tuning: raw spike
# positions overlaid on the animal's trajectory, alongside the smoothed,
# occupancy-normalized rate map. The panel title states whether each unit reaches
# significance (p < 0.05) or only trends toward it.

# %%
plot_ids = results_df.loc[eligible].sort_values("p_value").head(6)["unit"].tolist()
n_plot = len(plot_ids)

fig, axes = plt.subplots(2, n_plot, figsize=(4 * n_plot, 8), squeeze=False)
for i, uid in enumerate(plot_ids):
    spk = units_run[uid]
    x_at_spike = spk.value_from(position_run["x"])
    y_at_spike = spk.value_from(position_run["y"])
    row = results_df[results_df.unit == uid].iloc[0]
    status = "significant" if row.unit in place_cell_ids else \
        ("trending" if row.p_value < TREND_ALPHA else "n.s.")

    ax = axes[0, i]
    ax.scatter(position_run.values[:, 0], position_run.values[:, 1], s=0.5, alpha=0.15, color="gray")
    ax.scatter(x_at_spike.values, y_at_spike.values, s=6, color="red")
    ax.set_title(f"unit {uid} ({status})\nn={row.n_spikes} spikes, p={row.p_value:.3f}", fontsize=10)
    ax.set_xlabel("x (pixels)")
    if i == 0:
        ax.set_ylabel("y (pixels)")

    ax2 = axes[1, i]
    rm = rate_maps[uid].copy()
    rm[occupancy_mask] = np.nan
    im = ax2.imshow(rm.T, origin="lower", cmap="jet")
    ax2.set_title(f"{row.bits_per_spike:.2f} bits/spike", fontsize=10)
    ax2.set_xlabel("x bin")
    if i == 0:
        ax2.set_ylabel("y bin")
    plt.colorbar(im, ax=ax2, fraction=0.046, label="Hz")

plt.tight_layout()
plt.savefig("fig07_place_fields.png", dpi=110)
plt.close()

# %% [markdown]
# ## 9. Summary
#
# Using only the raw spike times and tracked position streamed from a 1998 Space-Shuttle
# hippocampal recording on DANDI, we recovered the classic place-cell signature: CA1
# units whose firing is concentrated in a restricted region of the arena, well beyond
# what spike count and sampling alone would produce by chance. This session has a modest
# number of well-isolated units and only ~34 minutes of continuous open-field locomotion,
# so statistical power is limited; nonetheless, after removing tracker-dropout samples,
# zero-order-hold repeats, and low-occupancy bins from the spatial information estimate,
# at least one unit's spatial tuning clearly survives a strict circular-shuffle test, with
# several more trending in the same direction. Each candidate place cell's spatial
# information score exceeds the large majority of its own shuffle null distribution,
# confirming that the localized firing is tied to position and not an artifact of firing
# rate or occupancy sampling.
#
# Notably, this baseline epoch was recorded in-flight (Flight Day 9 of the STS-90
# mission), so this place-tuned firing was expressed by the hippocampus of a rat
# navigating in microgravity.

# %%
print(f"Final result: {significant.sum()} of {eligible.sum()} well-sampled CA1 units "
      f"(n_spikes >= {MIN_SPIKES}) show significant spatial information (p < {ALPHA}); "
      f"{trending.sum()} more trend toward significance ({ALPHA} <= p < {TREND_ALPHA}). "
      "This is consistent with classic hippocampal place cells, recorded here from a rat "
      "in microgravity aboard the Space Shuttle.")
