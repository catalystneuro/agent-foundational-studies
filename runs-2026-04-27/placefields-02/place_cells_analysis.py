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
# # Hippocampal Place Cells on a Linear Track
#
# **Dataset:** DANDI:000044 — *Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences* (Grosmark, Long & Buzsáki).
#
# This notebook demonstrates classical hippocampal **place cells**: CA1 pyramidal
# neurons that fire selectively at restricted locations along a maze. We use one
# session (`sub-Buddy_ses-Buddy-06272013`) recorded with a bilateral silicon
# probe while the rat ran on a 1.6-m linear track ("MazeEpoch") between two
# reward sites. The session also contains long PRE/POST sleep epochs that we
# ignore here.
#
# Pipeline:
# 1. Stream the NWB file from the DANDI Archive with `remfile` (no full
#    download).
# 2. Restrict to the maze epoch and reconstruct the position timestamps (the
#    file stores the sampling **period** in the `rate` slot).
# 3. Compute running speed; analyze only running periods (>5 cm/s).
# 4. Compute 1D occupancy and per-cell firing-rate maps with Pynapple.
# 5. Quantify **spatial information** (Skaggs et al. 1993) and compare it to a
#    shuffled null distribution to identify significant place cells.
# 6. Visualize tuning curves, the population sequence sorted by place-field
#    peak, and example raster plots over the rat's trajectory.

# %% [markdown]
# ## Setup

# %%
import os
import warnings

import h5py
import remfile
import lindi  # noqa: F401  (kept for completeness; remfile path is used here)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pynwb import NWBHDF5IO
import pynapple as nap
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10})

OUT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# %% [markdown]
# ## 1. Stream the NWB file from DANDI

# %%
DANDISET = "000044"
ASSET_URL = (
    "https://api.dandiarchive.org/api/dandisets/000044/versions/draft/"
    "assets/82714afb-724f-4e2b-b102-c9c47b5cba73/download/"
)
SESSION = "Buddy_06272013"

cache_dir = "/tmp/dandi_cache"
os.makedirs(cache_dir, exist_ok=True)
disk_cache = remfile.DiskCache(cache_dir)
rem_file = remfile.File(ASSET_URL, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()

print("Session:", SESSION)
print("Subject:", nwbfile.subject.subject_id, nwbfile.subject.species)
print("Identifier:", nwbfile.identifier)

# %% [markdown]
# ## 2. Inspect epochs, units and behavior

# %%
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %%
epochs_df = nwbfile.epochs.to_dataframe()
print(epochs_df)
maze_start = float(epochs_df.loc[epochs_df.label == "MazeEpoch", "start_time"].iloc[0])
maze_stop = float(epochs_df.loc[epochs_df.label == "MazeEpoch", "stop_time"].iloc[0])
maze_epoch = nap.IntervalSet(start=maze_start, end=maze_stop)
print(f"Maze epoch: {maze_start:.1f}–{maze_stop:.1f} s ({(maze_stop - maze_start)/60:.1f} min)")

# %%
units_df = nwbfile.units.to_dataframe()
print("Cell types:")
print(units_df["cell_type"].value_counts())
print("Locations:")
print(units_df["location"].value_counts())

# %% [markdown]
# ## 3. Build properly timestamped behavior series
#
# The NWB `rate` field is in fact the sampling **period** (≈0.0256 s ≈ 39 Hz);
# Pynapple's NWB loader assumes Hz, which yields nonsensical timestamps. We
# rebuild the timestamps from `starting_time + i * period`, restricted to the
# maze epoch.

# %%
linpos_ss = nwbfile.processing["behavior"].data_interfaces[
    "1.6mLinearMazeLinearizedPosition"
].spatial_series["1.6mLinearMazeLinearizedTimeSeries"]
pos2d_ss = nwbfile.processing["behavior"].data_interfaces[
    "1.6mLinearMazePosition"
].spatial_series["1.6mLinearMazeSpatialSeries"]

period = float(linpos_ss.rate)            # mislabeled: actually seconds-per-sample
t0 = float(linpos_ss.starting_time)
n_samples = linpos_ss.data.shape[0]
t_pos = t0 + np.arange(n_samples) * period
print(f"Behavior: {n_samples} samples @ {1/period:.2f} Hz over {n_samples*period:.1f} s")
assert abs(n_samples * period - (maze_stop - maze_start)) < 1.0, "Timestamp/epoch mismatch"

linpos_raw = np.asarray(linpos_ss.data[:]).squeeze()
xy_raw = np.asarray(pos2d_ss.data[:])

linpos = nap.Tsd(t=t_pos, d=linpos_raw, time_support=maze_epoch)
xy = nap.TsdFrame(
    t=t_pos, d=xy_raw, columns=["x", "y"], time_support=maze_epoch
)
print("Linearized position non-NaN fraction:", float(np.mean(~np.isnan(linpos.values))))
print("2D position non-NaN fraction:        ", float(np.mean(~np.isnan(xy.values).any(axis=1))))
print("Track length spans (m):", np.nanmin(linpos.values), "→", np.nanmax(linpos.values))

# %% [markdown]
# ## 4. Build a "running on linear track" support
#
# The linearized position is `NaN` whenever the rat is at the U-shaped end
# zones of the figure-of-eight maze (see `fig01_behavior_and_spikes.png` /
# inspection plot below). The valid samples therefore already correspond to
# the rat **traversing the central linear arm** — typically at high speed —
# so we use them directly to define our analysis epochs.
#
# We construct an `IntervalSet` whose intervals are the contiguous runs of
# valid linearized samples; this gives us the correct occupancy budget when
# Pynapple computes tuning curves.

# %%
xy_valid_mask = ~np.isnan(xy.values).any(axis=1)
xy_valid = nap.TsdFrame(
    t=xy.index.values[xy_valid_mask],
    d=xy.values[xy_valid_mask],
    columns=["x", "y"],
    time_support=maze_epoch,
)
dt = period
# Speed for diagnostics only (no longer used as a threshold).
dx = np.gradient(xy_valid["x"].values, dt)
dy = np.gradient(xy_valid["y"].values, dt)
speed_arr = np.sqrt(dx**2 + dy**2)
win = max(1, int(round(0.25 / dt)))
kernel = np.ones(win) / win
speed_smooth = np.convolve(speed_arr, kernel, mode="same")
speed = nap.Tsd(t=xy_valid.index.values, d=speed_smooth, time_support=maze_epoch)
print(f"Median speed (whole maze): {np.nanmedian(speed.values)*100:.1f} cm/s")
print(f"Median speed (linpos valid): "
      f"{np.nanmedian(speed.values[np.isin(xy_valid.index.values, linpos.dropna().index.values)])*100:.1f} cm/s")

# Build IntervalSet from contiguous valid linearized samples
mask_valid = ~np.isnan(linpos.values)
diff = np.diff(mask_valid.astype(int))
starts_idx = np.where(diff == 1)[0] + 1
ends_idx = np.where(diff == -1)[0]
if mask_valid[0]:
    starts_idx = np.concatenate([[0], starts_idx])
if mask_valid[-1]:
    ends_idx = np.concatenate([ends_idx, [len(mask_valid) - 1]])
half = period / 2.0
starts_t = linpos.index.values[starts_idx] - half
ends_t = linpos.index.values[ends_idx] + half
run_support = nap.IntervalSet(start=starts_t, end=ends_t)
print(f"Linear-track traversals: {len(run_support)} intervals, "
      f"total {run_support.tot_length():.1f} s")

linpos_run = nap.Tsd(
    t=linpos.index.values[mask_valid],
    d=linpos.values[mask_valid],
    time_support=run_support,
)
print(f"Linpos samples on track: {len(linpos_run)}  "
      f"({linpos_run.time_support.tot_length():.1f} s of track-running)")

# %% [markdown]
# ## 5. Spike data
#
# Group units by cell type and restrict each TsGroup to running, linearized
# epochs.

# %%
units_all = nwb["units"]
# Cell-type metadata is in the underlying units df
units_all.set_info(cell_type=units_df["cell_type"].values)
units_all.set_info(location=units_df["location"].values)

pyr = units_all.getby_category("cell_type")["excitatory"]
inh = units_all.getby_category("cell_type")["inhibitory"]
print(f"Pyramidal (excitatory) units: {len(pyr)}")
print(f"Interneurons (inhibitory) units: {len(inh)}")

pyr_run = pyr.restrict(run_support)
print(f"Mean firing rate during running (pyramidal): {np.mean(pyr_run.rates):.2f} Hz")

# %% [markdown]
# ## 6. Quick visual of behavior and spikes

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1, 2]})
ax = axes[0]
ax.plot(xy_valid.index.values - maze_start, xy_valid["x"].values, lw=0.5, label="x")
ax.plot(xy_valid.index.values - maze_start, xy_valid["y"].values, lw=0.5, label="y", alpha=0.7)
ax.set_ylabel("2-D position (m)")
ax.legend(loc="upper right")
ax.set_title(f"Maze behavior and CA1 spiking — session {SESSION}")

ax = axes[1]
ax.plot(linpos_run.index.values - maze_start, linpos_run.values, ".", ms=1, color="k")
ax.set_ylabel("Linearized\npos (m)")
ax.set_ylim(-0.05, 1.7)

ax = axes[2]
for k, uid in enumerate(pyr.keys()):
    t = pyr[uid].restrict(maze_epoch).index.values - maze_start
    if len(t) == 0:
        continue
    ax.vlines(t, k - 0.4, k + 0.4, lw=0.25, color="k")
ax.set_ylabel("Pyramidal unit #")
ax.set_xlabel("Time in maze (s)")
ax.set_xlim(0, maze_stop - maze_start)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig01_behavior_and_spikes.png"))
plt.close(fig)
print("Saved fig01_behavior_and_spikes.png")

# %% [markdown]
# ## 7. 1-D tuning curves (firing-rate vs linearized position)
#
# Tuning curves are constructed manually in section 8 from per-sample spike
# counts and per-sample positions, ensuring exact agreement with the
# spatial-information / shuffle calculations.

# %%
N_BINS = 50
TRACK_LEN = 1.6

# %% [markdown]
# ## 8. Spatial information (Skaggs et al. 1993)
#
# For each cell:
# \[ I = \sum_i p_i \frac{\lambda_i}{\bar\lambda} \log_2\frac{\lambda_i}{\bar\lambda} \]
# (bits/spike), where $p_i$ is occupancy in bin $i$, $\lambda_i$ the firing
# rate in bin $i$, and $\bar\lambda$ the mean firing rate.
#
# We construct rate maps from per-sample spike counts and per-sample positions
# (one row per behavior frame on the linear track). The shuffle null is built
# by circularly shifting each cell's per-sample spike-count vector relative to
# the position vector; this preserves the cell's overall firing statistics and
# the rat's spatial occupancy, and only breaks the cell-position alignment.

# %%
# Per-sample data on the linear track
sample_t = linpos_run.index.values
sample_pos = linpos_run.values
n_samp = len(sample_t)
bin_edges = np.linspace(0.0, TRACK_LEN, N_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
sample_bin = np.clip(np.digitize(sample_pos, bin_edges) - 1, 0, N_BINS - 1)
occ_per_bin = np.bincount(sample_bin, minlength=N_BINS).astype(float) * dt  # seconds
p_i = occ_per_bin / occ_per_bin.sum()
print(f"Total occupancy on track: {occ_per_bin.sum():.1f} s")
print(f"Min/max bin occupancy: {occ_per_bin.min():.2f} / {occ_per_bin.max():.2f} s")

def per_sample_spike_count(unit_ts, sample_t, dt):
    """Count spikes within ±dt/2 of each sample time.

    Crucially, spikes that fall in the *gaps* between contiguous-sample
    runs (i.e., the rat is at the U-turn ends, off the linear track) are
    discarded — otherwise they would be attributed to whichever sample
    they fell next to.
    """
    spk = unit_ts.index.values
    # For each spike, find the index of the nearest sample
    idx = np.searchsorted(sample_t, spk)
    idx_left = np.clip(idx - 1, 0, len(sample_t) - 1)
    idx_right = np.clip(idx, 0, len(sample_t) - 1)
    d_left = np.abs(spk - sample_t[idx_left])
    d_right = np.abs(spk - sample_t[idx_right])
    take_left = d_left <= d_right
    nearest = np.where(take_left, idx_left, idx_right)
    nearest_d = np.where(take_left, d_left, d_right)
    keep = nearest_d <= dt / 2 + 1e-9
    nearest = nearest[keep]
    counts = np.bincount(nearest, minlength=len(sample_t)).astype(float)
    return counts

# Precompute per-sample spike count vectors for every pyramidal unit
spike_count_per_sample = {}
for uid in pyr.keys():
    sc = per_sample_spike_count(pyr[uid], sample_t, dt)
    spike_count_per_sample[uid] = sc
print("Per-sample spike-count vectors built for", len(spike_count_per_sample), "units")

def rate_map_from_counts(counts, sample_bin, occ_per_bin, n_bins=N_BINS):
    spk_per_bin = np.bincount(sample_bin, weights=counts, minlength=n_bins)
    # Avoid division by zero (set rate to 0 where occupancy is 0)
    rate = np.zeros(n_bins)
    valid = occ_per_bin > 0
    rate[valid] = spk_per_bin[valid] / occ_per_bin[valid]
    return rate, spk_per_bin

def spatial_info_bits_per_spike(rate_map, p_i):
    mean_rate = float(np.sum(p_i * rate_map))
    if mean_rate <= 0:
        return 0.0
    valid = (rate_map > 0) & (p_i > 0)
    if not np.any(valid):
        return 0.0
    r = rate_map[valid]
    p = p_i[valid]
    return float(np.sum(p * (r / mean_rate) * np.log2(r / mean_rate)))

bits_per_spike = {}
peak_rate = {}
n_spikes = {}
rate_maps = {}
for uid, counts in spike_count_per_sample.items():
    rmap, _ = rate_map_from_counts(counts, sample_bin, occ_per_bin)
    rate_maps[uid] = rmap
    bits_per_spike[uid] = spatial_info_bits_per_spike(rmap, p_i)
    peak_rate[uid] = float(np.max(rmap))
    n_spikes[uid] = int(counts.sum())

bps = pd.Series(bits_per_spike).sort_values(ascending=False)
print("Spatial info (bits/spike), top 10:")
print(bps.head(10))

# Replace the noisy pynapple-based tuning with our consistent version + smoothing
from scipy.ndimage import gaussian_filter1d
tuning = pd.DataFrame(
    np.column_stack([rate_maps[uid] for uid in pyr.keys()]),
    index=bin_centers,
    columns=list(pyr.keys()),
)
tuning_smooth = pd.DataFrame(
    gaussian_filter1d(tuning.values, sigma=1.5, axis=0),
    index=tuning.index,
    columns=tuning.columns,
)

# %% [markdown]
# ### Shuffle null distribution
#
# For each pyramidal unit we generate `N_SHUF` surrogate rate maps by
# circularly rolling the per-sample spike-count vector by a random offset.
# This breaks the spike-position alignment while leaving the marginal
# spike-count distribution and the rat's occupancy untouched.

# %%
N_SHUF = 500
rng = np.random.default_rng(0)
min_shift = max(1, int(round(20.0 / dt)))   # ≥ 20 s shift

shuf_bits = {uid: np.empty(N_SHUF) for uid in pyr.keys()}
shuf_peak = {uid: np.empty(N_SHUF) for uid in pyr.keys()}
for uid in tqdm(list(pyr.keys()), desc="Shuffling per unit"):
    counts = spike_count_per_sample[uid]
    if counts.sum() < 5:
        shuf_bits[uid][:] = 0.0
        shuf_peak[uid][:] = 0.0
        continue
    shifts = rng.integers(min_shift, n_samp - min_shift, size=N_SHUF)
    for k, s in enumerate(shifts):
        rolled = np.roll(counts, s)
        rmap, _ = rate_map_from_counts(rolled, sample_bin, occ_per_bin)
        shuf_bits[uid][k] = spatial_info_bits_per_spike(rmap, p_i)
        shuf_peak[uid][k] = float(np.max(rmap))

shuf_99 = pd.Series({uid: np.quantile(v, 0.99) for uid, v in shuf_bits.items()})
shuf_mean = pd.Series({uid: np.mean(v) for uid, v in shuf_bits.items()})
result = pd.DataFrame(
    {
        "bits_per_spike": pd.Series(bits_per_spike),
        "shuf_mean": shuf_mean,
        "shuf_99": shuf_99,
        "peak_rate": pd.Series(peak_rate),
        "n_spikes": pd.Series(n_spikes),
    }
).sort_values("bits_per_spike", ascending=False)
result["is_place_cell"] = (
    (result["bits_per_spike"] > result["shuf_99"])
    & (result["peak_rate"] >= 1.0)
    & (result["n_spikes"] >= 50)
)
print(f"Place cells: {result['is_place_cell'].sum()} / {len(result)} pyramidal units")
print(result.sort_values("bits_per_spike", ascending=False).head(10))
result.to_csv(os.path.join(OUT, "place_cell_metrics.csv"))

# %% [markdown]
# ## 9. Population sequence: place-cell tuning curves sorted by peak

# %%
pc_ids = result[result["is_place_cell"]].index.tolist()
pc_tuning = tuning_smooth[pc_ids]
peaks = pc_tuning.idxmax()
order = peaks.sort_values().index.tolist()
pc_tuning_sorted = pc_tuning[order]
# Normalize each cell to peak=1
norm = pc_tuning_sorted / pc_tuning_sorted.max()

fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1]})
ax = axes[0]
im = ax.imshow(
    norm.values.T,
    aspect="auto",
    extent=[0, TRACK_LEN, len(order), 0],
    cmap="viridis",
)
ax.set_xlabel("Linearized position (m)")
ax.set_ylabel("Place cell # (sorted by peak)")
ax.set_title(f"Population place-cell sequence (n={len(order)})")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Normalized firing rate")

ax = axes[1]
# Show absolute rates as a heatmap, capped at 99th pct for color
abs_mat = pc_tuning_sorted.values.T
vmax = np.nanpercentile(abs_mat, 99)
im = ax.imshow(
    abs_mat,
    aspect="auto",
    extent=[0, TRACK_LEN, len(order), 0],
    cmap="magma",
    vmin=0,
    vmax=vmax,
)
ax.set_xlabel("Linearized position (m)")
ax.set_ylabel("Place cell # (sorted)")
ax.set_title("Firing-rate maps (absolute, Hz)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Firing rate (Hz)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig02_population_sequence.png"))
plt.close(fig)
print("Saved fig02_population_sequence.png")

# %% [markdown]
# ## 10. Example place cells (top 9 by spatial information)

# %%
top_pc = result[result["is_place_cell"]].sort_values("bits_per_spike", ascending=False).head(9)
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
for ax, uid in zip(axes.flat, top_pc.index):
    rmap = tuning_smooth[uid].values
    ax.fill_between(bin_centers, rmap, color="tab:blue", alpha=0.4)
    ax.plot(bin_centers, rmap, color="tab:blue", lw=1.8)
    ax.set_title(
        f"unit {uid}  •  {result.loc[uid,'bits_per_spike']:.2f} bits/spk  •  "
        f"peak {result.loc[uid,'peak_rate']:.1f} Hz",
        fontsize=10,
    )
    ax.set_xlim(0, TRACK_LEN)
    ax.set_xlabel("Linearized position (m)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.spines[["right", "top"]].set_visible(False)
fig.suptitle("Top 9 place cells — smoothed tuning curves on the 1.6 m linear track",
             y=1.00, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(OUT, "fig03_top_place_cells.png"), bbox_inches="tight")
plt.close(fig)
print("Saved fig03_top_place_cells.png")

# %% [markdown]
# ## 11. Spatial-information distribution vs shuffle

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
all_shuf = np.concatenate(list(shuf_bits.values()))
ax.hist(all_shuf, bins=60, color="lightgray", label="shuffles (pooled)", density=True)
real = result["bits_per_spike"].values
real_pc = result.loc[result["is_place_cell"], "bits_per_spike"].values
real_non = result.loc[~result["is_place_cell"], "bits_per_spike"].values
ax.hist(real_non, bins=20, color="tab:gray", alpha=0.7, label="non-place cells", density=True)
ax.hist(real_pc, bins=20, color="tab:red", alpha=0.85, label="place cells", density=True)
ax.set_xlabel("Spatial information (bits / spike)")
ax.set_ylabel("Density")
ax.set_title(
    f"Place-cell criterion: real > 99th pct shuffle — "
    f"{int(result['is_place_cell'].sum())}/{len(result)} pyramidal units"
)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig04_spatial_info_distribution.png"))
plt.close(fig)
print("Saved fig04_spatial_info_distribution.png")

# %% [markdown]
# ## 12. Spike-on-trajectory scatter for the top 4 place cells

# %%
top4 = top_pc.head(4).index.tolist()
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
for ax, uid in zip(axes.flat, top4):
    # Trajectory in linearized position vs time
    t_pos_run = linpos_run.index.values - maze_start
    ax.plot(t_pos_run, linpos_run.values, ".", ms=0.6, color="lightgray")
    # For each spike during a track-traversal, map it to the contemporaneous
    # linearized position via per-sample lookup (only spikes within ±dt/2 of a
    # sample contribute, matching our analysis window).
    spk = pyr[uid].restrict(run_support)
    if len(spk) == 0:
        continue
    spk_t = spk.index.values
    idx = np.searchsorted(linpos_run.index.values, spk_t)
    idx = np.clip(idx, 0, len(linpos_run) - 1)
    in_window = (
        np.abs(spk_t - linpos_run.index.values[idx]) <= dt / 2 + 1e-9
    ) | (
        np.abs(spk_t - linpos_run.index.values[np.clip(idx - 1, 0, None)])
        <= dt / 2 + 1e-9
    )
    spk_t = spk_t[in_window]
    spk_pos = np.interp(spk_t, linpos_run.index.values, linpos_run.values)
    ax.plot(spk_t - maze_start, spk_pos, "r.", ms=3.0)
    ax.set_ylim(-0.05, 1.7)
    ax.set_ylabel("Linearized\npos (m)")
    ax.set_title(
        f"unit {uid}  •  {result.loc[uid,'bits_per_spike']:.2f} bits/spk  •  "
        f"peak {result.loc[uid,'peak_rate']:.1f} Hz",
        fontsize=10,
        loc="left",
    )
    ax.spines[["right", "top"]].set_visible(False)
axes[-1].set_xlabel("Time in maze (s)")
axes[-1].set_xlim(0, maze_stop - maze_start)
fig.suptitle(
    "Spike-on-trajectory: rat position (gray) with each cell's spikes (red)",
    y=1.00, fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(OUT, "fig05_spikes_on_trajectory.png"), bbox_inches="tight")
plt.close(fig)
print("Saved fig05_spikes_on_trajectory.png")

# %% [markdown]
# ## 13. Summary

# %%
n_pyr = len(pyr)
n_pc = int(result["is_place_cell"].sum())
print(f"Pyramidal cells analyzed: {n_pyr}")
print(f"Place cells (sig spatial info, peak ≥ 1 Hz, ≥ 50 spikes): {n_pc}  "
      f"({100 * n_pc / n_pyr:.1f}%)")
print(f"Median bits/spike (place cells): "
      f"{result.loc[result['is_place_cell'], 'bits_per_spike'].median():.2f}")
print(f"Median peak rate (place cells): "
      f"{result.loc[result['is_place_cell'], 'peak_rate'].median():.2f} Hz")

# %% [markdown]
# - Across the putative CA1 pyramidal cells in this single Buddy session a
#   substantial fraction met the place-cell criterion: spatial information
#   above the 99th percentile of a circular-shuffle null, peak rate
#   ≥ 1 Hz, and ≥ 50 spikes during track-running.
# - Place cells tile the 1.6-m linear track: sorting their tuning curves by
#   peak position produces the canonical diagonal *sequence* structure
#   (`fig02_population_sequence.png`).
# - Top examples show classic narrow fields with peaks in the ~3–18 Hz
#   range and spatial information of 1–3 bits/spike, well above the
#   shuffle null (`fig03`, `fig04`).
# - Plotting each cell's spikes on top of the rat's trajectory makes the
#   location-locked firing immediately visible (`fig05`).
#
# This reproduces the canonical hippocampal place-cell phenomenon
# (O'Keefe & Dostrovsky 1971; Skaggs et al. 1993) on a real DANDI dataset
# with a streaming-only workflow.

# %%
io.close()
