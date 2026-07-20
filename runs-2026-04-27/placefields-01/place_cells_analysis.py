# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hippocampal Place Cells from DANDI Archive
#
# This notebook demonstrates the classic phenomenon of **hippocampal place cells**:
# CA1 pyramidal neurons that fire selectively when an animal occupies a particular
# location in its environment.
#
# **Dataset**: DANDI [000044](https://dandiarchive.org/dandiset/000044/draft) —
# *Grosmark & Buzsáki (2016) "Diversity in neural firing dynamics supports both
# rigid and learned hippocampal sequences." Science 351:1440–1443.*
#
# Long-Evans rats run back and forth on a 1.6 m linear track for water reward
# while bilateral silicon probes record from dorsal hippocampus (CA1).
#
# We use a single session (`sub-Buddy_ses-Buddy-06272013`) and analyse units
# during the `MazeEpoch` (10717.9 – 13046.0 s).
#
# **Pipeline**
# 1. Stream the NWB file from S3 (no full download).
# 2. Reconstruct the position time-series.
# 3. Compute running speed, separate left- vs right-bound traversals.
# 4. Compute 1-D occupancy and per-cell firing-rate maps for each running direction.
# 5. Score each putative pyramidal neuron with spatial information (bits/spike).
# 6. Classify and visualise place cells (rate maps, raster on track, summary stats).

# %% [markdown]
# ## 1 — Setup

# %%
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

OUTDIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
os.makedirs(OUTDIR, exist_ok=True)

# %% [markdown]
# ## 2 — Stream the NWB file

# %%
S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/98b/25c/98b25cb1-310c-45f7-97cc-669fce2057b7"
CACHE_DIR = "/tmp/remfile_cache_placefields"
os.makedirs(CACHE_DIR, exist_ok=True)

disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
print(f"Subject: {nwbfile.subject.subject_id} | Session: {nwbfile.session_id}")
print(f"Lab: {nwbfile.lab} | Experimenter: {nwbfile.experimenter}")

# %% [markdown]
# ## 3 — Behavioural epochs and units

# %%
epochs_df = nwbfile.epochs.to_dataframe()
print(epochs_df)

units_df = nwbfile.units.to_dataframe()
print(f"\n{len(units_df)} units recorded")
print(units_df["cell_type"].value_counts())
print(units_df["location"].value_counts())

# Maze epoch — only time when the rat is running on the linear track
maze_start, maze_end = epochs_df.loc[epochs_df["label"] == "MazeEpoch", ["start_time", "stop_time"]].values[0]
maze_epoch = nap.IntervalSet(start=maze_start, end=maze_end)
print(f"\nMaze epoch: {maze_start:.1f} – {maze_end:.1f} s ({maze_epoch.tot_length():.1f} s)")

# %% [markdown]
# ## 4 — Reconstruct linearised position
#
# The NWB `SpatialSeries.rate` field in this file actually stores the sampling
# *period* (≈ 0.0256 s ≡ ~39 Hz). We rebuild the timestamps explicitly and wrap
# in a Pynapple `Tsd`.

# %%
lin_obj = nwbfile.processing["behavior"]["1.6mLinearMazeLinearizedPosition"][
    "1.6mLinearMazeLinearizedTimeSeries"
]
data = lin_obj.data[:].squeeze()
t0 = lin_obj.starting_time
period = lin_obj.rate
n = len(data)
print(f"n samples: {n}, t0: {t0:.1f}s, period: {period:.4f}s, fs: {1/period:.2f} Hz")

t = t0 + np.arange(n) * period

# Drop NaNs (samples when animal is off the linear segment)
valid = ~np.isnan(data)
lin_pos = nap.Tsd(t=t[valid], d=data[valid], time_support=maze_epoch)
print(f"Valid samples on track: {len(lin_pos)} ({100*valid.mean():.1f}% of maze)")
print(f"Position range: {lin_pos.values.min():.2f} – {lin_pos.values.max():.2f} m")

# Valid samples are sparse — they form contiguous *segments* on the linear
# track separated by large gaps (the rat leaves the linear segment between
# laps). Build an IntervalSet of those continuous on-track segments so that
# all subsequent velocity / occupancy / spike-count computations only use
# time spans during which we actually know the animal's linearised position.
GAP_THR = 5 * period  # treat any gap > 5 frames as a discontinuity
gaps = np.diff(lin_pos.t)
seg_breaks = np.where(gaps > GAP_THR)[0]
seg_starts_idx = np.concatenate([[0], seg_breaks + 1])
seg_ends_idx = np.concatenate([seg_breaks, [len(lin_pos) - 1]])
seg_starts = lin_pos.t[seg_starts_idx]
seg_ends = lin_pos.t[seg_ends_idx]
# Keep only segments lasting at least 0.5 s (long enough to estimate velocity)
seg_dur = seg_ends - seg_starts
keep = seg_dur >= 0.5
on_track = nap.IntervalSet(start=seg_starts[keep], end=seg_ends[keep])
print(f"Continuous on-track segments: {len(on_track)} (total {on_track.tot_length():.1f} s)")

# %% [markdown]
# ## 5 — Running speed and direction
#
# Velocity is estimated only within continuous on-track segments. We then
# split each segment into right-bound (`v > +threshold`) and left-bound
# (`v < -threshold`) sub-intervals.

# %%
SPEED_THR = 0.05  # 5 cm/s — animal is "running"

velocities = np.full(len(lin_pos), np.nan)
for s_i, e_i in zip(seg_starts_idx[keep], seg_ends_idx[keep]):
    if e_i - s_i < 2:
        continue
    seg_t = lin_pos.t[s_i:e_i + 1]
    seg_x = lin_pos.values[s_i:e_i + 1]
    # Central differences with edge handling
    v = np.gradient(seg_x, seg_t)
    velocities[s_i:e_i + 1] = v

velocity = nap.Tsd(t=lin_pos.t, d=velocities, time_support=maze_epoch)
# Smooth velocity within on-track segments only — restrict before smoothing
velocity = velocity.restrict(on_track)
velocity = velocity.smooth(std=0.25, size_factor=20)
speed = nap.Tsd(t=velocity.t, d=np.abs(velocity.values), time_support=on_track)

moving_mask = speed.values > SPEED_THR
print(f"Moving fraction (within on-track segments): {moving_mask.mean():.2%}")


def _direction_intervals(v_tsd, sign, thr):
    """Build IntervalSet of contiguous samples where sign*v > thr,
       respecting the velocity's IntervalSet time-support."""
    starts_all, ends_all = [], []
    for s, e in zip(v_tsd.time_support.start, v_tsd.time_support.end):
        seg = v_tsd.get(s, e)
        if len(seg) < 2:
            continue
        m = (sign * seg.values) > thr
        if m.sum() == 0:
            continue
        edges = np.diff(m.astype(int), prepend=0, append=0)
        seg_starts = seg.t[np.where(edges[:-1] == 1)[0]]
        seg_ends = seg.t[np.where(edges[1:] == -1)[0]]
        starts_all.append(seg_starts)
        ends_all.append(seg_ends)
    if not starts_all:
        return nap.IntervalSet(start=[], end=[])
    s = np.concatenate(starts_all)
    e = np.concatenate(ends_all)
    keep = e > s
    return nap.IntervalSet(start=s[keep], end=e[keep])


right_runs = _direction_intervals(velocity, +1, SPEED_THR)
left_runs = _direction_intervals(velocity, -1, SPEED_THR)
# Drop very short blips (< 0.25 s)
right_runs = right_runs.drop_short_intervals(0.25)
left_runs = left_runs.drop_short_intervals(0.25)
print(f"Right-bound traversals: {len(right_runs)} intervals, total {right_runs.tot_length():.1f} s")
print(f"Left-bound  traversals: {len(left_runs)}  intervals, total {left_runs.tot_length():.1f} s")

# %% [markdown]
# ## 6 — Behaviour visualisation

# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(lin_pos.t, lin_pos.values, ".", ms=1.5, color="k")
ax[0].set_ylabel("Linearised position (m)")
ax[0].set_title("Linear-track behaviour — sub-Buddy 2013-06-27")
ax[1].plot(velocity.t, velocity.values, lw=0.6, color="steelblue")
ax[1].axhline(SPEED_THR, color="r", lw=0.6, ls="--", label="+threshold")
ax[1].axhline(-SPEED_THR, color="r", lw=0.6, ls="--", label="-threshold")
ax[1].set_ylabel("Velocity (m/s)")
ax[1].set_xlabel("Time (s)")
ax[1].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig02_behaviour.png"), dpi=130)
plt.close()

# %% [markdown]
# ## 7 — Spike-time TsGroup
#
# Restrict to putative excitatory pyramidal cells in CA1. We further keep only
# units with a reasonable maze-epoch firing rate (between 0.1 and 10 Hz —
# excludes very-low-yield clusters and likely interneurons).

# %%
spike_dict = {int(uid): np.asarray(row.spike_times)
              for uid, row in units_df.iterrows()}
metadata = pd.DataFrame({
    "cell_type": units_df["cell_type"].values,
    "location": units_df["location"].values,
    "shank_id": units_df["shank_id"].values,
}, index=units_df.index.astype(int))

units = nap.TsGroup(spike_dict, metadata=metadata)
units_maze = units.restrict(maze_epoch)

# Maze-epoch rates
maze_rates = pd.Series(
    {uid: len(units_maze[uid]) / maze_epoch.tot_length() for uid in units_maze.keys()}
)
metadata["maze_rate_hz"] = maze_rates

is_pyr = (
    (metadata["cell_type"] == "excitatory")
    & (metadata["maze_rate_hz"] > 0.1)
    & (metadata["maze_rate_hz"] < 10.0)
)
pyr_ids = metadata.index[is_pyr].tolist()
print(f"Putative pyramidal cells passing rate filter: {len(pyr_ids)} / {len(metadata)}")

units_pyr = units_maze.getby_threshold("rate", 0.0)  # all
units_pyr = nap.TsGroup({uid: units_maze[uid] for uid in pyr_ids},
                        time_support=maze_epoch)

# %% [markdown]
# ## 8 — 1-D firing-rate maps
#
# Standard linear-track place-field analysis:
#
# * 64 spatial bins from 0 to 1.6 m.
# * Compute occupancy and spike count per bin separately for right- and left-bound
#   traversals (place fields on linear tracks are usually direction-specific).
# * Smooth with a Gaussian (~5 cm).

# %%
N_BINS = 64
TRACK_LEN = 1.6
bin_edges = np.linspace(0.0, TRACK_LEN, N_BINS + 1)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
SMOOTH_BINS = 2  # ~5 cm

from scipy.ndimage import gaussian_filter1d


def rate_map_1d(spike_times, lin_pos_tsd, run_intervals, bin_edges, smooth=SMOOTH_BINS):
    """Compute a 1-D firing-rate map restricted to a set of running intervals."""
    pos = lin_pos_tsd.restrict(run_intervals)
    if len(pos) == 0:
        return np.full(len(bin_edges) - 1, np.nan)
    # Occupancy in seconds per bin
    occ_counts, _ = np.histogram(pos.values, bins=bin_edges)
    dt_pos = period
    occupancy = occ_counts * dt_pos
    # Spike positions: interpolate position at each spike time inside the runs
    spk = nap.Ts(t=spike_times).restrict(run_intervals)
    if len(spk) == 0 or occupancy.sum() == 0:
        return np.full(len(bin_edges) - 1, 0.0)
    spk_pos = np.interp(spk.t, pos.t, pos.values, left=np.nan, right=np.nan)
    spk_pos = spk_pos[~np.isnan(spk_pos)]
    spk_counts, _ = np.histogram(spk_pos, bins=bin_edges)
    rate = np.divide(spk_counts, occupancy,
                     out=np.zeros_like(occupancy, dtype=float),
                     where=occupancy > 0)
    rate = gaussian_filter1d(rate, sigma=smooth)
    rate[occupancy == 0] = np.nan
    return rate


# Pre-compute occupancy distributions (for spatial information)
def _occupancy(lin_pos_tsd, run_intervals, bin_edges):
    pos = lin_pos_tsd.restrict(run_intervals)
    occ_counts, _ = np.histogram(pos.values, bins=bin_edges)
    occupancy = occ_counts * period
    return occupancy


occ_R = _occupancy(lin_pos, right_runs, bin_edges)
occ_L = _occupancy(lin_pos, left_runs, bin_edges)
print(f"Right-run occupancy (s): total={occ_R.sum():.1f}, "
      f"min={occ_R[occ_R > 0].min():.2f}, max={occ_R.max():.2f}")
print(f"Left-run  occupancy (s): total={occ_L.sum():.1f}, "
      f"min={occ_L[occ_L > 0].min():.2f}, max={occ_L.max():.2f}")

# %%
maps_R = {}
maps_L = {}
for uid in tqdm(pyr_ids, desc="Computing rate maps"):
    spk = units_pyr[uid].t
    maps_R[uid] = rate_map_1d(spk, lin_pos, right_runs, bin_edges)
    maps_L[uid] = rate_map_1d(spk, lin_pos, left_runs, bin_edges)

maps_R_arr = np.vstack([maps_R[u] for u in pyr_ids])
maps_L_arr = np.vstack([maps_L[u] for u in pyr_ids])

# %% [markdown]
# ## 9 — Spatial information (Skaggs et al. 1993)
#
# $$\mathrm{SI}\ (\text{bits/spike}) = \sum_i p_i \frac{\lambda_i}{\bar\lambda}
# \log_2 \frac{\lambda_i}{\bar\lambda}$$
#
# where $p_i$ is occupancy probability of bin $i$, $\lambda_i$ is the firing
# rate in bin $i$, and $\bar\lambda$ is the mean rate across the track.

# %%
def spatial_information(rate_map, occupancy):
    rate_map = np.asarray(rate_map, dtype=float)
    occupancy = np.asarray(occupancy, dtype=float)
    valid = (~np.isnan(rate_map)) & (occupancy > 0)
    if valid.sum() == 0:
        return np.nan
    p = occupancy[valid] / occupancy[valid].sum()
    lam = rate_map[valid]
    mean_lam = (p * lam).sum()
    if mean_lam <= 0:
        return 0.0
    ratio = lam / mean_lam
    safe = ratio > 0
    return float((p[safe] * ratio[safe] * np.log2(ratio[safe])).sum())


# SI for each direction; place-cell criterion: SI > 0.5 bits/spike in either direction
si_R = np.array([spatial_information(maps_R[u], occ_R) for u in pyr_ids])
si_L = np.array([spatial_information(maps_L[u], occ_L) for u in pyr_ids])
peak_R = np.array([np.nanmax(maps_R[u]) for u in pyr_ids])
peak_L = np.array([np.nanmax(maps_L[u]) for u in pyr_ids])

results = pd.DataFrame({
    "unit_id": pyr_ids,
    "maze_rate_hz": [metadata.loc[u, "maze_rate_hz"] for u in pyr_ids],
    "SI_right_bits": si_R,
    "SI_left_bits": si_L,
    "peak_rate_R": peak_R,
    "peak_rate_L": peak_L,
}).set_index("unit_id")

SI_THR = 0.5
PEAK_THR = 1.0  # Hz
results["is_place_cell"] = (
    ((results["SI_right_bits"] >= SI_THR) & (results["peak_rate_R"] >= PEAK_THR))
    | ((results["SI_left_bits"] >= SI_THR) & (results["peak_rate_L"] >= PEAK_THR))
)
n_place = int(results["is_place_cell"].sum())
print(f"Place cells: {n_place} / {len(results)} pyramidal units "
      f"({100*n_place/len(results):.1f}%)")
results.to_csv(os.path.join(OUTDIR, "place_cell_metrics.csv"))

# %% [markdown]
# ## 10 — Visualise individual place cells
#
# Top 12 by maximum spatial information (either direction). Each row shows:
# left-bound rate map, right-bound rate map, and a spike-on-track raster
# coloured by direction.

# %%
order = results["is_place_cell"]
top = (
    results[order]
    .assign(SI_max=lambda d: d[["SI_right_bits", "SI_left_bits"]].max(axis=1))
    .sort_values("SI_max", ascending=False)
    .head(12)
)
print(top)

n = len(top)
ncols = 3
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.6 * nrows))
axes = np.atleast_2d(axes)

for idx, (uid, row) in enumerate(top.iterrows()):
    ax = axes[idx // ncols, idx % ncols]
    ax.plot(bin_centres, maps_R[uid], lw=2, color="C3", label="→ right-bound")
    ax.plot(bin_centres, maps_L[uid], lw=2, color="C0", label="← left-bound")
    ax.set_title(f"unit {uid}  |  SI(R)={row.SI_right_bits:.2f} "
                 f"SI(L)={row.SI_left_bits:.2f}", fontsize=9)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Rate (Hz)")
    ax.set_xlim(0, TRACK_LEN)
    if idx == 0:
        ax.legend(fontsize=8, loc="upper right")

# Hide any unused panels
for j in range(n, nrows * ncols):
    axes[j // ncols, j % ncols].axis("off")
plt.suptitle("Top 12 place cells — direction-specific rate maps", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig03_top_place_cells.png"), dpi=130, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## 11 — Spike-on-track raster for an exemplar place cell

# %%
exemplar = top.index[0]
spk_t = units_pyr[exemplar].t
spk_pos = np.interp(spk_t, lin_pos.t, lin_pos.values, left=np.nan, right=np.nan)
spk_dir = np.interp(spk_t, velocity.t, velocity.values, left=np.nan, right=np.nan)

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1]})
ax[0].plot(lin_pos.t, lin_pos.values, color="0.7", lw=0.4)
right_spk = (spk_dir > 0) & ~np.isnan(spk_pos)
left_spk = (spk_dir < 0) & ~np.isnan(spk_pos)
ax[0].scatter(spk_t[right_spk], spk_pos[right_spk], s=8, color="C3",
              label="spike (right-bound)")
ax[0].scatter(spk_t[left_spk], spk_pos[left_spk], s=8, color="C0",
              label="spike (left-bound)")
ax[0].set_ylabel("Linearised position (m)")
ax[0].set_title(f"Spike raster on track — unit {exemplar}")
ax[0].legend(loc="upper right", fontsize=9)

ax[1].plot(bin_centres, maps_R[exemplar], color="C3", lw=2, label="right-bound rate")
ax[1].plot(bin_centres, maps_L[exemplar], color="C0", lw=2, label="left-bound rate")
ax[1].set_xlabel("Time (s)  |  bottom: position (m) x-axis is shared above only with time")
ax[1].set_ylabel("Rate (Hz)")
# The bottom panel shows rate vs position, share x is misleading - separate it
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig04_exemplar_raster.png"), dpi=130)
plt.close()

# Cleaner exemplar plot (separate axes for time-raster and rate-vs-position)
fig, axes = plt.subplots(1, 2, figsize=(13, 4),
                         gridspec_kw={"width_ratios": [2.4, 1]})
ax = axes[0]
ax.plot(lin_pos.t, lin_pos.values, color="0.75", lw=0.5)
ax.scatter(spk_t[right_spk], spk_pos[right_spk], s=10, color="C3",
           label="spike (right-bound)")
ax.scatter(spk_t[left_spk], spk_pos[left_spk], s=10, color="C0",
           label="spike (left-bound)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Linearised position (m)")
ax.set_title(f"Exemplar place cell {exemplar} — spikes on the track")
ax.legend(loc="upper right", fontsize=9)

ax = axes[1]
ax.plot(maps_R[exemplar], bin_centres, color="C3", lw=2, label="right-bound")
ax.plot(maps_L[exemplar], bin_centres, color="C0", lw=2, label="left-bound")
ax.set_xlabel("Firing rate (Hz)")
ax.set_ylabel("Position (m)")
ax.set_title("Tuning curves")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig04_exemplar_raster.png"), dpi=130)
plt.close()

# %% [markdown]
# ## 12 — Population summary
#
# *(a)* Distribution of spatial information bits/spike across all pyramidal cells.
#
# *(b)* Population rate-map heatmap, sorted by peak position. The diagonal "stripe"
# is the canonical signature of a population of place cells tiling the track.

# %%
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# (a) SI histogram
ax = fig.add_subplot(gs[0, 0])
ax.hist(np.maximum(si_R, si_L), bins=25, color="steelblue", edgecolor="k")
ax.axvline(SI_THR, color="r", ls="--", label=f"threshold = {SI_THR} bits/spike")
ax.set_xlabel("Spatial information (bits/spike, max over direction)")
ax.set_ylabel("# units")
ax.set_title(f"(a) Spatial information — {n_place}/{len(results)} pass threshold")
ax.legend(fontsize=9)

# (b) Place-cell count vs maze rate
ax = fig.add_subplot(gs[0, 1])
ax.scatter(results["maze_rate_hz"], np.maximum(si_R, si_L),
           c=results["is_place_cell"].map({True: "C3", False: "0.5"}),
           s=18, edgecolor="k", lw=0.3)
ax.axhline(SI_THR, color="r", ls="--", lw=0.7)
ax.set_xscale("log")
ax.set_xlabel("Mean firing rate during maze epoch (Hz)")
ax.set_ylabel("Spatial info (bits/spike)")
ax.set_title("(b) SI vs rate (red = place cell)")

# (c) Sorted population rate-map heatmap — right-bound
def _sort_rate_maps(arr, mask):
    """Z-score each row and sort by argmax of place cells, place rest below."""
    arr = np.array(arr, copy=True)
    arr_norm = arr / np.nanmax(arr, axis=1, keepdims=True)
    arr_norm[np.isnan(arr_norm)] = 0
    place_idx = np.where(mask)[0]
    other_idx = np.where(~mask)[0]
    peak = np.nanargmax(arr_norm[place_idx], axis=1)
    place_sorted = place_idx[np.argsort(peak)]
    return np.concatenate([place_sorted, other_idx]), arr_norm


is_place_arr = results["is_place_cell"].values
order_R, norm_R = _sort_rate_maps(maps_R_arr, is_place_arr)
order_L, norm_L = _sort_rate_maps(maps_L_arr, is_place_arr)

ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(norm_R[order_R], aspect="auto", origin="lower",
               extent=[0, TRACK_LEN, 0, len(order_R)], cmap="magma",
               vmin=0, vmax=1)
ax.axhline(is_place_arr.sum(), color="cyan", lw=1.0, ls="--")
ax.set_xlabel("Position (m)")
ax.set_ylabel("Unit (sorted by peak)")
ax.set_title("(c) Right-bound population rate maps  (above dashed line: place cells)")
plt.colorbar(im, ax=ax, label="Normalised rate")

ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(norm_L[order_L], aspect="auto", origin="lower",
               extent=[0, TRACK_LEN, 0, len(order_L)], cmap="magma",
               vmin=0, vmax=1)
ax.axhline(is_place_arr.sum(), color="cyan", lw=1.0, ls="--")
ax.set_xlabel("Position (m)")
ax.set_ylabel("Unit (sorted by peak)")
ax.set_title("(d) Left-bound population rate maps")
plt.colorbar(im, ax=ax, label="Normalised rate")

plt.suptitle("Hippocampal CA1 place-cell population — sub-Buddy 2013-06-27 (DANDI 000044)",
             fontsize=13)
plt.savefig(os.path.join(OUTDIR, "fig05_population.png"), dpi=130, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## 13 — Place-field width and peak position
#
# For each place cell we compute:
# * Peak firing rate
# * Place-field centre (location of peak rate)
# * Place-field width (FWHM of the rate map around the peak)

# %%
def field_metrics(rate_map, bin_centres):
    rm = np.array(rate_map, copy=True)
    rm[np.isnan(rm)] = 0.0
    if rm.max() <= 0:
        return np.nan, np.nan, np.nan
    peak_idx = int(np.argmax(rm))
    peak_rate = float(rm[peak_idx])
    half = peak_rate / 2.0
    above = rm >= half
    # Find contiguous block containing peak
    left = peak_idx
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak_idx
    while right < len(rm) - 1 and above[right + 1]:
        right += 1
    width = float(bin_centres[right] - bin_centres[left])
    return peak_rate, float(bin_centres[peak_idx]), width


metrics = []
for uid in pyr_ids:
    pr_R, pc_R, w_R = field_metrics(maps_R[uid], bin_centres)
    pr_L, pc_L, w_L = field_metrics(maps_L[uid], bin_centres)
    metrics.append((uid, pr_R, pc_R, w_R, pr_L, pc_L, w_L))

m_df = pd.DataFrame(
    metrics,
    columns=["unit_id", "peak_R", "centre_R", "width_R",
             "peak_L", "centre_L", "width_L"]
).set_index("unit_id")
results = results.join(m_df)
results.to_csv(os.path.join(OUTDIR, "place_cell_metrics.csv"))

place_mask = results["is_place_cell"].values
widths = np.concatenate([
    results.loc[place_mask & (results["peak_R"] >= PEAK_THR), "width_R"].values,
    results.loc[place_mask & (results["peak_L"] >= PEAK_THR), "width_L"].values,
])
centres = np.concatenate([
    results.loc[place_mask & (results["peak_R"] >= PEAK_THR), "centre_R"].values,
    results.loc[place_mask & (results["peak_L"] >= PEAK_THR), "centre_L"].values,
])
print(f"Place-field width (FWHM): mean={np.nanmean(widths)*100:.1f} cm, "
      f"median={np.nanmedian(widths)*100:.1f} cm, n={len(widths)}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(widths * 100, bins=20, color="seagreen", edgecolor="k")
ax[0].set_xlabel("Place-field width (cm, FWHM)")
ax[0].set_ylabel("# fields")
ax[0].set_title("(a) Place-field width distribution")

ax[1].hist(centres * 100, bins=20, color="indigo", edgecolor="k")
ax[1].set_xlabel("Place-field centre (cm along track)")
ax[1].set_ylabel("# fields")
ax[1].set_title("(b) Place-field centre distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig06_field_widths.png"), dpi=130)
plt.close()

# %% [markdown]
# ## 14 — Summary
#
# We streamed a single linear-track session from the Grosmark & Buzsáki (2016)
# Buzsaki-lab dataset on DANDI Archive, computed direction-specific 1-D rate
# maps for all putative pyramidal CA1 cells, scored each cell with Skaggs
# spatial information, and visualised the resulting place-cell population.
#
# Key results (printed to stdout above):
# * Number of place cells passing SI ≥ 0.5 bits/spike and peak ≥ 1 Hz.
# * Median FWHM place-field width.
# * Sorted population rate-map heatmap shows the canonical "diagonal stripe"
#   indicating place fields tiling the 1.6 m track.

# %%
print("Done.")
print(f"Outputs in: {OUTDIR}")
for f in sorted(os.listdir(OUTDIR)):
    if f.endswith((".png", ".csv", ".py", ".ipynb", ".md")):
        print("  ", f)
