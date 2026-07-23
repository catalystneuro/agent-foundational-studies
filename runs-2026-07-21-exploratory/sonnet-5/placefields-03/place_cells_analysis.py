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
# # Hippocampal Place Cells in Rat CA1 During Linear-Track Running
#
# This notebook demonstrates the classic hippocampal place cell phenomenon
# using real extracellular recordings from the DANDI Archive.
#
# **Dataset**: [DANDI:000044](https://dandiarchive.org/dandiset/000044) --
# "Diversity in neural firing dynamics supports both rigid and learned
# hippocampal sequences" (Grosmark & Buzsaki). Silicon-probe recordings were
# made from dorsal CA1 of rats running back and forth on a 1.6 m linear
# track for water reward at each end. We use session
# `sub-Buddy/sub-Buddy_ses-Buddy-06272013`.
#
# **Phenomenon**: Place cells are pyramidal neurons in the hippocampus that
# fire selectively when an animal occupies a specific location in its
# environment (the cell's "place field"). We identify place cells by
# computing spatial tuning curves of firing rate vs. position on the track,
# quantifying spatial information content (Skaggs et al. 1993), and testing
# significance against a shuffled null distribution.
#
# The NWB file is streamed directly from S3 (no full download) using
# `remfile`, and all spike/behavioral analysis uses `pynapple`.

# %% [markdown]
# ## Setup and Data Loading

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
from tqdm import tqdm

np.random.seed(0)
plt.rcParams["figure.dpi"] = 100

S3_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/98b/25c/"
    "98b25cb1-310c-45f7-97cc-669fce2057b7"
)  # sub-Buddy_ses-Buddy-06272013_behavior+ecephys.nwb (DANDI:000044)

# %% [markdown]
# Stream the NWB file with `remfile`, using a local disk cache so repeated
# reads of the same byte ranges don't re-download from S3.

# %%
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# ## Inspect Structure
#
# The file contains spike times (`units`), local field potentials (`LFP`),
# behavioral epochs, brain states (sleep scoring), and both raw (x, y) and
# linearized position on the maze.

# %%
units = nwb["units"]
epochs = nwb["epochs"]
print(f"{len(units)} units total")
print(units.metadata_columns)
print(epochs)

# %% [markdown]
# The recording spans a PRE-sleep epoch, a MazeEpoch of active running, and
# a POST-sleep epoch. We restrict all analysis to `MazeEpoch`.

# %%
maze_ep = epochs[epochs.label == "MazeEpoch"]
print(maze_ep)
print(f"MazeEpoch duration: {maze_ep.tot_length():.1f} s")

# %% [markdown]
# ### Fixing the position timestamps
#
# The `SpatialSeries` objects in this file store a `rate` attribute that is
# numerically the sampling *period* (in seconds) rather than the sampling
# *frequency* (in Hz) -- i.e. `rate` and `1/rate` are swapped. We verify this
# directly: dividing the number of position samples by the stored "rate"
# implies a session lasting ~41 days, whereas dividing by `1/rate` gives
# 2328.1 s, which matches the MazeEpoch duration exactly. We therefore invert
# the stored value to reconstruct correct timestamps rather than trusting
# pynapple's default (buggy) parsing of this field.

# %%
pos_grp = h5py_file["processing"]["behavior"]["1.6mLinearMazePosition"][
    "1.6mLinearMazeSpatialSeries"
]
stored_rate = pos_grp["starting_time"].attrs["rate"]
starting_time = pos_grp["starting_time"][()]
true_rate = 1.0 / stored_rate  # Hz, correcting the inverted metadata
pos_data = pos_grp["data"][:]  # (N, 2) array of (x, y) in meters
n_samples = pos_data.shape[0]
pos_time = starting_time + np.arange(n_samples) / true_rate

print(f"stored 'rate' field: {stored_rate:.6f} (looks like seconds/sample)")
print(f"corrected sampling rate: {true_rate:.3f} Hz")
implied_duration = n_samples / true_rate
print(f"implied duration with corrected rate: {implied_duration:.1f} s")
print(f"MazeEpoch duration: {maze_ep.tot_length():.1f} s (matches)")

# %% [markdown]
# A small cluster of tracked positions falls well outside the maze
# (`y < -1.5`), a brief tracking artifact. We drop NaNs and this outlier
# cluster, then build a `pynapple.TsdFrame` restricted to `MazeEpoch`.

# %%
valid = (
    ~np.isnan(pos_data[:, 0])
    & ~np.isnan(pos_data[:, 1])
    & (pos_data[:, 1] > -1.5)
)
position = nap.TsdFrame(
    t=pos_time[valid],
    d=pos_data[valid],
    columns=["x", "y"],
    time_support=maze_ep,
)
print(position)

# %% [markdown]
# ## Raw Data Visualization
#
# Before any analysis, inspect the raw behavioral trajectory, a snippet of
# LFP, and a spike raster to confirm the data look sensible.

# %%
lfp_grp = h5py_file["processing"]["ecephys"]["LFP"]["LFP"]
lfp_rate = lfp_grp["starting_time"].attrs["rate"]
lfp_start = lfp_grp["starting_time"][()]
snippet_start_s = maze_ep.start[0] + 60
sample0 = int((snippet_start_s - lfp_start) * lfp_rate)
n_lfp_samples = int(5 * lfp_rate)
lfp_snippet = lfp_grp["data"][sample0 : sample0 + n_lfp_samples, :4]
lfp_t = snippet_start_s + np.arange(n_lfp_samples) / lfp_rate

raster_window = nap.IntervalSet(start=snippet_start_s, end=snippet_start_s + 20)
spikes_window = units.restrict(raster_window)
traj_window = position.restrict(raster_window)

fig = plt.figure(figsize=(13, 9))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

ax0 = fig.add_subplot(gs[0, :])
for i in range(4):
    ax0.plot(lfp_t, lfp_snippet[:, i] / 1000 + i * 2, lw=0.5, color="k")
ax0.set_yticks([])
ax0.set_xlabel("Time (s)")
ax0.set_title("Raw LFP, 4 example channels (5 s snippet)")

ax1 = fig.add_subplot(gs[1, :])
for i, uid in enumerate(spikes_window.index):
    st = spikes_window[uid].times()
    ax1.plot(st, np.full_like(st, i), "|", color="k", markersize=4)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Unit #")
ax1.set_title(f"Spike raster, all {len(spikes_window)} units (20 s window)")

ax2 = fig.add_subplot(gs[2, 0])
ax2.plot(position["x"].values, position["y"].values, lw=0.2, alpha=0.4, color="gray")
ax2.plot(traj_window["x"].values, traj_window["y"].values, lw=1.5, color="crimson")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
ax2.set_title("Trajectory: full session (gray) + 20 s window (red)")
ax2.set_aspect("equal")

ax3 = fig.add_subplot(gs[2, 1])
ax3.plot(traj_window.t - traj_window.t[0], traj_window["x"].values, label="x")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("x position (m)")
ax3.set_title("Position along track (20 s window)")

fig.savefig("fig1_raw_data_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Preprocessing: Running Speed and Epoch Selection
#
# Place field analysis is restricted to periods of active locomotion,
# excluding immobility (during which firing can reflect sharp-wave ripple
# replay rather than the animal's instantaneous position). We compute
# running speed from the smoothed 2D trajectory and threshold at 5 cm/s.

# %%
pos_smooth = position.smooth(std=0.2)
dxy = np.diff(pos_smooth.values, axis=0)
dt = np.diff(pos_smooth.t)
speed_vals = np.sqrt((dxy**2).sum(axis=1)) / dt
speed = nap.Tsd(t=pos_smooth.t[1:], d=speed_vals, time_support=maze_ep)

SPEED_THRESHOLD = 0.05  # m/s
run_ep = (
    speed.threshold(SPEED_THRESHOLD)
    .time_support.drop_short_intervals(0.3)
    .merge_close_intervals(0.3)
)
print(f"Running epochs: {len(run_ep)} intervals, {run_ep.tot_length():.1f} s total")
print(f"({100 * run_ep.tot_length() / maze_ep.tot_length():.0f}% of MazeEpoch)")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(speed.values, bins=100, color="steelblue")
axes[0].axvline(SPEED_THRESHOLD, color="crimson", ls="--", label="5 cm/s threshold")
axes[0].set_xlabel("Speed (m/s)")
axes[0].set_ylabel("Count")
axes[0].set_title("Running speed distribution")
axes[0].set_xlim(0, 1)
axes[0].legend()

window2 = nap.IntervalSet(start=maze_ep.start[0] + 60, end=maze_ep.start[0] + 120)
sp_win = speed.restrict(window2)
run_win = run_ep.intersect(window2)
axes[1].plot(sp_win.t - window2.start[0], sp_win.values, color="steelblue", lw=0.8)
axes[1].axhline(SPEED_THRESHOLD, color="crimson", ls="--")
for s, e in zip(run_win.start, run_win.end):
    axes[1].axvspan(s - window2.start[0], e - window2.start[0], color="orange", alpha=0.3)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Speed (m/s)")
axes[1].set_title("Speed trace with detected running epochs (orange)")

fig.tight_layout()
fig.savefig("fig2_speed_and_running_epochs.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Select Candidate Place Cells
#
# We restrict to putative excitatory (pyramidal) CA1 units, since
# interneurons rarely show strong spatial tuning, and require at least 50
# spikes during running for a reliable tuning curve estimate.

# %%
units_maze = units.restrict(maze_ep)
exc_units = units_maze[units_maze.cell_type == "excitatory"]
units_run = exc_units.restrict(run_ep)

n_spikes_run = np.array([len(units_run[uid]) for uid in exc_units.index])
MIN_SPIKES = 50
good_ids = list(exc_units.index[n_spikes_run >= MIN_SPIKES])
candidate_units = exc_units[good_ids]

print(f"{len(exc_units)} excitatory units in MazeEpoch")
print(f"{len(good_ids)} pass the >= {MIN_SPIKES}-spike criterion during running")

# %% [markdown]
# ## Compute Place Fields (1D Tuning Curves)
#
# For each candidate unit, we compute the firing rate as a function of
# position along the track (x-coordinate, since the track is oriented along
# x), using only running epochs.

# %%
N_BINS = 40
pos_x_run = position["x"].restrict(run_ep)
tuning_curves = nap.compute_tuning_curves(
    candidate_units, pos_x_run, bins=N_BINS, epochs=run_ep
)
bin_centers = tuning_curves.coords["0"].values
print(tuning_curves)

# %% [markdown]
# ## Spatial Information and Shuffle Significance Test
#
# We quantify spatial selectivity with Skaggs spatial information
# (bits/spike) and assess significance by comparing each cell's observed
# value to a null distribution built from circularly time-shifted spike
# trains (200 shuffles per cell), which preserves each neuron's firing
# statistics and the animal's behavior while destroying their true temporal
# relationship.

# %%
mi_obs = nap.compute_mutual_information(tuning_curves)

t_start, t_end = maze_ep.start[0], maze_ep.end[0]
duration = t_end - t_start
N_SHUFFLES = 200
rng = np.random.default_rng(0)

null_mi = {uid: np.zeros(N_SHUFFLES) for uid in good_ids}
for i in tqdm(range(N_SHUFFLES), desc="Shuffling"):
    offset = rng.uniform(20, duration - 20)
    shuffled = {}
    for uid in good_ids:
        st = candidate_units[uid].times()
        newt = np.sort(((st - t_start + offset) % duration) + t_start)
        shuffled[uid] = nap.Ts(t=newt, time_support=maze_ep)
    shuf_group = nap.TsGroup(shuffled).restrict(run_ep)
    tc_shuf = nap.compute_tuning_curves(shuf_group, pos_x_run, bins=N_BINS, epochs=run_ep)
    mi_shuf = nap.compute_mutual_information(tc_shuf)
    for uid in good_ids:
        null_mi[uid][i] = mi_shuf.loc[uid, "bits/spike"]

pvals = {}
for uid in good_ids:
    obs = mi_obs.loc[uid, "bits/spike"]
    null = null_mi[uid]
    pvals[uid] = (np.sum(null >= obs) + 1) / (len(null) + 1)

mi_obs["p_value"] = [pvals[uid] for uid in mi_obs.index]
mi_obs["is_place_cell"] = mi_obs["p_value"] < 0.05
place_cell_ids = list(mi_obs.index[mi_obs["is_place_cell"]])

print(f"{len(place_cell_ids)} / {len(good_ids)} candidate units pass p < 0.05")
print(mi_obs.sort_values("bits/spike", ascending=False).head(10))

# %% [markdown]
# ## Example Place Cells: Tuning Curves and Spike Locations
#
# For a handful of the most spatially informative cells, plot the firing
# rate tuning curve alongside the animal's trajectory with spike locations
# overlaid ("spikes on path"), the classic place cell figure.

# %%
top_ids = mi_obs.loc[place_cell_ids].sort_values("bits/spike", ascending=False).index[:6]

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for row, uid in enumerate(top_ids):
    tc = tuning_curves.sel(unit=uid).values
    ax_tc = axes[row // 2, (row % 2) * 2]
    ax_tc.plot(bin_centers, tc, color="crimson")
    ax_tc.set_xlabel("Position (m)")
    ax_tc.set_ylabel("Rate (Hz)")
    bits = mi_obs.loc[uid, "bits/spike"]
    p = mi_obs.loc[uid, "p_value"]
    ax_tc.set_title(f"Unit {uid}: {bits:.2f} bits/spike, p={p:.3f}")

    ax_path = axes[row // 2, (row % 2) * 2 + 1]
    ax_path.plot(position["x"].values, position["y"].values, lw=0.2, color="gray", alpha=0.4)
    spk_pos = candidate_units[uid].restrict(run_ep).value_from(position)
    ax_path.scatter(spk_pos["x"].values, spk_pos["y"].values, s=4, color="crimson")
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.set_title(f"Unit {uid}: spike locations")
    ax_path.set_aspect("equal")

fig.tight_layout()
fig.savefig("fig3_example_place_cells.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Population Place Field Sequence
#
# Normalizing each significant place cell's tuning curve to its own peak and
# sorting cells by the position of that peak reveals that, as a population,
# place fields tile the entire length of the track.

# %%
place_tc = tuning_curves.sel(unit=place_cell_ids).values  # (n_cells, n_bins)
peak_bin = np.nanargmax(place_tc, axis=1)
order = np.argsort(peak_bin)
norm_tc = place_tc / np.nanmax(place_tc, axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    norm_tc[order],
    aspect="auto",
    extent=[bin_centers[0], bin_centers[-1], len(order), 0],
    cmap="viridis",
)
ax.set_xlabel("Position on track (m)")
ax.set_ylabel("Cell # (sorted by field location)")
ax.set_title(f"Place field sequence, {len(order)} significant place cells")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Normalized firing rate")
fig.tight_layout()
fig.savefig("fig4_place_field_sequence.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 2D Place Field Maps
#
# The track is effectively 1D, but plotting firing rate over the full 2D
# maze (including the reward zones at each end) confirms spatially localized
# firing rather than an artifact of the 1D projection.

# %%
pos_2d_run = position.restrict(run_ep)
tc_2d = nap.compute_tuning_curves(
    candidate_units[list(top_ids)], pos_2d_run, bins=[25, 15], epochs=run_ep
)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, uid in zip(axes.flat, top_ids):
    rate_map = tc_2d.sel(unit=uid).values.T
    im = ax.imshow(
        rate_map,
        origin="lower",
        extent=[
            tc_2d.coords["x"].values[0], tc_2d.coords["x"].values[-1],
            tc_2d.coords["y"].values[0], tc_2d.coords["y"].values[-1],
        ],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_title(f"Unit {uid}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, label="Hz", fraction=0.046)

fig.suptitle("2D firing rate maps, top place cells", y=1.02)
fig.tight_layout()
fig.savefig("fig5_2d_place_field_maps.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Spatial Information: Observed vs. Shuffled Null
#
# Across the population of candidate cells, observed spatial information is
# systematically higher than expected by chance, and this excess is
# concentrated in the units we classify as significant place cells.

# %%
all_null_flat = np.concatenate([null_mi[uid] for uid in good_ids])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(all_null_flat, bins=50, color="gray", alpha=0.6, density=True, label="Shuffled null")
axes[0].hist(
    mi_obs["bits/spike"].values, bins=50, color="crimson", alpha=0.7, density=True,
    label="Observed",
)
axes[0].set_xlabel("Spatial information (bits/spike)")
axes[0].set_ylabel("Density")
axes[0].set_title("Observed vs. shuffled spatial information")
axes[0].legend()

colors = np.where(mi_obs["is_place_cell"].values, "crimson", "gray")
order2 = np.argsort(mi_obs["bits/spike"].values)
axes[1].bar(
    range(len(good_ids)), mi_obs["bits/spike"].values[order2],
    color=np.array(colors)[order2],
)
axes[1].set_xlabel("Unit (sorted)")
axes[1].set_ylabel("Spatial information (bits/spike)")
axes[1].set_title("Significant (red, p<0.05) vs. non-significant (gray) cells")

fig.tight_layout()
fig.savefig("fig6_spatial_information_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Summary
#
# Of the excitatory CA1 units recorded during active running on the linear
# track, a majority show firing rate tuning curves that are significantly
# more spatially informative than expected from temporally shuffled
# controls, the defining signature of hippocampal place cells. As a
# population, these place fields tile the full length of the track, and
# their spike locations plotted on the animal's trajectory show the
# characteristic spatially restricted firing that defines a place field.

# %%
print(f"Candidate excitatory units (>= {MIN_SPIKES} spikes while running): {len(good_ids)}")
print(f"Significant place cells (shuffle test, p < 0.05): {len(place_cell_ids)}")
print(
    f"Fraction: {100 * len(place_cell_ids) / len(good_ids):.0f}%"
)
print(f"Median spatial info, place cells: {mi_obs.loc[place_cell_ids, 'bits/spike'].median():.2f} bits/spike")
print(f"Median spatial info, non-place cells: {mi_obs.loc[~mi_obs['is_place_cell'], 'bits/spike'].median():.2f} bits/spike")

io.close()
