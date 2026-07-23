# %% [markdown]
# # Hippocampal Place Cells in CA1 During Linear-Track Running
#
# This notebook demonstrates the classic hippocampal **place cell** phenomenon
# (O'Keefe & Dostrovsky, 1971) using real extracellular electrophysiology data
# from the DANDI Archive: as an animal moves through an environment, individual
# CA1 pyramidal neurons fire selectively when the animal occupies a specific
# location (the cell's "place field"), and collectively the population of place
# fields tiles the whole environment.
#
# **Dataset.** [DANDI:000044](https://dandiarchive.org/dandiset/000044) --
# "Diversity in neural firing dynamics supports both rigid and learned
# hippocampal sequences" (Grosmark & Buzsaki, *Science* 2016). Bilateral
# silicon-probe recordings from dorsal CA1 in Long-Evans rats running on a
# novel 1.6 m linear track for a water reward at each end, with the animal's
# position tracked via head-mounted LEDs. We analyze four sessions (four rats)
# that all used the same 1.6 m linear-track configuration.
#
# **Approach.**
# 1. Stream one session directly from DANDI (no local download) and inspect it.
# 2. Reconstruct a continuous linearized position from the raw 2D LED tracking
#    (the dataset's own linearized-position channel turned out to be very
#    sparse -- see below).
# 3. Restrict to running epochs, compute spatial tuning curves ("place fields")
#    for every recorded unit, and quantify spatial information (Skaggs et al.,
#    1993).
# 4. Use a circular-shift shuffle test to identify units with statistically
#    significant spatial tuning ("place cells").
# 5. Repeat across four sessions/rats and summarize population statistics.

# %% [markdown]
# ## Setup

# %%
import warnings

import h5py
import matplotlib

matplotlib.use("Agg")  # headless rendering; figures are saved to disk and shown inline when run as a notebook
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

nap.nap_config.suppress_conversion_warnings = True
# Pynapple emits benign warnings for empty epochs / zero-spike units that arise
# naturally during shuffling (a shifted spike train can fall entirely outside a
# short running bout); they don't indicate a problem and are silenced here.
warnings.filterwarnings("ignore", message="Some epochs have no duration")
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")

RNG_SEED = 0

# Four sessions from DANDI:000044 that all use the same 1.6 m linear track
# (the dandiset also includes circular-track and 2 m-track sessions from other
# days, which we exclude to keep the track geometry consistent across animals).
SESSIONS = {
    "Buddy_06272013": "https://dandiarchive.s3.amazonaws.com/blobs/98b/25c/98b25cb1-310c-45f7-97cc-669fce2057b7",
    "Achilles_10252013": "https://dandiarchive.s3.amazonaws.com/blobs/4f5/a84/4f5a84aa-a6e4-496a-9b23-535fa6fbd3ae",
    "Cicero_09012014": "https://dandiarchive.s3.amazonaws.com/blobs/ced/326/ced32609-16ab-4c4c-991f-8fa9f40e9401",
    "Gatsby_08022013": "https://dandiarchive.s3.amazonaws.com/blobs/810/e9d/810e9d83-a8f3-475a-9a71-634659b1c690",
}
PROTOTYPE_SESSION = "Buddy_06272013"


# %% [markdown]
# ## Loading a Session
#
# NWB files are streamed directly from the DANDI S3 bucket with `remfile`,
# using a local disk cache so repeated reads of the same byte ranges are fast.
# Loading into `pynapple.NWBFile` gives time-aware containers (`TsGroup` for
# spike trains, `IntervalSet` for epochs, etc.) for everything except position,
# which we reconstruct manually below.


# %%
def load_nwb(s3_url, cache_dir="/tmp/remfile_cache"):
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(s3_url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, io


nwbfile, io = load_nwb(SESSIONS[PROTOTYPE_SESSION])
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# The session contains a `units` TsGroup (68 CA1 single units, sorted into
# putative excitatory/inhibitory classes), an `epochs` IntervalSet
# (PRE-sleep / MAZE / POST-sleep), raw LFP, and 2D + "linearized" position
# tracked during the maze epoch.

# %%
print(nwbfile.units.to_dataframe()[["location", "shank_id", "cell_type"]].describe(include="all"))
epochs_df = nwbfile.intervals["epochs"].to_dataframe()
print(epochs_df)

# %% [markdown]
# ## A Data Quirk: the Position `rate` Field Is Actually a Period
#
# The raw `SpatialSeries` for position stores no explicit `timestamps` array,
# only a `rate` attribute plus a `starting_time`, which pynapple/pynwb use to
# reconstruct timestamps as `starting_time + arange(n) / rate`. Here,
# `rate = 0.0256`. Taken literally as a *frequency* in Hz, that implies a
# sample every ~39 seconds, stretching the 90,942-sample position trace across
# roughly 41 days -- obviously wrong for a single ~40 minute recording epoch.
#
# Treating the same number as a sampling *period* (dt) instead resolves this
# cleanly: `1 / 0.0256 ~= 39.06 Hz`, and `90942 * 0.0256 ~= 2328 s`, which
# matches the `MazeEpoch` duration (10717.9-13046.0 s = 2328.1 s) almost
# exactly, and the reconstructed trace's start time lines up with the maze
# epoch's start time. We use this corrected interpretation throughout.

# %%
beh = nwbfile.processing["behavior"].data_interfaces
pos_key = [k for k in beh.keys() if k != "states" and "Linearized" not in k][0]
spatial_series = beh[pos_key].spatial_series[list(beh[pos_key].spatial_series.keys())[0]]
dt = spatial_series.rate  # mislabeled: this is actually the sampling period, not a rate
n_samples = spatial_series.data.shape[0]
print(f"position stream: {pos_key}")
print(f"raw 'rate' field: {dt} -> interpreted as dt, implies true rate {1/dt:.2f} Hz")
print(f"n_samples * dt = {n_samples * dt:.1f} s  (MazeEpoch duration = "
      f"{epochs_df.loc[epochs_df.label == 'MazeEpoch', 'stop_time'].item() - epochs_df.loc[epochs_df.label == 'MazeEpoch', 'start_time'].item():.1f} s)")


# %% [markdown]
# ## Reconstructing a Continuous Linearized Position
#
# The dataset ships a pre-computed `*LinearizedTimeSeries`, but it turns out to
# be very sparse (valid on only ~7% of samples -- likely restricted to a
# subset of clean traversals by the original authors). Rather than throw away
# 93% of the running behavior, we recompute a linear position ourselves: the
# raw 2D LED position is projected onto the main axis of variance (PCA) of the
# trajectory, which is a standard and robust way to linearize a straight
# track. We validate this against the provided (sparse) linearized channel
# before relying on it.


# %%
def build_linear_position(nwbfile):
    beh = nwbfile.processing["behavior"].data_interfaces
    pos_key = [k for k in beh.keys() if k != "states" and "Linearized" not in k][0]
    spatial_series = beh[pos_key].spatial_series[list(beh[pos_key].spatial_series.keys())[0]]

    data = spatial_series.data[:]
    n = data.shape[0]
    dt = spatial_series.rate  # mislabeled sampling period, see above
    t = spatial_series.starting_time + np.arange(n) * dt

    valid = ~np.isnan(data).any(axis=1)
    xy = data[valid]
    tv = t[valid]

    centered = xy - xy.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    main_axis = evecs[:, np.argmax(evals)]
    explained_var_ratio = evals.max() / evals.sum()
    # elementwise multiply + sum instead of `centered @ main_axis`: numerically
    # identical, but avoids a spurious overflow RuntimeWarning from the
    # Accelerate BLAS backend on this array size (verified the matmul result
    # itself contains no NaN/Inf; this sidesteps the warning entirely).
    proj = np.sum(centered * main_axis[None, :], axis=1)

    lo, hi = np.percentile(proj, [0.5, 99.5])
    track_length_m = float(pos_key.split("m")[0])
    linpos_m = np.clip((proj - lo) / (hi - lo), 0, 1) * track_length_m

    linpos = nap.Tsd(t=tv, d=linpos_m)
    xy_tsd = nap.TsdFrame(t=tv, d=xy, columns=["x", "y"])
    return linpos, xy_tsd, pos_key, explained_var_ratio, track_length_m


linpos, xy_tsd, pos_key, evr, track_length_m = build_linear_position(nwbfile)
print(f"PCA main-axis explained variance ratio: {evr:.3f} (near 1 => trajectory is effectively 1D)")

# validate against the sparse provided linearized channel
lin_module_key = [k for k in beh.keys() if "Linearized" in k][0]
lin_series = beh[lin_module_key].spatial_series[list(beh[lin_module_key].spatial_series.keys())[0]]
lin_data_raw = lin_series.data[:].flatten()
t_raw = spatial_series.starting_time + np.arange(spatial_series.data.shape[0]) * spatial_series.rate
valid_raw = ~np.isnan(spatial_series.data[:]).any(axis=1)
lin_valid_mask = valid_raw & ~np.isnan(lin_data_raw)
provided = lin_data_raw[lin_valid_mask]
ours = linpos.d[np.cumsum(valid_raw)[lin_valid_mask] - 1]
corr = np.corrcoef(provided, ours)[0, 1]
print(f"correlation between our PCA linearization and the dataset's sparse linearized channel: r = {corr:.3f}")
print("(a near +-1 correlation confirms the PCA projection recovers the same track axis)")

# %% [markdown]
# ## Running-Epoch Detection
#
# Place fields should be estimated from active locomotion, excluding the long
# immobile bouts at the reward ports (where ripple/replay activity, not
# spatial coding, dominates firing). We smooth the linear position, take its
# derivative to get speed, and threshold at 5 cm/s.


# %%
def compute_run_epochs(linpos, speed_thresh=0.05, smooth_std=0.3):
    smoothed = linpos.smooth(std=smooth_std, size_factor=20)
    speed = np.abs(smoothed.derivative())
    speed = speed.smooth(std=smooth_std, size_factor=20)
    above = speed.threshold(speed_thresh, method="above")
    run_ep = above.time_support.drop_short_intervals(0.5).merge_close_intervals(0.2)
    return speed, run_ep


maze_ep = nwb["epochs"]
maze_ep = maze_ep[maze_ep.label == "MazeEpoch"]
linpos = linpos.restrict(maze_ep)
xy_tsd = xy_tsd.restrict(maze_ep)
speed, run_ep = compute_run_epochs(linpos)
print(f"running epochs: {len(run_ep)} bouts, {run_ep.tot_length():.0f} s total "
      f"out of {maze_ep.tot_length():.0f} s maze epoch "
      f"({100 * run_ep.tot_length() / maze_ep.tot_length():.0f}%)")

units_all = nwb["units"]
units_maze = units_all.restrict(maze_ep)
units_run = units_maze.restrict(run_ep)
linpos_run = linpos.restrict(run_ep)

# %% [markdown]
# ## Figure 1: Raw Data Overview
#
# Before any place-field analysis, we inspect the raw signals directly: the
# animal's linear position, the population spike raster, running speed, and a
# raw LFP trace. The LFP shows clear theta-band (~8 Hz) oscillations during
# running, the expected hippocampal signature of active locomotion.

# %%
win_start, win_end = 10820.0, 10950.0
win = nap.IntervalSet(start=win_start, end=win_end)

linpos_w = linpos.restrict(win)
speed_w = speed.restrict(win)
units_w = units_maze.restrict(win)
order = np.argsort(units_maze.location.values + units_maze.shank_id.values.astype(str))
unit_ids_sorted = units_maze.index[order]

ecephys = nwbfile.processing["ecephys"]
lfp_series = ecephys.data_interfaces["LFP"].electrical_series["LFP"]
lfp_rate = lfp_series.rate
lfp_conversion = lfp_series.conversion
lfp_start, lfp_end = 10871.0, 10880.0
i0, i1 = int(lfp_start * lfp_rate), int(lfp_end * lfp_rate)
lfp_chunk = lfp_series.data[i0:i1, 0].astype(float) * lfp_conversion * 1e3  # mV
lfp_t = np.arange(i0, i1) / lfp_rate

fig = plt.figure(figsize=(13, 11))
gs = fig.add_gridspec(4, 1, height_ratios=[1.3, 1.8, 0.8, 1.0], hspace=0.45)

ax0 = fig.add_subplot(gs[0])
ax0.plot(linpos_w.t - win_start, linpos_w.d, ".", ms=2, color="black")
for s, e in zip(run_ep.start, run_ep.end):
    if e > win_start and s < win_end:
        ax0.axvspan(max(s, win_start) - win_start, min(e, win_end) - win_start, color="orange", alpha=0.2)
ax0.set_ylabel("Linear position (m)")
ax0.set_title(f"{PROTOTYPE_SESSION} - MazeEpoch excerpt ({win_start:.0f}-{win_end:.0f} s). Orange = running epochs")
ax0.set_xlim(0, win_end - win_start)

ax1 = fig.add_subplot(gs[1], sharex=ax0)
for i, uid in enumerate(unit_ids_sorted):
    st = units_w[uid].t - win_start
    ax1.plot(st, np.full_like(st, i), "|", color="black", ms=6, mew=1)
ax1.set_ylabel("Unit # (sorted by shank)")
ax1.set_xlim(0, win_end - win_start)
ax1.set_ylim(-1, len(unit_ids_sorted))
ax1.set_title(f"Spike raster, all {len(unit_ids_sorted)} units")

ax2 = fig.add_subplot(gs[2], sharex=ax0)
ax2.plot(speed_w.t - win_start, speed_w.d, color="tab:blue", lw=0.8)
ax2.axhline(0.05, color="red", ls="--", lw=1, label="run threshold")
ax2.set_ylabel("Speed (m/s)")
ax2.set_xlabel("Time (s)")
ax2.legend(loc="upper right", fontsize=8)
ax2.set_xlim(0, win_end - win_start)

ax3 = fig.add_subplot(gs[3])
ax3.plot(lfp_t - lfp_start, lfp_chunk, color="tab:green", lw=0.8)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("LFP (mV)")
ax3.set_title(f"Raw LFP, shank1 channel, {lfp_start:.0f}-{lfp_end:.0f} s (theta oscillations during running)")

plt.savefig("fig1_raw_data_overview.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Tuning Curves and Spatial Information
#
# For each unit we compute a 1D tuning curve (firing rate as a function of
# linear position, occupancy-normalized) using `pynapple.compute_tuning_curves`,
# restricted to running epochs. Spatial information (Skaggs et al., 1993, in
# bits/spike) quantifies how much a cell's spike train tells you about
# position.

# %%
N_BINS = 40
tc = nap.compute_tuning_curves(units_run, linpos_run, bins=N_BINS, epochs=run_ep, feature_names=["position"])
mi = nap.compute_mutual_information(tc)
print(mi.describe())


# %% [markdown]
# ## Identifying Significant Place Cells: a Circular-Shift Shuffle Test
#
# A raw spatial-information value is not by itself evidence of a place field:
# even a randomly-firing cell has nonzero spatial information simply from
# sampling noise, particularly at low firing rates. The standard control
# (Kelemen & Fenton and similar) is to repeatedly and randomly circularly
# shift each cell's spike train in time (decoupling it from the position
# trace while preserving its overall firing statistics and autocorrelation),
# recompute spatial information under the shuffle, and compare the observed
# value to the resulting null distribution. A unit is classified as a place
# cell if its true spatial information exceeds the 95th percentile of its own
# shuffle distribution and its mean firing rate during running exceeds 0.2 Hz
# (to exclude cells too silent for a reliable estimate).


# %%
def shuffle_spatial_info(units, linpos, run_ep, maze_ep, n_bins, n_shuffles, rng):
    t0, t1 = maze_ep.start[0], maze_ep.end[0]
    duration = t1 - t0
    shuffled_info = np.zeros((n_shuffles, len(units)))

    for i in range(n_shuffles):
        shift = rng.uniform(20.0, duration - 20.0)
        shifted = {}
        for uid in units.index:
            st = units[uid].t
            new_t = t0 + np.mod((st - t0) + shift, duration)
            shifted[uid] = nap.Ts(t=np.sort(new_t))
        shifted_group = nap.TsGroup(shifted, time_support=maze_ep).restrict(run_ep)
        tc_shuf = nap.compute_tuning_curves(shifted_group, linpos, bins=n_bins, epochs=run_ep, feature_names=["position"])
        shuffled_info[i] = nap.compute_mutual_information(tc_shuf)["bits/spike"].values

    return shuffled_info


def analyze_session(s3_url, session_name, n_bins=40, n_shuffles=200, seed=0):
    nwbfile, io = load_nwb(s3_url)
    nwb = nap.NWBFile(nwbfile)

    units_all = nwb["units"]
    maze_ep = nwb["epochs"]
    maze_ep = maze_ep[maze_ep.label == "MazeEpoch"]

    linpos, xy_tsd, pos_key, evr, track_length_m = build_linear_position(nwbfile)
    linpos = linpos.restrict(maze_ep)
    xy_tsd = xy_tsd.restrict(maze_ep)
    speed, run_ep = compute_run_epochs(linpos)

    units_maze = units_all.restrict(maze_ep)
    units_run = units_maze.restrict(run_ep)
    linpos_run = linpos.restrict(run_ep)

    tc = nap.compute_tuning_curves(units_run, linpos_run, bins=n_bins, epochs=run_ep, feature_names=["position"])
    mi = nap.compute_mutual_information(tc)

    rng = np.random.default_rng(seed)
    shuf = shuffle_spatial_info(units_maze, linpos_run, run_ep, maze_ep, n_bins, n_shuffles, rng)
    thresh_95 = np.percentile(shuf, 95, axis=0)
    is_place_cell = (mi["bits/spike"].values > thresh_95) & (tc.attrs["rates"] > 0.2)

    return dict(
        session_name=session_name, nwbfile=nwbfile, io=io, units_all=units_all,
        units_maze=units_maze, units_run=units_run, maze_ep=maze_ep, run_ep=run_ep,
        linpos=linpos, linpos_run=linpos_run, xy_tsd=xy_tsd, speed=speed, tc=tc,
        mi=mi, shuf=shuf, thresh_95=thresh_95, is_place_cell=is_place_cell,
        track_length_m=track_length_m, pos_key=pos_key, explained_var_ratio=evr,
    )


res = analyze_session(SESSIONS[PROTOTYPE_SESSION], PROTOTYPE_SESSION, n_bins=N_BINS, n_shuffles=200, seed=RNG_SEED)
print(f"{PROTOTYPE_SESSION}: {res['is_place_cell'].sum()} / {len(res['is_place_cell'])} units are significant place cells")

# %% [markdown]
# ## Figure 2: The Place Field Map
#
# Normalizing each significant place cell's tuning curve to its own peak and
# sorting cells by the location of that peak reveals the hallmark signature of
# a place-cell population: a diagonal band of activity spanning the full
# extent of the track, i.e., the population of place fields collectively
# tiles the whole environment.

# %%
tc, is_pc, mi_res = res["tc"], res["is_place_cell"], res["mi"]
pc_ids = tc.coords["unit"].values[is_pc]
positions = tc.coords[tc.dims[1]].values
peak_bin = tc.sel(unit=pc_ids).argmax(dim=tc.dims[1]).values
pc_ids_sorted = pc_ids[np.argsort(peak_bin)]

norm_tc = np.stack([
    tc.sel(unit=uid).values / (np.nanmax(tc.sel(unit=uid).values) + 1e-12)
    for uid in pc_ids_sorted
])

fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.2, 1]})
im = axes[0].imshow(norm_tc, aspect="auto", cmap="viridis", origin="lower",
                     extent=[positions.min(), positions.max(), 0, len(pc_ids_sorted)])
axes[0].set_xlabel("Linear position (m)")
axes[0].set_ylabel("Place cell # (sorted by field location)")
axes[0].set_title(f"Place field map, sorted by peak location\n({len(pc_ids_sorted)} place cells, {PROTOTYPE_SESSION})")
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04).set_label("Normalized firing rate")

occ = tc.attrs["occupancy"]
occ = occ / occ.sum()
axes[1].bar(positions, occ, width=(positions[1] - positions[0]) * 0.9, color="grey")
axes[1].set_xlabel("Linear position (m)")
axes[1].set_ylabel("Fraction of running time")
axes[1].set_title("Occupancy during running epochs")

plt.tight_layout()
plt.savefig("fig2_place_field_heatmap.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Figure 3: Example Place Cells
#
# For the six most spatially informative cells, the top row shows every spike
# plotted at the animal's (x, y) position when it occurred (red) against the
# full trajectory (grey); the bottom row shows the corresponding 1D tuning
# curve. Spikes cluster tightly at one location on the track, matching the
# tuning curve peak.

# %%
si_vals = mi_res.loc[pc_ids, "bits/spike"].values
top_ids = pc_ids[np.argsort(-si_vals)][:6]
top_si = np.sort(si_vals)[::-1][:6]

fig, axes = plt.subplots(2, 6, figsize=(20, 7))
for i, uid in enumerate(top_ids):
    ax_top = axes[0, i]
    ax_top.plot(res["xy_tsd"].d[:, 0], res["xy_tsd"].d[:, 1], color="lightgrey", lw=0.5, zorder=1)
    spk_xy = res["units_run"][uid].value_from(res["xy_tsd"])
    ax_top.scatter(spk_xy.d[:, 0], spk_xy.d[:, 1], s=4, color="crimson", zorder=2)
    ax_top.set_title(f"unit {uid} ({res['units_run'].cell_type[uid]})\nSI={top_si[i]:.2f} bits/spk", fontsize=9)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_top.set_aspect("equal")

    ax_bot = axes[1, i]
    rate = tc.sel(unit=uid).values
    ax_bot.plot(positions, rate, color="tab:blue")
    ax_bot.fill_between(positions, 0, rate, alpha=0.3, color="tab:blue")
    ax_bot.set_xlabel("Position (m)")
    if i == 0:
        ax_bot.set_ylabel("Firing rate (Hz)")

fig.suptitle(f"Top 6 place cells by spatial information ({PROTOTYPE_SESSION}):\nspike locations (top) and place fields (bottom)", y=1.05)
plt.tight_layout()
plt.savefig("fig3_example_place_cells.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Figure 4: Spatial Information Statistics and the Shuffle Test
#
# Panel (a) shows the spatial-information distribution split by place-cell
# classification. Panel (b) illustrates the shuffle test for the single most
# informative place cell: the observed value (red) falls far outside its own
# null distribution (grey). Panel (c) shows that a larger fraction of
# excitatory (putative pyramidal) cells are classified as place cells than
# inhibitory (putative interneuron) cells, consistent with place coding being
# a principal-cell phenomenon in CA1. Panel (d) shows spatial information as a
# function of mean firing rate.

# %%
si = mi_res["bits/spike"].values
cell_type = res["units_all"].cell_type.values
rates = tc.attrs["rates"]
shuf, thresh_95 = res["shuf"], res["thresh_95"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
bins = np.linspace(0, np.nanpercentile(si, 99), 30)
ax.hist(si[~is_pc], bins=bins, alpha=0.6, label=f"non-place (n={(~is_pc).sum()})", color="grey")
ax.hist(si[is_pc], bins=bins, alpha=0.7, label=f"place cells (n={is_pc.sum()})", color="tab:orange")
ax.set_xlabel("Spatial information (bits/spike)"); ax.set_ylabel("# units")
ax.set_title("Spatial information distribution"); ax.legend(fontsize=9)

si_for_pc = np.where(is_pc, si, -np.inf)
best_idx = np.nanargmax(si_for_pc)
best_uid = tc.coords["unit"].values[best_idx]
ax = axes[0, 1]
ax.hist(shuf[:, best_idx], bins=30, color="lightgrey", label="shuffle distribution")
ax.axvline(thresh_95[best_idx], color="black", ls="--", label="95th pct (shuffle)")
ax.axvline(si[best_idx], color="crimson", lw=2, label="observed")
ax.set_xlabel("Spatial information (bits/spike)"); ax.set_ylabel("# shuffles")
ax.set_title(f"Shuffle test example, unit {best_uid}"); ax.legend(fontsize=8)

ax = axes[1, 0]
cts = np.unique(cell_type)
fracs = [(is_pc & (cell_type == ct)).sum() / (cell_type == ct).sum() for ct in cts]
counts = [(cell_type == ct).sum() for ct in cts]
bars = ax.bar(cts, fracs, color=["tab:blue", "tab:red"])
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"n={c}", ha="center", fontsize=9)
ax.set_ylabel("Fraction classified as place cells")
ax.set_title("Place cell fraction by cell type"); ax.set_ylim(0, 1.05)

ax = axes[1, 1]
ax.scatter(rates[~is_pc], si[~is_pc], s=15, color="grey", alpha=0.6, label="non-place")
ax.scatter(rates[is_pc], si[is_pc], s=15, color="tab:orange", alpha=0.8, label="place cells")
ax.set_xscale("log")
ax.set_xlabel("Mean firing rate during running (Hz)")
ax.set_ylabel("Spatial information (bits/spike)")
ax.set_title("Spatial information vs. firing rate")
ax.set_ylim(-0.1, np.nanpercentile(si, 98))
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("fig4_spatial_info_stats.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Scaling to Multiple Sessions
#
# The full pipeline above (linearize position, detect running, compute tuning
# curves, shuffle-test for significance) is repeated for three more sessions
# recorded from three other rats on the same 1.6 m linear track, to check that
# the phenomenon and its statistics are consistent across animals rather than
# an artifact of one session.

# %%
summary = {}
for name, url in SESSIONS.items():
    r = analyze_session(url, name, n_bins=N_BINS, n_shuffles=200, seed=RNG_SEED + hash(name) % 1000)
    cell_type_s = r["units_all"].cell_type.values
    n_exc = int((cell_type_s == "excitatory").sum())
    n_pc_exc = int((r["is_place_cell"] & (cell_type_s == "excitatory")).sum())
    print(f"{name}: n_units={len(r['units_all'])}, running={r['run_ep'].tot_length():.0f}s, "
          f"place_cells={r['is_place_cell'].sum()}/{len(r['is_place_cell'])}, "
          f"excitatory place_cells={n_pc_exc}/{n_exc} ({n_pc_exc/max(n_exc,1):.0%})")
    summary[name] = dict(
        n_exc=n_exc, n_pc_exc=n_pc_exc, si=r["mi"]["bits/spike"].values,
        cell_type=cell_type_s, is_pc=r["is_place_cell"], rates=r["tc"].attrs["rates"],
    )
    r["io"].close()

# %% [markdown]
# ## Figure 5: Population Summary Across Sessions
#
# (a) The fraction of excitatory units classified as significant place cells
# is consistently 50-70% across all four rats. (b) Pooling spatial information
# across sessions shows a clear excess of high-spatial-information cells among
# those classified as place cells. (c) Among excitatory cells, units that pass
# the place-cell criterion also tend to fire at higher rates during running
# than those that don't -- consistent with very sparsely-firing pyramidal
# cells not producing enough spikes during a single session to reach
# significance, rather than with place cells per se being high-rate.

# %%
names = list(summary.keys())
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
fracs = [summary[n]["n_pc_exc"] / summary[n]["n_exc"] for n in names]
counts = [summary[n]["n_exc"] for n in names]
bars = ax.bar(range(len(names)), fracs, color="tab:blue")
for i, (b, c) in enumerate(zip(bars, counts)):
    ax.text(i, b.get_height() + 0.02, f"n={c}", ha="center", fontsize=9)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.split("_")[0] for n in names])
ax.set_ylabel("Fraction of excitatory units classified as place cells")
ax.set_ylim(0, 1.0)
ax.set_title("Place cell fraction per session\n(1.6 m linear track, excitatory units)")

ax = axes[1]
exc_masks = {n: summary[n]["cell_type"] == "excitatory" for n in names}
all_si_pc = np.concatenate([summary[n]["si"][exc_masks[n] & summary[n]["is_pc"]] for n in names])
all_si_nonpc = np.concatenate([summary[n]["si"][exc_masks[n] & ~summary[n]["is_pc"]] for n in names])
bins = np.linspace(0, 3, 30)
ax.hist(all_si_nonpc, bins=bins, alpha=0.6, color="grey", label=f"non-place (n={len(all_si_nonpc)})")
ax.hist(all_si_pc, bins=bins, alpha=0.7, color="tab:orange", label=f"place cells (n={len(all_si_pc)})")
ax.set_xlabel("Spatial information (bits/spike)")
ax.set_ylabel("# excitatory units (pooled, 4 sessions)")
ax.set_title("Pooled spatial information\nacross 4 rats")
ax.legend(fontsize=9)

ax = axes[2]
all_rate_pc = np.concatenate([summary[n]["rates"][exc_masks[n] & summary[n]["is_pc"]] for n in names])
all_rate_nonpc = np.concatenate([summary[n]["rates"][exc_masks[n] & ~summary[n]["is_pc"]] for n in names])
bp = ax.boxplot([all_rate_nonpc, all_rate_pc], tick_labels=["non-place", "place cells"],
                showfliers=False, patch_artist=True)
for patch, color in zip(bp["boxes"], ["grey", "tab:orange"]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_yscale("log")
ax.set_ylabel("Mean firing rate during running (Hz)")
ax.set_title("Firing rate, place vs. non-place\n(pooled, excitatory units)")

plt.tight_layout()
plt.savefig("fig5_population_summary.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# Across four rats running on a novel 1.6 m linear track (DANDI:000044,
# Grosmark & Buzsaki 2016), 50-70% of putative excitatory CA1 units showed
# statistically significant spatial tuning by a circular-shift shuffle test on
# Skaggs spatial information, and their normalized tuning curves collectively
# tile the entire track (Figure 2). Individual example cells fire in
# spatially restricted zones matching their tuning curve peaks (Figure 3), and
# place coding was substantially enriched among excitatory relative to
# inhibitory units (Figure 4c), matching the classical characterization of
# hippocampal place cells as a principal-cell phenomenon. These results were
# consistent across four independent recording sessions from four different
# animals (Figure 5), demonstrating that place fields in dorsal CA1 are a
# robust, readily reproducible feature of hippocampal activity during
# spatial navigation.
