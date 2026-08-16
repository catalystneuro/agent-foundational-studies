# %% [markdown]
# # Hippocampal Place Cells on a Linear Track (DANDI 000044)
#
# This notebook demonstrates hippocampal place cells using a classic recording
# from the Buzsáki lab, streamed from the DANDI Archive.
#
# **Dataset:** DANDI dandiset
# [000044](https://dandiarchive.org/dandiset/000044) ("Diversity in neural
# firing dynamics supports both rigid and learned hippocampal sequences"),
# session `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`: 137
# single units from CA1 tetrodes (left and right) recorded while a rat ran
# back and forth on a 1.6 m linear track (~34 min maze epoch, ~42 laps).
#
# **Analysis:**
# 1. Stream the NWB file with remfile + disk cache (no full download).
# 2. Restrict to the maze epoch and use the dataset's linearized position
#    (defined while the animal runs on the track).
# 3. Compute occupancy-normalized firing rate maps for all units.
# 4. Quantify spatial tuning with Skaggs spatial information and assess
#    significance with circular time-shift shuffles.
# 5. Show that place fields tile the track and that many cells are
#    direction-selective.
# 6. Confirm the rate maps with Poisson GLMs (B-spline position basis, nemos).
#
# Everything runs end-to-end; figures are saved as PNG files.

# %% [markdown]
# ## Setup and streaming data access

# %%
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: save figures, never plt.show()
import matplotlib.pyplot as plt
import h5py
import remfile
import requests
import pynapple as nap
import jax
jax.config.update("jax_enable_x64", True)
import nemos as nmo
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

DANDISET = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
CACHE_DIR = "/tmp/remfile_cache_achilles"

def resolve_s3_url():
    """Resolve the direct S3 URL for the asset via the DANDI API."""
    api = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/"
    r = requests.get(api, params={"path": ASSET_PATH})
    r.raise_for_status()
    asset_id = r.json()["results"][0]["asset_id"]
    head = requests.head(f"{api}{asset_id}/download/", allow_redirects=False)
    head.raise_for_status()
    return head.headers["Location"]

s3_url = resolve_s3_url()
print("Streaming:", s3_url.split("?")[0])

disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# ## Inspect the session
#
# The NWB file contains a `TsGroup` of 137 units (with `cell_type`,
# `location`, `shank_id` metadata), behavioral epochs (PRE / Maze / POST),
# the 2D position (`1.6mLinearMazeSpatialSeries`, ~39 Hz), and a linearized
# position signal (`1.6mLinearMazeLinearizedTimeSeries`, 0-1.6 m) that is
# defined only while the animal is running on the track.

# %%
units = nwb["units"]
print("n units:", len(units))
print("metadata:", units.metadata_columns)
print("cell types:", np.unique(units.get_info("cell_type").values))
print("locations:", np.unique(units.get_info("location").values))
print("\nepochs:")
print(nwb["epochs"])

maze = nwb["epochs"][1]  # MazeEpoch
print("\nMaze epoch: %.1f to %.1f s (%.1f min)" % (
    maze.start[0], maze.end[0], (maze.end[0] - maze.start[0]) / 60))

# %%
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"].restrict(maze)
xy = nwb["1.6mLinearMazeSpatialSeries"].restrict(maze)

lin_t = lin.t
lin_v = lin.values[:, 0]
valid = ~np.isnan(lin_v)
xy_t = xy.t
xy_v = xy.values
dt = np.median(np.diff(lin_t))

print("position sampling rate: %.1f Hz" % (1 / dt))
print("linearized position defined %.1f%% of the epoch (%.0f s of running)" % (
    100 * valid.mean(), valid.sum() * dt))
print("track coordinate range: %.2f to %.2f m" % (np.nanmin(lin_v), np.nanmax(lin_v)))

# %% [markdown]
# ## Session overview
#
# The rat shuttles between reward sites at the two ends of a 1.6 m linear
# track. The linearized coordinate is only defined during runs (median speed
# while defined: ~50 cm/s); the animal spends the rest of the epoch at the
# reward platforms.

# %%
dx = np.gradient(xy_v[:, 0], dt)
dy = np.gradient(xy_v[:, 1], dt)
speed = gaussian_filter1d(np.nan_to_num(np.sqrt(dx**2 + dy**2)), sigma=int(0.5 / dt))
t0 = xy_t[0]

TRACK_LEN = 1.6
N_BINS = 50
BIN_EDGES = np.linspace(0, TRACK_LEN, N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
SMOOTH_SIGMA_BINS = 1.5
MIN_OCC_S = 0.1

occ_counts, _ = np.histogram(lin_v[valid], bins=BIN_EDGES)
occupancy = occ_counts * dt
occ_prob = occupancy / occupancy.sum()

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
ax = fig.add_subplot(gs[0, 0])
ax.plot(xy_v[:, 0], xy_v[:, 1], lw=0.05, alpha=0.4, color="steelblue")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("A  Trajectory during maze epoch", loc="left", fontsize=11)
ax.set_aspect("equal")

ax = fig.add_subplot(gs[0, 1])
ax.plot(lin_t[::10] - t0, lin_v[::10], lw=0.3, color="darkorange")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Linearized position (m)")
ax.set_title("B  Linearized position (runs only)", loc="left", fontsize=11)

ax = fig.add_subplot(gs[1, 0])
ax.plot(xy_t[::10] - t0, speed[::10] * 100, lw=0.3, color="seagreen")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (cm/s)")
ax.set_title("C  Running speed", loc="left", fontsize=11)
ax.set_xlim(0, xy_t[-1] - t0)

ax = fig.add_subplot(gs[1, 1])
ax.bar(BIN_CENTERS, occupancy, width=np.diff(BIN_CENTERS)[0], color="slategray")
ax.set_xlabel("Linearized position (m)"); ax.set_ylabel("Occupancy (s)")
ax.set_title("D  Occupancy on track", loc="left", fontsize=11)
plt.savefig("fig1_session_overview.png", dpi=150)
plt.close()
print("saved fig1_session_overview.png")

# %% [markdown]
# ## Firing rate maps, spatial information, and shuffle significance
#
# For each unit we bin spikes by the animal's linearized position (3.2 cm
# bins), divide by occupancy, and smooth with a Gaussian (sigma = 1.5 bins).
# Spatial tuning is quantified with Skaggs spatial information (bits/spike)
# and sparsity. Significance is assessed per unit with 500 circular
# time-shift shuffles of the spike train relative to the position signal:
# the shuffle destroys the spike-position relationship while preserving the
# temporal structure of both signals.
#
# A unit is called a **place cell** if it is excitatory, fires at >0.1 Hz on
# the track with a peak rate >1 Hz, has at least 30 on-track spikes, and its
# observed spatial information exceeds the shuffle distribution (p < 0.05).

# %%
# running direction from the linearized position (for direction-split maps)
idx_valid = np.where(valid)[0]
lin_interp = np.interp(np.arange(len(lin_v)), idx_valid, lin_v[idx_valid])
vel = gaussian_filter1d(np.gradient(lin_interp, dt), sigma=int(0.25 / dt))
direction = np.sign(vel)  # +1 running toward 1.6 m, -1 toward 0 m

occ_pos = np.histogram(lin_v[valid & (direction > 0)], bins=BIN_EDGES)[0] * dt
occ_neg = np.histogram(lin_v[valid & (direction < 0)], bins=BIN_EDGES)[0] * dt


def spike_bin_counts(spike_times):
    """Spike counts per position bin, using the nearest position sample."""
    if len(spike_times) == 0:
        z = np.zeros(N_BINS)
        return z, z.copy(), z.copy()
    pos_idx = np.searchsorted(lin_t, spike_times).clip(0, len(lin_t) - 1)
    pos_idx = pos_idx[valid[pos_idx]]
    b = np.digitize(lin_v[pos_idx], BIN_EDGES) - 1
    keep = (b >= 0) & (b < N_BINS)
    b = b[keep]
    d = direction[pos_idx][keep]
    counts = np.bincount(b, minlength=N_BINS).astype(float)
    counts_pos = np.bincount(b[d > 0], minlength=N_BINS).astype(float)
    counts_neg = np.bincount(b[d < 0], minlength=N_BINS).astype(float)
    return counts, counts_pos, counts_neg


def rate_map(counts, occ):
    with np.errstate(invalid="ignore", divide="ignore"):
        rm = counts / occ
    rm[occ < MIN_OCC_S] = np.nan
    return gaussian_filter1d(np.nan_to_num(rm), SMOOTH_SIGMA_BINS)


def skaggs_info(rm, occ_p):
    """Skaggs spatial information, bits per spike."""
    lam = np.nansum(rm * occ_p)
    if lam <= 0:
        return 0.0
    r = rm / lam
    with np.errstate(invalid="ignore", divide="ignore"):
        si = np.nansum(occ_p * r * np.log2(r))
    return float(si)


def sparsity(rm, occ_p):
    num = np.nansum(occ_p * rm) ** 2
    den = np.nansum(occ_p * rm**2)
    return float(num / den) if den > 0 else np.nan


N_SHUFFLE = 500
rng = np.random.default_rng(42)
epoch_dur = maze.end[0] - maze.start[0]
valid_dur = valid.sum() * dt

unit_keys = list(units.keys())
cell_type = units.get_info("cell_type").values
location = units.get_info("location").values

results = {}
for i, k in enumerate(tqdm(unit_keys, desc="rate maps + shuffles")):
    st = units[k].restrict(maze).t
    counts, counts_pos, counts_neg = spike_bin_counts(st)
    rm = rate_map(counts, occupancy)
    si = skaggs_info(np.nan_to_num(rm), occ_prob)
    n_track = int(counts.sum())

    if n_track >= 30:
        si_null = np.zeros(N_SHUFFLE)
        for s in range(N_SHUFFLE):
            shift = rng.uniform(20, epoch_dur - 20)
            st_sh = ((st - maze.start[0] + shift) % epoch_dur) + maze.start[0]
            c_sh, _, _ = spike_bin_counts(st_sh)
            si_null[s] = skaggs_info(np.nan_to_num(rate_map(c_sh, occupancy)), occ_prob)
        p_val = (np.sum(si_null >= si) + 1) / (N_SHUFFLE + 1)
        si_null_mean = si_null.mean()
    else:
        p_val, si_null_mean = np.nan, np.nan

    results[k] = dict(
        cell_type=cell_type[i], location=location[i],
        n_spikes_maze=len(st), n_spikes_track=n_track,
        mean_rate_track=n_track / valid_dur,
        rate_map=rm,
        rate_map_pos=rate_map(counts_pos, occ_pos),
        rate_map_neg=rate_map(counts_neg, occ_neg),
        spatial_info=si, sparsity=sparsity(np.nan_to_num(rm), occ_prob),
        shuffle_p=p_val, si_null_mean=si_null_mean,
    )

np.savez_compressed("placefield_results.npz",
                    unit_keys=np.array(unit_keys), occupancy=occupancy,
                    occ_pos=occ_pos, occ_neg=occ_neg, bin_centers=BIN_CENTERS,
                    results=np.array([results], dtype=object))

# %%
si_all = np.array([results[k]["spatial_info"] for k in unit_keys])
p_all = np.array([results[k]["shuffle_p"] for k in unit_keys])
rate_all = np.array([results[k]["mean_rate_track"] for k in unit_keys])
sp_all = np.array([results[k]["sparsity"] for k in unit_keys])
peak_all = np.array([np.nanmax(results[k]["rate_map"]) for k in unit_keys])
exc = np.array([results[k]["cell_type"] == "excitatory" for k in unit_keys])

is_place = (exc & (p_all < 0.05) & (rate_all > 0.1) & (peak_all > 1.0)
            & ~np.isnan(p_all))
place_keys = [k for k, m in zip(unit_keys, is_place) if m]
peak_place = {k: BIN_CENTERS[np.nanargmax(results[k]["rate_map"])] for k in place_keys}
si_place = {k: results[k]["spatial_info"] for k in place_keys}

print("excitatory units: %d, inhibitory: %d" % (exc.sum(), (~exc).sum()))
print("place cells: %d of %d excitatory (%.0f%%)" % (
    len(place_keys), exc.sum(), 100 * len(place_keys) / exc.sum()))
print("spatial information (excitatory): median %.2f, max %.2f bits/spike" % (
    np.median(si_all[exc]), si_all[exc].max()))

# %% [markdown]
# ## Example place cells
#
# Six place cells with fields at different positions along the track. Left:
# spike locations (red) over the trajectory. Middle: occupancy-normalized
# rate map. Right: rate maps computed separately for runs toward 1.6 m
# (blue) and toward 0 m (orange), showing that many cells fire only in one
# running direction.

# %%
def spike_positions(k):
    """xy and linearized position of each spike of unit k (nearest sample)."""
    st = units[k].restrict(maze).t
    ix = np.searchsorted(xy_t, st).clip(0, len(xy_t) - 1)
    il = np.searchsorted(lin_t, st).clip(0, len(lin_t) - 1)
    return st, xy_v[ix], lin_v[il], valid[il]


top_by_si = sorted(place_keys, key=lambda k: -si_place[k])
chosen = []
for k in top_by_si:
    if all(abs(peak_place[k] - peak_place[c]) > 0.2 for c in chosen):
        chosen.append(k)
    if len(chosen) == 6:
        break
chosen = sorted(chosen, key=lambda k: peak_place[k])
print("example cells:", chosen)

fig, axes = plt.subplots(6, 3, figsize=(12, 16))
for row, k in enumerate(chosen):
    st, sp_xy, sp_lin, sp_valid = spike_positions(k)
    rm = results[k]["rate_map"]

    ax = axes[row, 0]
    ax.plot(xy_v[:, 0], xy_v[:, 1], lw=0.05, alpha=0.3, color="gray")
    m = ~np.isnan(sp_xy[:, 0]) & sp_valid
    ax.plot(sp_xy[m, 0], sp_xy[m, 1], ".", ms=2, color="crimson")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if row == 0:
        ax.set_title("Spike locations (red)", fontsize=10)
    ax.set_ylabel(f"unit {k}", fontsize=9)

    ax = axes[row, 1]
    ax.plot(BIN_CENTERS, rm, color="black", lw=1.5)
    ax.fill_between(BIN_CENTERS, rm, color="black", alpha=0.2)
    ax.set_xlim(0, 1.6)
    if row == 0:
        ax.set_title("Firing rate map", fontsize=10)
    if row == 5:
        ax.set_xlabel("Position (m)")
    ax.set_ylabel("Hz", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.text(0.02, 0.85, f"SI={results[k]['spatial_info']:.2f}",
            transform=ax.transAxes, fontsize=8)

    ax = axes[row, 2]
    ax.plot(BIN_CENTERS, results[k]["rate_map_pos"], color="tab:blue",
            lw=1.2, label="runs to 1.6 m")
    ax.plot(BIN_CENTERS, results[k]["rate_map_neg"], color="tab:orange",
            lw=1.2, label="runs to 0 m")
    ax.set_xlim(0, 1.6)
    if row == 0:
        ax.set_title("By running direction", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    if row == 5:
        ax.set_xlabel("Position (m)")
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig("fig2_example_place_cells.png", dpi=150)
plt.close()
print("saved fig2_example_place_cells.png")

# %% [markdown]
# ## Top place cells by spatial information

# %%
top24 = top_by_si[:24]
fig, axes = plt.subplots(4, 6, figsize=(14, 9), sharex=True)
for ax, k in zip(axes.flat, top24):
    rm = results[k]["rate_map"]
    ax.fill_between(BIN_CENTERS, rm, color="darkblue", alpha=0.7)
    ax.set_xlim(0, 1.6)
    ax.set_title(f"u{k}  SI={results[k]['spatial_info']:.2f}", fontsize=8)
    ax.tick_params(labelsize=7)
for ax in axes[-1]:
    ax.set_xlabel("Position (m)", fontsize=8)
plt.suptitle("Top 24 place cells by spatial information", y=1.00, fontsize=12)
plt.tight_layout()
plt.savefig("fig3_top24_ratemaps.png", dpi=150)
plt.close()
print("saved fig3_top24_ratemaps.png")

# %% [markdown]
# ## The population tiles the track
#
# Normalized rate maps of all place cells, sorted by peak position, show the
# classic result: place fields are distributed across the whole track, so the
# population provides a code for every position. Splitting by running
# direction (right panel) shows that many fields appear in only one
# direction.

# %%
place_sorted = sorted(place_keys, key=lambda k: peak_place[k])
maps = np.array([results[k]["rate_map"] for k in place_sorted])
maps_norm = maps / np.nanmax(maps, axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax = axes[0]
im = ax.imshow(maps_norm, aspect="auto", cmap="viridis",
               extent=[0, 1.6, len(place_sorted), 0])
ax.set_xlabel("Position (m)"); ax.set_ylabel("Place cell (sorted by peak)")
ax.set_title(f"Normalized rate maps (n={len(place_sorted)})")
plt.colorbar(im, ax=ax, label="Norm. rate", shrink=0.8)

ax = axes[1]
maps_p = np.array([results[k]["rate_map_pos"] for k in place_sorted])
maps_n = np.array([results[k]["rate_map_neg"] for k in place_sorted])
both = np.hstack([maps_p, maps_n])
both_norm = both / np.nanmax(both, axis=1, keepdims=True)
im = ax.imshow(both_norm, aspect="auto", cmap="viridis",
               extent=[0, 3.2, len(place_sorted), 0])
ax.axvline(1.6, color="white", lw=1, ls="--")
ax.set_xlabel("left: runs to 1.6 m   |   right: runs to 0 m")
ax.set_ylabel("Place cell (sorted by peak)")
ax.set_title("Rate maps by running direction")
ax.set_xticks([0, 0.8, 1.6, 2.4, 3.2])
ax.set_xticklabels(["0", "0.8", "1.6 | 0", "0.8", "1.6"])
plt.colorbar(im, ax=ax, label="Norm. rate", shrink=0.8)
plt.tight_layout()
plt.savefig("fig4_population.png", dpi=150)
plt.close()
print("saved fig4_population.png")

# %% [markdown]
# ## Spatial information statistics
#
# Excitatory units (putative pyramidal cells) carry far more spatial
# information than inhibitory units (putative interneurons). The observed
# spatial information of place cells lies well above the circular-shift
# null distribution.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

ax = axes[0]
ax.hist(si_all[exc], bins=30, color="darkblue", alpha=0.8, label="excitatory")
ax.hist(si_all[~exc], bins=15, color="crimson", alpha=0.6, label="inhibitory")
ax.set_xlabel("Spatial information (bits/spike)")
ax.set_ylabel("Units")
ax.set_title("A  Spatial information distribution", loc="left", fontsize=11)
ax.legend(fontsize=8)

ax = axes[1]
null_mean = np.array([results[k]["si_null_mean"] for k in unit_keys])
m = exc & ~np.isnan(null_mean)
ax.scatter(null_mean[m], si_all[m], s=12,
           c=np.where(is_place[m], "darkblue", "lightgray"),
           edgecolors="none", alpha=0.8)
lim = [0, max(np.nanmax(null_mean[m]), np.nanmax(si_all[m])) * 1.05]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("Shuffle mean SI (bits/spike)")
ax.set_ylabel("Observed SI (bits/spike)")
ax.set_title("B  Observed vs shuffled spatial info", loc="left", fontsize=11)
ax.set_xlim(lim); ax.set_ylim(lim)

ax = axes[2]
m = exc & ~np.isnan(p_all)
ax.scatter(rate_all[m], si_all[m], s=12,
           c=np.where(is_place[m], "darkblue", "lightgray"),
           alpha=0.8, edgecolors="none")
ax.set_xscale("log")
ax.set_xlabel("Mean rate on track (Hz)")
ax.set_ylabel("Spatial information (bits/spike)")
ax.set_title("C  Rate vs spatial info (blue = place cell)", loc="left", fontsize=11)
plt.tight_layout()
plt.savefig("fig5_spatial_info_stats.png", dpi=150)
plt.close()
print("saved fig5_spatial_info_stats.png")

# %% [markdown]
# ## Direction selectivity
#
# On a linear track, many place cells fire only when the animal runs through
# the field in one direction. We quantify this with the correlation between
# the two direction-specific rate maps: values near 1 mean
# direction-invariant fields, values near or below 0 mean strongly
# direction-selective fields.

# %%
dir_corr = []
for k in place_keys:
    a = np.nan_to_num(results[k]["rate_map_pos"])
    b = np.nan_to_num(results[k]["rate_map_neg"])
    dir_corr.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan)
dir_corr = np.array(dir_corr)

order = np.argsort(dir_corr)
dir_examples = [place_keys[i] for i in order[:3]]

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for j, k in enumerate(dir_examples):
    ax = axes[j]
    ax.plot(BIN_CENTERS, results[k]["rate_map_pos"], color="tab:blue",
            lw=1.5, label="runs to 1.6 m")
    ax.plot(BIN_CENTERS, results[k]["rate_map_neg"], color="tab:orange",
            lw=1.5, label="runs to 0 m")
    ax.set_xlim(0, 1.6)
    ax.set_title(f"unit {k}, r={dir_corr[order[j]]:.2f}", fontsize=9)
    ax.set_xlabel("Position (m)", fontsize=8)
    if j == 0:
        ax.set_ylabel("Firing rate (Hz)")
        ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

ax = axes[3]
ax.hist(dir_corr[~np.isnan(dir_corr)], bins=25, color="slategray")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Correlation between direction maps")
ax.set_ylabel("Place cells")
ax.set_title("Directional consistency", fontsize=9)
plt.tight_layout()
plt.savefig("fig6_directionality.png", dpi=150)
plt.close()
print("saved fig6_directionality.png")
print("median direction correlation: %.2f" % np.nanmedian(dir_corr))
print("fraction of place cells with r < 0.5: %.2f" % np.nanmean(dir_corr < 0.5))

# %% [markdown]
# ## Model-based confirmation: Poisson GLM with a spline position basis
#
# As a complement to the histogram-based rate maps, we fit each place cell
# with a Poisson GLM whose only predictor is linearized position, expanded in
# 12 B-spline basis functions (nemos). Spikes are counted in the native
# position-sample bins (~26 ms). The GLM tuning curves should reproduce the
# empirical rate maps if the histogram estimate is sound. Goodness of fit is
# reported as Cohen's (deviance-based) pseudo-R^2; values of 0.05-0.3 are
# expected for position-only models of noisy spike trains.

# %%
N_BASIS = 12
basis = nmo.basis.BSplineEval(n_basis_funcs=N_BASIS)
X = np.asarray(basis.compute_features(lin_v)[valid])
grid = np.linspace(0, 1.6, 200)
X_grid = np.asarray(basis.compute_features(grid))
print("design matrix:", X.shape)


def cohen_pseudo_r2(model, X, y):
    """Deviance-based (Cohen) pseudo-R^2 for a fitted Poisson GLM."""
    mu = np.clip(np.asarray(model.predict(X)), 1e-12, None)
    ll_model = np.sum(y * np.log(mu) - mu)
    lam0 = max(y.mean(), 1e-12)
    ll_null = np.sum(y * np.log(lam0) - lam0)
    y_pos = y[y > 0]
    ll_sat = np.sum(y_pos * np.log(y_pos) - y_pos)
    denom = ll_sat - ll_null
    if abs(denom) < 1e-9:
        return np.nan
    return float(1 - (ll_sat - ll_model) / denom)


glm_curves, glm_peak, emp_peak, glm_score = {}, {}, {}, {}
for k in tqdm(place_keys, desc="GLM fits"):
    st = units[k].restrict(maze).t
    counts, _ = np.histogram(st, bins=np.append(lin_t, lin_t[-1] + dt))
    y = counts[valid].astype(float)
    model = nmo.glm.GLM(solver_name="LBFGS",
                        solver_kwargs=dict(tol=1e-10, maxiter=1000))
    model.fit(X, y)
    rate_grid = np.exp(model.intercept_ + X_grid @ model.coef_) / dt
    glm_curves[k] = rate_grid
    glm_peak[k] = grid[np.argmax(rate_grid)]
    emp_peak[k] = peak_place[k]
    glm_score[k] = cohen_pseudo_r2(model, X, y)

np.savez_compressed("glm_results.npz", place_keys=np.array(place_keys),
                    grid=grid,
                    curves=np.array([glm_curves[k] for k in place_keys]),
                    glm_peak=np.array([glm_peak[k] for k in place_keys]),
                    emp_peak=np.array([emp_peak[k] for k in place_keys]),
                    glm_score=np.array([glm_score[k] for k in place_keys]))

# %%
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, k in zip(axes.flat, chosen):
    rm = results[k]["rate_map"]
    ax.fill_between(BIN_CENTERS, rm, color="gray", alpha=0.4, label="empirical")
    ax.plot(grid, glm_curves[k], color="crimson", lw=2, label="GLM (Poisson, spline)")
    ax.set_xlim(0, 1.6)
    ax.set_title(f"unit {k}  (pseudo-R$^2$={glm_score[k]:.2f})", fontsize=10)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Firing rate (Hz)")
axes.flat[0].legend(fontsize=8)
plt.suptitle("GLM-predicted vs empirical position tuning", fontsize=12)
plt.tight_layout()
plt.savefig("fig7_glm_tuning.png", dpi=150)
plt.close()
print("saved fig7_glm_tuning.png")

# %%
gp = np.array([glm_peak[k] for k in place_keys])
ep = np.array([emp_peak[k] for k in place_keys])
gs = np.array([glm_score[k] for k in place_keys])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
ax.scatter(ep, gp, s=15, color="darkblue", alpha=0.7)
ax.plot([0, 1.6], [0, 1.6], "k--", lw=1)
ax.set_xlabel("Empirical rate-map peak (m)")
ax.set_ylabel("GLM tuning-curve peak (m)")
r_peak = np.corrcoef(ep, gp)[0, 1]
ax.set_title(f"A  Peak position agreement (r={r_peak:.3f})", loc="left", fontsize=11)

ax = axes[1]
ax.hist(gs[~np.isnan(gs)], bins=30, color="slategray")
ax.set_xlabel("GLM pseudo-R$^2$")
ax.set_ylabel("Place cells")
ax.set_title("B  GLM goodness of fit", loc="left", fontsize=11)
plt.tight_layout()
plt.savefig("fig8_glm_validation.png", dpi=150)
plt.close()
print("saved fig8_glm_validation.png")
print("GLM vs empirical peak correlation: %.3f" % r_peak)
print("median Cohen pseudo-R^2: %.3f" % np.nanmedian(gs))

# %% [markdown]
# ## Summary
#
# In this classic Buzsáki lab session, the large majority of active CA1
# pyramidal cells (85 of 120 excitatory units) show statistically
# significant spatial tuning on the 1.6 m linear track (circular-shift
# shuffle, p < 0.05). Their place fields tile the full extent of the track,
# and most are direction-selective: the median correlation between
# direction-specific rate maps is ~0.35, and 59% of place cells have
# r < 0.5. Poisson GLMs with a spline position basis independently reproduce
# the empirical rate maps (peak-position correlation r = 0.96), confirming
# that the histogram-based place field estimates are sound.
