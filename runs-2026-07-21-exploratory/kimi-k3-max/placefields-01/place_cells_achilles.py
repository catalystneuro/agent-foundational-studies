# %% [markdown]
# # Hippocampal Place Cells on a Linear Track
#
# This notebook demonstrates hippocampal place cells using a real recording from the
# [DANDI Archive](https://dandiarchive.org). When a rat runs along a track, individual
# CA1 pyramidal cells fire selectively at particular locations ("place fields"). We
# stream one classic session from the Buzsáki lab, extract run epochs on a 1.6 m linear
# maze, compute firing-rate maps for every recorded unit, quantify spatial tuning with
# Skaggs spatial information, assess significance with a circular time-shift shuffle,
# examine direction selectivity, and finally fit model-based tuning curves with a
# Poisson GLM (nemos).
#
# **Dataset:** DANDI dandiset
# [000044](https://dandiarchive.org/dandiset/000044) ("Diversity in neural firing
# dynamics supports both rigid and learned hippocampal sequences"), session
# `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`: 137 CA1 units (tetrodes,
# left and right CA1) recorded while rat Achilles ran back and forth on a 1.6 m linear
# maze for ~35 minutes, with PRE and POST sleep epochs.
#
# The 8.7 GB NWB file is streamed with `remfile` + a local disk cache; nothing is
# downloaded in full.

# %% [markdown]
# ## Setup and streaming data access

# %%
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: save figures, never plt.show()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import requests
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# DANDI asset: sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb (dandiset 000044)
ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"
CACHE_DIR = "/tmp/remfile_cache_placefields"

# Resolve the DANDI download redirect to a presigned S3 URL. We must not follow the
# redirect ourselves: the presigned URL is valid for GET only, and remfile issues its
# own HTTP range GETs against it.
r = requests.get(f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/",
                 allow_redirects=False, timeout=60, stream=True)
r.raise_for_status()
s3_url = r.headers["Location"]

rem_file = remfile.File(s3_url, disk_cache=remfile.DiskCache(CACHE_DIR))
h5py_file = h5py.File(rem_file, "r")
nwbfile = NWBHDF5IO(file=h5py_file).read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# The file contains spike times for 137 sorted units (`units`), 2D position
# (`1.6mLinearMazeSpatialSeries`, ~39 Hz), a linearized track coordinate
# (`1.6mLinearMazeLinearizedTimeSeries`, 0-1.6 m), and epoch labels
# (PRE / Maze / POST). We work in the MazeEpoch only.

# %%
units = nwb["units"]
unit_ids = np.array(list(units.keys()))
cell_type = np.array([units.get_info("cell_type")[u] for u in unit_ids])
location = np.array([units.get_info("location")[u] for u in unit_ids])
print(f"{len(unit_ids)} units: "
      f"{(cell_type == 'excitatory').sum()} excitatory, "
      f"{(cell_type == 'inhibitory').sum()} inhibitory")

epochs = nwb["epochs"]
maze = epochs[epochs["label"] == "MazeEpoch"]
print(maze)

pos = nwb["1.6mLinearMazeSpatialSeries"].restrict(maze)   # x, y in meters
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"].restrict(maze)  # 0-1.6 m, NaN off-track
t = pos.t
xy = pos.values
lin_v = lin.values[:, 0]
fs = lin.rate
print(f"position: {len(t)} samples at {fs:.1f} Hz over "
      f"{t[-1] - t[0]:.0f} s; linearized coordinate valid "
      f"{np.mean(~np.isnan(lin_v)) * 100:.1f}% of the epoch")

# %% [markdown]
# The linearized coordinate is defined only while the animal is actually running on
# the track stem (it is exactly a linear function of the x coordinate; the end
# platforms are excluded). We therefore define the **run epoch** as contiguous bouts
# with a valid linearized position and |speed| > 5 cm/s, bridging gaps < 0.5 s and
# dropping bouts < 1 s.

# %%
def mask_to_intervalset(t, mask, gap_tol=0.5, min_dur=1.0):
    """Boolean mask on timebase t -> IntervalSet, bridging short gaps."""
    idx = np.where(mask)[0]
    splits = np.where(np.diff(t[idx]) > gap_tol)[0]
    starts = np.concatenate([[idx[0]], idx[splits + 1]])
    ends = np.concatenate([idx[splits], [idx[-1]]])
    s, e = t[starts], t[ends]
    keep = (e - s) >= min_dur
    return nap.IntervalSet(start=s[keep], end=e[keep])


# speed from the linearized coordinate, lightly smoothed (~128 ms boxcar)
speed = np.convolve(np.gradient(lin_v, t), np.ones(5) / 5, mode="same")
direction = np.sign(speed)  # +1: increasing position, -1: decreasing

run_mask = ~np.isnan(lin_v) & (np.abs(speed) > 0.05)
run_ep = mask_to_intervalset(t, run_mask, gap_tol=0.5, min_dur=1.0)
print(f"run epoch: {len(run_ep)} bouts, {run_ep.tot_length():.1f} s total "
      f"({run_ep.tot_length() / (t[-1] - t[0]) * 100:.0f}% of the maze epoch)")

# Extract each unit's spikes during runs (this is the only heavy read from the stream)
spikes = {}
for u in tqdm(unit_ids, desc="extracting run-epoch spikes"):
    spikes[u] = units[u].restrict(run_ep).t
n_spikes_run = np.array([len(spikes[u]) for u in unit_ids])
print(f"spikes per unit during runs: median {np.median(n_spikes_run):.0f}, "
      f"max {n_spikes_run.max()}")

# %% [markdown]
# ## Figure 2: behavioral coverage
#
# Occupancy of each 2 cm position bin during runs (overall and per running
# direction), the speed distribution, and position over the whole epoch.

# %%
NB_BINS = 80  # 2 cm bins
TRACK_RANGE = (0.0, 1.6)
edges = np.linspace(*TRACK_RANGE, NB_BINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])

# run-epoch sample selection and a continuous "run-time" coordinate
in_run = np.zeros(len(t), dtype=bool)
for s, e in zip(run_ep.start, run_ep.end):
    in_run[(t >= s) & (t <= e)] = True
in_run &= ~np.isnan(lin_v)
idx_run = np.where(in_run)[0]
t_run, lin_run, dir_run = t[idx_run], lin_v[idx_run], direction[idx_run]
dt = np.median(np.diff(t))
rt = np.arange(len(t_run)) * dt  # cumulative run time, bouts concatenated

occ_s = np.histogram(lin_run, bins=edges)[0] / fs
occ_s_posdir = np.histogram(lin_run[dir_run > 0], bins=edges)[0] / fs
occ_s_negdir = np.histogram(lin_run[dir_run < 0], bins=edges)[0] / fs
print(f"run time kept: {rt[-1] + dt:.1f} s; occupancy per bin: "
      f"median {np.median(occ_s):.2f} s (min {occ_s.min():.2f}, max {occ_s.max():.2f})")
print(f"positive-direction time {occ_s_posdir.sum():.1f} s, "
      f"negative-direction time {occ_s_negdir.sum():.1f} s")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.16, wspace=0.3)

ax = axes[0]
ax.bar(centers, occ_s, width=centers[1] - centers[0], color="0.75", label="all runs")
ax.step(centers, occ_s_posdir, where="mid", color="tab:red", lw=1.4, label="positive dir.")
ax.step(centers, occ_s_negdir, where="mid", color="tab:blue", lw=1.4, label="negative dir.")
ax.set_xlabel("track position (m)")
ax.set_ylabel("occupancy (s)")
ax.set_title("A  Position occupancy during runs", loc="left")
ax.legend(frameon=False)

ax = axes[1]
sp = np.abs(speed[~np.isnan(speed)]) * 100
ax.hist(sp, bins=np.linspace(0, 120, 61), color="0.5")
ax.axvline(5, color="k", ls="--", lw=1, label="run threshold (5 cm/s)")
ax.set_xlabel("|speed| (cm/s)")
ax.set_ylabel("samples")
ax.set_title("B  Speed distribution (MazeEpoch)", loc="left")
ax.legend(frameon=False, loc="upper right")

ax = axes[2]
step = 20
ax.plot(t[::step] - t[0], lin_v[::step], ".", ms=1.2, color="0.4", alpha=0.6)
ax.set_xlabel("time in MazeEpoch (s)")
ax.set_ylabel("track position (m)")
ax.set_title("C  Position over the full maze epoch", loc="left")

fig.savefig("fig2_behavior.png")
plt.close(fig)
print("saved fig2_behavior.png")

# %% [markdown]
# ## Rate maps, spatial information, and shuffle significance
#
# For each unit we compute the occupancy-normalized firing-rate map (Gaussian
# smoothed, sigma = 2 bins = 4 cm), and Skaggs spatial information
#
# $$SI = \sum_i p_i \frac{r_i}{\bar r} \log_2 \frac{r_i}{\bar r}$$
#
# in bits per spike, where $r_i$ is the rate in bin $i$, $p_i$ the occupancy fraction,
# and $\bar r$ the mean rate. Significance uses 200 circular time shifts of the spike
# train within the concatenated run epoch: a unit is a **place cell** if it is
# classified excitatory, its observed SI exceeds the 95th percentile of its shuffle
# distribution, its smoothed peak rate is at least 1 Hz, and its mean run-epoch rate
# is at least 0.05 Hz.

# %%
SMOOTH_SIGMA_BINS = 2.0
MIN_OCC_S = 0.05
N_SHUFFLE = 200
RNG_SEED = 42


def smooth_ratemap(counts, occ_s):
    """Rate map from smoothed counts / smoothed occupancy; empty bins -> NaN."""
    c_s = gaussian_filter1d(np.nan_to_num(counts), SMOOTH_SIGMA_BINS)
    o_s = gaussian_filter1d(np.nan_to_num(occ_s), SMOOTH_SIGMA_BINS)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(o_s > 0, c_s / o_s, np.nan)


def spatial_information(rate, occ_s):
    """Skaggs spatial information (bits/spike) over bins with enough occupancy."""
    m = np.isfinite(rate) & (occ_s >= MIN_OCC_S)
    if m.sum() < 5:
        return np.nan
    p = occ_s[m] / occ_s[m].sum()
    r = rate[m]
    rbar = (p * r).sum()
    if rbar <= 0:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(p * (r / rbar) * np.log2(r / rbar))


n_units = len(unit_ids)
ratemaps_sm = np.full((n_units, NB_BINS), np.nan)
ratemaps_pos = np.full((n_units, NB_BINS), np.nan)
ratemaps_neg = np.full((n_units, NB_BINS), np.nan)
si = np.full(n_units, np.nan)
si_shuffle = np.full((n_units, N_SHUFFLE), np.nan)
mean_rate = np.full(n_units, np.nan)
peak_rate = np.full(n_units, np.nan)

rng = np.random.default_rng(RNG_SEED)
total_rt = rt[-1] + dt

for i, u in enumerate(tqdm(unit_ids, desc="rate maps + shuffles")):
    spk = spikes[u]
    if len(spk) < 20:
        continue
    # position and run-time of each spike (spikes land on the position timebase)
    spk_idx = np.clip(np.searchsorted(t_run, spk), 0, len(t_run) - 1)
    rt_spk, lin_spk, dir_spk = rt[spk_idx], lin_run[spk_idx], dir_run[spk_idx]

    counts = np.histogram(lin_spk, bins=edges)[0]
    ratemaps_sm[i] = smooth_ratemap(counts, occ_s)
    si[i] = spatial_information(ratemaps_sm[i], occ_s)
    mean_rate[i] = len(spk) / total_rt
    peak_rate[i] = np.nanmax(ratemaps_sm[i])

    ratemaps_pos[i] = smooth_ratemap(np.histogram(lin_spk[dir_spk > 0], bins=edges)[0],
                                     occ_s_posdir)
    ratemaps_neg[i] = smooth_ratemap(np.histogram(lin_spk[dir_spk < 0], bins=edges)[0],
                                     occ_s_negdir)

    # circular time-shift shuffle within the concatenated run epoch
    for j in range(N_SHUFFLE):
        rt_sh = (rt_spk + rng.uniform(dt, total_rt)) % total_rt
        lin_sh = np.interp(rt_sh, rt, lin_run)
        rate_sh = smooth_ratemap(np.histogram(lin_sh, bins=edges)[0], occ_s)
        si_shuffle[i, j] = spatial_information(rate_sh, occ_s)

si_thresh_95 = np.nanpercentile(si_shuffle, 95, axis=1)
is_place = ((cell_type == "excitatory")
            & (si > si_thresh_95)
            & (peak_rate >= 1.0)
            & (mean_rate >= 0.05))

print(f"\nplace cells: {is_place.sum()}/{(cell_type == 'excitatory').sum()} "
      f"excitatory units pass the shuffle test (p < 0.05) and rate criteria")
print(f"spatial information, excitatory: median "
      f"{np.nanmedian(si[cell_type == 'excitatory']):.2f} bits/spike")
print(f"spatial information, inhibitory: median "
      f"{np.nanmedian(si[cell_type == 'inhibitory']):.2f} bits/spike")

# %% [markdown]
# ## Figure 1: raw data and place-cell sequences
#
# Now that field locations are known we can build the raw-data figure: the
# trajectory, the speed profile of a run-dense 60 s window, the linearized position
# with spikes of six example place cells, and the population raster. In panel D,
# place cells are sorted by field location: each run sweeps through the sorted
# population, producing the visible sequential activation (place-cell sequences).

# %%
def pick_window(run_starts, dur=60.0):
    """The dur-second window containing the most run-bout starts."""
    best_t0, best_n = run_starts[0], 0
    for s in run_starts:
        n = ((run_starts >= s) & (run_starts < s + dur)).sum()
        if n > best_n:
            best_t0, best_n = s, n
    return best_t0, int(best_n)


WIN = 60.0
t0, n_runs = pick_window(run_ep.start, WIN)
win = (t >= t0) & (t < t0 + WIN)

fig = plt.figure(figsize=(11, 8.5))
gs = GridSpec(3, 2, height_ratios=[1.15, 1, 1], hspace=0.38, wspace=0.22,
              left=0.07, right=0.97, top=0.94, bottom=0.07)

ax = fig.add_subplot(gs[0, 0])
ax.plot(xy[:, 0], xy[:, 1], color="0.8", lw=0.3, zorder=1)
ax.plot(xy[win, 0], xy[win, 1], color="tab:blue", lw=1.0, zorder=2,
        label=f"{int(WIN)} s window shown below")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("A  Trajectory on the 1.6 m linear maze\n(MazeEpoch, 34.5 min)", loc="left")
ax.legend(loc="upper left", frameon=False, fontsize=8)
ax.set_aspect("equal")

ax = fig.add_subplot(gs[0, 1])
ax.plot(t[win] - t0, np.abs(speed[win]) * 100, color="0.4", lw=0.7)
for s, e in zip(run_ep.start, run_ep.end):
    if e < t0 or s > t0 + WIN:
        continue
    ax.axvspan(max(s, t0) - t0, min(e, t0 + WIN) - t0, color="tab:green", alpha=0.15, lw=0)
ax.axhline(5, color="k", ls="--", lw=0.8, label="run threshold (5 cm/s)")
ax.set_xlim(0, WIN)
ax.set_xlabel("time in window (s)")
ax.set_ylabel("|speed| (cm/s)")
ax.set_title(f"B  Speed during the window ({n_runs} runs, shaded)", loc="left")
ax.legend(loc="upper right", frameon=False, fontsize=8)

place_idx = np.where(is_place)[0]
peaks = np.array([centers[np.nanargmax(ratemaps_sm[i])]
                  if np.isfinite(ratemaps_sm[i]).any() else np.nan for i in place_idx])

ax = fig.add_subplot(gs[1, :])
ax.plot(t[win] - t0, lin_v[win], color="0.5", lw=0.8, zorder=1)
order = np.argsort(peaks)
picks6 = place_idx[order[np.linspace(0, len(order) - 1, 6).astype(int)]]
colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
for k, ui in enumerate(picks6):
    spk_win = spikes[unit_ids[ui]]
    spk_win = spk_win[(spk_win >= t0) & (spk_win < t0 + WIN)]
    ax.plot(spk_win - t0, np.interp(spk_win, t, lin_v), "o", ms=3,
            color=colors[k], label=f"unit {unit_ids[ui]}", zorder=3)
ax.set_xlim(0, WIN)
ax.set_ylim(-0.05, 1.8)
ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
ax.set_xlabel("time in window (s)")
ax.set_ylabel("track position (m)")
ax.set_title("C  Linearized position with spikes of six example place cells", loc="left")
ax.legend(loc="upper right", ncol=6, frameon=False, fontsize=8,
          columnspacing=0.8, handletextpad=0.1)

ax = fig.add_subplot(gs[2, :])
sort_idx = place_idx[np.argsort(peaks)]
for row, ui in enumerate(sort_idx):
    spk_win = spikes[unit_ids[ui]]
    spk_win = spk_win[(spk_win >= t0) & (spk_win < t0 + WIN)]
    ax.plot(spk_win - t0, np.full_like(spk_win, row), "|", ms=2.5, color="k", alpha=0.6)
ax.set_xlim(0, WIN)
ax.set_ylim(-1, len(sort_idx))
ax.set_xlabel("time in window (s)")
ax.set_ylabel("place cells (sorted)")
ax.set_title("D  Raster of all significant place cells, sorted by place-field location",
             loc="left")

fig.savefig("fig1_raw_data.png")
plt.close(fig)
print("saved fig1_raw_data.png")

# %% [markdown]
# ## Figure 3: example place cells
#
# One high-SI place cell per eighth of the track. Top: spike locations on the maze
# (red) over the full trajectory (gray). Bottom: direction-split rate maps. Many
# fields fire almost exclusively in one running direction.

# %%
def pick_example_cells(n=8):
    """One high-SI place cell per n-th of the track, so fields tile the track."""
    picks = []
    for b in range(n):
        lo, hi = b * 1.6 / n, (b + 1) * 1.6 / n
        cand = place_idx[(peaks >= lo) & (peaks < hi)]
        if len(cand):
            picks.append(cand[np.argmax(si[cand])])
    return picks


picks = pick_example_cells(8)

fig = plt.figure(figsize=(12, 5.2))
gs = GridSpec(2, len(picks), hspace=0.28, wspace=0.45,
              left=0.05, right=0.98, top=0.88, bottom=0.12)
for k, ui in enumerate(picks):
    u = unit_ids[ui]
    spk = spikes[u]
    ax = fig.add_subplot(gs[0, k])
    ax.plot(xy[:, 0], xy[:, 1], color="0.75", lw=0.3, zorder=1)
    spk_xy = np.column_stack([np.interp(spk, t, xy[:, 0]),
                              np.interp(spk, t, xy[:, 1])])
    ax.plot(spk_xy[:, 0], spk_xy[:, 1], "o", ms=1.1, color="tab:red",
            alpha=0.45, zorder=2, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"unit {u}\nSI = {si[ui]:.2f} bits/spk", fontsize=8)

    ax = fig.add_subplot(gs[1, k])
    ax.fill_between(centers, ratemaps_pos[ui], color="tab:red", alpha=0.35, lw=0)
    ax.plot(centers, ratemaps_pos[ui], color="tab:red", lw=1.2, label="pos. dir.")
    ax.fill_between(centers, ratemaps_neg[ui], color="tab:blue", alpha=0.35, lw=0)
    ax.plot(centers, ratemaps_neg[ui], color="tab:blue", lw=1.2, label="neg. dir.")
    ax.set_xlim(0, 1.6)
    ax.set_xticks([0, 0.8, 1.6])
    if k == 0:
        ax.set_ylabel("firing rate (Hz)")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_xlabel("position (m)", labelpad=1)
    ax.tick_params(labelsize=7)

fig.suptitle("Example place cells: spike locations on the maze (top) and "
             "direction-split rate maps (bottom)", fontsize=11, y=0.97)
fig.savefig("fig3_example_place_cells.png")
plt.close(fig)
print("saved fig3_example_place_cells.png")

# %% [markdown]
# ## Figure 4: population rate maps tile the track
#
# Normalized rate maps of all place cells, sorted by peak location, shown separately
# for the two running directions. The diagonal band shows that place fields cover the
# whole track in both directions.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
fig.subplots_adjust(left=0.07, right=0.95, top=0.86, bottom=0.13, wspace=0.12)
for ax, (rm, name) in zip(axes, [(ratemaps_pos, "positive direction"),
                                 (ratemaps_neg, "negative direction")]):
    R = rm[place_idx]
    peak_pos = np.array([centers[np.nanargmax(r)] if np.isfinite(r).any() else np.inf
                         for r in R])
    R = R[np.argsort(peak_pos)]
    with np.errstate(invalid="ignore", divide="ignore"):
        Rn = R / np.nanmax(R, axis=1, keepdims=True)
    im = ax.imshow(Rn, aspect="auto", cmap="viridis", origin="upper",
                   extent=[0, 1.6, len(Rn), 0], vmin=0, vmax=1)
    ax.set_xlabel("track position (m)")
    ax.set_title(f"{name} (n={len(Rn)} place cells)", loc="left")
axes[0].set_ylabel("place cells (sorted by peak)")
cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("normalized firing rate")
fig.suptitle("Population rate maps tile the linear track", fontsize=11, y=0.97)
fig.savefig("fig4_population_ratemaps.png")
plt.close(fig)
print("saved fig4_population_ratemaps.png")

# %% [markdown]
# ## Figure 5: spatial information and place-cell classification

# %%
exc = cell_type == "excitatory"
inh = cell_type == "inhibitory"

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.16, wspace=0.32)

ax = axes[0]
bins = np.linspace(0, np.nanmax(si[exc]) * 1.05, 50)
ax.hist(si[exc & np.isfinite(si)], bins=bins, color="tab:red", alpha=0.75,
        label=f"excitatory (n={exc.sum()})")
ax.hist(si[inh & np.isfinite(si)], bins=bins, color="tab:blue", alpha=0.75,
        label=f"inhibitory (n={inh.sum()})")
ax.set_xlabel("spatial information (bits/spike)")
ax.set_ylabel("units")
ax.set_title("A  Spatial information by cell type", loc="left")
ax.legend(frameon=False)

ax = axes[1]
m = exc & np.isfinite(si) & np.isfinite(si_thresh_95)
ax.scatter(si_thresh_95[m], si[m], s=10, alpha=0.6,
           c=np.where(is_place[m], "tab:red", "0.6"))
lim = [0, max(np.nanmax(si[m]), np.nanmax(si_thresh_95[m])) * 1.05]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("shuffle 95th percentile (bits/spike)")
ax.set_ylabel("observed SI (bits/spike)")
ax.set_title("B  Observed vs shuffle threshold", loc="left")

ax = axes[2]
m = exc & np.isfinite(si)
ax.scatter(peak_rate[m & ~is_place], si[m & ~is_place], s=10, color="0.6", alpha=0.6,
           label="not significant")
ax.scatter(peak_rate[m & is_place], si[m & is_place], s=10, color="tab:red", alpha=0.7,
           label=f"place cells (n={int(is_place.sum())})")
ax.set_xscale("log")
ax.set_xlabel("peak firing rate (Hz)")
ax.set_ylabel("spatial information (bits/spike)")
ax.set_title("C  Place-cell classification", loc="left")
ax.legend(frameon=False, loc="upper left")

fig.savefig("fig5_spatial_info.png")
plt.close(fig)
print("saved fig5_spatial_info.png")

# %% [markdown]
# ## Figure 6: direction selectivity
#
# Place fields on linear tracks are often directional. We compare each place cell's
# rate map between the two running directions: peak rates, the correlation between
# the two direction-specific maps, and the field locations.

# %%
peak_pos = np.nanmax(ratemaps_pos[place_idx], axis=1)
peak_neg = np.nanmax(ratemaps_neg[place_idx], axis=1)


def dir_corr(i):
    """Correlation between a cell's two direction maps over jointly occupied bins."""
    a, b = ratemaps_pos[i], ratemaps_neg[i]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
        return np.nan  # silent or flat in one direction
    return np.corrcoef(a[m], b[m])[0, 1]


corr = np.array([dir_corr(i) for i in place_idx])
loc_pos = centers[np.nanargmax(ratemaps_pos[place_idx], axis=1)]
loc_neg = centers[np.nanargmax(ratemaps_neg[place_idx], axis=1)]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.16, wspace=0.32)

ax = axes[0]
ax.scatter(peak_pos, peak_neg, s=12, color="0.4", alpha=0.7)
lim = [0, max(peak_pos.max(), peak_neg.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("peak rate, positive dir. (Hz)")
ax.set_ylabel("peak rate, negative dir. (Hz)")
ax.set_title("A  Peak rates by direction", loc="left")

ax = axes[1]
corr_f = corr[np.isfinite(corr)]
ax.hist(corr_f, bins=np.linspace(-0.4, 1, 37), color="0.5")
ax.axvline(np.median(corr_f), color="tab:red", ls="--", lw=1,
           label=f"median r = {np.median(corr_f):.2f}")
ax.set_xlabel("correlation between direction maps")
ax.set_ylabel("place cells")
ax.set_title("B  Directional stability of rate maps", loc="left")
ax.legend(frameon=False)

ax = axes[2]
ax.scatter(loc_pos, loc_neg, s=12, color="0.4", alpha=0.7)
ax.plot([0, 1.6], [0, 1.6], "k--", lw=1)
ax.set_xlim(0, 1.6)
ax.set_ylim(0, 1.6)
ax.set_xlabel("peak location, positive dir. (m)")
ax.set_ylabel("peak location, negative dir. (m)")
ax.set_title("C  Field location by direction", loc="left")

fig.savefig("fig6_directionality.png")
plt.close(fig)
print("saved fig6_directionality.png")
print(f"direction-map correlation: median {np.median(corr_f):.2f} "
      f"(n={len(corr_f)} cells with both-direction fields); "
      f"{(corr_f > 0.5).mean() * 100:.0f}% with r > 0.5")

# %% [markdown]
# ## Figure 7: model-based tuning curves with a Poisson GLM (nemos)
#
# As a regression-based complement to the binned rate maps, we fit each example cell
# with a Poisson GLM: spike count in 25.6 ms bins is modeled as
# $\exp(\beta_0 + \sum_k \beta_k B_k(x))$ where $B_k$ are 12 B-spline basis functions
# of track position. A small ridge penalty keeps coefficients finite for cells with
# near-zero rates over part of the track. The fitted curves closely match the
# empirical rate maps while providing a smooth, model-based estimate.

# %%
import jax
jax.config.update("jax_enable_x64", True)
import nemos as nmo

N_BASIS = 12
BIN_SIZE = 0.0256  # ~ position sampling period

pos_tsd = nap.Tsd(t=t, d=lin_v).restrict(run_ep)
basis = nmo.basis.BSplineEval(n_basis_funcs=N_BASIS)
grid = np.linspace(0, 1.6, 200)
X_grid = basis.compute_features(grid)

fig, axes = plt.subplots(2, 4, figsize=(11.5, 5))
axes = axes.ravel()
fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.11,
                    hspace=0.5, wspace=0.35)

for k, ui in enumerate(picks):
    u = unit_ids[ui]
    count = nap.Ts(spikes[u]).count(BIN_SIZE, ep=run_ep)
    pos_matched = pos_tsd.interpolate(count)
    m = ~np.isnan(pos_matched.values)
    X = basis.compute_features(pos_matched[m])
    y = count[m]

    model = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                        regularizer_strength=1e-4,
                        solver_kwargs={"maxiter": 1000, "tol": 1e-6})
    model.fit(X, y)
    rate_grid = np.asarray(model.predict(X_grid)) / BIN_SIZE

    ax = axes[k]
    ax.fill_between(centers, ratemaps_sm[ui], color="0.7", alpha=0.5, lw=0,
                    label="empirical (smoothed)")
    ax.plot(grid, rate_grid, color="tab:purple", lw=1.6,
            label="GLM (Poisson, B-splines)")
    ax.set_xlim(0, 1.6)
    ax.set_xticks([0, 0.8, 1.6])
    ax.set_title(f"unit {u}", fontsize=9)
    if k == 0:
        ax.set_ylabel("firing rate (Hz)")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_xlabel("position (m)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for k in range(len(picks), len(axes)):
    axes[k].axis("off")

fig.suptitle("Poisson GLM tuning curves (B-spline basis on position) vs empirical "
             "rate maps", fontsize=11, y=0.97)
fig.savefig("fig7_glm_tuning.png")
plt.close(fig)
print("saved fig7_glm_tuning.png")

# %% [markdown]
# ## Summary
#
# Streaming the classic Achilles linear-track session from DANDI 000044, we recover
# the defining signatures of hippocampal place cells:
#
# * **Most CA1 pyramidal cells are place cells.** 83 of 120 excitatory units show
#   spatially informative firing (Skaggs SI above the 95th percentile of 200 circular
#   time-shift shuffles, peak rate >= 1 Hz). Median SI is ~0.9 bits/spike for
#   excitatory units vs ~0 for inhibitory interneurons.
# * **Place fields tile the track.** Sorted population rate maps form a diagonal band
#   covering the full 1.6 m in both running directions, and runs drive the sorted
#   population in sequence (Figure 1D).
# * **Many fields are directional.** The median correlation between a cell's two
#   direction-specific rate maps is only ~0.3; a subpopulation fires almost
#   exclusively in one direction.
# * **A Poisson GLM with a B-spline position basis** reproduces the empirical rate
#   maps, providing smooth model-based tuning curves.
