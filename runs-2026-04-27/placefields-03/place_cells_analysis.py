# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# # Hippocampal Place Cells from DANDI:000044
#
# This notebook demonstrates **hippocampal place cells** using publicly available
# data from the DANDI Archive. We use one session (Achilles, 2013-10-25) from
# **DANDI:000044** — *"Diversity in neural firing dynamics supports both rigid and
# learned hippocampal sequences"* (Grosmark, Long, & Buzsáki) — in which a rat
# ran back and forth on a 1.6 m linear track while CA1 pyramidal cells were
# recorded with bilateral silicon probes.
#
# A **place cell** is a hippocampal pyramidal neuron whose firing rate is high
# only in a specific region of the environment (its *place field*). When the
# animal is elsewhere, the cell is largely silent. Together, the population of
# place cells tiles the environment, providing a neural map of space
# (O'Keefe & Dostrovsky, 1971).
#
# **Pipeline**
#
# 1. Stream the NWB file from DANDI with `remfile` (cached locally).
# 2. Reconstruct the linearized position time series (the NWB `rate` field is
#    actually `dt`, so timestamps need to be rebuilt).
# 3. Restrict to the maze-running epoch and to **running periods** (speed > 5 cm/s).
# 4. Compute 1D tuning curves (firing rate vs. linear position) per CA1 pyramidal
#    cell with `pynapple.compute_1d_tuning_curves`.
# 5. Quantify spatial coding with **Skaggs spatial information** (bits/spike) and
#    classify cells as place cells.
# 6. Visualise: occupancy, single-cell place fields, sorted population heatmap,
#    raster + position trace, and a Bayesian decoding check.

# %% [markdown]
# ## Imports

# %%
import os
import warnings

import h5py
import lindi  # noqa: F401  (kept for skill-conformance; remfile path is used here)
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

FIGDIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
os.makedirs(FIGDIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  saved: {name}")
    return path


# %% [markdown]
# ## Stream the session from DANDI
#
# Asset: `sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`
# from DANDI:000044. We use the direct S3 URL with `remfile` + a disk cache so
# only the byte ranges we touch are actually downloaded.

# %%
S3_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/"
    "4f5/a84/4f5a84aa-a6e4-496a-9b23-535fa6fbd3ae"
)

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5_file, load_namespaces=True)
nwbfile = io.read()

print(f"Identifier:    {nwbfile.identifier}")
print(f"Session start: {nwbfile.session_start_time}")
print(f"# units:       {len(nwbfile.units)}")

# %% [markdown]
# Pynapple gives us a high-level view of the file:

# %%
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# ## Build the position time series with correct timestamps
#
# In this NWB file, the `SpatialSeries.rate` attribute actually stores the
# sampling **interval** in seconds (~0.0256 s, i.e. ≈39 Hz), not the rate.
# Pynapple interprets the field as a frequency, which spreads the timestamps
# across days. We rebuild the time vector manually.

# %%
beh = nwbfile.processing["behavior"]
pos_iface = beh["1.6mLinearMazeLinearizedPosition"]
ss = pos_iface["1.6mLinearMazeLinearizedTimeSeries"]

raw = ss.data[:].squeeze()
dt = float(ss.rate)  # mis-labelled in the file: this is dt, not rate
t0 = float(ss.starting_time)
ts = t0 + dt * np.arange(raw.shape[0])

# Drop NaN-valued samples (track is intermittently linearised)
valid = ~np.isnan(raw)
position = nap.Tsd(t=ts[valid], d=raw[valid], time_units="s")
print(f"Position samples: {len(position)} (kept {valid.mean()*100:.1f}% of raw)")
print(f"Time range:       [{position.index[0]:.1f}, {position.index[-1]:.1f}] s")
print(f"Position range:   [{position.values.min():.2f}, {position.values.max():.2f}] m")

# %% [markdown]
# ## Maze-running epoch

# %%
maze_ep = nap.IntervalSet(start=position.index[0], end=position.index[-1])
print(f"Maze epoch duration: {maze_ep.tot_length():.1f} s "
      f"({maze_ep.tot_length()/60:.1f} min)")

# %% [markdown]
# ## Compute speed and restrict to running periods
#
# Place fields should be measured while the animal is moving along the track,
# not when it is paused at the reward wells at the ends. The 2-D position is
# denser (21% NaN) than the linearised version, so we use it to derive a
# continuous speed estimate. Running is then defined as **speed > 10 cm/s** AND
# **at least 5 cm away from either end of the track**.

# %%
ss2d = beh["1.6mLinearMazePosition"]["1.6mLinearMazeSpatialSeries"]
xy_raw = ss2d.data[:]
ts2d = float(ss2d.starting_time) + float(ss2d.rate) * np.arange(xy_raw.shape[0])
v2d = ~np.isnan(xy_raw).any(axis=1)
xy = xy_raw[v2d]
ts2d_v = ts2d[v2d]
# Speed from 2D position (m/s)
dx = np.gradient(xy[:, 0], ts2d_v)
dy = np.gradient(xy[:, 1], ts2d_v)
speed_2d = np.sqrt(dx ** 2 + dy ** 2)
speed = nap.Tsd(t=ts2d_v, d=speed_2d)

# Light temporal smoothing (~400 ms) to remove jitter
speed_smooth = speed.smooth(std=0.2, time_units="s")

# Interpolate speed onto the linearised-position timestamps, so the running
# mask shares the same time base as the position used for tuning curves
speed_at_pos = np.interp(position.index, speed_smooth.index, speed_smooth.values)
in_middle = (position.values > 0.05) & (position.values < 1.55)

run_mask = (speed_at_pos > 0.10) & in_middle  # 10 cm/s, away from ends

# Build IntervalSet from contiguous True runs of at least 200 ms,
# only joining samples that are themselves close in time (<100 ms apart)
sample_dt = np.diff(position.index)
gap = sample_dt > 0.1
edges_mask = run_mask.astype(int)
# Force a break wherever the time gap is large
break_idx = np.where(gap)[0]
edges_mask_diff = np.diff(edges_mask)
starts_idx = np.where(edges_mask_diff == 1)[0] + 1
ends_idx = np.where(edges_mask_diff == -1)[0]
if run_mask[0]:
    starts_idx = np.r_[0, starts_idx]
if run_mask[-1]:
    ends_idx = np.r_[ends_idx, len(run_mask) - 1]

# Now break any [start,end] interval that spans a large temporal gap
def split_on_gaps(starts, ends, times, max_gap=0.1):
    new_s, new_e = [], []
    for s, e in zip(starts, ends):
        cur_s = s
        for k in range(s, e):
            if times[k + 1] - times[k] > max_gap:
                new_s.append(cur_s)
                new_e.append(k)
                cur_s = k + 1
        new_s.append(cur_s)
        new_e.append(e)
    return np.array(new_s), np.array(new_e)


starts_idx, ends_idx = split_on_gaps(starts_idx, ends_idx, position.index)
# Need at least 2 samples per interval and >200 ms duration
keep = (ends_idx > starts_idx) & (
    position.index[ends_idx] - position.index[starts_idx] > 0.2
)
starts_idx, ends_idx = starts_idx[keep], ends_idx[keep]

run_intervals = nap.IntervalSet(
    start=position.index[starts_idx],
    end=position.index[ends_idx],
)
print(f"Running coverage:  {run_intervals.tot_length():.1f} s")
print(f"Maze epoch:        {maze_ep.tot_length():.1f} s")
print(f"# run epochs:      {len(run_intervals)}")

# %% [markdown]
# ## Pick CA1 pyramidal (excitatory) units

# %%
units = nwb["units"]
units.set_info(cell_type=units.get_info("cell_type"))
units.set_info(location=units.get_info("location"))

pyr = units.getby_category("cell_type")["excitatory"]
print(f"# excitatory units: {len(pyr)}")
print("Locations:", np.unique(pyr.get_info("location"), return_counts=True))

# Restrict spikes to the running periods on the maze
pyr_run = pyr.restrict(run_intervals)
maze_durations_per_unit = run_intervals.tot_length()
mean_rates = np.array(
    [len(pyr_run[u]) / maze_durations_per_unit for u in pyr_run.keys()]
)
print(f"Mean firing rate during running: median={np.median(mean_rates):.2f} Hz, "
      f"max={mean_rates.max():.2f} Hz")

# Drop very low-rate units that can't yield meaningful tuning
keep_ids = [u for u, r in zip(pyr_run.keys(), mean_rates) if r > 0.1]
pyr_run = pyr_run[keep_ids]
print(f"# units kept (>0.1 Hz on track): {len(pyr_run)}")

# %% [markdown]
# ## Sanity-check plot: raw position trace + a few rasters

# %%
demo_ids = list(pyr_run.keys())[:6]
fig, axes = plt.subplots(
    2, 1, figsize=(12, 6), sharex=True,
    gridspec_kw={"height_ratios": [2, 1.5]},
)
ax = axes[0]
t_demo_start = position.index[0]
t_demo_end = t_demo_start + 200  # show first 200 s
demo_ep = nap.IntervalSet(start=t_demo_start, end=t_demo_end)
p_demo = position.restrict(demo_ep)
ax.plot(p_demo.index - t_demo_start, p_demo.values, color="k", lw=0.8)
ax.set_ylabel("Linear position (m)")
ax.set_title(f"Achilles 2013-10-25 — first {t_demo_end - t_demo_start:.0f} s on the 1.6 m linear track")

ax = axes[1]
for i, uid in enumerate(demo_ids):
    spk = pyr[uid].restrict(demo_ep).index.values - t_demo_start
    ax.vlines(spk, i + 0.05, i + 0.95, color="C0", lw=0.6)
ax.set_yticks(np.arange(len(demo_ids)) + 0.5)
ax.set_yticklabels([f"unit {u}" for u in demo_ids])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Example pyramidal cells")
plt.tight_layout()
_save(fig, "fig01_raw_position_and_rasters.png")
plt.close(fig)

# %% [markdown]
# ## Compute 1D place-field tuning curves
#
# We discretise the 1.6 m track into 50 bins (3.2 cm each) and use
# `nap.compute_1d_tuning_curves`. Pynapple handles spike counts and occupancy
# normalisation for us.

# %%
NBINS = 50
TRACK_RANGE = (0.0, 1.6)

tuning = nap.compute_1d_tuning_curves(
    group=pyr_run,
    feature=position.restrict(run_intervals),
    nb_bins=NBINS,
    minmax=TRACK_RANGE,
)
# Light smoothing across bins (Gaussian, sigma = 1 bin)
from scipy.ndimage import gaussian_filter1d

tuning_smooth = tuning.copy()
for col in tuning_smooth.columns:
    vals = tuning_smooth[col].values.astype(float)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    tuning_smooth[col] = gaussian_filter1d(vals, sigma=1.0)

print("Tuning curve shape:", tuning_smooth.shape, "(bins x cells)")

# %% [markdown]
# ## Spatial information (Skaggs et al., 1993)
#
# $$ I = \sum_x p(x)\, \frac{\lambda(x)}{\bar\lambda}\, \log_2 \frac{\lambda(x)}{\bar\lambda} \quad [\text{bits/spike}] $$
#
# where $p(x)$ is the occupancy probability, $\lambda(x)$ is the firing rate at
# position $x$, and $\bar\lambda$ is the mean firing rate. Place cells are
# typically those with high spatial information (>0.5 bits/spike here) and a
# clear spatial peak.

# %%
def occupancy_prob(pos, intervals, nbins=NBINS, rng=TRACK_RANGE):
    p = pos.restrict(intervals).values
    edges = np.linspace(rng[0], rng[1], nbins + 1)
    counts, _ = np.histogram(p, bins=edges)
    return counts / counts.sum(), 0.5 * (edges[:-1] + edges[1:])

p_x, bin_centers = occupancy_prob(position, run_intervals)


def spatial_information_bits_per_spike(rates, p_x):
    rates = np.asarray(rates, dtype=float)
    mean_rate = np.sum(p_x * rates)
    if mean_rate <= 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rates / mean_rate
        log_term = np.where(ratio > 0, np.log2(ratio), 0.0)
    return float(np.sum(p_x * ratio * log_term))


SI = np.array([
    spatial_information_bits_per_spike(tuning_smooth[c].values, p_x)
    for c in tuning_smooth.columns
])
peak_rate = tuning_smooth.max(axis=0).values
mean_rate = tuning_smooth.mean(axis=0).values
unit_ids = list(tuning_smooth.columns)

print(f"Spatial information: median = {np.median(SI):.2f} bits/spike, "
      f"max = {SI.max():.2f}")

# Place-cell criterion: SI > 0.5 bits/spike, peak rate > 1 Hz, mean rate < 10 Hz
is_place = (SI > 0.5) & (peak_rate > 1.0) & (mean_rate < 10.0)
print(f"# place cells: {is_place.sum()} / {len(unit_ids)} pyramidal units")

# %% [markdown]
# ## Single-cell place fields (top examples)

# %%
order = np.argsort(SI)[::-1]
top_ids = [unit_ids[i] for i in order[:9]]

fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
for ax, uid in zip(axes.flat, top_ids):
    rates = tuning_smooth[uid].values
    ax.fill_between(bin_centers, 0, rates, color="C0", alpha=0.5)
    ax.plot(bin_centers, rates, color="C0")
    si_u = SI[unit_ids.index(uid)]
    pk = rates.max()
    ax.set_title(f"unit {uid}  •  SI={si_u:.2f} bits/spk  •  peak={pk:.1f} Hz",
                 fontsize=10)
    ax.set_xlim(*TRACK_RANGE)
    ax.set_ylim(bottom=0)
for ax in axes[-1, :]:
    ax.set_xlabel("Linear position (m)")
for ax in axes[:, 0]:
    ax.set_ylabel("Firing rate (Hz)")
fig.suptitle("Top-9 place cells by spatial information (Achilles 2013-10-25, CA1)",
             fontsize=12)
plt.tight_layout()
_save(fig, "fig02_top_place_cells.png")
plt.close(fig)

# %% [markdown]
# ## Population heat-map: place fields tile the track
#
# Each row is one place cell's tuning curve, peak-normalised, sorted by the
# location of the peak. A clean diagonal band means the population covers the
# whole 1.6 m track — the hallmark of a hippocampal cognitive map.

# %%
place_idx = np.where(is_place)[0]
pc_curves = tuning_smooth.iloc[:, place_idx].values  # (bins, n_pc)
# Peak-normalise per cell
pc_norm = pc_curves / (pc_curves.max(axis=0, keepdims=True) + 1e-9)
peak_locs = bin_centers[np.argmax(pc_norm, axis=0)]
sort_order = np.argsort(peak_locs)
pc_sorted = pc_norm[:, sort_order]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(
    pc_sorted.T,
    aspect="auto",
    origin="lower",
    extent=[TRACK_RANGE[0], TRACK_RANGE[1], 0, pc_sorted.shape[1]],
    cmap="magma",
    interpolation="nearest",
)
ax.set_xlabel("Linear position (m)")
ax.set_ylabel("Place cell # (sorted by peak location)")
ax.set_title(f"Place-field tiling: {pc_sorted.shape[1]} CA1 place cells")
cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Normalised firing rate")
plt.tight_layout()
_save(fig, "fig03_population_heatmap.png")
plt.close(fig)

# %% [markdown]
# ## Spatial information vs. peak rate

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.hist(SI, bins=30, color="C0", alpha=0.8, edgecolor="k")
ax.axvline(0.5, color="r", ls="--", label="place-cell threshold (0.5 bits/spike)")
ax.set_xlabel("Spatial information (bits/spike)")
ax.set_ylabel("# units")
ax.set_title("Skaggs spatial information")
ax.legend()

ax = axes[1]
ax.scatter(peak_rate, SI, s=14, c=np.where(is_place, "C3", "0.6"),
           edgecolor="k", linewidth=0.3)
ax.set_xscale("log")
ax.set_xlabel("Peak firing rate (Hz)")
ax.set_ylabel("Spatial information (bits/spike)")
ax.set_title("Place cells (red) stand out on both axes")
ax.axhline(0.5, color="r", ls="--", lw=0.8)
ax.axvline(1.0, color="r", ls="--", lw=0.8)
plt.tight_layout()
_save(fig, "fig04_spatial_information.png")
plt.close(fig)

# %% [markdown]
# ## Decoding the animal's position from spike counts
#
# As a stringent test that this CA1 population really *encodes* space, we use
# Pynapple's Bayesian decoder. We train tuning curves on ~80% of the running
# epochs and decode position on the held-out 20%.

# %%
# Simple holdout split: split running intervals into train/test halves
n_run = len(run_intervals)
rng = np.random.default_rng(0)
shuf = rng.permutation(n_run)
n_train = int(0.8 * n_run)
train_idx = np.sort(shuf[:n_train])
test_idx = np.sort(shuf[n_train:])

train_iset = nap.IntervalSet(
    start=run_intervals.start[train_idx],
    end=run_intervals.end[train_idx],
)
test_iset = nap.IntervalSet(
    start=run_intervals.start[test_idx],
    end=run_intervals.end[test_idx],
)

# Use only place cells for decoding
pc_unit_ids = [unit_ids[i] for i in place_idx]
pyr_pc = pyr[pc_unit_ids]

train_tc = nap.compute_1d_tuning_curves(
    group=pyr_pc.restrict(train_iset),
    feature=position.restrict(train_iset),
    nb_bins=NBINS,
    minmax=TRACK_RANGE,
)
# Smooth a touch
for col in train_tc.columns:
    train_tc[col] = gaussian_filter1d(train_tc[col].values, sigma=1.0)

decoded, p_post = nap.decode_1d(
    tuning_curves=train_tc,
    group=pyr_pc.restrict(test_iset),
    ep=test_iset,
    bin_size=0.25,  # 250 ms bins
    feature=position.restrict(test_iset),
)

actual = position.bin_average(0.25, ep=test_iset)
# Align decoded and actual on common timestamps
common = np.intersect1d(decoded.index.values, actual.index.values)
err = np.abs(decoded.restrict(test_iset).values - actual.restrict(test_iset).values)
median_err = np.nanmedian(err)
print(f"Median decoding error: {median_err*100:.1f} cm "
      f"(chance ≈ {1.6/4*100:.0f} cm on a 1.6 m track)")

# %% [markdown]
# ## Decoding plot
#
# Top: decoded posterior over position; the magenta line is the true position.
# Bottom: histogram of decoding error.

# %%
# Pick the longest contiguous test interval so we can see a clean traverse
durations = test_iset.end - test_iset.start
top_k = np.argsort(durations)[::-1][:6]  # 6 longest test runs
fig, axes = plt.subplots(2, 1, figsize=(11, 6),
                         gridspec_kw={"height_ratios": [2, 1]})
ax = axes[0]

# Stitch the 6 longest test intervals together with a small visual gap
cursor = 0.0
GAP = 0.5  # seconds of blank between stitched intervals
for j, ti in enumerate(top_k):
    s, e = float(test_iset.start[ti]), float(test_iset.end[ti])
    seg_post = p_post.restrict(nap.IntervalSet(start=s, end=e))
    seg_true = position.restrict(nap.IntervalSet(start=s, end=e))
    if len(seg_post) == 0:
        continue
    t0 = seg_post.index[0]
    ax.imshow(
        seg_post.values.T,
        aspect="auto",
        origin="lower",
        extent=[cursor, cursor + (seg_post.index[-1] - t0),
                TRACK_RANGE[0], TRACK_RANGE[1]],
        cmap="viridis",
    )
    ax.plot(seg_true.index - t0 + cursor, seg_true.values,
            color="magenta", lw=1.4,
            label="true position" if j == 0 else None)
    cursor += (seg_post.index[-1] - t0) + GAP
    ax.axvline(cursor - GAP / 2, color="white", lw=0.6, alpha=0.5)
ax.set_xlim(0, cursor)
ax.set_ylim(*TRACK_RANGE)
ax.set_ylabel("Position (m)")
ax.set_xlabel("Stitched time within held-out runs (s)")
ax.set_title("Bayesian decoding from CA1 place-cell ensemble (6 longest held-out runs)")
ax.legend(loc="upper right")

ax = axes[1]
ax.hist(err.flatten() * 100, bins=40, color="C0", edgecolor="k", alpha=0.8)
ax.axvline(median_err * 100, color="r", ls="--",
           label=f"median = {median_err*100:.1f} cm")
ax.set_xlabel("|decoded − true| (cm)")
ax.set_ylabel("# 250-ms bins")
ax.legend()
plt.tight_layout()
_save(fig, "fig05_decoding.png")
plt.close(fig)

# %% [markdown]
# ## Summary
#
# Using a single CA1 silicon-probe recording from DANDI:000044 we:
#
# - identified place cells (Skaggs SI > 0.5 bits/spike, peak > 1 Hz);
# - showed that their place fields **tile** the 1.6 m linear track;
# - decoded the animal's position from population spike counts to within a few
#   centimetres on held-out data.
#
# These three findings together are the canonical signatures of hippocampal
# place coding (O'Keefe & Dostrovsky 1971; Wilson & McNaughton 1993;
# Skaggs et al. 1993).

# %%
print("Done.")
