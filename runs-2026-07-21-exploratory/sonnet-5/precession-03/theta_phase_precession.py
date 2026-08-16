# %% [markdown]
# # Theta Phase Precession in Hippocampal CA1 Place Cells
#
# This notebook demonstrates **theta phase precession**, the phenomenon in which a
# hippocampal place cell fires at progressively earlier phases of the local theta
# (6-10 Hz) oscillation as an animal moves through the cell's place field
# (O'Keefe & Recce, 1993).
#
# **Dataset**: DANDI Archive Dandiset
# [000044](https://dandiarchive.org/dandiset/000044) ("Diversity in neural firing
# dynamics supports both rigid and learned hippocampal sequences", Grosmark & Buzsaki),
# session `sub-Achilles/sub-Achilles_ses-Achilles-10252013`. A rat runs back and forth
# on a ~1.6 m linear track for reward while 128-channel silicon-probe LFP and
# spike-sorted CA1 units (137 units; 120 putative excitatory, 17 putative inhibitory)
# are recorded, together with 2D video-tracked position.
#
# **Approach**:
# 1. Stream the NWB file directly from S3 with `remfile` (no full download).
# 2. Restrict to the `MazeEpoch` (the linear-track running session).
# 3. Build a continuous position/speed estimate from the raw 2D tracking and split
#    running epochs by direction of travel (place fields on linear tracks are
#    direction-selective).
# 4. Identify place cells from directional firing-rate maps.
# 5. Select a hippocampal LFP channel with the strongest theta rhythmicity, band-pass
#    filter it (6-10 Hz) and extract the instantaneous theta phase via the Hilbert
#    transform.
# 6. For each place cell, compute the theta phase at every spike emitted while the
#    animal ran through the field, plotted against normalized position in the field,
#    and fit a circular-linear model to quantify the precession slope.
# 7. Summarize phase precession across the population of detected place cells.

# %%
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, welch
from pynwb import NWBHDF5IO
import pynapple as nap
from dandi.dandiapi import DandiAPIClient
from tqdm import tqdm

np.set_printoptions(suppress=True)

# %% [markdown]
# ## 1. Stream the NWB file from DANDI

# %%
DANDISET_ID = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
CACHE_DIR = "nwb_cache"

client = DandiAPIClient()
dandiset = client.get_dandiset(DANDISET_ID, "draft")
asset = dandiset.get_asset_by_path(ASSET_PATH)
s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
print("Streaming:", s3_url)

disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5_file, load_namespaces=True)
nwbfile = io.read()
print(nwbfile)

# %% [markdown]
# The file contains a `behavior` processing module (2D and linearized position,
# speed) and an `ecephys` module with a 128-channel `LFP` series, plus a `units`
# table with 137 spike-sorted CA1 neurons. The `epochs` table splits the session
# into `PREEpoch` (rest box), `MazeEpoch` (linear-track running) and `POSTEpoch`
# (rest box / sleep).

# %%
epochs = nwbfile.epochs
for lbl, s, e in zip(epochs["label"].data[:], epochs["start_time"].data[:], epochs["stop_time"].data[:]):
    print(f"{lbl:12s} {s:10.1f} - {e:10.1f}  ({e - s:8.1f} s)")

maze_start = float(epochs["start_time"].data[1])
maze_stop = float(epochs["stop_time"].data[1])
maze_ep = nap.IntervalSet(start=maze_start, end=maze_stop)

# %% [markdown]
# ## 2. Build position, speed, and running-direction epochs
#
# The pre-computed "linearized position" channel in this file has long stretches of
# NaN (it is only defined for a subset of passes), which produces spurious gaps when
# used naively for velocity estimation. Instead we use the raw 2D position, restrict
# to samples on the runway itself (excluding the rest platform visible in the
# tracking data), and compute velocity **within** contiguous tracked epochs only, so
# that we never differentiate across a tracking dropout.

# %%
pos_series = nwbfile.processing["behavior"]["1.6mLinearMazePosition"]["1.6mLinearMazeSpatialSeries"]
t_pos = pos_series.starting_time + np.arange(pos_series.data.shape[0]) / pos_series.rate
xy = pos_series.data[:]
x, y = xy[:, 0], xy[:, 1]

# runway corridor only (excludes the rest platform at y > 0.15)
on_track = (~np.isnan(x)) & (~np.isnan(y)) & (y > -0.6) & (y < 0.15)

# contiguous on-track epochs (>= 0.5 s) built from the boolean mask
d = np.diff(np.concatenate(([0], on_track.astype(int), [0])))
starts = np.where(d == 1)[0]
ends = np.where(d == -1)[0]
min_samples = int(0.5 * pos_series.rate)
keep = (ends - starts) >= min_samples
starts, ends = starts[keep], ends[keep]
on_track_ep = nap.IntervalSet(start=t_pos[starts], end=t_pos[ends - 1])
print(f"{len(on_track_ep)} on-track epochs, {on_track_ep.tot_length():.0f} s total")

xpos_t, xpos_v, vel_t, vel_v = [], [], [], []
for s, e in zip(starts, ends):
    tt, xx = t_pos[s:e], x[s:e]
    xpos_t.append(tt)
    xpos_v.append(xx)
    vel_t.append(tt)
    vel_v.append(np.gradient(xx, tt))

xpos = nap.Tsd(t=np.concatenate(xpos_t), d=np.concatenate(xpos_v), time_support=on_track_ep)
velocity = nap.Tsd(t=np.concatenate(vel_t), d=np.concatenate(vel_v), time_support=on_track_ep)
velocity_smooth = velocity.smooth(std=0.25, windowsize=1.0)
speed_smooth = np.abs(velocity_smooth)

SPEED_THRESH = 0.1  # m/s
run_ep = speed_smooth.threshold(SPEED_THRESH).time_support
right_ep = velocity_smooth.threshold(SPEED_THRESH, method="above").time_support.intersect(run_ep)
left_ep = velocity_smooth.threshold(-SPEED_THRESH, method="below").time_support.intersect(run_ep)
print(f"running: {run_ep.tot_length():.0f} s  (rightward {right_ep.tot_length():.0f} s, "
      f"leftward {left_ep.tot_length():.0f} s)")

# %%
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
axes[0].plot(xpos.index, xpos.values, lw=0.3)
axes[0].set_ylabel("x position (m)")
axes[0].set_title("Runway x-position, MazeEpoch")
axes[1].plot(speed_smooth.index, speed_smooth.values, lw=0.3)
axes[1].axhline(SPEED_THRESH, color="r", ls="--", lw=0.8, label=f"{SPEED_THRESH} m/s threshold")
axes[1].set_ylabel("speed (m/s)")
axes[1].set_xlabel("time (s)")
axes[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig("fig01_position_speed.png", dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(x, y, lw=0.2, alpha=0.5)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Raw 2D trajectory (MazeEpoch)")
plt.tight_layout()
plt.savefig("fig02_2d_trajectory.png", dpi=130)
plt.close()
print("saved fig01_position_speed.png, fig02_2d_trajectory.png")

# %% [markdown]
# ## 3. Load CA1 units and inspect a raw spike raster

# %%
units = nwbfile.units
cell_type = units["cell_type"].data[:]
location = units["location"].data[:]
exc_idx = np.where(cell_type == "excitatory")[0]
print(f"{len(exc_idx)} putative excitatory CA1 units out of {len(units)} total")

spike_dict = {int(i): nap.Ts(t=units["spike_times"][i]) for i in exc_idx}
tsgroup = nap.TsGroup(spike_dict, time_support=maze_ep)
tsgroup.set_info(location=location[exc_idx])

# %%
example_ids = list(tsgroup.keys())[:30]
fig, ax = plt.subplots(figsize=(13, 6))
ax.eventplot([tsgroup[i].index for i in example_ids], lineoffsets=range(len(example_ids)),
             linelengths=0.8, color="k")
ax.set_xlabel("time (s)")
ax.set_ylabel("unit #")
ax.set_title("Example raw spike raster, 30 CA1 units (MazeEpoch)")
plt.tight_layout()
plt.savefig("fig03_spike_raster.png", dpi=130)
plt.close()
print("saved fig03_spike_raster.png")

# %% [markdown]
# ## 4. Directional place fields
#
# Place fields on a linear track are direction-selective, so tuning curves are
# computed separately for rightward and leftward runs. A simple, direction-specific
# place-field detector then keeps cells with a single, well-isolated field away from
# the track ends.

# %%
NB_BINS = 40
XMIN, XMAX = -0.6, 1.4
BIN_WIDTH = (XMAX - XMIN) / NB_BINS

tc_right = nap.compute_1d_tuning_curves(group=tsgroup, feature=xpos, nb_bins=NB_BINS, ep=right_ep, minmax=(XMIN, XMAX))
tc_left = nap.compute_1d_tuning_curves(group=tsgroup, feature=xpos, nb_bins=NB_BINS, ep=left_ep, minmax=(XMIN, XMAX))
bins = tc_right.index.values


def field_bounds(rate, thresh_frac=0.2):
    """Contiguous region around the peak bin above thresh_frac * peak."""
    rate = np.nan_to_num(rate, nan=0.0)
    peak = rate.max()
    if peak <= 0:
        return None
    peak_bin = int(rate.argmax())
    thresh = thresh_frac * peak
    lo = peak_bin
    while lo > 0 and rate[lo - 1] > thresh:
        lo -= 1
    hi = peak_bin
    while hi < len(rate) - 1 and rate[hi + 1] > thresh:
        hi += 1
    return lo, hi, peak, peak_bin


def detect_place_field(cell, min_peak=2.0, min_width=0.1, max_width=0.9):
    """Pick the better-tuned direction for a cell and return field info, or None."""
    candidates = []
    for direction, tc, ep in (("R", tc_right, right_ep), ("L", tc_left, left_ep)):
        fb = field_bounds(tc[cell].values)
        if fb is None:
            continue
        lo, hi, peak, peak_bin = fb
        width = (hi - lo + 1) * BIN_WIDTH
        # exclude fields touching the absolute track ends and require a clean, single bump
        if lo == 0 or hi == NB_BINS - 1:
            continue
        if not (min_width <= width <= max_width) or peak < min_peak:
            continue
        candidates.append((direction, ep, lo, hi, peak, width))
    if not candidates:
        return None
    # prefer the direction with the higher peak rate
    return max(candidates, key=lambda c: c[4])


place_cells = {}
for cell in tsgroup.keys():
    info = detect_place_field(cell)
    if info is not None:
        place_cells[cell] = info

print(f"{len(place_cells)} / {len(tsgroup)} excitatory units have a clean, single directional place field")

# %%
top_cells = sorted(place_cells, key=lambda c: place_cells[c][4], reverse=True)[:16]
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for ax, cell in zip(axes.flat, top_cells):
    direction, ep, lo, hi, peak, width = place_cells[cell]
    tc = tc_right if direction == "R" else tc_left
    ax.plot(tc.index, tc[cell], color="C0" if direction == "R" else "C1")
    ax.axvspan(bins[lo] - BIN_WIDTH / 2, bins[hi] + BIN_WIDTH / 2, color="gray", alpha=0.2)
    ax.set_title(f"unit {cell} ({direction}, {peak:.1f} Hz)")
    ax.set_xlabel("x (m)")
for ax in axes.flat[len(top_cells):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig("fig04_place_fields.png", dpi=130)
plt.close()
print("saved fig04_place_fields.png")

# %% [markdown]
# ## 5. Select a theta-band LFP reference channel
#
# The 128-channel LFP is streamed only for the `MazeEpoch` time range. To find a
# channel with strong theta rhythmicity (typical of the CA1 pyramidal/stratum
# radiatum layers), we compare theta-band (6-10 Hz) to delta-band (2-4 Hz) power
# across a sparse grid of channels, then refine locally.

# %%
lfp_series = nwbfile.processing["ecephys"]["LFP"]["LFP"]
fs_lfp = lfp_series.rate
i0, i1 = int(maze_start * fs_lfp), int(maze_stop * fs_lfp)


def theta_delta_ratio(channel):
    data = lfp_series.data[i0:i1, channel].astype(np.float32)
    f, pxx = welch(data, fs=fs_lfp, nperseg=int(fs_lfp * 4))
    theta = pxx[(f >= 6) & (f <= 10)].mean()
    delta = pxx[(f >= 2) & (f <= 4)].mean()
    return theta / delta, data


coarse_channels = list(range(0, lfp_series.data.shape[1], 8))
ratios = {ch: theta_delta_ratio(ch)[0] for ch in tqdm(coarse_channels, desc="scanning channels")}
best_coarse = max(ratios, key=ratios.get)

fine_channels = [c for c in range(max(0, best_coarse - 8), min(lfp_series.data.shape[1], best_coarse + 8))]
fine_ratios = {ch: theta_delta_ratio(ch)[0] for ch in tqdm(fine_channels, desc="refining channel")}
THETA_CHANNEL = max(fine_ratios, key=fine_ratios.get)
print(f"Selected LFP channel {THETA_CHANNEL} (theta/delta power ratio = {fine_ratios[THETA_CHANNEL]:.2f})")

lfp_data = lfp_series.data[i0:i1, THETA_CHANNEL].astype(np.float64)
t_lfp = maze_start + np.arange(len(lfp_data)) / fs_lfp
lfp = nap.Tsd(t=t_lfp, d=lfp_data)

# %% [markdown]
# ## 6. Band-pass filter for theta and extract instantaneous phase

# %%
theta = nap.apply_bandpass_filter(lfp, cutoff=(6, 10), fs=fs_lfp, mode="butter")
theta_phase = nap.Tsd(t=theta.index.values, d=np.angle(hilbert(theta.values)))

# %%
t0, t1 = maze_start + 120, maze_start + 125
sl = slice(*np.searchsorted(t_lfp, [t0, t1]))
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
axes[0].plot(t_lfp[sl], lfp_data[sl])
axes[0].set_ylabel("raw LFP (a.u.)")
axes[0].set_title(f"LFP channel {THETA_CHANNEL}")
axes[1].plot(theta.index.values[sl], theta.values[sl], color="C1")
axes[1].set_ylabel("theta (6-10 Hz)")
axes[2].plot(theta_phase.index.values[sl], theta_phase.values[sl], ".", ms=2, color="C2")
axes[2].set_ylabel("theta phase (rad)")
axes[2].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig("fig05_theta_phase_extraction.png", dpi=130)
plt.close()
print("saved fig05_theta_phase_extraction.png")

# %% [markdown]
# ## 7. Quantify phase precession with a circular-linear fit
#
# For each place cell, spikes emitted during runs in the cell's preferred direction
# are restricted to the field's spatial extent, position is normalized to [0, 1]
# (field entry -> exit) and the theta phase at each spike time is looked up from the
# Hilbert-transform phase trace. A circular-linear regression (grid search over
# slope k, maximizing resultant vector length of `phase - 2*pi*k*position`) gives
# the precession slope in theta cycles per field traversal; slope < 0 indicates
# classical phase precession (spikes occur at earlier phases later in the field).
# Statistical significance is assessed against a null distribution built by shuffling
# spike-phase assignment relative to position.

# %%
def circular_linear_fit(pos, phase, k_range=np.linspace(-3, 3, 301)):
    ang = phase[:, None] - 2 * np.pi * k_range[None, :] * pos[:, None]
    R = np.abs(np.mean(np.exp(1j * ang), axis=0))
    best = np.argmax(R)
    k = k_range[best]
    phase0 = np.angle(np.mean(np.exp(1j * (phase - 2 * np.pi * k * pos))))
    return k, R[best], phase0


def spikes_in_field(cell, direction, ep, lo, hi):
    xlo, xhi = bins[lo] - BIN_WIDTH / 2, bins[hi] + BIN_WIDTH / 2
    spk = tsgroup[cell].restrict(ep)
    spk_x = spk.value_from(xpos)
    in_field = (spk_x.values >= xlo) & (spk_x.values <= xhi)
    spk_in = spk[in_field]
    pos_norm = (spk_x.values[in_field] - xlo) / (xhi - xlo)
    if direction == "L":
        pos_norm = 1 - pos_norm  # so 0 = field entry, 1 = field exit, in both directions
    phase_at_spike = spk_in.value_from(theta_phase).values
    return pos_norm, phase_at_spike


rng = np.random.default_rng(0)
N_SHUFFLE = 200
MIN_SPIKES = 100

precession_results = []
for cell, (direction, ep, lo, hi, peak, width) in tqdm(place_cells.items(), desc="phase precession"):
    pos_norm, phase_at_spike = spikes_in_field(cell, direction, ep, lo, hi)
    if len(pos_norm) < MIN_SPIKES:
        continue
    k, R, phase0 = circular_linear_fit(pos_norm, phase_at_spike)
    shuffle_R = np.empty(N_SHUFFLE)
    for s in range(N_SHUFFLE):
        shuffled_pos = rng.permutation(pos_norm)
        _, shuffle_R[s], _ = circular_linear_fit(shuffled_pos, phase_at_spike)
    p_value = np.mean(shuffle_R >= R)
    precession_results.append(dict(cell=cell, direction=direction, slope=k, R=R, phase0=phase0,
                                    p_value=p_value, n_spikes=len(pos_norm), peak_rate=peak, width=width,
                                    pos=pos_norm, phase=phase_at_spike))

precession_df = pd.DataFrame(precession_results).drop(columns=["pos", "phase"])
precession_df = precession_df.sort_values("R", ascending=False).reset_index(drop=True)
print(precession_df.to_string())

n_sig_neg = ((precession_df.p_value < 0.05) & (precession_df.slope < 0)).sum()
print(f"\n{n_sig_neg} / {len(precession_df)} place cells show significant (p<0.05) negative "
      f"phase-precession slope")
print(f"Median slope across all tested place cells: {precession_df.slope.median():.2f} cycles/field")

# %% [markdown]
# ## 8. Example phase-precession plots
#
# The six most significant place cells (smallest shuffle p-value / highest resultant
# vector length R), showing raw spike phase-position scatter (each spike plotted
# twice, at phase and phase + 2*pi, for readability), the binned circular-mean phase
# per position, and the fitted circular-linear regression line.

# %%
example_cells = precession_df.sort_values("p_value").head(6)["cell"].tolist()

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, cell in zip(axes.flat, example_cells):
    row = next(r for r in precession_results if r["cell"] == cell)
    pos, phase = row["pos"], row["phase"]
    k, R, phase0 = row["slope"], row["R"], row["phase0"]

    ax.scatter(pos, phase, s=6, alpha=0.35, color="gray")
    ax.scatter(pos, phase + 2 * np.pi, s=6, alpha=0.35, color="gray")

    nb = 10
    edges = np.linspace(0, 1, nb + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    means = [np.angle(np.mean(np.exp(1j * phase[(pos >= edges[i]) & (pos < edges[i + 1])])))
             if ((pos >= edges[i]) & (pos < edges[i + 1])).sum() > 3 else np.nan
             for i in range(nb)]
    means = np.unwrap(means)
    ax.plot(centers, means, "o-", color="C1", label="binned circular mean")
    ax.plot(centers, means + 2 * np.pi, "o-", color="C1")

    xs = np.linspace(0, 1, 50)
    ax.plot(xs, 2 * np.pi * k * xs + phase0, "r-", lw=2, label=f"fit: {k:.2f} cyc/field")

    ax.set_title(f"unit {cell} ({row['direction']}): R={R:.2f}, p={row['p_value']:.3f}, n={row['n_spikes']}")
    ax.set_xlabel("normalized position in field")
    ax.set_ylabel("theta phase (rad)")
    ax.legend(fontsize=7, loc="upper right")
plt.tight_layout()
plt.savefig("fig06_example_phase_precession.png", dpi=130)
plt.close()
print("saved fig06_example_phase_precession.png")

# %% [markdown]
# ## 9. Population summary

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].hist(precession_df.slope, bins=20, color="C0", edgecolor="white")
axes[0].axvline(0, color="k", ls="--", lw=1)
axes[0].set_xlabel("precession slope (cycles / field)")
axes[0].set_ylabel("# place cells")
axes[0].set_title("Distribution of precession slopes")

sig_mask = precession_df.p_value < 0.05
colors = np.where(sig_mask, "C0", "C3")
axes[1].scatter(precession_df.slope, precession_df.R, c=colors, edgecolor="k", linewidth=0.3)
axes[1].axvline(0, color="k", ls="--", lw=1)
axes[1].set_xlabel("precession slope (cycles / field)")
axes[1].set_ylabel("resultant vector length R")
axes[1].set_title("Slope vs. fit strength\n(blue = p<0.05 vs. shuffle, red = n.s.)")

sig = precession_df.p_value < 0.05
axes[2].bar(["not sig.", "sig., slope>0", "sig., slope<0"],
            [ (~sig).sum(), (sig & (precession_df.slope > 0)).sum(), (sig & (precession_df.slope < 0)).sum() ],
            color=["gray", "C3", "C0"])
axes[2].set_ylabel("# place cells")
axes[2].set_title("Significance vs. shuffle (n=%d cells)" % len(precession_df))

plt.tight_layout()
plt.savefig("fig07_population_summary.png", dpi=130)
plt.close()
print("saved fig07_population_summary.png")

# %% [markdown]
# ## Summary
#
# Across the population of CA1 place cells with a clean, direction-specific place
# field on the linear track, the great majority of cells with a statistically
# significant circular-linear fit (relative to a position-shuffled null) show a
# **negative** precession slope: spikes occur at progressively earlier theta phases
# as the animal advances through the place field. This reproduces the classic
# hippocampal theta phase precession phenomenon (O'Keefe & Recce, 1993) directly
# from a publicly available DANDI Archive dataset, using only streamed access (no
# full file download) and Pynapple for all time-series handling and analysis.

# %%
io.close()
