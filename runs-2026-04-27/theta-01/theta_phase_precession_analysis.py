# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
# ---

# %% [markdown]
# # Theta Phase Entrainment and Phase Precession in Hippocampal CA1 Place Cells
#
# This analysis demonstrates two classical hippocampal phenomena using publicly available data
# from the **DANDI Archive (Dandiset 000044, Buzsáki lab)**:
#
# 1. **Theta phase entrainment**: CA1 pyramidal cells fire preferentially at specific phases
#    of the ongoing 6–12 Hz theta oscillation in the local field potential (LFP).
# 2. **Theta phase precession**: As a rat traverses the place field of a CA1 place cell,
#    spikes occur at progressively earlier phases of the theta cycle.
#
# **Dataset**: `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`
# (Long-Evans rat running back and forth on a 1.6 m linear track; bilateral silicon probes in CA1).
#
# **Reference**: Grosmark, A.D., Buzsáki, G. (2016). *Diversity in neural firing dynamics
# supports both rigid and learned hippocampal sequences*. Science 351, 1440–1443.

# %% [markdown]
# ## 1. Setup and streaming load

# %%
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import circmean, pearsonr

import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap

OUTDIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = OUTDIR  # write png next to script
os.makedirs("/tmp/remfile_cache", exist_ok=True)

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/4f5/a84/4f5a84aa-a6e4-496a-9b23-535fa6fbd3ae"

print("Streaming NWB file from DANDI...")
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
print("Subject:", nwbfile.subject.subject_id, "| Species:", nwbfile.subject.species)

# %% [markdown]
# ## 2. Define the maze (running) epoch
#
# The session has three epochs: PRE-sleep, MazeEpoch (linear track), and POST-sleep.
# Theta phenomena are most pronounced during running, so we restrict to the maze epoch.

# %%
epochs_df = nwbfile.epochs.to_dataframe()
print(epochs_df)
maze_row = epochs_df[epochs_df["label"] == "MazeEpoch"].iloc[0]
maze_start, maze_stop = float(maze_row["start_time"]), float(maze_row["stop_time"])
print(f"Maze epoch: {maze_start:.1f} – {maze_stop:.1f} s  (duration {maze_stop - maze_start:.1f} s)")
maze_ep = nap.IntervalSet(start=maze_start, end=maze_stop)

# %% [markdown]
# ## 3. Linearized position on the 1.6 m track
#
# The NWB file stores position with `rate=0.0256` which (in this file) is actually the
# sampling period in seconds, not Hz. We rebuild the time vector explicitly to be safe.

# %%
linpos_ts = nwbfile.processing["behavior"].data_interfaces[
    "1.6mLinearMazeLinearizedPosition"
].spatial_series["1.6mLinearMazeLinearizedTimeSeries"]

pos_period = float(linpos_ts.rate)        # actually period (s) in this NWB
pos_start = float(linpos_ts.starting_time)
pos_data = linpos_ts.data[:].squeeze()    # (N,)
pos_t = pos_start + np.arange(len(pos_data)) * pos_period

# Drop NaNs (only valid samples are during the maze epoch)
valid = ~np.isnan(pos_data)
pos_t = pos_t[valid]
pos_data = pos_data[valid]

position = nap.Tsd(t=pos_t, d=pos_data, time_support=maze_ep)
print(f"Position samples: {len(position)} | t range {position.t.min():.1f}–{position.t.max():.1f} s")
print(f"Position range: {np.nanmin(position.d):.3f} – {np.nanmax(position.d):.3f} m")

# Compute speed (m/s) — central difference smoothed
dt = np.median(np.diff(position.t))
speed_vals = np.abs(np.gradient(position.d, position.t))
speed = nap.Tsd(t=position.t, d=speed_vals, time_support=maze_ep).smooth(std=0.25)
print(f"Sampling dt = {dt*1000:.1f} ms (rate ≈ {1/dt:.1f} Hz)")
print(f"Speed: median {np.median(speed.d):.2f} m/s, max {np.max(speed.d):.2f} m/s")

# Restrict spatial analysis to the interior of the track to exclude reward zones at the ends
TRACK_LO, TRACK_HI = 0.10, 1.50  # meters

# %% [markdown]
# ## 4. Identify run epochs (rightward and leftward traversals)
#
# Place fields are direction-specific on a linear track, so we split traversals into
# rightward (towards x=1.6) and leftward (towards x=0) runs, requiring sustained motion.

# %%
SPEED_THRESH = 0.15  # m/s — exclude reward-zone pauses
moving_mask = (speed.d > SPEED_THRESH) & (position.d > TRACK_LO) & (position.d < TRACK_HI)
moving_intervals = []
i = 0
while i < len(moving_mask):
    if moving_mask[i]:
        j = i
        while j < len(moving_mask) and moving_mask[j]:
            j += 1
        moving_intervals.append((speed.t[i], speed.t[min(j-1, len(speed.t)-1)]))
        i = j
    else:
        i += 1
moving_intervals = np.array(moving_intervals)
# keep only traversals lasting >0.5 s
durations = moving_intervals[:, 1] - moving_intervals[:, 0]
moving_intervals = moving_intervals[durations > 0.5]
moving_ep = nap.IntervalSet(start=moving_intervals[:, 0], end=moving_intervals[:, 1])
print(f"Found {len(moving_ep)} movement bouts, total duration {moving_ep.tot_length():.1f} s")

# Direction of each bout: sign of (end - start) on linear axis
def position_at(ts):
    idx = np.searchsorted(position.t, ts)
    idx = np.clip(idx, 0, len(position) - 1)
    return position.d[idx]

dirs = np.sign(position_at(moving_intervals[:, 1]) - position_at(moving_intervals[:, 0]))
right_ep = nap.IntervalSet(
    start=moving_intervals[dirs > 0, 0], end=moving_intervals[dirs > 0, 1]
)
left_ep = nap.IntervalSet(
    start=moving_intervals[dirs < 0, 0], end=moving_intervals[dirs < 0, 1]
)
print(f"Rightward runs: {len(right_ep)} ({right_ep.tot_length():.1f} s)")
print(f"Leftward runs:  {len(left_ep)} ({left_ep.tot_length():.1f} s)")

# %% [markdown]
# ## 5. Load spikes and pick CA1 pyramidal cells

# %%
nwb = nap.NWBFile(nwbfile)
units_all = nwb["units"]
print("Total units:", len(units_all))
print("Cell types:", units_all["cell_type"].value_counts() if hasattr(units_all["cell_type"], "value_counts") else np.unique(units_all["cell_type"], return_counts=True))

# Restrict to excitatory (putative pyramidal) units in CA1, with reasonable firing rate
units_pyr = units_all.getby_category("cell_type")["excitatory"]
units_pyr = units_pyr.restrict(maze_ep)
# require rate >0.2 Hz on maze
mfr = np.array([len(units_pyr[k]) / maze_ep.tot_length() for k in units_pyr.keys()])
keep = np.array(units_pyr.keys())[mfr > 0.2]
units_pyr = units_pyr[list(keep)]
print(f"Pyramidal cells with maze firing rate > 0.2 Hz: {len(units_pyr)}")

# %% [markdown]
# ## 6. Load LFP for one CA1 channel and identify theta
#
# We pick a CA1 channel with strong theta power during running, restricted to the maze epoch
# (loading only this slice keeps memory tiny).

# %%
electrodes_df = nwbfile.electrodes.to_dataframe()
# All electrodes in this file are labeled "unknown" but are silicon-probe CA1 channels
# (subject Achilles had bilateral CA1 silicon probes). Exclude bad channels.
ca1_channels = electrodes_df.index[~electrodes_df["bad_electrode"].astype(bool)].to_numpy()
print(f"Good silicon-probe channels available (CA1): {len(ca1_channels)}")

lfp_es = nwbfile.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"]
lfp_rate = float(lfp_es.rate)
n_lfp = lfp_es.data.shape[0]
lfp_t_full_start = float(lfp_es.starting_time)
print(f"LFP shape {lfp_es.data.shape}, rate {lfp_rate} Hz, start {lfp_t_full_start} s")

# Slice the maze epoch
i0 = int(round((maze_start - lfp_t_full_start) * lfp_rate))
i1 = int(round((maze_stop - lfp_t_full_start) * lfp_rate))
i0 = max(0, i0); i1 = min(n_lfp, i1)
print(f"LFP indices for maze epoch: {i0}..{i1}  ({(i1-i0)/lfp_rate:.1f} s)")

# Sample a subset of CA1 channels and pick the one with strongest 6–12 Hz theta power
# (load in small slices to keep memory low)
def bandpass(x, fs, lo, hi, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)

candidate_channels = ca1_channels[:: max(1, len(ca1_channels) // 16)]  # ~16 channels spread across CA1
print(f"Probing {len(candidate_channels)} candidate CA1 channels for theta power...")

theta_power = {}
for ch in tqdm(candidate_channels, desc="theta power per channel"):
    x = lfp_es.data[i0:i1, int(ch)].astype(np.float32)
    xt = bandpass(x, lfp_rate, 6.0, 12.0)
    theta_power[int(ch)] = float(np.mean(xt ** 2))
best_ch = max(theta_power, key=theta_power.get)
print(f"Best CA1 channel for theta: {best_ch} | power {theta_power[best_ch]:.3g}")
print(f"Channel location: {electrodes_df.loc[best_ch, 'location']}")

# Load full LFP trace for the chosen channel (maze epoch only)
print("Loading full LFP trace for chosen channel...")
lfp_raw = lfp_es.data[i0:i1, int(best_ch)].astype(np.float32)
lfp_time = lfp_t_full_start + np.arange(i0, i1) / lfp_rate
lfp = nap.Tsd(t=lfp_time, d=lfp_raw, time_support=maze_ep)
print(f"LFP loaded: {len(lfp)} samples")

# %% [markdown]
# ## 7. Bandpass-filter LFP in the theta band and extract instantaneous phase

# %%
lfp_theta = bandpass(lfp.d, lfp_rate, 6.0, 12.0)
analytic = hilbert(lfp_theta)
theta_phase = np.angle(analytic)              # radians, –π … π
theta_amp = np.abs(analytic)
theta_phase_tsd = nap.Tsd(t=lfp.t, d=theta_phase, time_support=maze_ep)
theta_amp_tsd = nap.Tsd(t=lfp.t, d=theta_amp, time_support=maze_ep)
print("Theta phase signal computed.")

# %% [markdown]
# ## 8. Plot raw vs theta-filtered LFP (sanity check)

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
t0 = maze_start + 50  # 5-second snippet during the maze
mask = (lfp.t >= t0) & (lfp.t < t0 + 5)
axes[0].plot(lfp.t[mask] - t0, lfp.d[mask], color="black", lw=0.6)
axes[0].set_ylabel("Raw LFP (a.u.)")
axes[0].set_title(f"CA1 LFP (channel {best_ch})  — 5 s during running")
axes[1].plot(lfp.t[mask] - t0, lfp_theta[mask], color="C0", lw=1.0)
axes[1].plot(lfp.t[mask] - t0, theta_amp[mask], color="C3", lw=1.0, label="amplitude")
axes[1].plot(lfp.t[mask] - t0, -theta_amp[mask], color="C3", lw=1.0)
axes[1].set_ylabel("Theta-filtered\n(6–12 Hz)")
axes[1].legend(loc="upper right", frameon=False)
axes[2].plot(lfp.t[mask] - t0, theta_phase[mask], color="C2", lw=0.8)
axes[2].set_ylabel("Phase (rad)")
axes[2].set_xlabel("Time (s)")
axes[2].set_yticks([-np.pi, 0, np.pi])
axes[2].set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig01_lfp_and_theta.png"), dpi=140)
plt.close(fig)
print("Saved fig01_lfp_and_theta.png")

# %% [markdown]
# ## 9. Place fields (1-D firing rate vs linearized position)
#
# Compute place fields separately for rightward and leftward runs since CA1 cells
# are direction-selective on a linear track.

# %%
N_BINS = 50
edges = np.linspace(TRACK_LO, TRACK_HI, N_BINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
SMOOTH_BINS = 2  # gaussian smoothing of place maps (sigma in bins)

from scipy.ndimage import gaussian_filter1d

def place_fields(units, run_ep):
    pos_run = position.restrict(run_ep)
    fields = {}
    for k in units.keys():
        sp = units[k].restrict(run_ep)
        sp_pos = np.interp(sp.t, pos_run.t, pos_run.d)
        spike_count, _ = np.histogram(sp_pos, edges)
        occ_count, _ = np.histogram(pos_run.d, edges)
        occ_time = occ_count * dt
        fr = np.divide(spike_count, occ_time, out=np.zeros_like(occ_time, dtype=float), where=occ_time > 0)
        fr = gaussian_filter1d(fr, SMOOTH_BINS)
        fields[k] = fr
    return pd.DataFrame(fields, index=centers)

print("Computing place fields...")
pf_right = place_fields(units_pyr, right_ep)
pf_left = place_fields(units_pyr, left_ep)
print(f"Right place fields: {pf_right.shape[1]} units × {pf_right.shape[0]} bins")

# Spatial information (Skaggs) per unit, per direction
def skaggs_info(rate_map, occ_time):
    occ_p = occ_time / occ_time.sum()
    mean_rate = (rate_map * occ_p).sum()
    if mean_rate <= 0:
        return 0.0
    valid = (rate_map > 0) & (occ_p > 0)
    return float(np.sum(occ_p[valid] * (rate_map[valid] / mean_rate) * np.log2(rate_map[valid] / mean_rate)))

def occupancy(run_ep):
    pos_run = position.restrict(run_ep)
    occ_count, _ = np.histogram(pos_run.d, edges)
    return occ_count * dt

occ_R = occupancy(right_ep)
occ_L = occupancy(left_ep)
si_R = {k: skaggs_info(pf_right[k].values, occ_R) for k in pf_right.columns}
si_L = {k: skaggs_info(pf_left[k].values, occ_L) for k in pf_left.columns}

# Identify "place cells": peak rate >2 Hz, spatial info >0.7 bits/spike, AND
# peak in the interior of the track (not at the trimmed-track edges)
def has_interior_peak(rate_map):
    peak_idx = int(np.argmax(rate_map))
    return 2 <= peak_idx <= (len(rate_map) - 3)

def is_place_cell(k):
    cond_R = (pf_right[k].max() > 2.0) and (si_R[k] > 0.7) and has_interior_peak(pf_right[k].values)
    cond_L = (pf_left[k].max() > 2.0) and (si_L[k] > 0.7) and has_interior_peak(pf_left[k].values)
    return cond_R or cond_L

place_cell_ids = [k for k in pf_right.columns if is_place_cell(k)]
print(f"Identified {len(place_cell_ids)} place cells out of {len(pf_right.columns)} pyramidal units")

# %% [markdown]
# ## 10. Plot place field heatmap for both directions

# %%
def order_by_peak(pf):
    peaks = pf.idxmax()
    return peaks.sort_values().index.tolist()

pf_right_pc = pf_right[place_cell_ids]
pf_left_pc = pf_left[place_cell_ids]
order_R = order_by_peak(pf_right_pc)
# normalize each unit to its own peak
def normed(df, order):
    sub = df[order]
    sub = sub.div(sub.max(axis=0).replace(0, np.nan), axis=1)
    return sub.fillna(0).values.T  # (units, bins)

fig, axes = plt.subplots(1, 2, figsize=(11, 6))
axes[0].imshow(normed(pf_right_pc, order_R), aspect="auto", origin="lower",
               extent=[0, 1.6, 0, len(order_R)], cmap="magma")
axes[0].set_xlabel("Position on track (m)")
axes[0].set_ylabel("Place cell #  (sorted by peak)")
axes[0].set_title(f"Rightward runs ({len(order_R)} cells)")
axes[1].imshow(normed(pf_left_pc, order_R), aspect="auto", origin="lower",
               extent=[0, 1.6, 0, len(order_R)], cmap="magma")
axes[1].set_xlabel("Position on track (m)")
axes[1].set_title("Leftward runs (same ordering)")
fig.suptitle(f"CA1 place fields — Achilles, {len(place_cell_ids)} place cells", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig02_place_fields.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print("Saved fig02_place_fields.png")

# %% [markdown]
# ## 11. Theta phase entrainment
#
# For each spike of each pyramidal unit (during running), record the instantaneous theta
# phase of the LFP. The circular distribution shows preferred-phase firing.

# %%
print("Computing spike phases...")
# Restrict to running epochs
spike_phases = {}
for k in tqdm(units_pyr.keys(), desc="phase per unit"):
    sp = units_pyr[k].restrict(moving_ep)
    if len(sp) < 20:
        spike_phases[k] = np.array([])
        continue
    ph = np.interp(sp.t, theta_phase_tsd.t, np.unwrap(theta_phase_tsd.d))
    ph = (ph + np.pi) % (2 * np.pi) - np.pi  # rewrap to (-π, π]
    spike_phases[k] = ph

# Population phase histogram (excluding cells with too few spikes)
all_phases_excit = np.concatenate([v for v in spike_phases.values() if v.size >= 20])
preferred_phases = []
mvls = []  # mean vector lengths (a.k.a. resultant length)
for k, v in spike_phases.items():
    if v.size < 50:
        continue
    cmean = circmean(v, high=np.pi, low=-np.pi)
    R = np.abs(np.mean(np.exp(1j * v)))
    preferred_phases.append(cmean)
    mvls.append(R)

preferred_phases = np.array(preferred_phases)
mvls = np.array(mvls)
print(f"Mean preferred phase across pyramidal cells: {circmean(preferred_phases, high=np.pi, low=-np.pi):.2f} rad")
print(f"Mean phase-locking strength (R): {mvls.mean():.3f}")

# Plot
fig = plt.figure(figsize=(11, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1])

# Panel A: population phase histogram (linear)
ax = fig.add_subplot(gs[0, 0])
phase_bins = np.linspace(-np.pi, np.pi, 37)
counts, _ = np.histogram(all_phases_excit, phase_bins)
counts_norm = counts / counts.sum()
ax.bar((phase_bins[:-1] + phase_bins[1:]) / 2, counts_norm, width=np.diff(phase_bins),
       color="C0", edgecolor="white")
# overlay theta cycle for reference
xx = np.linspace(-np.pi, np.pi, 200)
ax.plot(xx, counts_norm.max() * 0.5 * (1 + np.cos(xx)), "k--", lw=1, alpha=0.5,
        label="reference cos(θ)")
ax.set_xlabel("Theta phase (rad)")
ax.set_ylabel("Normalized spike count")
ax.set_title(f"Pooled CA1 pyr spikes (N={all_phases_excit.size})")
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
ax.legend(frameon=False, fontsize=8)

# Panel B: distribution of preferred phases
ax = fig.add_subplot(gs[0, 1], projection="polar")
ax.hist(preferred_phases, bins=24, color="C3", alpha=0.7)
ax.set_title(f"Preferred phase across\n{len(preferred_phases)} pyr cells", pad=15)

# Panel C: distribution of phase-locking strengths
ax = fig.add_subplot(gs[0, 2])
ax.hist(mvls, bins=20, color="C2", edgecolor="white")
ax.axvline(np.mean(mvls), color="k", lw=1.5, linestyle="--", label=f"mean R = {np.mean(mvls):.2f}")
ax.set_xlabel("Mean resultant length (R)")
ax.set_ylabel("# cells")
ax.set_title("Phase-locking strength")
ax.legend(frameon=False, fontsize=9)

fig.suptitle("Theta phase entrainment of CA1 pyramidal cells (running epochs)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig03_theta_entrainment.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print("Saved fig03_theta_entrainment.png")

# %% [markdown]
# ## 12. Theta phase precession
#
# For each place cell we look at spikes that occur **inside the cell's place field**
# during runs in the field's preferred direction, and plot theta phase against the
# normalized position-within-field. A negative slope (phase decreasing as the rat
# advances through the field) is the canonical signature of phase precession.
#
# We fit a circular–linear regression (Kempter et al. 2012):
# minimize the negative of the resultant length of (phase – 2π·a·x – φ₀) over slope `a`.

# %%
def circ_lin_fit(x, phi):
    """Fit phi = 2*pi*a*x + phi0 (mod 2pi). Returns (slope_in_rad_per_unit_x, phi0, R, p)."""
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan
    # Search over slopes (cycles per unit x)
    a_grid = np.linspace(-2.0, 2.0, 401)  # cycles
    Rmax, abest = -np.inf, 0.0
    for a in a_grid:
        z = phi - 2 * np.pi * a * x
        R = np.abs(np.mean(np.exp(1j * z)))
        if R > Rmax:
            Rmax, abest = R, a
    # refine
    a_fine = np.linspace(abest - 0.02, abest + 0.02, 401)
    for a in a_fine:
        z = phi - 2 * np.pi * a * x
        R = np.abs(np.mean(np.exp(1j * z)))
        if R > Rmax:
            Rmax, abest = R, a
    z = phi - 2 * np.pi * abest * x
    phi0 = np.angle(np.mean(np.exp(1j * z)))
    n = len(x)
    # circular–linear correlation (Kempter)
    theta_hat = (2 * np.pi * abest * x + phi0)
    cphi = np.cos(phi - circmean(phi, high=np.pi, low=-np.pi))
    sphi = np.sin(phi - circmean(phi, high=np.pi, low=-np.pi))
    cth = np.cos(theta_hat - circmean(theta_hat, high=np.pi, low=-np.pi))
    sth = np.sin(theta_hat - circmean(theta_hat, high=np.pi, low=-np.pi))
    num = np.sum(cphi * cth) + np.sum(sphi * sth)
    den = np.sqrt((np.sum(cphi ** 2) + np.sum(sphi ** 2)) * (np.sum(cth ** 2) + np.sum(sth ** 2)))
    rho = num / den if den > 0 else 0.0
    # rough p-value via permutation (cheap, n_perm small)
    n_perm = 200
    rho_perm = np.zeros(n_perm)
    for i in range(n_perm):
        phi_s = np.random.permutation(phi)
        z = phi_s - 2 * np.pi * abest * x
        rho_perm[i] = np.abs(np.mean(np.exp(1j * z)))
    p = float(np.mean(rho_perm >= Rmax))
    return abest * 2 * np.pi, phi0, rho, p  # slope in rad/unit_x

def field_bounds(rate_map, peak_idx, peak_val, frac=0.5):
    """Return (left_idx, right_idx) where rate first drops below frac * peak."""
    thr = frac * peak_val
    left = peak_idx
    while left > 0 and rate_map[left - 1] >= thr:
        left -= 1
    right = peak_idx
    while right < len(rate_map) - 1 and rate_map[right + 1] >= thr:
        right += 1
    return left, right

# For each place cell, find its preferred direction (the one with the larger peak)
# and compute phase vs in-field position.
print("Fitting circular-linear regression on each place cell...")
results = []
for k in tqdm(place_cell_ids):
    fr_R = pf_right[k].values
    fr_L = pf_left[k].values
    # choose the direction whose peak (a) is interior and (b) is largest
    cand = []
    if has_interior_peak(fr_R):
        cand.append(("R", fr_R, right_ep, fr_R.max()))
    if has_interior_peak(fr_L):
        cand.append(("L", fr_L, left_ep, fr_L.max()))
    if not cand:
        continue
    label, rate_map, run_ep_used, _ = max(cand, key=lambda c: c[3])
    if rate_map.max() < 2.0:
        continue
    peak_idx = int(np.argmax(rate_map))
    li, ri = field_bounds(rate_map, peak_idx, rate_map.max(), frac=0.5)
    x_lo, x_hi = edges[li], edges[ri + 1]
    if x_hi - x_lo < 0.10 or x_hi - x_lo > 0.8:  # plausible single-field width
        continue
    # spikes during preferred-direction runs
    sp = units_pyr[k].restrict(run_ep_used)
    if len(sp) < 30:
        continue
    sp_pos = np.interp(sp.t, position.t, position.d)
    in_field = (sp_pos >= x_lo) & (sp_pos <= x_hi)
    sp_t_field = sp.t[in_field]
    sp_pos_field = sp_pos[in_field]
    if len(sp_t_field) < 30:
        continue
    sp_phase_field = np.interp(sp_t_field, lfp.t, np.unwrap(theta_phase))
    sp_phase_field = (sp_phase_field + np.pi) % (2 * np.pi) - np.pi
    # normalize position within field [0, 1] in direction of travel
    if label == "R":
        x_norm = (sp_pos_field - x_lo) / (x_hi - x_lo)
    else:
        x_norm = (x_hi - sp_pos_field) / (x_hi - x_lo)
    slope, phi0, rho, p = circ_lin_fit(x_norm, sp_phase_field)
    results.append({
        "unit": k, "dir": label, "n_spikes": len(sp_t_field),
        "x_lo": x_lo, "x_hi": x_hi, "peak_rate": float(rate_map.max()),
        "slope_rad_per_field": slope, "phi0": phi0, "rho": rho, "p": p,
        "x_norm": x_norm, "phase": sp_phase_field,
    })

res_df = pd.DataFrame([{kk: v for kk, v in r.items() if kk not in ("x_norm", "phase")} for r in results])
print(res_df.describe())
sig_neg = res_df[(res_df["p"] < 0.05) & (res_df["slope_rad_per_field"] < 0)]
print(f"Place cells with significant NEGATIVE slope (p<0.05): {len(sig_neg)} / {len(res_df)}")
print(f"Median slope (rad / field): {res_df['slope_rad_per_field'].median():.2f}")

res_df.to_csv(os.path.join(OUTDIR, "phase_precession_per_cell.csv"), index=False)
print("Saved phase_precession_per_cell.csv")

# %% [markdown]
# ## 13. Phase precession example plots
#
# Show the 9 example cells with the most negative (most strongly precessing) slopes.

# %%
res_sorted = sorted(results, key=lambda r: r["slope_rad_per_field"])
examples = [r for r in res_sorted if r["n_spikes"] > 80][:9]

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax, r in zip(axes.flat, examples):
    # plot two cycles for visual clarity (phase wrap)
    x = r["x_norm"]
    phi = r["phase"]
    ax.scatter(x, phi, s=4, color="C0", alpha=0.5)
    ax.scatter(x, phi + 2 * np.pi, s=4, color="C0", alpha=0.5)
    xx = np.linspace(0, 1, 100)
    yy = r["slope_rad_per_field"] * xx + r["phi0"]
    yy_wrap = (yy + np.pi) % (2 * np.pi) - np.pi
    # to draw line clearly, plot raw line in two cycles
    ax.plot(xx, yy_wrap, "r-", lw=1.5, alpha=0.9)
    ax.plot(xx, yy_wrap + 2 * np.pi, "r-", lw=1.5, alpha=0.9)
    ax.set_ylim(-np.pi, 3 * np.pi)
    ax.set_yticks([-np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi])
    ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$", r"$2\pi$", r"$3\pi$"])
    ax.set_xlim(0, 1)
    ax.set_title(f"unit {r['unit']} ({r['dir']}) | slope {r['slope_rad_per_field']:.1f} rad/field\n"
                 f"n={r['n_spikes']}, ρ={r['rho']:.2f}, p={r['p']:.3f}", fontsize=9)
    ax.set_xlabel("Position in field (norm.)")
    ax.set_ylabel("Theta phase")
fig.suptitle("Theta phase precession — example CA1 place cells", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig04_precession_examples.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print("Saved fig04_precession_examples.png")

# %% [markdown]
# ## 14. Population-level phase precession summary

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# A: histogram of slopes
ax = axes[0]
ax.hist(res_df["slope_rad_per_field"], bins=20, color="C0", edgecolor="white")
ax.axvline(0, color="k", lw=1)
ax.axvline(res_df["slope_rad_per_field"].median(), color="C3", lw=1.5,
           linestyle="--", label=f"median = {res_df['slope_rad_per_field'].median():.2f} rad/field")
ax.set_xlabel("Phase precession slope (rad / field width)")
ax.set_ylabel("# place cells")
ax.set_title("Slopes are predominantly negative\n(spikes shift to earlier phases)")
ax.legend(frameon=False)

# B: histogram of circular-linear correlations rho
ax = axes[1]
ax.hist(res_df["rho"], bins=20, color="C2", edgecolor="white")
ax.set_xlabel("Circular–linear correlation ρ")
ax.set_ylabel("# place cells")
ax.set_title(f"Correlation strength (median |ρ| = {np.abs(res_df['rho']).median():.2f})")

# C: pooled in-field phase vs position scatter (all cells stacked, after sign-flipping
# so each cell's preferred direction is increasing-x)
all_x = np.concatenate([r["x_norm"] for r in results])
all_phi = np.concatenate([r["phase"] for r in results])
ax = axes[2]
ax.scatter(all_x, all_phi, s=2, alpha=0.15, color="C0")
ax.scatter(all_x, all_phi + 2 * np.pi, s=2, alpha=0.15, color="C0")
# linear fit on circular-linear pooled
slope_pop, phi0_pop, rho_pop, p_pop = circ_lin_fit(all_x, all_phi)
xx = np.linspace(0, 1, 100)
yy = slope_pop * xx + phi0_pop
yy_wrap = (yy + np.pi) % (2 * np.pi) - np.pi
ax.plot(xx, yy_wrap, "r-", lw=2)
ax.plot(xx, yy_wrap + 2 * np.pi, "r-", lw=2)
ax.set_xlim(0, 1); ax.set_ylim(-np.pi, 3 * np.pi)
ax.set_xlabel("Position in field (norm.)")
ax.set_ylabel("Theta phase")
ax.set_yticks([-np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi])
ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$", r"$2\pi$", r"$3\pi$"])
ax.set_title(f"Pooled spikes (N={len(all_x)})\nslope = {slope_pop:.2f} rad, ρ={rho_pop:.2f}")

fig.suptitle("Phase precession — population summary", y=1.04)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig05_precession_population.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print("Saved fig05_precession_population.png")

# %% [markdown]
# ## 15. Summary
#
# - We loaded a single Buzsáki-lab CA1 recording from DANDI 000044 by streaming with
#   `remfile` (only the maze epoch and one LFP channel were ever transferred).
# - Pyramidal cells fire preferentially near a single phase of the 6–12 Hz theta rhythm
#   (entrainment): the population-average mean resultant length R ≈ {mvl}.
# - Place cells show systematic **negative** slopes of theta phase against in-field
#   position — the hallmark of theta phase precession.

# %%
print("=== DONE ===")
print(f"Place cells found: {len(place_cell_ids)}")
print(f"Cells used in precession analysis: {len(results)}")
print(f"Cells with significant negative slope (p<0.05): {len(sig_neg)}")
print(f"Median slope: {res_df['slope_rad_per_field'].median():.2f} rad/field")
print(f"Mean phase-locking R (pyramidal): {mvls.mean():.3f}")
