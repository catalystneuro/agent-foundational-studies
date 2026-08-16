# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
# ---

# %% [markdown]
# # Reach Direction and Velocity Tuning in Macaque Motor Cortex
#
# This notebook demonstrates **reach direction and velocity tuning** in single neurons
# recorded from primary motor cortex (M1) and dorsal premotor cortex (PMd) of a macaque
# performing a center-out reaching task with maze barriers.
#
# **Dataset**: DANDI [000128 — MC_Maze](https://dandiarchive.org/dandiset/000128)
# (Churchland & Kaufman, Shenoy lab, Stanford). Subject: monkey Jenkins.
#
# **Phenomenon**: Since Georgopoulos's seminal work (1982), motor cortical neurons
# are known to exhibit **cosine tuning** for reach direction: their firing rate is
# approximately a cosine function of the angle between the reach direction and the
# neuron's **preferred direction**. Many neurons additionally encode **movement speed**
# (Moran & Schwartz 1999) or even full velocity vectors. We will:
#
# 1. Load streaming data from S3 with `remfile` + Pynapple.
# 2. Compute each trial's reach direction from the active target position.
# 3. Build PSTHs locked to movement onset, split by reach direction.
# 4. Fit per-neuron **cosine tuning curves** for direction.
# 5. Fit a **velocity GLM** (NeMoS) using x/y hand velocity to verify each neuron's
#    preferred velocity vector.
# 6. Summarize the population distribution of preferred directions and tuning depth.

# %% [markdown]
# ## 1. Setup and data loading

# %%
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm.auto import tqdm

import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__))
                       if "__file__" in globals() else ".", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__))
                     if "__file__" in globals() else ".", "cache")
os.makedirs(CACHE, exist_ok=True)

S3_URL = ('https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/'
          'df3e3f73-50ab-42b4-8827-82664ddd474a')

disk_cache = remfile.DiskCache(CACHE)
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# The session contains 182 sorted units (M1 + PMd), 2295 successful center-out
# delayed reaches, and continuous hand position / velocity at 1 kHz.

# %%
units = nwb["units"]                    # TsGroup of spike times
trials = nwb["trials"]                  # IntervalSet (with metadata)
hand_pos = nwb["hand_pos"]              # TsdFrame, columns 0=x, 1=y (mm)
hand_vel = nwb["hand_vel"]              # TsdFrame, columns 0=vx, 1=vy (mm/s)
print(f"Number of units: {len(units)}")
print(f"Number of trials: {len(trials)}")
print(f"Hand pos shape:  {hand_pos.shape}, sampling: {1/np.median(np.diff(hand_pos.t)):.0f} Hz")
print(f"Hand vel shape:  {hand_vel.shape}")

# %% [markdown]
# ## 2. Compute per-trial reach direction
#
# Each trial's `active_target` indexes into `target_pos` to give the (x,y) target
# location relative to workspace center (mm). The reach direction angle is
# `atan2(y, x)`.

# %%
trial_df = nwbfile.trials.to_dataframe()
move_onset = trial_df["move_onset_time"].to_numpy()
go_cue = trial_df["go_cue_time"].to_numpy()
target_on = trial_df["target_on_time"].to_numpy()
start_t = trial_df["start_time"].to_numpy()
stop_t = trial_df["stop_time"].to_numpy()

active_target = trial_df["active_target"].to_numpy().astype(int)
target_xy = np.array(
    [trial_df["target_pos"].iloc[i][active_target[i]] for i in range(len(trial_df))],
    dtype=float,
)
reach_angle = np.arctan2(target_xy[:, 1], target_xy[:, 0])    # radians
reach_dist = np.hypot(target_xy[:, 0], target_xy[:, 1])

print(f"target distance (mm): mean {reach_dist.mean():.1f}, "
      f"median {np.median(reach_dist):.1f}")
print("Reach angle distribution (degrees):")
print(np.round(np.degrees(np.sort(np.unique(np.round(reach_angle, 2))))[:12], 1), "...")

# %% [markdown]
# Reaches are not at evenly-spaced angles: maze trials redirect the hand around
# barriers, so we will work with the **continuous reach-angle** as well as
# **8 angular bins** (45° wide) for tuning curves.

# %%
N_DIR_BINS = 8
edges = np.linspace(-np.pi, np.pi, N_DIR_BINS + 1)
dir_bin = np.digitize(reach_angle, edges) - 1
dir_bin = np.clip(dir_bin, 0, N_DIR_BINS - 1)
bin_centers = 0.5 * (edges[:-1] + edges[1:])
print("Trials per directional bin:", np.bincount(dir_bin, minlength=N_DIR_BINS))

# %% [markdown]
# ## 3. Visualize hand kinematics by reach direction

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

# Trajectories around movement onset, colored by direction
ax = axes[0]
cmap = cm.hsv
n_show = 400
sample_trials = np.random.RandomState(0).choice(len(trial_df), size=n_show, replace=False)
for ti in sample_trials:
    t0 = move_onset[ti]
    t1 = t0 + 0.6
    seg = hand_pos.restrict(nap.IntervalSet(start=t0, end=t1)).values
    if len(seg) < 5:
        continue
    seg = seg - seg[0]
    color = cmap((reach_angle[ti] + np.pi) / (2 * np.pi))
    ax.plot(seg[:, 0], seg[:, 1], color=color, alpha=0.35, lw=0.6)
ax.set_aspect("equal")
ax.set_xlabel("hand x (mm, relative to onset)")
ax.set_ylabel("hand y (mm, relative to onset)")
ax.set_title(f"Hand trajectories ({n_show} trials)\ncolor = reach angle")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)

# Speed profile per direction bin, locked to movement onset
ax = axes[1]
WIN = (-0.2, 0.6)
dt = 0.01
t_grid = np.arange(WIN[0], WIN[1], dt)
speed_per_bin = np.full((N_DIR_BINS, len(t_grid)), np.nan)
for b in range(N_DIR_BINS):
    idxs = np.where(dir_bin == b)[0]
    profs = []
    for ti in idxs:
        t0 = move_onset[ti]
        seg = hand_vel.restrict(nap.IntervalSet(start=t0 + WIN[0],
                                                end=t0 + WIN[1]))
        if len(seg) < 5:
            continue
        speed = np.hypot(seg.values[:, 0], seg.values[:, 1])
        prof = np.interp(t_grid, seg.t - t0, speed,
                         left=np.nan, right=np.nan)
        profs.append(prof)
    if profs:
        speed_per_bin[b] = np.nanmean(np.stack(profs), axis=0)

for b in range(N_DIR_BINS):
    ax.plot(t_grid, speed_per_bin[b],
            color=cmap((bin_centers[b] + np.pi) / (2 * np.pi)),
            label=f"{int(np.degrees(bin_centers[b]))}°")
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("time from move onset (s)")
ax.set_ylabel("hand speed (mm/s)")
ax.set_title("Mean speed profile by reach direction")
ax.legend(fontsize=7, ncol=2, title="reach angle")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_hand_kinematics.png"), dpi=150)
plt.close()

# %% [markdown]
# ## 4. PSTHs locked to movement onset, split by reach direction
#
# We pick a few example neurons and show their firing-rate profiles (spike density
# functions) across the 8 reach directions. Cosine tuning means the neuron will
# fire most strongly for a single preferred direction.

# %%
PSTH_WIN = (-0.3, 0.7)
PSTH_DT = 0.02
psth_t = np.arange(PSTH_WIN[0], PSTH_WIN[1], PSTH_DT)

def compute_psth(spike_times, ref_times, t_grid, dt):
    """PSTH (rate) averaged over reference events. Counts then /dt/n."""
    counts = np.zeros(len(t_grid))
    n = 0
    for t0 in ref_times:
        idx = np.searchsorted(spike_times, t0 + t_grid[0])
        idx2 = np.searchsorted(spike_times, t0 + t_grid[-1] + dt)
        rel = spike_times[idx:idx2] - t0
        if rel.size:
            bins = np.floor((rel - t_grid[0]) / dt).astype(int)
            bins = bins[(bins >= 0) & (bins < len(t_grid))]
            counts[bins] += 1
        n += 1
    return counts / (dt * max(n, 1))

# pre-extract spike-time arrays for fast looping
unit_spike_arrays = {u: np.asarray(units[u].t) for u in units.keys()}

# pick 6 high-firing-rate units that look strongly tuned
mean_rates = {u: len(unit_spike_arrays[u])
              / (units.time_support.tot_length()) for u in units.keys()}
# rank by depth of direction tuning measured as range of mean firing in a
# movement-execution window
WIN_EXEC = (0.05, 0.35)
def mean_rate_in_window(spk, ref, win):
    counts = np.array([
        np.searchsorted(spk, r + win[1]) - np.searchsorted(spk, r + win[0])
        for r in ref])
    return counts / (win[1] - win[0])

depth_score = {}
for u in units.keys():
    spk = unit_spike_arrays[u]
    r = mean_rate_in_window(spk, move_onset, WIN_EXEC)
    bin_means = np.array([r[dir_bin == b].mean() if (dir_bin == b).any() else 0
                          for b in range(N_DIR_BINS)])
    depth_score[u] = bin_means.max() - bin_means.min()
top_units = sorted(units.keys(), key=lambda u: -depth_score[u])[:6]
print("Top units by tuning depth (Hz range across directions):",
      [(u, round(depth_score[u], 1)) for u in top_units])

# %%
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for ax, u in zip(axes.flat, top_units):
    spk = unit_spike_arrays[u]
    for b in range(N_DIR_BINS):
        idxs = np.where(dir_bin == b)[0]
        if len(idxs) < 3:
            continue
        psth = compute_psth(spk, move_onset[idxs], psth_t, PSTH_DT)
        # smooth with a small Gaussian
        from scipy.ndimage import gaussian_filter1d
        psth = gaussian_filter1d(psth, sigma=2.0)
        ax.plot(psth_t, psth,
                color=cmap((bin_centers[b] + np.pi) / (2 * np.pi)),
                lw=1.4)
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_title(f"unit {u}  (depth={depth_score[u]:.1f} Hz)")
    ax.set_xlabel("time from move onset (s)")
    ax.set_ylabel("rate (Hz)")
plt.suptitle("PSTHs by reach direction — six most directionally tuned units",
             y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_psths_by_direction.png"), dpi=150)
plt.close()

# %% [markdown]
# ## 5. Cosine tuning curves
#
# For every neuron, we compute the mean firing rate during the execution window
# `[move_onset+50ms, move_onset+350ms]` for each reach angle, then fit
#
# $$ r(\theta) = b_0 + b_x\cos\theta + b_y\sin\theta $$
#
# which is equivalent to $r(\theta) = b_0 + M\cos(\theta - \theta_{\text{pref}})$
# with preferred direction $\theta_{\text{pref}} = \mathrm{atan2}(b_y, b_x)$ and
# tuning depth $M = \sqrt{b_x^2 + b_y^2}$.

# %%
n_units = len(units)
unit_ids = list(units.keys())
firing_per_trial = np.zeros((n_units, len(trial_df)))
for i, u in enumerate(unit_ids):
    firing_per_trial[i] = mean_rate_in_window(
        unit_spike_arrays[u], move_onset, WIN_EXEC)

# Per-direction-bin mean for tuning curve plot
bin_mean_rate = np.zeros((n_units, N_DIR_BINS))
bin_sem_rate = np.zeros((n_units, N_DIR_BINS))
for b in range(N_DIR_BINS):
    msk = dir_bin == b
    bin_mean_rate[:, b] = firing_per_trial[:, msk].mean(axis=1)
    bin_sem_rate[:, b] = (firing_per_trial[:, msk].std(axis=1)
                          / np.sqrt(msk.sum()))

# Cosine fit using the continuous reach angle of every trial
X = np.column_stack([np.ones_like(reach_angle),
                     np.cos(reach_angle),
                     np.sin(reach_angle)])
beta, *_ = np.linalg.lstsq(X, firing_per_trial.T, rcond=None)
b0, bx, by = beta
pref_dir = np.arctan2(by, bx)
tuning_depth = np.hypot(bx, by)

# R^2 of cosine fit per neuron
y_pred = X @ beta
ss_res = ((firing_per_trial.T - y_pred) ** 2).sum(axis=0)
ss_tot = ((firing_per_trial - firing_per_trial.mean(axis=1, keepdims=True)) ** 2
          ).sum(axis=1)
r2 = 1 - ss_res / np.maximum(ss_tot, 1e-9)

print(f"Median per-neuron cosine-fit R²: {np.median(r2):.3f}")
print(f"Fraction of neurons with R² > 0.05: "
      f"{(r2 > 0.05).mean():.2f}")

# %%
fig, axes = plt.subplots(2, 3, figsize=(13, 8),
                        subplot_kw={"projection": "polar"})
for ax, u in zip(axes.flat, top_units):
    i = unit_ids.index(u)
    theta_plot = np.linspace(-np.pi, np.pi, 200)
    fit = b0[i] + bx[i] * np.cos(theta_plot) + by[i] * np.sin(theta_plot)
    ax.plot(theta_plot, np.maximum(fit, 0), color="k", lw=2, label="cosine fit")
    ax.errorbar(bin_centers, bin_mean_rate[i], yerr=bin_sem_rate[i],
                fmt="o", color="tab:red", lw=1, ms=5)
    ax.set_title(f"unit {u}\nPD={np.degrees(pref_dir[i]):+.0f}°  "
                 f"R²={r2[i]:.2f}", pad=18)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
plt.suptitle("Reach-direction tuning curves (cosine fit, polar)", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_tuning_curves_polar.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## 6. Population distribution of preferred directions and tuning depth

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# (a) preferred direction distribution
ax = axes[0]
ax.hist(np.degrees(pref_dir[r2 > 0.05]), bins=24,
        color="steelblue", edgecolor="white")
ax.set_xlabel("preferred direction (°)")
ax.set_ylabel("# neurons")
ax.set_title(f"Preferred directions\n({(r2 > 0.05).sum()} of {n_units} neurons, "
             f"R²>0.05)")
ax.set_xlim(-180, 180)
ax.set_xticks([-180, -90, 0, 90, 180])

# (b) tuning depth (modulation amplitude) vs mean firing rate
ax = axes[1]
mean_rate = firing_per_trial.mean(axis=1)
sc = ax.scatter(mean_rate, tuning_depth, c=r2, s=12, cmap="viridis",
                vmin=0, vmax=0.4)
ax.set_xlabel("mean firing rate (Hz)")
ax.set_ylabel("cosine tuning depth $M$ (Hz)")
ax.set_title("Tuning depth vs mean rate")
plt.colorbar(sc, ax=ax, label="cosine fit $R^2$")

# (c) histogram of R²
ax = axes[2]
ax.hist(r2, bins=30, color="darkorange", edgecolor="white")
ax.axvline(np.median(r2), color="k", ls="--",
           label=f"median = {np.median(r2):.2f}")
ax.set_xlabel("cosine fit $R^2$")
ax.set_ylabel("# neurons")
ax.set_title("Goodness of fit")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_population_summary.png"), dpi=150)
plt.close()

# %% [markdown]
# ## 7. Velocity tuning via NeMoS GLM
#
# Direction tuning collapses each trial to a single rate. To assess whether neurons
# encode the **velocity vector** (direction × speed) at fine time scales, we fit a
# linearized Poisson GLM:
#
# $$ \log \lambda(t) = w_x v_x(t) + w_y v_y(t) + b $$
#
# using continuous hand velocity. The preferred velocity direction is
# $\mathrm{atan2}(w_y, w_x)$; the magnitude of $(w_x, w_y)$ measures speed sensitivity.
# We fit on movement-period samples only (between go-cue and trial end).

# %%
import nemos as nmo

# Build a movement-period IntervalSet and resample velocity at 50 Hz
move_set = nap.IntervalSet(start=move_onset - 0.05,
                           end=move_onset + 0.45)
BIN = 0.02
binned_vel = hand_vel.bin_average(BIN, ep=move_set)
binned_vel = binned_vel.dropna()           # drop edge bins
# Spike counts at the same binning
counts = units.count(BIN, ep=move_set)
# Align to the velocity time grid
common_t = np.intersect1d(np.round(binned_vel.t, 6),
                          np.round(counts.t, 6))
# Use shared time index
mask_v = np.isin(np.round(binned_vel.t, 6), common_t)
mask_c = np.isin(np.round(counts.t, 6), common_t)
v_arr = binned_vel.values[mask_v] / 100.0   # scale to ~unit range (cm/s/10)
y_arr = counts.values[mask_c].astype(float)
print("Design matrix shape:", v_arr.shape, "spike count matrix:", y_arr.shape)

# %%
# Fit a population GLM: identity on velocity, Poisson observations (default)
glm = nmo.glm.PopulationGLM(
    regularizer="Ridge",
    regularizer_strength=1e-3,
)
glm.fit(v_arr, y_arr)
W = np.asarray(glm.coef_)        # (n_features=2, n_neurons)
b_glm = np.asarray(glm.intercept_)
print("Coefficient matrix shape:", W.shape)

vel_pref_dir = np.arctan2(W[1], W[0])
vel_tuning_mag = np.hypot(W[0], W[1])

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.scatter(np.degrees(pref_dir), np.degrees(vel_pref_dir),
           c=r2, cmap="viridis", s=18, vmin=0, vmax=0.4)
# wrap-around dashed lines
for shift in [-360, 0, 360]:
    ax.plot([-180, 180], [-180 + shift, 180 + shift],
            "k--", lw=0.8, alpha=0.6)
ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
ax.set_xlabel("preferred direction (cosine fit, °)")
ax.set_ylabel("preferred direction (velocity GLM, °)")
ax.set_title("Direction tuning is consistent across methods")
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_yticks([-180, -90, 0, 90, 180])

ax = axes[1]
ax.scatter(tuning_depth, vel_tuning_mag, c=r2, cmap="viridis",
           s=18, vmin=0, vmax=0.4)
ax.set_xlabel("cosine tuning depth (Hz)")
ax.set_ylabel("velocity GLM weight magnitude")
ax.set_title("Speed sensitivity correlates with directional tuning depth")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_velocity_glm.png"), dpi=150)
plt.close()

# %% [markdown]
# ## 8. Speed-modulation: firing rate vs reach speed at preferred direction

# %%
# For each neuron, project velocity onto its preferred direction and bin
proj = np.cos(vel_pref_dir)[None, :] * v_arr[:, [0]] \
     + np.sin(vel_pref_dir)[None, :] * v_arr[:, [1]]
# proj has shape (T, n_neurons)
SPEED_BINS = np.linspace(-3, 3, 13)        # cm/s/10
mid = 0.5 * (SPEED_BINS[:-1] + SPEED_BINS[1:])
rate_vs_speed = np.full((n_units, len(mid)), np.nan)
for ni in range(n_units):
    p = proj[:, ni]
    for k in range(len(mid)):
        sel = (p >= SPEED_BINS[k]) & (p < SPEED_BINS[k + 1])
        if sel.sum() > 30:
            rate_vs_speed[ni, k] = y_arr[sel, ni].mean() / BIN

# Show top-tuned units
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for u in top_units:
    i = unit_ids.index(u)
    ax.plot(mid * 100, rate_vs_speed[i], "-o", lw=1.5, ms=4,
            label=f"unit {u}")
ax.set_xlabel("speed projected onto preferred direction (cm/s)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("Speed–rate relationship along each unit's preferred direction")
ax.legend(fontsize=8, ncol=2)
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_speed_rate.png"), dpi=150)
plt.close()

# %% [markdown]
# ## 9. Summary numbers

# %%
summary = {
    "n_units": int(n_units),
    "n_trials": int(len(trial_df)),
    "median_cosine_R2": float(np.median(r2)),
    "fraction_R2_above_0p05": float((r2 > 0.05).mean()),
    "fraction_R2_above_0p2": float((r2 > 0.2).mean()),
    "median_tuning_depth_Hz": float(np.median(tuning_depth)),
    "median_velocity_glm_weight_magnitude": float(np.median(vel_tuning_mag)),
    "median_abs_PD_difference_deg": float(np.median(np.abs(np.degrees(
        np.arctan2(np.sin(pref_dir - vel_pref_dir),
                   np.cos(pref_dir - vel_pref_dir)))))),
}
print("=== Summary ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__))
                       if "__file__" in globals() else ".",
                       "summary.txt"), "w") as fh:
    for k, v in summary.items():
        fh.write(f"{k}: {v}\n")

print("\nDone — figures saved to:", FIG_DIR)
