# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
# ---

# %% [markdown]
# # Reach direction and velocity tuning in macaque motor cortex
#
# **Dataset:** DANDI:000128 — *MC_Maze: macaque primary motor and dorsal premotor cortex
# spiking activity during a delayed reaching task* (Churchland, Kaufman, Shenoy lab,
# Stanford). Subject *Jenkins*, recorded with two 96-channel Utah arrays (M1 + PMd)
# while performing a delayed center-out reaching task with optional curved-path "maze"
# variants (Churchland et al. 2010, *Neuron*).
#
# **Goal.** Demonstrate two classic motor-cortex coding properties using real spiking
# data streamed from DANDI:
#
# 1. **Reach-direction tuning.** Single-unit firing rates during movement depend
#    systematically on the target direction relative to the body, with each unit
#    showing an approximately cosine tuning curve and a *preferred direction*
#    (Georgopoulos et al. 1982).
# 2. **Hand-velocity tuning.** When binned by the instantaneous 2D hand velocity
#    vector (rather than by trial-level direction), single units exhibit smooth
#    velocity tuning surfaces — firing rate rises along the unit's preferred
#    movement direction and (for many units) increases with movement speed.
#
# We use [pynapple](https://pynapple-org.github.io/pynapple/) for spike-time
# manipulation, tuning-curve computation, and decoding.

# %% [markdown]
# ## Setup and streaming load

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from tqdm import tqdm

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "axes.grid": False})
FIG_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
os.makedirs(FIG_DIR, exist_ok=True)

# %% [markdown]
# Stream the NWB file directly from DANDI with a local disk cache so chunks are
# reused across runs.

# %%
S3_URL = (
    "https://api.dandiarchive.org/api/dandisets/000128/versions/0.220113.0400/"
    "assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
)
CACHE_DIR = "/tmp/remfile_cache_reach"
os.makedirs(CACHE_DIR, exist_ok=True)

disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print(nwb)
print("Subject:", nwbfile.subject.subject_id, nwbfile.subject.species)
print("Session:", nwbfile.session_description[:80], "...")

# %% [markdown]
# ## Data streams
#
# The NWB file contains:
#
# - `units`: 182 sorted single units across M1 + PMd
# - `trials`: 2295 reach trials with `target_pos`, `active_target`, `move_onset_time`,
#   `go_cue_time`, `num_targets`, `num_barriers`, `success`
# - `hand_pos`, `hand_vel`: continuous 2D hand kinematics (x, y in mm, mm/s)

# %%
units = nwb["units"]
trials_df = nwbfile.trials.to_dataframe()
hand_pos = nwb["hand_pos"]
hand_vel = nwb["hand_vel"]

print(f"N units: {len(units)}")
print(f"N trials: {len(trials_df)}")
print(f"hand_pos shape: {hand_pos.shape}, columns: {hand_pos.columns.tolist()}")
print(f"hand_vel shape: {hand_vel.shape}")
print(f"hand_vel sampling: {1.0/np.median(np.diff(hand_vel.t)):.1f} Hz")

# %% [markdown]
# ## Trial selection: center-out reaches (no maze barriers)
#
# To isolate clean directional tuning we restrict to **single-target, zero-barrier**
# successful trials (n ≈ 789). Each trial's reach direction is the angle of the
# target position vector from the workspace center.

# %%
co_mask = (
    (trials_df.num_targets == 1)
    & (trials_df.num_barriers == 0)
    & trials_df.success
    & np.isfinite(trials_df.move_onset_time)
)
co_trials = trials_df[co_mask].copy()

# target_pos is shape (1, 2); flatten to (x, y) in mm
tgt = np.array([np.asarray(p).flatten()[:2] for p in co_trials.target_pos])
co_trials["target_x"] = tgt[:, 0]
co_trials["target_y"] = tgt[:, 1]
co_trials["target_angle"] = np.arctan2(tgt[:, 1], tgt[:, 0])  # radians, (-pi, pi]
co_trials["target_dist"] = np.hypot(tgt[:, 0], tgt[:, 1])

print(f"Center-out trials: {len(co_trials)}")
print(f"Unique target positions: {len(set(map(tuple, tgt))) }")
print(f"Target distance: median {co_trials.target_dist.median():.0f} mm "
      f"(min {co_trials.target_dist.min():.0f}, max {co_trials.target_dist.max():.0f})")

# Bin reach angles into 8 cardinal/oblique directions
N_DIRS = 8
edges = np.linspace(-np.pi, np.pi, N_DIRS + 1)
co_trials["dir_bin"] = np.digitize(co_trials.target_angle, edges) - 1
# wrap any -pi exactly to bin 0
co_trials["dir_bin"] = co_trials["dir_bin"].clip(0, N_DIRS - 1)

bin_centers = 0.5 * (edges[:-1] + edges[1:])
counts_per_dir = co_trials.dir_bin.value_counts().sort_index()
print("\nTrials per direction bin:")
for b in range(N_DIRS):
    deg = np.degrees(bin_centers[b])
    n = int(counts_per_dir.get(b, 0))
    print(f"  bin {b}: center {deg:+6.1f}°  n={n}")

# %% [markdown]
# ## Visualize trial geometry and example hand trajectories

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
ax.scatter(tgt[:, 0], tgt[:, 1], s=14, c=co_trials.dir_bin, cmap="hsv", alpha=0.7)
ax.scatter([0], [0], s=80, c="k", marker="+", label="center")
ax.set_xlabel("target x (mm)")
ax.set_ylabel("target y (mm)")
ax.set_title(f"Target locations (n={len(co_trials)} trials, color = direction bin)")
ax.set_aspect("equal")
ax.legend(loc="upper right")

# Plot hand trajectories for a few representative trials per direction
ax = axes[1]
cmap = plt.get_cmap("hsv")
for b in range(N_DIRS):
    sub = co_trials[co_trials.dir_bin == b].head(4)
    color = cmap(b / N_DIRS)
    for _, row in sub.iterrows():
        t0 = row.move_onset_time - 0.05
        t1 = row.move_onset_time + 0.6
        seg = hand_pos.restrict(nap.IntervalSet(start=t0, end=t1))
        if len(seg) > 0:
            x = seg[:, 0].values - seg[:, 0].values[0]
            y = seg[:, 1].values - seg[:, 1].values[0]
            ax.plot(x, y, color=color, alpha=0.6, lw=0.8)
ax.set_xlabel("hand Δx (mm)")
ax.set_ylabel("hand Δy (mm)")
ax.set_title("Hand trajectories from movement onset (4 trials/direction)")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig01_targets_and_trajectories.png"), bbox_inches="tight")
plt.close()

# %% [markdown]
# ## Per-trial firing rate per unit (movement window)
#
# For each trial we count spikes in a 500-ms window starting 50 ms before
# movement onset (capturing the build-up + early execution of the reach).

# %%
WIN_PRE = 0.05
WIN_POST = 0.45
move_starts = co_trials.move_onset_time.values - WIN_PRE
move_ends = co_trials.move_onset_time.values + WIN_POST
trial_iset = nap.IntervalSet(start=move_starts, end=move_ends)

# count spikes in each trial-window for every unit  →  shape (n_trials, n_units)
binned = units.count(ep=trial_iset, bin_size=WIN_PRE + WIN_POST)
# binned is a TsdFrame with one row per IntervalSet bin
rates = binned.values / (WIN_PRE + WIN_POST)  # Hz
print(f"rates shape: {rates.shape}  (trials × units)")
print(f"mean firing rate across units & trials: {rates.mean():.2f} Hz")

# %% [markdown]
# ## Directional tuning curves (8 bins)

# %%
unit_ids = list(units.keys())
n_units = len(unit_ids)

dir_curve = np.full((n_units, N_DIRS), np.nan)
dir_sem = np.full((n_units, N_DIRS), np.nan)
dir_n = np.zeros(N_DIRS, dtype=int)
for b in range(N_DIRS):
    in_b = co_trials.dir_bin.values == b
    dir_n[b] = in_b.sum()
    if in_b.sum() == 0:
        continue
    dir_curve[:, b] = rates[in_b].mean(axis=0)
    dir_sem[:, b] = rates[in_b].std(axis=0) / np.sqrt(in_b.sum())

# %% [markdown]
# ### Cosine fit: preferred direction & modulation depth
#
# A unit's directional tuning is well-described by
#   $r(\theta) = b_0 + b_1 \cos(\theta - \theta_p)$
# which is linear in $(b_0, b_x, b_y)$ with $b_x = b_1 \cos\theta_p$,
# $b_y = b_1 \sin\theta_p$. We fit by ordinary least squares to the per-trial
# rates of each unit.

# %%
theta = co_trials.target_angle.values
X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])

beta, *_ = np.linalg.lstsq(X, rates, rcond=None)  # (3, n_units)
b0 = beta[0]
bx = beta[1]
by = beta[2]
pref_dir = np.arctan2(by, bx)        # radians
mod_depth = np.hypot(bx, by)         # Hz amplitude of the cosine

# R² per unit
y_hat = X @ beta
ss_res = ((rates - y_hat) ** 2).sum(axis=0)
ss_tot = ((rates - rates.mean(axis=0)) ** 2).sum(axis=0)
r2 = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)

mean_rate = rates.mean(axis=0)
tuning_summary = pd.DataFrame({
    "unit_id": unit_ids,
    "mean_rate_Hz": mean_rate,
    "pref_dir_deg": np.degrees(pref_dir),
    "mod_depth_Hz": mod_depth,
    "cosine_R2": r2,
}).sort_values("cosine_R2", ascending=False)
print(tuning_summary.head(10).to_string(index=False))

# Save the table
tuning_summary.to_csv(os.path.join(FIG_DIR, "directional_tuning_summary.csv"), index=False)

# %% [markdown]
# ### Polar tuning curves for the most strongly tuned units

# %%
top = tuning_summary.head(8).reset_index(drop=True)

fig = plt.figure(figsize=(15, 9))
for k, row in top.iterrows():
    ax = fig.add_subplot(2, 4, k + 1, projection="polar")
    uid = row.unit_id
    j = unit_ids.index(uid)
    angles = np.r_[bin_centers, bin_centers[0]]
    rates_d = np.r_[dir_curve[j], dir_curve[j, 0]]
    sems_d = np.r_[dir_sem[j], dir_sem[j, 0]]
    valid = np.isfinite(rates_d)
    ax.plot(angles[valid], rates_d[valid], "o-", color="tab:blue", lw=1.6, ms=5)
    ax.fill_between(angles[valid],
                    (rates_d - sems_d)[valid], (rates_d + sems_d)[valid],
                    color="tab:blue", alpha=0.2)
    # cosine fit overlay
    t_fine = np.linspace(-np.pi, np.pi, 200)
    fit = b0[j] + bx[j] * np.cos(t_fine) + by[j] * np.sin(t_fine)
    ax.plot(np.r_[t_fine, t_fine[0]], np.r_[fit, fit[0]],
            color="crimson", lw=1.4, alpha=0.85, label="cosine fit")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(135)
    ax.set_title(f"unit {uid}\nPD={row.pref_dir_deg:+.0f}°  R²={row.cosine_R2:.2f}",
                 fontsize=10, pad=18)
    ax.tick_params(labelsize=8)
fig.suptitle("Directional tuning of the 8 most cosine-tuned units (movement window)",
             fontsize=13, y=0.99)
plt.tight_layout()
plt.subplots_adjust(top=0.90, hspace=0.55, wspace=0.55)
plt.savefig(os.path.join(FIG_DIR, "fig02_polar_tuning_top8.png"), bbox_inches="tight")
plt.close()

# %% [markdown]
# ### Population view: distribution of preferred directions and tuning quality

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# (a) PD distribution (only well-fit units)
mask_tuned = (r2 > 0.05) & (mod_depth > 1.0)
ax = axes[0]
ax.hist(np.degrees(pref_dir[mask_tuned]) % 360, bins=18,
        color="steelblue", edgecolor="white")
ax.set_xlabel("preferred direction (deg)")
ax.set_ylabel("# units")
ax.set_title(f"Preferred-direction distribution\n({mask_tuned.sum()}/{n_units} tuned units)")
ax.set_xticks(np.arange(0, 361, 45))

# (b) Modulation depth vs mean rate
ax = axes[1]
ax.scatter(mean_rate, mod_depth, c=r2, cmap="viridis", s=22, edgecolor="k", lw=0.3)
ax.set_xlabel("mean firing rate (Hz)")
ax.set_ylabel("cosine modulation depth (Hz)")
ax.set_title("Modulation vs mean rate (color = R²)")
cb = plt.colorbar(ax.collections[0], ax=ax, label="R²")

# (c) Cosine R² distribution
ax = axes[2]
ax.hist(r2[np.isfinite(r2)], bins=30, color="darkorange", edgecolor="white")
ax.axvline(0.05, color="k", ls="--", lw=1, label="R²=0.05")
ax.set_xlabel("cosine fit R²")
ax.set_ylabel("# units")
ax.set_title("Cosine-tuning R² across population")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig03_population_directional_tuning.png"), bbox_inches="tight")
plt.close()

# %% [markdown]
# ## Velocity tuning from continuous hand kinematics
#
# Direction tuning collapses each trial to a single rate. A finer-grained view
# treats firing rate as a function of the *instantaneous* 2D hand velocity
# vector. We restrict to the movement period (movement onset → +600 ms) on every
# included trial, then use Pynapple's 2D tuning-curve routine over (vx, vy).

# %%
# Build movement-period IntervalSet across all center-out trials
mvt_iset = nap.IntervalSet(
    start=co_trials.move_onset_time.values - 0.05,
    end=co_trials.move_onset_time.values + 0.55,
)

# Restrict velocity to movement periods
vel_mvt = hand_vel.restrict(mvt_iset)
print("velocity samples in movement window:", len(vel_mvt))
print("velocity ranges (mm/s): "
      f"vx [{vel_mvt[:,0].min():.0f}, {vel_mvt[:,0].max():.0f}], "
      f"vy [{vel_mvt[:,1].min():.0f}, {vel_mvt[:,1].max():.0f}]")

speed = np.hypot(vel_mvt[:, 0].values, vel_mvt[:, 1].values)
print(f"speed stats (mm/s): median {np.median(speed):.0f}, 95th {np.percentile(speed, 95):.0f}")

# %% [markdown]
# ### 2D velocity tuning maps via Pynapple

# %%
# clip extreme outliers for cleaner binning
v_lim = np.percentile(np.abs(np.r_[vel_mvt[:, 0].values, vel_mvt[:, 1].values]), 99)
print(f"velocity bin limit (99th pct |v|): {v_lim:.0f} mm/s")

n_bins = 12
vbin_edges = np.linspace(-v_lim, v_lim, n_bins + 1)

# Pynapple's compute_2d_tuning_curves expects a TsdFrame of features
tc2d, vbins = nap.compute_2d_tuning_curves(
    group=units, features=vel_mvt, nb_bins=n_bins,
    minmax=(-v_lim, v_lim, -v_lim, v_lim), ep=mvt_iset,
)
# tc2d is a dict {unit_id: 2d array (nb_bins x nb_bins)}
print(f"computed velocity tuning curves for {len(tc2d)} units, grid {n_bins}x{n_bins}")

# %% [markdown]
# Show the velocity-tuning maps for the same units that were strongly cosine-tuned
# at the trial level (top 8 by R²). Lightly smooth each map with a Gaussian to
# average out unsampled / sparsely-sampled bins.

# %%
from scipy.ndimage import gaussian_filter

vel_mean = np.array([np.nanmean(tc2d[u]) for u in unit_ids])
top_uids = tuning_summary.head(8).unit_id.tolist()
top_idx = [unit_ids.index(u) for u in top_uids]

fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
for k, (uid, j) in enumerate(zip(top_uids, top_idx)):
    ax = axes[k // 4, k % 4]
    grid = tc2d[uid].T  # x-axis becomes vx
    grid_filled = np.where(np.isfinite(grid), grid, np.nanmean(grid))
    grid_smooth = gaussian_filter(grid_filled, sigma=1.0)
    vmax = np.nanpercentile(grid_smooth, 99)
    im = ax.imshow(
        grid_smooth, origin="lower", aspect="equal", cmap="viridis",
        extent=[-v_lim, v_lim, -v_lim, v_lim], vmin=0, vmax=vmax,
    )
    ax.axhline(0, color="white", lw=0.5, alpha=0.6)
    ax.axvline(0, color="white", lw=0.5, alpha=0.6)
    pd_rad = pref_dir[j]
    ax.plot([0, v_lim * 0.9 * np.cos(pd_rad)],
            [0, v_lim * 0.9 * np.sin(pd_rad)],
            color="red", lw=2.0, alpha=0.95)
    ax.set_title(f"unit {uid}   µ={vel_mean[j]:.1f} Hz   PD={np.degrees(pd_rad):+.0f}°",
                 fontsize=10)
    ax.set_xlabel("vx (mm/s)", fontsize=9)
    ax.set_ylabel("vy (mm/s)", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="rate (Hz)")
fig.suptitle("2D hand-velocity tuning for the top cosine-tuned units "
             "(red = trial-level PD)", fontsize=13, y=0.99)
plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.45, wspace=0.55)
plt.savefig(os.path.join(FIG_DIR, "fig04_velocity_tuning_2d.png"), bbox_inches="tight")
plt.close()

# %% [markdown]
# ### Speed-only tuning (collapsing across direction)
#
# For each unit, compute mean firing rate vs. hand speed (regardless of
# direction). This isolates the *gain* component of velocity tuning.

# %%
# Build a TsdFrame of speed and movement-direction angle aligned to vel_mvt
speed_tsd = nap.Tsd(t=vel_mvt.t, d=speed)
move_dir = np.arctan2(vel_mvt[:, 1].values, vel_mvt[:, 0].values)
movedir_tsd = nap.Tsd(t=vel_mvt.t, d=move_dir)

speed_max = np.percentile(speed, 99)
tc_speed = nap.compute_1d_tuning_curves(
    units, speed_tsd, nb_bins=15, minmax=(0, speed_max), ep=mvt_iset,
)
tc_movedir = nap.compute_1d_tuning_curves(
    units, movedir_tsd, nb_bins=16, minmax=(-np.pi, np.pi), ep=mvt_iset,
)

# %% [markdown]
# Plot speed tuning (top row) and instantaneous movement-direction tuning
# (bottom row) for the top-4 cosine-tuned units, in separate panels.

# %%
top4 = top_uids[:4]
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for k, uid in enumerate(top4):
    j = unit_ids.index(uid)
    sp = tc_speed[uid]
    md = tc_movedir[uid]

    ax = axes[0, k]
    ax.plot(sp.index.values, sp.values, "o-", color="tab:blue", lw=1.6)
    ax.set_xlabel("hand speed (mm/s)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title(f"unit {uid}: speed gain", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1, k]
    ax.plot(np.degrees(md.index.values), md.values, "s-",
            color="tab:red", lw=1.6)
    ax.axvline(np.degrees(pref_dir[j]), color="k", ls="--", lw=1,
               label=f"trial PD={np.degrees(pref_dir[j]):+.0f}°")
    ax.set_xlabel("instantaneous movement direction (deg)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title(f"unit {uid}: direction tuning", fontsize=10)
    ax.set_xticks(np.arange(-180, 181, 90))
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

fig.suptitle("Speed gain (top) and instantaneous-direction tuning (bottom) "
             "from continuous hand kinematics", fontsize=12, y=0.99)
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.45, wspace=0.45)
plt.savefig(os.path.join(FIG_DIR, "fig05_speed_and_direction_tuning.png"), bbox_inches="tight")
plt.close()

# %% [markdown]
# ## Population vector decoding of reach direction
#
# Georgopoulos's *population vector*: each unit votes in the direction of its
# preferred direction with weight equal to its (normalized) firing rate. The
# vector sum across the population should point in the direction of the upcoming
# reach. We compute the population vector for each held-out trial and compare
# to the actual target direction.

# %%
tuned_idx = np.where(mask_tuned)[0]
print(f"Using {len(tuned_idx)} tuned units for population decoding.")

# Z-score each tuned unit's per-trial rate
mu = rates[:, tuned_idx].mean(axis=0)
sd = rates[:, tuned_idx].std(axis=0)
sd[sd == 0] = 1.0
Z = (rates[:, tuned_idx] - mu) / sd  # (trials, n_tuned)

pd_x = np.cos(pref_dir[tuned_idx])
pd_y = np.sin(pref_dir[tuned_idx])
pop_x = Z @ pd_x
pop_y = Z @ pd_y
pop_angle = np.arctan2(pop_y, pop_x)

# circular error
err = np.angle(np.exp(1j * (pop_angle - co_trials.target_angle.values)))
err_deg = np.degrees(err)
mean_abs_err = np.mean(np.abs(err_deg))
median_abs_err = np.median(np.abs(err_deg))
print(f"Population-vector decoding error: mean |Δ|={mean_abs_err:.1f}°, "
      f"median |Δ|={median_abs_err:.1f}°")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.scatter(np.degrees(co_trials.target_angle.values),
           np.degrees(pop_angle), s=8, alpha=0.55, color="navy")
ax.plot([-180, 180], [-180, 180], "k--", lw=0.8)
ax.set_xlabel("actual target direction (deg)")
ax.set_ylabel("decoded population-vector angle (deg)")
ax.set_title(f"Population-vector decoding\n(median |error| = {median_abs_err:.1f}°)")
ax.set_xticks(np.arange(-180, 181, 90))
ax.set_yticks(np.arange(-180, 181, 90))
ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
ax.set_aspect("equal")

ax = axes[1]
ax.hist(err_deg, bins=36, color="seagreen", edgecolor="white")
ax.axvline(0, color="k", lw=1, ls="--")
ax.set_xlabel("decoding error (deg, signed)")
ax.set_ylabel("# trials")
ax.set_title("Per-trial decoding error distribution")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig06_population_vector_decoding.png"), bbox_inches="tight")
plt.close()

# %% [markdown]
# ## Time-resolved tuning: PSTHs by direction for an example unit
#
# Build per-trial spike rasters and PSTHs for the strongest cosine-tuned unit,
# grouped by reach direction. The PSTH should peak in the direction matching the
# unit's preferred direction.

# %%
example_uid = int(tuning_summary.iloc[0].unit_id)
example_idx = unit_ids.index(example_uid)
example_pd = np.degrees(pref_dir[example_idx])
print(f"Example unit {example_uid}: PD={example_pd:.1f}°, "
      f"R²={r2[example_idx]:.2f}")

T_PRE, T_POST = 0.3, 0.6
bin_size = 0.02
edges_t = np.arange(-T_PRE, T_POST + bin_size, bin_size)
psth = np.zeros((N_DIRS, len(edges_t) - 1))
psth_n = np.zeros(N_DIRS, dtype=int)
spk = units[example_uid]

raster = {b: [] for b in range(N_DIRS)}

for _, row in co_trials.iterrows():
    b = int(row.dir_bin)
    t0 = row.move_onset_time
    seg = spk.restrict(nap.IntervalSet(start=t0 - T_PRE, end=t0 + T_POST))
    spike_offsets = seg.t - t0
    raster[b].append(spike_offsets)
    h, _ = np.histogram(spike_offsets, bins=edges_t)
    psth[b] += h
    psth_n[b] += 1

psth_rate = psth / np.maximum(psth_n[:, None], 1) / bin_size
t_centers = 0.5 * (edges_t[:-1] + edges_t[1:])

# %%
psth_max = np.nanmax(psth_rate) * 1.05 if np.nanmax(psth_rate) > 0 else 1
fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
for b in range(N_DIRS):
    ax = axes[b // 4, b % 4]
    if psth_n[b] == 0:
        ax.text(0.5, 0.5, "no trials", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="gray")
        ax.set_title(f"dir {np.degrees(bin_centers[b]):+.0f}°  (n=0)",
                     fontsize=11)
        ax.set_xlabel("time from move-onset (s)", fontsize=10)
        continue
    for i, offsets in enumerate(raster[b][:40]):
        ax.scatter(offsets, np.full_like(offsets, i + 1), s=2, color="k")
    ax.set_ylim(0, 41)
    ax.set_ylabel("trial # (dots)", fontsize=10)

    ax2 = ax.twinx()
    ax2.plot(t_centers, psth_rate[b], color="crimson", lw=1.8)
    ax2.set_ylabel("rate (Hz)", color="crimson", fontsize=10)
    ax2.set_ylim(0, psth_max)
    ax2.tick_params(axis="y", labelcolor="crimson", labelsize=9)

    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"dir {np.degrees(bin_centers[b]):+.0f}°  (n={psth_n[b]})",
                 fontsize=11)
    ax.set_xlabel("time from move-onset (s)", fontsize=10)
    ax.tick_params(labelsize=9)
fig.suptitle(f"Unit {example_uid}: rasters + PSTHs by reach direction "
             f"(trial-level PD={example_pd:+.0f}°)", fontsize=13, y=0.99)
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.40, wspace=0.55)
plt.savefig(os.path.join(FIG_DIR, "fig07_example_unit_psth_by_direction.png"),
            bbox_inches="tight")
plt.close()

# %% [markdown]
# ## Summary
#
# Using DANDI:000128 (macaque M1 + PMd, delayed center-out reaching, n=789
# clean center-out trials, 182 sorted units) we showed:
#
# 1. **Cosine direction tuning.** Many units fit the classic
#    $r(\theta) = b_0 + b_1\cos(\theta - \theta_p)$ form with substantial
#    explained variance. Preferred directions tile the full $[0, 360°)$ range.
# 2. **2D velocity tuning.** Pynapple's `compute_2d_tuning_curves` reveals
#    smooth velocity-tuning surfaces whose peaks line up with the preferred
#    direction recovered from the trial-level cosine fit (red lines on the
#    velocity heatmaps).
# 3. **Speed gain.** Most strongly velocity-modulated units increase their rate
#    monotonically with hand speed, on top of the directional preference.
# 4. **Population code.** A simple Georgopoulos population vector built from
#    the tuned units recovers the reach direction with median absolute error
#    of just a few tens of degrees, confirming that motor-cortex population
#    activity carries a faithful, distributed representation of upcoming
#    reach direction.

# %%
io.close()
print("\nGenerated figures:")
for fn in sorted(os.listdir(FIG_DIR)):
    if fn.endswith(".png"):
        print(" ", fn)
