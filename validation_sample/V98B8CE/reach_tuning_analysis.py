# %% [markdown]
# # Reach Direction and Velocity Tuning in Motor Cortex and Dorsal Premotor Cortex
#
# This notebook demonstrates two classic findings in motor cortical physiology,
# reach-direction tuning (Georgopoulos et al., 1982) and speed/velocity tuning
# (Moran & Schwartz, 1999), using real single-unit recordings from macaque M1
# and dorsal premotor cortex (PMd) recorded during a center-out-like reaching
# task with obstacles ("maze" task).
#
# **Dataset**: DANDI Archive Dandiset
# [000070](https://dandiarchive.org/dandiset/000070) ("Neural population
# dynamics during reaching", Churchland lab). We use a single session,
# `sub-Jenkins/sub-Jenkins_ses-20090912_behavior+ecephys.nwb`, which contains
# 192 sorted units from two 96-channel Utah arrays (96 in M1, 96 in PMd),
# hand position recorded at 1 kHz, and 1588 trials with detailed trial-timing
# and target-location metadata.
#
# The file is streamed directly from the DANDI S3 bucket with `remfile`
# (no full download) and analyzed with Pynapple.

# %% [markdown]
# ## Setup and Data Loading

# %%
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

warnings.filterwarnings("ignore", message="timestamps are not sorted")

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

S3_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/2b3/75e/"
    "2b375e1a-120d-4fc0-9d80-255712170214"
)

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %%
units = nwb["units"]
trials = nwb["trials"]
hand = nwb["Hand"]

locations = units.get_info("location")
is_m1 = locations.str.contains("M1").values
print(f"Total units: {len(units)}  |  M1: {is_m1.sum()}  |  PMd: {(~is_m1).sum()}")
print(f"Total trials: {len(trials)}")
print(f"Session duration: {hand.time_support['end'][-1] / 60:.1f} min "
      f"(hand samples only recorded during trials)")

# %% [markdown]
# ## Data Overview and Validation
#
# Before any analysis, we look at the raw hand-position trace and confirm
# spikes and behavior line up sensibly in time.

# %%
good_mask = (trials["discard_trial"].values == 0) & (trials["task_success"].values == 1)
good_idx = np.where(good_mask)[0]
print(f"Successful, non-discarded trials: {good_mask.sum()} / {len(trials)}")

example_trial = good_idx[0]
t0 = trials["start"][example_trial]
t1 = trials["end"][example_trial]
example_ep = nap.IntervalSet(start=t0 - 0.2, end=t1 + 0.2)
hand_ex = hand.restrict(example_ep)

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
axes[0].plot(hand_ex.t, hand_ex["x"].values, color="#1f77b4")
axes[0].axvline(trials["move_begins_time"].values[example_trial], color="k", ls="--", lw=1)
axes[0].axvline(trials["move_ends_time"].values[example_trial], color="k", ls="--", lw=1)
axes[0].set_ylabel("hand x (mm)")
axes[0].set_title(f"Example trial {example_trial}: hand position and example unit spikes")

axes[1].plot(hand_ex.t, hand_ex["y"].values, color="#ff7f0e")
axes[1].axvline(trials["move_begins_time"].values[example_trial], color="k", ls="--", lw=1)
axes[1].axvline(trials["move_ends_time"].values[example_trial], color="k", ls="--", lw=1)
axes[1].set_ylabel("hand y (mm)")

example_units = units[np.where(is_m1)[0][:15]]
for i, (uid, ts) in enumerate(example_units.items()):
    sp = ts.restrict(example_ep)
    axes[2].vlines(sp.t, i, i + 0.8, color="k", lw=1)
axes[2].set_ylabel("M1 unit #")
axes[2].set_xlabel("time (s)")
axes[2].set_title("raster, first 15 M1 units (dashed lines = movement onset/offset)")

plt.tight_layout()
plt.savefig("fig01_raw_data_overview.png", dpi=150)
plt.close()

# %% [markdown]
# ## Task Structure: Straight, Single-Target Reaches
#
# Most trials in this "maze" task involve virtual barriers that force curved
# reaches. To reproduce the classic center-out tuning-curve analysis we first
# isolate the subset of trials with **no barriers and a single target**,
# which are simple, roughly straight point-to-point reaches from a central
# hold position to one of nine peripheral targets spanning the workspace.

# %%
straight_mask = (
    good_mask
    & (trials["maze_num_barriers"].values == 0)
    & (trials["maze_num_targets"].values == 1)
)
straight_idx = np.where(straight_mask)[0]
print(f"Straight, single-target reaches: {len(straight_idx)}")

targets = np.stack(
    [trials["hit_target_position"].values[i] for i in straight_idx]
).astype(float)
angles_deg = np.degrees(np.arctan2(targets[:, 1], targets[:, 0]))
mb_s = trials["move_begins_time"].values[straight_idx]
me_s = trials["move_ends_time"].values[straight_idx]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
uniq_targets = np.unique(np.round(targets), axis=0)
cmap = plt.get_cmap("hsv")
for ut in uniq_targets:
    trial_sel = np.all(np.round(targets) == ut, axis=1)
    n_show = min(8, trial_sel.sum())
    shown = 0
    ang = np.degrees(np.arctan2(ut[1], ut[0]))
    color = cmap((ang + 180) / 360)
    for k in np.where(trial_sel)[0]:
        if shown >= n_show:
            break
        tr = straight_idx[k]
        ep = nap.IntervalSet(
            start=trials["start"][tr], end=trials["end"][tr]
        )
        h = hand.restrict(ep)
        axes[0].plot(h["x"].values, h["y"].values, color=color, alpha=0.6, lw=0.8)
        shown += 1
    axes[0].scatter(*ut, color=color, s=80, edgecolor="k", zorder=5)
axes[0].scatter(0, 0, color="k", marker="+", s=100, zorder=5, label="start")
axes[0].set_xlabel("hand x (mm)")
axes[0].set_ylabel("hand y (mm)")
axes[0].set_title("Example straight reach trajectories, by target")
axes[0].legend(loc="upper left")
axes[0].set_aspect("equal")

axes[1].hist(angles_deg, bins=36, range=(-180, 180), color="#555555")
axes[1].set_xlabel("target direction (deg)")
axes[1].set_ylabel("# trials")
axes[1].set_title("Target directions used across straight-reach trials")
plt.tight_layout()
plt.savefig("fig02_task_structure.png", dpi=150)
plt.close()

# %% [markdown]
# ## Discrete Reach-Direction Tuning (Cosine Model)
#
# For every unit, we compute the mean firing rate during the movement epoch
# (`move_begins_time` to `move_ends_time`) of each straight reach, then fit
# the classic cosine tuning model of Georgopoulos et al. (1982):
#
# $$ r(\theta) = b_0 + b_c \cos\theta + b_s \sin\theta $$
#
# The preferred direction is $\mathrm{atan2}(b_s, b_c)$ and the modulation
# depth is $\sqrt{b_c^2 + b_s^2}$.

# %%
move_ep_straight = nap.IntervalSet(start=mb_s, end=me_s)
counts_straight = units.count(ep=move_ep_straight)
durations_straight = me_s - mb_s
rates_straight = counts_straight.values / durations_straight[:, None]

theta = np.radians(angles_deg)
X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
beta, *_ = np.linalg.lstsq(X, rates_straight, rcond=None)
pred = X @ beta
ss_res = np.sum((rates_straight - pred) ** 2, axis=0)
ss_tot = np.sum((rates_straight - rates_straight.mean(axis=0)) ** 2, axis=0)
r2_cosine = 1 - ss_res / ss_tot
pref_dir_cosine = np.degrees(np.arctan2(beta[2], beta[1]))
depth_cosine = np.sqrt(beta[1] ** 2 + beta[2] ** 2)

print(f"Cosine-model R^2 across all {len(units)} units: "
      f"median={np.median(r2_cosine):.3f}, max={np.max(r2_cosine):.3f}")
print(f"Units with R^2 > 0.3: {np.sum(r2_cosine > 0.3)} "
      f"({np.sum(r2_cosine[is_m1] > 0.3)} in M1, "
      f"{np.sum(r2_cosine[~is_m1] > 0.3)} in PMd)")

# %%
top6 = np.argsort(-r2_cosine)[:6]
theta_fine = np.linspace(-np.pi, np.pi, 200)
fig, axes = plt.subplots(2, 3, figsize=(13, 10), subplot_kw={"projection": "polar"})
for ax, u in zip(axes.flat, top6):
    ax.scatter(theta, rates_straight[:, u], s=14, color="#1f77b4", alpha=0.5,
               label="single trial")
    fit_curve = beta[0, u] + beta[1, u] * np.cos(theta_fine) + beta[2, u] * np.sin(theta_fine)
    fit_curve = np.clip(fit_curve, 0, None)
    ax.plot(theta_fine, fit_curve, color="crimson", lw=2, label="cosine fit")
    area = "M1" if is_m1[u] else "PMd"
    ax.set_title(f"unit {units.index[u]} ({area})\n"
                 f"$R^2$={r2_cosine[u]:.2f}, PD={pref_dir_cosine[u]:.0f}$^\\circ$",
                 fontsize=10, pad=30)
    ax.set_theta_zero_location("E")
axes.flat[0].legend(loc="upper right", bbox_to_anchor=(1.35, 1.3), fontsize=8)
fig.suptitle("Reach-direction tuning during movement epoch (top 6 cosine-fit units)",
             fontsize=13)
plt.subplots_adjust(hspace=0.7, wspace=0.4, top=0.88, bottom=0.05)
plt.savefig("fig03_direction_tuning_polar.png", dpi=150)
plt.close()

# %%
fig = plt.figure(figsize=(11, 4.5))
ax0 = fig.add_subplot(1, 2, 1)
ax0.hist(r2_cosine[is_m1], bins=20, range=(0, 0.6), alpha=0.7, label="M1", color="#d62728")
ax0.hist(r2_cosine[~is_m1], bins=20, range=(0, 0.6), alpha=0.7, label="PMd", color="#2ca02c")
ax0.set_xlabel("cosine-tuning $R^2$")
ax0.set_ylabel("# units")
ax0.set_title("Direction-tuning strength (discrete targets)")
ax0.legend()

well_tuned = r2_cosine > 0.2
ax1 = fig.add_subplot(1, 2, 2, projection="polar")
ax1.scatter(np.radians(pref_dir_cosine[well_tuned & is_m1]),
            r2_cosine[well_tuned & is_m1], color="#d62728", label="M1", s=30)
ax1.scatter(np.radians(pref_dir_cosine[well_tuned & ~is_m1]),
            r2_cosine[well_tuned & ~is_m1], color="#2ca02c", label="PMd", s=30)
ax1.set_theta_zero_location("E")
ax1.set_title("Preferred directions ($R^2$>0.2)", pad=30)
ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.2), fontsize=8)
plt.tight_layout()
plt.savefig("fig04_direction_tuning_population.png", dpi=150)
plt.close()

# %% [markdown]
# The population of well-tuned units spans a broad range of preferred
# directions, consistent with a distributed population code for reach
# direction rather than a small set of "grandmother" direction cells.

# %% [markdown]
# ## Continuous Movement-Direction and Speed Tuning
#
# The discrete analysis above only uses 348 straight trials. To get a much
# larger, continuous sample and to directly test **speed** (not just target
# direction), we compute the instantaneous hand velocity vector across *all*
# successful reach epochs (including the curved, obstacle trials) and relate
# firing rate to instantaneous movement direction and speed using Pynapple
# tuning curves.

# %%
mb_all = trials["move_begins_time"].values[good_idx]
me_all = trials["move_ends_time"].values[good_idx]
valid = np.isfinite(mb_all) & np.isfinite(me_all) & (me_all > mb_all)
mb_all, me_all = mb_all[valid], me_all[valid]
move_ep_all = nap.IntervalSet(start=mb_all, end=me_all)

hand_move = hand.restrict(move_ep_all)
vel = hand_move.derivative(ep=move_ep_all)
speed_vals = np.sqrt(vel["x"].values ** 2 + vel["y"].values ** 2)
dir_vals = np.degrees(np.arctan2(vel["y"].values, vel["x"].values))

speed_tsd = nap.Tsd(t=vel.t, d=speed_vals, time_support=move_ep_all)
dir_tsd = nap.Tsd(t=vel.t, d=dir_vals, time_support=move_ep_all)

print(f"Continuous movement samples: {len(speed_tsd)} across {len(move_ep_all)} epochs")
print(f"Speed (mm/s): median={np.median(speed_vals):.0f}, "
      f"95th pct={np.percentile(speed_vals, 95):.0f}, max={np.max(speed_vals):.0f}")

# %%
tc_dir = nap.compute_tuning_curves(units, dir_tsd, bins=12, range=[(-180, 180)],
                                    epochs=move_ep_all)
tc_speed = nap.compute_tuning_curves(units, speed_tsd, bins=12,
                                      range=[(0, np.percentile(speed_vals, 99))],
                                      epochs=move_ep_all)

bin_centers_dir = tc_dir.coords[tc_dir.dims[1]].values
theta_bins = np.radians(bin_centers_dir)
tc_dir_vals = tc_dir.values
vec_x = np.sum(tc_dir_vals * np.cos(theta_bins), axis=1)
vec_y = np.sum(tc_dir_vals * np.sin(theta_bins), axis=1)
mean_rate_dir = np.sum(tc_dir_vals, axis=1)
r_len = np.sqrt(vec_x ** 2 + vec_y ** 2) / mean_rate_dir
pref_dir_vel = np.degrees(np.arctan2(vec_y, vec_x))

bin_centers_speed = tc_speed.coords[tc_speed.dims[1]].values
tc_speed_vals = tc_speed.values
speed_corr = np.array([
    np.corrcoef(bin_centers_speed, tc_speed_vals[i])[0, 1]
    for i in range(tc_speed_vals.shape[0])
])

print(f"Velocity-direction resultant length: median={np.median(r_len):.3f}, "
      f"max={np.max(r_len):.3f}")
print(f"Speed-tuning correlation: median={np.median(speed_corr):.3f}, "
      f"units with |r|>0.7: {np.sum(np.abs(speed_corr) > 0.7)}")

# %%
top6_dir_vel = np.argsort(-r_len)[:6]
fig, axes = plt.subplots(2, 3, figsize=(13, 10), subplot_kw={"projection": "polar"})
for ax, u in zip(axes.flat, top6_dir_vel):
    ax.plot(np.concatenate([theta_bins, theta_bins[:1]]),
            np.concatenate([tc_dir_vals[u], tc_dir_vals[u, :1]]),
            color="#1f77b4", lw=2, marker="o", ms=3)
    area = "M1" if is_m1[u] else "PMd"
    ax.set_title(f"unit {units.index[u]} ({area})\n"
                 f"resultant length={r_len[u]:.2f}, PD={pref_dir_vel[u]:.0f}$^\\circ$",
                 fontsize=10, pad=30)
    ax.set_theta_zero_location("E")
fig.suptitle("Continuous movement-direction tuning curves (all successful reaches)",
             fontsize=13)
plt.subplots_adjust(hspace=0.7, wspace=0.4, top=0.88, bottom=0.05)
plt.savefig("fig05_velocity_direction_tuning_polar.png", dpi=150)
plt.close()

# %%
top6_speed = np.argsort(-np.abs(speed_corr))[:6]
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, u in zip(axes.flat, top6_speed):
    ax.plot(bin_centers_speed, tc_speed_vals[u], "o-", color="#ff7f0e")
    area = "M1" if is_m1[u] else "PMd"
    ax.set_title(f"unit {units.index[u]} ({area}), r={speed_corr[u]:.2f}", fontsize=10)
    ax.set_xlabel("speed (mm/s)")
    ax.set_ylabel("firing rate (Hz)")
fig.suptitle("Speed tuning curves (top 6 |correlation| units)", fontsize=13)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig("fig06_speed_tuning_curves.png", dpi=150)
plt.close()

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(r_len[is_m1], bins=20, alpha=0.7, label="M1", color="#d62728")
axes[0].hist(r_len[~is_m1], bins=20, alpha=0.7, label="PMd", color="#2ca02c")
axes[0].set_xlabel("direction resultant vector length")
axes[0].set_ylabel("# units")
axes[0].set_title("Continuous movement-direction tuning strength")
axes[0].legend()

axes[1].hist(speed_corr[is_m1], bins=20, range=(-1, 1), alpha=0.7, label="M1", color="#d62728")
axes[1].hist(speed_corr[~is_m1], bins=20, range=(-1, 1), alpha=0.7, label="PMd", color="#2ca02c")
axes[1].axvline(0, color="k", lw=1)
axes[1].set_xlabel("firing-rate vs. speed correlation")
axes[1].set_ylabel("# units")
axes[1].set_title("Speed-tuning strength")
axes[1].legend()
plt.tight_layout()
plt.savefig("fig07_population_summary.png", dpi=150)
plt.close()

# %% [markdown]
# ## Summary
#
# - Using a subset of 348 straight, single-target reaches, we fit a classic
#   cosine tuning model to each unit's movement-epoch firing rate and found a
#   subpopulation of M1 and PMd neurons with clear directional tuning
#   (up to $R^2 \\approx 0.5$), with preferred directions distributed around
#   the full circle, consistent with a population code for reach direction.
# - Using continuous hand-velocity signals across all 1573 successful reaches
#   (including curved, obstacle trials), the same units show consistent
#   tuning to instantaneous movement direction, and a large fraction show
#   strong, often near-monotonic tuning to movement speed (|r| up to ~0.99),
#   consistent with the well-established finding that M1 (and, to a lesser
#   extent, PMd) activity co-varies with both the direction and the speed of
#   an ongoing reach.
# - Both effects are visible in single sessions from a single animal using
#   only openly available DANDI data and standard Pynapple tuning-curve
#   tools, without any simulated or synthetic data.

# %%
io.close()
print("Done. Figures saved:")
for f in [
    "fig01_raw_data_overview.png",
    "fig02_task_structure.png",
    "fig03_direction_tuning_polar.png",
    "fig04_direction_tuning_population.png",
    "fig05_velocity_direction_tuning_polar.png",
    "fig06_speed_tuning_curves.png",
    "fig07_population_summary.png",
]:
    print(" -", f)
