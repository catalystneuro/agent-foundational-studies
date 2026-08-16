# %% [markdown]
# # Reach Direction and Velocity Tuning in Primary Motor Cortex
#
# This notebook demonstrates two classic properties of primary motor cortex (M1)
# neurons during arm reaching: **directional tuning** (cosine-shaped modulation of
# firing rate with movement direction, Georgopoulos et al. 1982) and **speed/velocity
# tuning** (monotonic modulation of firing rate with movement speed, Moran & Schwartz
# 1999).
#
# ## Dataset
#
# We use Dandiset **000129** ("MC_RTT": *Macaque motor cortex spiking activity during
# self-paced reaching*), part of the Neural Latents Benchmark, contributed by the
# Sabes lab (O'Doherty, Cardoso, Makin). A rhesus macaque ("Indy") made continuous,
# self-paced reaches with a manipulandum to move a cursor to targets that appeared at
# random locations on a grid, with no inter-trial delay. The NWB file contains:
# - `units`: 130 sorted single/multi units recorded from a multi-electrode array in M1
# - `cursor_pos`, `finger_pos`: 2D/3D hand kinematics sampled at 1 kHz
# - `target_pos`: the on-screen target location, which changes each time a new target
#   is acquired
#
# We compute cursor velocity from position, derive instantaneous movement direction
# and speed, and relate them to single-unit firing rates using two independent
# approaches: (1) continuous tuning curves over the whole session and (2) discrete,
# per-reach firing rates keyed to individual target acquisitions. Agreement between
# the two approaches is used as an internal validation of the tuning estimates.

# %%
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, circmean

nap.nap_config.suppress_conversion_warnings = True
np.random.seed(0)

FIGDIR = "."

# %% [markdown]
# ## Load the NWB File
#
# We stream the file directly from the DANDI S3 bucket using `remfile`, with a local
# disk cache so repeated reads (e.g. re-running cells) do not re-download data.

# %%
S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/d4c/e0b/d4ce0ba6-a2fa-49d2-a724-bc79e3aebe0f"
# DANDI:000129/draft, sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print(nwb)
print()
print("Session:", nwbfile.session_description)
print("Subject:", nwbfile.subject.subject_id, nwbfile.subject.species, nwbfile.subject.age)

# %%
units = nwb["units"]
cursor_pos = nwb["cursor_pos"]
target_pos = nwb["target_pos"]

print(f"Number of units: {len(units)}")
print(f"Firing rate range: {units.rates.min():.2f} - {units.rates.max():.2f} Hz "
      f"(mean {units.rates.mean():.2f} Hz)")
print(f"cursor_pos: {cursor_pos.shape}, columns={list(cursor_pos.columns)}")
print(f"Session duration: {cursor_pos.index.max():.1f} s")

# %% [markdown]
# ## Inspect Raw Data Streams
#
# Before any analysis, we visualize a short snippet of the raw cursor trajectory and
# spike rasters to confirm the data look sensible.

# %%
cursor_pos = cursor_pos.dropna()  # drop ~300 samples (3 short gaps) with NaNs

t0, t1 = 100.0, 115.0
window = nap.IntervalSet(start=t0, end=t1)
cursor_win = cursor_pos.restrict(window)
units_win = units.restrict(window)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                          gridspec_kw={"height_ratios": [1, 1.6]})

axes[0].plot(cursor_win.index.values, cursor_win.values[:, 0], label="x", lw=1)
axes[0].plot(cursor_win.index.values, cursor_win.values[:, 1], label="y", lw=1)
axes[0].set_ylabel("Cursor position\n(mm)")
axes[0].legend(loc="upper right", frameon=False)
axes[0].set_title("Raw data validation: cursor position and population spike raster")

unit_ids_all = np.array(list(units.keys()))
sorted_ids = unit_ids_all[np.argsort(-units.rates)][:40]
for i, uid in enumerate(sorted_ids):
    spk = units_win[uid].index.values
    axes[1].vlines(spk, i, i + 0.8, color="k", lw=0.6)
axes[1].set_ylabel("Unit # (40 highest-rate\nunits shown)")
axes[1].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig1_raw_data_validation.png", dpi=150)
plt.close()
print("saved fig1_raw_data_validation.png")

# %% [markdown]
# ## Compute Velocity, Speed, and Direction
#
# Movement velocity is the numerical derivative of cursor position. We apply light
# Gaussian smoothing (20 ms std) before computing speed and direction, since raw
# finite-difference velocity at 1 kHz is dominated by tracking jitter, especially
# while the hand is nearly stationary.

# %%
velocity = cursor_pos.derivative()
velocity = velocity.smooth(std=0.02, size_factor=6)

speed = np.sqrt(np.sum(velocity.values ** 2, axis=1))
speed = nap.Tsd(t=velocity.index.values, d=speed, time_support=velocity.time_support)

direction = np.arctan2(velocity.values[:, 1], velocity.values[:, 0])
direction = (direction + 2 * np.pi) % (2 * np.pi)
direction = nap.Tsd(t=velocity.index.values, d=direction, time_support=velocity.time_support)

print(f"speed: min={speed.values.min():.1f}, median={np.median(speed.values):.1f}, "
      f"max={speed.values.max():.1f} (arbitrary cursor units / s)")

# %%
speed_win = speed.restrict(window)
dir_win = direction.restrict(window)

fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
axes[0].plot(speed_win.index.values, speed_win.values, color="tab:blue", lw=1)
axes[0].set_ylabel("Speed\n(units/s)")
axes[0].set_title("Derived kinematics: speed and movement direction")
axes[1].plot(dir_win.index.values, np.degrees(dir_win.values), '.', ms=2, color="tab:orange")
axes[1].set_ylabel("Direction\n(deg)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylim(0, 360)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig2_kinematics.png", dpi=150)
plt.close()
print("saved fig2_kinematics.png")

# %% [markdown]
# ## Define Movement Epochs
#
# Direction is only meaningful while the hand is actually moving (near zero speed,
# the angle is dominated by noise). We threshold the speed trace to isolate movement
# bouts and restrict directional-tuning analysis to those epochs. Speed tuning uses
# the full continuous trace (including near-zero speeds) since speed itself is
# well-defined at rest.

# %%
SPEED_THRESHOLD = 30.0
moving_ep = speed.threshold(SPEED_THRESHOLD, method="above").time_support
moving_ep = moving_ep.drop_short_intervals(0.05)

frac_moving = np.sum(moving_ep.end - moving_ep.start) / cursor_pos.index.max()
print(f"Movement epochs: {len(moving_ep)} bouts, "
      f"{np.sum(moving_ep.end - moving_ep.start):.1f} s total "
      f"({100*frac_moving:.1f}% of session)")

spikes_moving = units.restrict(moving_ep)
direction_moving = direction.restrict(moving_ep)

# %% [markdown]
# ## Directional Tuning (Continuous Approach)
#
# For each unit we compute a tuning curve of firing rate vs. instantaneous movement
# direction (24 bins over 0-2π), restricted to movement epochs, using
# `nap.compute_tuning_curves`. We then fit each tuning curve with a cosine function
# following Georgopoulos et al. (1982):
#
# $$f(\theta) = b_0 + b_1 \cos\theta + b_2 \sin\theta$$
#
# The preferred direction (PD) is $\mathrm{atan2}(b_2, b_1)$ and tuning strength is
# the fit $R^2$.

# %%
N_DIR_BINS = 24
tc_dir = nap.compute_tuning_curves(
    spikes_moving, direction_moving, bins=N_DIR_BINS,
    range=[(0, 2 * np.pi)], feature_names=["direction"],
)
angles = tc_dir.direction.values


def cosine_fn(theta, b0, b1, b2):
    return b0 + b1 * np.cos(theta) + b2 * np.sin(theta)


dir_tuning_results = []
for uid in tc_dir.unit.values:
    rates = tc_dir.sel(unit=uid).values
    if np.any(np.isnan(rates)):
        continue
    popt, _ = curve_fit(cosine_fn, angles, rates)
    b0, b1, b2 = popt
    pd = np.arctan2(b2, b1) % (2 * np.pi)
    amp = np.sqrt(b1 ** 2 + b2 ** 2)
    pred = cosine_fn(angles, *popt)
    ss_res = np.sum((rates - pred) ** 2)
    ss_tot = np.sum((rates - rates.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dir_tuning_results.append(dict(unit=uid, pd=pd, amplitude=amp, r2=r2,
                                    mean_rate=rates.mean(), tuning_curve=rates))

dir_tuning_results.sort(key=lambda d: -d["r2"])
r2_all = np.array([d["r2"] for d in dir_tuning_results])
print(f"Cosine-fit R^2 across {len(r2_all)} units: "
      f"median={np.median(r2_all):.2f}, n(R2>0.3)={np.sum(r2_all>0.3)}, "
      f"n(R2>0.5)={np.sum(r2_all>0.5)}")

# %%
fig, axes = plt.subplots(1, 6, subplot_kw={"projection": "polar"}, figsize=(18, 3.4))
for i, d in enumerate(dir_tuning_results[:6]):
    theta_plot = np.append(angles, angles[0])
    r_plot = np.append(d["tuning_curve"], d["tuning_curve"][0])
    axes[i].plot(theta_plot, r_plot, color="tab:blue", lw=1.5)
    axes[i].plot([d["pd"], d["pd"]], [0, r_plot.max()], color="tab:red", lw=1.5, ls="--")
    axes[i].set_title(f"unit {d['unit']}\nPD={np.degrees(d['pd']):.0f}°, "
                       f"R²={d['r2']:.2f}", fontsize=9, pad=22)
    axes[i].set_yticklabels([])
fig.suptitle("Directional tuning curves, top 6 units (movement direction, continuous)", y=1.12)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig3_direction_tuning_examples.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig3_direction_tuning_examples.png")

# %% [markdown]
# ## Population Summary of Directional Tuning
#
# We summarize direction tuning across the population two ways: (1) a heatmap of
# every unit's normalized tuning curve, sorted by preferred direction, which should
# show a diagonal band if PDs are distributed and tuning is genuine; and (2) a rose
# histogram of preferred directions across the population.

# %%
pds = np.array([d["pd"] for d in dir_tuning_results])
r2s = np.array([d["r2"] for d in dir_tuning_results])

# restrict the sorted heatmap to well-tuned units (R^2>0.3): including flat,
# untuned units would inject noise into the sort order and obscure the diagonal
# structure that reflects genuine, distributed preferred directions.
well_tuned_results = [d for d in dir_tuning_results if d["r2"] > 0.3]
well_tuned_sorted = sorted(well_tuned_results, key=lambda d: d["pd"])
heatmap = np.array([d["tuning_curve"] / d["tuning_curve"].max()
                     for d in well_tuned_sorted])

fig = plt.figure(figsize=(13, 5.5))
gs = GridSpec(1, 2, width_ratios=[1.3, 1], figure=fig)

ax0 = fig.add_subplot(gs[0])
im = ax0.imshow(heatmap, aspect="auto", cmap="viridis",
                 extent=[0, 360, len(heatmap), 0])
ax0.set_xlabel("Movement direction (deg)")
ax0.set_ylabel("Unit (sorted by preferred direction)")
ax0.set_title(f"Normalized directional tuning,\nwell-tuned units (n={len(heatmap)}, R²>0.3)")
plt.colorbar(im, ax=ax0, label="Normalized rate", fraction=0.046, pad=0.04)

ax1 = fig.add_subplot(gs[1], projection="polar")
well_tuned = r2s > 0.3
ax1.hist(pds[well_tuned], bins=18, color="tab:blue", alpha=0.8)
ax1.set_title(f"Preferred directions\n(n={well_tuned.sum()} units, R²>0.3)", pad=20)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig4_direction_population_summary.png", dpi=150)
plt.close()
print("saved fig4_direction_population_summary.png")

# %% [markdown]
# ## Speed Tuning (Continuous Approach)
#
# Speed has a highly right-skewed distribution (mostly slow/near-zero periods, a
# long tail of fast movements), so equal-width bins leave the fastest bins with
# almost no samples and unreliable rate estimates. We instead use quantile bin edges
# (20 bins, equal occupancy) computed over the full session, including near-zero
# speeds.

# %%
N_SPEED_BINS = 20
speed_edges = np.unique(np.quantile(speed.values, np.linspace(0, 1, N_SPEED_BINS + 1)))
tc_speed = nap.compute_tuning_curves(
    units, speed, bins=[speed_edges], feature_names=["speed"],
)

speed_tuning_results = []
for uid in tc_speed.unit.values:
    rates = tc_speed.sel(unit=uid).values
    valid = ~np.isnan(rates)
    rho, p = spearmanr(tc_speed.speed.values[valid], rates[valid])
    speed_tuning_results.append(dict(unit=uid, rho=rho, p=p, tuning_curve=rates))

speed_tuning_results.sort(key=lambda d: -abs(d["rho"]))
rhos = np.array([d["rho"] for d in speed_tuning_results])
print(f"Speed-rate Spearman rho across {len(rhos)} units: "
      f"median={np.median(rhos):.2f}, "
      f"n(positive)={np.sum(rhos>0)}, n(negative)={np.sum(rhos<0)}, "
      f"n(|rho|>0.5)={np.sum(np.abs(rhos)>0.5)}")

# %%
pos_examples = [d for d in speed_tuning_results if d["rho"] > 0][:3]
neg_examples = [d for d in speed_tuning_results if d["rho"] < 0][:3]

fig, axes = plt.subplots(1, 6, figsize=(18, 3))
for i, d in enumerate(pos_examples + neg_examples):
    axes[i].plot(tc_speed.speed.values, d["tuning_curve"], "o-", ms=4, color="tab:blue")
    axes[i].set_title(f"unit {d['unit']}\nρ={d['rho']:.2f}", fontsize=9)
    axes[i].set_xlabel("Speed")
    if i == 0:
        axes[i].set_ylabel("Firing rate (Hz)")
fig.suptitle("Speed tuning curves: 3 positively- and 3 negatively-tuned example units", y=1.05)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig5_speed_tuning_examples.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig5_speed_tuning_examples.png")

# %%
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.hist(rhos, bins=25, color="tab:purple", alpha=0.8, edgecolor="white")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("Spearman correlation (firing rate vs. speed)")
ax.set_ylabel("Number of units")
ax.set_title(f"Population speed tuning (n={len(rhos)} units)")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig6_speed_population_summary.png", dpi=150)
plt.close()
print("saved fig6_speed_population_summary.png")

# %% [markdown]
# ## Discrete Reach-Based Validation
#
# As an independent check, we segment the session into individual "reaches" defined
# by target acquisitions: whenever `target_pos` changes, we treat that as the onset
# of a new reach. The reach direction is defined geometrically as the vector from the
# cursor's position at target onset to the new target location (the direction the
# animal needed to move to acquire the target), and firing rate is computed in a
# 0-600 ms window after onset (truncated if the next target appears sooner). We then
# bin reaches into 8 direction octants and compare the resulting discrete preferred
# direction to the continuous, velocity-based preferred direction computed above.

# %%
tp = np.asarray(target_pos.values)
tt = target_pos.index.values
target_changed = np.where(np.any(np.abs(np.diff(tp, axis=0)) > 1e-9, axis=1))[0]
onset_idx = target_changed + 1
onset_times = tt[onset_idx]
new_targets = tp[onset_idx]

start_pos = nap.Ts(t=onset_times).value_from(cursor_pos)
reach_vec = new_targets - start_pos.values
reach_dir = np.arctan2(reach_vec[:, 1], reach_vec[:, 0])
reach_dir = (reach_dir + 2 * np.pi) % (2 * np.pi)

next_onset = np.append(onset_times[1:], onset_times[-1] + 1.0)
reach_end = np.minimum(next_onset, onset_times + 0.6)
valid = reach_end > onset_times + 0.05

reach_ep = nap.IntervalSet(start=onset_times[valid], end=reach_end[valid])
reach_dir = reach_dir[valid]
print(f"Segmented {len(reach_ep)} discrete reaches from target-acquisition events")

reach_counts = units.count(ep=reach_ep)
reach_durations = (reach_ep.end - reach_ep.start)
reach_rates = reach_counts.values / reach_durations[:, None]
reach_unit_ids = reach_counts.columns.values

N_OCTANTS = 8
octant_edges = np.linspace(0, 2 * np.pi, N_OCTANTS + 1)
octant_idx = np.clip(np.digitize(reach_dir, octant_edges) - 1, 0, N_OCTANTS - 1)
octant_centers = (octant_edges[:-1] + octant_edges[1:]) / 2

discrete_pd = {}
for j, uid in enumerate(reach_unit_ids):
    octant_rates = np.array([reach_rates[octant_idx == b, j].mean()
                              if np.any(octant_idx == b) else np.nan
                              for b in range(N_OCTANTS)])
    if np.any(np.isnan(octant_rates)):
        continue
    popt, _ = curve_fit(cosine_fn, octant_centers, octant_rates)
    b0, b1, b2 = popt
    discrete_pd[uid] = np.arctan2(b2, b1) % (2 * np.pi)

continuous_pd = {d["unit"]: d["pd"] for d in dir_tuning_results if d["r2"] > 0.3}
common_units = sorted(set(discrete_pd) & set(continuous_pd))
cont_vals = np.array([continuous_pd[u] for u in common_units])
disc_vals = np.array([discrete_pd[u] for u in common_units])
angular_diff = np.abs(np.angle(np.exp(1j * (cont_vals - disc_vals))))
print(f"Compared {len(common_units)} well-tuned units (continuous R²>0.3): "
      f"median angular difference = {np.degrees(np.median(angular_diff)):.1f} deg")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].scatter(np.degrees(cont_vals), np.degrees(disc_vals), s=25, alpha=0.7, color="tab:blue")
axes[0].plot([0, 360], [0, 360], "k--", lw=1, alpha=0.6)
axes[0].set_xlabel("Preferred direction, continuous velocity method (deg)")
axes[0].set_ylabel("Preferred direction, discrete reach method (deg)")
axes[0].set_title(f"Cross-method agreement (n={len(common_units)} units)\n"
                   f"median offset = {np.degrees(np.median(angular_diff)):.0f}°")
axes[0].set_xlim(0, 360)
axes[0].set_ylim(0, 360)

axes[1].hist(np.degrees(angular_diff), bins=20, color="tab:green", alpha=0.8)
axes[1].set_xlabel("Angular difference between methods (deg)")
axes[1].set_ylabel("Number of units")
axes[1].set_title("Distribution of cross-method PD offset")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig7_discrete_vs_continuous_validation.png", dpi=150)
plt.close()
print("saved fig7_discrete_vs_continuous_validation.png")

# %% [markdown]
# ## Peri-Reach Rasters by Direction
#
# For one strongly direction-tuned unit, we plot spike rasters aligned to reach onset
# (target acquisition), grouped into 4 direction quadrants, to visualize directional
# selectivity directly in the spike trains rather than through a binned tuning curve.

# %%
example_unit = dir_tuning_results[0]["unit"]  # highest R^2 from continuous fit
print(f"Example unit for peri-reach raster: {example_unit} "
      f"(PD={np.degrees(dir_tuning_results[0]['pd']):.0f} deg, "
      f"R2={dir_tuning_results[0]['r2']:.2f})")

quad_edges = np.linspace(0, 2 * np.pi, 5)
quad_labels = ["0-90°", "90-180°", "180-270°", "270-360°"]
quad_idx = np.clip(np.digitize(reach_dir, quad_edges) - 1, 0, 3)

spk_unit = units[example_unit]
perievent = nap.compute_perievent(spk_unit, nap.Ts(t=onset_times[valid]), window=(-0.2, 0.6))

fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})
bin_size = 0.02
psth_bins = np.arange(-0.2, 0.6 + bin_size, bin_size)
for q in range(4):
    trial_idx_in_quad = np.where(quad_idx == q)[0]
    row = 0
    all_spk_t = []
    for ti in trial_idx_in_quad:
        spk_t = perievent[ti].index.values
        axes[0, q].vlines(spk_t, row, row + 0.8, color="k", lw=0.7)
        all_spk_t.append(spk_t)
        row += 1
    axes[0, q].set_title(f"reach dir {quad_labels[q]}\n(n={len(trial_idx_in_quad)} reaches)")
    axes[0, q].axvline(0, color="tab:red", lw=1, ls="--")
    if q == 0:
        axes[0, q].set_ylabel("Reach #")

    all_spk_concat = np.concatenate(all_spk_t) if all_spk_t else np.array([])
    counts, _ = np.histogram(all_spk_concat, bins=psth_bins)
    rate = counts / (len(trial_idx_in_quad) * bin_size) if len(trial_idx_in_quad) else counts
    axes[1, q].bar(psth_bins[:-1], rate, width=bin_size, align="edge", color="tab:blue")
    axes[1, q].axvline(0, color="tab:red", lw=1, ls="--")
    axes[1, q].set_xlabel("Time from reach onset (s)")
    if q == 0:
        axes[1, q].set_ylabel("Rate (Hz)")

fig.suptitle(f"Unit {example_unit}: peri-reach raster and PSTH by reach-direction quadrant", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig8_raster_by_direction.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig8_raster_by_direction.png")

# %% [markdown]
# ## Summary
#
# Print a compact summary of the key statistics reported in the README.

# %%
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Units analyzed: {len(units)}")
print(f"Directional tuning (continuous, cosine fit): "
      f"{np.sum(r2_all>0.3)}/{len(r2_all)} units R²>0.3, "
      f"{np.sum(r2_all>0.5)}/{len(r2_all)} units R²>0.5")
print(f"Speed tuning (continuous, quantile bins): "
      f"{np.sum(np.abs(rhos)>0.5)}/{len(rhos)} units |rho|>0.5 "
      f"({np.sum(rhos>0.5)} positive, {np.sum(rhos<-0.5)} negative)")
print(f"Cross-method PD validation: {len(common_units)} units compared, "
      f"median angular offset {np.degrees(np.median(angular_diff)):.1f} deg")
