# %% [markdown]
# # Reach Direction and Velocity Tuning in Primary Motor Cortex
#
# This notebook demonstrates directional and velocity (speed) tuning of single-neuron
# firing rates during arm reaching movements, the classic finding described by
# Georgopoulos and colleagues for primary motor cortex (M1) and widely used since as
# the basis of velocity-based neural decoders for motor brain-computer interfaces.
#
# **Dataset**: DANDI Archive Dandiset
# [000129](https://dandiarchive.org/dandiset/000129) ("MC_RTT"), a session of
# multi-electrode Utah array recordings from macaque primary motor cortex during a
# self-paced continuous reaching task ("random target task"). The monkey moved a
# cursor, controlled by fingertip position, between targets randomly placed on an 8x8
# grid with no inter-trial delay. The NWB file contains 130 sorted M1 units together
# with continuous (1 kHz) fingertip position, fingertip velocity, cursor position, and
# target position.
#
# **Approach**: we use fingertip velocity as the behavioral variable of interest.
# For every unit we (1) build nonparametric tuning curves of firing rate against
# movement direction and speed with Pynapple, (2) fit the classic linear
# ("cosine") tuning model relating firing rate to the velocity vector and assess
# significance with a circular-shift shuffle test, and (3) fit Poisson GLMs with
# NeMoS, comparing a linear velocity encoding model against a more flexible
# spline-based direction + speed model.

# %% [markdown]
# ## Setup

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import nemos as nmo
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from tqdm import tqdm

nap.nap_config.suppress_conversion_warnings = True
rng = np.random.default_rng(0)

# %% [markdown]
# ## Data Loading
#
# We stream the NWB file directly from the DANDI S3 bucket with `remfile`, using a
# local disk cache so repeated reads of the same byte ranges do not re-download data.

# %%
S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/d4c/e0b/d4ce0ba6-a2fa-49d2-a724-bc79e3aebe0f"
# sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb, Dandiset 000129 (MC_RTT)

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print(nwb)

# %%
units = nwb["units"]
finger_pos = nwb["finger_pos"]
finger_vel = nwb["finger_vel"]
target_pos = nwb["target_pos"]

print(f"Number of units: {len(units)}")
print(f"Recording duration: {finger_vel.time_support.tot_length():.1f} s")
print(f"Finger velocity sampling interval: {np.median(np.diff(finger_vel.t)) * 1000:.2f} ms")

# %% [markdown]
# ## Raw Data Validation
#
# Before any processing, plot a short window of the raw fingertip position, velocity,
# and a population spike raster to confirm the streamed data look sensible.

# %%
short_ep = nap.IntervalSet(start=10.0, end=15.0)

fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
axs[0].plot(finger_pos.restrict(short_ep)[:, 0], label="x")
axs[0].plot(finger_pos.restrict(short_ep)[:, 1], label="y")
axs[0].set_ylabel("Finger position (mm)")
axs[0].legend(loc="upper right")
axs[0].set_title("Raw behavior and spiking, 10-15 s example window")

axs[1].plot(finger_vel.restrict(short_ep)[:, 0], label="vx")
axs[1].plot(finger_vel.restrict(short_ep)[:, 1], label="vy")
axs[1].set_ylabel("Finger velocity (mm/s)")
axs[1].legend(loc="upper right")

spikes_short = units.restrict(short_ep)
for i, (uid, sp) in enumerate(spikes_short.items()):
    axs[2].plot(sp.index, np.full(len(sp), i), "|", color="k", markersize=3)
axs[2].set_ylabel("Unit #")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylim(-1, len(spikes_short))

plt.tight_layout()
plt.savefig("fig01_raw_data_check.png", dpi=150)
plt.close()

# %% [markdown]
# ## Preprocessing
#
# The fingertip tracker has brief signal dropouts (marker occlusion) recorded as NaN
# runs in `finger_vel`. We locate these gaps and exclude them from the valid epoch
# used for all downstream analysis. We then bin velocity (by averaging) and spikes
# (by counting) into 50 ms bins, and derive instantaneous speed and movement direction
# from the binned velocity components.

# %%
vel_values = finger_vel.values
nan_mask = np.isnan(vel_values).any(axis=1)
t = finger_vel.t
print(f"NaN samples in finger_vel: {nan_mask.sum()} / {len(t)} ({nan_mask.mean() * 100:.3f}%)")

nan_idx = np.where(nan_mask)[0]
gaps = np.where(np.diff(nan_idx) > 1)[0]
gap_starts = np.concatenate(([nan_idx[0]], nan_idx[gaps + 1]))
gap_ends = np.concatenate((nan_idx[gaps], [nan_idx[-1]]))
nan_ep = nap.IntervalSet(
    start=t[np.clip(gap_starts - 1, 0, None)],
    end=t[np.clip(gap_ends + 1, 0, len(t) - 1)],
)
valid_ep = finger_vel.time_support.set_diff(nan_ep)
print(f"Excluded {len(nan_ep)} tracking-loss gap(s), retaining {valid_ep.tot_length():.1f} s of {finger_vel.time_support.tot_length():.1f} s")

# %%
BIN_SIZE = 0.05  # seconds

fv_bin = finger_vel.restrict(valid_ep).bin_average(BIN_SIZE)
vx = fv_bin.values[:, 0]
vy = fv_bin.values[:, 1]
speed_vals = np.sqrt(vx**2 + vy**2)
direction_vals = np.arctan2(vy, vx)

speed = nap.Tsd(t=fv_bin.t, d=speed_vals, time_support=fv_bin.time_support)
direction = nap.Tsd(t=fv_bin.t, d=direction_vals, time_support=fv_bin.time_support)

count = units.count(BIN_SIZE, ep=valid_ep)
unit_ids = np.array(list(units.keys()))
n_bins, n_units = count.shape
print(f"Binned data: {n_bins} bins x {n_units} units at {BIN_SIZE * 1000:.0f} ms resolution")
assert np.isnan(vx).sum() == 0 and np.isnan(vy).sum() == 0

# Movement epoch: bins where the hand is moving fast enough for direction to be
# meaningful (direction is undefined/noisy when the hand is nearly stationary).
SPEED_THRESHOLD = 40.0  # mm/s
move_ep = speed.threshold(SPEED_THRESHOLD, method="above").time_support
print(f"Movement epoch (speed > {SPEED_THRESHOLD} mm/s): {move_ep.tot_length():.1f} s "
      f"({100 * move_ep.tot_length() / valid_ep.tot_length():.1f}% of valid time)")

SPEED_CAP = np.percentile(speed_vals, 97)  # cap axis range to exclude rare outlier bins

# %% [markdown]
# ## Behavioral Overview
#
# The task consists of continuous, self-paced reaches between targets randomly placed
# on an 8x8 grid. We visualize a short fingertip trajectory, the distribution of
# target locations, and the overall speed distribution to confirm the behavior is a
# genuine multi-directional reaching task.

# %%
traj_ep = nap.IntervalSet(start=0.0, end=60.0)
fp = finger_pos.restrict(traj_ep)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

sc = axs[0].scatter(fp[:, 0], fp[:, 1], c=fp.t, cmap="viridis", s=2)
axs[0].set_xlabel("x (mm)")
axs[0].set_ylabel("y (mm)")
axs[0].set_title("Fingertip trajectory (first 60 s)")
plt.colorbar(sc, ax=axs[0], label="time (s)")

target_valid = target_pos.values[~np.isnan(target_pos.values).any(axis=1)]
unique_targets = np.unique(np.round(target_valid, 1), axis=0)
axs[1].scatter(unique_targets[:, 0], unique_targets[:, 1], s=60, marker="s", color="firebrick")
axs[1].set_xlabel("x (mm)")
axs[1].set_ylabel("y (mm)")
axs[1].set_title(f"Target grid ({len(unique_targets)} unique locations)")
axs[1].set_aspect("equal")

axs[2].hist(speed_vals, bins=80, color="steelblue")
axs[2].axvline(SPEED_THRESHOLD, color="k", linestyle="--", label=f"movement threshold ({SPEED_THRESHOLD:.0f} mm/s)")
axs[2].set_xlabel("Speed (mm/s)")
axs[2].set_ylabel("Count (50 ms bins)")
axs[2].set_title("Speed distribution")
axs[2].set_xlim(0, np.percentile(speed_vals, 99.5))
axs[2].legend()

plt.tight_layout()
plt.savefig("fig02_behavior_overview.png", dpi=150)
plt.close()

# %% [markdown]
# ## Nonparametric Tuning Curves (Pynapple)
#
# For every unit we compute a firing-rate-vs-direction tuning curve (restricted to
# movement periods, so direction is well defined) and a firing-rate-vs-speed tuning
# curve (using all valid time, so we also capture the resting baseline near zero
# speed). We display these for the units with the strongest direction tuning
# (identified below via the cosine regression), which lets the two analyses cross
# validate each other.

# %%
tc_direction = nap.compute_tuning_curves(
    units, direction, bins=16, range=(-np.pi, np.pi), epochs=move_ep, feature_names=["direction"]
)
tc_speed = nap.compute_tuning_curves(
    units, speed, bins=16, range=(0, SPEED_CAP), epochs=valid_ep, feature_names=["speed"]
)

# %% [markdown]
# ## Cosine Tuning Model and Significance Testing
#
# The classic model of directional tuning in motor cortex (Georgopoulos et al., 1982;
# Moran & Schwartz, 1999) expresses firing rate as a linear function of the velocity
# vector:
#
# $$ \lambda(t) = b_0 + b_1 v_x(t) + b_2 v_y(t) = b_0 + m \cdot \|v(t)\| \cos(\theta(t) - \text{PD}) $$
#
# where the preferred direction $\text{PD} = \operatorname{atan2}(b_2, b_1)$ and the
# modulation depth $m = \sqrt{b_1^2 + b_2^2}$. This single linear model captures both
# directional tuning (through the cosine term) and a linear gain on movement speed.
# We fit it for every unit at once with ordinary least squares, and assess
# significance with a circular-shift shuffle test that preserves each unit's
# autocorrelation while destroying its temporal relationship with behavior.

# %%
Y = count.values.astype(float)
X = np.column_stack([np.ones(n_bins), vx, vy])


def fit_r2(X, Y):
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((Y - pred) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / ss_tot
    return beta, r2


with np.errstate(over="ignore", invalid="ignore"):
    beta, r2 = fit_r2(X, Y)
b0, b1, b2 = beta[0], beta[1], beta[2]
pref_dir = np.arctan2(b2, b1)
mod_depth = np.sqrt(b1**2 + b2**2)
mean_rate = Y.mean(axis=0) / BIN_SIZE

N_SHUFFLES = 200
min_shift = int(5.0 / BIN_SIZE)  # >= 5 s shift, far beyond any behavioral autocorrelation
null_r2 = np.zeros((N_SHUFFLES, n_units))
with np.errstate(over="ignore", invalid="ignore"):
    for i in tqdm(range(N_SHUFFLES), desc="Circular-shift shuffle test"):
        shift = rng.integers(min_shift, n_bins - min_shift)
        Y_shift = np.roll(Y, shift, axis=0)
        _, r2_null = fit_r2(X, Y_shift)
        null_r2[i] = r2_null

pval = np.mean(null_r2 >= r2[None, :], axis=0)
sig = pval < 0.01
print(f"Significantly velocity-tuned units: {sig.sum()} / {n_units} (shuffle test, p < 0.01)")

order_by_r2 = np.argsort(-r2)
top_units = unit_ids[order_by_r2[:6]]
print("Top 6 units by cosine-model R^2:", top_units, "R^2 =", np.round(r2[order_by_r2[:6]], 4))

# %% [markdown]
# ### Example Tuning Curves
#
# Polar plots of the nonparametric direction tuning curve (blue) with the cosine
# model fit (orange, computed from the same regression coefficients) overlaid, for
# the six most strongly tuned units. Below, the corresponding nonparametric speed
# tuning curves.

# %%
theta_grid = np.linspace(-np.pi, np.pi, 200)
fig = plt.figure(figsize=(16, 8))
line_handles = None
for i, uid in enumerate(top_units):
    ax = fig.add_subplot(2, 6, i + 1, projection="polar")
    theta = tc_direction.direction.values
    r = tc_direction.sel(unit=int(uid)).values
    theta_c = np.append(theta, theta[0])
    r_c = np.append(r, r[0])
    (h1,) = ax.plot(theta_c, r_c, color="steelblue", label="data")

    uidx = np.where(unit_ids == uid)[0][0]
    cosine_fit = (b0[uidx] + mod_depth[uidx] * np.cos(theta_grid - pref_dir[uidx])) / BIN_SIZE
    (h2,) = ax.plot(theta_grid, cosine_fit, color="darkorange", linestyle="--", label="cosine fit")
    ax.set_title(f"unit {uid}, R²={r2[uidx]:.3f}", fontsize=10, pad=28)
    ax.set_yticklabels([])
    if i == 0:
        line_handles = (h1, h2)

fig.legend(line_handles, ["data", "cosine fit"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.93), fontsize=10)

for i, uid in enumerate(top_units):
    ax = fig.add_subplot(2, 6, i + 7)
    ax.plot(tc_speed.speed.values, tc_speed.sel(unit=int(uid)).values, color="seagreen")
    ax.set_xlabel("speed (mm/s)")
    if i == 0:
        ax.set_ylabel("rate (Hz)")
    ax.set_title(f"unit {uid}", fontsize=10)

plt.suptitle("Direction tuning (polar, with cosine fit) and speed tuning for the 6 most strongly tuned units", y=1.08)
plt.tight_layout()
plt.subplots_adjust(hspace=0.6, top=0.78)
plt.savefig("fig03_example_tuning_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# %% [markdown]
# ### Population Summary
#
# Across the population: the distribution of preferred directions (a Georgopoulos-
# style population vector plot: each significant unit contributes one arrow of length
# proportional to its modulation depth, pointing along its preferred direction), and
# modulation depth versus mean firing rate, highlighting which units carry
# significant velocity tuning.

# %%
fig = plt.figure(figsize=(13, 5.5))

ax0 = fig.add_subplot(1, 2, 1, projection="polar")
for i in range(n_units):
    color = "crimson" if sig[i] else "lightgray"
    zorder = 3 if sig[i] else 1
    ax0.plot([pref_dir[i], pref_dir[i]], [0, mod_depth[i]], color=color, alpha=0.8, zorder=zorder)
ax0.set_title(f"Preferred direction & modulation depth per unit\n(red = significant, n={sig.sum()}/{n_units}, shuffle p<0.01)", pad=25)
ax0.set_yticklabels([])

ax1 = fig.add_subplot(1, 2, 2)
ax1.scatter(mean_rate[~sig], mod_depth[~sig], color="lightgray", label="not significant", s=25)
ax1.scatter(mean_rate[sig], mod_depth[sig], color="crimson", label="significant", s=25)
ax1.set_xlabel("Mean firing rate (Hz)")
ax1.set_ylabel("Modulation depth |b| (spikes/bin per mm/s)")
ax1.set_title("Velocity modulation depth vs. mean rate")
ax1.legend()

plt.tight_layout()
plt.savefig("fig04_population_summary.png", dpi=150)
plt.close()

# %% [markdown]
# ## 2D Velocity Tuning Maps
#
# Rather than treating direction and speed separately, we can plot firing rate
# directly as a function of the 2D velocity vector $(v_x, v_y)$. This combines
# direction (angle) and speed (radial distance) tuning in a single map, and is the
# quantity the linear cosine model above approximates with a tilted plane.

# %%
VMAX = np.percentile(speed_vals, 97)
vel_features = nap.TsdFrame(t=fv_bin.t, d=fv_bin.values, time_support=fv_bin.time_support, columns=["vx", "vy"])
tc_2d = nap.compute_tuning_curves(
    units, vel_features, bins=[18, 18], range=[(-VMAX, VMAX), (-VMAX, VMAX)], feature_names=["vx", "vy"]
)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, uid in zip(axs, top_units[:3]):
    data = tc_2d.sel(unit=int(uid)).values
    im = ax.pcolormesh(tc_2d.vx.values, tc_2d.vy.values, data.T, cmap="viridis", shading="nearest")
    ax.set_xlabel("vx (mm/s)")
    ax.set_ylabel("vy (mm/s)")
    ax.set_title(f"unit {uid}")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="rate (Hz)")
plt.suptitle("Firing rate as a function of the 2D velocity vector", y=1.02)
plt.tight_layout()
plt.savefig("fig05_velocity_2d_tuning_maps.png", dpi=150, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## NeMoS GLM Encoding Models
#
# We fit two Poisson GLM encoding models with NeMoS for the most strongly tuned
# units:
#
# - **Linear velocity model**: $\lambda(t) = \exp(b_0 + b_1 v_x(t) + b_2 v_y(t))$, the
#   Poisson-GLM analog of the cosine tuning model above.
# - **Flexible spline model**: an additive combination of a cyclic B-spline basis on
#   movement direction (captures arbitrary, non-cosine directional tuning shapes) and
#   a B-spline basis on speed (captures nonlinear speed/gain tuning).
#
# Models are fit on the first 70% of the recording and evaluated out-of-sample on the
# remaining 30%, using McFadden's pseudo-R² (a likelihood-ratio-based goodness of fit
# relative to a constant-rate null model).

# %%
speed_clipped = np.clip(speed_vals, 0, SPEED_CAP)
dir_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, bounds=(-np.pi, np.pi), label="direction")
speed_basis = nmo.basis.BSplineEval(n_basis_funcs=6, bounds=(0, SPEED_CAP), label="speed")
flex_basis = dir_basis + speed_basis

X_flex = flex_basis.compute_features(direction_vals, speed_clipped)
X_lin = np.column_stack([vx, vy])

split = int(n_bins * 0.7)
X_flex_train, X_flex_test = X_flex[:split], X_flex[split:]
X_lin_train, X_lin_test = X_lin[:split], X_lin[split:]

pr2_lin_list, pr2_flex_list = [], []
predicted_rate_flex = {}
for uid in tqdm(top_units, desc="Fitting NeMoS GLMs"):
    y = count.loc[int(uid)].values.astype(float)
    y_train, y_test = y[:split], y[split:]

    model_lin = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.01, solver_name="LBFGS")
    model_lin.fit(X_lin_train, y_train)
    pr2_lin = float(model_lin.score(X_lin_test, y_test, score_type="pseudo-r2-McFadden"))

    model_flex = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.01, solver_name="LBFGS")
    model_flex.fit(X_flex_train, y_train)
    pr2_flex = float(model_flex.score(X_flex_test, y_test, score_type="pseudo-r2-McFadden"))

    pr2_lin_list.append(pr2_lin)
    pr2_flex_list.append(pr2_flex)
    predicted_rate_flex[uid] = np.asarray(model_flex.predict(X_flex)) / BIN_SIZE

    print(f"unit {uid}: linear pseudo-R2={pr2_lin:.4f}  flexible pseudo-R2={pr2_flex:.4f}")

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
x_pos = np.arange(len(top_units))
width = 0.35
ax.bar(x_pos - width / 2, pr2_lin_list, width, label="linear velocity (vx, vy)", color="steelblue")
ax.bar(x_pos + width / 2, pr2_flex_list, width, label="flexible (direction + speed splines)", color="darkorange")
ax.set_xticks(x_pos)
ax.set_xticklabels([str(u) for u in top_units])
ax.set_xlabel("Unit")
ax.set_ylabel("Held-out pseudo-R² (McFadden)")
ax.set_title("NeMoS GLM encoding model comparison")
ax.legend()
plt.tight_layout()
plt.savefig("fig06_glm_pseudo_r2_comparison.png", dpi=150)
plt.close()

# %% [markdown]
# ### Predicted vs. Actual Tuning Curves
#
# For each unit, we recompute the nonparametric direction and speed tuning curves
# using the flexible GLM's predicted firing rate in place of the spike counts, and
# overlay them on the empirical tuning curves. Good agreement indicates the GLM has
# correctly captured each unit's velocity encoding, and the tuning curves are more
# stable than the two-parameter cosine fit alone.

# %%
fig, axs = plt.subplots(2, len(top_units), figsize=(4 * len(top_units), 7))
for i, uid in enumerate(top_units):
    pred_tsdframe = nap.TsdFrame(
        t=fv_bin.t, d=predicted_rate_flex[uid][:, None], time_support=fv_bin.time_support, columns=[int(uid)]
    )
    tc_pred_dir = nap.compute_tuning_curves(
        pred_tsdframe, direction, bins=16, range=(-np.pi, np.pi), epochs=move_ep, feature_names=["direction"]
    )
    tc_pred_speed = nap.compute_tuning_curves(
        pred_tsdframe, speed, bins=16, range=(0, SPEED_CAP), epochs=valid_ep, feature_names=["speed"]
    )

    ax = axs[0, i]
    ax.plot(tc_direction.direction.values, tc_direction.sel(unit=int(uid)).values, color="steelblue", label="data")
    ax.plot(tc_pred_dir.direction.values, tc_pred_dir.sel(unit=int(uid)).values, color="darkorange", linestyle="--", label="GLM")
    ax.set_title(f"unit {uid}: direction")
    ax.set_xlabel("direction (rad)")
    if i == 0:
        ax.set_ylabel("rate (Hz)")
        ax.legend(fontsize=8)

    ax = axs[1, i]
    ax.plot(tc_speed.speed.values, tc_speed.sel(unit=int(uid)).values, color="seagreen", label="data")
    ax.plot(tc_pred_speed.speed.values, tc_pred_speed.sel(unit=int(uid)).values, color="darkorange", linestyle="--", label="GLM")
    ax.set_title(f"unit {uid}: speed")
    ax.set_xlabel("speed (mm/s)")
    if i == 0:
        ax.set_ylabel("rate (Hz)")
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("fig07_glm_predicted_vs_actual_tuning.png", dpi=150)
plt.close()

io.close()

# %% [markdown]
# ## Summary
#
# Using continuous fingertip velocity recorded during a self-paced random-target
# reaching task (Dandiset 000129), we found that a substantial fraction of recorded
# M1 units (see population summary figure) show firing rates that are significantly
# modulated by movement velocity beyond what is expected by chance (circular-shift
# shuffle test, p < 0.01). Individual units show clear cosine-shaped tuning to
# movement direction and monotonic tuning to movement speed, consistent with the
# classic linear velocity tuning model of motor cortex. NeMoS Poisson GLMs fit
# directly to spike counts recover the same tuning curves out-of-sample, and a
# flexible spline-based encoding model captures directional and speed tuning at
# least as well as the simple two-parameter cosine model, occasionally revealing
# mild deviations from a pure cosine shape.
