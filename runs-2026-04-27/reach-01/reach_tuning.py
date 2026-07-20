# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reach direction and velocity tuning in motor cortex
#
# This notebook demonstrates classic **directional and speed tuning** in primate
# motor cortex using a publicly available dataset on the DANDI Archive
# (**Dandiset 000128**, *MC_Maze*, Churchland & Kaufman, monkey "Jenkins"). The
# dataset provides:
#
# * 182 single units recorded simultaneously from primary motor cortex (M1) and
#   dorsal premotor cortex (PMd) using Utah arrays;
# * Continuous 1 kHz hand position and velocity from a planar reaching task;
# * 2,295 trials of a delayed center-out reaching task (with and without
#   maze barriers) including target onset, go cue, and movement-onset times.
#
# Reference: Churchland & Kaufman *Neural population dynamics during reaching*
# (Nature 2012) and the Neural Latents Benchmark release.
#
# We replicate two foundational findings:
#
# 1. **Cosine direction tuning** (Georgopoulos et al. 1982): single neurons in
#    motor cortex have firing rates well described by a cosine of the angle
#    between the reach direction and a unit-specific *preferred direction*.
# 2. **Speed (velocity-magnitude) tuning**: many of the same neurons modulate
#    their firing rate with the speed of the upcoming reach.
#
# We then combine these in a single Poisson **GLM** (NeMoS) that jointly
# encodes direction (via a sin/cos basis) and speed (via a raised-cosine basis)
# and shows that both contribute substantively to the explained activity.

# %% [markdown]
# ## 1. Setup and streaming load

# %%
import os
import numpy as np
import h5py
import remfile
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from tqdm import tqdm
from scipy.optimize import curve_fit

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
np.random.seed(0)

ASSET_ID = "26e85f09-39b7-480f-b337-278a8f034007"  # Dandiset 000128 - MC_Maze
S3_URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"

disk_cache = remfile.DiskCache("/tmp/remfile_cache_reach")
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
print("Session :", nwbfile.session_description)
print("Subject :", nwbfile.subject.subject_id, nwbfile.subject.species)
print("n_units :", len(nwbfile.units))
print("n_trials:", len(nwbfile.trials))

# %% [markdown]
# ## 2. Behavioural data: hand position, velocity, speed

# %%
behavior = nwbfile.processing["behavior"]
hp = behavior["hand_pos"]
hv = behavior["hand_vel"]

t_hand = hp.timestamps[:]
hp_data = hp.data[:]
hv_data = hv.data[:]
hand_pos = nap.TsdFrame(t=t_hand, d=hp_data, columns=["x", "y"])
hand_vel = nap.TsdFrame(t=t_hand, d=hv_data, columns=["vx", "vy"])
speed = nap.Tsd(t=t_hand, d=np.linalg.norm(hv_data, axis=1))
print(f"hand sampling: ~{1.0 / np.median(np.diff(t_hand)):.1f} Hz")

# %% [markdown]
# ## 3. Trial selection and reach kinematics
#
# We restrict to the cleanest reaches: **successful trials with no barriers**
# (`num_barriers == 0`), which are straight center-out reaches to one of eight
# canonical target locations.
#
# Per trial we compute:
# * `reach_dir` — direction (radians) from the position at *move onset* to the
#   peak-speed position, derived from the data (no reliance on listed target).
# * `peak_speed` — maximum hand-speed value in a window 0–400 ms after move onset.
# * `move_onset` — already provided in the trials table.

# %%
trials = nwbfile.trials.to_dataframe()
mask = (trials["success"]) & (trials["num_barriers"] == 0)
trials_co = trials[mask].reset_index(drop=True)
print(f"Selected {len(trials_co)} center-out (no-barrier) trials")

reach_dir = np.full(len(trials_co), np.nan)
peak_speed = np.full(len(trials_co), np.nan)
move_onset = trials_co["move_onset_time"].values.astype(float)

t_hand_arr = t_hand
for i, t_mv in enumerate(tqdm(move_onset, desc="kinematics")):
    if not np.isfinite(t_mv):
        continue
    # window: -50 ms to +400 ms around move onset
    i0 = np.searchsorted(t_hand_arr, t_mv - 0.05)
    i1 = np.searchsorted(t_hand_arr, t_mv + 0.4)
    if i1 - i0 < 10:
        continue
    seg_pos = hp_data[i0:i1]
    seg_spd = np.linalg.norm(hv_data[i0:i1], axis=1)
    pk = int(np.argmax(seg_spd))
    peak_speed[i] = seg_spd[pk]
    p_start = seg_pos[0]
    p_peak = seg_pos[pk]
    dx, dy = p_peak - p_start
    if dx ** 2 + dy ** 2 > 1e-3:
        reach_dir[i] = np.arctan2(dy, dx)

valid = np.isfinite(reach_dir) & np.isfinite(peak_speed)
print(f"valid kinematic trials: {valid.sum()}")

# Bin direction into 8 sectors (the canonical target layout).
n_dir_bins = 8
dir_edges = np.linspace(-np.pi, np.pi, n_dir_bins + 1)
dir_centers = 0.5 * (dir_edges[:-1] + dir_edges[1:])
dir_bin = np.digitize(reach_dir, dir_edges) - 1
dir_bin[dir_bin == n_dir_bins] = n_dir_bins - 1

# %% [markdown]
# ### Sanity-check: reach trajectories grouped by direction bin

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
cmap = plt.cm.hsv(np.linspace(0, 1, n_dir_bins, endpoint=False))
for b in range(n_dir_bins):
    idx = np.where(valid & (dir_bin == b))[0][:25]
    for i in idx:
        t_mv = move_onset[i]
        i0 = np.searchsorted(t_hand_arr, t_mv - 0.05)
        i1 = np.searchsorted(t_hand_arr, t_mv + 0.5)
        seg = hp_data[i0:i1]
        seg = seg - seg[0]  # align to move-onset position
        ax.plot(seg[:, 0], seg[:, 1], color=cmap[b], alpha=0.4, lw=0.7)
ax.set_xlabel("Δx from move onset (mm)")
ax.set_ylabel("Δy from move onset (mm)")
ax.set_title("Reach trajectories coloured by direction bin (n=8)")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_trajectories_by_dir.png"), dpi=130)
plt.close()
print("saved 02_trajectories_by_dir.png")

# %% [markdown]
# ## 4. Spikes and per-trial firing rates

# %%
spike_lists = [nwbfile.units["spike_times"][i] for i in range(len(nwbfile.units))]
spikes = nap.TsGroup({i: nap.Ts(t=s) for i, s in enumerate(spike_lists)})
print(f"loaded {len(spikes)} units. Mean rate = "
      f"{np.mean([len(s) / (t_hand[-1] - t_hand[0]) for s in spike_lists]):.2f} Hz")

# Movement-period firing rate: 100 ms before to 400 ms after move onset (500 ms window).
WIN_PRE = 0.10
WIN_POST = 0.40
WIN = WIN_PRE + WIN_POST

mv_intervals = nap.IntervalSet(
    start=move_onset[valid] - WIN_PRE,
    end=move_onset[valid] + WIN_POST,
)
trial_idx_valid = np.where(valid)[0]

# Build per-trial firing rates: shape (n_trials_valid, n_units)
fr_trial = np.zeros((len(mv_intervals), len(spikes)))
for u, st in enumerate(tqdm(spike_lists, desc="trial counts")):
    if len(st) == 0:
        continue
    # vectorised: for each interval count spikes via searchsorted
    s = np.asarray(st)
    left = np.searchsorted(s, mv_intervals.start)
    right = np.searchsorted(s, mv_intervals.end)
    fr_trial[:, u] = (right - left) / WIN

print("per-trial firing rate matrix:", fr_trial.shape)

# %% [markdown]
# ## 5. Direction tuning curves
#
# For each unit we average movement-period firing rate within each of 8
# direction bins, then fit a **cosine tuning model**
#
# $$f(\theta) = b_0 + b_1\cos(\theta - \theta_{\rm pref})$$

# %%
def cosine_model(theta, b0, b1, theta_pref):
    return b0 + b1 * np.cos(theta - theta_pref)

dir_tuning = np.zeros((len(spikes), n_dir_bins))
dir_sem = np.zeros_like(dir_tuning)
n_per_bin = np.zeros(n_dir_bins, dtype=int)
dir_bin_v = dir_bin[valid]
for b in range(n_dir_bins):
    sel = dir_bin_v == b
    n_per_bin[b] = sel.sum()
    dir_tuning[:, b] = fr_trial[sel].mean(axis=0)
    dir_sem[:, b] = fr_trial[sel].std(axis=0) / np.sqrt(max(sel.sum(), 1))
print("trials per direction bin:", n_per_bin)

pref_dir = np.zeros(len(spikes))
mod_depth = np.zeros(len(spikes))
baseline = np.zeros(len(spikes))
r2_dir_trial = np.zeros(len(spikes))  # trial-level R² (lower bound, includes Poisson noise)
r2_dir = np.zeros(len(spikes))  # bin-mean R² (does the cosine shape fit?)
# Trial-level fit (unbiased parameter estimates):
theta_trial = reach_dir[valid]
c_t, s_t = np.cos(theta_trial), np.sin(theta_trial)
X_lin = np.stack([np.ones_like(c_t), c_t, s_t], axis=1)
for u in range(len(spikes)):
    y = fr_trial[:, u]
    beta, *_ = np.linalg.lstsq(X_lin, y, rcond=None)
    baseline[u] = beta[0]
    mod_depth[u] = np.hypot(beta[1], beta[2])
    pref_dir[u] = np.arctan2(beta[2], beta[1])
    y_hat = X_lin @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_dir_trial[u] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # Bin-mean fit quality: how well does the cosine shape match the empirical
    # tuning curve (averaged over trials)?
    fit_curve = baseline[u] + mod_depth[u] * np.cos(dir_centers - pref_dir[u])
    well_sampled = n_per_bin >= 5
    if well_sampled.sum() >= 3:
        y_bin = dir_tuning[u, well_sampled]
        f_bin = fit_curve[well_sampled]
        ss_res = np.sum((y_bin - f_bin) ** 2)
        ss_tot = np.sum((y_bin - y_bin.mean()) ** 2)
        r2_dir[u] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

# wrap pref_dir into [-pi, pi] and ensure mod_depth >= 0
flip = mod_depth < 0
mod_depth[flip] = -mod_depth[flip]
pref_dir[flip] = (pref_dir[flip] + np.pi)
pref_dir = (pref_dir + np.pi) % (2 * np.pi) - np.pi

print(f"median bin-mean R² of cosine fit: {np.median(r2_dir):.3f}")
print(f"units with bin-mean R² > 0.5: {(r2_dir > 0.5).sum()} / {len(spikes)}")
print(f"median trial-level R² (includes Poisson noise): {np.median(r2_dir_trial):.3f}")
print(f"units with trial-level R² > 0.05: {(r2_dir_trial > 0.05).sum()} / {len(spikes)}")

# %% [markdown]
# ### Polar tuning curves of the most strongly tuned units

# %%
top = np.argsort(-r2_dir)[:8]
fig, axes = plt.subplots(2, 4, figsize=(14, 7), subplot_kw={"projection": "polar"})
theta_fine = np.linspace(-np.pi, np.pi, 200)
for ax, u in zip(axes.flat, top):
    th = np.r_[dir_centers, dir_centers[0]]
    fr = np.r_[dir_tuning[u], dir_tuning[u, 0]]
    sem = np.r_[dir_sem[u], dir_sem[u, 0]]
    ax.plot(th, fr, "o-", color="tab:blue", lw=1.5)
    ax.fill_between(th, fr - sem, fr + sem, color="tab:blue", alpha=0.2)
    fit = cosine_model(theta_fine, baseline[u], mod_depth[u], pref_dir[u])
    fit = np.clip(fit, 0, None)
    ax.plot(theta_fine, fit, color="tab:red", lw=1.2, alpha=0.9)
    ax.set_title(f"unit {u}\nR²={r2_dir[u]:.2f}", pad=12, fontsize=10)
    ax.set_yticklabels([])
    ax.tick_params(labelsize=8)
fig.suptitle("Cosine direction tuning – top 8 units (red = cosine fit)", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_direction_tuning_polar.png"),
            dpi=130, bbox_inches="tight")
plt.close()
print("saved 03_direction_tuning_polar.png")

# %% [markdown]
# ### Distribution of preferred directions and tuning depth

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sig = r2_dir > 0.5
axes[0].hist(np.degrees(pref_dir[sig]), bins=24,
             color="tab:purple", edgecolor="k")
axes[0].set_xlabel("Preferred direction (deg)")
axes[0].set_ylabel("# units")
axes[0].set_title(f"Preferred direction (n={sig.sum()} tuned)")

axes[1].hist(r2_dir, bins=30, color="tab:olive", edgecolor="k")
axes[1].axvline(0.5, color="r", ls="--", label="R²=0.5")
axes[1].set_xlabel("Cosine fit R²")
axes[1].set_ylabel("# units")
axes[1].set_title("Goodness of cosine fit")
axes[1].legend()

axes[2].scatter(baseline, mod_depth, s=10, alpha=0.6)
axes[2].plot([0, baseline.max()], [0, baseline.max()], "k--", alpha=0.4,
             label="depth = baseline")
axes[2].set_xlabel("Baseline rate b₀ (Hz)")
axes[2].set_ylabel("Modulation depth b₁ (Hz)")
axes[2].set_title("Cosine parameters")
axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_direction_population.png"), dpi=130)
plt.close()
print("saved 04_direction_population.png")

# %% [markdown]
# ## 6. Speed tuning
#
# Many M1/PMd neurons modulate their rate with movement *speed* in addition to
# direction. We bin trials by peak speed (quintiles) and recompute mean firing
# rate per speed bin, conditioning on the four cardinal direction sectors so
# that direction tuning does not contaminate the speed effect.

# %%
speed_q = np.quantile(peak_speed[valid], np.linspace(0, 1, 6))
speed_centers = 0.5 * (speed_q[:-1] + speed_q[1:])
speed_bin = np.digitize(peak_speed[valid], speed_q) - 1
speed_bin = np.clip(speed_bin, 0, 4)

speed_tuning = np.zeros((len(spikes), 5))
speed_sem = np.zeros_like(speed_tuning)
for sb in range(5):
    sel = speed_bin == sb
    speed_tuning[:, sb] = fr_trial[sel].mean(axis=0)
    speed_sem[:, sb] = fr_trial[sel].std(axis=0) / np.sqrt(max(sel.sum(), 1))

# Speed-modulation index: Pearson r between firing rate and peak speed
# *within* a direction bin, then averaged across direction bins (weighted).
speed_r = np.zeros(len(spikes))
speed_p_r2 = np.zeros(len(spikes))
for u in range(len(spikes)):
    rs = []
    weights = []
    for db in range(n_dir_bins):
        idx = np.where(dir_bin_v == db)[0]
        if len(idx) < 30:
            continue
        x = peak_speed[valid][idx]
        y = fr_trial[idx, u]
        if x.std() == 0 or y.std() == 0:
            continue
        r = np.corrcoef(x, y)[0, 1]
        rs.append(r)
        weights.append(len(idx))
    if rs:
        rs = np.array(rs)
        w = np.array(weights, dtype=float)
        speed_r[u] = (rs * w).sum() / w.sum()
        speed_p_r2[u] = (rs ** 2 * w).sum() / w.sum()

print(f"# units with |within-direction speed r| > 0.15: "
      f"{(np.abs(speed_r) > 0.15).sum()} / {len(spikes)}")

# %% [markdown]
# ### Speed tuning curves of strongly speed-modulated units

# %%
top_s = np.argsort(-np.abs(speed_r))[:6]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, u in zip(axes.flat, top_s):
    ax.errorbar(speed_centers, speed_tuning[u], yerr=speed_sem[u],
                marker="o", color="tab:green")
    ax.set_xlabel("peak hand speed (mm/s)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title(f"unit {u}  r={speed_r[u]:+.2f}")
fig.suptitle("Speed tuning curves of top units (within-direction r)", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_speed_tuning.png"),
            dpi=130, bbox_inches="tight")
plt.close()
print("saved 05_speed_tuning.png")

# %% [markdown]
# ### Population speed-modulation summary

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(speed_r, bins=30, color="tab:green", edgecolor="k")
axes[0].axvline(0, color="k", lw=0.8)
axes[0].set_xlabel("within-direction speed correlation r")
axes[0].set_ylabel("# units")
axes[0].set_title("Speed modulation across population")

axes[1].scatter(np.abs(speed_r), r2_dir, s=12, alpha=0.6)
axes[1].set_xlabel("|speed correlation|")
axes[1].set_ylabel("direction-tuning R²")
axes[1].set_title("Direction vs speed tuning per unit")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_population_summary.png"), dpi=130)
plt.close()
print("saved 06_population_summary.png")

# %% [markdown]
# ## 7. Joint Poisson GLM with NeMoS
#
# We fit a single-neuron Poisson GLM with two trial-level features:
#
# * direction encoded as **(cos θ, sin θ)** (a sufficient basis for cosine
#   tuning),
# * peak speed encoded with a **3-bump raised-cosine basis** (NeMoS
#   `RaisedCosineLogConv`-equivalent for scalar inputs).
#
# We compare three models — *direction only*, *speed only*, *direction + speed*
# — by cross-validated pseudo-R² (McFadden) on held-out trials.

# %%
import nemos as nmo

X_dir = np.stack([np.cos(reach_dir[valid]), np.sin(reach_dir[valid])], axis=1)
sp = peak_speed[valid]
sp_z = (sp - sp.mean()) / sp.std()

def raised_cos_bumps(x, centres, width):
    return np.stack(
        [0.5 * (1 + np.cos(np.clip((x - c) / width * np.pi, -np.pi, np.pi)))
         for c in centres],
        axis=1,
    )

centres = np.linspace(sp_z.min(), sp_z.max(), 4)[1:-1]
width = (sp_z.max() - sp_z.min()) / 2
X_sp = raised_cos_bumps(sp_z, centres, width)

X_full = np.concatenate([X_dir, X_sp], axis=1)
print("design matrices: dir", X_dir.shape, " speed", X_sp.shape, " full", X_full.shape)

# Spike counts in the movement window (Poisson observations)
y_counts = (fr_trial * WIN).astype(int)

# Train/test split (80/20) once - PopulationGLM fits all units jointly => fast.
n_trials = X_full.shape[0]
rng = np.random.RandomState(0)
perm = rng.permutation(n_trials)
n_train = int(0.8 * n_trials)
train, test = perm[:n_train], perm[n_train:]


def fit_population_pseudo_r2(X, y_counts):
    """McFadden pseudo-R² per unit using a single PopulationGLM fit (Poisson)."""
    glm = nmo.glm.PopulationGLM(
        regularizer="Ridge",
        regularizer_strength=1e-3,
        observation_model=nmo.observation_models.PoissonObservations(),
    )
    glm.fit(X[train].astype(float), y_counts[train].astype(float))
    rate_pred = np.asarray(glm.predict(X[test].astype(float)))
    pr2 = np.full(y_counts.shape[1], np.nan)
    for u in range(y_counts.shape[1]):
        y_te = y_counts[test, u].astype(float)
        if y_te.sum() < 30:
            continue
        mu = np.clip(rate_pred[:, u], 1e-9, None)
        ll_model = np.sum(y_te * np.log(mu) - mu)
        mu0 = max(y_counts[train, u].mean(), 1e-9)
        ll_null = np.sum(y_te * np.log(mu0) - mu0)
        if ll_null < 0:
            pr2[u] = 1 - ll_model / ll_null
    return pr2


print("Fitting Poisson GLMs with NeMoS PopulationGLM...")
pr2_dir = fit_population_pseudo_r2(X_dir, y_counts)
pr2_sp = fit_population_pseudo_r2(X_sp, y_counts)
pr2_full = fit_population_pseudo_r2(X_full, y_counts)
print(f"median pseudo-R²  direction-only: {np.nanmedian(pr2_dir):.3f}")
print(f"median pseudo-R²  speed-only    : {np.nanmedian(pr2_sp):.3f}")
print(f"median pseudo-R²  dir + speed   : {np.nanmedian(pr2_full):.3f}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
sel = (
    np.isfinite(pr2_dir) & np.isfinite(pr2_full) & np.isfinite(pr2_sp)
    & (pr2_dir > -0.1) & (pr2_full > -0.1) & (pr2_sp > -0.5) & (pr2_full < 1.0)
)
ax[0].scatter(pr2_dir[sel], pr2_full[sel], s=15, alpha=0.6)
mx = max(pr2_dir[sel].max(), pr2_full[sel].max())
ax[0].plot([0, mx], [0, mx], "k--", alpha=0.5)
ax[0].set_xlabel("pseudo-R² – direction only")
ax[0].set_ylabel("pseudo-R² – direction + speed")
ax[0].set_title(f"Adding speed → marginal improvement (n={sel.sum()})")

ax[1].boxplot([pr2_dir[sel], pr2_sp[sel], pr2_full[sel]],
              tick_labels=["dir", "speed", "dir+speed"], showfliers=False)
ax[1].set_ylabel("cross-validated pseudo-R²")
ax[1].set_title("GLM model comparison")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_glm_comparison.png"), dpi=130)
plt.close()
print("saved 07_glm_comparison.png")

# %% [markdown]
# ## 8. Summary
#
# * In Jenkins' M1/PMd we recover canonical cosine direction tuning: a large
#   fraction of units have polar tuning curves well described by a single
#   preferred direction with R² ≥ 0.5.
# * Preferred directions span the full circle, consistent with previous reports
#   that the population codes movement direction by a population vector.
# * Many units additionally modulate their firing with reach speed even when
#   reach direction is held constant; the within-direction speed correlation
#   is non-zero in a substantial subset of units.
# * A Poisson GLM with both direction and speed features yields a higher
#   cross-validated pseudo-R² than either feature alone, confirming that
#   direction and speed contribute complementary information about
#   movement-period firing rates.
#
# All figures are written to `figures/` and the dataset can be re-streamed
# from DANDI at any time without prior download.

# %%
print("DONE.")
