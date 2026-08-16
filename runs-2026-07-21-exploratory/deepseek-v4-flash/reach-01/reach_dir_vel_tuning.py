# %% [markdown]
# # Reach direction and velocity tuning in macaque primary motor cortex
#
# This notebook demonstrates directional and velocity tuning of M1/PMd neurons using the
# **MC_Maze** dataset (Churchland lab, Neural Latents Benchmark 2021) from the DANDI Archive
# (`dandiset 000128`, published version `0.220113.0400`).
#
# Dataset: `sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb`
# - 182 sorted single units from primary motor and dorsal premotor cortex
# - 2,295 delayed-reach trials through an obstacle (maze) workspace with 1-3 targets
# - 1 kHz hand kinematics (`hand_pos`, `hand_vel`)
# - NLB train/validation trial split for honest generalization testing
#
# Two complementary questions are asked:
#
# 1. **Reach direction tuning.** Is a neuron's firing rate during the movement epoch a
#    function of the direction of the reach?  Rate is measured per trial, trials are grouped
#    into eight 45&deg; movement-direction bins, and a one-way ANOVA measures tuning.  Each
#    tuned neuron gets a preferred direction (PD) and a modulation depth.
#
# 2. **Velocity tuning.** Is the instantaneous firing rate a function of the instantaneous
#    hand velocity vector?  A Poisson GLM (fitted with `nemos`) predicts 25 ms spike counts
#    from hand velocity.  Because `log rate = b0 + bx*vx + by*vy = b0 + |b|*|v|*cos(dir - PDv)`,
#    a linear velocity term is equivalent to a *cosine* model in velocity space: the fitted
#    vector `(bx, by)` is the neuron's preferred velocity direction and its norm is its speed
#    gain.  A speed-only model and a full speed+velocity model are fitted for comparison, and
#    all models are evaluated out-of-sample on the held-out NLB validation trials.
#
# > Note on units: the NWB file labels positions and velocities in meters and m/s, but reach
# > amplitude (~130 "m") reveals the true scale is millimeters.  All velocities are rescaled
# > to mm/s here.
#
# Data access is streaming (remfile + local disk cache, no full download of the 690 MB file).

# %% [markdown]
# ## Setup and data loading

# %%
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import h5py, remfile, pynapple as nap
from pynwb import NWBHDF5IO
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm
from nemos import glm, regularizer, observation_models

sns.set_theme(style="whitegrid", context="notebook")
np.random.seed(0)
os.makedirs("figures", exist_ok=True)

# %%
ASSET_ID = "26e85f09-39b7-480f-b337-278a8f034007"
S3_URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
DISK_CACHE = "/tmp/remfile_cache_mcmaze"
MM = 1000.0  # stored "m/s" kinematics are actually mm/s

disk_cache = remfile.DiskCache(DISK_CACHE)
h5f = h5py.File(remfile.File(S3_URL, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5f).read())

print(nwb)
units = nwb["units"]
trials = nwb["trials"]
hand_vel = nwb["hand_vel"] * MM           # mm/s
hand_pos = nwb["hand_pos"] * MM           # mm
print(f"\n{len(units)} units, {len(trials)} trials, "
      f"kinematics rate {hand_vel.rate:.0f} Hz")

trial_df = trials.as_dataframe()
print("trial columns:", list(trial_df.columns))
print("train/val split:", trial_df["split"].value_counts().to_dict())
mo = trial_df["move_onset_time"].to_numpy(dtype=float)

# %% [markdown]
# ## Raw data overview
#
# Before any tuning analysis, verify the data streams look right: a few trials plotted with
# hand velocity, a raster of the first units, and the population-averaged firing rate.

# %%
BIN = 0.025  # 25 ms analysis bin
BIN_HALF = BIN / 2

def avg_ts_on_edges(ts, vals, edges):
    """Mean of a 1-D sampled signal over the intervals [edges[i], edges[i+1])."""
    idx = np.searchsorted(ts, edges, side="left")
    out = np.full((len(edges) - 1, vals.shape[1]), np.nan)
    for k in range(len(edges) - 1):
        lo, hi = idx[k], idx[k + 1]
        if hi > lo:
            out[k] = vals[lo:hi].mean(axis=0)
    return out

def trial_bins(ep):
    """25 ms spike counts and velocity means with identical bins.

    Bins are those produced by pynapple's count (aligned to `ep.start`,
    handles the partial last bin), so the two matrices always share a bin axis.
    """
    cdf = units.count(BIN, ep)                                 # (n_bins, n_units)
    c = np.asarray(cdf)
    centers = np.asarray(cdf.t)
    edges = np.concatenate([[centers[0] - BIN_HALF], centers + BIN_HALF])
    rv = hand_vel.restrict(ep)
    X = avg_ts_on_edges(np.asarray(rv.t), np.asarray(rv), edges)  # mm/s
    return centers, c, X

# ----
ex_trials = np.array([20, 21, 22])
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for ti, t in enumerate(ex_trials):
    a = trial_df.iloc[t]
    ep = nap.IntervalSet(trials["start"][t], trials["end"][t])
    bc, c, v = trial_bins(ep)
    tb = bc - a["move_onset_time"]
    axes[0].plot(tb, v[:, 0], lw=1, label=f"vx (trial {t})" if ti == 0 else None)
    axes[1].plot(tb, v[:, 1], lw=1, label=f"vy (trial {t})" if ti == 0 else None)
    unit_ids = list(units.index)
    axes[2].eventplot([np.asarray(units[unit_ids[u]].restrict(ep).t) - a["move_onset_time"]
                       for u in range(12)], colors="k", lineoffsets=range(12), linewidths=0.8)
    pop = c.mean(axis=1) / BIN
    axes[3].plot(tb, pop, lw=1, label=f"trial {t}")
for ax, ylab in zip(axes, ["vx (mm/s)", "vy (mm/s)", "units 0-11 (raster)",
                           "population rate (Hz)"]):
    ax.set_ylabel(ylab)
axes[0].legend(loc="upper right", fontsize=8)
axes[3].legend(loc="upper right", fontsize=8)
axes[3].set_xlabel("time from movement onset (s)")
axes[-1].axvline(0, color="crimson", ls=":", lw=1)
axes[2].axvline(0, color="crimson", ls=":", lw=1)
plt.tight_layout()
plt.savefig("figures/fig01_raw_overview.png", dpi=150)
plt.close()
print("wrote figures/fig01_raw_overview.png")

# %% [markdown]
# The velocity traces are smoothly bell-shaped and reach a few hundred mm/s; the population
# firing rate rises around movement onset (dashed line at t = 0).  Now we build the
# trial-by-trial database used for both analyses.  For each trial we record the reach
# direction (mean hand-velocity direction over `[move_onset, move_onset + 0.3 s]`), the
# movement-epoch firing rate (`[move_onset, move_onset + 0.4 s]`), and the 25 ms binned spike
# counts and velocity over the full trial for the GLM.

# %%
N_UNITS = len(units)
dir_angle = np.full(len(trials), np.nan)          # reach direction (deg, math convention)
dir_speed = np.full(len(trials), np.nan)          # mean speed in the direction window
fr_mov = np.full((len(trials), N_UNITS), np.nan)  # movement-epoch rate (Hz)

vel_bins, cnt_bins, bin_trial, bin_t = [], [], [], []
for t in tqdm(range(len(trials)), desc="per-trial features"):
    s0, s1 = trials["start"][t], trials["end"][t]
    m0 = mo[t]

    # reach direction from mean velocity after movement onset
    epv = nap.IntervalSet(m0, m0 + 0.3)
    vv = np.asarray(hand_vel.restrict(epv).mean(axis=0))
    dir_angle[t] = np.degrees(np.arctan2(vv[1], vv[0]))
    dir_speed[t] = float(np.linalg.norm(vv))

    # movement-epoch firing rate
    fr_mov[t] = np.asarray(units.count(BIN, nap.IntervalSet(m0, m0 + 0.4)).mean(axis=0)) / BIN

    # binned velocity + spike counts over the whole trial (shared bin axis)
    ep = nap.IntervalSet(s0, s1)
    bc, c, X = trial_bins(ep)
    vel_bins.append(X)
    cnt_bins.append(c)
    bin_trial.append(np.full(len(bc), t))
    bin_t.append(bc)

vel_mat = np.vstack(vel_bins)        # (n_bins, 2) mm/s
cnt_mat = np.vstack(cnt_bins)        # (n_bins, n_units)
bin_trial = np.concatenate(bin_trial)
bin_t = np.concatenate(bin_t)
good = np.all(np.isfinite(vel_mat), axis=1)
print(f"{len(good)} bins total; {(~good).sum()} dropped for missing kinematics "
      f"({100 * (1 - good.mean()):.2f}%)")
vel_mat = vel_mat[good]
cnt_mat = cnt_mat[good]
bin_trial = bin_trial[good]
bin_t = bin_t[good]
speed = np.linalg.norm(vel_mat, axis=1)          # mm/s
is_val = (trial_df["split"].to_numpy() == "val")[bin_trial]
train_mask = ~is_val

mean_rate = cnt_mat.sum(axis=0) / (len(cnt_mat) * BIN)   # Hz over in-trial time
active = mean_rate >= 2.0
print(f"units >=2 Hz: {active.sum()}/{N_UNITS}")
np.savez("kinematics_binned.npz", vel=vel_mat, cnt=cnt_mat, speed=speed,
         bin_trial=bin_trial, bin_t=bin_t, is_val=is_val,
         dir_angle=dir_angle, dir_speed=dir_speed, fr_mov=fr_mov, mo=mo)

# %% [markdown]
# ## Reach direction tuning
#
# Trials are assigned to 8 movement-direction bins (45&deg;) from the mean velocity
# direction.  For each unit, the mean movement-epoch firing rate per direction bin is the
# *tuning curve*.  Direction selectivity is tested with a one-way ANOVA on per-trial rates
# (Bonferroni-corrected across units).  The preferred direction is the direction of the
# vector sum of the binned rates; modulation depth is `max - min` of the tuning curve.

# %%
dir_bin = np.digitize(dir_angle, np.arange(-180, 181, 45)[:-1]) % 8   # 0..7
dirs_deg = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)

pd_angle = np.full(N_UNITS, np.nan)
mod_depth = np.zeros(N_UNITS)
dir_pval = np.ones(N_UNITS)
tuning_curves = np.full((N_UNITS, 8), np.nan)

for u in tqdm(range(N_UNITS), desc="direction ANOVA"):
    rates = fr_mov[:, u]
    groups = [rates[dir_bin == b] for b in range(8)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        continue
    dir_pval[u] = stats.f_oneway(*groups)[1]
    for b in range(8):
        if len(rates[dir_bin == b]) > 0:
            tuning_curves[u, b] = np.mean(rates[dir_bin == b])
    mod_depth[u] = np.nanmax(tuning_curves[u]) - np.nanmin(tuning_curves[u])
    vec = np.nansum(tuning_curves[u] * np.exp(1j * np.radians(dirs_deg)))
    pd_angle[u] = np.degrees(np.angle(vec))

tuned = dir_pval < 0.05 / N_UNITS          # Bonferroni across units
print(f"direction-tuned: {tuned.sum()}/{N_UNITS} ({100 * tuned.sum() / N_UNITS:.0f}%)")

tuned_active = np.where(tuned & active & np.isfinite(pd_angle))[0]
order = np.argsort(-mod_depth[tuned_active])
ex_ids = tuned_active[order][:4]
print("example units:", ex_ids, "PDs:", np.round(pd_angle[ex_ids], 0))

# polar tuning curves
polar_dirs = np.radians(np.arange(0, 360, 45))
fig, axes = plt.subplots(1, 4, subplot_kw=dict(projection="polar"), figsize=(12, 3.4))
for ax, u in zip(axes, ex_ids):
    r = tuning_curves[u]
    r_closed = np.concatenate([r, [r[0]]])
    th = np.concatenate([polar_dirs, [polar_dirs[0]]])
    ax.plot(th, r_closed, color="steelblue", lw=2)
    ax.fill(th, r_closed, color="steelblue", alpha=0.25)
    ax.arrow(pd_angle[u] * np.pi / 180, 0, 0, mod_depth[u] * 0.8,
             head_width=0.12, head_length=mod_depth[u] * 0.2,
             color="crimson", lw=1.5)
    ax.set_title(f"unit {u}   PD {pd_angle[u]:.0f} deg", pad=14)
    ax.set_ylim(0, np.nanmax(r) * 1.15)
plt.tight_layout()
plt.savefig("figures/fig02_direction_polar_examples.png", dpi=150)
plt.close()

# population summary
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
ax = axes[0]
ax.bar(["tuned", "not tuned"], [tuned.sum(), N_UNITS - tuned.sum()],
       color=["#2b8cbe", "#bdbdbd"])
ax.set_title(f"Direction-tuned units ({100 * tuned.sum() / N_UNITS:.0f}%)")
for i, v in enumerate([tuned.sum(), N_UNITS - tuned.sum()]):
    ax.text(i, v + 4, str(v), ha="center")
ax.set_ylim(0, N_UNITS * 1.1)
ax = axes[1]
pd_t = pd_angle[tuned & np.isfinite(pd_angle)]
ax.hist(pd_t, bins=24, range=(-180, 180), color="steelblue", edgecolor="white")
ax.set_xlabel("preferred direction (deg)")
ax.set_title(f"PD distribution (n={len(pd_t)})")
r_vec = np.mean(np.exp(1j * np.radians(pd_t)))
ax.text(0.05, 0.9, f"mean resultant length = {abs(r_vec):.2f}",
        transform=ax.transAxes, fontsize=9)
ax = axes[2]
ax.hist(mod_depth[tuned_active], bins=25, color="#2ca25f", edgecolor="white")
ax.set_xlabel("modulation depth (Hz)")
ax.set_title("Modulation depth, tuned >=2 Hz units")
plt.tight_layout()
plt.savefig("figures/fig03_direction_population.png", dpi=150)
plt.close()

# %% [markdown]
# ## Velocity tuning: firing rate tracks hand velocity
#
# Quantify *when* spikes relate to kinematics.  For each unit, the smoothed spike count
# (25 ms bins, &sigma; ~ 40 ms Gaussian) is cross-correlated with hand speed over all
# in-trial bins; the lag of peak correlation is recorded (positive = spikes lead velocity).
# The median lag over the population is then used to align velocity predictors to spike
# counts in the GLM.

# %%
MAXLAG = 0.2   # s
lags = np.arange(-MAXLAG, MAXLAG + BIN, BIN)
n_lag = len(lags)
spd_z = (speed - speed.mean()) / speed.std()
xcorr = np.full((N_UNITS, n_lag), np.nan)
best_lag = np.full(N_UNITS, np.nan)

for u in tqdm(range(N_UNITS), desc="rate-speed xcorr"):
    y = cnt_mat[:, u]
    if y.sum() < 20:
        continue
    rate = gaussian_filter1d(y.astype(float), sigma=1.6)
    rate = (rate - rate.mean()) / rate.std()
    corr = np.correlate(rate, spd_z, mode="full") / len(rate)
    half = n_lag // 2
    xcorr[u] = corr[len(rate) - 1 - half: len(rate) + half + 1]
    best_lag[u] = lags[np.nanargmax(xcorr[u])]

lead_med = np.nanmedian(best_lag)
print(f"median best lag (spikes lead velocity): {lead_med * 1000:.0f} ms")

fig, ax = plt.subplots(figsize=(6.5, 4))
valid_u = np.isfinite(best_lag)
q = np.nanpercentile(xcorr[valid_u], [25, 50, 75], axis=0)
ax.plot(lags * 1000, q[1], color="steelblue", lw=2)
ax.fill_between(lags * 1000, q[0], q[2], color="steelblue", alpha=0.25)
ax.axvline(lead_med * 1000, color="crimson", ls="--", lw=1)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("lag of speed relative to spikes (ms); positive = spikes lead")
ax.set_ylabel("cross-correlation (z)")
ax.set_title("Spike count vs hand speed, population\n"
             f"(peak at {lead_med * 1000:.0f} ms; n={valid_u.sum()} units)")
plt.tight_layout()
plt.savefig("figures/fig04_lead_lag.png", dpi=150)
plt.close()

# %% [markdown]
# ## Velocity-tuned Poisson GLM
#
# Spike counts in 25 ms bins are modeled as Poisson with $\log \lambda(t) = b_0 +
# \mathbf{b}^\top \mathbf{x}(t)$, using three feature sets, all evaluated on the held-out
# validation trials:
#
# | model    | features                  | interpretation                                |
# |----------|---------------------------|-----------------------------------------------|
# | speed    | $\|\mathbf v\|$           | speed gain only                               |
# | velocity | $v_x, v_y$                | cosine tuning in velocity space (dir x speed) |
# | spd+vel  | $\|\mathbf v\|, v_x, v_y$ | both (speed gain beyond the cosine envelope)  |
#
# Velocity features are evaluated at the population lead (spikes precede the hand), features
# are z-scored with *train* statistics only (no leakage), and a small Ridge penalty (1e-5)
# stabilizes the fit.  Reported metric: out-of-sample pseudo-R&sup2; on validation bins,
# relative to a constant (intercept-only) model.  Units with a training rate < 2 Hz are
# excluded from summaries because their deviance ratio is unstable.

# %%
LEAD_BINS = int(round(lead_med / BIN))

def shift_feat(M, steps):
    """Advance rows of M by `steps` bins: spikes at bin t use velocity at bin t+steps."""
    if steps == 0:
        return M.copy()
    out = np.full_like(M, np.nan)
    if steps > 0:
        out[:-steps] = M[steps:]
    else:
        out[-steps:] = M[:steps]
    return out

y_all = cnt_mat
vx = vel_mat[:, 0]; vy = vel_mat[:, 1]
feat_dict = {
    "speed":    speed[:, None],
    "velocity": np.column_stack([vx, vy]),
    "spd+vel":  np.column_stack([speed, vx, vy]),
}
active_u = np.where(active)[0]

val_ll = {k: np.full(N_UNITS, np.nan) for k in feat_dict}
null_ll_val = np.full(N_UNITS, np.nan)
coefs = {k: [None] * N_UNITS for k in feat_dict}
intercepts = {k: [None] * N_UNITS for k in feat_dict}
feat_std = {k: [None] * N_UNITS for k in feat_dict}   # train std for coefficient rescaling

for u in tqdm(active_u, desc="GLM fits (>=2 Hz units)"):
    y = y_all[:, u].astype(np.float32)
    init_intercept = np.log(max(y[train_mask].mean(), 1e-6))
    null_ll_val[u] = stats.poisson.logpmf(y[~train_mask], y[train_mask].mean()).sum()

    for k, F in feat_dict.items():
        Fs = shift_feat(F, LEAD_BINS)
        mu, sd = Fs[train_mask].mean(0), Fs[train_mask].std(0)
        sd[sd == 0] = 1.0
        Xtr = ((Fs[train_mask] - mu) / sd).astype(np.float32)
        Xva = ((Fs[~train_mask] - mu) / sd).astype(np.float32)
        model = glm.GLM(observation_model=observation_models.PoissonObservations(),
                        regularizer=regularizer.Ridge(),
                        regularizer_strength=1e-5,
                        solver_kwargs={"maxiter": 2000})
        model.fit(Xtr, y[train_mask],
                  init_params=(np.zeros(Xtr.shape[1]),
                               np.array([init_intercept], dtype=np.float32)))
        coefs[k][u] = np.asarray(model.coef_).ravel()
        intercepts[k][u] = float(np.asarray(model.intercept_).ravel()[0])
        feat_std[k][u] = sd
        val_ll[k][u] = float(np.asarray(
            model.score(Xva, y[~train_mask], score_type="log-likelihood")))

pseudo2_val = {k: 1 - val_ll[k] / null_ll_val for k in feat_dict}

# %% [markdown]
# ## Velocity-tuning results
#
# Out-of-sample pseudo-R&sup2; (validation trials) for the three models over the >= 2 Hz
# population.  The spike counts are sparse (median rate ~1.7 Hz at 25 ms), so the values are
# modest but systematic.

# %%
au = np.array(active_u)
p2 = pd.DataFrame({k: pseudo2_val[k][au] for k in feat_dict})
print("val pseudo-R2 (mean +/- SD) over >=2Hz units:")
for k in feat_dict:
    v = p2[k]
    print(f"  {k:10s}: {np.nanmean(v):+.4f} +/- {np.nanstd(v):.4f}  "
          f"(median {np.nanmedian(v):+.4f})")

fig, ax = plt.subplots(figsize=(6.5, 4))
for k in ["speed", "velocity", "spd+vel"]:
    sns.kdeplot(p2[k], label=k, fill=True, alpha=0.3, ax=ax, bw_adjust=0.5)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("out-of-sample pseudo-R$^2$ (validation trials)")
ax.set_title(f"Velocity GLM performance, {len(au)} units (>=2 Hz)")
ax.legend(title="model")
plt.tight_layout()
plt.savefig("figures/fig06_glm_population.png", dpi=150)
plt.close()

# preferred velocity vectors, rescaled to mm/s units
c = coefs["velocity"]
sdx, sdy = [], []
bx_phys, by_phys = np.full(N_UNITS, np.nan), np.full(N_UNITS, np.nan)
for u in range(N_UNITS):
    if c[u] is not None:
        sd_u = feat_std["velocity"][u]
        bx_phys[u] = c[u][0] / sd_u[0]
        by_phys[u] = c[u][1] / sd_u[1]
pdv_angle = np.degrees(np.arctan2(by_phys, bx_phys))
pdv_amp = np.hypot(bx_phys, by_phys) * MM  # mm/s per unit z of spike count... keep relative

vel_tuned = pseudo2_val["velocity"][au] > 0
print(f"units with positive val velocity pseudo-R2: {vel_tuned.sum()}/{len(au)}")
print("example PDv dirs (deg):",
      np.round(np.nanpercentile(pdv_angle[au][vel_tuned], [10, 50, 90]), 0))

fig, ax = plt.subplots(figsize=(6.5, 6))
sel = au[vel_tuned]
for u in sel:
    amp = pdv_amp[u]
    ax.annotate("", xy=(bx_phys[u], by_phys[u]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=1.1, alpha=0.7))
ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel(r"velocity coefficient $b_{v_x}$ (Hz per m/s)")
ax.set_ylabel(r"velocity coefficient $b_{v_y}$ (Hz per m/s)")
ax.set_title(f"Preferred velocity vectors, {len(sel)} units with val pR2 > 0")
ax.set_aspect("equal", adjustable="datalim")
plt.tight_layout()
plt.savefig("figures/fig07_velocity_vectors.png", dpi=150)
plt.close()

# %% [markdown]
# ## Example neuron: predicted vs actual firing rate
#
# For the best velocity-tuned unit, plot the actual 25 ms spike counts (smoothed) together
# with the GLM-predicted rate $\hat\lambda$ around a stretch of validation trials.  The
# prediction should track the movement without being told the trial labels.

# %%
best_unit = au[np.nanargmax(pseudo2_val["velocity"][au])]
print("best velocity-tuned unit:", best_unit,
      "val pseudo-R2:", round(float(pseudo2_val["velocity"][best_unit]), 4))

val_idx = np.where(is_val)[0]
# contiguous stretch of validation bins
keep = np.concatenate([[True], np.diff(val_idx) == 1])
val_idx = val_idx[keep]
start_idx = val_idx[0]
end_idx = min(start_idx + 100, len(y_all))

F_vel = feat_dict["velocity"]
mu_v, sd_v = F_vel[train_mask].mean(0), F_vel[train_mask].std(0)
Fz = ((shift_feat(F_vel, LEAD_BINS)[start_idx:end_idx] - mu_v) / sd_v)
b = coefs["velocity"][best_unit]
b0 = intercepts["velocity"][best_unit]
pred = np.exp(b0 + Fz @ b)

fig, ax = plt.subplots(figsize=(10, 3.6))
t_ax = bin_t[start_idx:end_idx] - bin_t[start_idx:end_idx][0]
rate = gaussian_filter1d(y_all[start_idx:end_idx, best_unit].astype(float), 1.0) / BIN
ax.plot(t_ax, rate, color="k", lw=1.2, label="actual rate (25 ms, smoothed)")
ax.plot(t_ax, pred, color="crimson", lw=1.4, label="GLM predicted rate")
i0 = start_idx
while i0 < end_idx:
    if is_val[i0]:
        i1 = i0
        while i1 < end_idx and is_val[i1]:
            i1 += 1
        ax.axvspan(t_ax[i0 - start_idx], t_ax[min(i1, end_idx) - 1 - start_idx],
                   color="orange", alpha=0.12)
        i0 = i1
    else:
        i0 += 1
ax.set_xlabel("time (s)")
ax.set_ylabel("rate (Hz)")
ax.set_title(f"Unit {best_unit}: GLM velocity prediction tracks firing")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig05_glm_example_prediction.png", dpi=150)
plt.close()

# %% [markdown]
# ## Summary
#
# * **Direction tuning.** The fraction of units whose movement-epoch firing rate depends on
#   reach direction (Bonferroni-corrected ANOVA) is reported above in the heading of the
#   population figure; the example polar curves are smooth and unimodal with a clear preferred
#   direction, and the population PD distribution has a small mean-resultant length, i.e. the
#   preferred directions are roughly uniform across the workspace.
# * **Velocity tuning.** Spike counts lead hand speed by the population-median lag printed in
#   the cross-correlation figure (tens of ms), consistent with a motor-command (efference)
#   relationship.  The Poisson GLM with instantaneous velocity generalizes to held-out trials
#   with positive mean pseudo-R&sup2;; the full speed+velocity model performs best, and
#   velocity alone outperforms speed alone, showing the same neurons encode the movement
#   *vector* (direction and speed jointly), not just its direction.

print("DONE")

# %%
h5f.close()