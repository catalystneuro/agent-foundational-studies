# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reach direction and velocity tuning in macaque motor cortex
#
# **Dataset:** [DANDI:000128](https://dandiarchive.org/dandiset/000128) — *MC_Maze: macaque
# primary motor and dorsal premotor cortex spiking activity during delayed reaching*
# (Churchland & Kaufman, packaged for the Neural Latents Benchmark by Pei et al.).
# Session `sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb`: 182 sorted units recorded
# with two 96-channel Utah arrays in M1 and PMd, 2295 delayed reaches, and the hand position
# and velocity sampled at 1 kHz. (In this packaging every unit's electrode link points into
# the same 96-row electrode group, so the file does not support splitting the population into
# M1 and PMd; everything below treats it as one motor-cortical population.)
#
# **What this notebook demonstrates**
#
# 1. **Reach-direction tuning.** On straight (no-barrier) reaches, single units fire at a rate
#    that varies smoothly with the direction of the reach and is well described by a cosine,
#    the classical result of Georgopoulos et al. (1982). The tuning is present during the
#    instructed delay, before the hand moves at all.
# 2. **Velocity tuning.** Moving from one number per trial to the instantaneous hand velocity,
#    firing rate depends on both the *direction* and the *magnitude* of the velocity vector.
#    Poisson GLMs (NeMoS) fit on 20 ms bins show that a direction x speed model beats a
#    direction-only or speed-only model on held-out reaches, and that firing rate grows with
#    speed in a unit's preferred direction while staying flat or falling in the opposite
#    direction, i.e. speed acts as a gain on directional tuning.
# 3. **Read-out.** A linear decoder of the 182-unit population recovers the hand velocity
#    vector moment by moment on held-out reaches.
#
# Everything streams from the DANDI S3 bucket with `remfile` + a disk cache; nothing is
# downloaded in full. All analysis is done with `pynapple` objects and `nemos` GLMs.

# %% [markdown]
# ## Setup

# %%
import os
import warnings

import jax
import jax.numpy as jnp

# LBFGS on these population GLMs does not reach its tolerance in float32
jax.config.update("jax_enable_x64", True)

import h5py
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap
import remfile
from dandi.dandiapi import DandiAPIClient
from matplotlib.colors import LogNorm
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from scipy.stats import f as fdist
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

# NumPy 2.0 built against Apple's Accelerate BLAS raises spurious "divide by zero /
# overflow / invalid value encountered in matmul" RuntimeWarnings on ordinary matmuls of
# finite values (reproducible with two random finite matrices). The results are correct;
# only the warning is bogus, so it is silenced here to keep the output readable.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*",
                        category=RuntimeWarning)

CACHE_DIR = os.environ.get("REACH_CACHE", "/tmp/reach_cache")
os.makedirs(os.path.join(CACHE_DIR, "remfile"), exist_ok=True)

DANDISET_ID = "000128"
ASSET_PATH = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "font.size": 10})

# %% [markdown]
# ## Streaming the NWB file
#
# `remfile` serves HTTP range requests out of a disk cache, so only the byte ranges actually
# touched are transferred. The 1 kHz hand kinematics are the one stream we read end to end,
# so they are memoised as `.npy` on first use.

# %%
def load_mc_maze():
    """Return spikes (TsGroup), hand kinematics (TsdFrame) and the trial table."""
    client = DandiAPIClient()
    asset = client.get_dandiset(DANDISET_ID, "draft").get_asset_by_path(ASSET_PATH)
    url = asset.get_content_url(follow_redirects=1, strip_query=True)

    rem = remfile.File(url, disk_cache=remfile.DiskCache(os.path.join(CACHE_DIR, "remfile")))
    io = NWBHDF5IO(file=h5py.File(rem, "r"), load_namespaces=True)
    nwbfile = io.read()
    beh = nwbfile.processing["behavior"]

    def cached(name, fn):
        p = os.path.join(CACHE_DIR, name)
        if not os.path.exists(p):
            np.save(p, fn())
        return np.load(p)

    t = cached("maze_time.npy", lambda: beh["hand_pos"].timestamps[:])
    pos = cached("maze_hand_pos.npy", lambda: beh["hand_pos"].data[:])
    vel = cached("maze_hand_vel.npy", lambda: beh["hand_vel"].data[:])

    # The NWB conversion factor of 1e-3 maps the stored values onto metres, so the raw
    # numbers here are millimetres and millimetres per second.
    kin = nap.TsdFrame(t=t, d=np.column_stack([pos, vel]).astype(np.float32),
                       columns=["x", "y", "vx", "vy"])

    spike_times = nwbfile.units["spike_times"][:]
    spikes = nap.TsGroup({i: nap.Ts(t=np.asarray(s)) for i, s in enumerate(spike_times)})
    trials = nwbfile.trials.to_dataframe()
    return spikes, kin, trials, nwbfile


spikes, kin, trials, nwbfile = load_mc_maze()
n_units = len(spikes)
print(nwbfile.session_description)
print(f"{n_units} units, {len(trials)} trials, "
      f"{kin.index[-1] - kin.index[0]:.0f} s of 1 kHz hand kinematics")
print(spikes)

# %% [markdown]
# The task has two versions. `trial_version == 0` are straight reaches to a single visible
# target with no barriers; the remaining versions place barriers in the workspace so the
# monkey must curve around them. The straight reaches give a clean, classical
# centre-out-style dataset for direction tuning; all trials together give a much richer
# sampling of the velocity plane, which is what the continuous analysis needs.

# %%
print(trials["trial_version"].value_counts().sort_index())
straight = trials[trials.trial_version == 0].reset_index(drop=True)
onset_straight = straight["move_onset_time"].values
print(f"{len(straight)} straight reaches")

# %% [markdown]
# ## 1. Raw data validation
#
# Before any analysis: are the streams aligned and do they look like reaches? Reach direction
# is measured from the hand itself (displacement over the 400 ms after movement onset) rather
# than from the nominal target position, so it cannot be wrong about what the animal did.

# %%
kt = np.asarray(kin.index)
kv = kin.values
assert np.isfinite(kv).all(), "unexpected NaNs in the kinematics"

p0 = nap.Ts(onset_straight).value_from(kin)[["x", "y"]].values
p1 = nap.Ts(onset_straight + 0.4).value_from(kin)[["x", "y"]].values
theta_trial = np.arctan2(*(p1 - p0)[:, ::-1].T)

speed = nap.Tsd(t=kt, d=np.hypot(kv[:, 2], kv[:, 3]))
print("hand speed percentiles (mm/s):",
      np.percentile(speed.values, [50, 90, 99, 100]).round(0))
print("reach start position: mean", p0.mean(0).round(1), "sd", p0.std(0).round(1), "mm")

# %%
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)
cmap = plt.get_cmap("hsv")

ax = fig.add_subplot(gs[:2, 0])
for i in range(0, len(straight), 3):
    seg = kin.restrict(nap.IntervalSet(onset_straight[i] - 0.05, onset_straight[i] + 0.5))
    ax.plot(seg["x"].values, seg["y"].values, lw=0.6, alpha=0.6,
            color=cmap((theta_trial[i] + np.pi) / (2 * np.pi)))
ax.set(xlabel="hand x (mm)", ylabel="hand y (mm)",
       title="Straight reaches (no barriers)\ncoloured by reach direction")
ax.set_aspect("equal")

ax = fig.add_subplot(gs[2, 0])
sp_pe = nap.build_tensor(speed, nap.IntervalSet(onset_straight - 0.3, onset_straight + 0.7))
tt = np.arange(sp_pe.shape[1]) / 1000.0 - 0.3
m, sd = np.nanmean(sp_pe, 0), np.nanstd(sp_pe, 0)
ax.plot(tt, m, color="k")
ax.fill_between(tt, m - sd, m + sd, color="k", alpha=0.2)
ax.axvline(0, ls="--", c="r")
ax.set(xlabel="time from movement onset (s)", ylabel="speed (mm/s)", title="Mean hand speed")

t0 = trials["start_time"].iloc[300]
ep = nap.IntervalSet(t0, t0 + 12)
kseg = kin.restrict(ep)

ax = fig.add_subplot(gs[0, 1:])
ax.plot(kseg.index, kseg["x"].values, label="x")
ax.plot(kseg.index, kseg["y"].values, label="y")
for _, r in trials[(trials.start_time > t0) & (trials.start_time < t0 + 12)].iterrows():
    ax.axvline(r.move_onset_time, color="r", ls="--", lw=0.8)
ax.legend(loc="upper right", fontsize=8)
ax.set(ylabel="hand position (mm)",
       title="Raw behaviour and spiking (red dashed = movement onset)")

ax = fig.add_subplot(gs[1, 1:])
ax.plot(kseg.index, kseg["vx"].values, label="vx")
ax.plot(kseg.index, kseg["vy"].values, label="vy")
ax.legend(loc="upper right", fontsize=8)
ax.set(ylabel="hand velocity (mm/s)")

ax = fig.add_subplot(gs[2, 1:])
for row, u in enumerate(np.array(spikes.index)[np.argsort(spikes.rate.values)]):
    st = spikes[u].restrict(ep).index
    ax.plot(st, np.full_like(st, row), "|", ms=1.6, color="k")
ax.set(xlabel="time (s)", ylabel="unit (sorted by rate)", ylim=(-2, n_units + 2))

fig.suptitle("DANDI 000128 MC_Maze, monkey Jenkins: raw data validation", fontsize=13)
fig.savefig("fig01_raw_data.png", bbox_inches="tight")

# %% [markdown]
# Reaches radiate outward from a common start point, speed profiles are the expected
# single-peaked bells reaching ~750 mm/s, and the population raster shows clear bursts locked
# to each movement. The streams are aligned.

# %% [markdown]
# ## 2. Reach-direction tuning
#
# For every straight reach we take the mean firing rate in a movement window
# (50-350 ms after onset) and in a late-delay window (250-50 ms *before* onset), and fit
#
# $$r(\theta) = b_0 + b_1\cos\theta + b_2\sin\theta = b_0 + d\cos(\theta - \mathrm{PD})$$
#
# by least squares. $d$ is the modulation depth and PD the preferred direction. Significance
# comes from a permutation test that shuffles the reach directions across trials, which makes
# no assumption about the noise distribution.

# %%
MOVE_WIN = (0.05, 0.35)
PLAN_WIN = (-0.25, -0.05)


def trial_rates(spikes, onsets, window):
    """Mean firing rate (Hz) per unit per trial in a fixed window around each event."""
    ep = nap.IntervalSet(start=onsets + window[0], end=onsets + window[1])
    counts = nap.build_tensor(spikes, ep, bin_size=window[1] - window[0])
    return counts[:, :, 0] / (window[1] - window[0])


def cosine_fit(rates, angles):
    """Least-squares fit of r = b0 + d*cos(theta - PD) for every unit."""
    X = np.column_stack([np.ones_like(angles), np.cos(angles), np.sin(angles)])
    beta, *_ = np.linalg.lstsq(X, rates.T, rcond=None)
    resid = rates.T - X @ beta
    ss_res = (resid ** 2).sum(0)
    ss_tot = ((rates.T - rates.T.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    n, p = len(angles), 3
    f = (r2 / (p - 1)) / np.maximum((1 - r2) / (n - p), 1e-12)
    return dict(b0=beta[0], depth=np.hypot(beta[1], beta[2]),
                pd=np.arctan2(beta[2], beta[1]), r2=r2, p=fdist.sf(f, p - 1, n - p))


def permutation_p(rates, angles, n_perm=500, seed=0):
    """Permutation p-value for the cosine modulation depth, per unit."""
    rng = np.random.default_rng(seed)
    obs = cosine_fit(rates, angles)["depth"]
    null = np.empty((n_perm, rates.shape[0]))
    for i in tqdm(range(n_perm), desc="permutations", leave=False):
        null[i] = cosine_fit(rates, rng.permutation(angles))["depth"]
    return (null >= obs).mean(0)


def psth(spikes, onsets, window, bin_size=0.01, sigma_bins=2.0):
    ep = nap.IntervalSet(start=onsets + window[0], end=onsets + window[1])
    rate = nap.build_tensor(spikes, ep, bin_size=bin_size) / bin_size
    rate = gaussian_filter1d(rate, sigma_bins, axis=-1, mode="nearest")
    t = window[0] + bin_size * (np.arange(rate.shape[-1]) + 0.5)
    return rate, t


move_rates = trial_rates(spikes, onset_straight, MOVE_WIN)
plan_rates = trial_rates(spikes, onset_straight, PLAN_WIN)

fit_move = cosine_fit(move_rates, theta_trial)
fit_plan = cosine_fit(plan_rates, theta_trial)
perm_p = permutation_p(move_rates, theta_trial)
perm_p_plan = permutation_p(plan_rates, theta_trial)
# Many units in this packaging are nearly silent; population statistics and the
# Poisson pseudo-R^2 are both meaningless for them, so require a minimum rate.
active = move_rates.mean(1) > 0.5     # Hz, mean over the movement window
tuned = active & (perm_p < 0.01)

# 8 direction bins; the straight reaches sample 7 of them
n_dbin = 8
width = 2 * np.pi / n_dbin
dbin = np.floor((np.mod(theta_trial, 2 * np.pi) + width / 2) / width).astype(int) % n_dbin
centres = np.arange(n_dbin) * width
occupied = np.array([b for b in range(n_dbin) if (dbin == b).sum() > 5])
print("trials per direction bin:", np.bincount(dbin, minlength=n_dbin))

# cosine fit to the condition means (the classical Georgopoulos measure, free of
# single-trial spiking noise)
cond_mu = np.stack([move_rates[:, dbin == b].mean(1) for b in occupied], axis=1)
fit_cond = cosine_fit(cond_mu, centres[occupied])

print(f"{active.sum()}/{n_units} units fire above 0.5 Hz during movement")
print(f"direction-tuned during movement: {tuned.sum()}/{active.sum()} active units "
      f"({100*tuned.sum()/active.sum():.0f}%, permutation p < 0.01)")
print(f"direction-tuned during the delay: {(active & (perm_p_plan < 0.01)).sum()}"
      f"/{active.sum()} active units")
print(f"median cosine R^2 on condition means (tuned units): "
      f"{np.median(fit_cond['r2'][tuned]):.2f}")
print(f"median modulation depth (tuned units): {np.median(fit_move['depth'][tuned]):.1f} Hz")

# %% [markdown]
# ### Example units

# %%
rate_psth, tpsth = psth(spikes, onset_straight, (-0.4, 0.6))
ex = np.argsort(-fit_move["depth"] * (fit_cond["r2"] > 0.7))[:3]

cols = plt.get_cmap("hsv")(np.linspace(0, 1, n_dbin + 1)[:n_dbin])
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5),
                         gridspec_kw=dict(hspace=0.5, wspace=0.35))
for j, u in enumerate(ex):
    ax = axes[0, j]
    for b in occupied:
        ax.plot(tpsth, rate_psth[u][dbin == b].mean(0), color=cols[b], lw=1.4,
                label=f"{int(np.degrees(centres[b]))}$\\degree$")
    ax.axvline(0, ls="--", c="k", lw=0.8)
    ax.set(xlabel="time from movement onset (s)", title=f"unit {u}")
    if j == 0:
        ax.set_ylabel("firing rate (Hz)")
    if j == 2:
        ax.legend(title="reach direction", fontsize=7, title_fontsize=7,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    axes[1, j].remove()
    axp = fig.add_subplot(2, 3, 4 + j, projection="polar")
    mu = np.array([move_rates[u][dbin == b].mean() for b in occupied])
    se = np.array([move_rates[u][dbin == b].std() / np.sqrt((dbin == b).sum())
                   for b in occupied])
    th = np.append(centres[occupied], centres[occupied][0])
    axp.errorbar(th, np.append(mu, mu[0]), yerr=np.append(se, se[0]),
                 color="k", lw=1.5, marker="o", ms=4)
    grid = np.linspace(0, 2 * np.pi, 200)
    axp.plot(grid, fit_move["b0"][u] + fit_move["depth"][u] * np.cos(grid - fit_move["pd"][u]),
             color="crimson", lw=1.5)
    axp.set_title(f"PD = {np.degrees(fit_move['pd'][u]) % 360:.0f}$\\degree$,  "
                  f"$R^2_{{cond}}$ = {fit_cond['r2'][u]:.2f}", pad=30, fontsize=10)
    axp.tick_params(labelsize=7)
    axp.set_rlabel_position(157)

fig.suptitle("Directional tuning of single motor-cortex units (MC_Maze, straight reaches)\n"
             "top: PSTHs by reach direction (colour = direction); "
             "bottom: mean movement-epoch rate (black) with cosine fit (red)", fontsize=12)
fig.savefig("fig02_example_direction_tuning.png", bbox_inches="tight")

# %% [markdown]
# ### Population statistics
#
# The last panel recomputes the cosine fit in each 10 ms bin of the PSTH, which turns the
# modulation depth into a time course and shows when directional information appears.

# %%
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.42, top=0.86)

ax = fig.add_subplot(gs[0, 0], projection="polar")
ax.hist(fit_move["pd"][tuned], bins=np.linspace(-np.pi, np.pi, 25), color="steelblue")
ax.set_title(f"Preferred directions\n(n = {tuned.sum()} tuned units)", pad=30, fontsize=10)
ax.set_rlabel_position(157)
ax.tick_params(labelsize=7)

ax = fig.add_subplot(gs[0, 1])
ax.hist(fit_move["depth"][active], bins=30, color="steelblue", label="movement")
ax.hist(fit_plan["depth"][active], bins=30, color="darkorange", alpha=0.6, label="delay")
ax.set(xlabel="cosine modulation depth (Hz)", ylabel="units",
       title="Depth of directional modulation")
ax.legend()

ax = fig.add_subplot(gs[0, 2])
ax.scatter(fit_plan["depth"][active], fit_move["depth"][active], s=12, c="k", alpha=0.6)
lim = max(fit_move["depth"][active].max(), fit_plan["depth"][active].max()) * 1.05
ax.plot([0, lim], [0, lim], "r--", lw=0.8)
ax.set(xlabel="delay-period depth (Hz)", ylabel="movement depth (Hz)",
       title="Directional tuning is already\npresent before movement onset")

ax = fig.add_subplot(gs[1, 0])
ax.hist(fit_cond["r2"][tuned], bins=30, color="steelblue")
ax.axvline(np.median(fit_cond["r2"][tuned]), color="r", ls="--")
ax.set(xlabel="$R^2$ of cosine fit to condition means", ylabel="units",
       title=f"Cosine model captures directional tuning\n"
             f"median $R^2$ = {np.median(fit_cond['r2'][tuned]):.2f} (tuned units)")

ax = fig.add_subplot(gs[1, 1])
ax.hist(perm_p[active], bins=np.linspace(0, 1, 41), color="steelblue")
ax.axvline(0.01, color="r", ls="--")
ax.set(xlabel="permutation p (direction shuffled)", ylabel="active units",
       title=f"{100*tuned.sum()/active.sum():.0f}% of active units significantly\n"
             f"direction tuned (p < 0.01)")

ax = fig.add_subplot(gs[1, 2])
depth_t = np.stack([cosine_fit(rate_psth[:, :, k], theta_trial)["depth"]
                    for k in range(len(tpsth))], axis=1)
m = depth_t[tuned].mean(0)
se = depth_t[tuned].std(0) / np.sqrt(tuned.sum())
ax.plot(tpsth, m, color="crimson", lw=1.8, label="directional modulation depth")
ax.fill_between(tpsth, m - se, m + se, color="crimson", alpha=0.25)
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.set(xlabel="time from movement onset (s)", ylabel="mean depth (Hz)",
       title="Directional tuning ramps up before\nmovement and peaks near peak speed")
ax2 = ax.twinx()
sp_t = np.nanmean(nap.build_tensor(speed, nap.IntervalSet(onset_straight - 0.4,
                                                          onset_straight + 0.6)), 0)
ax2.plot(np.arange(len(sp_t)) / 1000 - 0.4, sp_t, color="gray", lw=1.2, label="hand speed")
ax2.set_ylabel("hand speed (mm/s)", color="gray")
ax2.tick_params(axis="y", colors="gray")
lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")

fig.suptitle(f"Population statistics of reach-direction tuning "
             f"({n_units} units, {len(straight)} straight reaches)", fontsize=13, y=0.99)
fig.savefig("fig03_population_direction_tuning.png", bbox_inches="tight")

# %% [markdown]
# ## 3. Velocity tuning
#
# Reach direction is one number per trial. The stronger claim is that firing rate tracks the
# *instantaneous* velocity vector. From here on we use all 2295 reaches (including the curved
# barrier trials, which sample the velocity plane far more densely than straight reaches do)
# in 20 ms bins.
#
# ### How far does spiking lead the hand?
#
# Motor cortex leads the arm, so the kinematics must be shifted before they can be regressed
# on the spikes. We scan the lead time with a simple linear model of the smoothed rate on
# $[v_x, v_y, \mathrm{speed}]$ and take the population optimum.

# %%
BIN = 0.02
LAG_BIN = 0.01
WIN = (-0.20, 0.60)
LAGS = np.arange(-0.20, 0.301, 0.02)

onset_all = trials["move_onset_time"].values
n_trials = len(onset_all)


def kin_tensor(bin_size, lag):
    """(n_trials, n_bins, 4) hand x, y, vx, vy at bin centres shifted by `lag`."""
    tc = WIN[0] + bin_size * (np.arange(round((WIN[1] - WIN[0]) / bin_size)) + 0.5)
    tt = onset_all[:, None] + tc[None, :] + lag
    return np.stack([np.interp(tt, kt, kv[:, j]) for j in range(4)], axis=-1), tc


def count_tensor(bin_size):
    ep = nap.IntervalSet(start=onset_all + WIN[0], end=onset_all + WIN[1])
    return nap.build_tensor(spikes, ep, bin_size=bin_size)


cnt_lag = count_tensor(LAG_BIN)
Y_all = gaussian_filter1d(cnt_lag / LAG_BIN, 3.0, axis=-1,
                          mode="nearest").reshape(n_units, -1).T
K0, _ = kin_tensor(LAG_BIN, 0.0)
mv = np.hypot(K0[:, :, 2], K0[:, :, 3]).ravel() > 50.0

r2_by_lag = np.empty((n_units, len(LAGS)))       # moving bins only
r2_by_lag_all = np.empty((n_units, len(LAGS)))   # every bin in the window
for i, lag in enumerate(tqdm(LAGS, desc="lag scan")):
    K, _ = kin_tensor(LAG_BIN, lag)
    v = K[:, :, 2:].reshape(-1, 2)
    X_full = np.column_stack([v, np.hypot(v[:, 0], v[:, 1])])
    for out, sel in ((r2_by_lag, mv), (r2_by_lag_all, slice(None))):
        X = X_full[sel] - X_full[sel].mean(0)
        Y = Y_all[sel] - Y_all[sel].mean(0)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        out[:, i] = 1 - ((Y - X @ beta) ** 2).sum(0) / (Y ** 2).sum(0)

opt_lag = LAGS[np.argmax(r2_by_lag.mean(0))]
opt_lag_all = LAGS[np.argmax(r2_by_lag_all.mean(0))]
best_lag = LAGS[np.argmax(r2_by_lag, axis=1)]
print(f"neural lead over the hand, moving bins: population optimum "
      f"{1000*opt_lag:.0f} ms, median single unit {1000*np.median(best_lag):.0f} ms")
print(f"  including delay-period bins the optimum shifts to {1000*opt_lag_all:.0f} ms")
del cnt_lag, Y_all, K0

# %% [markdown]
# ### Poisson GLMs over the velocity plane
#
# For each 20 ms bin in which the hand is moving (speed > 50 mm/s, so that direction is
# defined) we fit NeMoS Poisson GLMs with five feature sets:
#
# | model | features |
# |---|---|
# | speed only | B-spline basis on speed |
# | direction only | cyclic B-spline basis on direction |
# | linear velocity | $v_x, v_y$ (the classical Moran & Schwartz model) |
# | direction + speed | additive combination of the two bases |
# | direction $\times$ speed | outer product of the two bases (speed can rescale the whole directional tuning curve) |
#
# Models are scored by McFadden pseudo-$R^2$ against a constant-rate model, cross-validated
# over five contiguous blocks of held-out reaches.

# %%
cnt = count_tensor(BIN)
K, tc = kin_tensor(BIN, opt_lag)
n_bins = K.shape[1]
vx, vy = K[:, :, 2].ravel(), K[:, :, 3].ravel()
sp = np.hypot(vx, vy)
theta = np.arctan2(vy, vx)
counts = cnt.reshape(n_units, -1).T.astype(float)

moving = sp > 50.0
print(f"{moving.sum()}/{moving.size} bins with speed > 50 mm/s ({100*moving.mean():.0f}%)")
vx, vy, sp, theta, counts = (a[moving] for a in (vx, vy, sp, theta, counts))
trial_id = np.repeat(np.arange(n_trials), n_bins)[moving]

dir_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, label="direction")
spd_basis = nmo.basis.BSplineEval(n_basis_funcs=6, label="speed")
sp_clip = np.clip(sp, 0, np.percentile(sp, 99.5))

X_dir = np.asarray(dir_basis.compute_features(theta), dtype=float)
X_spd = np.asarray(spd_basis.compute_features(sp_clip), dtype=float)
X_mul = np.asarray((dir_basis * spd_basis).compute_features(theta, sp_clip), dtype=float)
X_add = np.column_stack([X_dir, X_spd])
X_lin = np.column_stack([vx, vy]) / 500.0

MODELS = {
    "speed only": X_spd,
    "direction only": X_dir,
    "linear velocity ($v_x, v_y$)": X_lin,
    "direction + speed": X_add,
    "direction $\\times$ speed": X_mul,
}


def make_glm():
    return nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=1e-4,
                                 solver_name="LBFGS",
                                 solver_kwargs={"tol": 1e-6, "maxiter": 1000})


def cv_pseudo_r2(X, y, n_splits=5):
    out = np.zeros((n_splits, y.shape[1]))
    for f, (tr_idx, _) in enumerate(KFold(n_splits=n_splits).split(np.arange(n_trials))):
        tr = np.isin(trial_id, tr_idx)
        glm = make_glm().fit(X[tr], y[tr])
        # aggregate over samples only, so the score stays per neuron
        out[f] = glm.score(X[~tr], y[~tr], score_type="pseudo-r2-McFadden",
                           aggregate_sample_scores=lambda a: jnp.mean(a, axis=0))
    return out.mean(0)


cv_scores = {}
for name, X in tqdm(MODELS.items(), desc="GLM model comparison"):
    cv_scores[name] = cv_pseudo_r2(X, counts)
    print(f"  {name:32s} median CV pseudo-R2 = "
          f"{np.median(cv_scores[name][active]):.4f}")

glm_full = make_glm().fit(X_mul, counts)

# %% [markdown]
# ### Tuning surfaces
#
# The fitted direction $\times$ speed model is evaluated on a grid to give each unit a firing
# rate surface over the velocity plane, next to the raw binned rates for comparison.

# %%
n_g = 60
g_dir = np.linspace(-np.pi, np.pi, n_g, endpoint=False)
g_spd = np.linspace(sp_clip.min(), sp_clip.max(), n_g)
GD, GS = np.meshgrid(g_dir, g_spd, indexing="ij")
X_grid = np.asarray((dir_basis * spd_basis).compute_features(GD.ravel(), GS.ravel()),
                    dtype=float)
surf = np.asarray(glm_full.predict(X_grid)).reshape(n_g, n_g, n_units) / BIN

n_db, n_sb = 16, 8
db = np.clip(((theta + np.pi) / (2 * np.pi) * n_db).astype(int), 0, n_db - 1)
sedges = np.percentile(sp, np.linspace(0, 99.5, n_sb + 1))  # matches the GLM grid
sb = np.clip(np.searchsorted(sedges, sp, side="right") - 1, 0, n_sb - 1)
emp = np.full((n_units, n_db, n_sb), np.nan)
occ = np.zeros((n_db, n_sb), int)
for i in range(n_db):
    for j in range(n_sb):
        m = (db == i) & (sb == j)
        occ[i, j] = m.sum()
        if m.sum() > 50:
            emp[:, i, j] = counts[m].mean(0) / BIN

# Preferred direction from the GLM: vector average of the speed-averaged direction
# profile. This is the same definition as the PD of the trial-level cosine fit, and
# much less noisy than the arg-max of the surface.
prof = surf.mean(axis=1)                       # (n_dir_grid, n_units)
pd_glm = np.angle((np.exp(1j * g_dir)[:, None] * (prof - prof.mean(0))).sum(0))
pref_idx = np.argmin(np.abs(np.angle(np.exp(1j * (g_dir[:, None] - pd_glm)))), axis=0)
anti_idx = (pref_idx + n_g // 2) % n_g
vel_signal = r2_by_lag.max(1)   # how much velocity signal a unit has at all
gain = np.empty((n_units, 2))
for u in range(n_units):
    for k, idx in enumerate([pref_idx[u], anti_idx[u]]):
        gain[u, k] = np.polyfit(g_spd, surf[idx, :, u], 1)[0] * 100  # Hz per 100 mm/s

# %%
cv_dir, cv_full = cv_scores["direction only"], cv_scores["direction $\\times$ speed"]
# Example units for display: well fit by the velocity GLM, but also fast enough that
# the measured rate map is not mostly empty (pseudo-R^2 alone favours sparse units).
bright = tuned & (move_rates.mean(1) > 10.0)
ex_v = np.argsort(-(cv_full * bright))[:3]

fig = plt.figure(figsize=(14.5, 9.5))
gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.38, top=0.86, bottom=0.07)
dir_edges = np.linspace(-180, 180, n_db + 1)

for j, u in enumerate(ex_v):
    ax = fig.add_subplot(gs[0, j])
    im = ax.pcolormesh(dir_edges, sedges, emp[u].T, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Hz" if j == 2 else None)
    ax.set(xlabel="hand direction (deg)", title=f"unit {u} — measured",
           xticks=[-180, -90, 0, 90, 180])
    if j == 0:
        ax.set_ylabel("hand speed (mm/s)")

    ax = fig.add_subplot(gs[1, j])
    im = ax.pcolormesh(np.degrees(g_dir), g_spd, surf[:, :, u].T, cmap="viridis",
                       shading="gouraud")
    fig.colorbar(im, ax=ax, label="Hz" if j == 2 else None)
    ax.set(xlabel="hand direction (deg)", title=f"unit {u} — Poisson GLM",
           xticks=[-180, -90, 0, 90, 180])
    if j == 0:
        ax.set_ylabel("hand speed (mm/s)")

ax = fig.add_subplot(gs[2, 0])
has_vel = active & (r2_by_lag.max(1) > 0.02)
ax.plot(1000 * LAGS, r2_by_lag[has_vel].T, color="k", alpha=0.10, lw=0.6)
ax.plot(1000 * LAGS, r2_by_lag.mean(0), color="crimson", lw=2.4, label="moving bins")
ax.plot(1000 * LAGS, r2_by_lag_all.mean(0), color="darkorange", lw=2.0, ls="--",
        label="all bins (incl. delay)")
ax.axvline(1000 * opt_lag, color="crimson", ls=":", lw=1)
ax.axvline(1000 * opt_lag_all, color="darkorange", ls=":", lw=1)
ax.set(xlabel="neural lead (ms)", ylabel="$R^2$ (linear velocity model)",
       title=f"Spiking leads the hand by {1000*opt_lag:.0f} ms\n"
             f"({1000*opt_lag_all:.0f} ms if the delay period is included)")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 1])
ax.hist(1000 * best_lag[has_vel], bins=np.arange(-210, 311, 20), color="steelblue")
ax.axvline(1000 * np.median(best_lag[has_vel]), color="crimson", ls="--",
           label=f"median {1000*np.median(best_lag[has_vel]):.0f} ms")
ax.set(xlabel="best lead per unit (ms)", ylabel="units",
       title=f"Single-unit optimal lead\n({has_vel.sum()} units with $R^2$ > 0.02)")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 2])
norm = np.maximum(surf.max(axis=(0, 1)), 1e-6)
pref = np.stack([surf[pref_idx[u], :, u] for u in range(n_units)]) / norm[:, None]
anti = np.stack([surf[anti_idx[u], :, u] for u in range(n_units)]) / norm[:, None]
for arr, col, lab in [(pref, "crimson", "preferred direction"),
                      (anti, "steelblue", "anti-preferred")]:
    m = arr[has_vel].mean(0)
    se = arr[has_vel].std(0) / np.sqrt(has_vel.sum())
    ax.plot(g_spd, m, color=col, lw=2.2, label=lab)
    ax.fill_between(g_spd, m - se, m + se, color=col, alpha=0.25)
ax.set(xlabel="hand speed (mm/s)", ylabel="firing rate (norm. to peak)",
       title="Rate grows with speed only in\nthe preferred direction")
ax.set_ylim(top=ax.get_ylim()[1] + 0.16)   # headroom so the legend clears the curves
ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Instantaneous velocity tuning in motor cortex (DANDI 000128, MC_Maze)\n"
             "top two rows: firing rate over the (direction, speed) plane, measured and "
             "GLM-predicted, for three example units", fontsize=12.5)
fig.savefig("fig04_velocity_tuning.png", bbox_inches="tight")

# %% [markdown]
# ### Model comparison and consistency with the trial-level analysis

# %%
ORDER = list(MODELS)
fig = plt.figure(figsize=(14.5, 8))
gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.36, top=0.86)

ax = fig.add_subplot(gs[0, 0])
med = [np.median(cv_scores[k][active]) for k in ORDER]
bars = ax.bar(range(len(ORDER)), med,
              color=["0.6", "steelblue", "seagreen", "goldenrod", "crimson"])
for b, m in zip(bars, med):
    ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}", ha="center", va="bottom",
            fontsize=8)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, rotation=35, ha="right", fontsize=8)
ax.set(ylabel="median CV pseudo-$R^2$",
       title="Poisson GLM model comparison\n(5-fold CV, held-out reaches)")

ax = fig.add_subplot(gs[0, 1])
ax.scatter(cv_dir[active], cv_full[active], s=14, c="k", alpha=0.55)
lim = [min(cv_dir[active].min(), 0),
       max(cv_dir[active].max(), cv_full[active].max()) * 1.05]
ax.plot(lim, lim, "r--", lw=0.9)
frac = (cv_full[tuned] > cv_dir[tuned]).mean()  # among trial-level tuned units
ax.set(xlabel="direction only", ylabel="direction $\\times$ speed", xlim=lim, ylim=lim,
       title=f"Adding speed helps {100*frac:.0f}% of\ndirection-tuned units")

ax = fig.add_subplot(gs[0, 2])
bins_g = np.linspace(-1.5, 3, 40)
gkeep = active & (vel_signal > 0.02)
ax.hist(gain[gkeep, 0], bins=bins_g, color="crimson", alpha=0.65, label="preferred direction")
ax.hist(gain[gkeep, 1], bins=bins_g, color="steelblue", alpha=0.65, label="anti-preferred")
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.set(xlabel="speed gain (Hz per 100 mm/s)", ylabel="units",
       title=f"Speed gain: median {np.median(gain[gkeep,0]):+.2f} vs "
             f"{np.median(gain[gkeep,1]):+.2f} Hz/100 mm/s")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 0])
delta = np.angle(np.exp(1j * (pd_glm - fit_move["pd"])))
keep = tuned & (vel_signal > 0.02)   # tuned on trials AND velocity signal in the GLM
circ_r = np.abs(np.exp(1j * delta[keep]).mean())
ax.scatter(np.degrees(fit_move["pd"]) % 360, np.degrees(pd_glm) % 360, s=10, c="0.75",
           label="all units")
ax.scatter(np.degrees(fit_move["pd"][keep]) % 360, np.degrees(pd_glm[keep]) % 360,
           s=16, c="k", alpha=0.75, label=f"n = {keep.sum()} with velocity signal")
ax.plot([0, 360], [0, 360], "r--", lw=0.9)
ax.set(xlabel="PD from trial-level cosine fit (deg)", ylabel="PD from velocity GLM (deg)",
       xticks=[0, 90, 180, 270, 360], yticks=[0, 90, 180, 270, 360],
       title=f"Preferred directions broadly agree\n(circular $r$ = {circ_r:.2f})")
ax.legend(fontsize=7, loc="lower right")

ax = fig.add_subplot(gs[1, 1])
ax.hist(np.degrees(delta[keep]), bins=np.arange(-180, 181, 15), color="steelblue")
ax.set(xlabel="PD(velocity GLM) $-$ PD(trial cosine) (deg)", ylabel="units",
       xticks=[-180, -90, 0, 90, 180],
       title=f"median |difference| = {np.median(np.abs(np.degrees(delta[keep]))):.0f}$\\degree$")

ax = fig.add_subplot(gs[1, 2])
im = ax.pcolormesh(dir_edges, sedges, occ.T, cmap="magma")
fig.colorbar(im, ax=ax, label="20 ms bins")
ax.set(xlabel="hand direction (deg)", ylabel="hand speed (mm/s)",
       xticks=[-180, -90, 0, 90, 180],
       title="Sampling of the velocity plane\n(all directions and speeds visited)")

fig.suptitle("Direction and speed are separate, jointly necessary determinants of "
             "motor-cortex firing", fontsize=13)
fig.savefig("fig05_velocity_model_comparison.png", bbox_inches="tight")

# %% [markdown]
# ## 4. Reading the velocity vector back out of the population
#
# A standard optimal linear estimator: predict $v_x$ and $v_y$ in each 20 ms bin from the
# spike counts of all units over the following 200 ms, ridge-regularised and cross-validated
# over held-out reaches.

# %%
N_LAG = 10
usable = n_bins - N_LAG
Xd = np.concatenate([cnt[:, :, l:l + usable] for l in range(N_LAG)], axis=0)
Xd = Xd.transpose(1, 2, 0).reshape(n_trials * usable, n_units * N_LAG)
Kd, _ = kin_tensor(BIN, 0.0)
Yd = np.column_stack([Kd[:, :usable, 2].ravel(), Kd[:, :usable, 3].ravel()])
print("decoder design matrix", Xd.shape, "targets", Yd.shape)

pred = np.empty_like(Yd)
for tr_idx, _ in KFold(n_splits=5).split(np.arange(n_trials)):
    tr = np.isin(np.repeat(np.arange(n_trials), usable), tr_idx)
    pred[~tr] = Ridge(alpha=50.0).fit(Xd[tr], Yd[tr]).predict(Xd[~tr])

r2_dec = 1 - ((Yd - pred) ** 2).sum(0) / ((Yd - Yd.mean(0)) ** 2).sum(0)
speed_true = np.hypot(*Yd.T)
ang_err = np.degrees(np.abs(np.angle(np.exp(
    1j * (np.arctan2(pred[:, 1], pred[:, 0]) - np.arctan2(Yd[:, 1], Yd[:, 0]))))))
fast = speed_true > 200
print(f"held-out R2: vx = {r2_dec[0]:.3f}, vy = {r2_dec[1]:.3f}")
print(f"median direction error (speed > 200 mm/s) = {np.median(ang_err[fast]):.1f} deg")

# %%
fig = plt.figure(figsize=(14.5, 8))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35, top=0.86)
t_plot = np.arange(30 * usable) * BIN
sl = slice(400 * usable, 430 * usable)
for k, (lab, col) in enumerate([("$v_x$", "tab:blue"), ("$v_y$", "tab:orange")]):
    ax = fig.add_subplot(gs[k, :2])
    ax.plot(t_plot, Yd[sl, k], color="k", lw=1.4, label="measured")
    ax.plot(t_plot, pred[sl, k], color=col, lw=1.4, label="decoded from spikes")
    for b in range(31):
        ax.axvline(b * usable * BIN, color="0.85", lw=0.6, zorder=0)
    ax.set(ylabel=f"{lab} (mm/s)",
           title="" if k else "Hand velocity decoded from the population on held-out "
                              "reaches (grey lines = reach boundaries)")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    if k:
        ax.set_xlabel("time (concatenated held-out reaches, s)")

ax = fig.add_subplot(gs[0, 2])
h = ax.hist2d(Yd[:, 0], pred[:, 0], bins=60, cmap="magma", norm=LogNorm(vmin=1),
              range=[[-900, 900], [-900, 900]])
fig.colorbar(h[3], ax=ax, label="20 ms bins")
ax.plot([-900, 900], [-900, 900], "w--", lw=1)
ax.set(xlabel="measured $v_x$ (mm/s)", ylabel="decoded $v_x$ (mm/s)",
       title=f"$R^2$ = {r2_dec[0]:.2f} ($v_x$), {r2_dec[1]:.2f} ($v_y$)")

ax = fig.add_subplot(gs[1, 2])
ax.hist(ang_err[fast], bins=np.arange(0, 181, 5), color="steelblue")
ax.axvline(np.median(ang_err[fast]), color="crimson", ls="--",
           label=f"median {np.median(ang_err[fast]):.0f}$\\degree$")
ax.set(xlabel="decoded $-$ measured direction (deg)", ylabel="20 ms bins",
       title="Direction read-out error\n(bins with speed > 200 mm/s)")
ax.legend(fontsize=8)

fig.suptitle("A linear read-out of the population recovers the hand velocity vector "
             "moment by moment", fontsize=13)
fig.savefig("fig06_population_decoding.png", bbox_inches="tight")

# %% [markdown]
# ## Summary
#
# Both phenomena are clearly present in this session.
#
# **Direction.** Of the units that fire at all during movement (mean rate above 0.5 Hz),
# about 85% are significantly tuned to reach direction (permutation test on the cosine
# modulation depth, p < 0.01), preferred directions cover the whole circle, and a cosine
# accounts for most of the variance across direction conditions. Directional tuning is already substantial during the instructed delay,
# before the hand leaves the start position, and its depth ramps up through the delay and
# peaks around movement onset, slightly ahead of peak hand speed.
#
# **Velocity.** Spiking leads the hand by tens of milliseconds during movement; if the
# instructed-delay bins are included in the same scan the apparent lead grows to 200 ms,
# because preparatory activity predicts a velocity that has not happened yet. Over the velocity plane, the
# firing-rate surfaces are not separable into "direction" and "speed" pieces: rate rises with
# speed along the preferred direction and is flat or falls along the opposite direction, which
# is the signature of speed acting as a gain on directional tuning rather than as an
# independent additive drive. Cross-validated Poisson GLMs make the same point quantitatively,
# with the direction x speed model scoring highest of the five feature sets tested and beating
# direction alone for essentially every unit. The absolute pseudo-$R^2$ values are small
# (about 0.018 for the best model) because a single 20 ms bin of one unit's spiking is mostly
# Poisson noise; what carries the argument is the ordering of the models, which is consistent
# across held-out folds, and not the absolute fraction of deviance explained.
#
# Two caveats are worth stating. The preferred directions recovered from the continuous
# velocity model only broadly agree with those from the trial-level cosine fits (circular
# $r$ = 0.58, median absolute difference about 36 degrees); the two measures are computed from
# different trial sets and different time windows, so exact agreement was not expected, but
# the scatter is larger than the classical picture of a single fixed preferred direction would
# suggest. And the lead time of spiking over the hand is not a single number: the population
# optimum on moving bins is 80 ms, while single-unit optima are spread over the whole range
# scanned, with one cluster at the negative edge.
#
# Finally, a linear read-out of the population tracks the measured hand velocity on held-out
# reaches ($R^2$ = 0.67 for $v_x$ and 0.51 for $v_y$), with a median direction error of about
# 18 degrees during fast movement, which is the practical statement of the same tuning.
