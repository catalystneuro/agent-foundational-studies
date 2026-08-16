# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reach direction and velocity tuning in macaque motor cortex
#
# **Dataset:** [DANDI:000128](https://dandiarchive.org/dandiset/000128), *MC_Maze*: macaque
# primary motor (M1) and dorsal premotor (PMd) cortex spiking activity during delayed reaching
# (Churchland & Kaufman, released through the Neural Latents Benchmark). One session from monkey
# Jenkins: 2295 successful trials, 182 sorted units recorded on two 96-channel Utah arrays, and
# hand position / velocity sampled at 1 kHz.
#
# **Task.** On each trial the monkey reaches from a central hold position to a peripheral target.
# On roughly a third of trials the workspace is empty and the reach is essentially straight; on
# the rest, virtual barriers force a curved trajectory around a maze. A variable delay separates
# target onset from the go cue, so planning and execution can be examined separately.
#
# **What this notebook demonstrates.**
#
# 1. Single units are strongly tuned for the direction of the reach, and the tuning is
#    approximately cosine-shaped (Georgopoulos et al., 1982).
# 2. That tuning is already present during the delay period, before movement.
# 3. Firing rate leads hand velocity by roughly 60 ms.
# 4. Movement **speed** scales the *gain* of the directional tuning rather than simply adding an
#    offset: the tuning curve grows with speed along the preferred direction and stays flat or
#    falls along the opposite direction (Moran & Schwartz, 1999).
# 5. A Poisson GLM (NeMoS) confirms that speed carries information beyond direction: adding a
#    speed term raises the cross-validated pseudo-R^2 by about 40% for 95% of units.
# 6. The instantaneous hand velocity vector can be read back out of the population with
#    R^2 ~ 0.6-0.7 and a median direction error under 20 degrees.
#
# All data are streamed from the DANDI S3 bucket with `remfile` (chunk-level disk caching); no
# whole-file download is performed. All analysis uses `pynapple` data structures.

# %% [markdown]
# ## Setup

# %%
import os
import numpy as np
import pandas as pd

np.seterr(all="ignore")

import matplotlib.pyplot as plt

import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

import jax

jax.config.update("jax_enable_x64", True)
import nemos as nmo

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

FIG = "figures"
CACHE = "cache"
os.makedirs(FIG, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

# S3 URL of sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb in DANDI:000128,
# resolved once with the DANDI API (see README for the resolving snippet).
URL = "https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a"

# %% [markdown]
# ## 1. Streaming the NWB file
#
# `remfile` fetches only the HDF5 chunks that are actually read and keeps them in a local disk
# cache, so re-running the notebook is fast without ever materialising the 690 MB file.

# %%
rem_file = remfile.File(URL, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()

print(nwbfile.session_description)
print("subject:", nwbfile.subject.subject_id, "|", nwbfile.subject.species)
print("behavior streams:", list(nwbfile.processing["behavior"].data_interfaces))
print("units:", len(nwbfile.units), "| trials:", len(nwbfile.trials))
print("electrode groups:", nwbfile.electrodes.to_dataframe().group_name.value_counts().to_dict())

# %% [markdown]
# ### A note on unit-to-array assignment
#
# The electrode table distinguishes the PMd array (channels 1-96) from the M1 array (97-192), but
# every one of the 182 units in this release points at a channel in the 1-87 range. The unit ->
# electrode mapping in this NWB conversion is therefore not usable to separate M1 from PMd, and we
# pool all units and speak of "motor cortex" throughout rather than making area-specific claims.

# %%
units_df = nwbfile.units.to_dataframe()
elec_idx = np.array([np.asarray(e.index)[0] for e in units_df["electrodes"]])
print("range of electrode indices referenced by units:", elec_idx.min(), "-", elec_idx.max())

# %% [markdown]
# ### Pull the arrays into pynapple objects
#
# Note that the NWB file labels the kinematics as metres, but the numeric values are millimetres
# (targets lie ~120 mm from the centre and peak speeds are ~800 mm/s). We keep mm and mm/s.

# %%
beh = nwbfile.processing["behavior"].data_interfaces
t_beh = beh["hand_vel"].timestamps[:]
hand_vel = beh["hand_vel"].data[:]
hand_pos = beh["hand_pos"].data[:]

vel = nap.TsdFrame(t=t_beh, d=hand_vel, columns=["vx", "vy"])
pos = nap.TsdFrame(t=t_beh, d=hand_pos, columns=["x", "y"])
spikes = nap.TsGroup({i: nap.Ts(t=np.asarray(s)) for i, s in enumerate(units_df["spike_times"])})

trials = nwbfile.trials.to_dataframe()
print(vel)
print(f"\n{len(spikes)} units, mean rate {np.mean(np.asarray(spikes.rates)):.1f} Hz, "
      f"session length {t_beh[-1]:.0f} s")

# %% [markdown]
# ## 2. Reach kinematics
#
# For each trial we take the movement-onset time supplied with the dataset, find the peak of the
# speed profile in the following 600 ms, and call the reach over when speed first drops below 20%
# of that peak. Reach direction is the direction of the hand displacement between onset and reach
# end.

# %%
dt = np.median(np.diff(t_beh[:1000]))
speed_full = np.hypot(hand_vel[:, 0], hand_vel[:, 1])
i_onset = np.searchsorted(t_beh, trials["move_onset_time"].values)
n_post = int(round(0.6 / dt))

rows = []
for k, i in enumerate(i_onset):
    sp = speed_full[i : i + n_post]
    i_pk = int(np.argmax(sp))
    below = np.where(sp[i_pk:] < 0.2 * sp[i_pk])[0]
    i_end = i_pk + (below[0] if len(below) else n_post - 1)
    disp = hand_pos[i + i_end] - hand_pos[i]
    rows.append(
        dict(
            t_onset=t_beh[i],
            t_end=t_beh[i + i_end],
            peak_speed=sp[i_pk],
            dur=i_end * dt,
            dist=np.hypot(*disp),
            direction=np.arctan2(disp[1], disp[0]),
        )
    )
K = pd.DataFrame(rows)
K["num_barriers"] = trials["num_barriers"].values
K["delay"] = trials["delay"].values
straight = K.num_barriers.values == 0  # unobstructed, near-straight reaches
print(K[["peak_speed", "dur", "dist"]].describe().loc[["mean", "std", "min", "max"]].to_string())
print(f"\nunobstructed reaches: {straight.sum()} / {len(K)}")

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
m = (t_beh > 100) & (t_beh < 120)
ax[0, 0].plot(t_beh[m], hand_pos[m, 0], lw=0.8, label="x")
ax[0, 0].plot(t_beh[m], hand_pos[m, 1], lw=0.8, label="y")
ax[0, 0].plot(t_beh[m], speed_full[m] / 10, lw=0.8, color="k", alpha=0.6, label="speed/10")
for mo in K.t_onset[(K.t_onset > 100) & (K.t_onset < 120)]:
    ax[0, 0].axvline(mo, color="r", ls=":", lw=0.8)
ax[0, 0].legend(fontsize=8)
ax[0, 0].set_xlabel("time (s)")
ax[0, 0].set_ylabel("mm  /  mm/s")
ax[0, 0].set_title("hand position & speed (red = move onset)")

cm = plt.get_cmap("hsv")
for k in range(0, len(K), 4):
    i = np.searchsorted(t_beh, K.t_onset[k])
    j = np.searchsorted(t_beh, K.t_end[k])
    p = hand_pos[i:j] - hand_pos[i]
    ax[0, 1].plot(p[:, 0], p[:, 1], lw=0.4, alpha=0.6,
                  color=cm((K.direction[k] % (2 * np.pi)) / (2 * np.pi)))
ax[0, 1].set_aspect("equal")
ax[0, 1].set_xlabel("x (mm)")
ax[0, 1].set_ylabel("y (mm)")
ax[0, 1].set_title("reach paths aligned to onset, coloured by endpoint direction")

ax[1, 0].hist(np.degrees(K.direction), bins=72, color="steelblue")
ax[1, 0].set_xlabel("reach direction (deg)")
ax[1, 0].set_ylabel("# trials")
ax[1, 0].set_title("endpoint direction distribution")

ax[1, 1].hist(K.peak_speed, bins=60, color="indianred")
ax[1, 1].set_xlabel("peak speed (mm/s)")
ax[1, 1].set_ylabel("# trials")
ax[1, 1].set_title("peak reach speed distribution")
fig.tight_layout()
fig.savefig(f"{FIG}/fig01_kinematics.png", dpi=140)

# %% [markdown]
# The maze trials produce clearly curved paths, and target geometry means the direction
# distribution is discrete rather than uniform. Peak speeds span roughly 450-1350 mm/s, a
# three-fold range that we will exploit below.

# %% [markdown]
# ## 3. Direction tuning during movement
#
# We count spikes in a window from 100 ms before to 300 ms after movement onset (motor cortex
# leads the movement, so the window is shifted early), convert to a rate, and fit
#
# $$ r(\theta) = b_0 + b_1\cos\theta + b_2\sin\theta = b_0 + d\cos(\theta - \theta_{pref}) $$
#
# Significance comes from a permutation test that shuffles the trial-to-direction assignment.

# %%
MOVE_WIN = (-0.10, 0.30)
PLAN_WIN = (-0.40, -0.05)  # relative to the go cue


def epoch_rates(spk, starts, win):
    ep = nap.IntervalSet(start=starts + win[0], end=starts + win[1])
    cnt = spk.restrict(ep).count(ep=ep, bin_size=win[1] - win[0])
    return np.asarray(cnt.values) / (win[1] - win[0])


def cosine_fit(rate, theta):
    X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    beta, *_ = np.linalg.lstsq(X, rate, rcond=None)
    pred = X @ beta
    r2 = 1 - ((rate - pred) ** 2).sum(0) / ((rate - rate.mean(0)) ** 2).sum(0)
    return dict(b0=beta[0], pref_dir=np.arctan2(beta[2], beta[1]),
                depth=np.hypot(beta[1], beta[2]), r2=r2,
                mdi=np.hypot(beta[1], beta[2]) / np.where(beta[0] > 0, beta[0], np.nan))


def permutation_p(rate, theta, nperm=500, seed=0):
    rng = np.random.default_rng(seed)
    obs = cosine_fit(rate, theta)["r2"]
    cnt = np.zeros_like(obs)
    for _ in tqdm(range(nperm), desc="permutations", leave=False):
        cnt += cosine_fit(rate, theta[rng.permutation(len(theta))])["r2"] >= obs
    return (cnt + 1) / (nperm + 1)


theta = K.direction.values
rate_mv = epoch_rates(spikes, K.t_onset.values, MOVE_WIN)
rate_pl = epoch_rates(spikes, trials["go_cue_time"].values, PLAN_WIN)
long_delay = K.delay.values >= 450  # the planning window must sit inside the delay period

res_all = cosine_fit(rate_mv, theta)
res_str = cosine_fit(rate_mv[straight], theta[straight])
res_pl = cosine_fit(rate_pl[long_delay], theta[long_delay])

tune = pd.DataFrame(
    dict(
        unit=np.arange(rate_mv.shape[1]),
        mean_rate=rate_mv.mean(0),
        r2=res_all["r2"], pref_dir=res_all["pref_dir"],
        r2_str=res_str["r2"], pref_dir_str=res_str["pref_dir"],
        depth_str=res_str["depth"], mdi_str=res_str["mdi"],
        r2_plan=res_pl["r2"], pref_dir_plan=res_pl["pref_dir"],
    )
)
tune["p_str"] = permutation_p(rate_mv[straight], theta[straight])
tune["p_plan"] = permutation_p(rate_pl[long_delay], theta[long_delay])

sig = tune.p_str < 0.01
print(f"{sig.mean():.0%} of {len(tune)} units are significantly direction-tuned "
      f"during movement (permutation p < 0.01)")
print(f"median cosine R^2: {tune.r2_str.median():.3f} (max {tune.r2_str.max():.3f}); "
      f"on all trials incl. curved maze reaches: {tune.r2.median():.3f}")
print(f"{(tune.p_plan < 0.01).mean():.0%} are already tuned during the delay period")

# %% [markdown]
# ### Rasters and PSTHs for the two most strongly tuned units

# %%
NBINS = 8
EDGES = np.linspace(-np.pi, np.pi, NBINS + 1)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
COLORS = [plt.get_cmap("hsv")(((c + np.pi) % (2 * np.pi)) / (2 * np.pi)) for c in CENTERS]
dbin = np.clip(np.digitize(theta, EDGES) - 1, 0, NBINS - 1)


def psth_by_direction(spk_unit, onsets, db, win=(-0.6, 0.6), bs=0.02):
    pe = nap.compute_perievent(spk_unit, nap.Ts(t=onsets), window=win)
    edges = np.arange(win[0], win[1] + bs, bs)
    rates = np.zeros((NBINS, len(edges) - 1))
    rasters = [[] for _ in range(NBINS)]
    for k in sorted(pe.keys()):
        s = pe[k].index.values
        rasters[db[k]].append(s)
        rates[db[k]] += np.histogram(s, edges)[0]
    for b in range(NBINS):
        rates[b] /= max(len(rasters[b]), 1) * bs
    return edges[:-1] + bs / 2, rates, rasters


ex2 = tune.sort_values("r2_str", ascending=False).unit.values[:2]
sel = np.where(straight)[0]
rng = np.random.default_rng(1)
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, height_ratios=[2.2, 1])
for j, u in enumerate(ex2):
    tcx, rates, rasters = psth_by_direction(spikes[u], K.t_onset.values[sel], dbin[sel])
    nonempty = [b for b in range(NBINS) if len(rasters[b]) > 0]
    axr = fig.add_subplot(gs[0, j])
    y, ticks = 0, []
    for b in nonempty:
        rs = rasters[b]
        if len(rs) > 30:
            rs = [rs[i] for i in rng.choice(len(rs), 30, replace=False)]
        y0 = y
        for s in rs:
            axr.plot(s, np.full_like(s, y), "|", ms=3.5, color=COLORS[b], markeredgewidth=0.9)
            y += 1
        ticks.append((y0 + y) / 2)
        axr.axhline(y, color="k", lw=0.4, alpha=0.3)
    axr.axvline(0, color="k", ls="--", lw=1)
    axr.set_xlim(-0.6, 0.6)
    axr.set_ylim(0, y)
    axr.set_yticks(ticks)
    axr.set_yticklabels([f"{np.degrees(CENTERS[b]):.0f}$\\degree$" for b in nonempty], fontsize=8)
    axr.set_ylabel("reach direction")
    axr.set_title(f"unit {u}  |  cosine $R^2$={tune.loc[u,'r2_str']:.2f}, "
                  f"PD={np.degrees(tune.loc[u,'pref_dir_str']):.0f}$\\degree$", pad=8)
    axp = fig.add_subplot(gs[1, j])
    for b in nonempty:
        axp.plot(tcx, rates[b], color=COLORS[b], lw=1.4,
                 label=f"{np.degrees(CENTERS[b]):.0f}$\\degree$")
    axp.axvline(0, color="k", ls="--", lw=1)
    axp.axvspan(*MOVE_WIN, color="gray", alpha=0.15)
    axp.set_xlim(-0.6, 0.6)
    axp.set_xlabel("time from movement onset (s)")
    axp.set_ylabel("firing rate (Hz)")
    if j == 1:
        axp.legend(fontsize=7, ncol=2, title="reach dir.", title_fontsize=7,
                   loc="upper left", bbox_to_anchor=(1.01, 1.05))
fig.suptitle("Direction-tuned responses on unobstructed reaches (shaded = analysis window)", y=0.98)
fig.savefig(f"{FIG}/fig02_raster_psth.png", dpi=140, bbox_inches="tight")

# %% [markdown]
# ### Polar tuning curves
#
# One direction bin (around -68 deg) contains no unobstructed reaches because of the target
# geometry, so it is left out of the plots.

# %%
top8 = tune.sort_values("r2_str", ascending=False).unit.values[:8]
thgrid = np.linspace(-np.pi, np.pi, 200)
fig, axes = plt.subplots(2, 4, figsize=(15, 8.5), subplot_kw=dict(projection="polar"))
for ax, u in zip(axes.ravel(), top8):
    r = rate_mv[straight, u]
    th = theta[straight]
    b = dbin[straight]
    nb = np.array([(b == i).sum() for i in range(NBINS)])
    mn = np.array([r[b == i].mean() if nb[i] else np.nan for i in range(NBINS)])
    se = np.array([r[b == i].std() / np.sqrt(nb[i]) if nb[i] else np.nan for i in range(NBINS)])
    X = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    curve = beta[0] + beta[1] * np.cos(thgrid) + beta[2] * np.sin(thgrid)
    rmax = np.nanmax([np.nanmax(mn + se), curve.max()]) * 1.12
    ax.plot(np.r_[CENTERS, CENTERS[0]], np.r_[mn, mn[0]], "o-", color="k", ms=4, lw=1.2)
    ax.errorbar(CENTERS, mn, yerr=se, fmt="none", ecolor="k", capsize=2, lw=1)
    ax.plot(thgrid, np.clip(curve, 0, None), color="crimson", lw=1.8)
    ax.annotate("", xy=(tune.loc[u, "pref_dir_str"], rmax * 0.95),
                xytext=(tune.loc[u, "pref_dir_str"], 0),
                arrowprops=dict(arrowstyle="-|>", color="navy", lw=1.6))
    ax.set_ylim(0, rmax)
    ax.set_rlabel_position(285)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_title(f"unit {u}   $R^2$={tune.loc[u,'r2_str']:.2f}   "
                 f"MDI={tune.loc[u,'mdi_str']:.2f}", fontsize=10, pad=22)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.4)
fig.suptitle("Reach-direction tuning curves on unobstructed reaches\n"
             "black = binned mean $\\pm$ SEM (Hz)   |   red = cosine fit   |   "
             "blue arrow = preferred direction", y=1.02, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=4.0, w_pad=2.0)
fig.savefig(f"{FIG}/fig03_polar_tuning.png", dpi=140, bbox_inches="tight")

# %% [markdown]
# ### Population summary
#
# The modulation depth index (MDI) is the cosine amplitude divided by the fitted baseline rate.

# %%
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
ax[0, 0].hist([tune.r2_str[sig], tune.r2_str[~sig]], bins=25, stacked=True,
              color=["steelblue", "lightgray"], label=["p<0.01", "n.s."])
ax[0, 0].set_xlabel("cosine $R^2$ (unobstructed reaches)")
ax[0, 0].set_ylabel("# units")
ax[0, 0].legend()
ax[0, 0].set_title(f"{sig.mean():.0%} of {len(tune)} units significantly tuned")

axp = fig.add_subplot(2, 3, 2, projection="polar")
ax[0, 1].remove()
axp.hist(tune.pref_dir_str[sig], bins=18, color="steelblue")
axp.set_title("preferred directions\n(significantly tuned units)", pad=18, fontsize=10)

ax[0, 2].hist(tune.mdi_str[sig & np.isfinite(tune.mdi_str)], bins=25, color="seagreen")
ax[0, 2].set_xlabel("modulation depth index (depth / baseline)")
ax[0, 2].set_ylabel("# units")
ax[0, 2].set_title("tuning strength")

ax[1, 0].scatter(tune.r2, tune.r2_str, s=14, c="k", alpha=0.6)
lim = [0, max(tune.r2_str.max(), tune.r2.max()) * 1.05]
ax[1, 0].plot(lim, lim, "r--", lw=1)
ax[1, 0].set_xlim(lim)
ax[1, 0].set_ylim(lim)
ax[1, 0].set_xlabel("$R^2$, all trials (incl. maze/curved)")
ax[1, 0].set_ylabel("$R^2$, unobstructed reaches")
ax[1, 0].set_title("endpoint direction explains straight\nreaches better than curved ones")

ax[1, 1].scatter(tune.r2_plan, tune.r2, s=14, c="k", alpha=0.6)
ax[1, 1].set_xlabel("$R^2$, late-delay (planning) window")
ax[1, 1].set_ylabel("$R^2$, peri-movement window")
ax[1, 1].set_title("planning vs execution tuning")

dd = np.angle(np.exp(1j * (tune.pref_dir_plan - tune.pref_dir)))
both = (tune.r2 > 0.02) & (tune.r2_plan > 0.02)
n = int(both.sum())
z = np.exp(1j * dd[both]).mean()
R = np.abs(z)
Z = n * R ** 2
p_ray = np.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n))
ax[1, 2].hist(np.degrees(dd[both]), bins=18, color="darkorange")
ax[1, 2].axvline(np.degrees(np.angle(z)), color="k", ls="--", lw=1.5)
ax[1, 2].set_xlabel("PD(plan) $-$ PD(move)  (deg)")
ax[1, 2].set_ylabel("# units")
ax[1, 2].set_title(f"PD partially preserved from planning to execution\n"
                   f"n={n}, mean $\\Delta$={np.degrees(np.angle(z)):.0f}$\\degree$, "
                   f"R={R:.2f}, Rayleigh p={p_ray:.3f}")
fig.tight_layout()
fig.savefig(f"{FIG}/fig04_population_direction.png", dpi=140)

print(f"planning-to-execution preferred-direction shift: mean {np.degrees(np.angle(z)):.0f} deg, "
      f"resultant R={R:.2f}, Rayleigh p={p_ray:.3g} (n={n})")

# %% [markdown]
# Preferred directions tile the circle with a modest bias, tuning is strong (median MDI ~0.5, i.e.
# the cosine amplitude is about half the baseline rate), and the same units are already tuned
# during the delay. The preferred direction is *partially* preserved from planning to execution:
# the distribution of PD differences is significantly clustered, but with a mean shift of about
# 33 degrees and a resultant length of only 0.31, so a substantial fraction of units do rotate.
#
# Note also that the cosine model does clearly worse on the curved maze reaches. Endpoint
# direction is simply the wrong description of what the hand did on those trials, which motivates
# moving to the continuous velocity signal.

# %% [markdown]
# ## 4. How far does neural activity lead the hand?
#
# We define a movement epoch per trial (onset to reach end, padded by 50 ms), bin spikes at 20 ms,
# smooth with a 50 ms Gaussian, and regress the smoothed rate on the hand velocity vector taken at
# a range of lags. The absolute $R^2$ is small because single-unit counts are Poisson-noisy; what
# matters is where the curve peaks.

# %%
ep_move = nap.IntervalSet(start=K.t_onset.values - 0.05, end=K.t_end.values + 0.05)
print(f"{len(ep_move)} movement epochs, {ep_move.tot_length():.0f} s in total")


def shifted_vel(v, lag):
    """Re-timestamp velocity so that spiking at t is paired with velocity at t + lag."""
    return nap.TsdFrame(t=v.index.values - lag, d=v.values, columns=["vx", "vy"])


lags = np.arange(-0.30, 0.305, 0.01)
cnt_lag = spikes.count(0.02, ep=ep_move).smooth(0.05)
tb_lag = cnt_lag.index.values
Y_lag = np.asarray(cnt_lag.values) / 0.02
R2_lag = np.zeros((len(lags), Y_lag.shape[1]))
for i, lag in enumerate(tqdm(lags, desc="lag scan")):
    V = np.asarray(vel.interpolate(nap.Ts(tb_lag + lag)).values)
    g = np.isfinite(V).all(1)
    X = np.column_stack([np.ones(g.sum()), V[g]])
    beta, *_ = np.linalg.lstsq(X, Y_lag[g], rcond=None)
    resid = Y_lag[g] - X @ beta
    R2_lag[i] = 1 - (resid ** 2).sum(0) / ((Y_lag[g] - Y_lag[g].mean(0)) ** 2).sum(0)

pop_lag = np.nanmean(R2_lag, 1)
LAG = float(lags[np.argmax(pop_lag)])
ok_lag = np.nanmax(R2_lag, 0) > 0.02
best_lag = lags[np.nanargmax(np.where(np.isnan(R2_lag), -np.inf, R2_lag)[:, ok_lag], 0)]
print(f"population optimum: neural activity leads velocity by {LAG*1000:.0f} ms")
print(f"per-unit optimum (n={ok_lag.sum()} units with R^2>0.02): median {np.median(best_lag)*1000:.0f} ms, "
      f"IQR {np.percentile(best_lag,[25,75])*1000} ms")

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].plot(lags * 1000, pop_lag, "k", lw=2)
ax[0].axvline(LAG * 1000, color="crimson", ls="--")
ax[0].axvline(0, color="gray", lw=0.8)
ax[0].set_xlabel("lag (ms):  neural activity leads velocity $\\rightarrow$")
ax[0].set_ylabel("mean $R^2$ (linear velocity model)")
ax[0].set_title(f"population optimum at {LAG*1000:.0f} ms")
for u in np.argsort(np.nanmax(R2_lag, 0))[-12:]:
    ax[1].plot(lags * 1000, R2_lag[:, u] / np.nanmax(R2_lag[:, u]), lw=0.9, alpha=0.8)
ax[1].axvline(0, color="gray", lw=0.8)
ax[1].set_xlabel("lag (ms)")
ax[1].set_ylabel("$R^2$ (normalised per unit)")
ax[1].set_title("12 most velocity-related units")
ax[2].hist(best_lag * 1000, bins=np.arange(-305, 306, 20), color="steelblue")
ax[2].axvline(np.median(best_lag) * 1000, color="crimson", ls="--",
              label=f"median {np.median(best_lag)*1000:.0f} ms")
ax[2].set_xlabel("per-unit optimal lag (ms)")
ax[2].set_ylabel("# units")
ax[2].legend()
ax[2].set_title(f"n={ok_lag.sum()} units with $R^2$>0.02")
fig.suptitle("Motor-cortical activity leads hand velocity", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{FIG}/fig06_lag_scan.png", dpi=140)

# %% [markdown]
# The population optimum sits at +60 ms, consistent with motor cortex driving rather than
# following the hand. Individual units span a wide range of lags (the counts piled up at the
# +/-300 ms edges belong to units whose $R^2$ was still rising at the edge of the scan, so their
# optima are not resolved here).

# %% [markdown]
# ## 5. Velocity tuning: direction and speed together
#
# At the trial level, peak reach speed is strongly confounded with reach direction, because the
# targets sit at different distances (correlation of peak speed with $\sin\theta$ is about
# $-0.47$). Within a reach, however, direction is roughly constant while speed sweeps from zero
# through the peak and back, so the *continuous* signal covers the direction x speed plane almost
# independently (correlations of 0.15 and 0.01). All speed results therefore come from the
# continuous signal.

# %%
vel_shift = shifted_vel(vel, LAG)
Vm = np.asarray(vel_shift.restrict(ep_move).values)
spd_m = np.hypot(Vm[:, 0], Vm[:, 1])
ang_m = np.arctan2(Vm[:, 1], Vm[:, 0])
mm = spd_m > 50
print(f"continuous: corr(speed, cos dir)={np.corrcoef(spd_m[mm], np.cos(ang_m[mm]))[0,1]:.3f}, "
      f"corr(speed, sin dir)={np.corrcoef(spd_m[mm], np.sin(ang_m[mm]))[0,1]:.3f}")
print(f"trial level: corr(peak speed, cos dir)={np.corrcoef(K.peak_speed, np.cos(K.direction))[0,1]:.3f}, "
      f"corr(peak speed, sin dir)={np.corrcoef(K.peak_speed, np.sin(K.direction))[0,1]:.3f}")

# %% [markdown]
# ### Tuning in the (vx, vy) plane

# %%
tc2d, bins2d = nap.compute_2d_tuning_curves(group=spikes, features=vel_shift, nb_bins=12,
                                            ep=ep_move, minmax=(-900, 900, -900, 900))
TC2D = np.array([tc2d[k] for k in sorted(tc2d)])
occ2d = np.histogram2d(Vm[:, 0], Vm[:, 1], bins=[12, 12], range=[[-900, 900], [-900, 900]])[0]
mask2d = occ2d > 1000  # at least 1 s of hand-velocity data per bin

# ### Tuning in polar (direction, speed) coordinates
NDIR, NSPD = 12, 6
SPD_RANGE = (50.0, 1150.0)
feat_polar = nap.TsdFrame(
    t=vel_shift.index.values,
    d=np.column_stack([np.arctan2(vel_shift.values[:, 1], vel_shift.values[:, 0]),
                       np.hypot(vel_shift.values[:, 0], vel_shift.values[:, 1])]),
    columns=["dir", "speed"],
)
tc_ds, bins_ds = nap.compute_2d_tuning_curves(
    group=spikes, features=feat_polar, nb_bins=(NDIR, NSPD), ep=ep_move,
    minmax=(-np.pi, np.pi, *SPD_RANGE))
TCDS = np.array([tc_ds[k] for k in sorted(tc_ds)])
dctr, sctr = bins_ds[0], bins_ds[1]
occ_ds = np.histogram2d(ang_m, spd_m, bins=[NDIR, NSPD], range=[[-np.pi, np.pi], SPD_RANGE])[0]
okb = occ_ds > 200

# preferred direction from the continuous maps, taken at the highest speeds
hi = np.nanmean(np.where(okb[None], TCDS, np.nan)[:, :, -3:], axis=2)
pref_dir_cont = np.angle(np.nansum(np.exp(1j * dctr)[None] * np.nan_to_num(hi), axis=1))
tune["pref_dir_cont"] = pref_dir_cont

# speed slope along the preferred and the opposite direction
TCm = np.where(okb[None], TCDS, np.nan)


def slope_at(u, ang):
    j = int(np.argmin(np.abs(np.angle(np.exp(1j * (dctr - ang))))))
    y = TCm[u, j]
    g = np.isfinite(y)
    return np.polyfit(sctr[g], y[g], 1)[0] * 100 if g.sum() > 2 else np.nan


tune["speed_slope_pd"] = [slope_at(u, pref_dir_cont[u]) for u in range(len(tune))]
tune["speed_slope_anti"] = [slope_at(u, pref_dir_cont[u] + np.pi) for u in range(len(tune))]
a, b_ = tune.speed_slope_pd[sig], tune.speed_slope_anti[sig]
g = np.isfinite(a) & np.isfinite(b_)
print(f"speed slope (Hz per 100 mm/s): preferred direction median {a.median():.2f}, "
      f"opposite direction {b_.median():.2f}")
print("Wilcoxon:", wilcoxon(a[g], b_[g]))

# %%
ex6 = tune.sort_values("r2_str", ascending=False).unit.values[:6]
fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
ext = [bins2d[0][0], bins2d[0][-1], bins2d[1][0], bins2d[1][-1]]
for ax, u in zip(axes.ravel()[:6], ex6):
    M = np.where(mask2d, TC2D[u], np.nan).T
    im = ax.imshow(M, origin="lower", extent=ext, cmap="viridis", aspect="equal")
    ax.arrow(0, 0, 650 * np.cos(pref_dir_cont[u]), 650 * np.sin(pref_dir_cont[u]),
             color="crimson", width=20, head_width=80, length_includes_head=True)
    ax.set_title(f"unit {u}", fontsize=10)
    ax.set_xlabel("$v_x$ (mm/s)")
    ax.set_ylabel("$v_y$ (mm/s)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Hz")
axo = axes.ravel()[6]
im = axo.imshow(occ_ds.T, origin="lower", cmap="magma", aspect="auto",
                extent=[-180, 180, sctr[0], sctr[-1]])
axo.set_xlabel("movement direction (deg)")
axo.set_ylabel("speed (mm/s)")
axo.set_title("sampling of the direction $\\times$ speed plane", fontsize=10)
plt.colorbar(im, ax=axo, fraction=0.046, label="samples (1 kHz)")
axc = axes.ravel()[7]
axc.scatter(np.degrees(K.direction), K.peak_speed, s=3, alpha=0.25, c="k")
axc.set_xlabel("reach direction (deg)")
axc.set_ylabel("peak speed (mm/s)")
axc.set_title("trial level: peak speed is confounded\nwith direction (target geometry)", fontsize=10)
fig.suptitle(f"Hand-velocity tuning during movement (activity leads velocity by {LAG*1000:.0f} ms; "
             f"red arrow = preferred direction)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{FIG}/fig05_velocity_maps.png", dpi=140)

# %% [markdown]
# ### Speed scales the gain of directional tuning

# %%
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.35)
for i, u in enumerate(ex6[:4]):
    M = TCm[u]
    a1 = fig.add_subplot(gs[0, i])
    im = a1.pcolormesh(np.degrees(dctr), sctr, M.T, cmap="viridis", shading="nearest")
    a1.set_xlabel("direction (deg)")
    if i == 0:
        a1.set_ylabel("speed (mm/s)")
    a1.set_title(f"unit {u}", fontsize=10)
    plt.colorbar(im, ax=a1, fraction=0.046, label="Hz" if i == 3 else None)
    a2 = fig.add_subplot(gs[1, i])
    jp = int(np.argmin(np.abs(np.angle(np.exp(1j * (dctr - pref_dir_cont[u]))))))
    ja = int(np.argmin(np.abs(np.angle(np.exp(1j * (dctr - pref_dir_cont[u] - np.pi))))))
    a2.plot(sctr, M[jp], "o-", color="crimson", label="preferred dir.")
    a2.plot(sctr, M[ja], "o-", color="steelblue", label="opposite dir.")
    a2.set_xlabel("speed (mm/s)")
    a2.set_ylabel("firing rate (Hz)")
    if i == 0:
        a2.legend(fontsize=8)

axa = fig.add_subplot(gs[2, 0])
axa.scatter(tune.speed_slope_anti[sig], tune.speed_slope_pd[sig], s=16, c="k", alpha=0.6)
lim = np.nanpercentile(np.r_[tune.speed_slope_pd[sig], tune.speed_slope_anti[sig]], [1, 99])
axa.plot(lim, lim, "r--", lw=1)
axa.axhline(0, lw=0.6, c="gray")
axa.axvline(0, lw=0.6, c="gray")
axa.set_xlabel("slope, opposite dir.")
axa.set_ylabel("slope, preferred dir.")
axa.set_title("speed slope (Hz per 100 mm/s)", fontsize=10)

axb = fig.add_subplot(gs[2, 1])
axb.hist([tune.speed_slope_pd[sig].dropna(), tune.speed_slope_anti[sig].dropna()], bins=20,
         color=["crimson", "steelblue"], label=["preferred", "opposite"])
axb.axvline(0, c="k", lw=0.8)
axb.legend(fontsize=8)
axb.set_xlabel("speed slope (Hz per 100 mm/s)")
axb.set_ylabel("# units")
axb.set_title(f"median {a.median():.2f} vs {b_.median():.2f}", fontsize=10)

axc = fig.add_subplot(gs[2, 2:])
aligned = np.full_like(TCDS, np.nan)
for u in range(len(TCDS)):
    sh = int(np.argmin(np.abs(np.angle(np.exp(1j * (dctr - pref_dir_cont[u]))))))
    mx = np.nanmax(TCm[u])
    if mx > 0:
        aligned[u] = np.roll(TCm[u], -sh, axis=0) / mx
A = np.nanmean(aligned[sig.values], 0)
rel = np.degrees(np.angle(np.exp(1j * (dctr - dctr[0]))))
order = np.argsort(rel)
for si in range(len(sctr)):
    axc.plot(np.sort(rel), A[order, si], "o-", lw=1.4,
             color=plt.get_cmap("plasma")(si / (len(sctr) - 1)), label=f"{sctr[si]:.0f} mm/s")
axc.set_xlabel("direction relative to each unit's preferred direction (deg)")
axc.set_ylabel("normalised firing rate")
axc.legend(fontsize=8, ncol=2, title="speed", title_fontsize=8)
axc.set_title(f"population-average tuning grows with speed (n={sig.sum()} tuned units)", fontsize=10)
fig.suptitle("Speed scales the gain of directional tuning", fontsize=13, y=0.98)
fig.savefig(f"{FIG}/fig07_speed_gain.png", dpi=140, bbox_inches="tight")

# %% [markdown]
# This is the key panel. Aligning every tuned unit to its own preferred direction and averaging
# gives a clean cosine whose *amplitude* grows monotonically with speed while the baseline is
# nearly unchanged. Equivalently, firing rate rises with speed along the preferred direction
# (median +0.18 Hz per 100 mm/s) and falls slightly along the opposite direction (-0.08 Hz per
# 100 mm/s), a highly significant difference. That is multiplicative gain modulation, not an
# additive speed signal.

# %% [markdown]
# ## 6. Poisson GLMs with NeMoS
#
# The tuning-curve analysis above is descriptive. To ask whether speed carries information
# *beyond* direction we fit four nested Poisson GLMs to 50 ms spike counts and compare their
# cross-validated deviance-based pseudo-$R^2$. Folds are grouped by reach, so no reach contributes
# to both training and test. Ridge strength is held fixed across models so the comparison is fair.
#
# This cell takes roughly 10 minutes on a laptop CPU.

# %%
BIN_GLM = 0.05
REG = 1e-3
NFOLD = 5

cnt_glm = spikes.count(BIN_GLM, ep=ep_move)
tb_glm = cnt_glm.index.values
Vg = np.asarray(shifted_vel(vel, LAG).interpolate(nap.Ts(tb_glm)).values)
spd_g = np.hypot(Vg[:, 0], Vg[:, 1])
ang_g = np.arctan2(Vg[:, 1], Vg[:, 0])
keep_bins = np.isfinite(spd_g) & (spd_g > 50.0)
trial_id = np.searchsorted(np.asarray(ep_move.start), tb_glm, side="right") - 1

y_all = np.asarray(cnt_glm.values)[keep_bins].astype(float)
ang_g, spd_g = ang_g[keep_bins], np.clip(spd_g[keep_bins], 50.0, 1200.0)
tid = trial_id[keep_bins]
keep_u = y_all.sum(0) >= 200  # need enough spikes to fit a Poisson GLM
y = y_all[:, keep_u]
print(f"design: {y.shape[0]} bins x {y.shape[1]}/{y_all.shape[1]} units")


def poisson_pr2(yt, mu):
    mu = np.clip(mu, 1e-9, None)
    ybar = yt.mean(0, keepdims=True)
    dev = 2 * np.sum(np.where(yt > 0, yt * np.log(yt / mu), 0.0) - (yt - mu), 0)
    dev0 = 2 * np.sum(np.where(yt > 0, yt * np.log(yt / ybar), 0.0) - (yt - ybar), 0)
    return 1 - dev / dev0


b_dir = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, label="direction")
b_spd = nmo.basis.BSplineEval(n_basis_funcs=5, label="speed")
MODELS = {
    "dir": (b_dir, (ang_g,)),
    "speed": (b_spd, (spd_g,)),
    "add": (b_dir + b_spd, (ang_g, spd_g)),
    "full": (b_dir * b_spd, (ang_g, spd_g)),
}

rng = np.random.default_rng(0)
utrials = np.unique(tid)
fold = (rng.permutation(len(utrials)) % NFOLD)[np.searchsorted(utrials, tid)]

scores = {}
for name, (basis, args) in MODELS.items():
    X = np.asarray(basis.compute_features(*args))
    s = np.zeros((NFOLD, y.shape[1]))
    for f in tqdm(range(NFOLD), desc=f"GLM [{name}] ({X.shape[1]} features)"):
        tr, te = fold != f, fold == f
        glm = nmo.glm.PopulationGLM(
            solver_name="LBFGS",
            regularizer=nmo.regularizer.Ridge(),
            regularizer_strength=REG,
            solver_kwargs=dict(tol=1e-7, maxiter=600),
        ).fit(X[tr], y[tr])
        s[f] = poisson_pr2(y[te], np.asarray(glm.predict(X[te])))
    scores[name] = s.mean(0)

for k in ["dir", "speed", "add", "full"]:
    print(f"  {k:6s} median CV pseudo-R^2 = {np.median(scores[k]):.4f}")
print("\ndirection + speed vs direction only:", wilcoxon(scores["add"], scores["dir"]))
print(f"units improved by adding speed: {np.mean(scores['add'] > scores['dir']):.1%}")
print("interaction vs separable:", wilcoxon(scores["full"], scores["add"]))
print(f"units improved by the interaction: {np.mean(scores['full'] > scores['add']):.1%}")

# %%
order_m = ["dir", "speed", "add", "full"]
names_m = {"dir": "direction", "speed": "speed", "add": "direction + speed",
           "full": "direction $\\times$ speed"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].violinplot([scores[k] for k in order_m], showmedians=True)
ax[0].set_xticks(range(1, 5))
ax[0].set_xticklabels([names_m[k] for k in order_m], fontsize=9, rotation=12, ha='right')
ax[0].set_ylabel("cross-validated pseudo-$R^2$")
ax[0].set_title("nested Poisson GLMs (NeMoS), 50 ms bins")
ax[0].axhline(0, color="k", lw=0.6)
for i, k in enumerate(order_m):
    ax[0].text(i + 1, np.median(scores[k]), f"  {np.median(scores[k]):.3f}", fontsize=8, va="bottom")
ax[1].scatter(scores["dir"], scores["add"], s=16, c="k", alpha=0.6)
lim = [min(scores["dir"].min(), 0), max(scores["add"].max(), scores["dir"].max()) * 1.05]
ax[1].plot(lim, lim, "r--", lw=1)
ax[1].set_xlim(lim)
ax[1].set_ylim(lim)
ax[1].set_xlabel("direction only")
ax[1].set_ylabel("direction + speed")
w = wilcoxon(scores["add"], scores["dir"])
ax[1].set_title(f"{np.mean(scores['add']>scores['dir']):.0%} of units improve when speed\n"
                f"is added (Wilcoxon p={w.pvalue:.1e})")
g1 = scores["add"] - scores["dir"]
g2 = scores["full"] - scores["add"]
ax[2].hist([g1, g2], bins=25, color=["mediumpurple", "darkorange"],
           label=["+ speed (separable)", "+ direction $\\times$ speed interaction"])
ax[2].axvline(0, color="k", lw=0.8)
ax[2].set_xlabel("gain in cross-validated pseudo-$R^2$")
ax[2].set_ylabel("# units")
ax[2].legend(fontsize=8)
ax[2].set_title(f"separable speed term: median +{np.median(g1):.4f}\n"
                f"further interaction: median +{np.median(g2):.4f}")
fig.suptitle("Speed carries information beyond direction; because of the exponential link the "
             "separable model\nis already multiplicative in rate, and the explicit interaction "
             "adds little", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(f"{FIG}/fig09_glm_models.png", dpi=140)

# %% [markdown]
# Direction alone reaches a median cross-validated pseudo-$R^2$ of 0.022, speed alone only 0.006,
# and direction + speed 0.030: adding speed improves 95% of units (Wilcoxon p ~ 1e-27). Adding an
# explicit direction x speed interaction on top of that gains very little (median +0.001, 65% of
# units, p = 0.04).
#
# That is not a contradiction of the gain modulation seen in section 5. A Poisson GLM uses an
# exponential link, so a model that is *additive* in $\log\lambda$,
# $\log\lambda = f(\theta) + g(v)$, is already *multiplicative* in rate,
# $\lambda = e^{f(\theta)} e^{g(v)}$. The separable GLM is therefore the formal statement of
# "speed scales the gain of directional tuning", and the fact that little is left for the
# interaction term says the gain modulation is close to purely multiplicative.

# %% [markdown]
# ## 7. Reading velocity back out of the population
#
# If direction and speed are jointly encoded, a linear readout of the population should recover
# the velocity vector. We smooth binned rates (100 ms Gaussian), fit ridge regression to
# $(v_x, v_y)$, and cross-validate with folds grouped by reach. For comparison we also run
# pynapple's Bayesian decoder on direction alone, using direction tuning curves estimated on the
# training folds.

# %%
BIN_DEC, SMOOTH = 0.05, 0.1
cnt_d = spikes.count(BIN_DEC, ep=ep_move)
rate_d = np.asarray(cnt_d.smooth(SMOOTH).values) / BIN_DEC
tb_d = cnt_d.index.values
Vd = np.asarray(shifted_vel(vel, LAG).interpolate(nap.Ts(tb_d)).values)
spd_d = np.hypot(Vd[:, 0], Vd[:, 1])
ang_d = np.arctan2(Vd[:, 1], Vd[:, 0])
keep_d = np.isfinite(spd_d) & (spd_d > 100)
grp = (np.searchsorted(np.asarray(ep_move.start), tb_d, side="right") - 1)[keep_d]
Xr, Yv, Ya = rate_d[keep_d], Vd[keep_d], ang_d[keep_d]

pred = np.zeros_like(Yv)
for tr, te in GroupKFold(n_splits=NFOLD).split(Xr, Yv, groups=grp):
    mu, sd = Xr[tr].mean(0), Xr[tr].std(0) + 1e-9
    pred[te] = Ridge(alpha=100.0).fit((Xr[tr] - mu) / sd, Yv[tr]).predict((Xr[te] - mu) / sd)
r2_dec = 1 - ((Yv - pred) ** 2).sum(0) / ((Yv - Yv.mean(0)) ** 2).sum(0)
ang_err = np.abs(np.angle(np.exp(1j * (np.arctan2(pred[:, 1], pred[:, 0]) - Ya))))
print(f"ridge decoding: R^2 vx={r2_dec[0]:.3f}, vy={r2_dec[1]:.3f}; "
      f"median direction error {np.degrees(np.median(ang_err)):.1f} deg (chance 90 deg)")

feat_ang = nap.Tsd(t=tb_d, d=ang_d)
folds_t = np.unique(grp) % NFOLD
fold_bin = folds_t[np.searchsorted(np.unique(grp), grp)]
dec_bayes = np.full(len(Ya), np.nan)
for f in tqdm(range(NFOLD), desc="Bayesian decoding"):
    tr_ep = nap.IntervalSet(start=np.asarray(ep_move.start)[folds_t != f],
                            end=np.asarray(ep_move.end)[folds_t != f])
    te_ep = nap.IntervalSet(start=np.asarray(ep_move.start)[folds_t == f],
                            end=np.asarray(ep_move.end)[folds_t == f])
    tc_f = nap.compute_1d_tuning_curves(group=spikes, feature=feat_ang, nb_bins=18,
                                        ep=tr_ep, minmax=(-np.pi, np.pi))
    dec, _ = nap.decode_1d(tuning_curves=tc_f, group=spikes, ep=te_ep, bin_size=BIN_DEC)
    dec_bayes[fold_bin == f] = np.asarray(
        dec.interpolate(nap.Ts(tb_d[keep_d][fold_bin == f])).values)
mb = np.isfinite(dec_bayes)
berr = np.abs(np.angle(np.exp(1j * (dec_bayes[mb] - Ya[mb]))))
print(f"Bayesian direction decoding: median error {np.degrees(np.median(berr)):.1f} deg, "
      f"{np.mean(berr < np.pi/4):.1%} within 45 deg (chance 25%)")

# %%
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
axt = fig.add_subplot(gs[0, :])
tb_keep = tb_d[keep_d]
seg = (tb_keep > 300) & (tb_keep < 340)
ts_seg = tb_keep[seg]
ts_seg = np.where(np.r_[False, np.diff(ts_seg) > 0.12], np.nan, ts_seg)  # break between reaches
axt.plot(ts_seg, Yv[seg, 0], "k", lw=1.8, label="true")
axt.plot(ts_seg, pred[seg, 0], "crimson", lw=1.5, label="decoded $v_x$")
axt.plot(ts_seg, Yv[seg, 1] - 1600, "k", lw=1.8)
axt.plot(ts_seg, pred[seg, 1] - 1600, "steelblue", lw=1.5, label="decoded $v_y$")
axt.text(ts_seg[np.isfinite(ts_seg)][0], 700, "$v_x$", fontsize=11)
axt.text(ts_seg[np.isfinite(ts_seg)][0], -900, "$v_y$", fontsize=11)
axt.set_xlabel("time (s); gaps are inter-reach intervals, which are excluded")
axt.set_ylabel("velocity (mm/s), $v_y$ offset")
axt.legend(ncol=3, fontsize=9)
axt.set_title(f"cross-validated ridge decoding of hand velocity from {len(spikes)} units "
              f"($R^2$: $v_x$={r2_dec[0]:.2f}, $v_y$={r2_dec[1]:.2f})")
idx = np.random.default_rng(0).choice(len(Yv), 4000, replace=False)
for i, (lab, col) in enumerate([("$v_x$", "crimson"), ("$v_y$", "steelblue")]):
    axs = fig.add_subplot(gs[1, i])
    axs.scatter(Yv[idx, i], pred[idx, i], s=3, alpha=0.15, c=col)
    axs.plot([-1000, 1000], [-1000, 1000], "k--", lw=1)
    axs.set_xlim(-1000, 1000)
    axs.set_ylim(-1000, 1000)
    axs.set_xlabel(f"true {lab} (mm/s)")
    axs.set_ylabel(f"decoded {lab} (mm/s)")
    axs.set_title(f"{lab}:  $R^2$={r2_dec[i]:.2f}")
axe = fig.add_subplot(gs[1, 2])
axe.hist(np.degrees(ang_err), bins=36, color="seagreen", alpha=0.75,
         label=f"ridge (median {np.degrees(np.median(ang_err)):.0f}$\\degree$)")
axe.hist(np.degrees(berr), bins=36, color="darkorange", alpha=0.6,
         label=f"Bayesian (median {np.degrees(np.median(berr)):.0f}$\\degree$)")
axe.axvline(90, color="k", ls="--", lw=1, label="chance median (90$\\degree$)")
axe.set_xlabel("absolute direction error (deg)")
axe.set_ylabel("# time bins")
axe.legend(fontsize=8)
axe.set_title("decoded movement direction")
fig.savefig(f"{FIG}/fig08_decoding.png", dpi=140, bbox_inches="tight")

# %% [markdown]
# ## Summary
#
# * **Direction.** 83% of the 182 units are significantly cosine-tuned for reach direction during
#   movement (permutation test, p < 0.01), with a median $R^2$ of 0.08 and a maximum of 0.64 on
#   unobstructed reaches. Preferred directions tile the circle. 70% are already tuned during the
#   delay period, and the preferred direction is partially preserved into execution.
# * **Timing.** Population activity leads hand velocity by about 60 ms.
# * **Speed.** Speed is not a separate additive signal: it multiplies the directional tuning.
#   Averaged over tuned units and aligned to each unit's preferred direction, the cosine amplitude
#   grows monotonically from ~140 to ~880 mm/s while the baseline barely moves. Per unit, the
#   speed slope is positive at the preferred direction (median +0.18 Hz per 100 mm/s) and negative
#   at the opposite direction (-0.08), Wilcoxon p ~ 1e-15.
# * **Model comparison.** In cross-validated Poisson GLMs, adding a speed term to a
#   direction-only model raises the median pseudo-$R^2$ from 0.022 to 0.030 and improves 95% of
#   units. Because of the exponential link this separable model is already multiplicative in rate,
#   and an explicit direction x speed interaction adds almost nothing on top of it.
# * **Decoding.** A linear readout of the population recovers the instantaneous velocity vector
#   with $R^2$ of 0.72 ($v_x$) and 0.62 ($v_y$) and a median direction error of 19 degrees.
#
# **Caveats.** Everything here comes from a single session of one animal. Unit-to-array identity
# is not recoverable from this NWB conversion, so no M1/PMd comparison is made. The decoding
# analysis is restricted to within-reach bins with speed above 100 mm/s, which is an easier regime
# than decoding across the whole session. The GLMs contain no spike-history or condition-invariant
# terms, so the absolute pseudo-$R^2$ values understate total explainable variance; only the
# relative comparison between nested models is intended to be read.
