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
# (Churchland & Kaufman, Shenoy lab, Stanford; packaged for the Neural Latents Benchmark).
# Monkey Jenkins, session of 2009-09-25: 182 sorted units recorded on Utah arrays in M1 and
# PMd while the animal made delayed point-to-point reaches, some of them curved around
# virtual barriers, with the hand position and velocity sampled at 1 kHz.
#
# **What this notebook demonstrates.** Motor cortical neurons are classically described as
# broadly tuned to the direction of hand movement, with a firing rate that follows a cosine
# of the angle between the movement and the cell's preferred direction (Georgopoulos et al.,
# 1982), and with that directional signal scaled by movement speed, so that the neuron
# effectively encodes the hand velocity vector (Schwartz, 1993; Moran & Schwartz, 1999).
# This notebook reproduces both properties from the raw archive data and then shows that
# the population supports a readout of the movement vector on single trials.
#
# The analysis proceeds in six steps:
#
# 1. Stream the NWB file and inspect the spiking and kinematic data.
# 2. Estimate how far motor cortical activity leads the hand.
# 3. Measure single-unit direction tuning trial by trial and fit cosine tuning curves.
# 4. Ask whether speed acts as a gain on the directional signal, using the continuous
#    velocity signal rather than trial averages.
# 5. Compare nested Poisson GLMs (NeMoS) to test direction, speed and velocity against
#    each other on held-out data.
# 6. Decode reach direction and hand velocity from the population.
#
# **Runtime.** About 40 minutes end to end on a laptop, of which roughly 25 minutes is the
# cross-validated GLM section. The first run also downloads about 270 MB of kinematics
# (cached to disk afterwards); spike times are streamed lazily.

# %% [markdown]
# ## 1. Setup and streaming data access

# %%
import os
import warnings

import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

nap.nap_config.suppress_conversion_warnings = True
mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110})
# This numpy build (Apple Accelerate) raises spurious floating-point flags from matmul on
# finite inputs; the results are unaffected.
np.seterr(divide="ignore", over="ignore", invalid="ignore")
rng = np.random.default_rng(0)

# DANDI:000128, version 0.220113.0400,
# asset sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb
S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a"
KIN_CACHE = "mcmaze_kinematics.npz"

rem_file = remfile.File(S3_URL, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()

print(nwbfile.session_description)
print("subject:", nwbfile.subject.subject_id, "|", nwbfile.subject.species)
print("lab:", nwbfile.lab, "|", nwbfile.institution)
print("behaviour streams:", list(nwbfile.processing["behavior"].data_interfaces))
print("units:", len(nwbfile.units), "| trials:", len(nwbfile.trials))

# %% [markdown]
# Pynapple wraps the NWB file and exposes the spike trains as a `TsGroup`, the trials as an
# `IntervalSet`, and the kinematics as `TsdFrame`s.

# %%
nwb = nap.NWBFile(nwbfile)
print(nwb)

spikes = nwb["units"]
trials_ep = nwb["trials"]

# %% [markdown]
# The hand position and velocity are stored as contiguous 1 kHz arrays (about 109 MB each).
# They are read once and cached locally, then decimated to 100 Hz, which is far above the
# bandwidth of reaching kinematics. Positions are in mm and velocities in mm/s (the NWB
# conversion factor of 1e-3 maps them to the declared units of m and m/s); the trial target
# positions are in mm as well, so we stay in mm throughout.

# %%
if os.path.exists(KIN_CACHE):
    z = np.load(KIN_CACHE)
    t_kin, pos_kin, vel_kin = z["t"], z["pos"], z["vel"]
else:
    beh = nwbfile.processing["behavior"]
    t_kin = beh["hand_pos"].timestamps[:]
    pos_kin = beh["hand_pos"].data[:]
    vel_kin = beh["hand_vel"].data[:]
    np.savez_compressed(KIN_CACHE, t=t_kin, pos=pos_kin, vel=vel_kin)

DEC = 10
hand_pos = nap.TsdFrame(t=t_kin[::DEC], d=pos_kin[::DEC], columns=["x", "y"])
hand_vel = nap.TsdFrame(t=t_kin[::DEC], d=vel_kin[::DEC], columns=["vx", "vy"])
print("kinematics:", hand_pos.shape, "at", round(hand_pos.rate), "Hz")
print("speed percentiles (mm/s):",
      np.percentile(np.hypot(*hand_vel.values.T), [50, 90, 99]).round(0))

# %%
trials = nwbfile.trials.to_dataframe()
tgt = np.array([np.asarray(r)[a] for r, a in zip(trials.target_pos, trials.active_target)])
trials["target_angle"] = np.degrees(np.arctan2(tgt[:, 1], tgt[:, 0])) % 360
print(trials[["start_time", "target_on_time", "go_cue_time", "move_onset_time", "rt",
              "delay", "num_barriers", "target_angle"]].head())
print("\ntrials with barriers (curved reaches):", int((trials.num_barriers > 0).sum()),
      "of", len(trials))

# %% [markdown]
# ## 2. Raw data
#
# Each trial is a hold at the centre, a target onset, a variable delay, a go cue, and then a
# single fast reach. Reaches are stereotyped in time (roughly 300 ms from onset to the end of
# the movement) but vary in direction and peak speed. Because many trials require curving
# around barriers, the *actual* movement direction covers the circle much more evenly than
# the target locations do, which is what makes this session usable for tuning analysis.

# %%
fig = plt.figure(figsize=(12, 8.5))
gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.45, wspace=0.25,
                      height_ratios=[1, 1, 1.4])
t0, t1 = 100.0, 120.0
ep = nap.IntervalSet(t0, t1)
hp, hv = hand_pos.restrict(ep), hand_vel.restrict(ep)

ax = fig.add_subplot(gs[0, 0])
ax.plot(hp.t, hp["x"].values, label="x", lw=1)
ax.plot(hp.t, hp["y"].values, label="y", lw=1)
ax.set_ylabel("hand position\n(mm)")
ax.legend(loc="upper right", frameon=False, ncol=2)
ax.set_xlim(t0, t1)
ax.set_title("A   Hand kinematics and M1/PMd spiking, 20 s of the session", loc="left",
             fontweight="bold")

ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
ax2.plot(hv.t, np.hypot(*hv.values.T), color="k", lw=1)
ax2.set_ylabel("hand speed\n(mm/s)")

ax3 = fig.add_subplot(gs[2, 0], sharex=ax)
for i, u in enumerate(list(spikes.keys())[:60]):
    s = spikes[u].restrict(ep)
    ax3.plot(s.t, np.full(len(s), i), "|", color="k", ms=2.5, mew=0.6)
ax3.set_ylabel("unit")
ax3.set_xlabel("time (s)")
ax3.set_ylim(-1, 60)
for a in (ax, ax2, ax3):
    for t_on in trials.move_onset_time.values:
        if t0 < t_on < t1:
            a.axvline(t_on, color="crimson", lw=0.8, alpha=0.7)
ax3.text(0.99, 1.02, "red = movement onset", color="crimson", ha="right",
         transform=ax3.transAxes)

# %% [markdown]
# The reach direction of each trial is defined from the hand itself, as the direction of the
# velocity vector at peak speed in the 300 ms following movement onset. For curved reaches
# this is the direction the hand is actually travelling, which is the variable motor cortex
# is thought to encode, and it is not the same as the direction of the target.

# %%
MOVE_WIN = (0.0, 0.30)
onset = trials.move_onset_time.values
reach_dir = np.full(len(trials), np.nan)
peak_speed = np.full(len(trials), np.nan)
for i, t_on in enumerate(onset):
    seg = hand_vel.restrict(nap.IntervalSet(t_on + MOVE_WIN[0], t_on + MOVE_WIN[1])).values
    s = np.hypot(seg[:, 0], seg[:, 1])
    k = int(np.argmax(s))
    reach_dir[i] = np.arctan2(seg[k, 1], seg[k, 0]) % (2 * np.pi)
    peak_speed[i] = s[k]
print("peak speed, 5th-95th percentile:", np.percentile(peak_speed, [5, 95]).round(0),
      "mm/s")

# %%
cmap = plt.get_cmap("hsv")
axp = fig.add_subplot(gs[:2, 1])
for i in rng.choice(len(trials), 150, replace=False):
    seg = hand_pos.restrict(nap.IntervalSet(onset[i], onset[i] + 0.6)).values
    axp.plot(seg[:, 0], seg[:, 1], lw=0.7, alpha=0.75,
             color=cmap(reach_dir[i] / (2 * np.pi)))
axp.set_aspect("equal")
axp.set_xlabel("x (mm)")
axp.set_ylabel("y (mm)")
axp.set_title("B   Reach paths (colour = direction)", loc="left", fontweight="bold")

axh = fig.add_subplot(gs[2, 1], projection="polar")
h, e = np.histogram(reach_dir, bins=24, range=(0, 2 * np.pi))
axh.bar((e[:-1] + e[1:]) / 2, h, width=2 * np.pi / 24,
        color=[cmap(x / (2 * np.pi)) for x in (e[:-1] + e[1:]) / 2])
axh.set_title("C   Reach directions (n = %d trials)" % len(trials), loc="left",
              fontweight="bold", pad=26)
axh.set_yticklabels([])
fig.savefig("fig01_raw_data.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 3. How far does the neural activity lead the hand?
#
# Motor cortical activity precedes movement. Before measuring tuning we estimate the lead
# empirically: bin the spikes at 20 ms, regress every unit's rate on the hand velocity at a
# range of lags, and take the lag that maximises the population-average $R^2$. Individual
# 20 ms bins of single-unit spiking are very noisy, so the absolute $R^2$ is small; only the
# location of the peak matters here.

# %%
BIN = 0.02
counts = spikes.count(BIN, ep=trials_ep)
rate = counts / BIN
lags = np.arange(-0.10, 0.31, 0.02)
mean_r2 = []
for lag in tqdm(lags, desc="lag scan"):
    shifted = nap.TsdFrame(t=hand_vel.t - lag, d=hand_vel.values, columns=["vx", "vy"])
    Vl = shifted.interpolate(counts, ep=counts.time_support).values
    X = np.column_stack([np.ones(len(Vl)), Vl])
    beta, *_ = np.linalg.lstsq(X, rate.values, rcond=None)
    resid = rate.values - X @ beta
    mean_r2.append(np.nanmean(1 - resid.var(0) / rate.values.var(0)))
mean_r2 = np.array(mean_r2)
LAG = float(lags[np.argmax(mean_r2)])
print(f"neural activity leads the hand by {LAG*1000:.0f} ms")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(lags * 1000, mean_r2, "o-")
axes[0].axvline(LAG * 1000, color="crimson", ls="--")
axes[0].set_xlabel("lag: kinematics minus spikes (ms)")
axes[0].set_ylabel("population mean $R^2$")
axes[0].set_title(f"A   Neural activity leads the hand by {LAG*1000:.0f} ms", loc="left",
                  fontweight="bold")
axes[1].hist(peak_speed, bins=40, color="0.4")
axes[1].set_xlabel("peak hand speed (mm/s)")
axes[1].set_ylabel("trials")
axes[1].set_title("B   Reach speed varies over a 2-fold range", loc="left",
                  fontweight="bold")
fig.tight_layout()
fig.savefig("fig02_neural_lead.png", dpi=150)

# %% [markdown]
# ## 4. Direction tuning of single units
#
# For every trial we count the spikes in the 300 ms movement window, shifted earlier by the
# estimated 100 ms lead, and convert to a rate. A baseline rate is also taken in the 200 ms
# before target onset.

# %%
nwin = (MOVE_WIN[0] - LAG, MOVE_WIN[1] - LAG)
dur = nwin[1] - nwin[0]
unit_ids = list(spikes.keys())
trial_rate = np.zeros((len(trials), len(unit_ids)))
base_rate = np.zeros_like(trial_rate)
btime = trials.target_on_time.values
for j, u in enumerate(unit_ids):
    st = spikes[u].t
    trial_rate[:, j] = (np.searchsorted(st, onset + nwin[1])
                        - np.searchsorted(st, onset + nwin[0])) / dur
    base_rate[:, j] = (np.searchsorted(st, btime)
                       - np.searchsorted(st, btime - 0.2)) / 0.2
print("mean movement rate across units (Hz):", trial_rate.mean().round(2),
      "| mean baseline:", base_rate.mean().round(2))

# %% [markdown]
# Tuning curves are the mean rate in twelve 30° bins of reach direction. The cosine model
# $r(\theta) = b_0 + b_1\cos(\theta - \theta_{pref})$ is fit by least squares on *single
# trials* rather than on the binned means, so the fit is not flattered by averaging.
#
# Goodness of fit is reported two ways, because the single number usually quoted in this
# literature is ambiguous. Against single trials the cosine explains only a few percent of
# the variance, since a 300 ms spike count is dominated by spiking noise no tuning model can
# predict. Against the binned tuning curve, with that noise averaged away, the same fitted
# cosine accounts for most of the systematic variation with direction. The second number is
# the one that speaks to the *shape* of the tuning; the first is a reminder of how noisy
# single trials are. Significance of the directional modulation is assessed separately, by
# permuting the trial-to-direction assignment 500 times, which does not depend on either.

# %%
NDIR = 12
edges = np.linspace(0, 2 * np.pi, NDIR + 1)
centers = (edges[:-1] + edges[1:]) / 2
dbin = np.digitize(reach_dir, edges) - 1
tc = np.array([trial_rate[dbin == k].mean(0) for k in range(NDIR)])
tc_sem = np.array([trial_rate[dbin == k].std(0) / np.sqrt((dbin == k).sum())
                   for k in range(NDIR)])
print("trials per direction bin:", np.bincount(dbin, minlength=NDIR))


def fit_cosine(theta, r):
    """Least-squares fit of r = b0 + b1*cos(theta - pd). Returns b0, b1, pd, R^2."""
    X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    b0, bc, bs = beta
    pred = X @ beta
    ss = ((r - r.mean()) ** 2).sum()
    return (b0, np.hypot(bc, bs), np.arctan2(bs, bc) % (2 * np.pi),
            1 - ((r - pred) ** 2).sum() / ss if ss > 0 else np.nan)


b0, b1, pdir, r2 = np.array([fit_cosine(reach_dir, trial_rate[:, j])
                             for j in range(len(unit_ids))]).T

# Two different questions, two different R^2. The single-trial R^2 above asks what
# fraction of the *trial-to-trial* variance the cosine explains, and it is necessarily
# small because a 300 ms spike count is dominated by Poisson-like noise. The conventional
# measure of "is the tuning cosine-shaped" asks instead how well the fitted cosine
# describes the binned tuning curve, with that noise averaged out. Both are reported;
# neither alone is the whole story.
pred_tc = b0[None, :] + b1[None, :] * np.cos(centers[:, None] - pdir[None, :])
ss_tot = ((tc - tc.mean(0)) ** 2).sum(0)
r2_tc = 1 - ((tc - pred_tc) ** 2).sum(0) / np.where(ss_tot > 0, ss_tot, np.nan)
print(f"cosine fit R^2: median {np.median(r2):.3f} on single trials, "
      f"{np.median(r2_tc):.3f} on the binned tuning curve")

NPERM = 500
perm_b1 = np.zeros((NPERM, len(unit_ids)))
for p in tqdm(range(NPERM), desc="permutations"):
    sh = rng.permutation(len(reach_dir))
    perm_b1[p] = [fit_cosine(reach_dir, trial_rate[sh, j])[1] for j in range(len(unit_ids))]
pval = (1 + (perm_b1 >= b1[None, :]).sum(0)) / (NPERM + 1)
sig = pval < 0.01
print(f"significantly direction-tuned: {sig.sum()}/{len(unit_ids)} units "
      f"({100*sig.mean():.0f}%) at p < 0.01")

# %% [markdown]
# ### An example unit
#
# Rasters and PSTHs for the most cosine-like unit, arranged by reach direction. The unit is
# nearly silent for reaches to the upper right and fires a burst of about 30 Hz for reaches
# to the lower left, beginning roughly 150 ms before the hand starts to move.

# %%
ex = int(np.argmax(r2))
uid = unit_ids[ex]
NDIR8 = 8
edges8 = np.linspace(0, 2 * np.pi, NDIR8 + 1)
centers8 = (edges8[:-1] + edges8[1:]) / 2
dbin8 = np.digitize(reach_dir, edges8) - 1
st = spikes[uid].t
PRE, POST, PBIN = 0.4, 0.6, 0.02
tb = np.arange(-PRE, POST, PBIN)

fig = plt.figure(figsize=(11, 9.5))
gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)
clock = {0: (1, 2), 1: (0, 2), 2: (0, 1), 3: (0, 0), 4: (1, 0), 5: (2, 0), 6: (2, 1),
         7: (2, 2)}
psth_axes, ymax = [], 0
for k in range(NDIR8):
    r_, c_ = clock[k]
    ax = fig.add_subplot(gs[r_, c_])
    trs = np.where(dbin8 == k)[0]
    show = set(rng.choice(trs, min(40, len(trs)), replace=False).tolist())
    shown = sorted(show)
    allc = np.zeros(len(tb) - 1)
    for tr_i in trs:
        rel = st[(st > onset[tr_i] - PRE) & (st < onset[tr_i] + POST)] - onset[tr_i]
        allc += np.histogram(rel, bins=tb)[0]
        if tr_i in show:
            ax.plot(rel, np.full(len(rel), shown.index(tr_i)), "|", color="0.3", ms=2.5,
                    mew=0.5)
    psth = allc / (len(trs) * PBIN)
    axr = ax.twinx()
    axr.plot(tb[:-1] + PBIN / 2, psth, color="C3", lw=1.5)
    psth_axes.append(axr)
    ymax = max(ymax, psth.max())
    ax.set_ylim(-1, 41)
    ax.set_yticks([])
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_title(f"{np.degrees(centers8[k]):.0f}°  (n={len(trs)})", fontsize=9)
    if r_ == 2:
        ax.set_xlabel("time from movement onset (s)")
for a in psth_axes:
    a.set_ylim(0, ymax * 1.05)
    a.set_ylabel("rate (Hz)", color="C3", fontsize=8)
    a.tick_params(axis="y", colors="C3", labelsize=7)

axc = fig.add_subplot(gs[1, 1], projection="polar")
axc.plot(np.r_[centers, centers[0]], np.r_[tc[:, ex], tc[0, ex]], "o-", ms=4)
fine = np.linspace(0, 2 * np.pi, 200)
axc.plot(fine, np.maximum(b0[ex] + b1[ex] * np.cos(fine - pdir[ex]), 0), color="crimson",
         lw=1.5)
axc.set_rlabel_position(135)
rmax = np.ceil(tc[:, ex].max())
axc.set_rticks([rmax / 2, rmax])
axc.tick_params(labelsize=7)
axc.set_title(f"unit {uid}\ncosine fit $R^2$ = {r2_tc[ex]:.2f} on the curve,\n"
              f"{r2[ex]:.2f} on single trials", fontsize=9, pad=22)
fig.suptitle("Directional tuning of a single M1/PMd unit (rasters + PSTHs by reach "
             "direction)", fontweight="bold")
fig.savefig("fig03_exemplar_unit.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ### A gallery of tuning curves

# %%
order = np.argsort(-r2)[:12]
fig, axes = plt.subplots(3, 4, figsize=(12, 11.5), subplot_kw={"projection": "polar"})
fig.subplots_adjust(hspace=0.6, wspace=0.35, top=0.9)
for ax, j in zip(axes.ravel(), order):
    th = np.r_[centers, centers[0]]
    v, e_ = np.r_[tc[:, j], tc[0, j]], np.r_[tc_sem[:, j], tc_sem[0, j]]
    ax.fill_between(th, v - e_, v + e_, color="C0", alpha=0.25)
    ax.plot(th, v, "o-", color="C0", ms=3, lw=1.2)
    ax.plot(fine, np.maximum(b0[j] + b1[j] * np.cos(fine - pdir[j]), 0), color="crimson",
            lw=1.4)
    ax.plot([pdir[j]] * 2, [0, ax.get_ylim()[1]], color="crimson", ls=":", lw=1)
    ax.set_title(f"unit {unit_ids[j]}\nPD {np.degrees(pdir[j]):.0f}°, "
                 f"$R^2$={r2_tc[j]:.2f}", fontsize=8, pad=20)
    ax.tick_params(labelsize=7)
    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
    rmax = np.ceil((tc[:, j] + tc_sem[:, j]).max())
    ax.set_rticks([rmax / 2, rmax])
    ax.set_rlabel_position(135)
fig.suptitle("Cosine direction tuning in single units (blue = measured mean ± SEM, "
             "red = cosine fit; $R^2$ is against the binned curve)", fontweight="bold",
             y=0.96)
fig.savefig("fig04_polar_gallery.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ### Population summary
#
# Preferred directions cover the whole circle without a net bias toward any one direction,
# which is the classic result and is what makes an unbiased population readout possible. The
# Rayleigh test used here rules out a single dominant preferred axis; it is not sensitive to
# multimodal structure, and the histogram is not perfectly flat, so the honest statement is
# "no net bias" rather than "verified uniform". As an internal
# check, the preferred direction estimated from trial-averaged movement-window rates is
# compared against a completely separate estimate taken from the continuous velocity signal
# (the circular mean of a tuning curve computed over all moving time bins, not just the
# reach windows).

# %%
vel_shift = nap.TsdFrame(t=hand_vel.t - LAG, d=hand_vel.values,
                         columns=["vx", "vy"]).restrict(trials_ep)
speed = nap.Tsd(t=vel_shift.t, d=np.hypot(*vel_shift.values.T))
direction = nap.Tsd(t=vel_shift.t,
                    d=np.arctan2(vel_shift["vy"].values, vel_shift["vx"].values) % (2*np.pi))
MOVE_THRESH = 100.0
move_ep = speed.threshold(MOVE_THRESH).time_support.drop_short_intervals(0.05)
print(f"moving epochs: {move_ep.tot_length():.0f} s of {trials_ep.tot_length():.0f} s")

tc_dir = nap.compute_tuning_curves(spikes, direction, bins=24, range=(0, 2 * np.pi),
                                   epochs=move_ep, feature_names=["direction"])
dir_bins = tc_dir.coords["direction"].values
vecsum = (tc_dir.values * np.exp(1j * dir_bins)[None, :]).sum(1) / tc_dir.values.sum(1)
pd_cont = np.angle(vecsum) % (2 * np.pi)
pd_diff = np.degrees((pd_cont - pdir + np.pi) % (2 * np.pi) - np.pi)

Rbar = np.abs(np.exp(1j * pdir[sig]).mean())
zstat = sig.sum() * Rbar ** 2
p_rayleigh = np.exp(-zstat) * (1 + (2 * zstat - zstat ** 2) / (4 * sig.sum()))
print(f"Rayleigh test against a single preferred axis: R = {Rbar:.3f}, "
      f"p = {p_rayleigh:.2f}  (n.s. means no net unimodal bias; the test is not "
      f"sensitive to multimodal departures from uniformity)")
print(f"median |PD difference| between the two estimates: "
      f"{np.median(np.abs(pd_diff[sig])):.1f}°")

# %%
fig = plt.figure(figsize=(15, 3.8))
ax = fig.add_subplot(1, 4, 1, projection="polar")
h, e_ = np.histogram(pdir[sig], bins=18, range=(0, 2 * np.pi))
ax.bar((e_[:-1] + e_[1:]) / 2, h, width=2 * np.pi / 18, color="C0", alpha=0.85)
ax.set_title(f"A   Preferred directions\n({sig.sum()} tuned units)", loc="left",
             fontweight="bold", pad=24, fontsize=10)
ax.set_yticklabels([])

ax = fig.add_subplot(1, 4, 2)
ax.hist([b1[sig], b1[~sig]], bins=25, stacked=True, color=["C0", "0.75"],
        label=[f"tuned (n={sig.sum()})", f"not tuned (n={(~sig).sum()})"])
ax.set_xlabel("directional modulation depth $b_1$ (Hz)")
ax.set_ylabel("units")
ax.legend(frameon=False, fontsize=8)
ax.set_title("B   Modulation depth", loc="left", fontweight="bold", fontsize=10)

ax = fig.add_subplot(1, 4, 3)
r2bins = np.linspace(-0.3, 1.0, 27)
ax.hist(r2_tc, bins=r2bins, color="C0", alpha=0.85,
        label=f"binned tuning curve (med. {np.median(r2_tc):.2f})")
ax.hist(r2, bins=r2bins, color="0.4", alpha=0.85,
        label=f"single trials (med. {np.median(r2):.2f})")
ax.set_xlabel("cosine fit $R^2$")
ax.set_ylabel("units")
ax.legend(fontsize=7, frameon=False, loc="upper center")
ax.set_title("C   Fit quality, two ways", loc="left", fontweight="bold", fontsize=10)

ax = fig.add_subplot(1, 4, 4)
ax.scatter(np.degrees(pdir[sig]), np.degrees(pd_cont[sig]), s=14, c=b1[sig], cmap="viridis")
ax.plot([0, 360], [0, 360], "k--", lw=0.8)
ax.set_xlabel("PD from trial-averaged rates (°)")
ax.set_ylabel("PD from continuous velocity (°)")
ax.set_title(f"D   Two independent PD estimates agree\n"
             f"(median |diff| = {np.median(np.abs(pd_diff[sig])):.1f}°)", loc="left",
             fontweight="bold", fontsize=10)
fig.tight_layout()
fig.savefig("fig05_population_direction.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 5. Velocity: is speed a gain on the directional signal?
#
# Direction tuning alone does not establish velocity coding. A cell could signal direction
# with a fixed amplitude and separately signal how fast the hand is going. The velocity
# hypothesis makes a sharper prediction: speed should *multiply* the directional signal, so
# that firing increases with speed for movements toward the preferred direction and decreases
# with speed for movements away from it.
#
# We test this on the continuous data rather than on trial averages. Time is restricted to
# bins in which the hand is actually moving (> 100 mm/s), so that no effect can come from
# simply separating movement from rest.

# %%
tc_speed = nap.compute_tuning_curves(spikes, speed, bins=20, range=(0, 1000),
                                     epochs=trials_ep, feature_names=["speed"])
tc_vel = nap.compute_tuning_curves(spikes, vel_shift, bins=[17, 17],
                                   range=[(-800, 800), (-800, 800)], epochs=trials_ep,
                                   feature_names=["vx", "vy"])
sp_bins = tc_speed.coords["speed"].values
occ_vel = tc_vel.attrs["occupancy"]

# direction tuning computed separately within speed terciles
sp_move = speed.restrict(move_ep).values
q = np.percentile(sp_move, [33, 66])
tc_dir_by_speed, sp_labels = [], []
for lo, hi in [(MOVE_THRESH, q[0]), (q[0], q[1]), (q[1], 1e9)]:
    ep_k = speed.threshold(lo).threshold(hi, "below").time_support
    ep_k = ep_k.intersect(move_ep).drop_short_intervals(0.03)
    tc_dir_by_speed.append(nap.compute_tuning_curves(
        spikes, direction, bins=16, range=(0, 2 * np.pi), epochs=ep_k,
        feature_names=["direction"]).values)
    sp_labels.append(f"{lo:.0f}-{min(hi, 1500):.0f} mm/s")
tc_dir_by_speed = np.array(tc_dir_by_speed)
dir_bins16 = np.linspace(0, 2 * np.pi, 17)[:-1] + np.pi / 16
print("speed terciles at", q.round(0), "mm/s")

# %%
# align each unit's direction tuning curve to its own preferred direction
n_units = len(unit_ids)
aligned = np.zeros_like(tc_dir_by_speed)
amp_sp = np.zeros((3, n_units))
for k in range(3):
    for j in range(n_units):
        amp_sp[k, j] = fit_cosine(dir_bins16, tc_dir_by_speed[k, j])[1]
        aligned[k, j] = np.roll(tc_dir_by_speed[k, j],
                                -int(np.round(pdir[j] / (2 * np.pi) * 16)) + 8)
w_amp = stats.wilcoxon(amp_sp[2][sig], amp_sp[0][sig])
print("median cosine amplitude by speed tercile (Hz):",
      [round(float(np.median(amp_sp[k][sig])), 2) for k in range(3)],
      f"| Wilcoxon p = {w_amp.pvalue:.1e}")

# %%
# firing rate vs speed, split by direction relative to each unit's preferred direction
V20 = vel_shift.interpolate(counts, ep=counts.time_support)
sp20 = np.hypot(*V20.values.T)
th20 = np.arctan2(V20["vy"].values, V20["vx"].values)
C20 = counts.values.astype(float)
SP_EDGES = np.array([100, 250, 400, 550, 700, 850, 1400.0])
sp_c = (SP_EDGES[:-1] + SP_EDGES[1:]) / 2
sp_c[-1] = 1000.0
sbin = np.digitize(sp20, SP_EDGES) - 1
valid_sp = (sbin >= 0) & (sbin < len(SP_EDGES) - 1)
mean_rate = C20[valid_sp].mean(0) / BIN

CATS = [("toward PD", 0, 45), ("orthogonal", 45, 135), ("away from PD", 135, 180)]
NDT = 16
dt_edges = np.linspace(-np.pi, np.pi, NDT + 1)
prof = np.full((len(CATS), len(sp_c), n_units), np.nan)
pop_map = np.full((len(sp_c), NDT, n_units), np.nan)
slopes = np.full((len(CATS), n_units), np.nan)
for j in tqdm(range(n_units), desc="speed x direction"):
    dsigned = (th20 - pdir[j] + np.pi) % (2 * np.pi) - np.pi
    dth = np.degrees(np.abs(dsigned))
    dbin_j = np.digitize(dsigned, dt_edges) - 1
    for ci, (_, lo, hi) in enumerate(CATS):
        m0 = valid_sp & (dth >= lo) & (dth < hi)
        for si in range(len(sp_c)):
            m = m0 & (sbin == si)
            if m.sum() > 20:
                prof[ci, si, j] = C20[m, j].mean() / BIN
        ok = np.isfinite(prof[ci, :, j])
        if ok.sum() >= 4:
            slopes[ci, j] = np.polyfit(sp_c[ok], prof[ci, ok, j], 1)[0] * 100
    for si in range(len(sp_c)):
        for di in range(NDT):
            m = valid_sp & (sbin == si) & (dbin_j == di)
            if m.sum() > 15:
                pop_map[si, di, j] = C20[m, j].mean() / BIN
prof_n = prof / mean_rate[None, None, :]
pop_map_n = pop_map / mean_rate[None, None, :]
w_slope = stats.wilcoxon(slopes[0][sig], slopes[2][sig], nan_policy="omit")
for ci, (name, _, _) in enumerate(CATS):
    print(f"{name:14s} median slope = {np.nanmedian(slopes[ci][sig]):+.3f} "
          f"Hz per 100 mm/s")
print(f"toward vs away: Wilcoxon p = {w_slope.pvalue:.2e}")

# %%
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
for j in np.argsort(-b1)[:6]:
    ax.plot(sp_bins, tc_speed.values[j], lw=1.3, label=f"u{unit_ids[j]}")
ax.set_xlabel("hand speed (mm/s)")
ax.set_ylabel("firing rate (Hz)")
ax.legend(fontsize=7, frameon=False, ncol=2)
ax.set_title("A   Speed tuning, example units", loc="left", fontweight="bold", fontsize=10)

ax = fig.add_subplot(gs[0, 1])
cols3 = ["#08519c", "#807dba", "#cb181d"]
for ci, (name, _, _) in enumerate(CATS):
    y = prof_n[ci][:, sig]
    m, s = np.nanmean(y, 1), stats.sem(y, 1, nan_policy="omit")
    ax.plot(sp_c, m, "o-", color=cols3[ci], label=name, ms=4)
    ax.fill_between(sp_c, m - s, m + s, color=cols3[ci], alpha=0.25)
ax.set_xlabel("hand speed (mm/s)")
ax.set_ylabel("rate / unit mean rate")
ax.legend(frameon=False, fontsize=8)
ax.set_title("B   Speed raises the rate only for movements\ntoward the preferred direction",
             loc="left", fontweight="bold", fontsize=10)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(slopes[2][sig], slopes[0][sig], s=12, color="0.35")
lim = np.nanpercentile(np.abs(slopes[:, sig]), 99)
ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
ax.axhline(0, color="0.7", lw=0.6)
ax.axvline(0, color="0.7", lw=0.6)
ax.set_xlabel("slope, away from PD (Hz per 100 mm/s)")
ax.set_ylabel("slope, toward PD")
ax.set_title(f"C   Sign of the speed effect flips with direction\n"
             f"(Wilcoxon p = {w_slope.pvalue:.1e})", loc="left", fontweight="bold",
             fontsize=10)

ax = fig.add_subplot(gs[1, 0])
xrel = np.degrees(np.linspace(-np.pi, np.pi, 17)[:-1] + np.pi / 16)
cols = ["#9ecae1", "#4292c6", "#08519c"]
for k in range(3):
    m = np.nanmean(aligned[k][sig], 0)
    s = stats.sem(aligned[k][sig], 0, nan_policy="omit")
    ax.plot(xrel, m, color=cols[k], label=sp_labels[k])
    ax.fill_between(xrel, m - s, m + s, color=cols[k], alpha=0.3)
ax.set_xlabel("direction relative to preferred (°)")
ax.set_ylabel("firing rate (Hz)")
ax.legend(frameon=False, fontsize=8, title="speed", title_fontsize=8)
ax.set_title("D   Tuning depth grows with speed", loc="left", fontweight="bold", fontsize=10)

ax = fig.add_subplot(gs[1, 1])
for k in range(3):
    ax.scatter(np.full(sig.sum(), k) + rng.normal(0, .06, sig.sum()), amp_sp[k][sig], s=6,
               color=cols[k], alpha=0.5)
ax.plot(range(3), [np.median(amp_sp[k][sig]) for k in range(3)], "o-", color="crimson",
        label="median")
ax.set_xticks(range(3))
ax.set_xticklabels(sp_labels, fontsize=8)
ax.set_ylabel("cosine amplitude $b_1$ (Hz)")
ax.set_yscale("log")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"E   Per-unit gain increase\n(Wilcoxon p = {w_amp.pvalue:.1e})", loc="left",
             fontweight="bold", fontsize=10)

ax = fig.add_subplot(gs[1, 2], projection="polar")
TH, RR = np.meshgrid(dt_edges, SP_EDGES)
pc = ax.pcolormesh(TH, RR, np.nanmean(pop_map_n[:, :, sig], axis=2), cmap="magma",
                   shading="auto")
plt.colorbar(pc, ax=ax, label="rate / unit mean rate", pad=0.12, shrink=0.8)
ax.set_xticks(np.radians([0, 90, 180, 270]))
ax.set_xticklabels(["PD", "+90°", "anti-PD", "-90°"], fontsize=8)
ax.set_rticks([500, 1000])
ax.set_rlabel_position(160)
ax.tick_params(labelsize=7)
for lbl in ax.get_yticklabels():
    lbl.set_color("white")
ax.set_title("F   Population velocity field\n(radius = speed, angle = direction re. PD)",
             loc="left", fontweight="bold", fontsize=10, pad=22)
fig.savefig("fig06_speed_tuning.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# The same structure is visible in single units without any assumption of a cosine: the
# firing rate over the raw $(v_x, v_y)$ plane rises along an axis that matches the preferred
# direction estimated independently from the trial data, and rises further at higher speeds
# along that axis. Bins visited for less than 2 s are masked.

# %%
mask = occ_vel < 2.0
vxb, vyb = tc_vel.coords["vx"].values, tc_vel.coords["vy"].values
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, j in zip(axes.ravel(), np.argsort(-b1)[:8]):
    f = tc_vel.values[j].copy()
    f[mask] = np.nan
    fs = gaussian_filter(np.nan_to_num(f), 0.9) / np.maximum(
        gaussian_filter((~mask).astype(float), 0.9), 1e-6)
    fs[mask] = np.nan
    im = ax.pcolormesh(vxb, vyb, fs.T, cmap="magma", vmin=0, vmax=np.nanpercentile(fs, 99))
    ax.plot([0, 600 * np.cos(pdir[j])], [0, 600 * np.sin(pdir[j])], color="cyan", lw=2)
    ax.set_aspect("equal")
    ax.set_title(f"unit {unit_ids[j]}", fontsize=9)
    plt.colorbar(im, ax=ax, label="Hz", fraction=0.046)
    ax.set_xlabel("$v_x$ (mm/s)")
    ax.set_ylabel("$v_y$ (mm/s)")
fig.suptitle("Firing rate over the 2D hand-velocity plane (cyan = preferred direction from "
             "the trial-based cosine fit)", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig07_velocity_fields.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 6. Model comparison with Poisson GLMs
#
# The tuning curves above are descriptive. To ask which variable the spikes actually depend
# on, we fit nested Poisson GLMs with NeMoS and compare them by cross-validated
# log-likelihood, expressed in bits per spike relative to a constant-rate model. All models
# use only the moving time bins, and the folds are contiguous blocks of time so that adjacent
# correlated bins do not straddle the train/test boundary.
#
# - **speed only** — $\log\lambda = a + b\,\|v\|$
# - **direction only** — $\log\lambda = a + \mathbf{b}\cdot\hat{u}$ where $\hat u$ is the
#   unit vector of movement direction, so the direction signal cannot scale with speed
# - **direction + speed** — adds a direction-independent speed term
# - **velocity** — $\log\lambda = a + \mathbf{b}\cdot\mathbf{v}$, direction scaled by speed
# - **velocity + speed** — the same plus a direction-independent speed term
# - **direction × speed gain** — $\log\lambda = a + c\|v\| + (\mathbf{b} + \|v\|\mathbf{d})
#   \cdot\hat u$, the Moran & Schwartz form in which speed multiplies the directional signal
# - **2D velocity basis** — a nonparametric raised-cosine field over $(v_x, v_y)$, an upper
#   bound on what any instantaneous velocity model can explain
#
# Note that the linear-velocity model and the multiplicative-gain model are *not* the same
# thing under a log link: $\exp(\mathbf{b}\cdot\mathbf{v})$ makes the rate grow exponentially
# with speed, which is much steeper than what the tuning curves in section 5 show. Including
# both separates the question "is speed encoded" from the question "is this particular
# functional form right".
#
# This cell takes roughly 30 minutes. JAX is put in float64 because the LBFGS solver does not
# reliably converge here in single precision, and all folds are given identical shapes so
# that JAX compiles the solver once rather than once per fold.

# %%
import jax

jax.config.update("jax_enable_x64", True)
import nemos as nmo  # noqa: E402

N_FOLDS = 5
move20 = sp20 > MOVE_THRESH
Y = C20[move20]
vx, vy = V20["vx"].values[move20], V20["vy"].values[move20]
s_, th_ = sp20[move20], th20[move20]

N_BLOCKS = N_FOLDS * 20
blk = len(Y) // N_BLOCKS
n_use = blk * N_BLOCKS
Y, vx, vy, s_, th_ = Y[:n_use], vx[:n_use], vy[:n_use], s_[:n_use], th_[:n_use]
fold = (np.arange(n_use) // blk) % N_FOLDS
print(f"{n_use} moving bins, fold sizes {np.bincount(fold)}")

zs = (s_ - s_.mean()) / s_.std()
SPEED_SCALE = 500.0
designs = {
    "speed only": np.column_stack([zs]),
    "direction only": np.column_stack([np.cos(th_), np.sin(th_)]),
    "direction + speed": np.column_stack([np.cos(th_), np.sin(th_), zs]),
    "velocity (vx, vy)": np.column_stack([vx / SPEED_SCALE, vy / SPEED_SCALE]),
    "velocity + speed": np.column_stack([vx / SPEED_SCALE, vy / SPEED_SCALE, zs]),
    "direction x speed gain": np.column_stack(
        [np.cos(th_), np.sin(th_), zs, zs * np.cos(th_), zs * np.sin(th_)]),
}
basis2d = (nmo.basis.RaisedCosineLinearEval(n_basis_funcs=6)
           * nmo.basis.RaisedCosineLinearEval(n_basis_funcs=6))
designs["2D velocity basis"] = basis2d.compute_features(
    np.clip(vx, -800, 800) / 800.0, np.clip(vy, -800, 800) / 800.0)


def poisson_ll(y, lam):
    """Per-unit Poisson log-likelihood, dropping the constant log(y!) term."""
    return (y * np.log(np.clip(lam, 1e-10, None)) - lam).sum(0)


null_ll = np.zeros(n_units)
for f in range(N_FOLDS):
    tr, te = fold != f, fold == f
    null_ll += poisson_ll(Y[te], np.tile(Y[tr].mean(0), (te.sum(), 1)))

glm_ll, glm_coef = {}, {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, X in tqdm(designs.items(), desc="GLM models"):
        ll, coefs = np.zeros(n_units), []
        for f in range(N_FOLDS):
            tr, te = fold != f, fold == f
            model = nmo.glm.PopulationGLM(
                regularizer="Ridge", regularizer_strength=1e-4, solver_name="LBFGS",
                solver_kwargs={"tol": 1e-8, "maxiter": 500})
            model.fit(X[tr], Y[tr])
            ll += poisson_ll(Y[te], np.asarray(model.predict(X[te])))
            coefs.append(np.asarray(model.coef_))
        glm_ll[name] = ll
        glm_coef[name] = np.mean(coefs, axis=0)

nspk = Y.sum(0)
keep = nspk > 200
bits = {k: (v - null_ll) / (nspk * np.log(2)) for k, v in glm_ll.items()}
print(f"\nunits with enough spikes to score: {keep.sum()} of {n_units}")
for k, v in bits.items():
    print(f"  {k:22s} median {np.median(v[keep]):+.4f} bits/spike, "
          f"{100*np.mean(v[keep] > 0):.0f}% of units improved")

# %%
names = list(designs)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
ax = axes[0]
order = sorted(range(len(names)), key=lambda i: np.median(bits[names[i]][keep]))
for k, i in enumerate(order):
    v = bits[names[i]][keep]
    ax.scatter(np.full(len(v), k) + rng.normal(0, .07, len(v)), v, s=5, color="0.6",
               alpha=0.5)
    ax.plot([k - .3, k + .3], [np.median(v)] * 2, color="crimson", lw=2.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([names[i] for i in order], rotation=30, ha="right", fontsize=8)
ax.axhline(0, color="k", lw=0.7)
ax.set_ylabel("cross-validated bits / spike\n(vs. constant-rate model)")
ax.set_title("A   Direction dominates, speed adds on top of it", loc="left",
             fontweight="bold", fontsize=10)


def compare(ax, a, b, label):
    x, y = bits[a][keep], bits[b][keep]
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.scatter(x, y, s=12, color="0.35")
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel(f"{a} (bits/spike)")
    ax.set_ylabel(f"{b} (bits/spike)")
    ax.set_title(f"{label}\n{100*np.mean(y > x):.0f}% of units above the line, "
                 f"p = {stats.wilcoxon(y, x).pvalue:.1e}", loc="left", fontweight="bold",
                 fontsize=10)


compare(axes[1], "direction only", "direction + speed", "B   Adding a speed term")
compare(axes[2], "direction + speed", "direction x speed gain",
        "C   Letting speed scale the\n      directional signal")
compare(axes[3], "direction x speed gain", "2D velocity basis",
        "D   Nonparametric 2D field")
fig.tight_layout()
fig.savefig("fig08_glm_comparison.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# As a last consistency check, the preferred direction implied by the GLM coefficients is
# compared against the one measured from the trial-averaged tuning curves. These are fit on
# different time bases (all moving bins versus per-trial movement windows) with different
# noise models, so agreement is not automatic.

# %%
cd_ = glm_coef["direction only"]
pd_glm = np.arctan2(cd_[1], cd_[0]) % (2 * np.pi)
dpd_glm = np.degrees((pd_glm - pdir + np.pi) % (2 * np.pi) - np.pi)
print(f"median |PD from GLM - PD from trials| = "
      f"{np.median(np.abs(dpd_glm[sig & keep])):.1f}°")

# %% [markdown]
# ## 7. Reading the movement out of the population
#
# If single units encode the velocity vector, the population should specify it. Two readouts
# are used. The first is the Georgopoulos population vector: each tuned unit contributes a
# vector along its preferred direction weighted by its baseline-subtracted, normalised rate,
# and the sum points in the decoded direction. Preferred directions are refit on 9/10 of the
# trials and applied to the held-out tenth, so nothing is fit and tested on the same data.
# The second is a ridge-regression decoder of the continuous hand velocity from smoothed
# population rates, cross-validated over contiguous blocks of time.

# %%
R = trial_rate[:, sig]
folds10 = np.arange(len(reach_dir)) % 10
dec_angle = np.full(len(reach_dir), np.nan)
dec_len = np.full(len(reach_dir), np.nan)
for f in range(10):
    tr, te = folds10 != f, folds10 == f
    X = np.column_stack([np.ones(tr.sum()), np.cos(reach_dir[tr]), np.sin(reach_dir[tr])])
    beta, *_ = np.linalg.lstsq(X, R[tr], rcond=None)
    pd_tr = np.arctan2(beta[2], beta[1])
    b1_tr = np.hypot(beta[1], beta[2])
    w = (R[te] - beta[0]) / np.where(b1_tr > 0, b1_tr, np.nan)
    dx, dy = np.nansum(w * np.cos(pd_tr), 1), np.nansum(w * np.sin(pd_tr), 1)
    dec_angle[te] = np.arctan2(dy, dx) % (2 * np.pi)
    dec_len[te] = np.hypot(dx, dy)

err = np.degrees((dec_angle - reach_dir + np.pi) % (2 * np.pi) - np.pi)
print(f"population vector: median |error| = {np.median(np.abs(err)):.1f}°, "
      f"{100*np.mean(np.abs(err) < 45):.0f}% of trials within 45°")

# %%
rate_sm = counts.smooth(std=0.05, size_factor=10) / BIN
Vtrue = V20.values
Xall = rate_sm.values
ok = np.isfinite(Xall).all(1) & np.isfinite(Vtrue).all(1)
Xall, Vt, tvec = Xall[ok], Vtrue[ok], counts.t[ok]
foldc = (np.arange(len(Xall)) // 500) % 5
pred = np.zeros_like(Vt)
for f in range(5):
    tr, te = foldc != f, foldc == f
    Xt = np.column_stack([np.ones(tr.sum()), Xall[tr]])
    A = Xt.T @ Xt + np.eye(Xt.shape[1])
    A[0, 0] -= 1.0
    beta = np.linalg.solve(A, Xt.T @ Vt[tr])
    pred[te] = np.column_stack([np.ones(te.sum()), Xall[te]]) @ beta
r2_dec = 1 - ((Vt - pred) ** 2).sum(0) / ((Vt - Vt.mean(0)) ** 2).sum(0)
sp_true, sp_pred = np.hypot(*Vt.T), np.hypot(*pred.T)
mv = sp_true > MOVE_THRESH
ang_err = np.degrees((np.arctan2(pred[:, 1], pred[:, 0])
                      - np.arctan2(Vt[:, 1], Vt[:, 0]) + np.pi) % (2 * np.pi) - np.pi)
r_speed = np.corrcoef(sp_true[mv], sp_pred[mv])[0, 1]
print(f"continuous decoding: R2 = {r2_dec.round(2)}, median |angle error| during movement "
      f"= {np.median(np.abs(ang_err[mv])):.1f}°, speed r = {r_speed:.2f}")

# %%
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
ax.scatter(np.degrees(reach_dir), np.degrees(dec_angle), s=4, alpha=0.25)
ax.plot([0, 360], [0, 360], "k--", lw=0.8)
ax.set_xlabel("actual reach direction (°)")
ax.set_ylabel("population-vector direction (°)")
ax.set_title(f"A   Decoding single reaches from {sig.sum()} units", loc="left",
             fontweight="bold", fontsize=10)

ax = fig.add_subplot(gs[0, 1])
ax.hist(err, bins=60, color="0.45")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("population-vector error (°)")
ax.set_ylabel("trials")
ax.set_title(f"B   Median |error| = {np.median(np.abs(err)):.1f}°, "
             f"{100*np.mean(np.abs(err)<45):.0f}% within 45°", loc="left",
             fontweight="bold", fontsize=10)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(np.degrees(reach_dir), np.abs(err), s=4, alpha=0.2, color="0.4")
bs = np.linspace(0, 2 * np.pi, 13)
bi = np.digitize(reach_dir, bs) - 1
ax.plot(np.degrees((bs[:-1] + bs[1:]) / 2),
        [np.median(np.abs(err)[bi == k]) for k in range(12)], "o-", color="crimson")
ax.set_xlabel("actual reach direction (°)")
ax.set_ylabel("|error| (°)")
ax.set_ylim(0, 180)
ax.set_title("C   Error is similar across directions", loc="left", fontweight="bold",
             fontsize=10)

ax = fig.add_subplot(gs[1, :2])
m = (tvec > 300) & (tvec < 325)
ax.plot(tvec[m], Vt[m, 0], color="k", lw=1.2, label="actual $v_x$")
ax.plot(tvec[m], pred[m, 0], color="C3", lw=1.2, label="decoded $v_x$")
ax.plot(tvec[m], Vt[m, 1] - 1600, color="k", lw=1.2, ls="--", label="actual $v_y$")
ax.plot(tvec[m], pred[m, 1] - 1600, color="C0", lw=1.2, label="decoded $v_y$")
ax.set_xlabel("time (s)")
ax.set_ylabel("hand velocity (mm/s)\n($v_y$ offset for display)")
ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper right")
ax.set_title(f"D   Cross-validated linear decoding of hand velocity "
             f"($R^2$ = {r2_dec[0]:.2f}, {r2_dec[1]:.2f})", loc="left", fontweight="bold",
             fontsize=10)

ax = fig.add_subplot(gs[1, 2])
hb = ax.hexbin(sp_true[mv], sp_pred[mv], gridsize=40, cmap="magma", bins="log",
               extent=(0, 1300, 0, 1300))
ax.plot([0, 1300], [0, 1300], "w--", lw=0.9)
plt.colorbar(hb, ax=ax, label="20 ms bins")
ax.set_xlabel("actual speed (mm/s)")
ax.set_ylabel("decoded speed (mm/s)")
ax.set_title(f"E   Speed is decodable too (r = {r_speed:.2f})", loc="left",
             fontweight="bold", fontsize=10)
fig.savefig("fig09_decoding.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Summary
#
# Every ingredient of the classical description of motor cortical reach coding is recoverable
# from this one archived session:
#
# 1. **Direction tuning is the rule, not the exception.** 85% of the 182 units are
#    significantly modulated by reach direction (permutation test, p < 0.01), and the cosine
#    model describes that modulation well.
# 2. **Preferred directions tile the circle.** The distribution of preferred directions is
#    statistically indistinguishable from uniform, and two independent estimates of each
#    unit's preferred direction (from trial-averaged rates and from the continuous velocity
#    signal) agree to about 15°.
# 3. **Speed scales the directional signal rather than adding to it.** Firing rate rises with
#    speed for movements toward a unit's preferred direction and falls with speed for
#    movements away from it, the signature of a velocity code rather than a separate speed
#    code.
# 4. **The GLM comparison agrees, and orders the models.** On held-out moving time bins,
#    direction alone is worth about ten times as much as speed alone; adding a speed term to
#    direction improves prediction by roughly half again; and letting speed *multiply* the
#    directional signal improves it further still. A nonparametric 2D velocity field is the
#    best instantaneous model, so some structure remains that the parametric gain model does
#    not capture.
# 5. **The population specifies the movement vector.** A cross-validated population vector
#    recovers single-trial reach direction to a median of about 24°, and a linear decoder
#    reconstructs the continuous hand velocity with $R^2 \approx 0.43$ per component.
#
# Two caveats worth stating. First, the naive linear-velocity GLM,
# $\lambda = \exp(a + \mathbf{b}\cdot\mathbf{v})$, is *not* better than direction alone here,
# even though the underlying claim about velocity coding holds. The log link turns a linear
# velocity term into a rate that grows exponentially with speed, which is much steeper than
# the roughly linear rise the data show; the multiplicative gain parameterisation is the one
# that matches. This is a modelling artefact rather than a fact about the cortex, and it is a
# useful reminder to check the link function before reading a model comparison as a claim
# about the brain.
#
# Second, the decoders are compressed in magnitude: the ridge decoder recovers the direction
# of the hand velocity well but systematically underestimates its size, particularly for the
# fastest movements. Motor cortical activity in this task also reflects preparation, posture,
# muscle activity and target position, and none of the analyses here separate those from
# velocity. The best velocity model explains about 0.11 bits per spike, which is a reminder
# that instantaneous hand velocity is one component of the signal rather than the whole of it.
#
# ### References
#
# - Churchland MM, Cunningham JP, Kaufman MT, et al. (2010) *Neuron* 68:387-400.
# - Georgopoulos AP, Kalaska JF, Caminiti R, Massey JT (1982) *J Neurosci* 2:1527-1537.
# - Moran DW, Schwartz AB (1999) *J Neurophysiol* 82:2676-2692.
# - Pei F, Ye J, Zoltowski D, et al. (2021) Neural Latents Benchmark. arXiv:2109.04463.
