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
# This notebook demonstrates, using real data streamed from the DANDI Archive, that neurons in
# macaque motor cortex are tuned to the **direction** of a reach and that this directional signal
# is **scaled by hand speed**, which is the defining signature of tuning to *velocity* rather than
# to direction alone.
#
# **Datasets**
#
# | | primary | replication |
# |---|---|---|
# | Dandiset | [DANDI:000128](https://dandiarchive.org/dandiset/000128) `MC_Maze` | [DANDI:000129](https://dandiarchive.org/dandiset/000129) `MC_RTT` |
# | Subject | macaque Jenkins | macaque Indy |
# | Task | delayed centre-out reaching, with and without virtual barriers | self-paced random-target reaching |
# | Units | 182 sorted units | 130 sorted units |
# | Behaviour | hand position / velocity at 1 kHz | finger position / velocity at 1 kHz |
#
# Both are from the Neural Latents Benchmark release of previously published recordings
# (Churchland, Kaufman and Shenoy for `MC_Maze`; Makin, O'Doherty and Sabes for `MC_RTT`).
#
# **What is shown**
#
# 1. Single units fire more for reaches in some directions than others, and a cosine describes
#    that dependence well (median $R^2 = 0.72$ across direction-binned means).
# 2. The directional signal leads the hand: it peaks about 125 ms before peak hand speed, and
#    population decoding of hand velocity peaks at a lag of +100 ms.
# 3. Firing rate is not a function of direction alone. At a fixed direction relative to a unit's
#    preferred direction, rate grows with speed for movements toward the preferred direction and
#    falls with speed for movements away from it. A Poisson GLM given the full velocity vector
#    roughly doubles the cross-validated pseudo-$R^2$ of a direction-only model.
# 4. Hand velocity can be decoded from the population ($R^2 = 0.55$ during movement), and the
#    classic Georgopoulos population vector recovers reach direction to a median error of 26°.
# 5. All of the above replicates in an independent dataset, subject and task.
#
# **A note on the data.** In `MC_Maze` the electrode table describes two Utah arrays (M1 and PMd),
# but every unit's electrode reference resolves into the PMd block of that table, with only 87
# distinct electrode rows for 182 units. That is not consistent with the documented dual-array
# recording, so the unit-to-area mapping in this file is not trustworthy and no analysis below
# splits units by area. All 182 units are treated as a single motor-cortical population.

# %% [markdown]
# ## Setup
#
# Files are streamed from the DANDI S3 bucket with `remfile` plus a local disk cache; nothing is
# downloaded in full. Spike times, kinematics and analysis all go through pynapple.

# %%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # this notebook is run non-interactively
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import nemos as nmo

ASSETS = {
    "MC_Maze": "https://api.dandiarchive.org/api/assets/26e85f09-39b7-480f-b337-278a8f034007/download/",
    "MC_RTT":  "https://api.dandiarchive.org/api/assets/2ae6bf3c-788b-4ece-8c01-4b4a5680b25b/download/",
}
CACHE = "/tmp/remfile_cache"

BIN_PSTH = 0.01      # s, PSTH bin
BIN_CONT = 0.05      # s, bin for all continuous (velocity) analyses
WIN_W = 0.30         # s, width of the trial rate window
SPD_TH = 100.0       # mm/s, threshold defining "the hand is moving" in MC_Maze
DELAY_WIN = (0.10, 0.40)   # s after target onset: the planning epoch, before the go cue


def open_nwb(key):
    f = remfile.File(ASSETS[key], disk_cache=remfile.DiskCache(CACHE))
    return NWBHDF5IO(file=h5py.File(f, "r"), load_namespaces=True).read()


def obs_intervals_set(nwbfile):
    """Spikes are only recorded inside these epochs, so every analysis is restricted to them."""
    oi = nwbfile.units["obs_intervals"][0]
    return nap.IntervalSet(start=oi[:, 0], end=oi[:, 1])


def load_units(nwbfile, ep):
    u = nwbfile.units
    spikes = {i: np.asarray(u["spike_times"][i]) for i in range(len(u.id))}
    return nap.TsGroup(spikes, time_support=ep,
                       metadata={"heldout": np.asarray(u["heldout"][:]).astype(bool)})


def lstsq_r2(X, Yv):
    """Least-squares fit of every column of Yv on X; returns coefficients and per-column R^2."""
    with np.errstate(all="ignore"):
        B = np.linalg.lstsq(X, Yv, rcond=None)[0]
        sst = ((Yv - Yv.mean(0)) ** 2).sum(0)
        r2 = 1 - ((Yv - X @ B) ** 2).sum(0) / np.where(sst > 0, sst, np.nan)
    return B, r2


# %% [markdown]
# ## 1. Load MC_Maze and check every data stream before using it

# %%
nwbfile = open_nwb("MC_Maze")
ep = obs_intervals_set(nwbfile)
units = load_units(nwbfile, ep)
spk = [units[k].t for k in units.index]

beh = nwbfile.processing["behavior"].data_interfaces
hand_pos, hand_vel = beh["hand_pos"], beh["hand_vel"]
t = np.asarray(hand_pos.timestamps[:])
pos = np.asarray(hand_pos.data[:])
vel = np.asarray(hand_vel.data[:])
spd = np.hypot(vel[:, 0], vel[:, 1])
trials = nwbfile.trials.to_dataframe()

dt = np.diff(t)
print("subject %s, %s" % (nwbfile.subject.subject_id, nwbfile.subject.species))
print("kinematics: %d samples, %.1f s span, median dt %.4f s, %d gaps > 10 ms (the inter-trial breaks)"
      % (len(t), t[-1] - t[0], np.median(dt), (dt > 0.01).sum()))
print("NaNs: position %d, velocity %d" % (np.isnan(pos).sum(), np.isnan(vel).sum()))
print("stored in mm (NWB conversion %g gives metres); speed median %.0f, p99 %.0f, max %.0f mm/s"
      % (hand_pos.conversion, np.median(spd), np.percentile(spd, 99), spd.max()))
print("units: %d, firing rate median %.1f Hz (range %.2f-%.1f)"
      % (len(units), np.median(units.rate), units.rate.min(), units.rate.max()))
print("observation intervals: %d covering %.0f s; trials: %d (%d successful)"
      % (len(ep), ep.tot_length(), len(trials), trials.success.sum()))

# %% [markdown]
# Trials with `num_barriers == 0` are unobstructed centre-out reaches, so the hand travels
# more or less straight to the target. Those are used for the trial-based direction analysis.
# The curved maze trials are kept for the continuous velocity analysis, where they are an asset:
# they broaden the distribution of hand velocities considerably.

# %%
sel = trials[(trials.num_barriers == 0) & trials.success].copy()
onset = sel.move_onset_time.values
i0, i1 = np.searchsorted(t, onset), np.searchsorted(t, onset + 0.2)

# Reach direction is measured from the hand itself rather than assumed from the target.
mv = np.stack([vel[a:b].mean(0) for a, b in zip(i0, i1)])
theta = np.arctan2(mv[:, 1], mv[:, 0])
peak_speed = np.array([spd[a:b].max() for a, b in zip(i0, np.searchsorted(t, onset + 0.4))])
tgt = np.stack([r.target_pos[r.active_target] for r in sel.itertuples()]).astype(float)
circ_err = np.abs(np.angle(np.exp(1j * (theta - np.arctan2(tgt[:, 1], tgt[:, 0])))))

print("straight reaches: %d" % len(sel))
print("measured hand direction vs target direction: median difference %.1f deg, p95 %.1f deg"
      % (np.degrees(np.median(circ_err)), np.degrees(np.percentile(circ_err, 95))))
print("peak speed: median %.0f mm/s (range %.0f-%.0f)" % (np.median(peak_speed), peak_speed.min(), peak_speed.max()))
print("every PSTH window lies inside a trial: min(onset-start) %.2f s, min(stop-onset) %.2f s"
      % ((onset - sel.start_time.values).min(), (sel.stop_time.values - onset).min()))

# mean speed profile, used later to locate peak hand speed
prof_off = np.arange(-50, 301)
PEAK_SPD_MS = float(prof_off[np.argmax(spd[np.searchsorted(t, onset)[:, None] + prof_off[None, :]].mean(0))])
print("mean hand speed peaks %+.0f ms after movement onset" % PEAK_SPD_MS)

# %% [markdown]
# ### Figure 1: raw data validation
#
# Before any analysis, look at the streams directly: reach paths, speed profiles, the sampled
# target directions, raw velocity traces and a spike raster.

# %%
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)
ang_col = plt.cm.hsv((theta + np.pi) / (2 * np.pi))

ax = fig.add_subplot(gs[0, 0])
for k, r in enumerate(sel.itertuples()):
    if k % 6:
        continue
    m = (t >= r.move_onset_time - 0.05) & (t <= r.move_onset_time + 0.6)
    ax.plot(pos[m, 0], pos[m, 1], lw=0.6, alpha=0.6, color=ang_col[k])
ax.set(xlabel="hand x (mm)", ylabel="hand y (mm)", aspect="equal",
       title="Straight (no-barrier) reach paths\ncoloured by measured reach direction")

ax = fig.add_subplot(gs[0, 1])
for k, r in enumerate(sel.itertuples()):
    if k % 6:
        continue
    m = (t >= r.move_onset_time - 0.2) & (t <= r.move_onset_time + 0.6)
    ax.plot(t[m] - r.move_onset_time, spd[m], lw=0.4, alpha=0.3, color="k")
ax.axvline(0, color="r", ls="--", lw=1)
ax.axvline(PEAK_SPD_MS / 1000, color="steelblue", ls=":", lw=2)
ax.set(xlabel="time from movement onset (s)", ylabel="speed (mm/s)",
       title="Hand speed profiles\n(blue dotted = mean peak, %+.0f ms)" % PEAK_SPD_MS)

ax = fig.add_subplot(gs[0, 2], projection="polar")
ax.hist(theta, bins=36, color="steelblue")
ax.set_rlabel_position(285)
ax.tick_params(labelsize=8)
ax.set_title("Reach directions sampled\n(%d straight trials)" % len(sel), pad=28)

t0 = trials.move_onset_time.iloc[10] - 1.0
m = (t >= t0) & (t <= t0 + 8)
ax = fig.add_subplot(gs[1, :])
ax.plot(t[m], vel[m, 0], lw=0.9, label="$v_x$")
ax.plot(t[m], vel[m, 1], lw=0.9, label="$v_y$")
ax.plot(t[m], spd[m], lw=1.2, color="k", label="speed")
for r in trials.itertuples():
    if t0 <= r.move_onset_time <= t0 + 8:
        ax.axvline(r.move_onset_time, color="r", ls="--", lw=1)
ax.legend(ncol=3, fontsize=8)
ax.set(xlabel="time (s)", ylabel="mm/s", title="Hand velocity (red dashed = movement onset)")

ax = fig.add_subplot(gs[2, :])
for row, ui in enumerate(np.argsort(units.rate.values)[::-1][:60]):
    s = spk[ui]
    s = s[(s >= t0) & (s <= t0 + 8)]
    ax.plot(s, np.full_like(s, row), "|", ms=3, color="C0")
for r in trials.itertuples():
    if t0 <= r.move_onset_time <= t0 + 8:
        ax.axvline(r.move_onset_time, color="r", ls="--", lw=1)
ax.set(xlim=(t0, t0 + 8), xlabel="time (s)", ylabel="unit",
       title="Spike raster, 60 highest-rate units")
fig.savefig("fig01_data_overview.png", dpi=140, bbox_inches="tight")
print("saved fig01_data_overview.png")

# %% [markdown]
# ## 2. Reach-direction tuning
#
# For each trial the firing rate is counted in a 300 ms window near movement onset. Rather than
# fixing that window by hand, its position is scanned and chosen to maximise the population's
# directional modulation. A cosine, $r = b_0 + d \cos(\theta - \mathrm{PD})$, is fitted by least
# squares; significance is assessed by permuting the trial-to-direction assignment 1000 times and
# comparing the observed modulation depth $d$ against the resulting null.

# %%
def epoch_rates(events, win):
    """Exact spike counts in [event+win0, event+win1], as rates (trials x units)."""
    a, b = events + win[0], events + win[1]
    cnt = np.stack([np.searchsorted(s, b) - np.searchsorted(s, a) for s in spk], 1)
    return cnt / (win[1] - win[0])


def cosine_fit(R, th, n_perm=1000, seed=0):
    X = np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1)
    B, r2 = lstsq_r2(X, R)
    out = dict(b0=B[0], depth=np.hypot(B[1], B[2]), pd=np.arctan2(B[2], B[1]), r2=r2)
    if n_perm:
        rng = np.random.default_rng(seed)
        null = np.empty((n_perm, R.shape[1]))
        for i in range(n_perm):
            Bs = np.linalg.lstsq(X[rng.permutation(len(th))], R, rcond=None)[0]
            null[i] = np.hypot(Bs[1], Bs[2])
        out["p"] = (null >= out["depth"]).mean(0)
    return out


offsets = np.round(np.arange(-0.30, 0.31, 0.05), 3)
scan = np.array([np.nanmean(cosine_fit(epoch_rates(onset, (o, o + WIN_W)), theta, n_perm=0)["depth"])
                 for o in offsets])
BEST_OFF = offsets[np.argmax(scan)]
MOVE_WIN = (BEST_OFF, BEST_OFF + WIN_W)
print("best %d ms window starts %+.0f ms relative to movement onset, i.e. centred %+.0f ms"
      % (WIN_W * 1000, BEST_OFF * 1000, (BEST_OFF + WIN_W / 2) * 1000))

R_move = epoch_rates(onset, MOVE_WIN)
R_delay = epoch_rates(sel.target_on_time.values, DELAY_WIN)
alive = R_move.sum(0) > 0
fit_move = cosine_fit(R_move, theta)
fit_delay = cosine_fit(R_delay, theta)
sig = (fit_move["p"] < 0.01) & alive

print("directionally tuned during movement (permutation p < 0.01): %d/%d units (%.0f%%)"
      % (sig.sum(), alive.sum(), 100 * sig.sum() / alive.sum()))
print("directionally tuned during the delay period: %d/%d units"
      % (((fit_delay["p"] < 0.01) & alive).sum(), alive.sum()))
print("modulation depth of tuned units: median %.1f Hz" % np.median(fit_move["depth"][sig]))

# %% [markdown]
# The sampled target directions are strongly non-uniform (see figure 1), so empirical tuning
# curves use 30° bins and only bins holding at least 15 trials are shown. The cosine fit itself
# uses every trial and does not depend on this binning.

# %%
NB = 12
edges = np.linspace(-np.pi, np.pi, NB + 1)
centers = (edges[:-1] + edges[1:]) / 2
dbin = np.clip(np.digitize(theta, edges) - 1, 0, NB - 1)
n_per = np.bincount(dbin, minlength=NB)
keep = n_per >= 15
kb = np.where(keep)[0]
print("direction bins retained: %d of %d, with %s trials each" % (keep.sum(), NB, n_per[keep]))

tc = np.full((NB, R_move.shape[1]), np.nan)
tc_sem = np.full_like(tc, np.nan)
for b in kb:
    tc[b] = R_move[dbin == b].mean(0)
    tc_sem[b] = R_move[dbin == b].std(0) / np.sqrt(n_per[b])

cpred = fit_move["b0"] + fit_move["depth"] * np.cos(centers[kb][:, None] - fit_move["pd"])
with np.errstate(all="ignore"):
    sst = ((tc[kb] - tc[kb].mean(0)) ** 2).sum(0)
    r2_binned = 1 - ((tc[kb] - cpred) ** 2).sum(0) / np.where(sst > 0, sst, np.nan)
print("cosine R^2 against direction-binned means, tuned units: median %.2f" % np.median(r2_binned[sig]))
print("(single-trial R^2 is far lower, median %.2f, because 300 ms Poisson counts are noisy)"
      % np.median(fit_move["r2"][sig]))

# PSTHs on a uniform grid; every window was verified above to lie inside an observed trial
grid = np.arange(0, t[-1] + BIN_PSTH, BIN_PSTH)
C = gaussian_filter1d(np.stack([np.histogram(s, bins=grid)[0] for s in spk], 1) / BIN_PSTH, 2.5, axis=0)
off = np.arange(int(round(-0.5 / BIN_PSTH)), int(round(0.7 / BIN_PSTH)))
psth = C[np.searchsorted(grid, onset)[:, None] + off[None, :]]
psth_dir = np.full((NB, len(off), C.shape[1]), np.nan)
for b in kb:
    psth_dir[b] = psth[dbin == b].mean(0)
tax = off * BIN_PSTH

# %% [markdown]
# ### Figure 2: four example units

# %%
ex = np.argsort(np.where(sig, fit_move["depth"], -1))[::-1][:4]
cols = plt.cm.hsv((centers + np.pi) / (2 * np.pi))
fig = plt.figure(figsize=(16, 8.5))
gs = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.42, height_ratios=[1, 1.1])
for j, ui in enumerate(ex):
    ax = fig.add_subplot(gs[0, j])
    for b in kb:
        ax.plot(tax, psth_dir[b, :, ui], color=cols[b], lw=1.4)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvspan(*MOVE_WIN, color="0.85", zorder=0)
    ax.set_title("unit %d" % units.index[ui], fontsize=11)
    ax.set_xlabel("time from movement onset (s)")
    if j == 0:
        ax.set_ylabel("firing rate (Hz)")
        ax.text(0.02, 0.98, "shading = rate window", transform=ax.transAxes, fontsize=7, va="top")

    ax = fig.add_subplot(gs[1, j], projection="polar")
    th_c = np.append(centers[kb], centers[kb][0] + 2 * np.pi)
    ax.errorbar(th_c, np.append(tc[kb, ui], tc[kb[0], ui]),
                yerr=np.append(tc_sem[kb, ui], tc_sem[kb[0], ui]),
                color="k", marker="o", ms=4, lw=1.4, capsize=2, label="observed")
    tt = np.linspace(-np.pi, np.pi, 200)
    ax.plot(tt, np.maximum(fit_move["b0"][ui] + fit_move["depth"][ui] * np.cos(tt - fit_move["pd"][ui]), 0),
            color="crimson", lw=2, label="cosine fit")
    ax.set_rlabel_position(285)
    ax.tick_params(labelsize=8)
    ax.set_title("PD %.0f$\\degree$  depth %.1f Hz  $R^2_{bin}$=%.2f"
                 % (np.degrees(fit_move["pd"][ui]), fit_move["depth"][ui], r2_binned[ui]), pad=20, fontsize=9)
    if j == 0:
        ax.legend(loc="lower left", bbox_to_anchor=(-0.35, -0.30), fontsize=8, frameon=False)
fig.suptitle("Reach-direction tuning in macaque motor cortex (DANDI:000128 MC_Maze, %d straight reaches)\n"
             "top: direction-conditioned PSTHs (line colour = reach direction);   "
             "bottom: mean rate in the %+d to %+d ms window with cosine fit"
             % (len(sel), MOVE_WIN[0] * 1000, MOVE_WIN[1] * 1000), y=0.99, fontsize=12)
fig.savefig("fig02_direction_tuning_examples.png", dpi=140, bbox_inches="tight")
print("saved fig02_direction_tuning_examples.png")

# %% [markdown]
# ### Figure 3: population summary
#
# The window scan is worth reading carefully. The directional signal is strongest for a window
# centred essentially on movement onset, which is roughly 125 ms *before* the hand reaches peak
# speed. This is the first indication that the activity leads the movement rather than following it.
#
# The delay-period panel is a negative result and is labelled as such. Many units are direction
# tuned while the monkey waits for the go cue, but their preferred directions during the delay do
# not predict their preferred directions during the movement (circular correlation near zero).

# %%
fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

ax = fig.add_subplot(gs[0, 0], projection="polar")
ax.hist(fit_move["pd"][sig], bins=np.linspace(-np.pi, np.pi, 25), color="steelblue", edgecolor="w", lw=0.5)
ax.set_rlabel_position(285)
ax.tick_params(labelsize=8)
ax.set_title("Preferred directions\n(%d tuned units)" % sig.sum(), pad=30, fontsize=11)

ax = fig.add_subplot(gs[0, 1])
ax.hist(fit_move["depth"][sig], bins=25, color="steelblue", edgecolor="w")
ax.axvline(np.median(fit_move["depth"][sig]), color="crimson", lw=2)
ax.set(xlabel="modulation depth (Hz)", ylabel="units",
       title="Depth of directional modulation\nmedian %.1f Hz" % np.median(fit_move["depth"][sig]))

ax = fig.add_subplot(gs[0, 2])
ax.hist(r2_binned[sig], bins=np.linspace(0, 1, 26), color="steelblue", edgecolor="w")
ax.axvline(np.median(r2_binned[sig]), color="crimson", lw=2)
ax.set(xlabel="$R^2$ of cosine fit (direction-binned means)", ylabel="units",
       title="Cosine tuning is a good description\nmedian $R^2$ = %.2f" % np.median(r2_binned[sig]))

ax = fig.add_subplot(gs[1, 0])
ctr = (offsets + WIN_W / 2) * 1000
bc = (BEST_OFF + WIN_W / 2) * 1000
ax.plot(ctr, scan, "o-", color="k")
ax.axvline(bc, color="crimson", ls="--", label="best window centre (%+.0f ms)" % bc)
ax.axvline(PEAK_SPD_MS, color="steelblue", ls=":", lw=2, label="peak hand speed (%+.0f ms)" % PEAK_SPD_MS)
ax.legend(fontsize=8)
ax.set(xlabel="centre of the %d ms rate window\nrelative to movement onset (ms)" % (WIN_W * 1000),
       ylabel="mean modulation depth (Hz)",
       title="Direction signal peaks %d ms\nbefore peak hand speed" % (PEAK_SPD_MS - bc))

ax = fig.add_subplot(gs[1, 1])
both = sig & (fit_delay["p"] < 0.01)
a_, b_ = fit_move["pd"][both], fit_delay["pd"][both]
am, bm = np.angle(np.mean(np.exp(1j * a_))), np.angle(np.mean(np.exp(1j * b_)))
rcc = (np.sum(np.sin(a_ - am) * np.sin(b_ - bm))
       / np.sqrt(np.sum(np.sin(a_ - am) ** 2) * np.sum(np.sin(b_ - bm) ** 2)))
dl = np.degrees(np.angle(np.exp(1j * (b_ - a_))))
ax.scatter(np.degrees(a_), np.degrees(b_), s=14, color="steelblue")
ax.plot([-180, 180], [-180, 180], "k--", lw=1)
ax.set(xlabel="preferred direction, movement (deg)", ylabel="preferred direction, delay (deg)",
       xticks=[-180, -90, 0, 90, 180], yticks=[-180, -90, 0, 90, 180],
       title="Delay-period PDs do NOT predict movement PDs\n"
             "%d units tuned in both; circular $r$ = %.2f, %.0f%% within 45$\\degree$ (chance 25%%)"
             % (both.sum(), rcc, 100 * (np.abs(dl) < 45).mean()))
ax.title.set_fontsize(10)

ax = fig.add_subplot(gs[1, 2])
pref_bin = np.array([kb[np.argmin(np.abs(np.angle(np.exp(1j * (centers[kb] - p)))))] for p in fit_move["pd"]])
anti_bin = np.array([kb[np.argmin(np.abs(np.angle(np.exp(1j * (centers[kb] - p - np.pi)))))] for p in fit_move["pd"]])
uu = np.where(sig)[0]
diff = np.stack([psth_dir[pref_bin[u], :, u] - psth_dir[anti_bin[u], :, u] for u in uu])
diff = diff / np.abs(diff).max(1, keepdims=True)
im = ax.imshow(diff[np.argsort(np.argmax(diff, 1))], aspect="auto", cmap="RdBu_r",
               norm=TwoSlopeNorm(0, -1, 1), extent=[tax[0], tax[-1], len(uu), 0], interpolation="nearest")
ax.axvline(0, color="k", ls="--", lw=1)
ax.set(xlabel="time from movement onset (s)", ylabel="unit (sorted by peak time)",
       title="Preferred minus anti-preferred rate\n(each unit normalised)")
fig.colorbar(im, ax=ax, label="normalised rate difference")
fig.suptitle("Population summary of reach-direction tuning (DANDI:000128 MC_Maze, monkey Jenkins, %d units)"
             % len(units), y=0.965, fontsize=13)
fig.savefig("fig03_direction_population.png", dpi=140, bbox_inches="tight")
print("saved fig03_direction_population.png")

# %% [markdown]
# ## 3. From direction to velocity
#
# The trial-based analysis above collapses each reach to a single direction and a single number.
# The rest of the notebook drops that abstraction and works directly with the continuous 1 kHz
# hand velocity, binned to 50 ms and paired with spike counts.
#
# A lag is applied by shifting the velocity timestamps before binning, so that spikes in the bin
# at time $\tau$ are paired with hand velocity at $\tau + \mathrm{lag}$. A positive lag therefore
# means neural activity **leads** the hand.

# %%
counts = units.count(BIN_CONT, ep)
Y = counts.values / BIN_CONT


def lagged_velocity(lag):
    v = nap.TsdFrame(t=t - lag, d=vel, columns=["vx", "vy"])
    out = v.restrict(ep).bin_average(BIN_CONT, ep)
    assert np.allclose(out.t, counts.t), "velocity and spike-count bin grids disagree"
    return out.values


# Cross-validation folds are contiguous blocks of ~60 trials, so that train and test bins are
# never taken from the same reach (neighbouring bins are strongly autocorrelated).
blocks = np.searchsorted(ep.start, counts.t, side="right") // 60


def decode_r2(Xr, Vt, blk, n_fold=5, return_pred=False):
    f = blk % n_fold
    pred = np.empty_like(Vt)
    with np.errstate(all="ignore"):
        for k in range(n_fold):
            tr, te = f != k, f == k
            pred[te] = Ridge(alpha=1.0).fit(Xr[tr], Vt[tr]).predict(Xr[te])
    r2 = 1 - ((Vt - pred) ** 2).sum() / ((Vt - Vt.mean(0)) ** 2).sum()
    return (r2, pred) if return_pred else r2


V0 = lagged_velocity(0.0)
moving0 = np.isfinite(V0).all(1) & (np.hypot(V0[:, 0], V0[:, 1]) > SPD_TH)
print("%d bins of %.0f ms; %d (%.0f%%, %.0f s) have hand speed above %.0f mm/s"
      % (len(V0), BIN_CONT * 1000, moving0.sum(), 100 * moving0.mean(), moving0.sum() * BIN_CONT, SPD_TH))

# %% [markdown]
# ### The lag between motor cortex and the hand
#
# The lag scan runs over **all** observed bins rather than movement bins only. Within a single
# reach the velocity direction barely changes, so a movement-only scan is nearly flat; it is the
# transitions between rest and movement that carry the timing information.

# %%
lags = np.round(np.arange(-0.30, 0.301, 0.02), 3)
r2_lag = np.full((len(lags), Y.shape[1]), np.nan)
r2_lag_pop = np.full(len(lags), np.nan)
Ysm = gaussian_filter1d(Y, 2.0, axis=0)          # ~100 ms smoothing, for the population decode
for i, lg in enumerate(tqdm(lags, desc="lag scan")):
    V = lagged_velocity(lg)
    m = np.isfinite(V).all(1)
    r2_lag[i] = lstsq_r2(np.column_stack([np.ones(m.sum()), V[m]]), Y[m])[1]
    r2_lag_pop[i] = decode_r2(Ysm[m], V[m], blocks[m])

BEST_LAG = lags[np.argmax(r2_lag_pop)]
su_curve = np.nanmedian(r2_lag[:, sig], 1)
best_per_unit = lags[np.nanargmax(r2_lag, 0)]
print("population decode of hand velocity peaks at %+.0f ms (cross-validated R^2 %.3f; %.3f at zero lag)"
      % (BEST_LAG * 1000, r2_lag_pop.max(), r2_lag_pop[np.isclose(lags, 0)][0]))
print("single-unit velocity model peaks at %+.0f ms" % (1000 * lags[np.argmax(su_curve)]))
print("per-unit best lag across tuned units: median %+.0f ms, IQR [%+.0f, %+.0f] ms"
      % tuple(1000 * np.percentile(best_per_unit[sig], [50, 25, 75])))
print("positive lag means the neural activity leads the hand")

# %% [markdown]
# ### Velocity tuning at the optimal lag

# %%
V = lagged_velocity(BEST_LAG)
mov = np.isfinite(V).all(1) & (np.hypot(V[:, 0], V[:, 1]) > SPD_TH)
Vm, Yv = V[mov], Y[mov]
speed = np.hypot(Vm[:, 0], Vm[:, 1])
vdir = np.arctan2(Vm[:, 1], Vm[:, 0])

Bv, r2_vel = lstsq_r2(np.column_stack([np.ones(mov.sum()), Vm]), Yv)
pd_vel = np.arctan2(Bv[2], Bv[1])
gain = np.hypot(Bv[1], Bv[2])
dpd = np.angle(np.exp(1j * (pd_vel - fit_move["pd"])))
print("moving-bin speed: median %.0f mm/s (p10 %.0f, p90 %.0f)"
      % (np.median(speed), *np.percentile(speed, [10, 90])))
print("velocity gain of tuned units: median %.1f Hz change at 500 mm/s along the preferred direction"
      % (500 * np.median(gain[sig])))
print("preferred direction from the continuous velocity model vs the trial-based cosine fit:")
print("  median |difference| %.1f deg, %.0f%% within 45 deg (chance 25%%)"
      % (np.degrees(np.median(np.abs(dpd[sig]))), 100 * (np.abs(dpd[sig]) < np.pi / 4).mean()))

# %% [markdown]
# The decisive analysis. Each bin is labelled by hand speed and by the angle between hand
# direction and that unit's own preferred direction, then the mean rate is taken in every
# speed-by-angle cell. If units coded direction alone, the curves for different relative
# directions would be flat in speed and merely offset from one another. They are not: they fan
# apart as speed grows.

# %%
rel = np.angle(np.exp(1j * (vdir[:, None] - pd_vel[None, :])))
sedges = np.array([100, 200, 300, 400, 500, 650, 900])
scent = (sedges[:-1] + sedges[1:]) / 2
aedges = np.linspace(-np.pi, np.pi, 7)
acent = (aedges[:-1] + aedges[1:]) / 2
sb = np.digitize(speed, sedges) - 1
grid = np.full((len(scent), len(acent), Y.shape[1]), np.nan)
for j in tqdm(range(Y.shape[1]), desc="speed x direction"):
    ab = np.digitize(rel[:, j], aedges) - 1
    for a in range(len(acent)):
        for s in range(len(scent)):
            k = (ab == a) & (sb == s)
            if k.sum() > 30:
                grid[s, a, j] = Yv[k, j].mean()
g_norm = grid[..., sig] / np.nanmean(grid[..., sig], (0, 1))
pop_grid = np.nanmean(g_norm, 2)
ipd = int(np.argmin(np.abs(acent)))
iap = int(np.argmax(np.abs(acent)))
iorth = int(np.argmin(np.abs(np.abs(acent) - np.pi / 2)))
print("normalised population rate, slowest to fastest speed bin:")
print("  toward the preferred direction: %.2f -> %.2f" % (pop_grid[0, ipd], pop_grid[-1, ipd]))
print("  away from the preferred direction: %.2f -> %.2f" % (pop_grid[0, iap], pop_grid[-1, iap]))

LIM = 700
Vtsd = nap.TsdFrame(t=counts.t[mov], d=Vm, columns=["vx", "vy"])
tc2d_d, edges2d = nap.compute_2d_tuning_curves(units, Vtsd, 13, minmax=(-LIM, LIM, -LIM, LIM))
tc2d = np.stack([tc2d_d[k] for k in units.index])
occ, _, _ = np.histogram2d(Vm[:, 0], Vm[:, 1], bins=13, range=[[-LIM, LIM]] * 2)
xe, ye = edges2d


def smooth_nan(M, valid, sigma=0.8):
    """Gaussian-smooth a rate map while ignoring bins with too little occupancy."""
    num = gaussian_filter(np.where(valid, M, 0.0), sigma, mode="nearest")
    den = gaussian_filter(valid.astype(float), sigma, mode="nearest")
    return np.where(valid, np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan), np.nan)


# %% [markdown]
# ### Figure 4: velocity tuning

# %%
fig = plt.figure(figsize=(15.5, 10))
gs = fig.add_gridspec(3, 4, hspace=0.66, wspace=0.45)

ax = fig.add_subplot(gs[0, :2])
ax.plot(lags * 1000, r2_lag_pop / r2_lag_pop.max(), "o-", color="crimson",
        label="population decode of hand velocity (peak $R^2$ = %.2f)" % r2_lag_pop.max())
ax.plot(lags * 1000, su_curve / su_curve.max(), "s-", color="steelblue", ms=4,
        label="single unit, linear velocity model (peak median $R^2$ = %.4f)" % su_curve.max())
ax.axvline(BEST_LAG * 1000, color="crimson", ls="--")
ax.axvline(0, color="0.6", lw=1)
ax.legend(fontsize=8, loc="lower center")
ax.set(xlabel="lag (ms):  positive = neural activity leads the hand",
       ylabel="$R^2$ (normalised to peak)",
       title="Motor cortex leads the hand by ~%d ms" % (BEST_LAG * 1000))

ax = fig.add_subplot(gs[0, 2])
ax.hist(np.degrees(dpd[sig]), bins=np.linspace(-180, 180, 37), color="steelblue", edgecolor="w")
ax.set(xlabel="PD(velocity model) - PD(trial cosine fit)  (deg)", ylabel="units",
       xticks=[-180, -90, 0, 90, 180],
       title="Two independent estimates of PD agree\nmedian |difference| %.0f$\\degree$"
             % np.degrees(np.median(np.abs(dpd[sig]))))

ax = fig.add_subplot(gs[0, 3])
ax.hist(gain[sig] * 500, bins=25, color="steelblue", edgecolor="w")
ax.set(xlabel="rate change at 500 mm/s along PD (Hz)", ylabel="units",
       title="Velocity gain\nmedian %.1f Hz" % np.median(gain[sig] * 500))

for j, ui in enumerate(ex):          # the same four units as figure 2
    ax = fig.add_subplot(gs[1, j])
    im = ax.imshow(smooth_nan(tc2d[ui].T, occ.T >= 25), origin="lower",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="viridis", interpolation="nearest")
    L = 0.75 * xe[-1]
    ax.arrow(0, 0, L * np.cos(pd_vel[ui]), L * np.sin(pd_vel[ui]), color="w", width=18,
             length_includes_head=True)
    ax.set(xlabel="$v_x$ (mm/s)", ylabel="$v_y$ (mm/s)" if j == 0 else "",
           title="unit %d (arrow = PD)" % ui)
    fig.colorbar(im, ax=ax, label="Hz" if j == 3 else "")

ax = fig.add_subplot(gs[2, 0:2])
im = ax.imshow(pop_grid, origin="lower", aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(1.0),
               extent=[np.degrees(acent[0]) - 30, np.degrees(acent[-1]) + 30, scent[0] - 50, scent[-1] + 125])
ax.set(xlabel="hand direction relative to each unit's PD (deg)", ylabel="hand speed (mm/s)",
       xticks=[-150, -90, -30, 30, 90, 150],
       title="Population firing rate depends on direction AND speed\n"
             "(mean over %d tuned units, normalised to each unit's mean rate)" % sig.sum())
fig.colorbar(im, ax=ax, label="rate / mean rate")

ax = fig.add_subplot(gs[2, 2:])
for i, lab, c in [(ipd, "toward PD", "crimson"), (iorth, "orthogonal to PD", "0.4"),
                  (iap, "away from PD", "steelblue")]:
    mu = np.nanmean(g_norm[:, i, :], 1)
    se = np.nanstd(g_norm[:, i, :], 1) / np.sqrt(sig.sum())
    ax.errorbar(scent, mu, yerr=se, marker="o", lw=2, color=c, label=lab, capsize=3)
ax.axhline(1, color="k", lw=0.8, ls=":")
ax.legend(fontsize=9)
ax.set(xlabel="hand speed (mm/s)", ylabel="rate / mean rate",
       title="Speed scales the directional signal\n(the signature of velocity, not direction, tuning)")
fig.suptitle("Hand-velocity tuning in macaque motor cortex (DANDI:000128 MC_Maze, %d ms bins, movement periods)"
             % (BIN_CONT * 1000), y=0.965, fontsize=13)
fig.savefig("fig04_velocity_tuning.png", dpi=140, bbox_inches="tight")
print("saved fig04_velocity_tuning.png")

# %% [markdown]
# ## 4. Nested Poisson GLMs with NeMoS
#
# The tuning curves above are descriptive. To ask which kinematic variable actually predicts
# spiking, a set of nested Poisson GLMs is fitted to the same movement bins and compared by
# 5-fold cross-validated McFadden pseudo-$R^2$, with folds again defined by contiguous blocks
# of trials.
#
# One caveat matters for reading the result. A GLM uses an exponential inverse link, so the
# "velocity, linear" model ($\log \lambda = b_0 + b_x v_x + b_y v_y$) forces rate to grow
# *exponentially* with speed, which real motor cortex does not do. That model is therefore
# handicapped and is included only for completeness. The fair general velocity model is the 2D
# spline over $(v_x, v_y)$, which can represent any smooth dependence on the velocity vector.

# %%
SC = 500.0                                    # scale velocities to O(1) for conditioning
SLO, SHI = SPD_TH, float(np.percentile(speed, 99.5))
VLO, VHI = -800.0, 800.0
# B-spline bases return NaN outside their bounds, so inputs are clipped into range.
Bs = nmo.basis.BSplineEval(6, bounds=(SLO, SHI)).compute_features(np.clip(speed, SLO, SHI))
vb = nmo.basis.BSplineEval(5, bounds=(VLO, VHI)) * nmo.basis.BSplineEval(5, bounds=(VLO, VHI))
Bv2 = vb.compute_features(np.clip(Vm[:, 0], VLO, VHI), np.clip(Vm[:, 1], VLO, VHI))
Bd = np.column_stack([np.cos(vdir), np.sin(vdir)])

designs = {
    "direction only":      Bd,
    "speed only":          Bs,
    "direction + speed":   np.column_stack([Bd, Bs]),
    "velocity, linear":    Vm / SC,
    "velocity, 2D spline": Bv2,
}
Ycnt = np.asarray(counts.values[mov], dtype=float)
blk_mov = blocks[mov]
for k, X in designs.items():
    print("  %-20s %2d features, %d NaNs" % (k, X.shape[1], np.isnan(np.asarray(X)).sum()))


def poisson_dev(y, lam):
    lam = np.maximum(lam, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / lam), 0.0)
    return 2 * (term - (y - lam)).sum(0)


def cv_pseudo_r2(X, Yc, blk, n_fold=5):
    f = blk % n_fold
    pred = np.empty_like(Yc)
    for k in range(n_fold):
        tr, te = f != k, f == k
        g = nmo.glm.PopulationGLM(observation_model="Poisson", regularizer="Ridge",
                                  regularizer_strength=1e-4, solver_name="LBFGS",
                                  solver_kwargs={"tol": 1e-10, "maxiter": 400})
        g.fit(X[tr], Yc[tr])
        pred[te] = np.asarray(g.predict(X[te]))
    null = np.repeat(Yc.mean(0, keepdims=True), len(Yc), 0)
    return 1 - poisson_dev(Yc, pred) / poisson_dev(Yc, null)


glm_res = {}
for k, X in tqdm(designs.items(), desc="fitting GLMs"):
    glm_res[k] = cv_pseudo_r2(np.asarray(X, dtype=float), Ycnt, blk_mov)
    print("  %-20s cross-validated pseudo-R^2: median %.4f over tuned units"
          % (k, np.median(glm_res[k][sig])))

kv, kd = "velocity, 2D spline", "direction only"
win = (glm_res[kv] > glm_res[kd])[sig]
gain_frac = (glm_res[kv][sig] - glm_res[kd][sig]) / np.maximum(glm_res[kv][sig], 1e-9)
print("the general velocity model beats direction-only in %d/%d tuned units (%.0f%%), "
      "median improvement %.0f%%" % (win.sum(), sig.sum(), 100 * win.mean(), 100 * np.median(gain_frac)))

# %% [markdown]
# ### Figure 5: model comparison

# %%
names = list(glm_res.keys())
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw=dict(wspace=0.32))
ax = axes[0]
bp = ax.boxplot([glm_res[k][sig] for k in names], tick_labels=[n.replace(" ", "\n") for n in names],
                showfliers=False, patch_artist=True)
for p in bp["boxes"]:
    p.set_facecolor("lightsteelblue")
ax.tick_params(axis="x", labelsize=8)
ax.text(0.02, 0.98, "the linear velocity model is handicapped by the\nexponential link (it makes rate grow "
                    "exponentially\nwith speed); the 2D spline is the general version",
        transform=ax.transAxes, fontsize=6.5, va="top", color="0.35")
ax.set(ylabel="cross-validated pseudo-$R^2$",
       title="Nested Poisson GLMs (NeMoS)\n%d directionally tuned units" % sig.sum())

ax = axes[1]
ax.scatter(glm_res[kd][sig], glm_res[kv][sig], s=16, color="steelblue")
lim = [0, max(glm_res[kd][sig].max(), glm_res[kv][sig].max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1)
ax.set(xlim=lim, ylim=lim, xlabel="pseudo-$R^2$, direction only",
       ylabel="pseudo-$R^2$, velocity (2D spline)",
       title="A general velocity model beats direction alone\n%.0f%% of units above the diagonal"
             % (100 * win.mean()))

ax = axes[2]
ax.hist(100 * gain_frac, bins=np.linspace(-50, 100, 31), color="steelblue", edgecolor="w")
ax.axvline(0, color="k", lw=1)
ax.axvline(100 * np.median(gain_frac), color="crimson", lw=2)
ax.set(xlabel="gain from modelling speed\n(% of velocity-model pseudo-$R^2$)", ylabel="units",
       title="Median improvement %.0f%%" % (100 * np.median(gain_frac)))
fig.suptitle("Direction alone is not enough: firing rate encodes hand velocity (direction $\\times$ speed)",
             y=1.02, fontsize=13)
fig.savefig("fig05_glm_model_comparison.png", dpi=140, bbox_inches="tight")
print("saved fig05_glm_model_comparison.png")

# %% [markdown]
# ## 5. Population decoding
#
# Two decoders are compared. The **population vector** is the classic Georgopoulos construction:
# each unit contributes a vector along its own preferred direction, weighted by its z-scored
# firing rate, and the vectors are summed. It fits nothing to the data beyond the preferred
# directions. The **ridge decoder** is a fitted linear readout of the full velocity vector,
# evaluated only on held-out folds.

# %%
Rsm = gaussian_filter1d(Y, 2.0, axis=0)
z = (Rsm - Rsm.mean(0)) / np.maximum(Rsm.std(0), 1e-9)
U = np.stack([np.cos(pd_vel), np.sin(pd_vel)], 1)
PV = z[:, sig] @ U[sig]
pv_len = np.hypot(PV[:, 0], PV[:, 1])
Vfull = lagged_velocity(BEST_LAG)
true_ang = np.arctan2(Vfull[:, 1], Vfull[:, 0])
spd_full = np.hypot(Vfull[:, 0], Vfull[:, 1])
pv_err = np.degrees(np.angle(np.exp(1j * (np.arctan2(PV[:, 1], PV[:, 0])[mov] - true_ang[mov]))))
r_len = np.corrcoef(pv_len[mov], spd_full[mov])[0, 1]
print("population vector: median |direction error| %.1f deg, %.0f%% within 45 deg (chance 25%%)"
      % (np.median(np.abs(pv_err)), 100 * (np.abs(pv_err) < 45).mean()))
print("population vector length vs hand speed: r = %.2f" % r_len)

finite = np.isfinite(Vfull).all(1)
_, pred_f = decode_r2(Rsm[finite], Vfull[finite], blocks[finite], return_pred=True)
pred = np.full_like(Vfull, np.nan)
pred[finite] = pred_f


def r2(a, b):
    return 1 - ((a - b) ** 2).sum() / ((a - a.mean(0)) ** 2).sum()


dec_err = np.degrees(np.angle(np.exp(1j * (np.arctan2(pred[:, 1], pred[:, 0])[mov] - true_ang[mov]))))
r_spd = np.corrcoef(np.hypot(pred[mov, 0], pred[mov, 1]), spd_full[mov])[0, 1]
print("ridge decode over all observed bins: R^2 vx %.3f, vy %.3f, overall %.3f"
      % (r2(Vfull[finite, 0], pred[finite, 0]), r2(Vfull[finite, 1], pred[finite, 1]),
         r2(Vfull[finite], pred[finite])))
print("ridge decode over movement bins only: R^2 %.3f" % r2(Vfull[mov], pred[mov]))
print("ridge decode: median |direction error| %.1f deg; decoded vs actual speed r = %.2f"
      % (np.median(np.abs(dec_err)), r_spd))

# %% [markdown]
# ### Figure 6: decoding

# %%
fig = plt.figure(figsize=(15.5, 9))
gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.30, height_ratios=[1, 1, 1.15])
sl = slice(int(np.argmax(counts.t > 300)), int(np.argmax(counts.t > 300)) + 320)
for r, (comp, lab) in enumerate([(0, "$v_x$"), (1, "$v_y$")]):
    ax = fig.add_subplot(gs[r, :])
    ax.plot(counts.t[sl], Vfull[sl, comp], color="k", lw=1.6, label="actual hand velocity")
    ax.plot(counts.t[sl], pred[sl, comp], color="crimson", lw=1.4, label="decoded from %d units" % len(units))
    ax.set_ylabel("%s (mm/s)" % lab)
    if r == 0:
        ax.legend(ncol=2, fontsize=9, loc="upper right")
        ax.set_title("Hand velocity decoded from the population (5-fold cross-validated, %+.0f ms lag)"
                     % (BEST_LAG * 1000))
    else:
        ax.set_xlabel("time (s)")

ax = fig.add_subplot(gs[2, 0])
bins = np.linspace(-180, 180, 49)
ax.hist(pv_err, bins=bins, color="steelblue", alpha=0.85,
        label="population vector (median |err| %.0f$\\degree$)" % np.median(np.abs(pv_err)))
ax.hist(dec_err, bins=bins, color="crimson", alpha=0.55,
        label="ridge decoder (median |err| %.0f$\\degree$)" % np.median(np.abs(dec_err)))
ax.axhline(mov.sum() / (len(bins) - 1), color="k", ls=":", lw=1.2, label="chance (uniform)")
ax.legend(fontsize=7.5, loc="upper left")
ax.set(xlabel="decoded minus actual reach direction (deg)", ylabel="bins", xticks=[-180, -90, 0, 90, 180],
       title="Reach direction is decoded\nfrom single %d ms bins" % (BIN_CONT * 1000))

ax = fig.add_subplot(gs[2, 1])
ax.hexbin(spd_full[mov], np.hypot(pred[mov, 0], pred[mov, 1]), gridsize=45, cmap="Blues", bins="log", mincnt=1)
ax.plot([0, 1000], [0, 1000], "k--", lw=1)
ax.set(xlim=(0, 1000), ylim=(0, 1000), xlabel="actual hand speed (mm/s)", ylabel="decoded speed (mm/s)",
       title="Speed is decoded too, though ridge\nshrinkage compresses it: r = %.2f" % r_spd)

ax = fig.add_subplot(gs[2, 2])
ax.hexbin(spd_full[mov], pv_len[mov], gridsize=45, cmap="Purples", bins="log", mincnt=1)
ax.set(xlabel="actual hand speed (mm/s)", ylabel="population vector length (a.u.)",
       title="Population vector length is only\nweakly related to speed: r = %.2f" % r_len)
fig.suptitle("Population coding of reach velocity (DANDI:000128 MC_Maze)", y=0.965, fontsize=13)
fig.savefig("fig06_population_decoding.png", dpi=140, bbox_inches="tight")
print("saved fig06_population_decoding.png")

# %% [markdown]
# ## 6. Replication in an independent dataset
#
# `MC_RTT` (DANDI:000129) is a different monkey performing a different task: instead of cued,
# delayed centre-out reaches it makes continuous self-paced reaches to randomly placed targets.
# The movements are slower and more continuous, so the movement threshold is set to that session's
# own 60th speed percentile rather than reusing the MC_Maze value.
#
# Significance here is assessed with a circular-shift permutation, which destroys the pairing
# between neural activity and kinematics while preserving the autocorrelation of both. That is the
# appropriate null for continuous data, where a plain shuffle would be far too permissive.

# %%
rtt = open_nwb("MC_RTT")
ep_r = obs_intervals_set(rtt)
units_r = load_units(rtt, ep_r)
fv = rtt.processing["behavior"].data_interfaces["finger_vel"]
vv = np.asarray(fv.data[:])
tv = (np.asarray(fv.timestamps[:]) if fv.timestamps is not None
      else fv.starting_time + np.arange(len(vv)) / fv.rate)
print("MC_RTT: subject %s, %d units, %d observation intervals totalling %.0f s"
      % (rtt.subject.subject_id, len(units_r), len(ep_r), ep_r.tot_length()))
print("finger velocity: %d samples at %.0f Hz; speed percentiles (mm/s) %s"
      % (len(vv), 1 / np.median(np.diff(tv)), np.round(np.nanpercentile(np.hypot(vv[:, 0], vv[:, 1]), [50, 90, 99]), 1)))
print("NaN samples: %d of %d (%.3f%%), excluded by the isfinite masks below"
      % (np.isnan(vv).any(1).sum(), len(vv), 100 * np.isnan(vv).any(1).mean()))

counts_r = units_r.count(BIN_CONT, ep_r)
Y_r = counts_r.values / BIN_CONT
blocks_r = np.arange(len(counts_r.t)) // 200          # ~10 s blocks for cross-validation


def lagged_r(lag):
    return nap.TsdFrame(t=tv - lag, d=vv, columns=["vx", "vy"]).restrict(ep_r).bin_average(BIN_CONT, ep_r).values


SPD_TH_R = float(np.nanpercentile(np.hypot(*lagged_r(0.0).T), 60))
print("movement threshold for this session (60th speed percentile): %.0f mm/s" % SPD_TH_R)

Ysm_r = gaussian_filter1d(Y_r, 2.0, axis=0)
pop_r = np.full(len(lags), np.nan)
su_r = np.full((len(lags), Y_r.shape[1]), np.nan)
for i, lg in enumerate(tqdm(lags, desc="MC_RTT lag scan")):
    Vr_ = lagged_r(lg)
    m = np.isfinite(Vr_).all(1)
    su_r[i] = lstsq_r2(np.column_stack([np.ones(m.sum()), Vr_[m]]), Y_r[m])[1]
    pop_r[i] = decode_r2(Ysm_r[m], Vr_[m], blocks_r[m])
BEST_R = lags[np.argmax(pop_r)]
print("MC_RTT population decode peaks at %+.0f ms (R^2 %.3f); single-unit peak %+.0f ms"
      % (BEST_R * 1000, pop_r.max(), 1000 * lags[np.nanargmax(np.nanmedian(su_r, 1))]))

Vr_ = lagged_r(BEST_R)
mov_r = np.isfinite(Vr_).all(1) & (np.hypot(Vr_[:, 0], Vr_[:, 1]) > SPD_TH_R)
Vmr, Yvr = Vr_[mov_r], Y_r[mov_r]
sp_r = np.hypot(Vmr[:, 0], Vmr[:, 1])
th_r = np.arctan2(Vmr[:, 1], Vmr[:, 0])
Br, _ = lstsq_r2(np.column_stack([np.ones(mov_r.sum()), Vmr]), Yvr)
pd_r = np.arctan2(Br[2], Br[1])
obs_amp = np.hypot(Br[1], Br[2])

rng = np.random.default_rng(0)
null = np.empty((300, Y_r.shape[1]))
for i in tqdm(range(300), desc="MC_RTT circular-shift permutations"):
    Vs = np.roll(Vmr, rng.integers(500, len(Vmr) - 500), axis=0)
    null[i] = np.hypot(*lstsq_r2(np.column_stack([np.ones(len(Vs)), Vs]), Yvr)[0][1:3])
sig_r = (null >= obs_amp).mean(0) < 0.01
print("velocity-tuned units (circular-shift permutation p < 0.01): %d/%d (%.0f%%)"
      % (sig_r.sum(), len(sig_r), 100 * sig_r.mean()))

rel_r = np.angle(np.exp(1j * (th_r[:, None] - pd_r[None, :])))
sedges_r = np.percentile(sp_r, [0, 20, 40, 60, 80, 95, 100])
scent_r = (sedges_r[:-1] + sedges_r[1:]) / 2
sb_r = np.clip(np.digitize(sp_r, sedges_r) - 1, 0, len(scent_r) - 1)
grid_r = np.full((len(scent_r), len(acent), Y_r.shape[1]), np.nan)
for j in range(Y_r.shape[1]):
    ab = np.digitize(rel_r[:, j], aedges) - 1
    for a in range(len(acent)):
        for s in range(len(scent_r)):
            k = (ab == a) & (sb_r == s)
            if k.sum() > 30:
                grid_r[s, a, j] = Yvr[k, j].mean()
g_r = grid_r[..., sig_r] / np.nanmean(grid_r[..., sig_r], (0, 1))
pg_r = np.nanmean(g_r, 2)
print("normalised rate toward PD: %.2f (slow) -> %.2f (fast); away from PD: %.2f -> %.2f"
      % (pg_r[0, ipd], pg_r[-1, ipd], pg_r[0, iap], pg_r[-1, iap]))

r2dec_r, pred_r = decode_r2(Ysm_r[mov_r], Vmr, blocks_r[mov_r], return_pred=True)
derr_r = np.degrees(np.angle(np.exp(1j * (np.arctan2(pred_r[:, 1], pred_r[:, 0]) - th_r))))
print("MC_RTT ridge decode on movement bins: R^2 %.3f, median |direction error| %.1f deg"
      % (r2dec_r, np.median(np.abs(derr_r))))

# %% [markdown]
# ### Figure 7: replication

# %%
fig, ax = plt.subplots(1, 4, figsize=(17, 4.3), gridspec_kw=dict(wspace=0.36))
ax[0].plot(lags * 1000, pop_r, "o-", color="crimson")
ax[0].axvline(BEST_R * 1000, color="crimson", ls="--")
ax[0].axvline(0, color="0.6", lw=1)
ax[0].set(xlabel="lag (ms): positive = neural leads hand", ylabel="cross-validated $R^2$",
          title="Motor cortex leads the hand\nby %+.0f ms" % (BEST_R * 1000))
ax[1].hist(np.degrees(pd_r[sig_r]), bins=np.linspace(-180, 180, 25), color="seagreen", edgecolor="w")
ax[1].set(xlabel="preferred direction (deg)", ylabel="units", xticks=[-180, -90, 0, 90, 180],
          title="Preferred directions\n%d/%d units velocity-tuned" % (sig_r.sum(), len(sig_r)))
for i, lab, c in [(ipd, "toward PD", "crimson"), (iorth, "orthogonal", "0.4"),
                  (iap, "away from PD", "steelblue")]:
    mu = np.nanmean(g_r[:, i, :], 1)
    se = np.nanstd(g_r[:, i, :], 1) / np.sqrt(sig_r.sum())
    ax[2].errorbar(scent_r, mu, yerr=se, marker="o", lw=2, color=c, label=lab, capsize=3)
ax[2].axhline(1, color="k", ls=":", lw=0.8)
ax[2].legend(fontsize=8)
ax[2].set(xlabel="finger speed (mm/s)", ylabel="rate / mean rate",
          title="Speed scales the directional signal\n(replicates MC_Maze)")
ax[3].hist(derr_r, bins=np.linspace(-180, 180, 37), color="seagreen", edgecolor="w")
ax[3].set(xlabel="decoded minus actual direction (deg)", ylabel="bins", xticks=[-180, -90, 0, 90, 180],
          title="Velocity decoding\n$R^2$ = %.2f, median error %.0f$\\degree$"
                % (r2dec_r, np.median(np.abs(derr_r))))
fig.suptitle("Replication in an independent dataset: DANDI:000129 MC_RTT, monkey Indy, %d M1 units, "
             "self-paced random-target reaching" % len(units_r), y=1.06, fontsize=13)
fig.savefig("fig07_rtt_replication.png", dpi=140, bbox_inches="tight")
print("saved fig07_rtt_replication.png")

# %% [markdown]
# ## Summary
#
# Across two independent datasets, two monkeys and two reaching tasks, motor-cortical units are
# tuned to the direction of the reach, a cosine describes that tuning well, and the strength of
# the directional signal scales with hand speed. The last point is what distinguishes velocity
# tuning from direction tuning, and it is visible three separate ways: in the speed-by-relative-
# direction grids, in the two-dimensional velocity rate maps, and in the roughly doubled
# cross-validated pseudo-$R^2$ of a general velocity GLM over a direction-only GLM. Both the
# single-unit models and the population decoders are best when neural activity is advanced by
# about 100 ms relative to the hand, consistent with motor cortex driving the movement.
#
# **Limitations.** The unit-to-array mapping in the MC_Maze file is internally inconsistent, so
# no M1-versus-PMd comparison was attempted. Direction sampling in MC_Maze is uneven, which leaves
# two of twelve 30° bins too sparse to use and probably contributes to the non-uniform
# distribution of preferred directions. The GLMs model instantaneous kinematics only and include
# no spike-history term, so a unit's own refractoriness and bursting are not accounted for.
# Finally, all of this is correlational: tuning to velocity in this sense is a statement about
# what firing rate predicts, not a claim that velocity is the variable the cortex explicitly
# represents.
