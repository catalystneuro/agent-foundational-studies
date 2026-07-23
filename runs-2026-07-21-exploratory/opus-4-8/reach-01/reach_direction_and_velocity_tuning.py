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
# **Primary dataset:** [DANDI:000128](https://dandiarchive.org/dandiset/000128), *MC_Maze: macaque primary
# motor and dorsal premotor cortex spiking activity during delayed reaching*
# (Churchland & Kaufman, released as part of the Neural Latents Benchmark).
# Monkey Jenkins, session 2009-09-25, 182 sorted units recorded from Utah arrays in M1 and PMd,
# 2295 successful trials of a delayed center-out maze task. Cursor position, hand position and hand
# velocity were recorded at 1 kHz.
#
# The task is a delayed reaching task. A target appears, the monkey holds through a variable delay,
# a go cue is given, and the monkey reaches. On roughly a third of the trials the workspace is
# empty and the reach is straight; on the remaining trials virtual barriers force a curved path to
# the same target. That mix is useful here, because it partially decorrelates *where the target is*
# from *which way the hand is actually moving* at any instant.
#
# **What this notebook demonstrates**
#
# 1. Single units in M1/PMd are tuned for the direction of the upcoming and ongoing reach, and the
#    tuning is well described by a cosine (Georgopoulos et al., 1982).
# 2. The same units are tuned for the direction of the *instantaneous hand velocity* throughout the
#    reach, not only for the direction of the target, and firing rate also grows with hand speed
#    (Moran & Schwartz, 1999).
# 3. The neural signal leads the kinematics: direction tuning measured on 20 ms bins is strongest
#    when spikes are paired with the velocity roughly 100 ms in the future.
# 4. A Poisson GLM (NeMoS) shows that a direction x speed (i.e. full velocity) model explains about
#    twice as much held-out deviance as direction alone.
# 5. Hand velocity can be linearly decoded from the population moment by moment
#    (cross-validated R^2 ~ 0.55), and the discrete reach direction can be decoded from delay-period
#    activity alone, well before the movement starts.
# 6. All of the velocity results replicate in a second dataset, [DANDI:000129](
#    https://dandiarchive.org/dandiset/000129) (MC_RTT, monkey Indy, self-paced random-target
#    reaching). That dataset also resolves a confound the center-out task cannot: because its
#    targets appear at random locations, hand position carries no information about hand velocity,
#    and the position encoding model that looked competitive in MC_Maze collapses.
#
# Both NWB files are streamed from the DANDI S3 bucket through LINDI; nothing is downloaded in
# full. Expect roughly 30-45 minutes end to end, most of it in the two GLM sections.

# %%
import os
import pickle

import matplotlib
matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from tqdm.auto import tqdm

import lindi
import nemos as nmo
import pynapple as nap
from pynwb import NWBHDF5IO
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import jax
jax.config.update("jax_enable_x64", True)

CACHE = "./cache"
os.makedirs(CACHE, exist_ok=True)
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Streaming the session
#
# LINDI serves a JSON index of the HDF5 chunk layout, so `pynwb` can read the 691 MB NWB asset
# lazily over HTTP with a local chunk cache.

# %%
LINDI_URL = ("https://lindi.neurosift.org/dandi/dandisets/000128/assets/"
             "26e85f09-39b7-480f-b337-278a8f034007/nwb.lindi.json")

f = lindi.LindiH5pyFile.from_lindi_file(
    LINDI_URL, local_cache=lindi.LocalCache(cache_dir=f"{CACHE}/lindi"))
nwbfile = NWBHDF5IO(file=f, mode="r").read()

print(nwbfile.session_description)
print("subject:", nwbfile.subject.subject_id, nwbfile.subject.species)
print("behaviour streams:", list(nwbfile.processing["behavior"].data_interfaces))
print("units:", len(nwbfile.units), " trials:", len(nwbfile.trials))

# %% [markdown]
# Behaviour is sampled at 1 kHz but only *within* trials, so the timestamp vector has small gaps
# between trials. That matters later: any lag/convolution operation has to respect trial
# boundaries rather than treating the session as one uniform grid.
#
# `hand_pos` sits in a frame offset from the target coordinates by about 35 mm, while `cursor_pos`
# is in the same frame as the targets (median endpoint error 7 mm). Reach geometry is therefore
# computed from `cursor_pos`; velocity is taken from `hand_vel` (a constant offset does not affect
# the derivative). All stored values are in mm and mm/s.

# %%
beh = nwbfile.processing["behavior"].data_interfaces
npz = f"{CACHE}/behavior.npz"
if os.path.exists(npz):
    z = np.load(npz)
    t_beh, hand_vel_raw, cursor_raw = z["t"], z["hv"], z["cp"]
else:
    t_beh = np.asarray(beh["hand_pos"].timestamps[:])
    hand_vel_raw = np.asarray(beh["hand_vel"].data[:], dtype=np.float32)
    cursor_raw = np.asarray(beh["cursor_pos"].data[:], dtype=np.float32)
    np.savez_compressed(npz, t=t_beh, hv=hand_vel_raw, cp=cursor_raw)

hand_vel = nap.TsdFrame(t=t_beh, d=hand_vel_raw, columns=["vx", "vy"])
cursor = nap.TsdFrame(t=t_beh, d=cursor_raw, columns=["x", "y"])

units_df = nwbfile.units.to_dataframe()
spikes_all = nap.TsGroup({i: np.asarray(s) for i, s in enumerate(units_df["spike_times"])})
trials_raw = nwbfile.trials.to_dataframe()

print(hand_vel)
print("session duration: %.0f s, NaNs in kinematics: %d"
      % (t_beh[-1], np.isnan(hand_vel_raw).sum() + np.isnan(cursor_raw).sum()))

# %% [markdown]
# ## 2. Trial geometry
#
# For each trial the *reach direction* is the angle from the hand's position at movement onset to
# the active target. Trials are also split into barrier-free (straight) and maze (curved) reaches.

# %%
N_DIR_BINS = 8
DIR_EDGES = np.linspace(-np.pi, np.pi, N_DIR_BINS + 1)
DIR_CENTERS = 0.5 * (DIR_EDGES[:-1] + DIR_EDGES[1:])

tgt = np.array([np.asarray(p)[a] for p, a in
                zip(trials_raw["target_pos"], trials_raw["active_target"])], dtype=float)
i_mo = np.searchsorted(cursor.index, trials_raw["move_onset_time"].values)
p0 = cursor.values[i_mo].astype(float)
disp = tgt - p0

trials = trials_raw.copy()
trials["reach_angle"] = np.arctan2(disp[:, 1], disp[:, 0])
trials["reach_dist"] = np.hypot(disp[:, 0], disp[:, 1])
trials["dir_bin"] = np.clip(np.digitize(trials["reach_angle"], DIR_EDGES) - 1, 0, N_DIR_BINS - 1)
trials["is_straight"] = trials["num_barriers"] == 0

print(trials[["reach_angle", "reach_dist", "dir_bin", "is_straight"]].describe().T)
print("straight / curved trials:", trials["is_straight"].value_counts().to_dict())
print("occupied direction bins:", sorted(trials["dir_bin"].unique()))

# %% [markdown]
# Only 7 of the 8 direction bins are occupied (the maze conditions leave a gap in the
# lower-right quadrant), so all direction analyses below use the 7 populated bins, and the cosine
# fits use the continuous angle rather than the bins.

# %% [markdown]
# ## 3. Raw data
#
# Population raster with the simultaneous kinematics, units sorted by the preferred velocity
# direction computed later in the notebook (sorting is applied at plot time).

# %%
UNITS_MIN_RATE = 1.0  # Hz; drop very sparse units, they carry no usable tuning estimate
keep = np.array([u for u in spikes_all.index if spikes_all[u].rate > UNITS_MIN_RATE])
spikes = spikes_all[list(keep)]
print("kept %d of %d units above %.1f Hz" % (len(keep), len(spikes_all), UNITS_MIN_RATE))


def plot_raw_session(unit_order, unit_colors, first_trial=8, n_trials=8, fname="fig00_raw_session.png"):
    sl = trials.iloc[first_trial:first_trial + n_trials]
    ep = nap.IntervalSet(start=sl["start_time"].iloc[0], end=sl["stop_time"].iloc[-1])
    fig, ax = plt.subplots(4, 1, figsize=(14, 9.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1, 1, 1], hspace=.12, top=0.93))
    for i, u in enumerate(unit_order):
        s = np.asarray(spikes[u].restrict(ep).index)
        ax[0].plot(s, np.full_like(s, i), "|", color=unit_colors[i], ms=2.8, mew=.7)
    ax[0].set(ylabel="unit (sorted by preferred\nvelocity direction)")
    c = cursor.restrict(ep)
    ax[1].plot(c.index, c.values[:, 0], label="x")
    ax[1].plot(c.index, c.values[:, 1], label="y")
    ax[1].set(ylabel="cursor\nposition (mm)")
    ax[1].legend(ncol=2, fontsize=8, loc="upper right")
    v = hand_vel.restrict(ep)
    ax[2].plot(v.index, v.values[:, 0], label="$v_x$")
    ax[2].plot(v.index, v.values[:, 1], label="$v_y$")
    ax[2].set(ylabel="hand\nvelocity (mm/s)")
    ax[2].legend(ncol=2, fontsize=8, loc="upper right")
    ax[3].plot(v.index, np.hypot(*v.values.T), "k")
    ax[3].set(ylabel="speed\n(mm/s)", xlabel="time (s)")
    for a in ax:
        for _, r in sl.iterrows():
            a.axvline(r["target_on_time"], color="tab:green", lw=.8, alpha=.7)
            a.axvline(r["go_cue_time"], color="tab:orange", lw=.8, alpha=.7)
            a.axvline(r["move_onset_time"], color="tab:red", lw=.9, alpha=.8)
        a.set_xlim(ep.start[0], ep.end[0])
    h = [plt.Line2D([], [], color=c_, label=l) for c_, l in
         [("tab:green", "target on"), ("tab:orange", "go cue"), ("tab:red", "move onset")]]
    fig.suptitle("DANDI:000128 MC_Maze, monkey Jenkins - M1/PMd population activity and hand "
                 "kinematics (%d consecutive trials)" % n_trials, y=0.985)
    fig.legend(handles=h, ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 0.962),
               frameon=False)
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)


# %% [markdown]
# ## 4. Behaviour
#
# Straight reaches fan out from the centre; maze reaches curve around the barriers to reach the
# same targets. Speed profiles are the expected bell shape peaking ~140 ms after movement onset.

# %%
def perievent_continuous(tsd, events, window=(-0.4, 0.8), dt=0.001):
    """Align a within-trial regularly-sampled series to events.

    Behaviour is 1 kHz within trials with gaps between them, so pynapple's `compute_perievent`
    (which requires one global uniform grid) does not apply. NaN marks missing samples.
    """
    lags = np.arange(round(window[0] / dt), round(window[1] / dt) + 1) * dt
    t = np.asarray(tsd.index)
    v = np.atleast_2d(np.asarray(tsd.values).T).T
    out = np.full((len(lags), len(events), v.shape[1]), np.nan)
    for j, e in enumerate(events):
        idx = np.searchsorted(t, e + lags)
        ok = (idx < len(t)) & (np.abs(t[np.clip(idx, 0, len(t) - 1)] - (e + lags)) < dt)
        out[ok, j] = v[idx[ok]]
    return lags, out


hsv = plt.get_cmap("hsv")
dir_colors = [hsv((c % (2 * np.pi)) / (2 * np.pi)) for c in DIR_CENTERS]
speed = nap.Tsd(t=hand_vel.index, d=np.hypot(*hand_vel.values.T))

fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
for ax, straight, ttl in [(axes[0], True, "Barrier-free reaches (straight)"),
                          (axes[1], False, "Maze reaches (curved)")]:
    sub = trials[trials["is_straight"] == straight]
    for _, row in sub.sample(min(240, len(sub)), random_state=0).iterrows():
        seg = cursor.get(row["move_onset_time"], row["move_onset_time"] + 0.8).values
        ax.plot(seg[:, 0], seg[:, 1], lw=0.6, alpha=0.55, color=dir_colors[int(row["dir_bin"])])
    ax.set(title=f"{ttl}  (n={len(sub)})", xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
    ax.set_xlim(-160, 160)
    ax.set_ylim(-160, 160)

lags, pe = perievent_continuous(speed, trials["move_onset_time"].values, window=(-0.4, 0.8))
pe = pe[:, :, 0]
axes[2].plot(lags, np.nanmedian(pe, 1), "k", lw=2)
axes[2].fill_between(lags, *np.nanpercentile(pe, [25, 75], axis=1), alpha=.25, color="k")
axes[2].axvline(0, color="r", ls="--", label="move onset")
axes[2].set(xlabel="time from move onset (s)", ylabel="hand speed (mm/s)",
            title="Speed profile (median $\\pm$ IQR)")
axes[2].legend()

ax = plt.subplot(1, 4, 4, projection="polar")
axes[3].remove()
rose_edges = np.linspace(-np.pi, np.pi, 33)
cnt, _ = np.histogram(trials["reach_angle"], bins=rose_edges)
ax.bar(rose_edges[:-1] + np.pi / 32, cnt, width=2 * np.pi / 32,
       color=[hsv(((a + np.pi / 32) % (2 * np.pi)) / (2 * np.pi)) for a in rose_edges[:-1]])
ax.set_title("Reach direction distribution\n(n=%d trials)" % len(trials), pad=22)
ax.set_rlabel_position(255)
ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig("fig01_behavior_overview.png", dpi=140)
plt.close(fig)

# %% [markdown]
# ## 5. Reach-direction tuning
#
# Per-trial firing rates in four windows, then an ordinary least-squares cosine fit
# $r = b_0 + b_1\cos\theta + b_2\sin\theta$ per unit. The preferred direction is
# $\arctan2(b_2, b_1)$, the modulation depth is $\sqrt{b_1^2+b_2^2}$, and the two cosine terms are
# tested jointly with an F test. The pre-target baseline window acts as a negative control.

# %%
def trial_rates(spike_group, tr, ref, t0, t1):
    a, b = tr[ref].values + t0, tr[ref].values + t1
    counts = np.stack([np.searchsorted(np.asarray(spike_group[u].index), b)
                       - np.searchsorted(np.asarray(spike_group[u].index), a)
                       for u in spike_group.index], axis=1)
    return counts / (t1 - t0)


def cosine_fit(rates, angles):
    X = np.column_stack([np.ones_like(angles), np.cos(angles), np.sin(angles)])
    beta, *_ = np.linalg.lstsq(X, rates, rcond=None)
    ss_res = ((rates - X @ beta) ** 2).sum(0)
    ss_tot = ((rates - rates.mean(0)) ** 2).sum(0)
    n = len(angles)
    fstat = (ss_tot - ss_res) / 2 / (ss_res / (n - 3))
    return pd.DataFrame(dict(b0=beta[0], pd=np.arctan2(beta[2], beta[1]),
                             mod_depth=np.hypot(beta[1], beta[2]),
                             r2=1 - ss_res / ss_tot, f=fstat, p=stats.f.sf(fstat, 2, n - 3)))


EPOCHS = {"baseline": ("target_on_time", -0.25, -0.05),
          "preparatory": ("go_cue_time", -0.30, 0.0),
          "perimovement": ("move_onset_time", -0.05, 0.25),
          "late_movement": ("move_onset_time", 0.25, 0.55)}

epoch_rates, dir_fits = {}, {}
for name, (ref, t0, t1) in EPOCHS.items():
    R = trial_rates(spikes, trials, ref, t0, t1)
    fit = cosine_fit(R, trials["reach_angle"].values)
    fit["epoch"] = name
    epoch_rates[name], dir_fits[name] = R, fit
    print(f"{name:14s} tuned (p<0.01): {(fit['p'] < 0.01).sum():3d}/{len(fit)}   "
          f"median R2={fit['r2'].median():.3f}   median depth={fit['mod_depth'].median():.2f} Hz")
all_fits = pd.concat(dir_fits.values())

# %% [markdown]
# Straight and curved reaches to the same targets, fit separately. If the cells encoded the target
# rather than the movement, the two fits would agree.

# %%
m_straight = trials["is_straight"].values
fit_straight = cosine_fit(trial_rates(spikes, trials[m_straight], "move_onset_time", -0.05, 0.25),
                          trials["reach_angle"].values[m_straight])
fit_curved = cosine_fit(trial_rates(spikes, trials[~m_straight], "move_onset_time", -0.05, 0.25),
                        trials["reach_angle"].values[~m_straight])
for nm, ft in [("straight", fit_straight), ("curved", fit_curved)]:
    print(f"{nm:9s} only: tuned {(ft['p'] < 0.01).sum():3d}/{len(ft)}  "
          f"median R2={ft['r2'].median():.3f}  median depth={ft['mod_depth'].median():.2f} Hz")

# %% [markdown]
# ### Example units

# %%
def psth(spike_times, events, window=(-0.6, 0.8), bin_size=0.02, sigma=0.03):
    st = np.asarray(spike_times)
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    counts = np.zeros(len(edges) - 1)
    per_trial = []
    for e in events:
        i0, i1 = np.searchsorted(st, [e + window[0], e + window[1]])
        rel = st[i0:i1] - e
        per_trial.append(rel)
        counts += np.histogram(rel, edges)[0]
    rate = counts / (len(events) * bin_size)
    if sigma:
        k = np.exp(-0.5 * (np.arange(-4 * sigma, 4 * sigma + bin_size, bin_size) / sigma) ** 2)
        rate = np.convolve(rate, k / k.sum(), mode="same")
    return 0.5 * (edges[:-1] + edges[1:]), rate, per_trial


fit_s = fit_straight.copy()
fit_s.index = keep
straight_trials = trials[m_straight]
bins_present = sorted(straight_trials["dir_bin"].unique())

# three well-tuned units with distinct preferred directions
examples, used_pd = [], []
for u, row in fit_s.sort_values("r2", ascending=False).iterrows():
    if all(abs(np.angle(np.exp(1j * (row["pd"] - p)))) > np.pi / 4 for p in used_pd):
        examples.append(u)
        used_pd.append(row["pd"])
    if len(examples) == 3:
        break
print("example units:", examples, "PDs:", np.round(np.degrees(used_pd)))

fig = plt.figure(figsize=(15, 4.4 * len(examples) + 0.8))
gs = fig.add_gridspec(len(examples), 3, width_ratios=[1.5, 1.2, 1.0], hspace=.5, wspace=.32,
                      top=0.90, bottom=0.06)
handles = []
for r, u in enumerate(examples):
    st = np.asarray(spikes[u].index)
    ax_r = fig.add_subplot(gs[r, 0])
    ax_p = fig.add_subplot(gs[r, 1])
    ax_t = fig.add_subplot(gs[r, 2], projection="polar")
    y = 0
    for b in bins_present:
        ev = straight_trials.loc[straight_trials["dir_bin"] == b, "move_onset_time"].values
        col = dir_colors[b]
        _, _, per = psth(st, ev[:40], window=(-0.6, 0.8), sigma=0)
        for rel in per:
            ax_r.plot(rel, np.full_like(rel, y), "|", color=col, ms=3, mew=.7)
            y += 1
        tt, rate, _ = psth(st, ev, window=(-0.9, 1.1))   # wide, then trim edge artifacts
        vis = (tt >= -0.6) & (tt <= 0.8)
        ln, = ax_p.plot(tt[vis], rate[vis], color=col, lw=1.8,
                        label=f"{np.degrees(DIR_CENTERS[b]):.0f}$\\degree$")
        if r == 0:
            handles.append(ln)
        y += 4
    for a in (ax_r, ax_p):
        a.axvline(0, color="k", ls="--", lw=1)
        a.set_xlabel("time from move onset (s)")
        a.set_xlim(-0.6, 0.8)
    ax_r.set(ylabel="trials (grouped by direction)", title=f"unit {u}: raster", yticks=[])
    ax_p.set(ylabel="firing rate (Hz)", title=f"unit {u}: direction-conditioned PSTH")
    R = trial_rates(spikes[[u]], straight_trials, "move_onset_time", -0.05, 0.25)[:, 0]
    db = straight_trials["dir_bin"].values
    mu = np.array([R[db == b].mean() for b in bins_present])
    se = np.array([R[db == b].std() / np.sqrt((db == b).sum()) for b in bins_present])
    th = DIR_CENTERS[bins_present]
    ax_t.errorbar(np.r_[th, th[0]], np.r_[mu, mu[0]], yerr=np.r_[se, se[0]],
                  color="k", marker="o", ms=4, lw=1.5, capsize=2)
    ft = fit_s.loc[u]
    g = np.linspace(-np.pi, np.pi, 200)
    ax_t.plot(g, np.clip(ft["b0"] + ft["mod_depth"] * np.cos(g - ft["pd"]), 0, None), "r-", lw=2)
    ax_t.set_title(f"unit {u}: cosine fit\nPD={np.degrees(ft['pd']):.0f}$\\degree$, "
                   f"MD={ft['mod_depth']:.1f} Hz, $R^2$={ft['r2']:.2f}", pad=26, fontsize=10)
    ax_t.set_rlabel_position(200)
    ax_t.tick_params(labelsize=8)
fig.legend(handles=handles, ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, 0.975),
           title="reach direction (target angle)", fontsize=9, title_fontsize=10)
fig.suptitle("Reach-direction tuning in macaque M1/PMd (MC_Maze, barrier-free trials)",
             y=0.995, fontsize=13)
plt.savefig("fig02_example_units_direction.png", dpi=140)
plt.close(fig)

# %% [markdown]
# ## 6. Velocity tuning
#
# The trial-level analysis collapses each reach to a single direction. To ask about velocity we
# need the moment-by-moment relationship, so spikes and kinematics are binned at 20 ms inside the
# trial intervals.

# %%
BIN = 0.02
trial_ep = nap.IntervalSet(start=trials["start_time"].values, end=trials["stop_time"].values)
counts_tsd = spikes.count(BIN, ep=trial_ep)
vel_binned = hand_vel.bin_average(BIN, ep=trial_ep)
pos_binned = cursor.bin_average(BIN, ep=trial_ep)

M = dict(t=counts_tsd.index.values,
         counts=np.asarray(counts_tsd.values),
         vel=np.asarray(vel_binned.values),
         pos=np.asarray(pos_binned.values),
         trial=np.asarray(trial_ep.in_interval(nap.Ts(counts_tsd.index.values))).astype(int),
         bin_size=BIN)
M["speed"] = np.hypot(M["vel"][:, 0], M["vel"][:, 1])
M["vel_angle"] = np.arctan2(M["vel"][:, 1], M["vel"][:, 0])
print("binned matrix:", M["counts"].shape, " moving bins (>100 mm/s):", (M["speed"] > 100).sum())


def shift_by_lag(M, lag_bins):
    """Pair spike counts at bin i with kinematics at bin i+lag_bins, never crossing a trial.

    Positive lag_bins means the neural activity LEADS the kinematics it is paired with.
    """
    n = len(M["t"])
    src = np.arange(n) + lag_bins
    ok = (src >= 0) & (src < n)
    ok[ok] &= M["trial"][src[ok]] == M["trial"][np.arange(n)[ok]]
    return np.arange(n)[ok], src[ok]


def cosine_fit_counts(counts, angle, bin_size):
    return cosine_fit(counts / bin_size, angle)


# %% [markdown]
# ### How far ahead does the neural signal sit?
#
# Sweeping the pairing lag and re-fitting the cosine at each one gives the lead time at which
# direction tuning is strongest.

# %%
LAGS = np.arange(-0.40, 0.52, 0.02)
md_by_lag, r2_by_lag = [], []
for lag in tqdm(LAGS, desc="lag sweep"):
    i_n, i_k = shift_by_lag(M, int(round(lag / BIN)))
    mv = M["speed"][i_k] > 100
    ft = cosine_fit_counts(M["counts"][i_n][mv], M["vel_angle"][i_k][mv], BIN)
    md_by_lag.append(ft["mod_depth"].values)
    r2_by_lag.append(ft["r2"].values)
md_by_lag = np.array(md_by_lag)
BEST_LAG = LAGS[int(np.argmax(np.median(md_by_lag, 1)))]
per_unit_lag = LAGS[np.argmax(md_by_lag, axis=0)]
strong = md_by_lag.max(0) > np.median(md_by_lag.max(0))
print("population-median tuning peaks at a neural lead of %d ms" % round(BEST_LAG * 1000))
print("per-unit optimal lead (well-modulated half): median %d ms, IQR %s"
      % (round(np.median(per_unit_lag[strong]) * 1000),
         np.round(np.percentile(per_unit_lag[strong], [25, 75]) * 1000)))

# %%
i_n, i_k = shift_by_lag(M, int(round(BEST_LAG / BIN)))
moving = M["speed"][i_k] > 100
C = M["counts"][i_n][moving]
A = M["vel_angle"][i_k][moving]
S = M["speed"][i_k][moving]
vel_fit = cosine_fit_counts(C, A, BIN)
print("velocity-direction tuning at %d ms lead: %d/%d units p<0.01, median depth %.2f Hz"
      % (BEST_LAG * 1000, (vel_fit["p"] < 0.01).sum(), len(vel_fit),
         vel_fit["mod_depth"].median()))


def tuning_2d(counts, vel, bin_size, nbins=13, vmax=600.0, min_occ=25):
    edges = np.linspace(-vmax, vmax, nbins + 1)
    ix = np.digitize(vel[:, 0], edges) - 1
    iy = np.digitize(vel[:, 1], edges) - 1
    ok = (ix >= 0) & (ix < nbins) & (iy >= 0) & (iy < nbins)
    occ = np.zeros((nbins, nbins))
    np.add.at(occ, (iy[ok], ix[ok]), 1)
    maps = np.full((counts.shape[1], nbins, nbins), np.nan)
    for u in range(counts.shape[1]):
        s = np.zeros((nbins, nbins))
        np.add.at(s, (iy[ok], ix[ok]), counts[ok, u])
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / occ / bin_size
        m[occ < min_occ] = np.nan
        maps[u] = m
    return edges, maps


def speed_tuning(counts, spd, angle, pref_dir, bin_size, nbins=10, halfwidth=np.pi / 4,
                 rest_thresh=50.0):
    """Rate vs speed for movement in each unit's preferred direction; slow bins (direction
    ill-defined) are pooled into the first, near-zero-speed point."""
    qs = np.linspace(rest_thresh, np.percentile(spd, 99), nbins)
    ctrs = np.r_[0.5 * rest_thresh, 0.5 * (qs[:-1] + qs[1:])]
    idx = np.digitize(spd, qs)
    out = np.full((counts.shape[1], nbins), np.nan)
    for u in range(counts.shape[1]):
        near = (np.abs(np.angle(np.exp(1j * (angle - pref_dir[u])))) < halfwidth) | (spd < rest_thresh)
        for b in range(nbins):
            m = near & (idx == b)
            if m.sum() > 30:
                out[u, b] = counts[m, u].mean() / bin_size
    return ctrs, out


vel_edges, vel_maps = tuning_2d(M["counts"][i_n], M["vel"][i_k], BIN)
speed_ctr, speed_tun = speed_tuning(M["counts"][i_n], M["speed"][i_k], M["vel_angle"][i_k],
                                    vel_fit["pd"].values, BIN)
print("population mean rate in the PD: %.1f Hz at rest -> %.1f Hz at the fastest speed bin"
      % (np.nanmean(speed_tun[:, 0]), np.nanmean(speed_tun[:, -1])))

# %%
nb = 18
vedges = np.linspace(-np.pi, np.pi, nb + 1)
vctr = 0.5 * (vedges[:-1] + vedges[1:])
ib = np.clip(np.digitize(A, vedges) - 1, 0, nb - 1)
tc = np.stack([[C[ib == b, u].mean() / BIN for b in range(nb)] for u in range(C.shape[1])])
tc_se = np.stack([[C[ib == b, u].std() / np.sqrt((ib == b).sum()) / BIN for b in range(nb)]
                  for u in range(C.shape[1])])

order = np.argsort(-vel_fit["r2"].values)[:4]
fig = plt.figure(figsize=(16, 10.5))
gs = fig.add_gridspec(3, 4, hspace=.75, wspace=.42)
for j, u in enumerate(order):
    ax = fig.add_subplot(gs[0, j], projection="polar")
    ax.errorbar(np.r_[vctr, vctr[0]], np.r_[tc[u], tc[u][0]], yerr=np.r_[tc_se[u], tc_se[u][0]],
                color="k", lw=1.5, marker=".", ms=4)
    g = np.linspace(-np.pi, np.pi, 200)
    ax.plot(g, np.clip(vel_fit["b0"][u] + vel_fit["mod_depth"][u] * np.cos(g - vel_fit["pd"][u]),
                       0, None), "r", lw=2)
    ax.set_title(f"unit {keep[u]}\nPD={np.degrees(vel_fit['pd'][u]):.0f}$\\degree$, "
                 f"MD={vel_fit['mod_depth'][u]:.1f} Hz", pad=24, fontsize=10)
    ax.set_rlabel_position(np.degrees(vel_fit["pd"][u]) + 150)
    ax.tick_params(labelsize=7)
fig.text(0.5, 0.975, "Firing rate vs. instantaneous hand-velocity direction "
         f"(20 ms bins, speed > 100 mm/s, neural lead {BEST_LAG * 1000:.0f} ms)",
         ha="center", fontsize=12)

ext = [vel_edges[0], vel_edges[-1], vel_edges[0], vel_edges[-1]]
for j, u in enumerate(order):
    ax = fig.add_subplot(gs[1, j])
    im = ax.imshow(vel_maps[u], origin="lower", extent=ext, cmap="viridis")
    ax.arrow(0, 0, 380 * np.cos(vel_fit["pd"][u]), 380 * np.sin(vel_fit["pd"][u]), color="w",
             width=14, head_width=48, length_includes_head=True)
    ax.set(xlabel="$v_x$ (mm/s)", ylabel="$v_y$ (mm/s)", title=f"unit {keep[u]}: 2-D velocity map")
    plt.colorbar(im, ax=ax, label="rate (Hz)", fraction=.046)

ax = fig.add_subplot(gs[2, 0])
ax.plot(LAGS * 1000, np.median(md_by_lag, 1), "k-o", ms=3)
ax.axvline(BEST_LAG * 1000, color="r", ls="--", label=f"peak {BEST_LAG * 1000:.0f} ms")
ax.set(xlabel="neural lead time (ms)", ylabel="median modulation depth (Hz)",
       title="Direction tuning vs. neural lead")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 1])
ax.hist(per_unit_lag[strong] * 1000, bins=np.arange(-410, 530, 40), color="0.4")
ax.axvline(np.median(per_unit_lag[strong]) * 1000, color="r", ls="--",
           label=f"median {np.median(per_unit_lag[strong]) * 1000:.0f} ms")
ax.set(xlabel="per-unit optimal lead (ms)", ylabel="units",
       title="Optimal lead (well-modulated units)")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 2])
norm = speed_tun / np.nanmax(speed_tun, 1, keepdims=True)
for r in norm[np.argsort(-vel_fit["r2"].values)[:40]]:
    ax.plot(speed_ctr, r, color="0.75", lw=.7)
ax.plot(speed_ctr, np.nanmean(norm, 0), "r-o", lw=2.2, ms=4, label="population mean")
ax.set(xlabel="hand speed (mm/s)", ylabel="normalized rate",
       title="Speed tuning within each unit's PD ($\\pm45\\degree$)")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 3], projection="polar")
sig = vel_fit["p"].values < 0.01
ax.hist(vel_fit["pd"].values[sig], bins=np.linspace(-np.pi, np.pi, 25), color="steelblue")
ax.set_title(f"Preferred velocity directions\n({sig.sum()}/{len(vel_fit)} units, p<0.01)",
             pad=24, fontsize=10)
ax.set_rlabel_position(255)
ax.tick_params(labelsize=8)
plt.savefig("fig03_velocity_tuning.png", dpi=135, bbox_inches="tight")
plt.close(fig)

# now that preferred velocity directions are known, draw the raw-data figure with units sorted
pd_order = np.argsort(vel_fit["pd"].values)
plot_raw_session(keep[pd_order],
                 [hsv((p % (2 * np.pi)) / (2 * np.pi)) for p in vel_fit["pd"].values[pd_order]])

# %% [markdown]
# ## 7. Population summary of direction tuning

# %%
def decode_direction(rates, labels, groups, n_folds=5):
    pred = np.empty(len(labels), dtype=int)
    for tr_i, te_i in GroupKFold(n_splits=n_folds).split(rates, labels, groups=groups):
        sc = StandardScaler().fit(rates[tr_i])
        m = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(rates[tr_i]), labels[tr_i])
        pred[te_i] = m.predict(sc.transform(rates[te_i]))
    return pred


labels = trials["dir_bin"].values
folds = np.arange(len(labels)) % 5
dir_acc = {}
for nm, key in [("baseline", "baseline"), ("delay", "preparatory"), ("movement", "perimovement")]:
    dir_acc[nm] = (decode_direction(epoch_rates[key], labels, folds) == labels).mean()
    print(f"{nm:10s} 7-way direction decoding accuracy {dir_acc[nm]:.3f} "
          f"(chance {1 / len(np.unique(labels)):.3f})")

# %%
ep_order = ["baseline", "preparatory", "perimovement", "late_movement"]
ep_lbl = ["baseline\n(pre-target)", "delay\n(-300-0 ms\nre go cue)", "peri-move\n(-50-250 ms\nre onset)",
          "late move\n(250-550 ms\nre onset)"]
ep_col = ["0.6", "tab:blue", "tab:red", "tab:orange"]

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.30)

ax = fig.add_subplot(gs[0, 0])
frac = [(dir_fits[e]["p"] < 0.01).mean() for e in ep_order]
ax.bar(range(4), frac, color=ep_col)
ax.axhline(0.01, color="k", ls=":", label="chance (p<0.01)")
ax.set(xticks=range(4), ylabel="fraction of units cosine-tuned", ylim=(0, 1.05),
       title="Directional tuning by task epoch")
ax.set_xticklabels(ep_lbl, fontsize=7.5)
ax.legend(fontsize=8)
for i, v in enumerate(frac):
    ax.text(i, v + .02, f"{v:.2f}", ha="center", fontsize=9)

ax = fig.add_subplot(gs[0, 1])
data = [dir_fits[e]["mod_depth"].values for e in ep_order]
bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=.6)
for p, c in zip(bp["boxes"], ep_col):
    p.set_facecolor(c)
for i, dd in enumerate(data):
    ax.plot(np.random.default_rng(i).normal(i + 1, .07, len(dd)), dd, ".", color="k", ms=2, alpha=.4)
ax.set(xticks=range(1, 5), ylabel="cosine modulation depth (Hz)", title="Depth of direction tuning")
ax.set_xticklabels(ep_lbl, fontsize=7.5)

ax = fig.add_subplot(gs[0, 2], projection="polar")
mv, pr = dir_fits["perimovement"], dir_fits["preparatory"]
ax.hist(mv["pd"].values[mv["p"].values < 0.01], bins=np.linspace(-np.pi, np.pi, 25),
        color="tab:red", alpha=.8, label="peri-movement")
ax.hist(pr["pd"].values[pr["p"].values < 0.01], bins=np.linspace(-np.pi, np.pi, 25),
        histtype="step", color="tab:blue", lw=2, label="preparatory")
ax.set_title("Preferred directions", pad=26)
ax.set_rlabel_position(250)
ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.25, -0.1))
ax.tick_params(labelsize=8)

ax = fig.add_subplot(gs[1, 0])
both = ((pr["p"].values < 0.01) & (mv["p"].values < 0.01)
        & (pr["mod_depth"].values > 0.5) & (mv["mod_depth"].values > 1.0))
a1, a2 = pr["pd"].values[both], mv["pd"].values[both]
m1, m2 = np.angle(np.exp(1j * a1).mean()), np.angle(np.exp(1j * a2).mean())
circ_r = (np.sum(np.sin(a1 - m1) * np.sin(a2 - m2))
          / np.sqrt(np.sum(np.sin(a1 - m1) ** 2) * np.sum(np.sin(a2 - m2) ** 2)))
dd = np.degrees(np.angle(np.exp(1j * (a2 - a1))))
ax.plot(np.degrees(a1), np.degrees(a2), "o", ms=4, color="k", alpha=.6)
ax.plot([-180, 180], [-180, 180], "r--", lw=1)
ax.set(xlabel="preferred direction, delay (deg)", ylabel="preferred direction, movement (deg)",
       xlim=(-185, 185), ylim=(-185, 185),
       title=f"Delay vs. movement PD (well-tuned units)\ncirc. r = {circ_r:.2f}, "
             f"median |shift| = {np.median(np.abs(dd)):.0f}$\\degree$, n={both.sum()}")

ax = fig.add_subplot(gs[1, 1])
both2 = (fit_straight["p"].values < 0.01) & (fit_curved["p"].values < 0.01)
d2 = np.degrees(np.angle(np.exp(1j * (fit_curved["pd"].values[both2]
                                      - fit_straight["pd"].values[both2]))))
ax.hist(d2, bins=np.arange(-180, 190, 15), color="0.4")
ax.axvline(0, color="r", ls="--")
ax.set(xlabel="PD(curved reaches) - PD(straight reaches) (deg)", ylabel="units",
       title="Target direction is not the whole story:\nPD shifts when the path is curved "
             f"(median |shift| {np.median(np.abs(d2)):.0f}$\\degree$)")

ax = fig.add_subplot(gs[1, 2])
nms = list(dir_acc)
ax.bar(range(len(nms)), [dir_acc[n] for n in nms], color=["0.6", "tab:blue", "tab:red"])
ax.axhline(1 / len(np.unique(labels)), color="k", ls=":", label="chance")
for i, n in enumerate(nms):
    ax.text(i, dir_acc[n] + .02, f"{dir_acc[n]:.2f}", ha="center")
ax.set(xticks=range(len(nms)), ylabel="7-way decoding accuracy", ylim=(0, 1),
       title="Reach direction decoded from\npopulation activity")
ax.set_xticklabels(nms, fontsize=9)
ax.legend(fontsize=8)
fig.suptitle(f"Population summary: reach-direction tuning in {len(keep)} M1/PMd units", y=0.98)
plt.savefig("fig04_population_direction.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 8. Poisson GLM encoding models (NeMoS)
#
# Nested comparison on 20 ms bins during movement (speed > 50 mm/s), with the neural activity
# paired to kinematics at the optimal lead. Each model is a `PopulationGLM` with a Ridge penalty,
# fitted with 5-fold cross-validation grouped by trial; the score is the fraction of Poisson
# deviance explained on held-out trials, relative to a constant-rate model.
#
# Note that the spline `Eval` bases are density-normalised, so their output scale goes as
# 1/range. Rescaling every input to [0, 1] first keeps the feature blocks comparable, which
# matters once a Ridge penalty is applied; without it the speed block is ~1000x smaller than the
# direction block and gets regularised away.

# %%
def deviance_explained(y, rate, rate_null):
    def ll(r):
        r = np.clip(r, 1e-9, None)
        return (y * np.log(r) - r - gammaln(y + 1)).sum(0)
    ll_sat = ll(np.clip(y, 1e-9, None))
    return 1 - (ll_sat - ll(rate)) / (ll_sat - ll(rate_null))


def unit_scale(v):
    lo, hi = np.min(v), np.max(v)
    return v if hi - lo == 0 else (v - lo) / (hi - lo)


def fit_cv(X, Y, groups, n_folds=5, reg=1e-3):
    ug = np.unique(groups)
    fold_of = {g: i % n_folds for i, g in enumerate(ug)}
    fold = np.array([fold_of[g] for g in groups])
    de = np.full((n_folds, Y.shape[1]), np.nan)
    for k in tqdm(range(n_folds), desc="CV folds", leave=False):
        tr_i, te_i = fold != k, fold == k
        m = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=reg,
                                  solver_name="LBFGS",
                                  solver_kwargs={"tol": 1e-7, "maxiter": 300})
        m.fit(X[tr_i], Y[tr_i])
        de[k] = deviance_explained(Y[te_i], np.asarray(m.predict(X[te_i])),
                                   np.tile(Y[tr_i].mean(0), (te_i.sum(), 1)))
    return de.mean(0)


glm_mask = M["speed"][i_k] > 50.0
Y_glm = M["counts"][i_n][glm_mask].astype(float)
glm_groups = M["trial"][i_n][glm_mask]
feat = dict(angle=M["vel_angle"][i_k][glm_mask],
            speed=unit_scale(M["speed"][i_k][glm_mask]),
            x=unit_scale(M["pos"][i_k][glm_mask, 0]),
            y=unit_scale(M["pos"][i_k][glm_mask, 1]))

b_ang = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, label="direction")
b_spd = nmo.basis.MSplineEval(n_basis_funcs=5, label="speed")
b_px = nmo.basis.BSplineEval(n_basis_funcs=5, label="pos_x")
b_py = nmo.basis.BSplineEval(n_basis_funcs=5, label="pos_y")
MODELS = {"speed only": (b_spd, ("speed",)),
          "direction only": (b_ang, ("angle",)),
          "direction x speed": (b_ang * b_spd, ("angle", "speed")),
          "position": (b_px * b_py, ("x", "y")),
          "dir x speed + position": (b_ang * b_spd + b_px * b_py, ("angle", "speed", "x", "y"))}

glm_scores = {}
for name, (basis, argnames) in MODELS.items():
    X = np.asarray(basis.compute_features(*[feat[a] for a in argnames]))
    glm_scores[name] = fit_cv(X, Y_glm, glm_groups)
    print(f"{name:24s} n_features={X.shape[1]:3d}  median held-out deviance explained = "
          f"{np.median(glm_scores[name]):.4f}")

# %% [markdown]
# ## 9. Decoding hand velocity from the population

# %%
def lagged_population(counts, trial, lags_bins):
    n, u = counts.shape
    out = np.full((n, u * len(lags_bins)), np.nan)
    idx = np.arange(n)
    for j, L in enumerate(lags_bins):
        src = idx - L
        ok = (src >= 0) & (src < n)
        ok[ok] &= trial[src[ok]] == trial[idx[ok]]
        out[np.ix_(ok, np.arange(u) + j * u)] = counts[src[ok]]
    return out


mo_of_bin = trials["move_onset_time"].values[M["trial"]]
peri = (M["t"] - mo_of_bin > -0.2) & (M["t"] - mo_of_bin < 0.8)
X_dec = lagged_population(M["counts"][peri].astype(np.float32), M["trial"][peri],
                          np.arange(0, 12, 2))   # 0-220 ms of neural history
good = ~np.isnan(X_dec).any(1)
X_dec, Y_dec, g_dec = X_dec[good], M["vel"][peri][good], M["trial"][peri][good]
pred = np.full_like(Y_dec, np.nan)
for tr_i, te_i in tqdm(list(GroupKFold(n_splits=5).split(X_dec, Y_dec, groups=g_dec)),
                       desc="decoding folds"):
    sc = StandardScaler().fit(X_dec[tr_i])
    m = RidgeCV(alphas=np.logspace(0, 5, 12)).fit(sc.transform(X_dec[tr_i]), Y_dec[tr_i])
    pred[te_i] = m.predict(sc.transform(X_dec[te_i]))
r2_vel = 1 - ((Y_dec - pred) ** 2).sum(0) / ((Y_dec - Y_dec.mean(0)) ** 2).sum(0)
ang_err = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(Y_dec[:, 1], Y_dec[:, 0])
                                                  - np.arctan2(pred[:, 1], pred[:, 0]))))))
fast = np.hypot(*Y_dec.T) > 100
print("velocity decoding: R2(vx)=%.3f, R2(vy)=%.3f, median direction error %.1f deg"
      % (r2_vel[0], r2_vel[1], np.median(ang_err[fast])))

# %%
fig = plt.figure(figsize=(16, 9.5))
gs = fig.add_gridspec(2, 3, hspace=.38, wspace=.30)

ax = fig.add_subplot(gs[0, 0])
names = list(glm_scores)
vals = [glm_scores[n] for n in names]
bp = ax.boxplot(vals, showfliers=False, patch_artist=True, widths=.6, vert=False)
for p, c in zip(bp["boxes"], ["0.7", "tab:blue", "tab:red", "tab:green", "tab:purple"]):
    p.set_facecolor(c)
for i, v in enumerate(vals):
    ax.plot(v, np.random.default_rng(i).normal(i + 1, .07, len(v)), ".", color="k", ms=2.5, alpha=.4)
ax.axvline(0, color="k", lw=.8)
ax.set(yticks=range(1, len(names) + 1), xlabel="held-out Poisson deviance explained",
       title=f"Poisson GLM encoding models (NeMoS)\n5-fold CV grouped by trial, {len(keep)} units")
ax.set_yticklabels(names, fontsize=9)

ax = fig.add_subplot(gs[0, 1])
ax.plot(glm_scores["direction only"], glm_scores["direction x speed"], "o", ms=5, color="k", alpha=.6)
lim = [0, max(glm_scores["direction x speed"].max(), glm_scores["direction only"].max()) * 1.05]
ax.plot(lim, lim, "r--")
ax.set(xlim=lim, ylim=lim, xlabel="direction only", ylabel="direction $\\times$ speed",
       title="Adding speed improves the fit for\n%d/%d units"
             % ((glm_scores["direction x speed"] > glm_scores["direction only"]).sum(), len(keep)))

ax = fig.add_subplot(gs[0, 2])
ax.plot(glm_scores["position"], glm_scores["direction x speed"], "o", ms=5, color="k", alpha=.6)
lim = [0, max(glm_scores["position"].max(), glm_scores["direction x speed"].max()) * 1.05]
ax.plot(lim, lim, "r--")
ax.set(xlim=lim, ylim=lim, xlabel="position model",
       ylabel="velocity model (direction $\\times$ speed)",
       title="Velocity vs. position as predictors\n(velocity better for %d/%d units)"
             % ((glm_scores["direction x speed"] > glm_scores["position"]).sum(), len(keep)))

ax = fig.add_subplot(gs[1, :2])
sel = np.unique(g_dec)[40:46]
mask = np.isin(g_dec, sel)
tt = np.arange(mask.sum()) * BIN
ax.plot(tt, Y_dec[mask, 0], "k", lw=1.6, label="actual $v_x$")
ax.plot(tt, pred[mask, 0], "tab:red", lw=1.4, label="decoded $v_x$")
ax.plot(tt, Y_dec[mask, 1] - 1500, "k", lw=1.6)
ax.plot(tt, pred[mask, 1] - 1500, "tab:blue", lw=1.4, label="decoded $v_y$")
for b in np.flatnonzero(np.diff(g_dec[mask]) != 0):
    ax.axvline(tt[b], color="0.8", lw=1)
ax.text(0.4, 400, "$v_x$", fontsize=11)
ax.text(0.4, -1100, "$v_y$", fontsize=11)
ax.set(xlabel="concatenated peri-movement time (s)", ylabel="velocity (mm/s), $v_y$ offset",
       title="Cross-validated linear decoding of hand velocity from %d units "
             "($R^2_{v_x}$=%.2f, $R^2_{v_y}$=%.2f)" % (len(keep), r2_vel[0], r2_vel[1]))
ax.legend(ncol=3, fontsize=9, loc="lower left")

ax = fig.add_subplot(gs[1, 2])
ax.hist(ang_err[fast], bins=np.arange(0, 185, 5), color="0.4")
ax.axvline(np.median(ang_err[fast]), color="r", ls="--",
           label="median %.0f$\\degree$" % np.median(ang_err[fast]))
ax.set(xlabel="decoded - actual movement direction (deg)", ylabel="20 ms bins",
       title="Instantaneous direction decoding error\n(bins with speed > 100 mm/s)")
ax.legend(fontsize=9)
plt.savefig("fig05_glm_and_decoding.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 10. Replication on a second dataset, and the position/velocity confound
#
# Everything above comes from one session of one monkey in one task, and the task has a specific
# weakness: in a center-out reach, where the hand *is* and where it is *going* are tightly coupled,
# because every reach starts at the centre and runs outward. That is why the position model scored
# nearly as well as the velocity model in section 8, and it means the model comparison on its own
# cannot say which variable the cells encode.
#
# [DANDI:000129](https://dandiarchive.org/dandiset/000129), *MC_RTT* (Makin, O'Doherty, Cardoso &
# Sabes; also released through the Neural Latents Benchmark), settles it. Monkey Indy makes
# self-paced reaches to targets that appear at random locations in the workspace, 130 sorted units
# from M1, 649 s of continuous recording with finger position and velocity at 1 kHz. Because
# successive targets are random, hand position carries essentially no information about hand
# velocity, so a position-only model has nothing to free-ride on.
#
# The same pipeline is run on this dataset: 20 ms bins, lag sweep, cosine fits, the NeMoS GLM
# comparison and linear velocity decoding.

# %%
RTT_LINDI = ("https://lindi.neurosift.org/dandi/dandisets/000129/assets/"
             "2ae6bf3c-788b-4ece-8c01-4b4a5680b25b/nwb.lindi.json")

f_rtt = lindi.LindiH5pyFile.from_lindi_file(
    RTT_LINDI, local_cache=lindi.LocalCache(cache_dir=f"{CACHE}/lindi"))
nwb_rtt = NWBHDF5IO(file=f_rtt, mode="r").read()
beh_rtt = nwb_rtt.processing["behavior"].data_interfaces

cp = beh_rtt["cursor_pos"]                       # regularly sampled: rate, not timestamps
t_rtt = cp.starting_time + np.arange(cp.data.shape[0]) / cp.rate

npz_r = f"{CACHE}/rtt_behavior.npz"
if os.path.exists(npz_r):
    z = np.load(npz_r)
    pos_r, vel_r, tgt_r = z["pos"], z["vel"], z["tgt"]
else:
    pos_r = np.asarray(cp.data[:], dtype=np.float32)
    vel_r = np.asarray(beh_rtt["finger_vel"].data[:], dtype=np.float32)
    tgt_r = np.asarray(beh_rtt["target_pos"].data[:], dtype=np.float32)
    np.savez_compressed(npz_r, t=t_rtt, pos=pos_r, vel=vel_r, tgt=tgt_r)

spz_r = f"{CACHE}/rtt_spikes.npz"
if os.path.exists(spz_r):
    z = np.load(spz_r)
    st_rtt = [z[str(i)] for i in range(len(z.files))]
else:
    st_rtt = [np.asarray(s) for s in nwb_rtt.units["spike_times"][:]]
    np.savez_compressed(spz_r, **{str(i): s for i, s in enumerate(st_rtt)})
obs_rtt = np.asarray(nwb_rtt.units["obs_intervals"][0], dtype=float)

print(nwb_rtt.session_description[:105])
print("subject:", nwb_rtt.subject.subject_id, "| %.0f s | %d units" % (t_rtt[-1], len(st_rtt)))
print("observed intervals (s):", np.round(obs_rtt, 1).tolist())

# 600 samples of the behaviour are NaN; drop them before building pynapple objects
fin = np.isfinite(pos_r).all(1) & np.isfinite(vel_r).all(1) & np.isfinite(tgt_r).all(1)
print("dropping %d non-finite behaviour samples of %d" % ((~fin).sum(), len(fin)))
t_r, pos_r, vel_r, tgt_r = t_rtt[fin], pos_r[fin], vel_r[fin], tgt_r[fin]

vel_rtt = nap.TsdFrame(t=t_r, d=vel_r.astype(float), columns=["vx", "vy"])
pos_rtt = nap.TsdFrame(t=t_r, d=pos_r.astype(float), columns=["x", "y"])
spikes_rtt_all = nap.TsGroup({i: s for i, s in enumerate(st_rtt)})

obs_ep = nap.IntervalSet(start=obs_rtt[:, 0], end=obs_rtt[:, 1])
keep_rtt = np.array([u for u in spikes_rtt_all.index
                     if spikes_rtt_all[u].restrict(obs_ep).rate > UNITS_MIN_RATE])
spikes_rtt = spikes_rtt_all[list(keep_rtt)]
print("kept %d of %d units above %.1f Hz" % (len(keep_rtt), len(spikes_rtt_all), UNITS_MIN_RATE))

jump = np.flatnonzero(np.abs(np.diff(tgt_r, axis=0)).sum(1) > 1e-6)
print("%d target jumps, median inter-target interval %.2f s"
      % (len(jump), np.median(np.diff(t_r[jump + 1]))))

# %% [markdown]
# Same 20 ms binning as before. The session is continuous rather than trial-structured, so the
# unit of "do not cross this boundary" is the observation interval, and cross-validation groups are
# 5 s blocks of the session instead of trials.

# %%
counts_r = spikes_rtt.count(BIN, ep=obs_ep)
velb_r = vel_rtt.bin_average(BIN, ep=obs_ep)
posb_r = pos_rtt.bin_average(BIN, ep=obs_ep)
seg_r = np.asarray(obs_ep.in_interval(nap.Ts(counts_r.index.values))).astype(float)

ok_r = np.isfinite(velb_r.values).all(1) & np.isfinite(posb_r.values).all(1) & np.isfinite(seg_r)
Mr = dict(t=counts_r.index.values[ok_r],
          counts=np.asarray(counts_r.values)[ok_r],
          vel=np.asarray(velb_r.values)[ok_r],
          pos=np.asarray(posb_r.values)[ok_r],
          seg=seg_r[ok_r].astype(int))
Mr["speed"] = np.hypot(Mr["vel"][:, 0], Mr["vel"][:, 1])
Mr["vel_angle"] = np.arctan2(Mr["vel"][:, 1], Mr["vel"][:, 0])
Mr["block"] = (Mr["t"] // 5.0).astype(int)
print("binned matrix:", Mr["counts"].shape, " moving bins (>100 mm/s):", (Mr["speed"] > 100).sum())


def shift_by_lag_key(Mx, lag_bins, key):
    """As `shift_by_lag`, but the boundary variable is named by `key` (trial or observation seg)."""
    n = len(Mx["t"])
    src = np.arange(n) + lag_bins
    ok = (src >= 0) & (src < n)
    ok[ok] &= Mx[key][src[ok]] == Mx[key][np.arange(n)[ok]]
    return np.arange(n)[ok], src[ok]


# %% [markdown]
# ### Velocity-direction tuning and neural lead time

# %%
md_lag_r = []
for lag in tqdm(LAGS, desc="MC_RTT lag sweep"):
    a, b = shift_by_lag_key(Mr, int(round(lag / BIN)), "seg")
    mvr = Mr["speed"][b] > 100
    md_lag_r.append(cosine_fit(Mr["counts"][a][mvr] / BIN, Mr["vel_angle"][b][mvr])["mod_depth"].values)
md_lag_r = np.array(md_lag_r)
BEST_LAG_R = LAGS[int(np.argmax(np.median(md_lag_r, 1)))]
print("MC_RTT: velocity-direction tuning peaks at a neural lead of %d ms (MC_Maze: %d ms)"
      % (round(BEST_LAG_R * 1000), round(BEST_LAG * 1000)))

j_n, j_k = shift_by_lag_key(Mr, int(round(BEST_LAG_R / BIN)), "seg")
mv_r = Mr["speed"][j_k] > 100
Cr, Ar, Sr = Mr["counts"][j_n][mv_r], Mr["vel_angle"][j_k][mv_r], Mr["speed"][j_k][mv_r]
vel_fit_r = cosine_fit(Cr / BIN, Ar)
print("MC_RTT: %d/%d units tuned for instantaneous velocity direction (p<0.01), median depth %.2f Hz"
      % ((vel_fit_r["p"] < 0.01).sum(), len(vel_fit_r), vel_fit_r["mod_depth"].median()))

ib_r = np.clip(np.digitize(Ar, vedges) - 1, 0, nb - 1)
tc_r = np.stack([[Cr[ib_r == b, u].mean() / BIN for b in range(nb)] for u in range(Cr.shape[1])])
tc_se_r = np.stack([[Cr[ib_r == b, u].std() / np.sqrt((ib_r == b).sum()) / BIN for b in range(nb)]
                    for u in range(Cr.shape[1])])
vel_edges_r, vel_maps_r = tuning_2d(Mr["counts"][j_n], Mr["vel"][j_k], BIN, nbins=11, vmax=400.0)
sctr_r, spd_tun_r = speed_tuning(Mr["counts"][j_n], Mr["speed"][j_k], Mr["vel_angle"][j_k],
                                 vel_fit_r["pd"].values, BIN)
print("MC_RTT: population mean rate in the PD %.1f Hz at rest -> %.1f Hz at the fastest bin"
      % (np.nanmean(spd_tun_r[:, 0]), np.nanmean(spd_tun_r[:, -1])))

# %% [markdown]
# ### How big is the confound in each task?
#
# A direct measure: how well does hand *position* predict hand *velocity*? Fit a cross-validated
# ridge regression from a 2-D spline expansion of $(x, y)$ to $(v_x, v_y)$ in each dataset. If that
# $R^2$ is high, a position-only encoding model can mimic a velocity model without encoding
# velocity at all.

# %%
def position_predicts_velocity(Mx, groups, n_folds=5):
    bx = nmo.basis.BSplineEval(n_basis_funcs=6, label="x")
    by = nmo.basis.BSplineEval(n_basis_funcs=6, label="y")
    Xp = np.asarray((bx * by).compute_features(unit_scale(Mx["pos"][:, 0]),
                                               unit_scale(Mx["pos"][:, 1])))
    Yp = Mx["vel"]
    pr = np.full_like(Yp, np.nan)
    for tr_i, te_i in GroupKFold(n_splits=n_folds).split(Xp, Yp, groups=groups):
        sc = StandardScaler().fit(Xp[tr_i])
        mm = RidgeCV(alphas=np.logspace(-2, 4, 10)).fit(sc.transform(Xp[tr_i]), Yp[tr_i])
        pr[te_i] = mm.predict(sc.transform(Xp[te_i]))
    return 1 - ((Yp - pr) ** 2).sum(0) / ((Yp - Yp.mean(0)) ** 2).sum(0)


confound = {}
for nm_, Mx_, gk_ in [("MC_Maze (center-out)", M, "trial"), ("MC_RTT (random target)", Mr, "block")]:
    confound[nm_] = position_predicts_velocity(Mx_, Mx_[gk_])
    print("%-24s cross-validated R2 of velocity predicted from position: vx=%+.3f  vy=%+.3f"
          % (nm_, confound[nm_][0], confound[nm_][1]))

# %% [markdown]
# ### The GLM comparison, repeated where position and velocity are decorrelated

# %%
glm_mask_r = Mr["speed"][j_k] > 50.0
Y_glm_r = Mr["counts"][j_n][glm_mask_r].astype(float)
groups_r = Mr["block"][j_n][glm_mask_r]
feat_r = dict(angle=Mr["vel_angle"][j_k][glm_mask_r],
              speed=unit_scale(Mr["speed"][j_k][glm_mask_r]),
              x=unit_scale(Mr["pos"][j_k][glm_mask_r, 0]),
              y=unit_scale(Mr["pos"][j_k][glm_mask_r, 1]))

# fresh basis objects: NeMoS Eval bases record the input shape on first use
rb_ang = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, label="direction")
rb_spd = nmo.basis.MSplineEval(n_basis_funcs=5, label="speed")
rb_px = nmo.basis.BSplineEval(n_basis_funcs=5, label="pos_x")
rb_py = nmo.basis.BSplineEval(n_basis_funcs=5, label="pos_y")
MODELS_R = {"speed only": (rb_spd, ("speed",)),
            "direction only": (rb_ang, ("angle",)),
            "direction x speed": (rb_ang * rb_spd, ("angle", "speed")),
            "position": (rb_px * rb_py, ("x", "y")),
            "dir x speed + position": (rb_ang * rb_spd + rb_px * rb_py,
                                       ("angle", "speed", "x", "y"))}
assert list(MODELS_R) == list(MODELS)

glm_scores_r = {}
for name, (basis, argnames) in MODELS_R.items():
    Xr = np.asarray(basis.compute_features(*[feat_r[a] for a in argnames]))
    glm_scores_r[name] = fit_cv(Xr, Y_glm_r, groups_r)
    print(f"MC_RTT {name:24s} median held-out deviance explained = "
          f"{np.median(glm_scores_r[name]):.4f}")
print("MC_RTT: velocity model beats position model for %d/%d units (MC_Maze: %d/%d)"
      % ((glm_scores_r["direction x speed"] > glm_scores_r["position"]).sum(), len(keep_rtt),
         (glm_scores["direction x speed"] > glm_scores["position"]).sum(), len(keep)))
print("MC_RTT: adding speed helps %d/%d units (MC_Maze: %d/%d)"
      % ((glm_scores_r["direction x speed"] > glm_scores_r["direction only"]).sum(), len(keep_rtt),
         (glm_scores["direction x speed"] > glm_scores["direction only"]).sum(), len(keep)))

# %% [markdown]
# ### Velocity decoding on MC_RTT

# %%
X_dec_r = lagged_population(Mr["counts"].astype(np.float32), Mr["seg"], np.arange(0, 12, 2))
good_r = ~np.isnan(X_dec_r).any(1)
X_dec_r, Y_dec_r, g_dec_r = X_dec_r[good_r], Mr["vel"][good_r], Mr["block"][good_r]
pred_r = np.full_like(Y_dec_r, np.nan)
for tr_i, te_i in tqdm(list(GroupKFold(n_splits=5).split(X_dec_r, Y_dec_r, groups=g_dec_r)),
                       desc="MC_RTT decoding folds"):
    sc = StandardScaler().fit(X_dec_r[tr_i])
    mm = RidgeCV(alphas=np.logspace(0, 5, 12)).fit(sc.transform(X_dec_r[tr_i]), Y_dec_r[tr_i])
    pred_r[te_i] = mm.predict(sc.transform(X_dec_r[te_i]))
r2_vel_r = 1 - ((Y_dec_r - pred_r) ** 2).sum(0) / ((Y_dec_r - Y_dec_r.mean(0)) ** 2).sum(0)
ang_err_r = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(Y_dec_r[:, 1], Y_dec_r[:, 0])
                                                    - np.arctan2(pred_r[:, 1], pred_r[:, 0]))))))
fast_r = np.hypot(*Y_dec_r.T) > 100
print("MC_RTT velocity decoding: R2(vx)=%.3f, R2(vy)=%.3f, median direction error %.1f deg"
      % (r2_vel_r[0], r2_vel_r[1], np.median(ang_err_r[fast_r])))

# %%
fig = plt.figure(figsize=(16.5, 12.5))
gs = fig.add_gridspec(3, 4, hspace=.62, wspace=.46, top=0.90, bottom=0.055, left=0.06, right=0.975)
order_r = np.argsort(-vel_fit_r["r2"].values)[:3]

for j, u in enumerate(order_r):
    ax = fig.add_subplot(gs[0, j], projection="polar")
    ax.errorbar(np.r_[vctr, vctr[0]], np.r_[tc_r[u], tc_r[u][0]],
                yerr=np.r_[tc_se_r[u], tc_se_r[u][0]], color="k", lw=1.4, marker=".", ms=4)
    g = np.linspace(-np.pi, np.pi, 200)
    ax.plot(g, np.clip(vel_fit_r["b0"][u] + vel_fit_r["mod_depth"][u]
                       * np.cos(g - vel_fit_r["pd"][u]), 0, None), "r", lw=2)
    ax.set_title(f"MC_RTT unit {keep_rtt[u]}\nPD={np.degrees(vel_fit_r['pd'][u]):.0f}$\\degree$, "
                 f"MD={vel_fit_r['mod_depth'][u]:.1f} Hz", pad=22, fontsize=10)
    ax.set_rlabel_position(np.degrees(vel_fit_r["pd"][u]) + 150)
    ax.tick_params(labelsize=7)

ax = fig.add_subplot(gs[0, 3], projection="polar")
sig_r = vel_fit_r["p"].values < 0.01
ax.hist(vel_fit_r["pd"].values[sig_r], bins=np.linspace(-np.pi, np.pi, 25), color="seagreen")
ax.set_title(f"Preferred velocity directions\n({sig_r.sum()}/{len(vel_fit_r)} units, p<0.01)",
             pad=22, fontsize=10)
ax.set_rlabel_position(255)
ax.tick_params(labelsize=8)

ext_r = [vel_edges_r[0], vel_edges_r[-1], vel_edges_r[0], vel_edges_r[-1]]
for j, u in enumerate(order_r[:2]):
    ax = fig.add_subplot(gs[1, j])
    im = ax.imshow(vel_maps_r[u], origin="lower", extent=ext_r, cmap="viridis")
    ax.arrow(0, 0, 320 * np.cos(vel_fit_r["pd"][u]), 320 * np.sin(vel_fit_r["pd"][u]), color="w",
             width=12, head_width=42, length_includes_head=True)
    ax.set(xlabel="$v_x$ (mm/s)", ylabel="$v_y$ (mm/s)",
           title=f"unit {keep_rtt[u]}: 2-D velocity map")
    plt.colorbar(im, ax=ax, label="rate (Hz)", fraction=.046)

ax = fig.add_subplot(gs[1, 2])
for lg, md_, lbl, c in [(LAGS, md_by_lag, f"MC_Maze (Jenkins, {len(keep)} units)", "tab:red"),
                        (LAGS, md_lag_r, f"MC_RTT (Indy, {len(keep_rtt)} units)", "seagreen")]:
    mm_ = np.median(md_, 1)
    ax.plot(np.asarray(lg) * 1000, mm_ / mm_.max(), "-o", ms=3, color=c, label=lbl)
    ax.axvline(np.asarray(lg)[int(np.argmax(mm_))] * 1000, color=c, ls="--", lw=1)
ax.set(xlabel="neural lead time (ms)", ylabel="median modulation depth (normalized)",
       title="Neural activity leads the kinematics\nin both datasets")
ax.legend(fontsize=7.5, loc="lower center")

ax = fig.add_subplot(gs[1, 3])
nz = speed_tun / np.nanmax(speed_tun, 1, keepdims=True)
nr = spd_tun_r / np.nanmax(spd_tun_r, 1, keepdims=True)
ax.plot(speed_ctr, np.nanmean(nz, 0), "-o", color="tab:red", lw=2, ms=4, label="MC_Maze")
ax.plot(sctr_r, np.nanmean(nr, 0), "-o", color="seagreen", lw=2, ms=4, label="MC_RTT")
ax.set(xlabel="hand speed (mm/s)", ylabel="normalized rate",
       title="Speed tuning within each unit's PD\n(population mean, $\\pm45\\degree$)")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 0])
cnames = list(confound)
w = 0.35
for i, comp in enumerate(["$v_x$", "$v_y$"]):
    ax.bar(np.arange(2) + (i - .5) * w, [confound[n][i] for n in cnames], width=w, label=comp,
           color=["tab:blue", "tab:cyan"][i])
ax.axhline(0, color="k", lw=.8)
ax.set(xticks=range(2), ylabel="cross-validated $R^2$", ylim=(-0.08, 0.72),
       title="How much of hand velocity is\npredictable from hand position?")
ax.set_xticklabels(["MC_Maze\n(center-out)", "MC_RTT\n(random target)"], fontsize=8.5)
ax.legend(fontsize=8, loc="upper center", ncol=2)
for i, n in enumerate(cnames):
    for k in (0, 1):
        ax.text(i + (k - .5) * w, confound[n][k] + .02, f"{confound[n][k]:.2f}",
                ha="center", fontsize=7.5)

ax = fig.add_subplot(gs[2, 1:3])
mod_lbl = ["speed", "direction", "direction $\\times$ speed", "position", "dir $\\times$ speed + pos"]
mods = list(MODELS)
yy = np.arange(len(mods))
ax.barh(yy + .19, [np.median(glm_scores[m]) for m in mods], height=.36, color="tab:red",
        label=f"MC_Maze ({len(keep)} units)")
ax.barh(yy - .19, [np.median(glm_scores_r[m]) for m in mods], height=.36, color="seagreen",
        label=f"MC_RTT ({len(keep_rtt)} units)")
for i, m in enumerate(mods):
    ax.text(np.median(glm_scores[m]) + .0008, i + .19, f"{np.median(glm_scores[m]):.3f}",
            va="center", fontsize=8)
    ax.text(np.median(glm_scores_r[m]) + .0008, i - .19, f"{np.median(glm_scores_r[m]):.3f}",
            va="center", fontsize=8)
ax.set(yticks=yy, xlabel="median held-out Poisson deviance explained",
       title="Poisson GLM encoding models (NeMoS): the position model only rivals the velocity\n"
             "model in the task where position and velocity are confounded")
ax.set_yticklabels(mod_lbl, fontsize=9)
ax.set_xlim(0, 0.048)
ax.legend(fontsize=8.5, loc="lower right")

ax = fig.add_subplot(gs[2, 3])
sel_r = np.unique(g_dec_r)[30:34]
mk = np.isin(g_dec_r, sel_r)
tt = np.arange(mk.sum()) * BIN
ax.plot(tt, Y_dec_r[mk, 0], "k", lw=1.5, label="actual $v_x$")
ax.plot(tt, pred_r[mk, 0], "seagreen", lw=1.3, label="decoded $v_x$")
ax.plot(tt, Y_dec_r[mk, 1] - 900, "k", lw=1.5)
ax.plot(tt, pred_r[mk, 1] - 900, "tab:blue", lw=1.3, label="decoded $v_y$")
ax.text(0.3, 300, "$v_x$", fontsize=10)
ax.text(0.3, -650, "$v_y$", fontsize=10)
ax.set(xlabel="time (s)", ylabel="velocity (mm/s), $v_y$ offset",
       title="MC_RTT velocity decoding\n$R^2$=%.2f / %.2f, median dir. error %.0f$\\degree$"
             % (r2_vel_r[0], r2_vel_r[1], np.median(ang_err_r[fast_r])))
ax.legend(fontsize=7.5, loc="upper right")

fig.suptitle("Replication on a second dataset: MC_RTT (DANDI:000129, monkey Indy, self-paced "
             "random-target reaching)", y=0.975, fontsize=13)
plt.savefig("fig06_replication_mc_rtt.png", dpi=135, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# The replication holds on every point that matters. Roughly 80% of MC_RTT units are cosine-tuned
# for the direction of the instantaneous hand velocity, the tuning peaks at a neural lead of the
# same order (80 ms here, 100 ms in MC_Maze), rate grows with speed within the preferred direction
# with the same saturating shape, and hand velocity decodes linearly from the population with a
# median instantaneous direction error of about 20 degrees, essentially the MC_Maze figure.
#
# The confound analysis is the part that could not be done with MC_Maze alone. In the center-out
# task, hand position predicts hand velocity with a cross-validated $R^2$ of 0.49 ($v_x$) and 0.33
# ($v_y$); in the random-target task it predicts essentially nothing ($R^2 \approx 0$). And the GLM
# comparison tracks that exactly: in MC_Maze the position model (0.022) is statistically
# indistinguishable from the velocity model (0.024) and actually wins for 47 of 115 units, whereas
# in MC_RTT the position model collapses to 0.002 against 0.015 for velocity and loses for 77 of
# 94 units. The apparent position code in MC_Maze was the task geometry, not the neurons.
#
# One difference is worth reporting rather than glossing over. Adding speed to the direction model
# helps almost every MC_Maze unit (114/115) but fewer than half of the MC_RTT units (42/94), even
# though the population median still improves. MC_RTT contributes about a fifth as many
# above-threshold bins as MC_Maze, so the 40-feature direction x speed model overfits the weakly
# modulated units; the gain is concentrated in the well-modulated ones.

# %% [markdown]
# ## 11. Is the per-bin significance real? A block-shift null
#
# Every "$n$ of $m$ units tuned for instantaneous velocity direction" figure above rests on an F
# test that treats 20 ms bins as independent samples. They are not. Firing rate and hand velocity
# are both strongly autocorrelated within a reach, so the effective number of independent samples
# is far smaller than the tens of thousands of bins the test is handed, and the nominal p-values
# are anticonservative.
#
# The honest null is to circularly shift the neural series against the kinematics by a large
# offset (at least 40 s, wrapping within the session). That preserves the autocorrelation of both
# signals exactly while destroying the true pairing, so anything left is what autocorrelation
# alone can manufacture. A unit counts as tuned if its observed $R^2$ beats the shifted $R^2$ on
# at least 95% of 100 shifts.

# %%
def block_shift_null(Mx, best_lag, key, n_shuffles=100, speed_thresh=100.0, seed=0):
    """Observed cosine fit and a null built by circularly shifting counts against kinematics."""
    i_n, i_k = shift_by_lag_key(Mx, int(round(best_lag / BIN)), key)
    mv = Mx["speed"][i_k] > speed_thresh
    i_n, i_k = i_n[mv], i_k[mv]
    A = Mx["vel_angle"][i_k]
    obs = cosine_fit(Mx["counts"][i_n] / BIN, A)

    n = len(Mx["t"])
    guard = int(round(40.0 / BIN))  # keep the shift well clear of zero and of a full wrap
    offsets = np.random.default_rng(seed).integers(guard, n - guard, size=n_shuffles)
    null = np.empty((n_shuffles, Mx["counts"].shape[1]))
    for j, k in enumerate(tqdm(offsets, desc="block shifts", leave=False)):
        null[j] = cosine_fit(np.roll(Mx["counts"], k, axis=0)[i_n] / BIN, A)["r2"].values
    p_emp = (null >= obs["r2"].values[None, :]).sum(0) / (n_shuffles + 1)
    return obs, null, p_emp


shuf = {}
for nm, Mx, lag, key in [("MC_Maze", M, BEST_LAG, "trial"), ("MC_RTT", Mr, BEST_LAG_R, "seg")]:
    obs_s, null_s, p_s = block_shift_null(Mx, lag, key)
    shuf[nm] = (obs_s, null_s, p_s)
    print("%s: %d/%d tuned by the nominal F test, %d/%d by the block-shift null; "
          "observed median R2 %.4f vs null 99.9th pct %.5f"
          % (nm, (obs_s["p"] < 0.01).sum(), len(obs_s), (p_s < 0.05).sum(), len(obs_s),
             np.median(obs_s["r2"]), np.percentile(null_s, 99.9)))

# %%
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
for col, (nm, colr, n_lab) in enumerate([("MC_Maze", "#2b6cb0", "Jenkins, 115 units"),
                                         ("MC_RTT", "#c05621", "Indy, 94 units")]):
    obs_s, null_s, p_s = shuf[nm]
    r2 = obs_s["r2"].values
    order = np.argsort(r2)
    x = np.arange(len(r2))

    ax = axes[0, col]
    ax.fill_between(x, null_s.min(axis=0)[order], np.percentile(null_s, 99.9, axis=0)[order],
                    color="0.7", label="block-shift null (min to 99.9th pct)")
    ax.plot(x, r2[order], ".", ms=5, color=colr, label="observed")
    ns = p_s[order] >= 0.05
    ax.plot(x[ns], r2[order][ns], "x", ms=6, color="0.35", label="not significant")
    ax.set_yscale("log")
    ax.set_xlabel("unit (sorted by observed $R^2$)")
    ax.set_ylabel("cosine fit $R^2$")
    ax.set_title(f"{nm} ({n_lab})", pad=8)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    ratio = r2 / np.percentile(null_s, 99.9, axis=0)
    ax = axes[1, col]
    ax.hist(ratio, bins=np.logspace(np.log10(min(ratio.min(), 0.5) * 0.8),
                                    np.log10(ratio.max() * 1.3), 34), color=colr, alpha=0.85)
    ax.axvline(1.0, color="k", ls="--", lw=1.5)
    ax.text(1.15, ax.get_ylim()[1] * 0.92, "null 99.9th pct", fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("observed $R^2$ / that unit's null 99.9th percentile")
    ax.set_ylabel("units")
    ax.set_title("nominal F test: %d/%d   block-shift null: %d/%d   (median %.0fx null)"
                 % ((obs_s["p"] < 0.01).sum(), len(obs_s), (p_s < 0.05).sum(), len(obs_s),
                    np.median(ratio)), fontsize=10, pad=8)

fig.suptitle("Velocity-direction tuning survives a null that preserves the autocorrelation of "
             "both signals", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.955))
plt.savefig("fig07_shuffle_control.png", dpi=140)
plt.close(fig)

# %% [markdown]
# The concern was real but the result is not affected by it. In MC_Maze the block-shift null
# returns exactly the same 111 of 115 units, and the median unit's observed $R^2$ is about 12
# times its own null 99.9th percentile. In MC_RTT the null count is 80 of 94, slightly *more* than
# the 75 the nominal F test gives, with a median unit about 3 times its null. The shifted fits sit
# two orders of magnitude below the observed ones in MC_Maze, so autocorrelation on its own cannot
# manufacture tuning at anything like the observed scale. The units the two criteria disagree
# about in MC_RTT are all in the weakly modulated tail, which is where a shorter recording should
# make the tests disagree.
#
# This control applies to the per-bin velocity analyses. The trial-level direction results in
# section 5 do not need it: there each trial contributes one rate, and trials are separated in
# time, so the independence assumption is reasonable there. The GLM and decoding results are
# cross-validated with grouping by trial or by block, which handles the same issue by construction.

# %% [markdown]
# ## 12. Summary
#
# | result | MC_Maze (Jenkins, DANDI:000128) | MC_RTT (Indy, DANDI:000129) |
# |---|---|---|
# | units analysed (rate > 1 Hz) | 115 of 182 | 94 of 130 |
# | task | delayed center-out, 2295 trials | self-paced random target, 649 s |
# | cosine-tuned for reach direction, peri-movement | 93% (3% at pre-target baseline) | n/a (no trial structure) |
# | cosine-tuned for reach direction, delay period | 83% | n/a |
# | tuned for instantaneous velocity direction (nominal F test) | 111 / 115 | 75 / 94 |
# | the same, against a block-shift null | 111 / 115 | 80 / 94 |
# | neural lead at which velocity tuning peaks | ~100 ms | ~80 ms |
# | mean rate in the preferred direction, rest -> fastest | ~4 -> ~8.5 Hz | ~4.6 -> ~10.7 Hz |
# | velocity predictable from position (CV R^2, vx / vy) | **0.49 / 0.33** | **-0.00 / -0.00** |
# | GLM deviance explained, direction only | 0.013 | 0.012 |
# | GLM deviance explained, direction x speed | 0.024 | 0.015 |
# | GLM deviance explained, position | 0.022 | 0.002 |
# | GLM deviance explained, velocity + position | 0.040 | 0.018 |
# | velocity decoding R^2 (vx, vy) | 0.59, 0.52 | 0.40, 0.41 |
# | median instantaneous direction decoding error | 21 deg | 20 deg |
# | 7-way reach direction decoding (baseline / delay / movement) | 0.22 / 0.59 / 0.81 | n/a |
#
# The direction code is present before the movement begins (83% of units tuned during the delay,
# 59% decoding accuracy from delay activity alone) and strengthens during the reach. It is not
# simply a code for the target: when the same targets are reached along a curved path forced by
# the barriers, preferred directions shift by a median of 27 degrees, and the moment-by-moment
# analysis shows the cells follow the instantaneous velocity vector rather than the straight-line
# direction to the goal. Every velocity result replicates in a second monkey performing a
# different reaching task, recorded by a different laboratory.
#
# The position/velocity question is worth restating because it is the one place where the first
# dataset was genuinely ambiguous. In the center-out task the position model scored 0.022 against
# 0.024 for the velocity model, which looks like evidence for a position code. It is not: in that
# task hand position predicts hand velocity with a cross-validated R^2 of 0.49, so a position model
# gets the velocity code for free. In the random-target task, where position predicts nothing about
# velocity, the position model falls to 0.002 while the velocity model holds at 0.015. The velocity
# tuning is real, and the most economical reading of the position result is that it was task
# geometry. That inference should be stated with its limits: the two datasets differ in more than
# target placement (different monkey, array, laboratory, and a session about a tenth as long), so
# the argument rests on the measured mediator, the position-to-velocity $R^2$ of 0.49 against 0.00,
# and not on the dataset contrast by itself. A within-animal manipulation, randomising target
# placement for part of one session, would settle it more firmly than anything available here.
#
# The per-bin significance counts were checked against a null that preserves the autocorrelation
# of both the spike counts and the kinematics (section 11), because the F test on 20 ms bins
# assumes an independence the data do not have. The counts survive it.
#
# Two caveats remain. First, the delay-period and movement preferred directions of the same unit
# are only weakly related (circular r = 0.15), consistent with reports that preparatory and
# movement-epoch tuning in M1/PMd are largely dissociated, so "the" preferred direction of a unit
# is epoch-specific rather than a fixed property. Second, MC_RTT contributes about a fifth as many
# above-threshold bins as MC_Maze, and the direction x speed model correspondingly overfits its
# weakly modulated units (it improves only 42 of 94, against 114 of 115 in MC_Maze), so the speed
# contribution is less firmly established in the replication than the direction contribution.

# %%
print("figures written:")
for fn in sorted(fn for fn in os.listdir(".") if fn.endswith(".png")):
    print("  ", fn)
