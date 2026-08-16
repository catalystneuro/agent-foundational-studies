# %% [markdown]
# # Reach direction and velocity tuning in macaque motor cortex (DANDI 000128, MC_Maze)
#
# This notebook demonstrates two classical properties of motor cortical neurons using real
# data streamed from the DANDI Archive:
#
# 1. **Reach-direction tuning**: M1/PMd neurons modulate their firing rate with the
#    direction of an upcoming/ongoing reach, often following an approximate cosine
#    tuning profile (Georgopoulos et al., 1982).
# 2. **Velocity/speed tuning**: firing rates also scale with hand speed and are
#    temporally coupled to the hand velocity, with neural activity tending to *lead*
#    the kinematics by tens of milliseconds.
#
# **Dataset**: [DANDI 000128](https://dandiarchive.org/dandiset/000128) — MC_Maze from the
# Neural Latents Benchmark '21 (Churchland lab). Extracellular recordings from primary motor
# (M1) and dorsal premotor cortex (PMd) of a macaque ("Jenkins") performing a delayed
# center-out reaching task with virtual barriers that force curved trajectories. The file
# used here (`sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb`) contains 182 sorted
# units, 2295 trials, and 1 kHz hand position/velocity.
#
# **Methods**: streaming NWB access with `remfile` + disk cache, data handling and tuning
# curves with `pynapple`, and Poisson GLM encoding models with `nemos`.
#
# Note on units: the NWB file labels hand position/velocity as meters and m/s, but the
# values are clearly on a millimeter scale (reach amplitudes ~130), so we report mm and mm/s.

# %% [markdown]
# ## Setup and streaming data access

# %%
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import nemos as nmo
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson
from tqdm import tqdm

DANDISET = "000128"
VERSION = "0.220113.0400"
ASSET_PATH = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"
CACHE_DIR = "/tmp/remfile_cache_mcmaze"

def get_download_url(dandiset, version, path):
    """Resolve an asset download URL via the DANDI REST API (handles pagination)."""
    url = f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/{version}/assets/"
    while url:
        r = requests.get(url, params={"page_size": 1000})
        r.raise_for_status()
        d = r.json()
        for a in d["results"]:
            if a["path"] == path:
                return f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"
        url = d.get("next")
    raise FileNotFoundError(path)

s3_url = get_download_url(DANDISET, VERSION, ASSET_PATH)
disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

units = nwb["units"]          # TsGroup: sorted spike times
hp = nwb["hand_pos"]          # TsdFrame: hand x/y at 1 kHz
hv = nwb["hand_vel"]          # TsdFrame: hand vx/vy at 1 kHz
trials = nwbfile.trials.to_dataframe()
unit_ids = np.array(list(units.keys()))
n_units = len(units)
print(f"{n_units} units, {len(trials)} trials")
print("trial columns:", list(trials.columns))

# %% [markdown]
# ## Trial structure and reach direction
#
# Each trial presents 1–3 candidate targets; `active_target` indexes the cued one in the
# (ragged) `target_pos` array. We define the **reach direction** of a trial as the angle of
# the vector from the hand position at movement onset to the cued target.

# %%
def active_pos(row):
    p = np.asarray(row["target_pos"])
    return p if p.ndim == 1 else p[int(row["active_target"])]

trials["target_xy"] = trials.apply(active_pos, axis=1)
onset_t = trials["move_onset_time"].values
go_t = trials["go_cue_time"].values

pos_at_onset = np.stack([hp.get(t - 0.005, t + 0.005).values.mean(axis=0)
                         for t in tqdm(onset_t, desc="hand pos at move onset")])
txy = np.stack(trials["target_xy"].values)
reach_vec = txy - pos_at_onset
trials["reach_dir"] = np.arctan2(reach_vec[:, 1], reach_vec[:, 0])
trials["reach_dist"] = np.linalg.norm(reach_vec, axis=1)
print("reach distance (mm):", trials["reach_dist"].describe().round(1).to_dict())
print("all trials successful:", (trials["success"] == 1).all())
print("split counts:", trials["split"].value_counts().to_dict())

# %% [markdown]
# ## Behavior overview
#
# The monkey makes curved reaches around virtual barriers from a central start position to
# peripheral targets. Speed profiles are bell-shaped and tightly locked to the annotated
# movement onset.

# %%
fig = plt.figure(figsize=(14, 4.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.3, 0.9], wspace=0.3)
rng = np.random.default_rng(3)
idx = rng.choice(len(trials), size=150, replace=False)

axa = fig.add_subplot(gs[0, 0])
for i in idx:
    seg = hp.get(onset_t[i] - 0.05, onset_t[i] + 0.55)
    axa.plot(seg[:, 0].values, seg[:, 1].values,
             color=plt.cm.hsv((trials["reach_dir"].values[i] + np.pi) / (2 * np.pi)),
             alpha=0.45, lw=0.8)
axa.set_xlabel("x (mm)"); axa.set_ylabel("y (mm)")
axa.set_title("Reach trajectories (n=150)\ncolored by reach direction", fontsize=10)
axa.set_aspect("equal")

axb = fig.add_subplot(gs[0, 1])
segs = []
for i in idx[:40]:
    seg = hv.get(onset_t[i] - 0.3, onset_t[i] + 0.7)
    axb.plot(seg.t - onset_t[i], np.linalg.norm(seg.values, axis=1),
             color="steelblue", alpha=0.25, lw=0.8)
for i in idx:
    segs.append(np.linalg.norm(hv.get(onset_t[i] - 0.3, onset_t[i] + 0.7).values, axis=1))
segs = np.array(segs)
axb.plot(seg.t - onset_t[i], segs.mean(axis=0), color="k", lw=2, label="mean")
axb.axvline(0, color="k", ls="--", lw=1)
axb.set_xlabel("time from move onset (s)"); axb.set_ylabel("hand speed (mm/s)")
axb.set_title("Hand speed around movement onset\n(40 example trials + mean)", fontsize=10)
axb.legend(fontsize=8)

axc = fig.add_subplot(gs[0, 2], projection="polar")
bins = np.linspace(-np.pi, np.pi, 25)
cnt, _ = np.histogram(trials["reach_dir"], bins=bins)
th = (bins[:-1] + bins[1:]) / 2
axc.bar(th, cnt, width=np.diff(bins), color="seagreen", edgecolor="k", lw=0.4)
axc.set_theta_zero_location("E"); axc.set_theta_direction(1)
axc.set_title(f"Reach direction distribution\n(n={len(trials)} trials)", fontsize=10, pad=22)
axc.tick_params(labelsize=8); axc.set_rlabel_position(250)
axc.tick_params(axis="y", labelsize=7)

fig.suptitle("MC_Maze behavior: delayed center-out reaching with virtual barriers (monkey Jenkins)",
             fontsize=12, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("fig1_behavior.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig1_behavior.png")

# %% [markdown]
# ## Raw neural data
#
# A 5 s segment with 60 simultaneously recorded units and the hand kinematics. Bursts of
# population activity accompany each reach (dashed lines = movement onsets).

# %%
i0 = np.searchsorted(onset_t, 1000.0)
t0, t1 = onset_t[i0] - 1.2, onset_t[i0] + 3.6
seg_v = hv.get(t0, t1)
spd = np.linalg.norm(seg_v.values, axis=1)

fig, (ax_r, ax_k) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.08))
n_show = 60
show_units = np.linspace(0, n_units - 1, n_show).astype(int)
for row, u in enumerate(show_units):
    st = np.asarray(units[unit_ids[u]].get(t0, t1).t)
    ax_r.scatter(st, np.full(st.shape, row, dtype=float), s=3, color="k", marker="|")
for ot in onset_t:
    if t0 <= ot <= t1:
        ax_r.axvline(ot, color="crimson", ls="--", lw=1)
        ax_k.axvline(ot, color="crimson", ls="--", lw=1)
ax_k.plot([], [], color="crimson", ls="--", label="move onset")
ax_r.set_ylabel("unit"); ax_r.set_ylim(-0.5, n_show - 0.5)
ax_r.set_yticks([0, 29, 59]); ax_r.set_yticklabels([1, 30, 60])
ax_r.set_title("Simultaneously recorded M1/PMd units (raster) and hand kinematics", fontsize=11)

ax_k.plot(seg_v.t, seg_v.values[:, 0], color="steelblue", lw=1, label="velocity x")
ax_k.plot(seg_v.t, seg_v.values[:, 1], color="darkorange", lw=1, label="velocity y")
ax_k.plot(seg_v.t, spd, color="k", lw=1.4, label="speed")
ax_k.set_xlabel("time (s)"); ax_k.set_ylabel("velocity / speed (mm/s)")
ax_k.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("fig2_raw_data.png", dpi=150)
plt.close()
print("saved fig2_raw_data.png")

# %% [markdown]
# ## Direction tuning of peri-movement firing rates
#
# For each unit we compute the firing rate in a peri-movement window (−50 to +400 ms around
# movement onset) on every trial, then quantify direction tuning three ways:
#
# - **Cosine fit**: rate = b0 + A·cos(θ − PD), fit by linear regression on [cos θ, sin θ].
# - **ANOVA** across the populated 45° direction bins.
# - **Shuffle test** on the rate-weighted mean resultant length (500 shuffles).
#
# A unit is called direction-tuned when both tests pass p < 0.01. We repeat the analysis in
# the delay epoch (−400 to 0 ms around the go cue) for comparison.

# %%
def trial_rates(win, align_t):
    """Firing rate (Hz) per unit per trial in window win around align_t."""
    ep = nap.IntervalSet(start=align_t + win[0], end=align_t + win[1])
    cnt = units.count(win[1] - win[0], ep=ep)  # one row per trial
    assert cnt.shape[0] == len(align_t)
    return cnt.values / (win[1] - win[0])

MOVE_WIN = (-0.05, 0.40)
DELAY_WIN = (-0.40, 0.0)
rates_move = trial_rates(MOVE_WIN, onset_t)   # (n_trials, n_units)
rates_delay = trial_rates(DELAY_WIN, go_t)

NBINS = 8
bin_edges = np.linspace(-np.pi, np.pi, NBINS + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
dir_bin = np.digitize(trials["reach_dir"].values, bin_edges) - 1
populated = [b for b in range(NBINS) if (dir_bin == b).sum() > 1]
print("trials per direction bin:", np.bincount(dir_bin, minlength=NBINS),
      "(the lower-left quadrant of the workspace has no targets)")

def tuning_stats(rates, dirs, n_shuf=500):
    """Cosine fit + ANOVA + shuffle test for each unit. rates: (n_trials, n_units)."""
    rng = np.random.default_rng(0)
    n_trials, n_units = rates.shape
    X = np.column_stack([np.ones(n_trials), np.cos(dirs), np.sin(dirs)])
    beta, *_ = np.linalg.lstsq(X, rates, rcond=None)
    pred = X @ beta
    ss_res = ((rates - pred) ** 2).sum(axis=0)
    ss_tot = ((rates - rates.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    resultant = (np.abs((rates * np.exp(1j * dirs[:, None])).sum(axis=0))
                 / np.maximum(rates.sum(axis=0), 1e-12))
    p_anova = np.full(n_units, np.nan)
    for u in range(n_units):
        p_anova[u] = stats.f_oneway(*[rates[dir_bin == b, u] for b in populated])[1]
    shuf = np.zeros((n_shuf, n_units))
    for s in tqdm(range(n_shuf), desc="shuffle test", leave=False):
        rs = rates[rng.permutation(n_trials)]
        shuf[s] = (np.abs((rs * np.exp(1j * dirs[:, None])).sum(axis=0))
                   / np.maximum(rs.sum(axis=0), 1e-12))
    return dict(pd=np.arctan2(beta[2], beta[1]), amp=np.hypot(beta[1], beta[2]),
                base=beta[0], mod_depth=np.hypot(beta[1], beta[2]) / np.maximum(beta[0], 1e-6),
                r2=r2, resultant=resultant, p_anova=p_anova,
                p_shuf=(shuf >= resultant).mean(axis=0))

tun_move = tuning_stats(rates_move, trials["reach_dir"].values)
tun_delay = tuning_stats(rates_delay, trials["reach_dir"].values)
sig_move = (tun_move["p_anova"] < 0.01) & (tun_move["p_shuf"] < 0.01)
sig_delay = (tun_delay["p_anova"] < 0.01) & (tun_delay["p_shuf"] < 0.01)
print(f"direction-tuned (movement epoch): {sig_move.sum()}/{n_units} = {sig_move.mean()*100:.0f}%")
print(f"direction-tuned (delay epoch):    {sig_delay.sum()}/{n_units} = {sig_delay.mean()*100:.0f}%")

def binned_tc(rates):
    tc = np.full((rates.shape[1], NBINS), np.nan)
    se = np.full((rates.shape[1], NBINS), np.nan)
    for b in range(NBINS):
        m = dir_bin == b
        if m.sum() > 1:
            tc[:, b] = rates[m].mean(axis=0)
            se[:, b] = rates[m].std(axis=0) / np.sqrt(m.sum())
    return tc, se

tc_move, tc_move_se = binned_tc(rates_move)

# %% [markdown]
# ### Example units
#
# Four strongly tuned units with different preferred directions. Black: mean ± sem per
# direction bin; red: cosine fit (clipped at zero for display).

# %%
order = np.argsort(-tun_move["mod_depth"] * sig_move)
chosen = []
for u in order:
    if not sig_move[u]:
        continue
    if all(abs(np.angle(np.exp(1j * (tun_move["pd"][u] - tun_move["pd"][v])))) > 0.6 for v in chosen):
        chosen.append(u)
    if len(chosen) == 4:
        break

th_fine = np.linspace(-np.pi, np.pi, 200)
fig = plt.figure(figsize=(13, 7))
for i, u in enumerate(chosen):
    ax = fig.add_subplot(2, 4, i + 1)
    m = ~np.isnan(tc_move[u])
    ax.errorbar(np.degrees(bin_centers[m]), tc_move[u, m], yerr=tc_move_se[u, m],
                fmt="o", color="k", ms=4, capsize=2, label="data (mean ± sem)")
    fit = tun_move["base"][u] + tun_move["amp"][u] * np.cos(th_fine - tun_move["pd"][u])
    ax.plot(np.degrees(th_fine), np.clip(fit, 0, None), color="crimson", lw=1.5, label="cosine fit")
    ax.set_xlabel("reach direction (deg)")
    if i == 0:
        ax.set_ylabel("firing rate (Hz)")
        ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"unit {unit_ids[u]}  (PD={np.degrees(tun_move['pd'][u]):.0f}°, "
                 f"depth={tun_move['mod_depth'][u]:.2f})", fontsize=9)
    ax.set_xlim(-180, 180); ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_ylim(bottom=0)

    axp = fig.add_subplot(2, 4, i + 5, projection="polar")
    th_b = np.concatenate([bin_centers[m], bin_centers[m][:1] + 2 * np.pi])
    r_b = np.concatenate([tc_move[u, m], tc_move[u, m][:1]])
    axp.plot(th_b, r_b, "o-", color="k", ms=4, lw=1)
    axp.plot(np.concatenate([th_fine, th_fine[:1] + 2 * np.pi]),
             np.clip(np.concatenate([fit, fit[:1]]), 0, None), color="crimson", lw=1.5)
    axp.set_theta_zero_location("E"); axp.set_theta_direction(1)
    axp.tick_params(labelsize=7); axp.set_rlabel_position(22)
fig.suptitle("Direction tuning of example M1/PMd units (peri-movement rates, −50 to +400 ms around move onset)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("fig3_example_direction_tuning.png", dpi=150)
plt.close()
print("saved fig3_example_direction_tuning.png")

# %% [markdown]
# ### Peri-movement rasters and PETHs by direction
#
# Spike rasters and smoothed perievent time histograms aligned to movement onset, split by
# reach-direction bin, for two example units.

# %%
PRE, POST, BIN = 0.4, 0.8, 0.01
nbins = int(round((PRE + POST) / BIN))
ep = nap.IntervalSet(start=onset_t - PRE, end=onset_t + POST)
counts = units.count(BIN, ep=ep)
C3 = counts.values.reshape(len(onset_t), nbins, n_units)  # (trials, time, units)
t_ax = np.linspace(-PRE + BIN / 2, POST - BIN / 2, nbins)
bin_color = {b: plt.cm.hsv((bin_centers[b] + np.pi) / (2 * np.pi)) for b in populated}

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True,
                         gridspec_kw=dict(height_ratios=[1, 1.2], hspace=0.08))
for col, u in enumerate(chosen[:2]):
    ax_r, ax_p = axes[0, col], axes[1, col]
    row = 0
    for b in populated:
        for tr in np.where(dir_bin == b)[0][:25]:
            spk = np.where(C3[tr, :, u] > 0)[0]
            ax_r.scatter(t_ax[spk], np.full(spk.shape, row, dtype=float),
                         s=1, color=bin_color[b], marker="|")
            row += 1
        ax_r.axhline(row - 0.5, color="gray", lw=0.3, alpha=0.5)
    ax_r.axvline(0, color="k", ls="--", lw=1)
    ax_r.set_ylabel("trials (by direction)")
    ax_r.set_title(f"unit {unit_ids[u]} (PD={np.degrees(tun_move['pd'][u]):.0f}°)", fontsize=10)
    ax_r.set_ylim(-0.5, row - 0.5)
    for b in populated:
        peth = gaussian_filter1d(C3[dir_bin == b][:, :, u].mean(axis=0) / BIN, 3)
        ax_p.plot(t_ax, peth, color=bin_color[b], lw=1.2,
                  label=f"{int(round(np.degrees(bin_centers[b])))}°")
    ax_p.axvline(0, color="k", ls="--", lw=1)
    ax_p.set_xlabel("time from move onset (s)")
    if col == 0:
        ax_p.set_ylabel("firing rate (Hz)")
    ax_p.legend(fontsize=7, title="direction", title_fontsize=7, ncol=2, loc="upper right")
fig.suptitle("Peri-movement spiking by reach direction (raster: 25 trials per direction bin)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("fig4_psth_by_direction.png", dpi=150)
plt.close()
print("saved fig4_psth_by_direction.png")

# %% [markdown]
# ### Population summary of direction tuning
#
# Preferred directions cover the workspace; PDs are modestly consistent between the delay
# and movement epochs; and the population response separates preferred from anti-preferred
# reaches well before movement onset (the go cue precedes onset by the reaction time).

# %%
# population PETH: preferred vs anti-preferred direction
pop_pref, pop_anti = [], []
for u in np.where(sig_move)[0]:
    d_ang = np.angle(np.exp(1j * (trials["reach_dir"].values - tun_move["pd"][u])))
    pref_tr = np.abs(d_ang) < np.pi / 4
    anti_tr = np.abs(d_ang) > 3 * np.pi / 4
    if pref_tr.sum() < 5 or anti_tr.sum() < 5:
        continue
    pop_pref.append(gaussian_filter1d(C3[pref_tr][:, :, u].mean(axis=0) / BIN, 3))
    pop_anti.append(gaussian_filter1d(C3[anti_tr][:, :, u].mean(axis=0) / BIN, 3))
pop_pref, pop_anti = np.array(pop_pref), np.array(pop_anti)

def circ_corr(a, b):
    """Fisher-Lee circular correlation."""
    sa = np.sin(a[:, None] - a[None, :]); sb = np.sin(b[:, None] - b[None, :])
    iu = np.triu_indices(len(a), 1)
    return (sa[iu] * sb[iu]).sum() / np.sqrt((sa[iu] ** 2).sum() * (sb[iu] ** 2).sum())

fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

axa = fig.add_subplot(gs[0, 0], projection="polar")
pd_m = tun_move["pd"][sig_move]
bins = np.linspace(-np.pi, np.pi, 17)
cnts, _ = np.histogram(pd_m, bins=bins)
axa.bar((bins[:-1] + bins[1:]) / 2, cnts, width=np.diff(bins),
        color="steelblue", edgecolor="k", lw=0.5)
axa.set_theta_zero_location("E"); axa.set_theta_direction(1)
axa.set_title(f"Preferred directions, movement epoch\n(n={sig_move.sum()} tuned units)",
              fontsize=10, pad=24)
axa.tick_params(labelsize=8)

axb = fig.add_subplot(gs[0, 1])
both = sig_move & sig_delay
d = np.angle(np.exp(1j * (tun_move["pd"][both] - tun_delay["pd"][both])))
axb.hist(np.degrees(d), bins=np.arange(-180, 181, 20), color="seagreen", edgecolor="k", lw=0.5)
axb.axvline(0, color="k", ls="--", lw=1)
axb.set_xlabel("PD(movement) − PD(delay) (deg)"); axb.set_ylabel("units")
axb.set_title(f"PD stability across epochs (n={both.sum()} tuned in both)\n"
              f"circular corr = {circ_corr(tun_delay['pd'][both], tun_move['pd'][both]):.2f}",
              fontsize=10)

axc = fig.add_subplot(gs[0, 2])
axc.hist(tun_move["mod_depth"][sig_move], bins=20, alpha=0.7, color="steelblue",
         label=f"movement (n={sig_move.sum()})")
axc.hist(tun_delay["mod_depth"][sig_delay], bins=20, alpha=0.7, color="darkorange",
         label=f"delay (n={sig_delay.sum()})")
axc.set_xlabel("modulation depth (cosine amp / baseline)"); axc.set_ylabel("units")
axc.set_title("Direction tuning strength", fontsize=10)
axc.legend(fontsize=8)

axd = fig.add_subplot(gs[1, 0])
fracs = [sig_delay.mean(), sig_move.mean(), (sig_delay & sig_move).mean()]
axd.bar(["delay", "movement", "both"], [f * 100 for f in fracs],
        color=["darkorange", "steelblue", "slategray"], edgecolor="k", lw=0.5)
for i, f in enumerate(fracs):
    axd.text(i, f * 100 + 1, f"{f*100:.0f}%", ha="center", fontsize=10)
axd.set_ylabel(f"% of {n_units} units"); axd.set_ylim(0, 90)
axd.set_title("Fraction significantly direction-tuned\n(ANOVA & shuffle p<0.01)", fontsize=10)

axe = fig.add_subplot(gs[1, 1])
m_p, s_p = pop_pref.mean(axis=0), pop_pref.std(axis=0) / np.sqrt(len(pop_pref))
m_a, s_a = pop_anti.mean(axis=0), pop_anti.std(axis=0) / np.sqrt(len(pop_anti))
axe.plot(t_ax, m_p, color="crimson", lw=1.5, label="preferred dir. (±45°)")
axe.fill_between(t_ax, m_p - s_p, m_p + s_p, color="crimson", alpha=0.25)
axe.plot(t_ax, m_a, color="navy", lw=1.5, label="anti-preferred dir. (±45°)")
axe.fill_between(t_ax, m_a - s_a, m_a + s_a, color="navy", alpha=0.25)
axe.axvline(0, color="k", ls="--", lw=1)
axe.set_xlabel("time from move onset (s)"); axe.set_ylabel("firing rate (Hz)")
axe.set_title(f"Population PETH (n={len(pop_pref)} tuned units, mean ± sem)", fontsize=10)
axe.legend(fontsize=8)

axf = fig.add_subplot(gs[1, 2])
axf.hist(tun_move["r2"][sig_move], bins=20, color="steelblue", edgecolor="k", lw=0.5)
axf.set_xlabel("cosine fit R² (single-trial rates)"); axf.set_ylabel("units")
axf.set_title(f"Cosine fit quality, movement-tuned units\n"
              f"median R²={np.median(tun_move['r2'][sig_move]):.2f}", fontsize=10)

fig.suptitle("Population summary: reach-direction tuning in macaque M1/PMd (MC_Maze, DANDI 000128)",
             fontsize=12)
plt.savefig("fig5_population_direction.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig5_population_direction.png")

# %% [markdown]
# ## Velocity and speed tuning
#
# During movement epochs (move onset to +600 ms) we compute, with pynapple's
# `compute_tuning_curves`:
#
# - **Speed tuning curves**: firing rate vs hand speed (occupancy-corrected).
# - **Velocity-direction tuning curves**: firing rate vs the angle of the instantaneous
#   velocity vector; the preferred direction from these should agree with the trial-based
#   reach-direction PD.
#
# We also compute the cross-correlation between each unit's smoothed firing rate and hand
# speed over ±300 ms lags to estimate the neural-kinematic delay (positive lag = spikes
# lead speed).

# %%
MOVE_EPOCH = nap.IntervalSet(start=onset_t, end=onset_t + 0.6)
print("loading full hand velocity ...")
hv_t, hv_v = hv.t, hv.values
speed = np.linalg.norm(hv_v, axis=1)
vel_ang = np.arctan2(hv_v[:, 1], hv_v[:, 0])
speed_tsd = nap.Tsd(t=hv_t, d=speed)
ang_tsd = nap.Tsd(t=hv_t, d=vel_ang)

speed_max = np.percentile(speed, 99)
tc_speed = nap.compute_tuning_curves(units, speed_tsd, bins=12, range=(0, speed_max),
                                     epochs=MOVE_EPOCH, feature_names=["speed"])
tc_vdir = nap.compute_tuning_curves(units, ang_tsd, bins=12, range=(-np.pi, np.pi),
                                    epochs=MOVE_EPOCH, feature_names=["vel_dir"])
speed_centers = tc_speed.speed.values
vdir_centers = tc_vdir.vel_dir.values
tc_speed = np.asarray(tc_speed)  # (n_units, 12)
tc_vdir = np.asarray(tc_vdir)

# preferred direction from the velocity-direction tuning curve (first Fourier component)
pd_vel = np.angle((tc_vdir * np.exp(1j * vdir_centers[None, :])).sum(axis=1))

# spike-rate x speed cross-correlation
BIN = 0.01
counts = units.count(BIN, ep=MOVE_EPOCH)
spd_binned = speed_tsd.bin_average(BIN, ep=MOVE_EPOCH)
n_b = min(len(counts), len(spd_binned))
Cs = gaussian_filter1d(counts.values[:n_b].astype(float), sigma=5, axis=0) / BIN
S = spd_binned.values[:n_b]
Cz = (Cs - Cs.mean(axis=0)) / Cs.std(axis=0)
Sz = (S - S.mean()) / S.std()
lags = np.arange(-30, 31)  # 10 ms bins
xcorr = np.full((len(lags), n_units), np.nan)
for li, lag in enumerate(lags):
    if lag < 0:
        a, b = Cz[-lag:], Sz[:lag]
    elif lag > 0:
        a, b = Cz[:-lag], Sz[lag:]
    else:
        a, b = Cz, Sz
    xcorr[li] = (a * b[:, None]).mean(axis=0)
peak_r = np.nanmax(xcorr, axis=0)
peak_lag = lags[np.nanargmax(xcorr, axis=0)] * BIN * 1000  # ms; >0 = spikes lead speed
print(f"median peak r: {np.median(peak_r):.3f}; median peak lag: {np.median(peak_lag):.0f} ms")

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# (a) example speed tuning curves
rng_speed = tc_speed.max(axis=1) - tc_speed.min(axis=1)
for u in np.argsort(-rng_speed)[:4]:
    axes[0, 0].plot(speed_centers, tc_speed[u], lw=1.5, label=f"unit {unit_ids[u]}")
axes[0, 0].set_xlabel("hand speed (mm/s)"); axes[0, 0].set_ylabel("firing rate (Hz)")
axes[0, 0].set_title("Example speed tuning curves", fontsize=10)
axes[0, 0].legend(fontsize=8)

# (b) population speed tuning (z-scored per unit)
zn = (tc_speed - tc_speed.mean(axis=1, keepdims=True)) / np.maximum(tc_speed.std(axis=1, keepdims=True), 1e-9)
axes[0, 1].plot(speed_centers, zn.mean(axis=0), color="k", lw=2)
axes[0, 1].fill_between(speed_centers, zn.mean(axis=0) - zn.std(axis=0) / np.sqrt(n_units),
                        zn.mean(axis=0) + zn.std(axis=0) / np.sqrt(n_units),
                        color="gray", alpha=0.4)
axes[0, 1].set_xlabel("hand speed (mm/s)"); axes[0, 1].set_ylabel("z-scored rate")
axes[0, 1].set_title(f"Population speed tuning (n={n_units}, mean ± sem)", fontsize=10)

# (c) example velocity-direction tuning curves
for u in chosen[:3]:
    axes[0, 2].plot(np.degrees(vdir_centers), tc_vdir[u], lw=1.5, label=f"unit {unit_ids[u]}")
axes[0, 2].set_xlabel("velocity direction (deg)"); axes[0, 2].set_ylabel("firing rate (Hz)")
axes[0, 2].set_title("Example velocity-direction tuning", fontsize=10)
axes[0, 2].legend(fontsize=8)

# (d) PD from velocity direction vs PD from reach direction
axd = axes[1, 0]
dpd = np.angle(np.exp(1j * (pd_vel[sig_move] - tun_move["pd"][sig_move])))
axd.hist(np.degrees(dpd), bins=np.arange(-180, 181, 20), color="slateblue",
         edgecolor="k", lw=0.5, alpha=0.8)
axd.axvline(0, color="k", ls="--", lw=1)
axd.set_xlabel("PD(velocity) − PD(reach direction) (deg)"); axd.set_ylabel("units")
axd.set_title(f"Velocity PD vs reach PD (n={sig_move.sum()} tuned)\n"
              f"circular corr = {circ_corr(tun_move['pd'][sig_move], pd_vel[sig_move]):.2f}",
              fontsize=10)

# (e) peak correlation distribution
axe2 = axes[1, 1]
axe2.hist(peak_r, bins=25, color="teal", edgecolor="k", lw=0.5)
axe2.axvline(np.median(peak_r), color="k", ls="--", label=f"median r={np.median(peak_r):.2f}")
axe2.set_xlabel("peak spike-rate × speed correlation"); axe2.set_ylabel("units")
axe2.set_title("Speed modulation of firing rate", fontsize=10)
axe2.legend(fontsize=8)

# (f) peak lag distribution for clearly speed-modulated units
axf2 = axes[1, 2]
modulated = peak_r >= 0.1
axf2.hist(peak_lag[modulated], bins=np.arange(-300, 301, 20), color="purple",
          edgecolor="k", lw=0.5, alpha=0.7)
axf2.axvline(0, color="k", ls="-", lw=1)
axf2.axvline(np.median(peak_lag[modulated]), color="k", ls="--",
             label=f"median={np.median(peak_lag[modulated]):.0f} ms")
axf2.set_xlabel("lag of peak correlation (ms; >0 = spikes lead speed)")
axf2.set_ylabel("units")
axf2.set_title(f"Neural-kinematic lag (n={modulated.sum()} units with r≥0.1)", fontsize=10)
axf2.legend(fontsize=8)

fig.suptitle("Velocity and speed tuning in M1/PMd during reaching", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("fig6_speed_tuning.png", dpi=150)
plt.close()
print("saved fig6_speed_tuning.png")

# %% [markdown]
# ## Poisson GLM encoding of direction and speed (nemos)
#
# As a regression-based complement, we fit a Poisson GLM per unit predicting spike counts
# in 10 ms bins from instantaneous kinematics: [cos θ_v, sin θ_v, speed] where θ_v is the
# velocity direction. Models are fit on the dataset's own `train` trials and evaluated on
# the held-out `val` trials with McFadden-style pseudo-R². Ablations (direction-only,
# speed-only) quantify each signal's unique contribution.
#
# Population summaries are restricted to units with movement-epoch rates ≥ 2 Hz, because
# pseudo-R² is unstable for near-silent units.

# %%
counts = units.count(BIN, ep=MOVE_EPOCH)
hv_bin = hv.bin_average(BIN, ep=MOVE_EPOCH)
n_b = min(len(counts), len(hv_bin))
C = counts.values[:n_b]
V = hv_bin.values[:n_b]
speed_b = np.linalg.norm(V, axis=1)
vang_b = np.arctan2(V[:, 1], V[:, 0])
X_full = np.column_stack([np.cos(vang_b), np.sin(vang_b), speed_b / 500.0])
nbins_per_trial = int(round(0.6 / BIN))
trial_mask = np.repeat((trials["split"].values == "train"), nbins_per_trial)[:n_b]
print(f"{n_b} bins, train fraction {trial_mask.mean():.2f}")

def poisson_ll(y, mu):
    return poisson.logpmf(y, np.clip(mu, 1e-9, None)).sum()

ll = {k: np.zeros(n_units) for k in ["full", "dir", "spd", "null"]}
for u in tqdm(range(n_units), desc="GLM fits"):
    y = C[:, u].astype(float)
    ytr, yte = y[trial_mask], y[~trial_mask]
    ll["null"][u] = poisson_ll(yte, np.full_like(yte, ytr.mean()))
    for k, X in [("full", X_full), ("dir", X_full[:, :2]), ("spd", X_full[:, 2:3])]:
        m = nmo.glm.GLM(solver_name="LBFGS")
        init = (np.zeros(X.shape[1]), np.array([np.log(np.clip(ytr.mean(), 1e-6, None))]))
        m.fit(X[trial_mask], ytr, init_params=init)
        ll[k][u] = poisson_ll(yte, np.asarray(m.predict(X[~trial_mask])))

pr2 = {k: 1 - ll[k] / ll["null"] for k in ["full", "dir", "spd"]}
dll_dir = ll["full"] - ll["spd"]   # unique direction contribution
dll_spd = ll["full"] - ll["dir"]   # unique speed contribution
mean_rate = C.mean(axis=0) / BIN
rate_ok = mean_rate >= 2.0
print(f"units with rate >= 2 Hz: {rate_ok.sum()}/{n_units}")
print(f"median held-out pseudo-R2 (rate>=2Hz): full={np.median(pr2['full'][rate_ok]):.4f}, "
      f"dir={np.median(pr2['dir'][rate_ok]):.4f}, spd={np.median(pr2['spd'][rate_ok]):.4f}")
print(f"units better than null (full model): {(pr2['full'][rate_ok] > 0).mean()*100:.0f}%")

# refit the best active unit for the example trace
best = int(np.argmax(np.where(rate_ok, pr2["full"], -np.inf)))
y = C[:, best].astype(float)
m = nmo.glm.GLM(solver_name="LBFGS")
init = (np.zeros(3), np.array([np.log(np.clip(y[trial_mask].mean(), 1e-6, None))]))
m.fit(X_full[trial_mask], y[trial_mask], init_params=init)
mu_best = np.asarray(m.predict(X_full[~trial_mask])) / BIN
yte_best = y[~trial_mask] / BIN

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

win = 400
s0 = int(np.argmax(np.convolve(yte_best, np.ones(win), mode="valid")))
seg = slice(s0, s0 + win)
tt = np.arange(win) * BIN
axes[0].plot(tt, gaussian_filter1d(yte_best[seg], 5), color="k", lw=1.2, label="actual (smoothed)")
axes[0].plot(tt, mu_best[seg], color="crimson", lw=1.2, label="GLM prediction")
axes[0].set_xlabel("time within held-out trials (s)"); axes[0].set_ylabel("firing rate (Hz)")
axes[0].set_title(f"Example encoding: unit {unit_ids[best]} "
                  f"(held-out pseudo-R²={pr2['full'][best]:.2f})", fontsize=10)
axes[0].legend(fontsize=8)

axes[1].scatter(dll_spd[rate_ok], dll_dir[rate_ok], s=12, alpha=0.6,
                color="slateblue", edgecolor="none")
lim = np.nanpercentile(np.abs(np.concatenate([dll_spd[rate_ok], dll_dir[rate_ok]])), 98)
axes[1].plot([-lim, lim], [-lim, lim], color="k", ls="--", lw=1)
axes[1].axhline(0, color="gray", lw=0.5); axes[1].axvline(0, color="gray", lw=0.5)
axes[1].set_xlabel("unique speed contribution (Δ log-likelihood)")
axes[1].set_ylabel("unique direction contribution (Δ log-likelihood)")
n_dir = (dll_dir[rate_ok] > dll_spd[rate_ok]).sum()
axes[1].set_title(f"Direction vs speed encoding per unit (rate ≥ 2 Hz)\n"
                  f"(n={n_dir}/{rate_ok.sum()} units: direction > speed)", fontsize=10)

meds = [np.median(pr2[k][rate_ok]) for k in ["dir", "spd", "full"]]
q25 = [np.percentile(pr2[k][rate_ok], 25) for k in ["dir", "spd", "full"]]
q75 = [np.percentile(pr2[k][rate_ok], 75) for k in ["dir", "spd", "full"]]
axes[2].bar(["direction\nonly", "speed\nonly", "direction\n+ speed"], meds,
            yerr=[np.array(meds) - np.array(q25), np.array(q75) - np.array(meds)],
            color=["steelblue", "teal", "crimson"], edgecolor="k", lw=0.5, capsize=4)
axes[2].set_ylabel("held-out pseudo-R² (median, IQR)")
axes[2].set_title("Poisson GLM encoding of kinematics\n(10 ms bins, movement epochs)", fontsize=10)

plt.tight_layout()
plt.savefig("fig7_glm_encoding.png", dpi=150)
plt.close()
print("saved fig7_glm_encoding.png")

# %% [markdown]
# ## Summary
#
# - **Direction tuning is widespread**: ~71% of the 182 M1/PMd units are significantly
#   direction-tuned during movement execution (ANOVA + shuffle, p<0.01), and ~43% already
#   show tuning during the delay period, consistent with motor preparation.
# - **Cosine tuning**: example units show clean cosine-shaped tuning; across tuned units
#   the median modulation depth is ~0.5 (cosine amplitude / baseline rate). Single-trial
#   cosine R² is modest (median ~0.08), as expected for curved reaches and single-trial rates.
# - **Speed tuning**: firing rates increase with hand speed (population z-scored tuning
#   rises from rest and saturates), and the spike-rate/speed cross-correlation peaks at
#   positive lags (median +35 ms across all units, +30 ms among units with r ≥ 0.1),
#   i.e. neural activity leads the kinematics, consistent with a causal role in movement
#   generation.
# - **Velocity direction agrees with reach direction**: PDs estimated from instantaneous
#   velocity-direction tuning curves align with trial-based reach-direction PDs.
# - **GLM encoding**: Poisson GLMs using velocity direction + speed beat the null model on
#   held-out trials for the large majority of active units; direction and speed each carry
#   unique information, with their combination best.
