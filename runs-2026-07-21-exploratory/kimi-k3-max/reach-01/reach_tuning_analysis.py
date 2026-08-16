# %% [markdown]
# # Reach Direction and Velocity Tuning in Macaque Motor Cortex
#
# **Dataset:** [DANDI:000128](https://dandiarchive.org/dandiset/000128): *MC_Maze: macaque primary
# motor and dorsal premotor cortex spiking activity during delayed reaching* (Churchland & Shenoy labs,
# Neural Latents Benchmark '21). Subject Jenkins, one full session: 182 sorted units (86 M1, 96 PMd),
# 2295 successful delayed-reach trials, hand position/velocity at 1 kHz.
#
# **Question:** Are motor-cortical neurons tuned for reach **direction** (Georgopoulos-style cosine
# tuning) and for hand **velocity/speed**, and does neural activity lead the movement?
#
# **Approach:**
# 1. Stream the NWB file from DANDI with `remfile` (no full download) and load with `pynapple`.
# 2. Direction tuning: firing rate vs cued target direction in the movement and delay epochs;
#    preferred direction (vector sum), modulation depth, one-way ANOVA, cosine fit.
# 3. Speed/velocity tuning: rate vs hand speed conditional on moving toward/away from the preferred
#    direction; rate–speed cross-correlation lag analysis.
# 4. Encoding model: Poisson GLM (`nemos`) of spike counts from basis-expanded lagged hand velocity,
#    compared against a speed-only model with blocked cross-validation.
#
# All data are streamed from the DANDI Archive; nothing is simulated.

# %% [markdown]
# ## 1. Setup and streaming data access

# %%
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless run: save figures, never plt.show()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.special import gammaln
import pickle, os

import remfile
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# DANDI:000128 draft, asset sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb
ASSET_URL = ("https://api.dandiarchive.org/api/dandisets/000128/versions/draft/"
             "assets/26e85f09-39b7-480f-b337-278a8f034007/download/")

disk_cache = remfile.DiskCache("/tmp/remfile_cache_mcmaze")   # local chunk cache
rem_file = remfile.File(ASSET_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# The file contains a `TsGroup` of 182 units, an `IntervalSet` of 2295 trials (with target
# positions, go-cue and movement-onset times), and 1 kHz behavioral series (`hand_pos`,
# `hand_vel`, `cursor_pos`, `eye_pos`). Note on units: the NWB `unit` attributes say meters
# and m/s, but the magnitudes (workspace ±130, peak speed ~800) show the stored values are
# actually mm and mm/s; we convert to cm and cm/s for display.

# %% [markdown]
# ## 2. Task structure and behavior
#
# Each trial presents one cued target (plus distractors in some trials) among virtual barriers;
# after a delay the go cue signals the reach. We extract the cued target per trial and its angle.

# %%
trials_df = nwbfile.intervals["trials"].to_dataframe()
target_xy = np.array([row["target_pos"][row["active_target"]] for _, row in trials_df.iterrows()],
                     dtype=float)
tr = dict(
    start=trials_df["start_time"].values.astype(float),
    stop=trials_df["stop_time"].values.astype(float),
    go_cue=trials_df["go_cue_time"].values.astype(float),
    move_onset=trials_df["move_onset_time"].values.astype(float),
    target_on=trials_df["target_on_time"].values.astype(float),
    rt=trials_df["rt"].values.astype(float),
    delay=trials_df["delay"].values.astype(float),
    target_xy=target_xy,
)
tr["target_angle"] = (np.degrees(np.arctan2(target_xy[:, 1], target_xy[:, 0])) + 360) % 360
n_trials = len(tr["start"])
print(f"{n_trials} trials; RT median {np.median(tr['rt']):.0f} ms; "
      f"delay median {np.median(tr['delay']):.0f} ms")

# Units: spike times, array (M1/PMd), NLB held-out flag
units_df = nwbfile.units.to_dataframe()
elec_tbl = nwbfile.electrodes.to_dataframe()
elec_of_unit = np.asarray(nwbfile.units["electrodes"].data[:])
id2row = {eid: i for i, eid in enumerate(elec_tbl.index.values)}
unit_group = np.array([g.replace("electrode_group_", "")
                       for g in elec_tbl["group_name"].values[[id2row[e] for e in elec_of_unit]]])
un = dict(
    spike_times=[np.asarray(st, dtype=float) for st in units_df["spike_times"].values],
    group=unit_group,
    heldout=units_df["heldout"].values.astype(bool),
    unit_ids=units_df.index.values.astype(int),
)
n_units = len(un["spike_times"])
print(f"{n_units} units: {(unit_group=='M1').sum()} M1, {(unit_group=='PMd').sum()} PMd")

# Behavior at 1 kHz (stored as mm / mm/s despite the meter attrs; convert to cm / cm/s)
beh = nwbfile.processing["behavior"]
t_beh = beh["hand_vel"].timestamps[:]
pos = beh["hand_pos"].data[:].astype(np.float64)
vel = beh["hand_vel"].data[:].astype(np.float64) / 10.0      # -> cm/s
speed = np.hypot(vel[:, 0], vel[:, 1])
print(f"behavior: {len(t_beh)} samples at ~{1/np.median(np.diff(t_beh[:2000])):.0f} Hz, "
      f"session {t_beh[-1]/3600:.2f} h, peak speed {np.percentile(speed, 99.9):.0f} cm/s")

# %% [markdown]
# ### Figure 1: task and behavior overview

# %%
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.3)
cmap = plt.cm.hsv
ang = tr["target_angle"]

# A: example trajectories + targets
ax = fig.add_subplot(gs[0, 0])
rng = np.random.default_rng(0)
for i in rng.choice(n_trials, 60, replace=False):
    m = (t_beh >= tr["move_onset"][i] - 0.05) & (t_beh <= tr["move_onset"][i] + 0.5)
    ax.plot(pos[m, 0], pos[m, 1], color=cmap(ang[i]/360), lw=0.7, alpha=0.6)
ax.scatter(target_xy[:, 0], target_xy[:, 1], c=ang/360, cmap="hsv", s=14,
           edgecolors="k", linewidths=0.3, zorder=5)
ax.scatter([0], [0], marker="+", c="k", s=80)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_aspect("equal")
ax.set_title("A  Reach trajectories & targets", loc="left", fontsize=11)

# B: example trial speed profile with events
ax = fig.add_subplot(gs[0, 1])
i = 100
m = (t_beh >= tr["start"][i] - 0.2) & (t_beh <= tr["stop"][i] + 0.3)
ax.plot(t_beh[m] - tr["go_cue"][i], speed[m], "k", lw=1.2)
for ev, c, lbl in [(tr["target_on"][i], "tab:blue", "target on"),
                   (tr["go_cue"][i], "tab:green", "go cue"),
                   (tr["move_onset"][i], "tab:red", "move onset")]:
    ax.axvline(ev - tr["go_cue"][i], color=c, ls="--", lw=1)
    ax.text(ev - tr["go_cue"][i] + 0.02, 78, lbl, rotation=90, va="top", fontsize=8, color=c)
ax.set_xlabel("time from go cue (s)"); ax.set_ylabel("hand speed (cm/s)")
ax.set_title("B  Example trial speed profile", loc="left", fontsize=11)

# C: RT and reach-duration distributions
ax = fig.add_subplot(gs[0, 2])
ax.hist(tr["rt"], bins=40, alpha=0.7, label="reaction time", color="tab:blue")
ax.hist((tr["stop"] - tr["move_onset"])*1000, bins=40, alpha=0.7, label="reach duration",
        color="tab:red")
ax.set_xlabel("duration (ms)"); ax.set_ylabel("trials"); ax.legend(fontsize=9)
ax.set_title("C  Trial timing", loc="left", fontsize=11)

# D: mean speed aligned to move onset
ax = fig.add_subplot(gs[1, 0])
win = np.arange(-0.3, 0.8, 0.001)
prof = np.full((n_trials, len(win)), np.nan)
for i in range(n_trials):
    m0 = np.searchsorted(t_beh, tr["move_onset"][i] + win[0])
    if m0 + len(win) <= len(t_beh):
        prof[i] = speed[m0:m0+len(win)]
ax.plot(win, np.nanmean(prof, 0), "k", lw=1.5)
sem = np.nanstd(prof, 0)/np.sqrt(n_trials)
ax.fill_between(win, np.nanmean(prof, 0)-sem, np.nanmean(prof, 0)+sem, alpha=0.3)
ax.axvline(0, color="tab:red", ls="--", lw=1)
ax.set_xlabel("time from move onset (s)"); ax.set_ylabel("hand speed (cm/s)")
ax.set_title("D  Speed aligned to move onset", loc="left", fontsize=11)

# E: spike raster + kinematics snippet
ax = fig.add_subplot(gs[1, 1:])
s0, s1 = 600.0, 608.0
order = np.argsort([len(s) for s in un["spike_times"]])[::-1][:25]
for k, u in enumerate(order):
    st = un["spike_times"][u]
    st = st[(st >= s0) & (st < s1)]
    ax.vlines(st, k, k+0.8, color="k", lw=0.5)
m = (t_beh >= s0) & (t_beh < s1)
sc = 0.12
ax.plot(t_beh[m], -3 + vel[m, 0]*sc, color="tab:blue", lw=1, label="vx")
ax.plot(t_beh[m], -9 + vel[m, 1]*sc, color="tab:orange", lw=1, label="vy")
ax.plot(t_beh[m], -16 + speed[m]*sc, color="tab:red", lw=1.2, label="speed")
for s_ in tr["start"]:
    if s0 <= s_ < s1: ax.axvline(s_, color="gray", ls=":", lw=0.7)
for mo in tr["move_onset"]:
    if s0 <= mo < s1: ax.axvline(mo, color="tab:red", ls=":", lw=0.7)
ax.set_yticks([]); ax.set_xlabel("time (s)"); ax.set_xlim(s0, s1)
ax.legend(loc="lower right", fontsize=8, ncol=3, framealpha=0.9)
ax.set_title("E  Spike raster (25 units) with hand velocity (gray: trial start, red: move onset)",
             loc="left", fontsize=11)

fig.suptitle("fig01  MC_Maze (DANDI 000128, sub-Jenkins): delayed reaching task overview", fontsize=13)
fig.savefig("figures/fig01_task_behavior.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# Reaches curve around the virtual barriers (A), speed is bell-shaped peaking ~200 ms after
# move onset (D), and the raster already shows direction-dependent modulation (E).

# %% [markdown]
# ## 3. Direction tuning
#
# Classic Georgopoulos analysis: for each unit, mean firing rate per target direction in two
# epochs: the **movement epoch** (move onset −50 ms to +400 ms) and the **delay epoch**
# (last 300 ms before the go cue, trials with delay ≥ 350 ms). Target angles are grouped into
# 30° bins (10 of 12 bins are occupied; the task has no straight-down targets). Per unit we
# compute the preferred direction (rate-weighted circular mean), modulation depth
# (max − min rate across bins), a one-way ANOVA across direction bins, and a cosine fit
# r(θ) = b0 + A·cos(θ − PD).

# %%
BIN_CENTERS = np.arange(0, 360, 30)
bin_id = np.floor(((tr["target_angle"] + 15) % 360) / 30).astype(int)
occupied = np.array([b for b in range(12) if (bin_id == b).sum() >= 20])
WIN_MOVE = (-0.05, 0.40)
WIN_DELAY = (-0.30, 0.0)
delay_ok = tr["delay"] >= 350

def window_rates(spk, events, win):
    lo, hi = win
    i0 = np.searchsorted(spk, events + lo)
    i1 = np.searchsorted(spk, events + hi)
    return (i1 - i0) / (hi - lo)

def circ_pd(angles_deg, rates):
    th = np.deg2rad(angles_deg)
    z = np.sum(rates * np.exp(1j*th))
    return (np.rad2deg(np.angle(z)) % 360), np.abs(z)/np.sum(rates)

def cosine_fit(angles_deg, rates):
    th = np.deg2rad(angles_deg)
    X = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
    beta, *_ = np.linalg.lstsq(X, rates, rcond=None)
    pred = X @ beta
    ss_res = np.sum((rates-pred)**2); ss_tot = np.sum((rates-rates.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return beta[0], np.hypot(beta[1], beta[2]), np.rad2deg(np.arctan2(beta[2], beta[1])) % 360, r2

STAT_KEYS = ["pd_move", "R_move", "depth_move", "anova_p_move", "cos_r2_move", "cos_pd_move",
             "pd_delay", "R_delay", "depth_delay", "anova_p_delay", "mean_rate_move",
             "mean_rate_delay"]
res = {k: np.full(n_units, np.nan) for k in STAT_KEYS}
bin_means = np.full((n_units, 12), np.nan)

for u in range(n_units):
    spk = un["spike_times"][u]
    r_move = window_rates(spk, tr["move_onset"], WIN_MOVE)
    r_delay = window_rates(spk, tr["go_cue"], WIN_DELAY)
    res["mean_rate_move"][u] = r_move.mean()
    res["mean_rate_delay"][u] = r_delay[delay_ok].mean()
    for tag, r, mask in [("move", r_move, np.ones(n_trials, bool)), ("delay", r_delay, delay_ok)]:
        bm = np.full(12, np.nan)
        for b in occupied:
            bm[b] = r[(bin_id == b) & mask].mean()
        if tag == "move":
            bin_means[u] = bm
        vals = bm[occupied]
        if np.all(vals == 0) or np.isnan(vals).any():
            continue
        pd_, R_ = circ_pd(BIN_CENTERS[occupied], vals)
        res[f"pd_{tag}"][u] = pd_
        res[f"R_{tag}"][u] = R_
        res[f"depth_{tag}"][u] = vals.max() - vals.min()
        _, p = stats.f_oneway(*[r[(bin_id == b) & mask] for b in occupied])
        res[f"anova_p_{tag}"][u] = p
        if tag == "move":
            _, _, cpd, r2 = cosine_fit(BIN_CENTERS[occupied], vals)
            res["cos_r2_move"][u] = r2
            res["cos_pd_move"][u] = cpd

sig_move = res["anova_p_move"] < 0.01
sig_delay = res["anova_p_delay"] < 0.01
print(f"direction-tuned during movement (ANOVA p<0.01): {sig_move.sum()}/{n_units}")
print(f"direction-tuned during delay:                   {np.nansum(sig_delay)}/{n_units}")
for g in ["M1", "PMd"]:
    m = un["group"] == g
    print(f"  {g}: move {np.sum(sig_move & m)}/{m.sum()}, delay {np.nansum(sig_delay & m)}/{m.sum()}")

# %% [markdown]
# ### Figure 2: example unit: rasters and PETHs by reach direction

# %%
bincolors = {b: cmap(BIN_CENTERS[b]/360) for b in occupied}
score_ex = res["depth_move"] * np.sqrt(res["mean_rate_move"]) * res["cos_r2_move"]
ex = int(np.nanargmax(score_ex))
spk = un["spike_times"][ex]
mo = tr["move_onset"]
T0, T1, bin_s = -0.3, 0.6, 0.01
edges = np.arange(T0, T1+bin_s, bin_s)
ct = (edges[:-1]+edges[1:])/2

fig = plt.figure(figsize=(15, 9))
for k, b in enumerate(occupied[np.argsort(BIN_CENTERS[occupied])]):
    ax = fig.add_subplot(3, 4, k+1)
    tb = np.where(bin_id == b)[0]
    for j, ti in enumerate(tb):
        s = spk[(spk >= mo[ti]+T0) & (spk < mo[ti]+T1)] - mo[ti]
        ax.vlines(s, j+0.5, j+1.5, color=bincolors[b], lw=0.4)
    counts_p = np.zeros(len(edges)-1)
    for ti in tb:
        s = spk[(spk >= mo[ti]+T0) & (spk < mo[ti]+T1)] - mo[ti]
        counts_p += np.histogram(s, edges)[0]
    peth = gaussian_filter1d(counts_p/len(tb)/bin_s, sigma=2)
    ax2 = ax.twinx()
    ax2.plot(ct, peth, "k", lw=1.5)
    ax2.set_ylim(0, np.nanmax(peth)*1.3+1); ax2.set_yticks([])
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlim(T0, T1); ax.set_ylim(0.5, len(tb)+0.5); ax.set_yticks([])
    if k >= 6:
        ax.set_xlabel("time from move onset (s)", fontsize=9)
    ax.set_title(f"{BIN_CENTERS[b]}°  (n={len(tb)})", fontsize=10, color=bincolors[b])
fig.suptitle(f"fig02  Unit {un['unit_ids'][ex]} ({un['group'][ex]}): rasters & PETHs by reach direction",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/fig02_direction_rasters.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 3: polar tuning curves for the 12 most modulated units

# %%
top = np.argsort(np.nan_to_num(score_ex))[::-1][:12]
fig = plt.figure(figsize=(13, 10))
for k, u in enumerate(top):
    ax = fig.add_subplot(3, 4, k+1, projection="polar")
    vals = bin_means[u][occupied]
    th = np.deg2rad(BIN_CENTERS[occupied])
    th_c = np.concatenate([th, th[:1]]); v_c = np.concatenate([vals, vals[:1]])
    ax.plot(th_c, v_c, "o-", color="tab:blue", lw=1.5, ms=4)
    pd_ = res["pd_move"][u]
    ax.plot([np.deg2rad(pd_)]*2, [0, vals.max()*1.1], color="tab:red", lw=1.5, ls="--")
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_title(f"u{un['unit_ids'][u]} {un['group'][u]}\nPD={pd_:.0f}°, depth={res['depth_move'][u]:.0f} sp/s",
                 fontsize=9, pad=22)
    ax.set_rlabel_position(0); ax.tick_params(labelsize=7)
fig.suptitle("fig03  Movement-period direction tuning (polar): top 12 units by modulation", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/fig03_tuning_curves_examples.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 4: population summary of direction tuning

# %%
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
bins = np.arange(0, 361, 20)
for g, c in [("M1", "tab:red"), ("PMd", "tab:blue")]:
    m = sig_move & (un["group"] == g)
    ax.hist(res["pd_move"][m], bins=bins, alpha=0.6, color=c, label=f"{g} (n={m.sum()})")
ax.set_xlabel("preferred direction (deg)"); ax.set_ylabel("units"); ax.legend(fontsize=9)
ax.set_title("A  Preferred direction distribution (movement)", loc="left", fontsize=11)

ax = fig.add_subplot(gs[0, 1])
ax.hist(res["depth_move"][sig_move], bins=30, color="tab:blue", alpha=0.75,
        label=f"sig (n={sig_move.sum()})")
ax.hist(res["depth_move"][~sig_move], bins=30, color="gray", alpha=0.6,
        label=f"not sig (n={(~sig_move).sum()})")
ax.set_xlabel("modulation depth (sp/s)"); ax.set_ylabel("units"); ax.legend(fontsize=9)
ax.set_title("B  Direction modulation depth", loc="left", fontsize=11)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(res["depth_move"][sig_move], res["cos_r2_move"][sig_move], s=12, alpha=0.6, c="tab:blue")
ax.scatter(res["depth_move"][~sig_move], res["cos_r2_move"][~sig_move], s=12, alpha=0.6, c="gray")
ax.set_xlabel("modulation depth (sp/s)"); ax.set_ylabel("cosine fit R²")
ax.set_title("C  Cosine tuning fit quality", loc="left", fontsize=11)

ax = fig.add_subplot(gs[1, 0])
okd = ~np.isnan(res["depth_delay"])
ax.scatter(res["depth_delay"][okd], res["depth_move"][okd], s=12, alpha=0.5,
           c=["tab:red" if g == "M1" else "tab:blue" for g in un["group"][okd]])
lim = np.nanmax(np.concatenate([res["depth_delay"], res["depth_move"]]))
ax.plot([0, lim], [0, lim], "k--", lw=0.8)
ax.set_xlabel("delay-period depth (sp/s)"); ax.set_ylabel("movement-period depth (sp/s)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color="tab:red", label="M1"),
                   Line2D([], [], marker="o", ls="", color="tab:blue", label="PMd")], fontsize=9)
ax.set_title("D  Planning vs execution modulation", loc="left", fontsize=11)

ax = fig.add_subplot(gs[1, 1])
both = sig_move & sig_delay
d_pd = (res["pd_move"][both] - res["pd_delay"][both] + 180) % 360 - 180
ax.hist(d_pd, bins=np.arange(-180, 181, 20), color="tab:green", alpha=0.8)
ax.set_xlabel("PD(move) − PD(delay), wrapped (deg)"); ax.set_ylabel("units")
ax.set_title(f"E  PD stability across epochs (n={both.sum()})", loc="left", fontsize=11)

ax = fig.add_subplot(gs[1, 2])
u = int(np.nanargmax(res["cos_r2_move"] * (res["depth_move"] > 10)))
th = np.deg2rad(BIN_CENTERS[occupied])
X = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
beta, *_ = np.linalg.lstsq(X, bin_means[u][occupied], rcond=None)
thf = np.linspace(0, 2*np.pi, 200)
ax.plot(np.rad2deg(thf), beta[0] + beta[1]*np.cos(thf) + beta[2]*np.sin(thf), "k-", lw=1.5,
        label="cosine fit")
ax.plot(BIN_CENTERS[occupied], bin_means[u][occupied], "o", color="tab:blue", label="data")
ax.set_xlabel("reach direction (deg)"); ax.set_ylabel("firing rate (sp/s)"); ax.legend(fontsize=9)
ax.set_title(f"F  Cosine fit, unit {un['unit_ids'][u]} (R²={res['cos_r2_move'][u]:.2f})",
             loc="left", fontsize=11)

fig.suptitle("fig04  Population direction tuning: 182 units (86 M1, 96 PMd), MC_Maze sub-Jenkins",
             fontsize=13)
fig.savefig("figures/fig04_population_direction.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# Nearly every unit (174/182) is significantly direction-tuned during movement, and 129/182
# during the delay period. Preferred directions tile the workspace, tuning is typically
# cosine-like (median fit R² ≈ 0.7 for deeply modulated units), and preferred directions are
# largely stable from planning to execution (median |ΔPD| ≈ 37°).

# %% [markdown]
# ## 4. Speed and velocity tuning
#
# We now bin spikes and hand kinematics into 50 ms bins within trials and ask how firing rate
# depends on instantaneous hand **speed** and **velocity direction**. Because direction tuning
# dominates, speed effects are isolated by conditioning on movement direction relative to each
# unit's preferred direction (toward PD: within ±60°; away: >120°). We also compute the
# rate–speed cross-correlation as a function of lag to estimate the neural–behavioral delay.

# %%
BIN = 0.05
bt_list, vx_l, vy_l, sp_l, tr_l = [], [], [], [], []
for i in range(n_trials):
    e = np.arange(tr["start"][i], tr["stop"][i], BIN)
    if len(e) < 2:
        continue
    c_ = (e[:-1]+e[1:])/2
    bi = np.clip(np.searchsorted(t_beh, c_), 0, len(t_beh)-1)
    bt_list.append(c_); vx_l.append(vel[bi, 0]); vy_l.append(vel[bi, 1]); sp_l.append(speed[bi])
    tr_l.append(np.full(len(c_), i))
bin_times = np.concatenate(bt_list)
vx = np.concatenate(vx_l); vy = np.concatenate(vy_l); spd = np.concatenate(sp_l)
trial_of_bin = np.concatenate(tr_l)
n_bins = len(bin_times)

counts = np.zeros((n_bins, n_units), dtype=np.float32)
left_edges = bin_times - BIN/2
for u in range(n_units):
    spk = un["spike_times"][u]
    bi = np.searchsorted(left_edges, spk, side="right") - 1
    ok = (bi >= 0) & (bi < n_bins) & (spk < left_edges[np.clip(bi, 0, n_bins-1)] + BIN - 1e-9)
    np.add.at(counts[:, u], bi[ok], 1)
rates = counts / BIN
print(f"{n_bins} bins, {int(counts.sum())} spikes placed, mean rate {rates.mean():.2f} sp/s")

# speed tuning conditional on direction relative to PD
vdir = (np.degrees(np.arctan2(vy, vx)) + 360) % 360
fast = spd > 15
dtheta = np.abs(((vdir[None, :] - res["pd_move"][:, None]) + 180) % 360 - 180)
toward = (dtheta <= 60) & fast[None, :]
away = (dtheta >= 120) & fast[None, :]
spd_q = np.quantile(spd[fast], np.linspace(0, 1, 11))
spd_qc = (spd_q[:-1]+spd_q[1:])/2
sq_bin = np.clip(np.searchsorted(spd_q, spd)-1, 0, 9)
rate_toward = np.full((10, n_units), np.nan)
rate_away = np.full((10, n_units), np.nan)
for b in range(10):
    mb = sq_bin == b
    rate_toward[b] = [rates[toward[u] & mb, u].mean() if (toward[u] & mb).sum() > 5 else np.nan
                      for u in range(n_units)]
    rate_away[b] = [rates[away[u] & mb, u].mean() if (away[u] & mb).sum() > 5 else np.nan
                    for u in range(n_units)]
slope_toward = np.array([stats.linregress(spd_qc[~np.isnan(rate_toward[:, u])],
                                          rate_toward[~np.isnan(rate_toward[:, u]), u]).slope
                         if (~np.isnan(rate_toward[:, u])).sum() > 4 else np.nan
                         for u in range(n_units)])
slope_away = np.array([stats.linregress(spd_qc[~np.isnan(rate_away[:, u])],
                                        rate_away[~np.isnan(rate_away[:, u]), u]).slope
                       if (~np.isnan(rate_away[:, u])).sum() > 4 else np.nan
                       for u in range(n_units)])
wilc = stats.wilcoxon((slope_toward - slope_away)[sig_move])
print(f"speed slope toward PD vs away (sig units): Wilcoxon p = {wilc.pvalue:.2e}")

# rate-speed cross-correlation vs lag (z-scored within trials)
lags = np.arange(-0.4, 0.401, BIN)
lag_bins = (lags/BIN).round().astype(int)
spd_z = np.zeros_like(spd)
rate_z = np.zeros_like(rates)
for i in np.unique(trial_of_bin):
    m = trial_of_bin == i
    spd_z[m] = (spd[m]-spd[m].mean())/(spd[m].std()+1e-9)
    rate_z[m] = (rates[m]-rates[m].mean(0))/(rates[m].std(0)+1e-9)
cc = np.zeros((len(lags), n_units))
tri_ids = np.unique(trial_of_bin)
for li, lb in enumerate(lag_bins):
    ps = np.zeros(n_units); ns = 0
    for i in tri_ids:
        m = np.where(trial_of_bin == i)[0]
        a, b = (m[:-lb], m[lb:]) if lb > 0 else (m[-lb:], m[:lb]) if lb < 0 else (m, m)
        if len(a) == 0:
            continue
        ps += (rate_z[a]*spd_z[b][:, None]).sum(0); ns += len(a)
    cc[li] = ps/ns
peak_lag = lags[np.argmax(cc, axis=0)]
peak_cc = cc.max(axis=0)
coupled = sig_move & (peak_cc > 0.05)
print(f"rate-speed coupling: median peak lag {np.median(peak_lag[coupled])*1000:.0f} ms "
      f"(n={coupled.sum()} coupled units; + = neural leads)")

# velocity-direction PD from early-reach bins (0-250 ms after move onset, speed > 10 cm/s)
trel = bin_times - tr["move_onset"][trial_of_bin]
early = (trel >= 0) & (trel <= 0.25) & (spd > 10)
vbin = np.floor(((vdir + 15) % 360)/30).astype(int)
occ_v = np.array([b for b in range(12) if ((vbin == b) & early).sum() >= 30])
vt = np.full((12, n_units), np.nan)
for b in occ_v:
    vt[b] = rates[(vbin == b) & early].mean(0)
th = np.deg2rad(BIN_CENTERS[occ_v])
z = np.nansum(vt[occ_v]*np.exp(1j*th)[:, None], axis=0)
pd_vel = (np.degrees(np.angle(z))+360) % 360
R_vel = np.abs(z)/np.nansum(vt[occ_v], axis=0)
okv = sig_move & (R_vel > 0.1)
d_pv = (res["pd_move"][okv] - pd_vel[okv] + 180) % 360 - 180
print(f"PD from target direction vs early-reach velocity: n={okv.sum()}, "
      f"median |dPD|={np.median(np.abs(d_pv)):.0f}°")

# %% [markdown]
# ### Figure 5: speed and velocity tuning

# %%
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
for u in np.argsort(np.nan_to_num(res["depth_move"]))[::-1][:6]:
    ax.plot(spd_qc, rate_toward[:, u], "-", lw=1.5, alpha=0.8, label=f"u{un['unit_ids'][u]}")
ax.set_xlabel("hand speed (cm/s)"); ax.set_ylabel("firing rate (sp/s)"); ax.legend(fontsize=7)
ax.set_title("A  Rate vs speed, moving toward PD", loc="left", fontsize=11)

ax = fig.add_subplot(gs[0, 1])
norm = np.nanmean(np.concatenate([rate_toward, rate_away], axis=0), axis=0, keepdims=True)
rtn, ran = rate_toward/norm, rate_away/norm
ax.errorbar(spd_qc, np.nanmean(rtn, 1), yerr=np.nanstd(rtn, 1)/np.sqrt(n_units),
            color="tab:red", lw=1.5, label="toward PD (±60°)")
ax.errorbar(spd_qc, np.nanmean(ran, 1), yerr=np.nanstd(ran, 1)/np.sqrt(n_units),
            color="tab:blue", lw=1.5, label="away from PD (>120°)")
ax.set_xlabel("hand speed (cm/s)"); ax.set_ylabel("normalized rate"); ax.legend(fontsize=9)
ax.set_title("B  Population speed modulation by direction", loc="left", fontsize=11)

ax = fig.add_subplot(gs[0, 2])
ax.hist(slope_toward[sig_move]*10, bins=40, alpha=0.7, color="tab:red", label="toward PD")
ax.hist(slope_away[sig_move]*10, bins=40, alpha=0.7, color="tab:blue", label="away from PD")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("speed slope (sp/s per 10 cm/s)"); ax.set_ylabel("units"); ax.legend(fontsize=9)
ax.set_title(f"C  Speed slopes (Wilcoxon p={wilc.pvalue:.1e})", loc="left", fontsize=11)

ax = fig.add_subplot(gs[1, 0])
for u in np.argsort(np.nan_to_num(res["depth_move"]))[::-1][:3]:
    ax.plot(lags*1000, cc[:, u], lw=1, alpha=0.7)
ax.plot(lags*1000, cc[:, sig_move].mean(1), "k", lw=2.5, label="population mean")
ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("lag (ms; + = neural leads hand speed)"); ax.set_ylabel("rate–speed correlation (z)")
ax.legend(fontsize=8)
ax.set_title("D  Rate–speed cross-correlation vs lag", loc="left", fontsize=11)

ax = fig.add_subplot(gs[1, 1])
ax.hist(peak_lag[coupled]*1000, bins=np.arange(-400, 401, 50), color="tab:purple", alpha=0.8)
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.median(peak_lag[coupled])*1000, color="tab:red", ls="--", lw=1.2,
           label=f"median {np.median(peak_lag[coupled])*1000:.0f} ms (n={coupled.sum()})")
ax.set_xlabel("peak lag (ms)"); ax.set_ylabel("units"); ax.legend(fontsize=9)
ax.set_title("E  Distribution of optimal lags", loc="left", fontsize=11)

ax = fig.add_subplot(gs[1, 2])
ax.scatter(res["pd_move"][okv], pd_vel[okv], s=12, alpha=0.6, c="tab:green")
ax.plot([0, 360], [0, 360], "k--", lw=0.8)
ax.set_xlabel("PD from target direction (deg)"); ax.set_ylabel("PD from hand velocity (deg)")
ax.set_xlim(0, 360); ax.set_ylim(0, 360)
ax.set_title(f"F  PD: target dir vs early-reach velocity (n={okv.sum()}, med |Δ|={np.median(np.abs(d_pv)):.0f}°)",
             loc="left", fontsize=10)

fig.suptitle("fig05  Speed and velocity tuning: 182 units, MC_Maze sub-Jenkins", fontsize=13)
fig.savefig("figures/fig05_speed_tuning.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# Rate grows with hand speed when the hand moves toward a unit's preferred direction and falls
# with speed when moving away (Wilcoxon p ≈ 5e-6), the signature of the cosine-direction ×
# speed model of Moran & Schwartz. The rate–speed cross-correlation peaks at positive lags for
# most units (median ≈ +100 ms): motor-cortical activity *leads* the kinematics.

# %% [markdown]
# ## 5. Poisson GLM encoding of velocity (nemos)
#
# Finally we fit an encoding model per population: spike count in 50 ms bins from hand
# kinematics convolved with a temporal basis (4 raised-cosine functions over a 500 ms window).
# Because neural activity leads the hand, kinematics are shifted 250 ms earlier, giving an
# effective lag range of +200 ms (lead) to −250 ms (lag). Three feature sets are compared with
# 5-fold blocked cross-validation:
#
# - **speed**: convolved hand speed (4 features), direction-blind
# - **velocity**: convolved vx and vy (8 features), linear velocity = cosine direction × speed
# - **velocity+speed**: both (12 features)
#
# Per-unit test log-likelihoods quantify what each signal adds beyond the other, and the
# predicted-vs-actual rate correlation summarizes absolute encoding quality per unit.

# %%
import jax
jax.config.update("jax_enable_x64", True)
import nemos as nmo

SHIFT, WIN, NB = 5, 10, 4   # tap k sees kinematics at (SHIFT-1-k) bins: lags +200..-250 ms
def shift_sig(x, k):
    y = np.empty_like(x); y[:-k] = x[k:]; y[-k:] = x[-k:]
    return y

basis = nmo.basis.RaisedCosineLinearConv(n_basis_funcs=NB, window_size=WIN)
X_vx = np.asarray(basis.compute_features(shift_sig(vx, SHIFT)))
X_vy = np.asarray(basis.compute_features(shift_sig(vy, SHIFT)))
X_sp = np.asarray(basis.compute_features(shift_sig(spd, SHIFT)))
X_vel = np.concatenate([X_vx, X_vy], axis=1)
X_full = np.concatenate([X_vx, X_vy, X_sp], axis=1)

_, widx = np.unique(trial_of_bin, return_index=True)
within = np.arange(n_bins) - widx[trial_of_bin]
tlen = np.bincount(trial_of_bin)[trial_of_bin]
valid = (within >= WIN) & (within < tlen - SHIFT) & ~np.isnan(X_full).any(1)
print(f"valid bins for GLM: {valid.sum()}/{n_bins}")

def poisson_ll(y, mu):
    mu = np.clip(mu, 1e-9, None)
    return (y*np.log(mu) - mu - gammaln(y+1)).mean(0)

# movement-epoch mask: bins within [-0.1, 0.5] s of move onset (direction signal lives here;
# the rest of trial time is mostly hold periods where direction is irrelevant)
trel_bin = bin_times - tr["move_onset"][trial_of_bin]
move_mask = (trel_bin >= -0.1) & (trel_bin <= 0.5)

folds = np.array_split(np.unique(trial_of_bin), 5)
feat_sets = {"speed": X_sp, "velocity": X_vel, "velocity+speed": X_full}
test_ll = {name: np.zeros((5, n_units)) for name in feat_sets}
test_ll_move = {name: np.zeros((5, n_units)) for name in feat_sets}
r_enc_move = np.full(n_units, np.nan)   # predicted-vs-actual rate correlation, full model
for fi, te_tr in enumerate(folds):
    te = valid & np.isin(trial_of_bin, te_tr)
    trn = valid & ~np.isin(trial_of_bin, te_tr)
    for name, X in feat_sets.items():
        glm = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=0.01,
                                    solver_name="LBFGS",
                                    solver_kwargs={"tol": 1e-10, "maxiter": 1000})
        glm.fit(X[trn], counts[trn])
        test_ll[name][fi] = poisson_ll(counts[te], np.asarray(glm.predict(X[te])))
        tem = te & move_mask
        test_ll_move[name][fi] = poisson_ll(counts[tem], np.asarray(glm.predict(X[tem])))
        if name == "velocity+speed":
            pred_s = gaussian_filter1d(np.asarray(glm.predict(X[te]))/BIN, 2, axis=0)
            act_s = gaussian_filter1d(counts[te]/BIN, 2, axis=0)
            tm = move_mask[te]
            for u in range(n_units):
                if act_s[tm, u].std() > 0:
                    r = np.corrcoef(act_s[tm, u], pred_s[tm, u])[0, 1]
                    r_enc_move[u] = r if np.isnan(r_enc_move[u]) else (r_enc_move[u]*fi + r)/(fi+1)
    print(f"fold {fi} done", flush=True)

mean_ll = {k: v.mean(0) for k, v in test_ll.items()}
mean_ll_move = {k: v.mean(0) for k, v in test_ll_move.items()}
beat = mean_ll["velocity"] > mean_ll["speed"]
beat_move = mean_ll_move["velocity"] > mean_ll_move["speed"]
print(f"velocity beats speed-only (all bins): {beat.sum()}/{n_units}")
print(f"velocity beats speed-only (movement bins): {beat_move.sum()}/{n_units} "
      f"({(beat_move & sig_move).sum()}/{sig_move.sum()} among direction-tuned)")
print(f"velocity+speed beats velocity (all bins): "
      f"{(mean_ll['velocity+speed'] > mean_ll['velocity']).sum()}/{n_units}")

# full-data velocity fit -> temporal filters, GLM preferred direction and lag
glm_full = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=0.01,
                                 solver_name="LBFGS",
                                 solver_kwargs={"tol": 1e-10, "maxiter": 1000})
glm_full.fit(X_vel[valid], counts[valid])
_, kernels = basis.evaluate_on_grid(WIN)
coef = np.asarray(glm_full.coef_)
fx = kernels @ coef[:NB]
fy = kernels @ coef[NB:2*NB]
lag_axis = (SHIFT - 1 - np.arange(WIN)) * BIN * 1000  # conv orientation verified empirically
fnorm = np.hypot(fx, fy)
peak_k = np.argmax(fnorm, axis=0)
pd_glm = (np.degrees(np.arctan2(fy[peak_k, np.arange(n_units)],
                                fx[peak_k, np.arange(n_units)])) + 360) % 360
pref_lag = lag_axis[peak_k]

# %% [markdown]
# ### Figure 6: GLM results

# %%
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.35)

# A: example unit temporal filters
ax = fig.add_subplot(gs[0, 0])
u = ex
ax.plot(lag_axis, fx[:, u], "o-", color="tab:blue", label="vx filter", ms=4)
ax.plot(lag_axis, fy[:, u], "o-", color="tab:orange", label="vy filter", ms=4)
ax.plot(lag_axis, fnorm[:, u], "k--", lw=1, label="norm")
ax.axvline(0, color="gray", ls=":", lw=0.8)
ax.set_xlabel("lag (ms; + = neural leads)"); ax.set_ylabel("filter weight")
ax.legend(fontsize=8)
ax.set_title(f"A  Velocity filters, unit {un['unit_ids'][u]}\n(GLM PD={pd_glm[u]:.0f}°)",
             loc="left", fontsize=10)

# B: predicted vs actual rate for example unit (held-out trials)
ax = fig.add_subplot(gs[0, 1:3])
te = valid & np.isin(trial_of_bin, folds[0])
glm1 = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.01, solver_name="LBFGS",
                   solver_kwargs={"tol": 1e-10, "maxiter": 1000})
glm1.fit(X_full[valid & ~np.isin(trial_of_bin, folds[0])],
         counts[valid & ~np.isin(trial_of_bin, folds[0]), u])
mu_u = np.asarray(glm1.predict(X_full[te]))/BIN
t_te = bin_times[te]
r_actual = gaussian_filter1d(rates[te, u], sigma=2)
r_pred = gaussian_filter1d(mu_u, sigma=2)
seg = (t_te > t_te[0]+40) & (t_te < t_te[0]+58)
ax.plot(t_te[seg], r_actual[seg], "k", lw=1.3, label="actual (smoothed)")
ax.plot(t_te[seg], r_pred[seg], "tab:red", lw=1.3, label="GLM prediction")
ax.set_xlabel("time (s)"); ax.set_ylabel("firing rate (sp/s)"); ax.legend(fontsize=9)
r2_enc = np.corrcoef(r_actual, r_pred)[0, 1]**2
ax.set_title(f"B  Encoding quality, unit {un['unit_ids'][u]}, held-out trials (R²={r2_enc:.2f})",
             loc="left", fontsize=10)

# C: model comparison, all bins
ax = fig.add_subplot(gs[0, 3])
ax.scatter(mean_ll["speed"], mean_ll["velocity"], s=12, alpha=0.6,
           c=["tab:red" if g == "M1" else "tab:blue" for g in un["group"]])
lims = [min(mean_ll["speed"].min(), mean_ll["velocity"].min()),
        max(mean_ll["speed"].max(), mean_ll["velocity"].max())]
ax.plot(lims, lims, "k--", lw=0.8)
ax.set_xlabel("test LL, speed-only"); ax.set_ylabel("test LL, velocity")
ax.set_title(f"C  All bins: velocity vs speed\n({beat.sum()}/{n_units} above diagonal)",
             loc="left", fontsize=9)

# D: model comparison, movement bins only
ax = fig.add_subplot(gs[1, 0])
ax.scatter(mean_ll_move["speed"], mean_ll_move["velocity"], s=12, alpha=0.6,
           c=["tab:red" if g == "M1" else "tab:blue" for g in un["group"]])
lims = [min(mean_ll_move["speed"].min(), mean_ll_move["velocity"].min()),
        max(mean_ll_move["speed"].max(), mean_ll_move["velocity"].max())]
ax.plot(lims, lims, "k--", lw=0.8)
ax.set_xlabel("test LL, speed-only"); ax.set_ylabel("test LL, velocity")
ax.set_title(f"D  Movement bins: velocity vs speed\n({beat_move.sum()}/{n_units} above diagonal)",
             loc="left", fontsize=9)

# E: what each signal adds beyond the other (movement bins)
ax = fig.add_subplot(gs[1, 1])
d_dir = mean_ll_move["velocity+speed"] - mean_ll_move["speed"]     # direction beyond speed
d_spd = mean_ll_move["velocity+speed"] - mean_ll_move["velocity"]  # speed beyond direction
ax.hist(d_dir, bins=40, color="tab:red", alpha=0.65,
        label=f"direction beyond speed ({(d_dir>0).sum()}/{n_units})")
ax.hist(d_spd, bins=40, color="tab:blue", alpha=0.65,
        label=f"speed beyond direction ({(d_spd>0).sum()}/{n_units})")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Δ test LL (movement bins)"); ax.set_ylabel("units")
ax.legend(fontsize=8)
p_dir = stats.wilcoxon(d_dir).pvalue
p_spd = stats.wilcoxon(d_spd).pvalue
ax.set_title(f"E  Independent direction & speed info\n(Wilcoxon p={p_dir:.0e} / {p_spd:.0e})",
             loc="left", fontsize=9)

# F: encoding quality across the population (held-out movement bins)
ax = fig.add_subplot(gs[1, 2])
ax.hist(r_enc_move[sig_move], bins=30, color="tab:purple", alpha=0.8,
        label=f"direction-tuned (n={sig_move.sum()})")
ax.hist(r_enc_move[~sig_move], bins=30, color="gray", alpha=0.55,
        label=f"not tuned (n={(~sig_move).sum()})")
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.nanmedian(r_enc_move), color="tab:red", ls="--", lw=1.2,
           label=f"median {np.nanmedian(r_enc_move):.2f}")
ax.set_xlabel("predicted vs actual rate correlation (r)"); ax.set_ylabel("units")
ax.legend(fontsize=8)
ax.set_title("F  Encoding quality, velocity+speed model\n(held-out movement bins)",
             loc="left", fontsize=9)

# G: GLM PD vs tuning-curve PD
ax = fig.add_subplot(gs[1, 3])
okg = sig_move & (fnorm.max(0) > np.percentile(fnorm.max(0), 25))
d_glm = (res["pd_move"][okg] - pd_glm[okg] + 180) % 360 - 180
ax.scatter(res["pd_move"][okg], pd_glm[okg], s=12, alpha=0.6, c="tab:green")
ax.plot([0, 360], [0, 360], "k--", lw=0.8)
ax.set_xlabel("PD from tuning curve (deg)"); ax.set_ylabel("PD from GLM filters (deg)")
ax.set_xlim(0, 360); ax.set_ylim(0, 360)
ax.set_title(f"G  GLM vs tuning-curve PD\n(n={okg.sum()}, med |Δ|={np.median(np.abs(d_glm)):.0f}°)",
             loc="left", fontsize=10)

fig.suptitle("fig06  Poisson GLM encoding of hand velocity (nemos PopulationGLM, 5-fold blocked CV)",
             fontsize=13)
fig.savefig("figures/fig06_glm_encoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# On held-out movement-epoch bins, adding velocity direction to a speed-only model improves
# test likelihood for nearly every unit, and adding speed to a velocity-only model does the same
# (panel E): direction and speed are encoded independently and complementarily, the
# Moran & Schwartz result recovered in an encoding-model framework. The full model reconstructs
# held-out firing-rate dynamics well (panels B, F), and the preferred directions read out from
# the fitted velocity filters agree with the tuning-curve estimates (panel G). The fitted
# filters are temporally broad (panel A), so the precise neural-to-hand delay is best read from
# the model-free cross-correlation in Figure 5 rather than from filter peaks.

# %% [markdown]
# ## 6. Summary
#
# - **Direction tuning:** 174/182 units (96%) are significantly tuned to reach direction during
#   movement (one-way ANOVA across 30° direction bins, p < 0.01), and 129/182 during the
#   instructed-delay period. Tuning is broadly cosine-shaped and preferred directions are
#   largely preserved from planning to execution (median |ΔPD| ≈ 37°).
# - **Speed/velocity tuning:** firing rate scales with hand speed when moving toward the
#   preferred direction and anti-scales when moving away (Wilcoxon p ≈ 5e-6), matching the
#   cosine-direction × speed encoding model. Rate–speed cross-correlation peaks at
#   ≈ +100 ms, i.e. cortical activity leads the hand.
# - **Encoding model:** in a Poisson GLM on lagged kinematics, direction and speed each improve
#   held-out predictions beyond the other for nearly all units (Wilcoxon p ≈ 1e-31), and
#   GLM-inferred preferred directions agree with the tuning-curve estimates
#   (median |ΔPD| ≈ 37°).
#
# These replicate the classical Georgopoulos/Moran–Schwartz picture of motor-cortical reach
# coding in a modern open dataset streamed directly from the DANDI Archive.

# %%
with open("data/results_summary.pkl", "wb") as f:
    pickle.dump(dict(res=res, bin_means=bin_means, occupied=occupied,
                     sig_move=sig_move, sig_delay=sig_delay,
                     slope_toward=slope_toward, slope_away=slope_away,
                     peak_lag=peak_lag, peak_cc=peak_cc, pd_vel=pd_vel, R_vel=R_vel,
                     mean_ll=mean_ll, mean_ll_move=mean_ll_move, pd_glm=pd_glm,
                     pref_lag=pref_lag, r_enc_move=r_enc_move), f)
print("done. figures in figures/, summary in data/results_summary.pkl")
