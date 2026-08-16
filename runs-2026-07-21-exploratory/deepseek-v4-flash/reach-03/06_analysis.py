"""
Reach direction tuning in macaque dorsal premotor cortex (PMd), MC_Maze DANDI
000128, Jenkins "full" session.

Firing rates are computed per trial in a 0.7 s window after movement onset.
Direction selectivity is tested with a 1-way ANOVA over the (non-empty)
45-degree reach-direction bins; preferred direction via circular statistics;
tuning depth via a modulation index (max-min)/(max+min).
"""
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import pynapple as nap
import mcmaze_data as md

OUTDIR = "."

# ----------------------------------------------------------------------
print("Loading NWB ...")
nwbfile, nwb = md.load_nwb()
units = nwb["units"]
tr = md.build_trial_table(nwbfile, nwb)

unit_ids = list(units.index)
n_units = len(units)
n_trials = len(tr["direction"])
print(f"{n_units} units, {n_trials} trials")

# region (all PMd in this file, but keep the mapping for bookkeeping)
elec_locs = np.asarray(nwbfile.electrodes["location"][:])
unit_elec = md.unit_electrode_indices()
unit_region = np.array([elec_locs[e] for e in unit_elec])
print("units per region:", {r: int((unit_region == r).sum()) for r in np.unique(unit_region)})

# ----------------------------------------------------------------------
MOV = 0.7
starts = tr["move"]
move_epochs = nap.IntervalSet(start=starts, end=starts + MOV)
angles = tr["direction"]

NBINS = 8
bin_edges = np.linspace(-np.pi, np.pi, NBINS + 1)
dir_idx = np.clip(np.digitize(angles, bin_edges) - 1, 0, NBINS - 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_counts = np.bincount(dir_idx)
nonempty = bin_counts > 0
print("trials per direction bin:", bin_counts)
print("empty bins:", np.where(~nonempty)[0])

# ----------------------------------------------------------------------
# per-trial spike counts during movement (vectorized with searchsorted)
# ----------------------------------------------------------------------
spike_times = [np.asarray(units[uid].t) for uid in tqdm(units.index, desc="spike arrays")]
counts = np.zeros((n_units, n_trials))
for iu in tqdm(range(n_units), desc="binning spikes"):
    sp = spike_times[iu]
    counts[iu] = np.searchsorted(sp, starts + MOV) - np.searchsorted(sp, starts)
trial_rate = counts / MOV

# overall spike rate (for the report)
mean_rate = trial_rate.mean(axis=1)

# ----------------------------------------------------------------------
# Direction selectivity per unit
# ----------------------------------------------------------------------
pref_dir = np.full(n_units, np.nan)
mod_index = np.full(n_units, np.nan)
circ_r = np.full(n_units, np.nan)
fstat = np.full(n_units, np.nan)
pval = np.full(n_units, np.nan)

for iu in tqdm(range(n_units), desc="direction ANOVA"):
    r = trial_rate[iu]
    groups = [r[dir_idx == b] for b in range(NBINS) if nonempty[b]]
    f, p = stats.f_oneway(*groups)
    pval[iu], fstat[iu] = p, f
    mb = np.array([r[dir_idx == b].mean() for b in range(NBINS) if nonempty[b]])
    cats = bin_centers[nonempty]
    z = np.sum(mb * np.exp(1j * cats))
    pref_dir[iu] = np.angle(z)
    circ_r[iu] = np.abs(z) / mb.sum()
    mod_index[iu] = (mb.max() - mb.min()) / (mb.max() + mb.min() + 1e-12)

q = stats.false_discovery_control(pval)
tuned = q < 0.05
print(f"\nFraction direction-tuned (BH q<0.05): {tuned.mean():.3f}  ({tuned.sum()}/{n_units})")
print(f"Modulation index: median {np.median(mod_index):.3f}, "
      f"mean {mod_index.mean():.3f}")

np.savez("direction_analysis.npz",
         pref_dir=pref_dir, mod_index=mod_index, circ_r=circ_r,
         pval=pval, fstat=fstat, tuned=tuned,
         mean_rate=mean_rate, bin_centers=bin_centers, bin_counts=bin_counts)

# ------------------------------------------------------------------
# Figure 1: raw overview (speed + spikes for 12 s)
# ------------------------------------------------------------------
vel = np.asarray(nwb["hand_vel"].values)
vt = np.asarray(nwb["hand_vel"].t)
speed = np.linalg.norm(vel, axis=1) / 1000.0  # mm/s -> m/s

t0, t1 = 400.0, 412.0
sel = (vt >= t0) & (vt <= t1)
n_show = 40
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
axes[0].plot(vt[sel], speed[sel], color="k", lw=1.2)
axes[0].set_ylabel("hand speed\n(m/s)")
axes[0].set_title("Raw neural and kinematic activity (Jenkins, MC_Maze)")

for i in range(n_show):
    sp = spike_times[i]
    m = (sp >= t0) & (sp <= t1)
    axes[1].plot(sp[m], np.full(m.sum(), i), ".", ms=1, color="k")
axes[1].set_ylabel("unit")
axes[1].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig01_raw_overview.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 2: polar tuning curves for the 6 most strongly-tuned units
# ----------------------------------------------------------------------
order = np.argsort(mod_index)[::-1]
ex_ids = [uid for uid in order if tuned[uid]][:6] if tuned.any() else list(order[:6])

fig, axes = plt.subplots(2, 3, figsize=(12, 8), subplot_kw={"projection": "polar"})
for k, ax in enumerate(axes.flat):
    i = ex_ids[k]
    mb = np.array([trial_rate[i, dir_idx == b].mean() for b in range(NBINS) if nonempty[b]])
    cats = bin_centers[nonempty]
    vals = np.concatenate([[mb[-1]], mb, [mb[0]]])
    catsp = np.concatenate([[cats[-1] - 2*np.pi], cats, [cats[0] + 2*np.pi]])
    ax.plot(catsp, vals, "-o", lw=1.5, ms=3)
    ax.fill(catsp, vals, alpha=0.25)
    ax.plot([0, pref_dir[iu]], [0, vals.max() * 0.9], "r-", lw=2)
    ax.set_title(f"unit {units.index[iu]} (MI={mod_index[iu]:.2f}, "
                 f"rate={mean_rate[iu]:.0f} Hz)", fontsize=9)
    ax.set_xticks(bin_centers[nonempty])
    ax.set_xticklabels(np.round(np.degrees(bin_centers[nonempty])).astype(int))
fig.suptitle("Polar tuning curves during movement (reach direction)")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig02_direction_tuning_curves.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# Figure 3: population summary
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# (a) PD rose
ax = axes[0]
ax.hist(np.degrees(pref_dir[tuned]), bins=24, color="#4C72B0")
ax.set_xlabel("preferred direction (deg)")
ax.set_ylabel("units")
ax.set_title(f"PD of tuned units (n={tuned.sum()})")

# (b) tuning depth distribution, split tuned/not
ax = axes[1]
ax.hist(mod_index[~tuned], bins=20, color="#DD8452", alpha=0.7, label="not tuned")
ax.hist(mod_index[tuned], bins=20, color="#4C72B0", alpha=0.7, label="tuned")
ax.axvline(np.nanmedian(mod_index), color="k", ls="--")
ax.set_xlabel("modulation index (max-min)/(max+min)")
ax.set_ylabel("units")
ax.legend()

# (c) rate vs. PD scatter
ax = axes[2]
ax.scatter(mean_rate, np.degrees(pref_dir), s=8, alpha=0.6, c=tuned)
ax.set_xlabel("mean rate (Hz)")
ax.set_ylabel("preferred direction (deg)")

fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig03_direction_summary.png", dpi=150)
plt.close(fig)

print("part1 done")