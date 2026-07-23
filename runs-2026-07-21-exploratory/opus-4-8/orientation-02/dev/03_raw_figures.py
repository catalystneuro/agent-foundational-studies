"""Figures 1-2: raw activity during the drifting-grating block, and one example unit.

These are validation figures. Before computing any tuning metric they show that the
spike times, the stimulus table and the running trace are aligned as expected.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import oslib

SES = "715093703"
D = np.load(f"extracted/{SES}.npz", allow_pickle=True)
SP = np.load(f"extracted/{SES}_dg_spikes.npz", allow_pickle=True)

area = D["area"]
dg_ori, dg_tf = D["dg_ori"], D["dg_tf"]
dg_start, dg_stop = D["dg_start"], D["dg_stop"]
rates = D["dg_rates"]
unit_ids = D["unit_ids"]

spikes = nap.TsGroup({int(u): nap.Ts(SP[f"u{u}"].astype(float)) for u in SP["unit_ids"]},
                     metadata={"area": area})

DIRS = np.array(sorted(np.unique(dg_ori)))
COL = dict(zip(DIRS, plt.cm.hsv(np.linspace(0, 0.92, len(DIRS)))))

# ---------------------------------------------------------------- figure 1
# A 60 s slice of the drifting-grating block: unit raster grouped by area, the
# population rate, and the running speed, with stimulus epochs shaded by direction.
t0 = dg_start[120]
t1 = t0 + 30.0
win = nap.IntervalSet(start=t0, end=t1)

# Subsample to at most SHOW_PER_AREA units per area so individual spike trains stay
# visible; the full population still drives the summed rate below.
SHOW_PER_AREA = 14
rng = np.random.default_rng(0)
pick = []
for a in oslib.VISUAL_CORTEX + oslib.THALAMUS:
    idx = np.where(area == a)[0]
    if len(idx) == 0:
        continue
    pick.append(rng.choice(idx, min(SHOW_PER_AREA, len(idx)), replace=False))
pick = np.concatenate(pick)
sorted_ids = unit_ids[pick]
sorted_area = area[pick]

fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.12})

ax = axes[0]
for row, uid in enumerate(sorted_ids):
    t = spikes[int(uid)].restrict(win).t
    ax.plot(t, np.full_like(t, row), "|", color="k", ms=4.0, mew=0.7)
# area bands on the right
bounds, labels = [], []
for a in oslib.VISUAL_CORTEX + oslib.THALAMUS:
    idx = np.where(sorted_area == a)[0]
    if len(idx) == 0:
        continue
    ax.axhline(idx[-1] + 0.5, color="0.4", lw=0.8, ls="--")
    labels.append((a, idx.mean()))
for a, y in labels:
    ax.text(t1 + 0.25, y, a, va="center", fontsize=9,
            color="C0" if a.startswith("VIS") else "C1", fontweight="bold")
ax.set_ylabel("unit (sorted by area)")
ax.set_ylim(-1, len(sorted_ids))
ax.set_title(f"Session {SES}: drifting-grating block, {t1 - t0:.0f} s slice\n"
             "coloured bands = 2 s grating presentations (colour codes direction); "
             "white = inter-trial grey screen", pad=10)

pop = spikes.count(bin_size=0.02, ep=win).sum(axis=1) / (0.02 * len(spikes))
pop = pop.smooth(0.03)
axes[1].plot(pop.t, pop.values, color="k", lw=0.9)
axes[1].set_ylabel("population\nrate (Hz)")

# Running speed is stored per drifting-grating trial by the extraction step, so it
# is drawn as a step trace across the trials falling in this window.
sel_run = np.where((dg_start < t1) & (dg_stop > t0))[0]
axes[2].step(dg_start[sel_run], D["dg_speed"][sel_run], where="post", color="C2", lw=1.4)
axes[2].axhline(1.0, color="0.5", ls=":", lw=1)
axes[2].set_ylabel("running speed\n(cm/s, trial mean)")
axes[2].set_xlabel("time (s)")

# stimulus shading on all three axes
sel = np.where((dg_start < t1) & (dg_stop > t0))[0]
for ax_ in axes:
    for i in sel:
        ax_.axvspan(dg_start[i], dg_stop[i], color=COL[dg_ori[i]], alpha=0.22, lw=0)
    ax_.set_xlim(t0, t1)

# direction legend
handles = [plt.Line2D([], [], color=COL[d], lw=8, alpha=0.5, label=f"{int(d)}°") for d in DIRS]
axes[0].legend(handles=handles, ncol=8, loc="lower center", bbox_to_anchor=(0.5, 1.14),
               frameon=False, fontsize=9, columnspacing=1.0, handlelength=1.4)
fig.savefig("fig01_raw_activity.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig01_raw_activity.png")

# ---------------------------------------------------------------- figure 2
# Pick a strongly orientation-tuned VISp unit and show its raster and PSTH by direction.
finite = np.nanmean(rates, axis=1)
by_dir = np.stack([np.nanmean(rates[:, dg_ori == d], axis=1) for d in DIRS], axis=1)
gosi, gdsi, pref_ori, pref_dir = oslib.osi_dsi(by_dir, DIRS)
cand = np.where((area == "VISp") & (finite > 2.0))[0]
best = cand[np.argmax(gosi[cand])]
uid = int(unit_ids[best])
print(f"example unit {uid} (VISp): mean rate {finite[best]:.1f} Hz, gOSI {gosi[best]:.2f}, "
      f"preferred orientation {pref_ori[best]:.0f} deg")
np.save("extracted/example_unit.npy", np.array([best, uid]))

PRE, POST = 0.5, 2.5
fig, axes = plt.subplots(2, len(DIRS), figsize=(18, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1], "hspace": 0.28, "wspace": 0.18})
ymax_psth = 0
for j, d in enumerate(DIRS):
    tr = np.where(dg_ori == d)[0]
    ev = nap.Ts(dg_start[tr])
    pe = nap.compute_perievent(spikes[uid], ev, window=(-PRE, POST))
    axr = axes[0, j]
    for k, key in enumerate(pe.keys()):
        t = pe[key].t
        axr.plot(t, np.full_like(t, k), "|", color=COL[d], ms=3.5, mew=0.8)
    axr.axvspan(0, 2.0, color="0.85", zorder=0)
    axr.set_title(f"{int(d)}°", color=COL[d], fontweight="bold", pad=6)
    axr.set_ylim(-1, len(pe))
    if j == 0:
        axr.set_ylabel("trial")
    else:
        axr.set_yticklabels([])

    # PSTH from the same perievent spikes.
    edges = np.arange(-PRE, POST + 1e-9, 0.05)
    allsp = np.concatenate([pe[k].t for k in pe.keys()]) if len(pe) else np.array([])
    h, _ = np.histogram(allsp, bins=edges)
    psth = h / (len(pe) * 0.05)
    axp = axes[1, j]
    axp.fill_between(edges[:-1] + 0.025, psth, step="mid", color=COL[d], alpha=0.75)
    axp.axvspan(0, 2.0, color="0.85", zorder=0)
    axp.set_xlabel("time from onset (s)")
    ymax_psth = max(ymax_psth, psth.max())
    if j == 0:
        axp.set_ylabel("rate (Hz)")
    else:
        axp.set_yticklabels([])
for j in range(len(DIRS)):
    axes[1, j].set_ylim(0, ymax_psth * 1.05)
fig.suptitle(f"Unit {uid} (VISp, session {SES}): responses to each grating direction "
             f"(gOSI = {gosi[best]:.2f}, preferred orientation {pref_ori[best]:.0f}°)",
             y=1.02, fontsize=13)
fig.savefig("fig02_example_unit.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig02_example_unit.png")
