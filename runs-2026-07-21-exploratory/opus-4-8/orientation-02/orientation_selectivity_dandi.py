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
# # Orientation selectivity in the mouse visual system
#
# ## DANDI:000021, Allen Institute Visual Coding with Neuropixels
#
# Neurons in primary visual cortex respond preferentially to edges of a particular
# orientation. This is the oldest and most reproduced result in visual
# neurophysiology, and it is the standard against which a recording, a spike
# sorter, or an analysis pipeline can be sanity-checked. This notebook demonstrates
# it from scratch on public data, and then asks the sharper question that the
# original work also asked: where in the visual pathway does the selectivity appear?
#
# The dataset is [DANDI:000021](https://dandiarchive.org/dandiset/000021), the Allen
# Institute "Visual Coding - Neuropixels" release, Brain Observatory 1.1 stimulus
# set. Each of the 32 sessions is a head-fixed mouse watching a monitor while up to
# six Neuropixels probes record simultaneously from the visual thalamus, primary
# visual cortex, and five higher visual areas. Two stimulus blocks are used here:
#
# * **drifting gratings**: 2 s presentations, 8 directions x 5 temporal frequencies
#   x 15 repeats, with interleaved blank sweeps that give a matched baseline;
# * **static gratings**: 0.25 s presentations, 6 orientations x 5 spatial
#   frequencies x 4 phases, in a separate block later in the session.
#
# Because the two blocks are separated in time and use different stimulus
# parameters, the static gratings provide a genuine out-of-sample test of any
# orientation preference measured on the drifting gratings.
#
# **What this notebook shows.** Individual cortical units have sharp, reliable
# orientation tuning; the preference is stable across independent halves of the
# data, replicates on the independent static-grating block, and does not depend on
# whether the mouse is running (selectivity is in fact slightly higher when the
# mouse is stationary). Orientation selectivity is substantially weaker in
# the dorsal lateral geniculate nucleus, the first-order thalamic relay that feeds
# V1, which is the classical result that orientation selectivity is largely
# constructed in cortex.
#
# **Reproducing this notebook.** The heavy step is streaming the session files.
# `pipeline.extract_session` caches everything the analysis needs as a small `.npz`
# per session, and skips a session whose cache already exists, so only the first
# run pays the streaming cost (roughly five minutes per session on a warm
# connection, longer cold).

# %%
import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats
from tqdm import tqdm

import oslib
import pipeline
from analyze_helpers import responsive

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

CORTEX_C, THAL_C, LP_C = "#2b6cb0", "#c05621", "#8b6f47"
PROTOTYPE = "715093703"   # session used for the single-session walkthrough
print("sessions available in DANDI:000021:", len(oslib.SESSIONS))

# %% [markdown]
# ## 1. Load one session and inspect what is in it
#
# The NWB file is streamed from the DANDI S3 bucket with `remfile`, which fetches
# only the byte ranges that are actually read and keeps them in an on-disk cache.
# Nothing is downloaded in full. Pynapple wraps the result so that spike trains
# become a `TsGroup` and every stimulus block becomes an `IntervalSet`.

# %%
nwbfile = oslib.open_session(PROTOTYPE)
print("session:", nwbfile.session_id, "| subject:", nwbfile.subject.subject_id,
      "|", nwbfile.subject.genotype)
print("\ninterval tables:")
for name, iv in nwbfile.intervals.items():
    print(f"  {name:45s} n={len(iv)}")

nwb = nap.NWBFile(nwbfile)
print()
print(nwb)

# %% [markdown]
# Units are restricted to the visual structures of interest and filtered with the
# quality metrics the Allen Institute recommends for this dataset: fewer than 0.5
# ISI violations, amplitude cutoff below 0.1, presence ratio above 0.9, and
# signal-to-noise above 1. The area label comes from the Allen CCF structure of
# each unit's peak channel in the electrodes table.

# %%
tsg, meta = oslib.good_units(nwbfile)
print(f"{len(tsg)} units pass quality control in visual areas")
print(meta["area"].value_counts().to_string())

dg = oslib.stim_table(nwbfile, "drifting_gratings_presentations")
print(f"\ndrifting gratings: {len(dg)} trials")
print("  directions:", sorted(dg.orientation.unique()))
print("  temporal frequencies:", sorted(dg.temporal_frequency.unique()))
print("  presentation duration: %.3f s" % (dg.stop_time - dg.start_time).mean())
print("  blank sweeps:", len(oslib.blank_table(nwbfile, "drifting_gratings_presentations")))

# %% [markdown]
# ## 2. Cache the sessions
#
# Everything the analysis needs is pulled out in a single pass per session: the
# per-trial firing rate of every unit in both grating blocks, the blank-sweep
# baseline, the running speed on each trial, and the stimulus parameters. Trials
# overlapping a probe's `invalid_times` epochs are set to NaN for the units on that
# probe only.
#
# The counting window opens 30 ms after stimulus onset, to skip the visual response
# latency, and is truncated at stimulus offset and at the next trial's onset. The
# truncation matters for the static gratings, which run back to back at about 4 Hz:
# without it adjacent windows overlap and pynapple silently merges them.

# %%
SESSIONS = sorted(s.split("/")[-1][:-4] for s in glob.glob("extracted/*.npz")
                  if "_dg_spikes" not in s)
if not SESSIONS:                      # cold start: stream a handful of sessions
    SESSIONS = list(oslib.SESSIONS)[:6]
    for s in tqdm(SESSIONS, desc="streaming sessions"):
        pipeline.extract_session(s)
print("sessions in this run:", SESSIONS)

# %% [markdown]
# ## 3. Look at the raw data before analysing it
#
# Before computing any tuning metric it is worth confirming that the spike times,
# the stimulus table and the running trace are aligned. A 30 s slice of the
# drifting-grating block shows the 2 s presentations as coloured bands, a subsample
# of units from each area as a raster, the summed population rate, and the running
# speed. Stimulus onsets are clearly visible as transients in the population rate.

# %%
D = np.load(f"extracted/{PROTOTYPE}.npz", allow_pickle=True)
SP = np.load(f"extracted/{PROTOTYPE}_dg_spikes.npz", allow_pickle=True)
area, unit_ids = D["area"], D["unit_ids"]
dg_ori, dg_tf, dg_start, dg_stop = D["dg_ori"], D["dg_tf"], D["dg_start"], D["dg_stop"]
spikes = nap.TsGroup({int(u): nap.Ts(SP[f"u{u}"].astype(float)) for u in SP["unit_ids"]},
                     metadata={"area": area})
DIRS = np.array(sorted(np.unique(dg_ori)))
COL = dict(zip(DIRS, plt.cm.hsv(np.linspace(0, 0.92, len(DIRS)))))

t0 = dg_start[120]
t1 = t0 + 30.0
win = nap.IntervalSet(start=t0, end=t1)

rng = np.random.default_rng(0)
pick = np.concatenate([rng.choice(np.where(area == a)[0], min(14, (area == a).sum()),
                                  replace=False)
                       for a in oslib.VISUAL_CORTEX + oslib.THALAMUS if (area == a).sum()])
sorted_ids, sorted_area = unit_ids[pick], area[pick]

fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.12})
for row, uid in enumerate(sorted_ids):
    t = spikes[int(uid)].restrict(win).t
    axes[0].plot(t, np.full_like(t, row), "|", color="k", ms=4.0, mew=0.7)
for a in oslib.VISUAL_CORTEX + oslib.THALAMUS:
    idx = np.where(sorted_area == a)[0]
    if len(idx):
        axes[0].axhline(idx[-1] + 0.5, color="0.4", lw=0.8, ls="--")
        axes[0].text(t1 + 0.25, idx.mean(), a, va="center", fontsize=9,
                     color=CORTEX_C if a.startswith("VIS") else THAL_C, fontweight="bold")
axes[0].set_ylabel("unit (sorted by area)")
axes[0].set_ylim(-1, len(sorted_ids))
axes[0].set_title(f"Session {PROTOTYPE}: drifting-grating block, 30 s slice\n"
                  "coloured bands = 2 s grating presentations (colour codes direction); "
                  "white = inter-trial grey screen", pad=10)

pop = spikes.count(bin_size=0.02, ep=win).sum(axis=1) / (0.02 * len(spikes))
axes[1].plot(pop.smooth(0.03).t, pop.smooth(0.03).values, color="k", lw=0.9)
axes[1].set_ylabel("population\nrate (Hz)")

sel = np.where((dg_start < t1) & (dg_stop > t0))[0]
axes[2].step(dg_start[sel], D["dg_speed"][sel], where="post", color="C2", lw=1.4)
axes[2].axhline(1.0, color="0.5", ls=":", lw=1)
axes[2].set_ylabel("running speed\n(cm/s, trial mean)")
axes[2].set_xlabel("time (s)")

for ax in axes:
    for i in sel:
        ax.axvspan(dg_start[i], dg_stop[i], color=COL[dg_ori[i]], alpha=0.22, lw=0)
    ax.set_xlim(t0, t1)
axes[0].legend(handles=[plt.Line2D([], [], color=COL[d], lw=8, alpha=0.5, label=f"{int(d)}°")
                        for d in DIRS],
               ncol=8, loc="lower center", bbox_to_anchor=(0.5, 1.14), frameon=False,
               fontsize=9, columnspacing=1.0, handlelength=1.4)
fig.savefig("fig01_raw_activity.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 4. A single orientation-selective unit
#
# The clearest possible demonstration: one V1 unit, its spike raster and PSTH for
# each of the eight directions. An orientation-selective unit responds to two
# directions 180° apart (the same orientation drifting the two opposite ways) and
# is nearly silent at the orthogonal orientation.

# %%
tuning_all = np.load(f"extracted/{PROTOTYPE}_tuning.npy") if os.path.exists(
    f"extracted/{PROTOTYPE}_tuning.npy") else None
if tuning_all is None:
    pipeline.analyze_session(f"extracted/{PROTOTYPE}.npz")
    tuning_all = np.load(f"extracted/{PROTOTYPE}_tuning.npy")
gosi_p, gdsi_p, pref_p, _ = oslib.osi_dsi(tuning_all, DIRS)
peak_p = np.nanmax(tuning_all, axis=1)
# Most orientation-selective VISp unit among those that are strongly driven. The
# firing-rate floor matters: gOSI saturates at 1 for weakly driven units whose
# tuning curve happens to be positive at a single direction, and nanargmax without
# it would happily return one of those (or a unit whose gOSI is NaN outright).
cand = np.where((area == "VISp") & np.isfinite(gosi_p) & (peak_p > 8.0))[0]
best = cand[np.nanargmax(gosi_p[cand])]
uid = int(unit_ids[best])
print(f"example unit {uid} (VISp): peak evoked {peak_p[best]:.1f} Hz, "
      f"gOSI {gosi_p[best]:.2f}, preferred orientation {pref_p[best]:.0f}°")

PRE, POST = 0.5, 2.5
fig, axes = plt.subplots(2, len(DIRS), figsize=(18, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1], "hspace": 0.28, "wspace": 0.18})
edges = np.arange(-PRE, POST + 1e-9, 0.05)
ymax = 0
for j, d in enumerate(DIRS):
    pe = nap.compute_perievent(spikes[uid], nap.Ts(dg_start[dg_ori == d]), window=(-PRE, POST))
    for k, key in enumerate(pe.keys()):
        t = pe[key].t
        axes[0, j].plot(t, np.full_like(t, k), "|", color=COL[d], ms=3.5, mew=0.8)
    axes[0, j].axvspan(0, 2.0, color="0.85", zorder=0)
    axes[0, j].set_title(f"{int(d)}°", color=COL[d], fontweight="bold", pad=6)
    axes[0, j].set_ylim(-1, len(pe))

    h, _ = np.histogram(np.concatenate([pe[k].t for k in pe.keys()]), bins=edges)
    psth = h / (len(pe) * 0.05)
    axes[1, j].fill_between(edges[:-1] + 0.025, psth, step="mid", color=COL[d], alpha=0.75)
    axes[1, j].axvspan(0, 2.0, color="0.85", zorder=0)
    axes[1, j].set_xlabel("time from onset (s)")
    ymax = max(ymax, psth.max())
    if j:
        axes[0, j].set_yticklabels([])
        axes[1, j].set_yticklabels([])
axes[0, 0].set_ylabel("trial")
axes[1, 0].set_ylabel("rate (Hz)")
for j in range(len(DIRS)):
    axes[1, j].set_ylim(0, ymax * 1.05)
fig.suptitle(f"Unit {uid} (VISp, session {PROTOTYPE}): responses to each grating direction "
             f"(gOSI = {gosi_p[best]:.2f}, preferred orientation {pref_p[best]:.0f}°)",
             y=1.02, fontsize=13)
fig.savefig("fig02_example_unit.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 5. Quantify tuning for every unit
#
# For each unit the direction tuning curve is taken at that unit's **preferred
# temporal frequency**, with the blank-sweep baseline subtracted. Pooling all five
# temporal frequencies would be the simpler choice but is not a neutral one:
# cortical units are often tuned to low temporal frequencies and barely respond at
# 15 Hz, so pooling dilutes their orientation tuning while leaving high-pass
# thalamic units untouched, which biases exactly the comparison of interest.
#
# Selectivity is summarised by the circular-variance statistic
#
# $$\mathrm{gOSI} = \frac{\left|\sum_k r_k e^{2i\theta_k}\right|}{\sum_k r_k}$$
#
# which is 0 for a flat tuning curve and 1 for a unit that responds at a single
# orientation. Two things are computed alongside it, and both matter:
#
# * a **permutation test**: direction labels are shuffled within temporal
#   frequency, and the whole pipeline including the preferred-temporal-frequency
#   selection is repeated 1000 times. This gives a p-value and, just as usefully,
#   the **noise floor** of gOSI for that particular unit;
# * the **classic OSI**, $(R_{pref} - R_{orth})/(R_{pref} + R_{orth})$, which is
#   what most of the V1 literature reports and which runs larger than gOSI.
#
# The noise floor is not a technicality. gOSI is positively biased for units with
# few spikes: a tuning curve that is positive at one direction and zero elsewhere
# scores 1.0 no matter how few spikes produced it. Subtracting the per-unit null
# mean makes the statistic comparable across units and across areas that differ in
# firing rate, which cortex and thalamus certainly do.

# %%
if os.path.exists("units.pkl"):
    U = pd.read_pickle("units.pkl")
else:
    U = pd.concat([pipeline.analyze_session(f"extracted/{s}.npz")
                   for s in tqdm(SESSIONS, desc="analysing")], ignore_index=True)
    U.to_pickle("units.pkl")

R = responsive(U)
print(f"{len(U)} units from {U.session.nunique()} sessions ({U.subject.nunique()} mice); "
      f"{len(R)} visually responsive")
print(R.groupby("area").apply(lambda g: pd.Series({
    "n": len(g), "frac_sig_p<0.01": (g.p_perm < 0.01).mean(),
    "median_gOSI": g.gosi.median(), "median_gOSI_noise_corrected": g.gosi_corrected.median(),
    "median_OSI_classic": g.osi_classic.median(), "median_gDSI": g.gdsi.median(),
}), include_groups=False).to_string())

# %%
AREAS = [a for a in oslib.VISUAL_CORTEX + oslib.THALAMUS if (R.area == a).sum() >= 15]
ACOL = {a: (CORTEX_C if a in oslib.VISUAL_CORTEX else THAL_C) for a in AREAS}


def _load(suffix):
    return {p.split("/")[-1].split("_")[0]: np.load(p)
            for p in glob.glob(f"extracted/*_{suffix}.npy")}


TUN, TUN_RAW, TUN_SEM = _load("tuning"), _load("tuningraw"), _load("tuningsem")
HALVES = {}
for p_ in glob.glob("extracted/*_half*.npy"):
    ses_, which = p_.split("/")[-1].split("_half")
    HALVES.setdefault(ses_, {})[int(which[0]) - 1] = np.load(p_)

_UIDX = {}


def unit_index(row):
    if row.session not in _UIDX:
        d = np.load(f"extracted/{row.session}.npz", allow_pickle=True)
        _UIDX[row.session] = {int(u): i for i, u in enumerate(d["unit_ids"])}
    return _UIDX[row.session][int(row.unit_id)]


# %% [markdown]
# ## 6. Tuning curves
#
# Twelve units in polar coordinates. Curves are raw firing rate, not
# baseline-subtracted, so that the spontaneous rate is visible as a reference
# circle: a baseline-subtracted polar plot clipped at zero looks far more dramatic
# than the data warrant. Rows are the most selective cortical units, median
# cortical units, and the most selective LGd units.

# %%
strong = R[R.peak_evoked > 8.0]
cx = strong[(strong.region == "cortex") & (strong.p_perm < 0.01)].sort_values(
    "gosi_corrected", ascending=False)
lg = strong[strong.area == "LGd"].sort_values("gosi_corrected", ascending=False)
mid = len(cx) // 2
picks = pd.concat([cx.head(4), cx.iloc[mid:mid + 4], lg.head(4)])
ROW_LABELS = ["most selective\ncortical units", "median\ncortical units",
              "most selective\nLGd units"]

fig, axes = plt.subplots(3, 4, figsize=(15, 12), subplot_kw={"projection": "polar"})
th = np.deg2rad(np.r_[DIRS, DIRS[0]])
fine = np.linspace(0, 2 * np.pi, 200)
for ax, (_, row) in zip(axes.ravel(), picks.iterrows()):
    i = unit_index(row)
    t, e = TUN_RAW[row.session][i], TUN_SEM[row.session][i]
    col = ACOL.get(row.area, "0.4")
    ax.plot(th, np.r_[t, t[0]], "-o", color=col, ms=5, lw=2)
    ax.fill_between(th, np.r_[t - e, t[0] - e[0]], np.r_[t + e, t[0] + e[0]],
                    color=col, alpha=0.25, lw=0)
    ax.plot(fine, np.full_like(fine, row.baseline_rate), color="0.35", lw=1.2, ls=":")
    top = np.nanmax(t + e) * 1.05
    for a in (row.pref_ori, row.pref_ori + 180):
        ax.plot([np.deg2rad(a)] * 2, [0, top], color="k", lw=1.4, ls="--", alpha=0.7)
    ax.set_ylim(0, top)
    ax.set_title(f"{row.area}  unit {row.unit_id}\ngOSI {row.gosi:.2f}   "
                 f"pref {row.pref_ori:.0f}°   p {row.p_perm:.3f}", fontsize=10, pad=22)
    ax.set_theta_zero_location("E")
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
for r_, lab in enumerate(ROW_LABELS):
    axes[r_, 0].text(-0.32, 0.5, lab, transform=axes[r_, 0].transAxes, rotation=90,
                     va="center", ha="center", fontsize=11, fontweight="bold")
fig.suptitle("Direction tuning curves at each unit's preferred temporal frequency\n"
             "firing rate in Hz +/- SEM; dotted circle = spontaneous rate on blank sweeps; "
             "dashed line = preferred orientation axis\n"
             "gOSI is computed on the response *above* the dotted circle, so a curve that "
             "looks round on a high baseline can still score high", fontsize=13, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig03_polar_gallery.png", dpi=140)

# %% [markdown]
# ## 7. Orientation selectivity across the visual hierarchy
#
# Panel (d) deserves a note. Aligning every unit's tuning curve to its own peak and
# averaging will produce a peak even from pure noise, because the alignment uses
# the same data as the readout. Here the preferred bin is chosen from one half of
# the repeats and the curve is read out from the *other* half, so an untuned
# population gives a flat line and the comparison between areas is interpretable.

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
data = [R.loc[R.area == a, "gosi_corrected"].dropna().values for a in AREAS]
bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                medianprops=dict(color="k", lw=2))
for patch, a in zip(bp["boxes"], AREAS):
    patch.set_facecolor(ACOL[a])
    patch.set_alpha(0.6)
jit = np.random.default_rng(0)
for i, (a, v) in enumerate(zip(AREAS, data)):
    ax.plot(i + 1 + jit.uniform(-0.18, 0.18, len(v)), v, ".", color=ACOL[a], ms=3, alpha=0.45)
ax.set_xticks(np.arange(1, len(AREAS) + 1))
ax.set_xticklabels(AREAS, fontsize=9)
ax.set_ylabel("noise-corrected gOSI")
ax.set_title("(a) Orientation selectivity by area", loc="left", fontweight="bold")

ax = axes[0, 1]
frac, ci = [], []
for a in AREAS:
    g = R[R.area == a]
    k, n = (g.p_perm < 0.01).sum(), len(g)
    frac.append(k / n)
    lo, hi = stats.beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5)
    ci.append([k / n - lo, hi - k / n])
ax.bar(AREAS, frac, yerr=np.array(ci).T, color=[ACOL[a] for a in AREAS], capsize=4)
ax.axhline(0.01, color="k", ls=":", label="chance (p < 0.01)")
for i, a in enumerate(AREAS):
    ax.text(i, 0.015, f"n={(R.area == a).sum()}", ha="center", va="bottom", fontsize=8,
            color="white", fontweight="bold")
ax.set_ylabel("fraction significantly orientation tuned")
ax.set_title("(b) Significant tuning (permutation test, p < 0.01)", loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False)

ax = axes[1, 0]
for label, sub, col in [("visual cortex", R[R.region == "cortex"], CORTEX_C),
                        ("LGd (first-order relay)", R[R.area == "LGd"], THAL_C),
                        ("LP (higher-order)", R[R.area == "LP"], LP_C)]:
    v = np.sort(sub.gosi_corrected.dropna().values)
    ax.step(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=2.2,
            label=f"{label}, n={len(v)}, median {np.median(v):.3f}")
ctx = R.loc[R.region == "cortex", "gosi_corrected"].dropna()
lgd = R.loc[R.area == "LGd", "gosi_corrected"].dropna()
u_stat, p_ctx_lgd = stats.mannwhitneyu(ctx, lgd)
ax.set_xlabel("noise-corrected gOSI")
ax.set_ylabel("cumulative fraction of units")
ax.set_title(f"(c) Cortex vs thalamic relay\ncortex vs LGd: Mann-Whitney p = {p_ctx_lgd:.2g}",
             loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False, loc="lower right")

ax = axes[1, 1]
x = np.array([-90, -45, 0, 45])
for label, sub, col in [("visual cortex", R[R.region == "cortex"], CORTEX_C),
                        ("LGd", R[R.area == "LGd"], THAL_C)]:
    curves = []
    for _, row in sub.iterrows():
        i = unit_index(row)
        o1 = np.clip(HALVES[row.session][0][i], 0, None)
        o2 = np.clip(HALVES[row.session][1][i], 0, None)
        o1, o2 = o1[:4] + o1[4:], o2[:4] + o2[4:]
        if not (np.isfinite(o1).all() and np.isfinite(o2).all()) or o2.max() <= 0:
            continue
        curves.append(np.roll(o2, 2 - int(np.argmax(o1))) / o2.max())
    c = np.array(curves)
    m, sem = c.mean(axis=0), c.std(axis=0) / np.sqrt(len(c))
    ax.errorbar(np.r_[x, 90], np.r_[m, m[0]], yerr=np.r_[sem, sem[0]], marker="o",
                color=col, lw=2, capsize=3, label=f"{label} (n={len(c)})")
ax.set_xticks(np.arange(-90, 91, 45))
ax.set_xlabel("orientation relative to preferred (deg)")
ax.set_ylabel("response, normalised to each unit's peak")
ax.set_title("(d) Cross-validated population tuning\n(preferred bin from held-out repeats)",
             loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False)

fig.suptitle(f"Orientation selectivity across the mouse visual system "
             f"({R.session.nunique()} sessions, {R.subject.nunique()} mice, "
             f"{len(R)} visually responsive units)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig04_population.png", dpi=140)

# %% [markdown]
# ## 8. Is the preference real? Three controls
#
# A tuning curve computed from the same data that selected the preferred
# orientation proves very little on its own. Three independent checks:
#
# 1. **Split-half reliability.** Repeats of each condition are split odd/even and
#    the preferred orientation is estimated separately in each half.
# 2. **Cross-stimulus replication.** The preferred orientation from the
#    drifting-grating block is compared with the one from the static-grating block,
#    recorded later in the session with different spatial frequencies, different
#    presentation duration, and no motion at all.
# 3. **Locomotion.** Running strongly modulates gain in mouse visual cortex. gOSI
#    is recomputed on stationary and running trials separately. The two are not
#    identical: selectivity is modestly but significantly *higher* on stationary
#    trials, which is the expected consequence of locomotion boosting responses at
#    all orientations. What matters for the demonstration is that the preferred
#    orientation is essentially unchanged and the two estimates are strongly
#    correlated, so the tuning is not an artefact of running.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
sub = R[(R.p_perm < 0.01) & (R.region == "cortex")]
d_half = oslib.circ_dist_180(sub.pref_half1, sub.pref_half2)
ax.hist(d_half, bins=np.linspace(0, 90, 19), color=CORTEX_C, alpha=0.8, density=True)
ax.axhline(1 / 90, color="k", ls="--", label="chance (uniform)")
r_half, p_half, n_half = oslib.circ_corr_180(sub.pref_half1, sub.pref_half2)
ax.set_xlabel("|preferred orientation difference| between halves (deg)")
ax.set_ylabel("density")
ax.set_title(f"(a) Split-half reliability, cortex\ncircular r = {r_half:.2f}, "
             f"p < {p_half:.1e}, n = {n_half}", loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False)

ax = axes[1]
sub2 = R[(R.p_perm < 0.01) & (R.region == "cortex") & R.sg_pref_ori.notna() & (R.sg_gosi > 0.1)]
ax.plot(sub2.pref_ori, sub2.sg_pref_ori, "o", color=CORTEX_C, ms=4, alpha=0.55)
ax.plot([0, 180], [0, 180], "k--", lw=1)
r2, p2, n2 = oslib.circ_corr_180(sub2.pref_ori, sub2.sg_pref_ori)
ax.set_xlabel("preferred orientation, drifting gratings (deg)")
ax.set_ylabel("preferred orientation, static gratings (deg)")
ax.set_xlim(0, 180)
ax.set_ylim(0, 180)
ax.set_title(f"(b) Cross-stimulus replication, cortex\ncircular r = {r2:.2f}, "
             f"p < {p2:.1e}, n = {n2}", loc="left", fontweight="bold")

ax = axes[2]
sub3 = R[R.gosi_still.notna() & R.gosi_move.notna() & (R.p_perm < 0.01)]
for reg, col in [("cortex", CORTEX_C), ("thalamus", THAL_C)]:
    s = sub3[sub3.region == reg]
    ax.plot(s.gosi_still, s.gosi_move, "o", color=col, ms=4, alpha=0.5, label=reg)
lim = [0, max(sub3.gosi_still.max(), sub3.gosi_move.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1)
rr = stats.spearmanr(sub3.gosi_still, sub3.gosi_move)
wil = stats.wilcoxon(sub3.gosi_still, sub3.gosi_move)
pref_shift = np.nanmedian(oslib.circ_dist_180(sub3.pref_ori_still, sub3.pref_ori))
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("gOSI, stationary trials")
ax.set_ylabel("gOSI, running trials")
ax.set_title(f"(c) Locomotion control (n = {len(sub3)})\n"
             f"median gOSI {sub3.gosi_still.median():.2f} still vs "
             f"{sub3.gosi_move.median():.2f} running, p = {wil.pvalue:.0e}\n"
             f"Spearman r = {rr.statistic:.2f}; pref. orientation shifts {pref_shift:.0f}°",
             loc="left", fontweight="bold", fontsize=10)
ax.legend(fontsize=9, frameon=False)

fig.suptitle("Orientation preference is reliable, replicates on an independent stimulus, "
             "and survives restricting to stationary trials", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig05_controls.png", dpi=140)

# %% [markdown]
# ## 9. Which orientations are preferred?
#
# Mouse visual cortex is reported to over-represent the cardinal orientations,
# particularly horizontal. The distribution of preferred orientations is plotted
# mirrored, because orientation has period 180°, and tested for a bias with a
# Rayleigh test on the doubled angles.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), subplot_kw={"projection": "polar"})
ray = {}
for ax, (label, sub, col) in zip(axes, [
        ("visual cortex", R[(R.region == "cortex") & (R.p_perm < 0.01)], CORTEX_C),
        ("visual thalamus", R[(R.region == "thalamus") & (R.p_perm < 0.01)], THAL_C)]):
    edges = np.arange(0, 181, 15)
    h, _ = np.histogram(sub.pref_ori, bins=edges)
    ax.bar(np.deg2rad(np.arange(0, 360, 15)) + np.deg2rad(7.5), np.r_[h, h],
           width=np.deg2rad(14), color=col, alpha=0.8)
    a2 = 2 * np.deg2rad(sub.pref_ori.values)
    Rbar, n_ray = np.abs(np.mean(np.exp(1j * a2))), len(a2)
    p_ray = np.exp(np.sqrt(1 + 4 * n_ray + 4 * (n_ray ** 2 - (n_ray * Rbar) ** 2)) - (1 + 2 * n_ray))
    ray[label] = (Rbar, p_ray, n_ray)
    ax.set_title(f"{label} (n={n_ray})\nRayleigh R = {Rbar:.3f}, p = {p_ray:.2g}", pad=25)
    ax.set_theta_zero_location("E")
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
fig.suptitle("Distribution of preferred orientations (mirrored: orientation has period 180°)",
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig("fig06_preferred_orientation.png", dpi=140)

# %% [markdown]
# ## 10. Population decoding and a NeMoS encoding GLM
#
# Two questions single-unit tuning curves cannot answer.
#
# **Decoding.** A linear discriminant reports which of the four orientations was on
# the screen from the single-trial spike counts of N randomly chosen units, with
# 5-fold cross-validation and a shuffled-label control. Sweeping N matters: at 32
# units both cortex and LGd are near ceiling, so a single population size would say
# little. The slope at small N is what reveals per-neuron information.
#
# **Encoding.** A NeMoS Poisson GLM models each unit's single-trial spike count from
# temporal frequency alone, or from temporal frequency plus orientation. Orientation
# enters through a cyclic B-spline basis evaluated at twice the direction, making
# the basis 180° periodic and therefore blind to direction: the full model can only
# improve by capturing orientation tuning. The reported quantity is the gain in
# held-out Poisson log-likelihood as a McFadden pseudo-$R^2$, so unlike gOSI it
# cannot be inflated by fitting the tuning curve to noise.
#
# These are computed by `06_decode.py` and `07_glm.py`, which are run here if their
# result files are missing.

# %%
if not os.path.exists("decoding_results.csv"):
    os.system("python 06_decode.py")
if not os.path.exists("glm_results.csv"):
    os.system("python 07_glm.py")
dec = pd.read_csv("decoding_results.csv")
glm = pd.read_csv("glm_results.csv")

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))

ax = axes[0]
for label, col in [("visual cortex", CORTEX_C), ("LGd", THAL_C)]:
    sub = dec[dec.group == label]
    if not len(sub):
        continue
    g = sub.groupby("n_units")["accuracy"]
    ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values, marker="o",
                color=col, lw=2, capsize=3, label=label)
    gs = sub.groupby("n_units")["shuffled"].mean()
    ax.plot(gs.index, gs.values, "--", color=col, lw=1.2, alpha=0.6)
ax.axhline(0.25, color="k", ls=":", lw=1)
ax.text(1.0, 0.265, "chance (4 orientations)", ha="left", fontsize=9)
ax.set_xscale("log", base=2)
ax.set_xticks(sorted(dec.n_units.unique()))
ax.set_xticklabels(sorted(dec.n_units.unique()))
ax.set_xlabel("number of units in the population")
ax.set_ylabel("decoding accuracy (5-fold CV)")
ax.set_ylim(0.15, 1.02)
ax.set_title("(a) Orientation decoded from spike counts\n"
             "(dashed = same decoder on shuffled labels)", loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False, loc="upper left")

ax = axes[1]
for label, sub, col in [("visual cortex", glm[glm.region == "cortex"], CORTEX_C),
                        ("LGd", glm[glm.area == "LGd"], THAL_C),
                        ("LP", glm[glm.area == "LP"], LP_C)]:
    if not len(sub):
        continue
    v = np.sort(sub.dR2_orientation.dropna().values)
    ax.step(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=2.2,
            label=f"{label}, n={len(v)}, median {np.median(v):.3f}")
ax.axvline(0, color="k", ls=":", lw=1)
gctx = glm.loc[glm.region == "cortex", "dR2_orientation"].dropna()
glgd = glm.loc[glm.area == "LGd", "dR2_orientation"].dropna()
u_glm, p_glm = stats.mannwhitneyu(gctx, glgd)
ax.set_xlabel("held-out pseudo-$R^2$ gained by adding orientation")
ax.set_ylabel("cumulative fraction of units")
ax.set_title(f"(b) NeMoS Poisson GLM, cross-validated\ncortex vs LGd: "
             f"Mann-Whitney p = {p_glm:.2g}", loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False, loc="lower right")

ax = axes[2]
for label, sub, col in [("cortex", glm[glm.region == "cortex"], CORTEX_C),
                        ("thalamus", glm[glm.region == "thalamus"], THAL_C)]:
    ax.plot(sub.gosi_corrected, sub.dR2_orientation, "o", color=col, ms=4, alpha=0.45, label=label)
ok = glm.gosi_corrected.notna() & glm.dR2_orientation.notna()
rho = stats.spearmanr(glm.gosi_corrected[ok], glm.dR2_orientation[ok])
ax.set_xlabel("noise-corrected gOSI (tuning-curve statistic)")
ax.set_ylabel("GLM orientation pseudo-$R^2$ gain")
ax.set_title(f"(c) Two independent measures agree\nSpearman r = {rho.statistic:.2f}, "
             f"p = {rho.pvalue:.1e}, n = {ok.sum()}", loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False)

fig.suptitle("Population decoding and single-trial encoding of orientation", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("fig07_decoding_glm.png", dpi=140)

# %% [markdown]
# ## 11. Summary

# %%
lines = [
    f"sessions: {sorted(U.session.unique())}",
    f"mice: {U.subject.nunique()}   units passing QC: {len(U)}   visually responsive: {len(R)}",
    "",
    R.groupby("area").apply(lambda g: pd.Series({
        "n": len(g), "frac_sig_p<0.01": (g.p_perm < 0.01).mean(),
        "median_gOSI": g.gosi.median(), "median_gOSI_noise_corrected": g.gosi_corrected.median(),
        "median_OSI_classic": g.osi_classic.median(), "median_gDSI": g.gdsi.median(),
    }), include_groups=False).to_string(),
    "",
    f"cortex vs LGd noise-corrected gOSI: medians {ctx.median():.4f} vs {lgd.median():.4f}, "
    f"Mann-Whitney U={u_stat:.0f}, p={p_ctx_lgd:.3g}",
    f"cortex fraction significantly tuned: {(R[R.region == 'cortex'].p_perm < 0.01).mean():.3f}; "
    f"LGd: {(R[R.area == 'LGd'].p_perm < 0.01).mean():.3f}",
    f"split-half circular r (cortex, tuned) = {r_half:.3f}, p < {p_half:.3g}, n = {n_half}",
    f"drifting vs static circular r (cortex, tuned) = {r2:.3f}, p < {p2:.3g}, n = {n2}",
    f"locomotion: median gOSI {sub3.gosi_still.median():.3f} stationary vs "
    f"{sub3.gosi_move.median():.3f} running, Wilcoxon p = {wil.pvalue:.3g}; "
    f"Spearman r = {rr.statistic:.3f}; median |preferred-orientation shift| = "
    f"{pref_shift:.1f} deg, n = {len(sub3)}",
    "",
    "preferred-orientation Rayleigh test (doubled angles):",
    *[f"  {k}: R = {v[0]:.3f}, p = {v[1]:.3g}, n = {v[2]}" for k, v in ray.items()],
    "",
    "population decoding (4 orientations, chance 0.25):",
    dec.groupby(["group", "n_units"])[["accuracy", "shuffled"]].mean().to_string(),
    "",
    "NeMoS GLM, held-out pseudo-R2 gained by orientation:",
    glm.groupby("area")["dR2_orientation"].agg(["count", "median", "mean"]).to_string(),
    f"cortex vs LGd: medians {gctx.median():.4f} vs {glgd.median():.4f}, "
    f"Mann-Whitney U={u_glm:.0f}, p={p_glm:.3g}",
    f"gOSI vs GLM gain: Spearman r = {rho.statistic:.3f}, p = {rho.pvalue:.3g}",
]
summary = "\n".join(lines)
open("stats_summary.txt", "w").write(summary + "\n")
print(summary)

# %% [markdown]
# ### What the numbers say
#
# Orientation selectivity is unambiguously present in mouse visual cortex in this
# dataset. A majority of visually responsive cortical units are significantly
# orientation tuned by a permutation test that controls for firing rate and for
# temporal-frequency tuning, the median classic OSI in V1 is in the range reported
# in the mouse V1 literature, and preferred orientation is highly reproducible both
# across independent halves of the drifting-grating repeats and across the separate
# static-grating block, which shares no stimulus parameters with it beyond
# orientation itself. The effect survives restricting the analysis to trials where
# the mouse was stationary.
#
# The comparison with the thalamus is the more interesting result and it is a
# graded one rather than an all-or-none one. LGd, the first-order relay that feeds
# V1, has clearly weaker orientation selectivity than cortex by every measure used
# here, but it is not zero: a minority of LGd units are significantly tuned, and
# populations of LGd units support orientation decoding well above chance. That is
# consistent with the published finding that mouse dLGN contains a genuine
# orientation- and direction-selective subpopulation, and it means the textbook
# statement that orientation selectivity first appears in cortex should be read as
# a statement about degree rather than about presence and absence. LP, a
# higher-order nucleus that receives heavy cortical input, sits closer to cortex
# than to LGd, which is what its connectivity predicts.
#
# ### Limitations
#
# The unit counts are unequal across areas because probe placement targeted cortex,
# so LGd is the most sparsely sampled structure here and its confidence intervals
# are correspondingly wide. gOSI is compared after subtracting a per-unit
# permutation noise floor, which corrects the leading rate-dependent bias but is
# not a complete correction. Receptive-field position relative to the monitor is
# not controlled: units whose receptive field falls near the edge of the screen see
# a truncated grating, which adds noise to the tuning estimates in every area.
