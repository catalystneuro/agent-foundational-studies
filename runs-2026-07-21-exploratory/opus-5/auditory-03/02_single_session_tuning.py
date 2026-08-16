"""Step 2: frequency tuning of single units in one session.

Writes figures/02_example_units.png and figures/03_session_population.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib import colormaps

from common import (BASELINE_WIN, EVOKED_WIN, EXAMPLE_SESSION, FIGDIR, FREQS,
                    FREQ_LABELS, RESDIR, TONE_DUR)
from tuning import analyze_session, psth_matrix

os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(RESDIR, exist_ok=True)

name, asset = EXAMPLE_SESSION
units, tmean, tsem, sess = analyze_session(name, asset)
trials, spikes = sess["trials"], sess["spikes"]
onsets = trials.start_time.values
freq_of_trial = trials.stim_frequency.values
units.to_csv(f"{RESDIR}/units_{name}.csv", index=False)

print(f"{name}: {len(units)} units")
print(f"  tone-responsive (FDR q<0.01): {units.responsive.sum()} "
      f"({units.enhanced.sum()} enhanced, {units.suppressed.sum()} suppressed)")
print(f"  frequency-tuned (Kruskal-Wallis, FDR q<0.01, enhanced): "
      f"{units.freq_tuned.sum()}")
tuned = units[units.freq_tuned]
print(f"  BF counts: {tuned.bf_hz.value_counts().sort_index().to_dict()}")
print(f"  median selectivity index {tuned.selectivity_index.median():.2f}, "
      f"sparseness {tuned.sparseness.median():.2f}")
print(f"  BF reproducible across trial halves in "
      f"{100 * tuned.bf_split_match.mean():.0f}% of tuned units (chance 20%)")
print(f"  median onset latency at BF: {1e3 * np.nanmedian(tuned.latency_s):.0f} ms")

FCOLORS = colormaps["viridis"](np.linspace(0, 0.92, len(FREQS)))

# ----------------------------------------------------------------- figure 2
# Three well-tuned example units with different preferred frequencies.
examples = []
for b in [0, 2, 4]:
    cand = tuned[(tuned.bf_idx == b)].sort_values("best_hz", ascending=False)
    if len(cand):
        examples.append(int(cand.iloc[0].unit))
print(f"  example units: {examples}")

bin_size = 0.002
edges = np.arange(-0.05, 0.150 + bin_size / 2, bin_size)
centers = (edges[:-1] + bin_size / 2) * 1e3
n_raster = 90   # trials per frequency drawn in the raster

fig, axes = plt.subplots(len(examples), 3, figsize=(14, 3.5 * len(examples)),
                         gridspec_kw=dict(width_ratios=[1.1, 1.1, 0.9], hspace=0.55,
                                          wspace=0.3))
for r, u in enumerate(examples):
    row = units[units.unit == u].iloc[0]
    ax = axes[r, 0]
    y = 0
    for fi, f in enumerate(FREQS):
        ons = onsets[freq_of_trial == f][:n_raster]
        peri = nap.compute_perievent(spikes[u], nap.Ts(ons), window=(-0.05, 0.15))
        for k in peri.index:
            t = peri[k].t
            ax.plot(t * 1e3, np.full(t.size, y), "|", color=FCOLORS[fi], ms=2.4,
                    mew=0.6)
            y += 1
        ax.axhline(y, color="0.8", lw=0.5)
    ax.axvspan(0, TONE_DUR * 1e3, color="tab:red", alpha=0.18, lw=0)
    ax.set_ylim(0, y)
    ax.set_xlim(-50, 150)
    ax.set_ylabel("trial (grouped by frequency)")
    ax.set_xlabel("time from tone onset (ms)")
    ax.set_title(f"unit {u}: raster, {n_raster} trials/frequency", fontsize=10)
    for fi, f in enumerate(FREQS):
        ax.text(152, (fi + 0.5) * n_raster, f"{FREQ_LABELS[fi]} kHz",
                color=FCOLORS[fi], va="center", fontsize=8)

    ax = axes[r, 1]
    for fi, f in enumerate(FREQS):
        ons = onsets[freq_of_trial == f]
        p = psth_matrix(nap.TsGroup({u: spikes[u]}), ons, edges)[0]
        ax.plot(centers, p / (len(ons) * bin_size), color=FCOLORS[fi], lw=1.2,
                label=f"{FREQ_LABELS[fi]} kHz")
    ax.axvspan(0, TONE_DUR * 1e3, color="tab:red", alpha=0.18, lw=0)
    ax.set_xlabel("time from tone onset (ms)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title(f"unit {u}: PSTH by frequency (all trials)", fontsize=10)
    if r == 0:
        ax.legend(fontsize=8, ncol=2)

    ax = axes[r, 2]
    i = int(np.flatnonzero(np.asarray(spikes.index) == u)[0])
    ax.errorbar(FREQS / 1e3, tmean[:, i], yerr=tsem[:, i], marker="o", color="k",
                capsize=3, lw=1.5)
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(FREQS / 1e3)
    ax.set_xticklabels(FREQ_LABELS)
    ax.set_xlabel("tone frequency (kHz)")
    ax.set_ylabel("evoked rate\n(Hz, baseline-subtracted)")
    ax.set_title(f"unit {u}: tuning curve\nBF = {row.bf_hz / 1e3:.0f} kHz, "
                 f"SI = {row.selectivity_index:.2f}, "
                 f"q = {row.q_frequency:.1e}", fontsize=10)

fig.suptitle(f"Frequency tuning of single auditory-cortex units — DANDI:000986, "
             f"{name}", fontsize=13, y=0.985)
fig.savefig(f"{FIGDIR}/02_example_units.png", dpi=150, bbox_inches="tight")
print(f"wrote {FIGDIR}/02_example_units.png")
plt.close(fig)

# ----------------------------------------------------------------- figure 3
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)

# (a) population tuning heatmap, units sorted by BF then by peak
ax = fig.add_subplot(gs[0, 0])
sel = units.freq_tuned.values
M = tmean[:, sel]
M = M / np.abs(M).max(axis=0, keepdims=True)
order = np.lexsort((-M.max(axis=0), np.argmax(M, axis=0)))
im = ax.imshow(M[:, order].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
               interpolation="nearest")
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FREQ_LABELS)
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("frequency-tuned unit (sorted by BF)")
ax.set_title("(a) Normalised tuning curves", fontsize=11)
plt.colorbar(im, ax=ax, label="evoked rate / peak")

# (b) BF distribution
ax = fig.add_subplot(gs[0, 1])
counts = [int((tuned.bf_idx == i).sum()) for i in range(len(FREQS))]
ax.bar(range(len(FREQS)), counts, color=FCOLORS)
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FREQ_LABELS)
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("units")
ax.set_title(f"(b) Preferred frequency, {len(tuned)} tuned units", fontsize=11)

# (c) split-half reproducibility of BF
ax = fig.add_subplot(gs[0, 2])
conf = np.zeros((len(FREQS), len(FREQS)))
for a, b in zip(tuned.bf_split_a, tuned.bf_split_b):
    conf[np.flatnonzero(FREQS == a)[0], np.flatnonzero(FREQS == b)[0]] += 1
conf = conf / conf.sum(axis=1, keepdims=True)
im = ax.imshow(conf, cmap="magma", vmin=0, vmax=1)
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FREQ_LABELS)
ax.set_yticks(range(len(FREQS)))
ax.set_yticklabels(FREQ_LABELS)
ax.set_xlabel("BF from odd trials (kHz)")
ax.set_ylabel("BF from even trials (kHz)")
ax.set_title(f"(c) BF reproducibility\n{100 * tuned.bf_split_match.mean():.0f}% match "
             f"(chance 20%)", fontsize=11)
plt.colorbar(im, ax=ax, label="fraction of row")

# (d) selectivity and sparseness
ax = fig.add_subplot(gs[1, 0])
ax.hist(tuned.selectivity_index.dropna(), bins=np.linspace(0, 1, 21),
        color="tab:purple", alpha=0.75, label="selectivity index")
ax.hist(tuned.sparseness.dropna(), bins=np.linspace(0, 1, 21), histtype="step",
        color="k", lw=1.5, label="lifetime sparseness")
ax.set_xlabel("value")
ax.set_ylabel("units")
ax.set_title("(d) Tuning selectivity", fontsize=11)
ax.legend(fontsize=8)

# (e) tuning curves aligned to each unit's BF
ax = fig.add_subplot(gs[1, 1])
oct_axis = np.arange(-4, 5)
aligned = np.full((sel.sum(), oct_axis.size), np.nan)
for j, i in enumerate(np.flatnonzero(sel)):
    b = int(units.bf_idx.values[i])
    peak = tmean[b, i]
    for k in range(len(FREQS)):
        aligned[j, np.flatnonzero(oct_axis == k - b)[0]] = tmean[k, i] / peak
m = np.nanmean(aligned, axis=0)
se = np.nanstd(aligned, axis=0) / np.sqrt(np.sum(~np.isnan(aligned), axis=0))
ok = np.sum(~np.isnan(aligned), axis=0) >= 5
ax.errorbar(oct_axis[ok], m[ok], yerr=se[ok], marker="o", color="tab:red", capsize=3)
ax.axhline(0, color="0.6", lw=0.8, ls=":")
ax.axhline(0.5, color="0.8", lw=0.8, ls="--")
ax.set_xlabel("distance from BF (octaves)")
ax.set_ylabel("evoked rate / peak")
ax.set_title("(e) BF-aligned population tuning", fontsize=11)

# (f) onset latency
ax = fig.add_subplot(gs[1, 2])
ax.hist(1e3 * tuned.latency_s.dropna(), bins=np.arange(0, 60, 2), color="tab:blue")
ax.axvline(1e3 * np.nanmedian(tuned.latency_s), color="k", ls="--",
           label=f"median {1e3 * np.nanmedian(tuned.latency_s):.0f} ms")
ax.set_xlabel("onset latency at BF (ms)")
ax.set_ylabel("units")
ax.set_title("(f) Response latency", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle(f"Auditory-cortex population, single session ({name})", fontsize=13,
             y=0.965)
fig.savefig(f"{FIGDIR}/03_session_population.png", dpi=150, bbox_inches="tight")
print(f"wrote {FIGDIR}/03_session_population.png")
