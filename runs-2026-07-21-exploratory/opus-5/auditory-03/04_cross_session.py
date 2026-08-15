"""Step 4: pooled results across all sessions, and the decoding figures.

Writes figures/04_cross_session.png and figures/05_decoding.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps

from common import FIGDIR, FREQS, FREQ_LABELS, RESDIR

os.makedirs(FIGDIR, exist_ok=True)

units = pd.read_csv(f"{RESDIR}/units_all.csv")
summary = pd.read_csv(f"{RESDIR}/session_summary.csv")
dec = np.load(f"{RESDIR}/decoding.npz", allow_pickle=True)
tuned = units[units.freq_tuned]
tune_cols = [f"tune_{int(f)}" for f in FREQS]
FCOLORS = colormaps["viridis"](np.linspace(0, 0.92, len(FREQS)))

print(f"{len(units)} units, {len(summary)} sessions, {units.subject.nunique()} mice")
print(f"responsive {100 * units.responsive.mean():.0f}%, "
      f"tuned {100 * units.freq_tuned.mean():.0f}%")
print(f"per-session tuned fraction: {summary.frac_tuned.min():.2f}-"
      f"{summary.frac_tuned.max():.2f}")
print(f"pooled BF counts: "
      f"{tuned.bf_hz.value_counts().sort_index().astype(int).to_dict()}")
print(f"BF split-half match {100 * tuned.bf_split_match.mean():.0f}% (chance 20%)")
print(f"median SI {tuned.selectivity_index.median():.2f}, "
      f"sparseness {tuned.sparseness.median():.2f}, "
      f"latency {1e3 * tuned.latency_s.median():.0f} ms")

# ----------------------------------------------------------------- figure 4
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.32)

# (a) yield per session
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(summary))
ax.bar(x, 100 * summary.n_responsive / summary.n_units, color="0.75",
       label="tone-responsive")
ax.bar(x, 100 * summary.n_tuned / summary.n_units, color="tab:red",
       label="frequency-tuned")
ax.set_xticks(x)
ax.set_xticklabels(summary.session, rotation=90, fontsize=7)
ax.set_ylabel("% of units")
ax.set_title("(a) Yield per session", fontsize=11)
ax.legend(fontsize=8, loc="lower left")

# (b) pooled BF distribution, per mouse
ax = fig.add_subplot(gs[0, 1])
mice = sorted(tuned.subject.unique())
bottom = np.zeros(len(FREQS))
for m in mice:
    c = np.array([(tuned[(tuned.subject == m)].bf_hz == f).sum() for f in FREQS])
    ax.bar(range(len(FREQS)), c, bottom=bottom, label=m)
    bottom += c
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FREQ_LABELS)
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("units")
ax.set_title(f"(b) Preferred frequency, {len(tuned)} tuned units", fontsize=11)
ax.legend(fontsize=7, ncol=2, title="mouse", title_fontsize=7)

# (c) BF-aligned tuning, pooled and per session
ax = fig.add_subplot(gs[0, 2])
oct_axis = np.arange(-4, 5)
T = tuned[tune_cols].values
bf = tuned.bf_idx.values
aligned = np.full((len(tuned), oct_axis.size), np.nan)
for j in range(len(tuned)):
    peak = T[j, bf[j]]
    for k in range(len(FREQS)):
        aligned[j, np.flatnonzero(oct_axis == k - bf[j])[0]] = T[j, k] / peak
for s in summary.session:
    sel = (tuned.session == s).values
    m = np.nanmean(aligned[sel], axis=0)
    ax.plot(oct_axis, m, color="0.8", lw=0.8)
m = np.nanmean(aligned, axis=0)
se = np.nanstd(aligned, axis=0) / np.sqrt(np.sum(~np.isnan(aligned), axis=0))
ax.errorbar(oct_axis, m, yerr=se, marker="o", color="tab:red", capsize=3,
            label="all sessions pooled")
ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
ax.axhline(0, color="0.6", lw=0.8, ls=":")
ax.set_xlabel("distance from BF (octaves)")
ax.set_ylabel("evoked rate / peak")
ax.set_title("(c) BF-aligned tuning\n(grey = individual sessions)", fontsize=11)
ax.legend(fontsize=8)

# (d) selectivity per mouse
ax = fig.add_subplot(gs[1, 0])
data = [tuned[tuned.subject == m].selectivity_index.dropna() for m in mice]
ax.boxplot(data, labels=mice, showfliers=False)
ax.set_ylabel("frequency selectivity index")
ax.set_xlabel("mouse")
ax.set_ylim(0, 1)
ax.set_title("(d) Selectivity by mouse", fontsize=11)

# (e) split-half BF reproducibility
ax = fig.add_subplot(gs[1, 1])
ax.bar(x, 100 * summary.bf_match, color="tab:blue")
ax.axhline(20, color="k", ls="--", lw=1, label="chance (5 frequencies)")
ax.set_xticks(x)
ax.set_xticklabels(summary.session, rotation=90, fontsize=7)
ax.set_ylabel("% tuned units with matching BF")
ax.set_title("(e) BF reproducibility, odd vs even trials", fontsize=11)
ax.legend(fontsize=8)

# (f) latency and tuning-curve reliability
ax = fig.add_subplot(gs[1, 2])
ax.hist(1e3 * tuned.latency_s.dropna(), bins=np.arange(0, 62, 2),
        color="tab:blue")
med = 1e3 * tuned.latency_s.median()
ax.axvline(med, color="k", ls="--", label=f"median {med:.0f} ms")
ax.set_xlabel("onset latency at BF (ms)")
ax.set_ylabel("units")
ax.set_title("(f) Response latency, pooled", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle(f"Frequency tuning across {len(summary)} sessions / "
             f"{units.subject.nunique()} mice — DANDI:000986 ({len(units)} units)",
             fontsize=13, y=0.965)
fig.savefig(f"{FIGDIR}/04_cross_session.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {FIGDIR}/04_cross_session.png")

# ----------------------------------------------------------------- figure 5
names = list(dec["names"])
conf = dec["confusions"]
curve = dec["acc_curve"]
sizes = dec["subset_sizes"]
pooled = conf.mean(axis=0)

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2),
                         gridspec_kw=dict(wspace=0.42))

ax = axes[0]
im = ax.imshow(pooled, cmap="magma", vmin=0, vmax=1)
for i in range(len(FREQS)):
    for j in range(len(FREQS)):
        ax.text(j, i, f"{pooled[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="w" if pooled[i, j] < 0.6 else "k")
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FREQ_LABELS)
ax.set_yticks(range(len(FREQS)))
ax.set_yticklabels(FREQ_LABELS)
ax.set_xlabel("decoded frequency (kHz)")
ax.set_ylabel("presented frequency (kHz)")
ax.set_title("(a) Confusion matrix\nmean of 15 sessions", fontsize=11)
plt.colorbar(im, ax=ax, label="P(decoded | presented)")

ax = axes[1]
ax.bar(x, 100 * summary.decode_acc, color="tab:green", label="decoder")
ax.bar(x, 100 * summary.decode_acc_shuffled, color="0.5",
       label="shuffled labels")
ax.axhline(20, color="k", ls="--", lw=1, label="chance")
ax.set_xticks(x)
ax.set_xticklabels(summary.session, rotation=90, fontsize=7)
ax.set_ylabel("5-way accuracy (%)")
ax.set_title("(b) Single-trial decoding", fontsize=11)
ax.legend(fontsize=8, loc="lower left")

ax = axes[2]
ok = ~np.isnan(curve).all(axis=1)
ax.errorbar(sizes[ok], 100 * np.nanmean(curve[ok], axis=1),
            yerr=100 * np.nanstd(curve[ok], axis=1), marker="o",
            color="tab:green", capsize=3)
ax.axhline(20, color="k", ls="--", lw=1, label="chance")
ax.set_xscale("log")
ax.set_xlabel("units in the decoded population")
ax.set_ylabel("accuracy (%)")
ax.set_title("(c) Accuracy vs population size\n(LA11_ses-1, 5 draws per size)",
             fontsize=11)
ax.legend(fontsize=8)

ax = axes[3]
d = np.log2(FREQS[None, :] / FREQS[:, None])
err = pooled.copy()
np.fill_diagonal(err, np.nan)
err = err / np.nansum(err, axis=1, keepdims=True)   # error distribution only
offs = np.unique(d[d != 0])
p = np.array([np.nanmean(err[d == o]) for o in offs])
ax.bar(offs, p, width=0.6, color="tab:purple")
ax.axhline(1 / (len(FREQS) - 1), color="k", ls="--", lw=1,
           label="uniform errors")
ax.set_xlabel("decoded − presented (octaves)")
ax.set_ylabel("fraction of errors")
ax.set_title("(d) Where the errors go", fontsize=11)
ax.legend(fontsize=8)
near = np.nanmean(err[np.abs(d) == 1])
far = np.nanmean(err[np.abs(d) == 4])
print(f"error structure: {near:.2f} of errors at 1 octave vs {far:.2f} at "
      f"4 octaves (uniform would be {1 / (len(FREQS) - 1):.2f})")

fig.suptitle("Single-trial decoding of tone frequency from the 5-55 ms "
             "population response", fontsize=13, y=1.02)
fig.savefig(f"{FIGDIR}/05_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {FIGDIR}/05_decoding.png")
print(f"decoding accuracy {100 * summary.decode_acc.mean():.0f}% "
      f"(range {100 * summary.decode_acc.min():.0f}-"
      f"{100 * summary.decode_acc.max():.0f}%), "
      f"shuffled {100 * summary.decode_acc_shuffled.mean():.1f}%")
