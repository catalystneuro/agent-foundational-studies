"""Population-level figures pooled over all 15 sessions / 5 mice of DANDI 000986."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import dandi_auditory as da

d = np.load("population_results.npz", allow_pickle=True)
freqs = d["freqs"]
n_freq = len(freqs)
tuning, tuning_even = d["tuning"], d["tuning_even"]
psth, bins = d["psth"], d["bins"]
responsive = d["responsive_p"] < 0.01
tuned = d["tuned_p"] < 0.01
bf, bf_odd = d["bf"], d["bf_odd"]
subject, session = d["subject"], d["session"]
n_units = tuning.shape[1]
print(f"{n_units} units, {responsive.sum()} responsive, {tuned.sum()} frequency-tuned")

sel = responsive & tuned                       # units used for the tuning population
oct_axis = np.log2(freqs / freqs[0])

# ---------------------------------------------------------------- figure 4
fig = plt.figure(figsize=(14.5, 8))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.55, width_ratios=[1.15, 1, 1])

# (a) heatmap of tuning curves normalized to each unit's peak, sorted by best frequency
ax = fig.add_subplot(gs[0, 0])
T = tuning[:, sel].T
T = T / (np.abs(T).max(axis=1, keepdims=True) + 1e-12)
order = np.lexsort((-T.max(axis=1), bf[sel]))
im = ax.imshow(T[order], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
               interpolation="nearest", extent=[-0.5, n_freq - 0.5, len(order), 0])
ax.set_xticks(range(n_freq))
ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("unit (sorted by best frequency)")
ax.set_title("(a) Peak-normalized tuning,\n%d tuned units" % len(order), fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.046)
cb.set_label("rate / peak", fontsize=8)

# (b) mean tuning curve of each best-frequency group (BF from odd trials,
#     curve from even trials, so the peak is not a selection artifact)
ax = fig.add_subplot(gs[0, 1])
for i, f in enumerate(freqs):
    grp = sel & (bf_odd == i)
    if grp.sum() < 5:
        continue
    E = tuning_even[:, grp]
    E = E / (np.abs(E).max(axis=0, keepdims=True) + 1e-12)
    m, e = E.mean(axis=1), stats.sem(E, axis=1)
    ax.plot(oct_axis, m, color=da.FREQ_COLORS[i], marker="o", lw=1.8,
            label=f"BF {f/1000:g} kHz (n={grp.sum()})")
    ax.fill_between(oct_axis, m - e, m + e, color=da.FREQ_COLORS[i], alpha=0.25)
ax.set_xticks(oct_axis)
ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("normalized evoked rate")
ax.set_title("(b) Cross-validated tuning by BF group", fontsize=10)
ax.legend(fontsize=7, frameon=False)

# (c) BF distribution per mouse
ax = fig.add_subplot(gs[0, 2])
subs = sorted(set(subject[sel]))
width = 0.8 / len(subs)
for j, sb in enumerate(subs):
    grp = sel & (subject == sb)
    frac = np.array([np.mean(bf[grp] == i) for i in range(n_freq)])
    ax.bar(np.arange(n_freq) + j * width - 0.4, frac, width=width,
           label=f"{sb} (n={grp.sum()})")
ax.set_xticks(range(n_freq))
ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("fraction of tuned units")
ax.set_title("(c) Best-frequency distribution per mouse", fontsize=10)
ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center")

# (d) population PSTH at BF vs worst frequency
ax = fig.add_subplot(gs[1, 0])
idx = np.where(sel)[0]
at_bf = np.stack([psth[bf[i], :, i] for i in idx])
worst = np.argmin(tuning, axis=0)
at_worst = np.stack([psth[worst[i], :, i] for i in idx])
base = at_bf[:, bins < 0].mean(axis=1, keepdims=True)
for arr, c, lab in [(at_bf - base, "#882255", "best frequency"),
                    (at_worst - base, "#4477AA", "worst frequency")]:
    m, e = arr.mean(0), stats.sem(arr, axis=0)
    ax.plot(bins, m, color=c, lw=2, label=lab)
    ax.fill_between(bins, m - e, m + e, color=c, alpha=0.3)
ax.axvspan(0, 0.025, color="0.9", zorder=0)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("time from tone onset (s)")
ax.set_ylabel("evoked rate (Hz)")
ax.set_title("(d) Population PSTH", fontsize=10)
ax.legend(fontsize=8, frameon=False)

# (e) selectivity: sparseness of the tuning curve
ax = fig.add_subplot(gs[1, 1])
sp = d["sparse"]
ax.hist(sp[sel & np.isfinite(sp)], bins=30, color="#117733", alpha=0.85, label="tuned")
ax.hist(sp[responsive & ~tuned & np.isfinite(sp)], bins=30, color="0.7",
        alpha=0.8, label="untuned")
ax.set_xlabel("lifetime sparseness of evoked rate\nacross the 5 tones")
ax.set_ylabel("# units")
ax.set_title("(e) Tuning selectivity\nmedian = %.2f (tuned units)"
             % np.nanmedian(sp[sel]), fontsize=10)
ax.legend(fontsize=8, frameon=False)

# (f) per-session yield
ax = fig.add_subplot(gs[1, 2])
summ = pd.read_csv("session_summary.csv")
x = np.arange(len(summ))
ax.bar(x, summ["n_units"], color="0.8", label="recorded")
ax.bar(x, summ["n_responsive"], color="#4477AA", label="sound-responsive")
ax.bar(x, summ["n_tuned"], color="#882255", label="frequency-tuned")
ax.set_xticks(x)
ax.set_xticklabels([p.split("/")[1].replace("_behavior.nwb", "").replace("sub-", "")
                    for p in summ["path"]], rotation=90, fontsize=6)
ax.set_ylabel("# units")
ax.set_title("(f) Yield per session", fontsize=10)
ax.legend(fontsize=7, frameon=False)

fig.suptitle("Frequency tuning in mouse auditory cortex, DANDI 000986 "
             f"({len(subs)} mice, {len(summ)} sessions, {n_units} units)", y=0.97)
fig.savefig("fig04_population_tuning.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 5
# split-half reliability of the best frequency: the acid test that tuning is real
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))

bf_e = d["bf_even"]
M = np.zeros((n_freq, n_freq))
for a, b in zip(bf_odd[sel], bf_e[sel]):
    M[a, b] += 1
M = M / M.sum(axis=1, keepdims=True)
im = axes[0].imshow(M, cmap="magma", vmin=0, vmax=1)
axes[0].set_xticks(range(n_freq)); axes[0].set_xticklabels([f"{f/1000:g}" for f in freqs])
axes[0].set_yticks(range(n_freq)); axes[0].set_yticklabels([f"{f/1000:g}" for f in freqs])
axes[0].set_xlabel("BF from even trials (kHz)")
axes[0].set_ylabel("BF from odd trials (kHz)")
agree = np.mean(bf_odd[sel] == bf_e[sel])
axes[0].set_title("BF is stable across trial halves\n%.0f%% agreement (chance %.0f%%)"
                  % (100 * agree, 100 / n_freq), fontsize=10)
fig.colorbar(im, ax=axes[0], fraction=0.046)

# evoked vs baseline rate
axes[1].scatter(d["baseline"][~sel], d["evoked"][~sel], s=6, color="0.7", label="other")
axes[1].scatter(d["baseline"][sel], d["evoked"][sel], s=6, color="#882255", label="tuned")
lim = [0.05, max(d["evoked"].max(), d["baseline"].max()) * 1.2]
axes[1].plot(lim, lim, "k--", lw=1)
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlim(*lim); axes[1].set_ylim(*lim)
axes[1].set_xlabel("baseline rate (Hz)")
axes[1].set_ylabel("tone-evoked rate (Hz)")
axes[1].set_title("Tones drive the population", fontsize=10)
axes[1].legend(fontsize=8, frameon=False)

# modulation depth
depth = (tuning.max(axis=0) - tuning.min(axis=0))
axes[2].hist(np.log10(depth[sel] + 1e-3), bins=40, color="#882255", alpha=0.85,
             label="tuned")
axes[2].hist(np.log10(depth[~tuned] + 1e-3), bins=40, color="0.7", alpha=0.8,
             label="untuned")
axes[2].set_xlabel("log10 best-minus-worst evoked rate (Hz)")
axes[2].set_ylabel("# units")
axes[2].set_title("Depth of frequency modulation\nmedian %.1f Hz (tuned)"
                  % np.median(depth[sel]), fontsize=10)
axes[2].legend(fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig("fig05_tuning_validation.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("BF split-half agreement %.3f (chance %.3f)" % (agree, 1 / n_freq))
print("median modulation depth %.2f Hz" % np.median(depth[sel]))
print("wrote fig04_population_tuning.png, fig05_tuning_validation.png")
