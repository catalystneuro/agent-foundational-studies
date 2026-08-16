"""Re-plot the population figures from cached results with corrected normalisation."""

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

with open("all_sessions.pkl", "rb") as fh:
    results = pickle.load(fh)
d = np.load("pooled_tuning.npz", allow_pickle=True)


def bh_fdr(p, q=0.05):
    n = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, n + 1) / n
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(n, bool)
    sig[order[:k]] = True
    return sig


ufreq = d["ufreq"]
delta = d["delta"]
spars, null_spars = d["spars"], d["null_spars"]
latency, rel = d["latency"], d["rel"]
sig_resp, sig_tune = bh_fdr(d["p_resp"]), bh_fdr(d["p_tune"])
excited = delta.max(axis=0) > 0
tuned = sig_resp & sig_tune & (d["p_spars"] < 0.05) & excited
bf_idx = np.argmax(delta, axis=0)

n_total = delta.shape[1]
print("units total %d | responsive %d | freq-tuned %d | + shuffle ctrl & excited %d (%.0f%%)"
      % (n_total, sig_resp.sum(), sig_tune.sum(), tuned.sum(), 100 * tuned.mean()))
print("suppressed-only units excluded: %d"
      % (sig_resp & sig_tune & (d["p_spars"] < 0.05) & ~excited).sum())
ok = np.isfinite(spars) & np.isfinite(null_spars)
print("sparseness: data %.3f vs shuffled %.3f, Wilcoxon p = %.3g"
      % (np.median(spars[ok]), np.median(null_spars[ok]),
         stats.wilcoxon(spars[ok], null_spars[ok])[1]))

# rectified, peak-normalised tuning
norm = np.clip(delta[:, tuned], 0, None)
norm = norm / norm.max(axis=0, keepdims=True)
sp_t = spars[tuned]
order = np.lexsort((-sp_t, bf_idx[tuned]))

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2),
                         gridspec_kw={"width_ratios": [1.25, 1, 1]})
im = axes[0].imshow(norm[:, order].T, aspect="auto", cmap="magma", vmin=0, vmax=1,
                    interpolation="nearest", extent=[-0.5, 4.5, norm.shape[1], 0])
axes[0].set_xticks(range(5))
axes[0].set_xticklabels(["%g" % (f / 1000) for f in ufreq])
axes[0].set_xlabel("tone frequency (kHz)")
axes[0].set_ylabel("unit (sorted by best frequency)")
axes[0].set_title("Peak-normalised tuning of %d tuned units\n(15 sessions, 5 mice)"
                  % tuned.sum())
plt.colorbar(im, ax=axes[0], label="evoked rate / peak")

cnt = np.bincount(bf_idx[tuned], minlength=5)
axes[1].bar(range(5), 100 * cnt / cnt.sum(), color="C0")
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(["%g" % (f / 1000) for f in ufreq])
axes[1].set_xlabel("best frequency (kHz)")
axes[1].set_ylabel("% of tuned units")
axes[1].set_title("Best frequencies tile the tested range")
axes[1].axhline(20, color="k", ls="--", lw=1)
axes[1].text(4.4, 20.6, "uniform", ha="right", fontsize=8)

bins = np.linspace(0, 1, 26)
axes[2].hist(spars[tuned], bins=bins, alpha=0.8, label="observed", color="C0")
axes[2].hist(null_spars[tuned], bins=bins, alpha=0.65,
             label="frequency-shuffled", color="0.5")
axes[2].set_xlabel("lifetime sparseness across frequencies")
axes[2].set_ylabel("units")
axes[2].set_title("Tuning strength vs shuffle control")
axes[2].legend(frameon=False)
for a in axes[1:]:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig05_population_tuning.png", dpi=140)
print("saved fig05")

# --- Figure 6 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
cmap = plt.get_cmap("viridis")
for i in range(5):
    m = tuned & (bf_idx == i)
    y = np.clip(delta[:, m], 0, None)
    y = y / y.max(axis=0, keepdims=True)
    axes[0].errorbar(ufreq / 1000, y.mean(axis=1),
                     yerr=y.std(axis=1) / np.sqrt(m.sum()),
                     marker="o", color=cmap(i / 4), capsize=3,
                     label="BF %g kHz (n=%d)" % (ufreq[i] / 1000, m.sum()))
axes[0].set_xscale("log", base=2)
axes[0].set_xticks(ufreq / 1000)
axes[0].set_xticklabels(["%g" % (f / 1000) for f in ufreq])
axes[0].set_xlabel("tone frequency (kHz)")
axes[0].set_ylabel("normalised evoked rate")
axes[0].set_title("Mean tuning curve, grouped by best frequency")
axes[0].set_ylim(0, 1.42)
axes[0].legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center")

frac = [100 * bh_fdr(r["p_tune"]).mean() for r in results]
names = [r["name"] for r in results]
subjects = sorted({r["subject"] for r in results})
cols = {s: "C%d" % i for i, s in enumerate(subjects)}
axes[1].bar(range(len(frac)), frac, color=[cols[r["subject"]] for r in results])
axes[1].set_xticks(range(len(frac)))
axes[1].set_xticklabels(names, rotation=90, fontsize=7)
axes[1].set_ylabel("% units frequency-tuned")
axes[1].set_title("Consistency across sessions (colour = mouse)")
axes[1].axhline(np.mean(frac), color="k", ls="--", lw=1)
handles = [plt.Rectangle((0, 0), 1, 1, color=cols[s]) for s in subjects]
axes[1].set_ylim(0, 122)
axes[1].legend(handles, subjects, fontsize=7, frameon=False, ncol=5, loc="upper left")

axes[2].hist(latency[tuned], bins=np.arange(0, 105, 5), color="C3")
axes[2].set_xlabel("onset latency at best frequency (ms)")
axes[2].set_ylabel("units")
axes[2].set_title("Response latency (median %.0f ms)" % np.nanmedian(latency[tuned]))
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig06_population_summary.png", dpi=140)
print("saved fig06")

np.savez("pooled_final.npz", tuned=tuned, bf_idx=bf_idx, excited=excited,
         sig_resp=sig_resp, sig_tune=sig_tune)
