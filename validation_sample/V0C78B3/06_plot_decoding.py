"""Figures for population decoding of tone frequency."""
import pickle

import matplotlib.pyplot as plt
import numpy as np

with open("decoding_results.pkl", "rb") as fh:
    res = pickle.load(fh)
sessions = list(res)
freqs = res[sessions[0]]["freqs"]
khz = [f"{int(f/1000)}" for f in freqs]

fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))

# ---- (a) confusion matrix, averaged over sessions ---------------------------
cm = np.mean([r["cm"] for r in res.values()], axis=0)
im = ax[0].imshow(cm, cmap="magma", vmin=0, vmax=cm.max())
ax[0].set_xticks(range(len(freqs)), khz)
ax[0].set_yticks(range(len(freqs)), khz)
ax[0].set_xlabel("decoded frequency (kHz)")
ax[0].set_ylabel("presented frequency (kHz)")
ax[0].set_title("a  confusion matrix (mean of 15 sessions)", loc="left", fontsize=10)
for i in range(len(freqs)):
    for j in range(len(freqs)):
        ax[0].text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                   color="w" if cm[i, j] < 0.6 * cm.max() else "k")
plt.colorbar(im, ax=ax[0], label="P(decoded | presented)")

# ---- (b) accuracy per session ------------------------------------------------
acc = np.array([res[s]["acc"] for s in sessions])
shuf = np.array([res[s]["acc_shuffled"] for s in sessions])
n_units = np.array([res[s]["n_units"] for s in sessions])
order = np.argsort(n_units)
x = np.arange(len(sessions))
ax[1].bar(x - 0.2, acc[order], 0.4, color="tab:red", label="observed")
ax[1].bar(x + 0.2, shuf[order], 0.4, color="0.6", label="shuffled labels")
ax[1].axhline(1 / len(freqs), color="k", ls="--", lw=1, label="chance (0.20)")
ax[1].set_xticks(x)
ax[1].set_xticklabels([f"{sessions[i]}\n({n_units[i]}u)" for i in order],
                      rotation=90, fontsize=6.5)
ax[1].set_ylabel("5-fold CV accuracy")
ax[1].set_title("b  single-trial frequency decoding", loc="left", fontsize=10)
ax[1].legend(fontsize=8, frameon=False)

# ---- (c) accuracy vs number of units -----------------------------------------
curve_sessions = [s for s in sessions if res[s]["curve"]]
for s in curve_sessions:
    c = res[s]["curve"]
    ns = sorted(c)
    ax[2].errorbar(ns, [c[n][0] for n in ns], yerr=[c[n][1] for n in ns],
                   marker="o", ms=3, lw=1.2, capsize=2, label=f"{s} ({res[s]['n_units']}u)")
ax[2].axhline(1 / len(freqs), color="k", ls="--", lw=1, label="chance")
ax[2].set_xscale("log")
ax[2].set_xlabel("number of simultaneously recorded units")
ax[2].set_ylabel("decoding accuracy")
ax[2].set_title("c  accuracy grows with population size", loc="left", fontsize=10)
ax[2].legend(fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig("fig04_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"mean accuracy {acc.mean():.3f}, shuffled {shuf.mean():.3f}")
print("mean |octave error|:",
      np.mean([res[s]["oct_err"] for s in sessions]).round(3))
print("saved fig04_decoding.png")
