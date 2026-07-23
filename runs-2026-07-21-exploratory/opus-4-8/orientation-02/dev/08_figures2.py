"""Figure 7: population decoding of orientation and the NeMoS encoding GLM."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

dec = pd.read_csv("decoding_results.csv")
glm = pd.read_csv("glm_results.csv")
CORTEX_C, THAL_C = "#2b6cb0", "#c05621"

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))

# (a) decoding accuracy versus population size
ax = axes[0]
for label, col in [("visual cortex", CORTEX_C), ("LGd", THAL_C)]:
    sub = dec[dec.group == label]
    if not len(sub):
        continue
    g = sub.groupby("n_units")["accuracy"]
    m, s = g.mean(), g.std()
    ax.errorbar(m.index, m.values, yerr=s.values, marker="o", color=col, lw=2,
                capsize=3, label=label)
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

# (b) GLM orientation contribution
ax = axes[1]
for label, sub, col in [("visual cortex", glm[glm.region == "cortex"], CORTEX_C),
                        ("LGd", glm[glm.area == "LGd"], THAL_C),
                        ("LP", glm[glm.area == "LP"], "#8b6f47")]:
    if not len(sub):
        continue
    v = np.sort(sub.dR2_orientation.dropna().values)
    ax.step(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=2.2,
            label=f"{label}, n={len(v)}, median {np.median(v):.3f}")
ax.axvline(0, color="k", ls=":", lw=1)
ax.set_xlabel("held-out pseudo-$R^2$ gained by adding orientation")
ax.set_ylabel("cumulative fraction of units")
ctx = glm.loc[glm.region == "cortex", "dR2_orientation"].dropna()
lgd = glm.loc[glm.area == "LGd", "dR2_orientation"].dropna()
u_stat, p_gl = stats.mannwhitneyu(ctx, lgd)
ax.set_title(f"(b) NeMoS Poisson GLM, cross-validated\ncortex vs LGd: "
             f"Mann-Whitney p = {p_gl:.2g}", loc="left", fontweight="bold")
ax.legend(fontsize=9, frameon=False, loc="lower right")

# (c) the descriptive statistic and the model-based one agree
ax = axes[2]
for label, sub, col in [("cortex", glm[glm.region == "cortex"], CORTEX_C),
                        ("thalamus", glm[glm.region == "thalamus"], THAL_C)]:
    ax.plot(sub.gosi_corrected, sub.dR2_orientation, "o", color=col, ms=4, alpha=0.45,
            label=label)
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
plt.close(fig)
print("wrote fig07_decoding_glm.png")

with open("stats_summary.txt", "a") as fh:
    fh.write("\n--- population decoding (4 orientations, chance 0.25) ---\n")
    fh.write(dec.groupby(["group", "n_units"])[["accuracy", "shuffled"]].mean().to_string() + "\n")
    fh.write("\n--- NeMoS GLM: held-out pseudo-R2 gained by orientation ---\n")
    fh.write(glm.groupby("area")["dR2_orientation"].agg(["count", "median", "mean"]).to_string() + "\n")
    fh.write(f"cortex vs LGd: medians {ctx.median():.4f} vs {lgd.median():.4f}, "
             f"Mann-Whitney U={u_stat:.0f}, p={p_gl:.3g}\n")
    fh.write(f"gOSI vs GLM gain: Spearman r = {rho.statistic:.3f}, p = {rho.pvalue:.3g}\n")
