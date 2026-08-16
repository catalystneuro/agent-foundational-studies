import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Our study: mean total rubric score (out of 24), fable folded into opus-4-8, firstrun excluded.
# Artificial Analysis Intelligence Index (reasoning/max variant, artificialanalysis.ai, Aug 2026).
models = ["Haiku 4.5", "Sonnet 5", "Opus 4.8", "Opus 5"]
aa     = np.array([30, 55, 57, 63], float)   # x: AA Intelligence Index
score  = np.array([5.5, 19.7, 22.5, 23.5], float)  # y: mean rubric points / 24
passr  = np.array([0, 74, 94, 100], float)   # for annotation

r = np.corrcoef(aa, score)[0,1]
# rank correlation (more honest at n=4)
from scipy.stats import spearmanr
rho, _ = spearmanr(aa, score)

fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=300)
ax.scatter(aa, score, s=70, color="#1b3a6b", zorder=3)
offsets = {"Haiku 4.5": (8, 4), "Sonnet 5": (-6, -16), "Opus 4.8": (-8, 9), "Opus 5": (9, -3)}
ha_map = {"Haiku 4.5": "left", "Sonnet 5": "center", "Opus 4.8": "right", "Opus 5": "left"}
for m, x, y in zip(models, aa, score):
    ax.annotate(m, (x, y), textcoords="offset points", xytext=offsets[m],
                ha=ha_map[m], fontsize=9, color="#222")
# least-squares guide line
b, a = np.polyfit(aa, score, 1)
xs = np.linspace(aa.min()-2, aa.max()+2, 50)
ax.plot(xs, a + b*xs, color="#b04a3a", lw=1.2, ls="--", zorder=2, alpha=0.8)

ax.set_xlabel("Artificial Analysis Intelligence Index", fontsize=10)
ax.set_ylabel("Mean rubric score (of 24)", fontsize=10)
ax.set_ylim(0, 25)
ax.set_xlim(25, 68)
ax.tick_params(labelsize=9)
ax.spines[["top","right"]].set_visible(False)
ax.text(0.03, 0.97, f"Pearson r = {r:.2f}\nSpearman ρ = {rho:.2f}  (n = 4)",
        transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color="#444")
fig.tight_layout()
fig.savefig("paper_figures/aa_intelligence_vs_score.png", bbox_inches="tight")
print(f"Pearson r={r:.3f}  Spearman rho={rho:.3f}")
print("saved paper_figures/aa_intelligence_vs_score.png")
