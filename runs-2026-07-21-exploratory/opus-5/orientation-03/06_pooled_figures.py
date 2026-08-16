"""Stage 6: pooled results across sessions."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import orientation_lib as ol

pooled = pd.read_csv("cache/pooled_results.csv", dtype={"session_id": str})
tun = np.load("cache/pooled_tuning.npz")
tc_b = tun["tc_b_aligned"]
dirs = np.arange(0, 360, 45)

print("%d units, %d sessions, %d mice"
      % (len(pooled), pooled.session_id.nunique(), pooled.subject_id.nunique()))

order = ["VISp", "VISl", "VISrl", "VISal", "VISam", "VISpm", "LGd", "LP", "CA1"]
order = [r for r in order if (pooled.region == r).sum() >= 30]
col = {r: ("C0" if r.startswith("VIS") else ("C1" if r in ol.THALAMUS else "0.6"))
       for r in order}

summary = (pooled.groupby("region")
           .agg(n=("gosi", "size"), n_resp=("responsive", "sum"),
                frac_resp=("responsive", "mean"), n_sig=("sig_ori", "sum"),
                frac_sig=("sig_ori", "mean"))
           .loc[order])
med = pooled[pooled.responsive].groupby("region").gosi.median().loc[order]
summary["median_gosi_responsive"] = med
print(summary.round(3).to_string())
summary.to_csv("results_by_region.csv")

# ---------------------------------------------------------------------------
# FIGURE 9: pooled selectivity by region, with per-session points
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

ax = axes[0]
data = [pooled[(pooled.region == r) & pooled.responsive].gosi.values for r in order]
parts = ax.violinplot(data, showmedians=True, widths=0.85)
for pc, r in zip(parts["bodies"], order):
    pc.set_facecolor(col[r])
    pc.set_alpha(0.75)
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels(order, rotation=45)
ax.set_ylabel("global OSI")
ax.set_title("Orientation selectivity, all sessions pooled\n(visually responsive units)",
             fontsize=10)

ax = axes[1]
for i, r in enumerate(order):
    per_ses = pooled[pooled.region == r].groupby("session_id").sig_ori.mean()
    ax.bar(i, pooled[pooled.region == r].sig_ori.mean(), color=col[r])
    ax.plot(np.random.normal(i, 0.07, len(per_ses)), per_ses.values, "o", ms=4,
            color="k", alpha=0.7, mfc="none")
ax.axhline(0.01, color="r", ls="--", lw=1, label="test alpha (0.01)")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45)
ax.set_ylabel("fraction orientation-selective")
ax.set_title("Significantly tuned units\n(bars = pooled, circles = single sessions)",
             fontsize=10)
ax.legend(fontsize=8)

ax = axes[2]
rel = np.r_[dirs[:5], dirs[5:] - 360]
o = np.argsort(rel)
for group, name, c in [(pooled.region.isin(ol.VISUAL_CORTEX), "visual cortex", "C0"),
                       (pooled.region.isin(ol.THALAMUS), "thalamus", "C1"),
                       (pooled.region == "CA1", "CA1 (control)", "0.5")]:
    m = (group & pooled.responsive).values
    z = tc_b[:, m]
    z = z / np.where(z.mean(0) == 0, np.nan, z.mean(0))
    z = z[o]
    mu = np.nanmean(z, 1)
    se = np.nanstd(z, 1) / np.sqrt(np.isfinite(z).sum(1))
    ax.errorbar(np.sort(rel), mu, yerr=se, marker="o", ms=4, color=c, capsize=3,
                label="%s (n=%d)" % (name, m.sum()))
ax.axhline(1, color="k", ls="--", lw=0.8)
ax.set_xticks(np.sort(rel))
ax.set_xlabel("direction relative to preferred (deg)")
ax.set_ylabel("rate / mean rate")
ax.set_title("Cross-validated population tuning\n(peak from half A, response from half B)",
             fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig09_pooled_by_region.png", dpi=140, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# FIGURE 10: preferred-orientation distribution and tuning width
# ---------------------------------------------------------------------------
sel = pooled[pooled.region.isin(ol.VISUAL_CORTEX) & pooled.sig_ori]
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

ax = axes[0]
bins = np.arange(0, 181, 15)
h, _ = np.histogram(sel.pref_ori, bins=bins)
ax.bar(bins[:-1] + 7.5, h, width=13, color="C0")
ax.axhline(len(sel) / (len(bins) - 1), color="k", ls="--", lw=1, label="uniform")
chi2, p = stats.chisquare(h)
ax.set_xlabel("preferred orientation (deg)")
ax.set_ylabel("number of units")
ax.set_title("Preferred orientations across cortex\n"
             "$\\chi^2$ vs uniform: p = %.1e (n=%d)" % (p, len(sel)), fontsize=10)
ax.set_xticks(np.arange(0, 181, 45))
ax.legend(fontsize=8)

ax = axes[1]
bins = np.arange(0, 95, 7.5)
ax.hist(sel.ori_diff_dg_sg.dropna(), bins=bins, color="C2", alpha=0.75, density=True,
        label="drifting vs static gratings")
ax.hist(sel.split_half_diff.dropna(), bins=bins, histtype="step", lw=2, color="C0",
        density=True, label="split-half (drifting)")
ax.axhline(1 / 90, color="k", ls="--", lw=1, label="chance")
ax.set_xlabel("|$\\Delta$ preferred orientation| (deg)")
ax.set_ylabel("density")
ax.set_title("Preferred orientation is stable\nacross trials and across stimuli",
             fontsize=10)
ax.legend(fontsize=8)

ax = axes[2]
for group, name, c in [(pooled.region.isin(ol.VISUAL_CORTEX), "visual cortex", "C0"),
                       (pooled.region.isin(ol.THALAMUS), "thalamus", "C1")]:
    s = pooled[group & pooled.responsive]
    ax.plot(s.gosi, s.gdsi, "o", ms=3, alpha=0.45, color=c, label=name)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("global OSI")
ax.set_ylabel("global DSI")
ax.set_title("Orientation vs direction selectivity\n(pooled)", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig10_pooled_validation.png", dpi=140, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# quantitative summary written to disk
# ---------------------------------------------------------------------------
ctx = pooled[pooled.region.isin(ol.VISUAL_CORTEX)]
thal = pooled[pooled.region.isin(ol.THALAMUS)]
ca1 = pooled[pooled.region == "CA1"]
u, p_mw = stats.mannwhitneyu(ctx[ctx.responsive].gosi, thal[thal.responsive].gosi,
                             alternative="greater")
lines = [
    "units analysed: %d (%d sessions, %d mice)"
    % (len(pooled), pooled.session_id.nunique(), pooled.subject_id.nunique()),
    "visual cortex: %d units, %d responsive, %d orientation-selective (%.1f%%)"
    % (len(ctx), ctx.responsive.sum(), ctx.sig_ori.sum(), 100 * ctx.sig_ori.mean()),
    "thalamus:      %d units, %d responsive, %d orientation-selective (%.1f%%)"
    % (len(thal), thal.responsive.sum(), thal.sig_ori.sum(), 100 * thal.sig_ori.mean()),
    "CA1 control:   %d units, %d responsive, %d orientation-selective (%.1f%%)"
    % (len(ca1), ca1.responsive.sum(), ca1.sig_ori.sum(), 100 * ca1.sig_ori.mean()),
    "median gOSI (responsive): cortex %.3f, thalamus %.3f, CA1 %.3f"
    % (ctx[ctx.responsive].gosi.median(), thal[thal.responsive].gosi.median(),
       ca1[ca1.responsive].gosi.median()),
    "cortex > thalamus gOSI, Mann-Whitney U p = %.2e" % p_mw,
    "cortex selective units: split-half |d pref ori| median %.1f deg (chance 45)"
    % sel.split_half_diff.median(),
    "cortex selective units: drifting vs static |d pref ori| median %.1f deg"
    % sel.ori_diff_dg_sg.median(),
]
print("\n".join(lines))
open("summary_stats.txt", "w").write("\n".join(lines) + "\n")
