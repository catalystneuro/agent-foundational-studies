"""Figure 3: population statistics of orientation selectivity across 10 sessions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import plotting as pl

p = pd.read_pickle("results_pooled_units.pkl")
ALPHA = 0.01
AREAS = ["VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm", "LGd", "LP", "CA1", "CA3", "DG"]
areas = [a for a in AREAS if (p["location"] == a).sum() >= 20]

fig = plt.figure(figsize=(13, 9.5))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32, left=0.07, right=0.97, top=0.90, bottom=0.08)

# --- A: gOSI cumulative distributions --------------------------------------
ax = fig.add_subplot(gs[0, 0])
for grp in pl.REGION_ORDER:
    v = np.sort(p.loc[p.region_group == grp, "dg_gOSI"].dropna().values)
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=pl.REGION_COLORS[grp], lw=1.8,
            label=f"{grp} (n={len(v)})")
ax.set_xlabel("global OSI (drifting gratings)")
ax.set_ylabel("cumulative fraction of units")
ax.set_xlim(0, 1)
ax.legend(fontsize=7.5, loc="lower right")
kc = p.loc[p.region_group == "visual cortex", "dg_gOSI"].dropna()
kh = p.loc[p.region_group == "hippocampus", "dg_gOSI"].dropna()
u = stats.mannwhitneyu(kc, kh)
ax.set_title(f"A  gOSI distributions\ncortex vs hippocampus: U-test p = {u.pvalue:.1e}", loc="left")

# --- B: fraction of significantly tuned units per area ---------------------
ax = fig.add_subplot(gs[0, 1])
frac = p.groupby("location")["dg_p_gOSI"].apply(lambda x: (x < ALPHA).mean()).reindex(areas)
frac_sg = p.groupby("location")["sg_p_gOSI"].apply(lambda x: (x < ALPHA).mean()).reindex(areas)
n = p.groupby("location").size().reindex(areas)
x = np.arange(len(areas))
import dandi_io as dio
cols = [pl.REGION_COLORS[dio.region_group(a)] for a in areas]
ax.bar(x - 0.2, frac.values, 0.4, color=cols, label="drifting")
ax.bar(x + 0.2, frac_sg.values, 0.4, color=cols, alpha=0.5, label="static")
ax.axhline(ALPHA, color="k", ls="--", lw=0.9)
ax.text(len(areas) - 0.5, ALPHA + 0.015, f"chance ({ALPHA})", ha="right", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([f"{a} ({n[a]})" for a in areas], fontsize=7.5, rotation=55, ha="right")
ax.set_ylabel(f"fraction with p < {ALPHA}")
ax.set_ylim(0, 1)
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("B  Orientation-tuned units by area\n(shuffle test on gOSI)", loc="left")

# --- C: orientation vs direction selectivity ------------------------------
ax = fig.add_subplot(gs[0, 2])
for grp in ["hippocampus", "visual thalamus", "visual cortex"]:
    d = p[p.region_group == grp]
    ax.scatter(d["dg_gOSI"], d["dg_gDSI"], s=4, alpha=0.35, lw=0,
               color=pl.REGION_COLORS[grp], label=grp)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("global OSI")
ax.set_ylabel("global DSI")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=7.5, markerscale=2.5, loc="upper right")
cx = p[p.region_group == "visual cortex"]
ax.set_title("C  Orientation vs direction selectivity\n"
             f"cortex: median OSI {cx.dg_gOSI.median():.2f} > DSI {cx.dg_gDSI.median():.2f}", loc="left")

# --- D: preferred orientation distribution --------------------------------
ax = fig.add_subplot(gs[1, 0], projection="polar")
sig = p[(p.region_group == "visual cortex") & (p.sg_p_gOSI < ALPHA)]
edges = np.arange(0, 181, 15)
h, _ = np.histogram(sig["sg_pref_ori"], bins=edges)
theta = np.deg2rad((edges[:-1] + 7.5) * 2)
ax.bar(theta, h, width=np.deg2rad(30), color=pl.REGION_COLORS["visual cortex"], alpha=0.8)
ax.set_thetagrids(np.arange(0, 360, 45), [f"{int(a)}°" for a in np.arange(0, 180, 22.5)])
ax.set_yticklabels([])
# Rayleigh test for a non-uniform preferred-orientation distribution. Orientation
# lives on a 180 deg circle, so angles are doubled before the circular statistic.
ang = np.deg2rad(2 * sig["sg_pref_ori"].values)
n_sig = len(ang)
R = np.abs(np.sum(np.exp(1j * ang)))
p_ray = np.exp(np.sqrt(1 + 4 * n_sig + 4 * (n_sig ** 2 - R ** 2)) - (1 + 2 * n_sig))
ax.set_title(f"D  Preferred orientation, tuned cortical units\n(n={n_sig}; axes doubled; "
             f"Rayleigh p={p_ray:.1e})", loc="left", pad=22)

# --- E: tuning width ------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
# Width is only interpretable for units that are actually tuned and well fit;
# the 30 deg stimulus spacing bounds the estimate to [15 deg, 90 deg].
w = p[(p.sg_p_gOSI < ALPHA) & (p.sg_fit_r2 > 0.8) & (p.sg_hwhm < 85)]
for grp in pl.REGION_ORDER:
    v = w.loc[w.region_group == grp, "sg_hwhm"].dropna()
    if len(v) < 10:
        continue
    ax.hist(v, bins=np.arange(15, 95, 5), histtype="step", lw=1.8, density=True,
            color=pl.REGION_COLORS[grp], label=f"{grp} (n={len(v)}, med {v.median():.0f}°)")
ax.set_xlim(15, 85)
ax.set_xlabel("von Mises half-width at half max (deg)\n(bounded below at 15° by the 30° stimulus spacing)")
ax.set_ylabel("density")
ax.legend(fontsize=7.5)
ax.set_title("E  Orientation tuning width\n(significantly tuned units, static gratings)", loc="left")

# --- F: session-by-session consistency ------------------------------------
ax = fig.add_subplot(gs[1, 2])
bysess = (p.groupby(["session_id", "region_group"])["dg_p_gOSI"]
          .agg(frac=lambda x: (x < ALPHA).mean(), n="size").reset_index())
bysess = bysess[bysess["n"] >= 20]
for i, grp in enumerate(pl.REGION_ORDER):
    d = bysess[bysess.region_group == grp]
    ax.scatter(np.full(len(d), i) + np.linspace(-0.18, 0.18, len(d)), d["frac"],
               s=26, color=pl.REGION_COLORS[grp], zorder=3)
    ax.hlines(d["frac"].median(), i - 0.3, i + 0.3, color="k", lw=1.6, zorder=4)
ax.axhline(ALPHA, color="k", ls="--", lw=0.9)
ax.set_xticks(range(3))
ax.set_xticklabels([g.replace(" ", "\n") for g in pl.REGION_ORDER])
ax.set_ylabel(f"fraction with p < {ALPHA}")
ax.set_ylim(-0.03, 1)
ax.set_title("F  Per-session consistency\n(one point per session, ≥20 units)", loc="left")

fig.suptitle(f"Orientation selectivity across {p.session_id.nunique()} Neuropixels sessions "
             f"({len(p)} QC-passing units), DANDI:000021", y=0.965, fontsize=12)
fig.savefig("fig03_population.png", bbox_inches="tight")
print("wrote fig03_population.png")
