"""Polar tuning gallery and single-session population selectivity."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import oslib

mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

SES = "715093703"
AREA_COLORS = {"VISp": "#1f4e9c", "VISl": "#3f7fbf", "VISrl": "#6aa9d8",
               "VISam": "#8fbf8f", "VISpm": "#5f9e5f", "LGd": "#d1671a"}

d = np.load(f"extracted/{SES}.npz", allow_pickle=True)
area = d["area"]
dg = np.load(f"extracted/{SES}_dg_tuning.npy", allow_pickle=True).item()
ids = d["unit_ids"]
dirs = dg["levels"]

# ---------------------------------------------------------------- figure 3
def polar_gallery(ax, i, color, label):
    th = np.deg2rad(np.append(dirs, dirs[0]))
    r = np.append(dg["tc"][i], dg["tc"][i][0])
    e = np.append(dg["sem"][i], dg["sem"][i][0])
    ax.plot(th, r, "-o", color=color, ms=3, lw=1.3)
    ax.fill_between(th, np.clip(r - e, 0, None), r + e, color=color, alpha=0.25)
    ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, dg["baseline"][i]),
            ls="--", color="0.55", lw=0.9)
    ax.set_theta_zero_location("E")
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0", "", "180", "270"], fontsize=7)
    ax.tick_params(pad=-1)
    ax.set_yticklabels([])
    ax.set_title(label, fontsize=7.5, pad=10)


fig, axes = plt.subplots(2, 6, figsize=(13.5, 6.4), subplot_kw=dict(projection="polar"))
for row, aname in enumerate(["VISp", "LGd"]):
    # Units spanning the gOSI range of the area rather than only its extreme tail.
    idx = np.flatnonzero((area == aname) & (dg["tc"].max(axis=1) > 3))
    order_g = idx[np.argsort(dg["gosi"][idx])]
    pcts = [95, 80, 65, 50, 35, 20]
    picks = [order_g[min(len(order_g) - 1, int(round(p / 100 * (len(order_g) - 1))))] for p in pcts]
    for col, (p, i) in enumerate(zip(pcts, picks)):
        polar_gallery(
            axes[row, col], i, AREA_COLORS[aname],
            f"unit {int(ids[i])}  ({p}th pct)\ngOSI {dg['gosi'][i]:.2f}   DSI {dg['dsi'][i]:.2f}\n"
            f"peak {dg['tc'][i].max():.1f} Hz",
        )
    axes[row, 0].text(-0.42, 0.5, aname, transform=axes[row, 0].transAxes, rotation=90,
                      va="center", ha="center", fontsize=13, fontweight="bold",
                      color=AREA_COLORS[aname])
fig.suptitle(
    "Direction tuning curves of units spanning the gOSI range of each area "
    f"(session {SES}, units with peak rate > 3 Hz); dashed circle = blank-sweep rate",
    fontsize=11, y=1.0,
)
fig.subplots_adjust(hspace=0.62, wspace=0.42, left=0.06, right=0.99, top=0.80, bottom=0.05)
fig.savefig("fig03_polar_gallery.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("fig03 done")

# ---------------------------------------------------------------- figure 4
areas_present = [a for a in oslib.AREAS if (area == a).sum() >= 15]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))

ax = axes[0]
for aname in ("VISp", "LGd"):
    v = np.sort(dg["gosi"][area == aname])
    ax.plot(v, np.arange(1, v.size + 1) / v.size, color=AREA_COLORS[aname], lw=2,
            label=f"{aname} (n={v.size})")
ax.set_xlabel("gOSI")
ax.set_ylabel("cumulative fraction of units")
ax.legend(frameon=False, loc="lower right")
u = stats.mannwhitneyu(dg["gosi"][area == "VISp"], dg["gosi"][area == "LGd"])
ax.set_title(f"gOSI distribution\nMann-Whitney p = {u.pvalue:.1e}", fontsize=10)

ax = axes[1]
frac = [dg["tuned"][area == a].mean() for a in areas_present]
n = [(area == a).sum() for a in areas_present]
ax.bar(range(len(areas_present)), frac, color=[AREA_COLORS[a] for a in areas_present])
for i, (f, nn) in enumerate(zip(frac, n)):
    ax.text(i, f + 0.015, f"{f:.0%}\nn={nn}", ha="center", fontsize=7.5)
ax.set_xticks(range(len(areas_present)))
ax.set_xticklabels(areas_present, rotation=45, ha="right")
ax.set_ylabel("fraction significantly\norientation tuned")
ax.set_ylim(0, max(frac) * 1.3)
ax.set_title("ANOVA p<0.01 and\npermuted gOSI p<0.05", fontsize=10)

ax = axes[2]
for aname in ("LGd", "VISp"):
    m = area == aname
    ax.scatter(dg["gosi"][m], dg["dsi"][m], s=16, alpha=0.7,
               color=AREA_COLORS[aname], label=aname, edgecolor="none")
lim = max(dg["gosi"].max(), dg["dsi"].max()) * 1.05
ax.plot([0, lim], [0, lim], ls=":", color="0.6", lw=1)
ax.set_xlabel("gOSI (orientation)")
ax.set_ylabel("DSI (direction)")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.legend(frameon=False)
ax.set_title("orientation vs direction\nselectivity", fontsize=10)

ax = axes[3]
rng = np.random.default_rng(0)
for i, aname in enumerate(areas_present):
    v = dg["gosi"][area == aname]
    boot = np.median(rng.choice(v, (2000, v.size)), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ax.errorbar(i, np.median(v), yerr=[[np.median(v) - lo], [hi - np.median(v)]],
                fmt="o", color=AREA_COLORS[aname], capsize=4, ms=7)
ax.set_xticks(range(len(areas_present)))
ax.set_xticklabels(areas_present, rotation=45, ha="right")
ax.set_ylabel("median gOSI")
ax.set_title("median gOSI by area\n(95% bootstrap CI)", fontsize=10)

fig.suptitle(f"Population orientation selectivity, session {SES}", fontsize=12, y=1.04)
fig.tight_layout()
fig.savefig("fig04_population_single_session.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("fig04 done")
for a in areas_present:
    m = area == a
    print(f"  {a:6s} n={m.sum():3d} tuned={dg['tuned'][m].mean():5.1%} "
          f"median gOSI={np.median(dg['gosi'][m]):.3f} median DSI={np.median(dg['dsi'][m]):.3f}")
