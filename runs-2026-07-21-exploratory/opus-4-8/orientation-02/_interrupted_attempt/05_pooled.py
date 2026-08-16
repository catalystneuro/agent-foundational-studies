"""Pool tuning across sessions; drifting vs static gratings; cardinal bias."""
import glob
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

import oslib

mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
AREA_COLORS = {"VISp": "#1f4e9c", "VISl": "#3f7fbf", "VISrl": "#6aa9d8",
               "VISam": "#8fbf8f", "VISpm": "#5f9e5f", "LGd": "#d1671a"}
POOL = "extracted/pooled.pkl"


def rayleigh(angles_rad):
    """Rayleigh test for non-uniformity of a circular sample."""
    n = angles_rad.size
    r = np.abs(np.exp(1j * angles_rad).sum()) / n
    z = n * r**2
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - (n * r) ** 2)) - (1 + 2 * n))
    return r, p


def build():
    rows = []
    files = sorted(glob.glob("extracted/*.npz"))
    for f in tqdm(files, desc="tuning"):
        d = np.load(f, allow_pickle=True)
        ses = str(d["session"])
        dgt = oslib.analyze_gratings(d["dg_rates"], d["dg_ori"], d["dg_tf"], n_perm=1000)
        sgt = oslib.analyze_gratings(d["sg_rates"], d["sg_ori"], d["sg_sf"], n_perm=1000)
        rows.append(dict(session=ses, area=d["area"], unit_ids=d["unit_ids"],
                         snr=d["snr"], dg=dgt, sg=sgt))
        print(f"  {ses}: {len(d['unit_ids'])} units, "
              f"VISp median gOSI {np.median(dgt['gosi'][d['area'] == 'VISp']):.3f}", flush=True)
    pickle.dump(rows, open(POOL, "wb"))
    return rows


if __name__ == "__main__":
    rows = pickle.load(open(POOL, "rb")) if os.path.exists(POOL) else build()

    area = np.concatenate([r["area"] for r in rows])
    session = np.concatenate([[r["session"]] * len(r["area"]) for r in rows])
    dirs = rows[0]["dg"]["levels"]
    oris = rows[0]["sg"]["levels"]
    dg = {k: np.concatenate([r["dg"][k] for r in rows]) for k in
          ("gosi", "dsi", "tuned", "pref_ori", "baseline", "p_anova")}
    dg["tc"] = np.concatenate([r["dg"]["tc"] for r in rows])
    sg = {k: np.concatenate([r["sg"][k] for r in rows]) for k in ("gosi", "tuned", "pref_ori")}
    sg["tc"] = np.concatenate([r["sg"]["tc"] for r in rows])
    peak = dg["tc"].max(axis=1)

    areas = [a for a in oslib.AREAS if (area == a).sum() >= 30]
    print(f"\n{len(rows)} sessions, {len(area)} units, areas {areas}")

    # ------------------------------------------------------------ figure 5
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))

    ax = axes[0]
    for a in ("VISp", "LGd"):
        v = np.sort(dg["gosi"][area == a])
        ax.plot(v, np.arange(1, v.size + 1) / v.size, color=AREA_COLORS[a], lw=2.2,
                label=f"{a} (n={v.size})")
    u = stats.mannwhitneyu(dg["gosi"][area == "VISp"], dg["gosi"][area == "LGd"])
    ax.set_xlabel("gOSI")
    ax.set_ylabel("cumulative fraction of units")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"pooled gOSI, drifting gratings\nMann-Whitney p = {u.pvalue:.1e}", fontsize=10)

    ax = axes[1]
    rng = np.random.default_rng(0)
    for i, a in enumerate(areas):
        v = dg["gosi"][area == a]
        boot = np.median(rng.choice(v, (2000, v.size)), axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        med = np.median(v)
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], fmt="o", ms=7,
                    color=AREA_COLORS[a], capsize=4)
        ax.text(i + 0.12, med, f"n={v.size}", fontsize=7, va="center")
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, rotation=45, ha="right")
    ax.set_ylabel("median gOSI")
    ax.set_title("median gOSI by area\n(95% bootstrap CI)", fontsize=10)

    ax = axes[2]
    fr = [dg["tuned"][area == a].mean() for a in areas]
    ax.bar(range(len(areas)), fr, color=[AREA_COLORS[a] for a in areas])
    for i, f in enumerate(fr):
        ax.text(i, f + 0.01, f"{f:.0%}", ha="center", fontsize=8)
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, rotation=45, ha="right")
    ax.set_ylabel("fraction significantly tuned")
    ax.set_ylim(0, max(fr) * 1.25)
    ax.set_title("significant direction tuning\n(ANOVA p<0.01 and perm. gOSI p<0.05)", fontsize=10)

    ax = axes[3]
    per_ses = np.array([
        [np.median(r["dg"]["gosi"][r["area"] == a]) if (r["area"] == a).sum() >= 10 else np.nan
         for a in ("VISp", "LGd")]
        for r in rows
    ])
    ok = ~np.isnan(per_ses).any(axis=1)
    for v in per_ses[ok]:
        ax.plot([0, 1], v, "-o", color="0.6", ms=5, lw=1)
    ax.plot([0, 1], np.nanmedian(per_ses[ok], axis=0), "-o", color="k", lw=2.5, ms=8,
            label="median across sessions")
    w = stats.wilcoxon(per_ses[ok, 0], per_ses[ok, 1])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["VISp", "LGd"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("median gOSI in session")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"per-session paired comparison\nWilcoxon p = {w.pvalue:.1e} "
                 f"(n={ok.sum()} sessions)", fontsize=10)

    fig.suptitle(f"Orientation selectivity pooled across {len(rows)} sessions of DANDI:000021",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig("fig05_pooled_selectivity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("fig05 done")

    # ------------------------------------------------------------ figure 6
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    xs = np.arange(dirs.size) * 45
    for a in ("VISp", "LGd"):
        m = (area == a) & (peak > 1)
        al = oslib.align_to_preferred(dg["tc"][m], dirs)
        mu, se = np.nanmean(al, axis=0), stats.sem(al, axis=0, nan_policy="omit")
        ax.plot(xs, mu, "-o", color=AREA_COLORS[a], lw=2, label=f"{a} (n={m.sum()})")
        ax.fill_between(xs, mu - se, mu + se, color=AREA_COLORS[a], alpha=0.25)
    ax.axvline(90, ls="--", color="0.6", lw=1)
    ax.axvline(180, ls=":", color="0.4", lw=1)
    ax.text(92, 0.95, "orthogonal", fontsize=8, color="0.4")
    ax.text(182, 0.95, "opposite direction,\nsame orientation", fontsize=8, color="0.4")
    ax.set_xlabel("direction relative to each unit's preferred (deg)")
    ax.set_ylabel("normalized firing rate")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("population tuning aligned to\neach unit's preferred direction", fontsize=10)

    ax = axes[1]
    for a in ("VISp", "LGd"):
        m = (area == a) & (peak > 1)
        al = oslib.align_to_preferred(sg["tc"][m], oris)
        xs2 = np.arange(oris.size) * 30
        mu, se = np.nanmean(al, axis=0), stats.sem(al, axis=0, nan_policy="omit")
        ax.plot(xs2, mu, "-o", color=AREA_COLORS[a], lw=2, label=f"{a} (n={m.sum()})")
        ax.fill_between(xs2, mu - se, mu + se, color=AREA_COLORS[a], alpha=0.25)
    ax.axvline(90, ls="--", color="0.6", lw=1)
    ax.set_xlabel("orientation relative to preferred (deg)")
    ax.set_ylabel("normalized firing rate")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("same, for static gratings\n(orientation only, no motion)", fontsize=10)

    ax = axes[2]
    for a in ("VISp", "LGd"):
        v = np.sort(sg["gosi"][area == a])
        ax.plot(v, np.arange(1, v.size + 1) / v.size, color=AREA_COLORS[a], lw=2.2, label=a)
    u2 = stats.mannwhitneyu(sg["gosi"][area == "VISp"], sg["gosi"][area == "LGd"])
    ax.set_xlabel("gOSI (static gratings)")
    ax.set_ylabel("cumulative fraction of units")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"static-grating gOSI\nMann-Whitney p = {u2.pvalue:.1e}", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig06_aligned_tuning.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("fig06 done")

    # ------------------------------------------------------------ figure 7
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    sel = (area == "VISp") & dg["tuned"] & (peak > 1)
    po = dg["pref_ori"][sel]
    r_, p_ray = rayleigh(np.deg2rad(po) * 2)
    ax.hist(po, bins=np.arange(0, 181, 15), color=AREA_COLORS["VISp"], alpha=0.85)
    for c in (0, 90, 180):
        ax.axvline(c, ls="--", color="k", lw=1)
    ax.set_xlabel("preferred orientation (deg)")
    ax.set_ylabel("number of VISp units")
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_title(f"cardinal bias in VISp (n={sel.sum()})\nRayleigh on 2$\\theta$: p = {p_ray:.1e}",
                 fontsize=10)

    ax = axes[1]
    both = (area == "VISp") & dg["tuned"] & sg["tuned"] & (peak > 1)
    delta = oslib.circ_dist_180(dg["pref_ori"][both], sg["pref_ori"][both])
    rng = np.random.default_rng(1)
    null = oslib.circ_dist_180(dg["pref_ori"][both], rng.permutation(sg["pref_ori"][both]))
    bins = np.arange(0, 91, 7.5)
    ax.hist(delta, bins=bins, color=AREA_COLORS["VISp"], alpha=0.85, label="observed")
    ax.hist(null, bins=bins, histtype="step", color="k", lw=1.5, label="shuffled units")
    ks = stats.mannwhitneyu(delta, null)
    ax.set_xlabel("|preferred orientation difference| (deg)")
    ax.set_ylabel("number of VISp units")
    ax.legend(frameon=False)
    ax.set_title(f"drifting vs static gratings, same unit\nmedian {np.median(delta):.0f}$\\degree$ "
                 f"vs {np.median(null):.0f}$\\degree$ shuffled (p = {ks.pvalue:.1e})", fontsize=10)

    ax = axes[2]
    for a in ("VISp", "LGd"):
        m = (area == a) & (peak > 1)
        ax.scatter(dg["gosi"][m], dg["dsi"][m], s=9, alpha=0.5, color=AREA_COLORS[a],
                   edgecolor="none", label=a)
    ax.plot([0, 1], [0, 1], ls=":", color="0.5")
    ax.set_xlabel("gOSI (orientation selectivity)")
    ax.set_ylabel("DSI (direction selectivity)")
    ax.legend(frameon=False)
    frac = np.mean(dg["gosi"][(area == "VISp") & (peak > 1)] > dg["dsi"][(area == "VISp") & (peak > 1)])
    ax.set_title(f"orientation vs direction, pooled\n{frac:.0%} of VISp units have gOSI > DSI",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("fig07_preference_structure.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("fig07 done")

    # ------------------------------------------------------------ summary
    with open("pooled_stats.txt", "w") as fh:
        fh.write(f"sessions: {[r['session'] for r in rows]}\n")
        fh.write(f"total quality-filtered units: {len(area)}\n\n")
        fh.write(f"{'area':6s} {'n':>5s} {'medGOSI':>8s} {'medDSI':>7s} {'%tuned':>7s} "
                 f"{'medGOSIstatic':>14s}\n")
        for a in areas:
            m = area == a
            fh.write(f"{a:6s} {m.sum():5d} {np.median(dg['gosi'][m]):8.3f} "
                     f"{np.median(dg['dsi'][m]):7.3f} {dg['tuned'][m].mean():7.1%} "
                     f"{np.median(sg['gosi'][m]):14.3f}\n")
        fh.write(f"\nVISp vs LGd gOSI Mann-Whitney p = {u.pvalue:.3e}\n")
        fh.write(f"per-session paired Wilcoxon p = {w.pvalue:.3e} (n={ok.sum()})\n")
        fh.write(f"VISp cardinal bias Rayleigh p = {p_ray:.3e}\n")
        fh.write(f"drifting-vs-static preferred orientation: median |delta| = "
                 f"{np.median(delta):.1f} deg (shuffled {np.median(null):.1f} deg)\n")
        alv = oslib.align_to_preferred(dg["tc"][(area == "VISp") & (peak > 1)], dirs)
        mu = np.nanmean(alv, axis=0)
        fh.write(f"VISp aligned population tuning (normalized): {np.round(mu, 3).tolist()}\n")
        alg = oslib.align_to_preferred(dg["tc"][(area == "LGd") & (peak > 1)], dirs)
        fh.write(f"LGd  aligned population tuning (normalized): "
                 f"{np.round(np.nanmean(alg, axis=0), 3).tolist()}\n")
    print(open("pooled_stats.txt").read())
    np.savez("extracted/pooled_summary.npz", area=area, session=session,
             dg_gosi=dg["gosi"], dg_dsi=dg["dsi"], dg_tuned=dg["tuned"],
             dg_pref=dg["pref_ori"], dg_tc=dg["tc"], sg_gosi=sg["gosi"],
             sg_tuned=sg["tuned"], sg_pref=sg["pref_ori"], sg_tc=sg["tc"],
             dirs=dirs, oris=oris, peak=peak)
