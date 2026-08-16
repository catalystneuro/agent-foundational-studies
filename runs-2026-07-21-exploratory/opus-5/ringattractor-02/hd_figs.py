"""Figure-drawing routines for the head-direction ring-attractor analysis.

Every function takes the results dictionary `R` assembled by the analysis
notebook and writes one PNG into `outdir`.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

import hd_core as hc
import hd_glm as hg

STATE_COL = dict(wake="#1b6ca8", rem="#c0392b", nrem="#7d3c98")
STATE_LAB = dict(wake="wake (arena)", rem="REM sleep", nrem="NREM sleep")


def _setup(R):
    # `res` is only needed from fig03 onwards; fig01/fig02 run before it exists
    return (R.get("res"), R["st"], R["keep"], R["pref"],
            f"{R['subject']} / {R['session']}")


# ---------------------------------------------------------------- fig 1 ----
def fig01(R, outdir="figures"):
    res, st, keep, pref, SES = _setup(R)
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.7, 1.1, 1.2], hspace=0.55, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])
    home = R["epochs"]["home_cage"]
    for i, (s, col) in enumerate([("wake", "#95a5a6"), ("rem", STATE_COL["rem"]),
                                  ("nrem", STATE_COL["nrem"])]):
        ep = R["states"][s].intersect(home)   # scoring is only used inside the home cage
        ax.broken_barh(list(zip(ep.start, ep.end - ep.start)), (0.55, 0.35),
                       facecolors=col, label=s.upper())
    for k, ep in R["epochs"].items():
        c = "#f39c12" if k.startswith("wake") else "#bdc3c7"
        ax.broken_barh(list(zip(ep.start, ep.end - ep.start)), (0.1, 0.3), facecolors=c)
    for k, ep in R["epochs"].items():
        ax.text(float(ep.start[0]) + 30, 0.16, k, fontsize=8)
    ax.set_yticks([0.25, 0.72]); ax.set_yticklabels(["epoch", "sleep state\n(home cage)"])
    ax.set_xlabel("time (s)"); ax.set_ylim(0, 1)
    ax.legend(ncol=3, fontsize=8, loc="upper right", framealpha=.9)
    ax.set_title(f"{SES}: session structure  "
                 f"({len(keep)} head-direction cells of {len(st)} units)")

    ax = fig.add_subplot(gs[1, 0])
    hd = R["hd"]; pos = R["pos"]
    beh = R["beh"]
    p = pos.restrict(beh); h = hd.restrict(beh)
    hv = np.mod(np.angle(np.interp(p.t, h.t, np.cos(h.d)) + 1j * np.interp(p.t, h.t, np.sin(h.d))),
                hc.TWOPI)
    sc = ax.scatter(p.values[::10, 0], p.values[::10, 1], c=hv[::10], cmap="hsv", s=1)
    ax.set_title("arena trajectory, coloured by head direction", fontsize=10)
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)"); ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="HD (rad)", fraction=0.046)

    ax = fig.add_subplot(gs[1, 1:])
    t0 = float(beh.start[0]) + 400
    hh = hd.restrict(hc.nap.IntervalSet(t0, t0 + 60))
    ax.plot(hh.t, hh.d, ".", ms=1.5, color="k")
    ax.set_ylabel("head direction (rad)"); ax.set_xlabel("time (s)")
    ax.set_title("measured head direction (60 s of foraging)", fontsize=10)
    ax.set_xlim(t0, t0 + 60)

    ax = fig.add_subplot(gs[2, :])
    order = keep[np.argsort(pref)]
    for j, u in enumerate(order):
        s = R["spikes"][int(u)]
        s = s[(s > t0) & (s < t0 + 60)]
        ax.plot(s, np.full_like(s, j), "|", ms=4, color="k", lw=.5)
    ax.set_ylabel("HD cell\n(sorted by preferred direction)")
    ax.set_xlabel("time (s)"); ax.set_xlim(t0, t0 + 60)
    ax2 = ax.twinx()
    hv2 = np.mod(hh.d, hc.TWOPI) / hc.TWOPI * len(order)
    hv2[np.abs(np.diff(hv2, prepend=hv2[0])) > len(order) / 2] = np.nan   # hide 2pi wraps
    ax2.plot(hh.t, hv2, color="#e74c3c", lw=1.5, alpha=.8)
    ax2.set_ylim(0, len(order)); ax2.set_yticks([])
    ax.set_title("population raster: the active subset of cells tracks head direction (red)",
                 fontsize=10)
    path = f"{outdir}/fig01_session_overview.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- fig 2 ----
def fig02(R, outdir="figures"):
    res, st, keep, pref, SES = _setup(R)
    tc = R["tc"]
    fig = plt.figure(figsize=(14, 7.6))
    gs = GridSpec(3, 6, figure=fig, hspace=0.55, wspace=0.6,
                  height_ratios=[1, 1, 1.25])
    ex = keep[np.argsort(pref)][np.linspace(0, len(keep) - 1, 12).astype(int)]
    ang = np.append(tc.index.values, tc.index.values[0])
    for i, u in enumerate(ex):
        ax = fig.add_subplot(gs[i // 6, i % 6], projection="polar")
        r = np.append(tc[u].values, tc[u].values[0])
        ax.plot(ang, r, color="#1b6ca8", lw=1.3)
        ax.fill_between(ang, 0, r, color="#1b6ca8", alpha=.25)
        ax.set_xticks(np.arange(0, hc.TWOPI, np.pi / 2))
        ax.set_xticklabels(["0", "", "π", ""], fontsize=7)
        ax.set_yticks([]); ax.set_title(f"#{u}  {r.max():.0f} Hz", fontsize=8, pad=6)

    ax = fig.add_subplot(gs[2, 0:2])
    ax.scatter(st["mvl"], st["stability"], s=14, c=np.where(st["keep"], "#1b6ca8", "#bdc3c7"))
    ax.axvline(0.25, color="k", ls=":", lw=1); ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("tuning strength (mean vector length)")
    ax.set_ylabel("split-half tuning\nstability (r)")
    ax.set_title(f"HD-cell selection ({len(keep)} kept)", fontsize=10)

    ax = fig.add_subplot(gs[2, 2:4], projection="polar")
    ax.hist(pref, bins=24, color="#1b6ca8", alpha=.8)
    ax.set_yticks([]); ax.set_xticklabels(["0", "", "π/2", "", "π", "", "3π/2", ""], fontsize=7)
    ax.set_title("preferred directions\ncover the circle", fontsize=10, y=1.18)

    ax = fig.add_subplot(gs[2, 4:6])
    o = np.argsort(pref)
    M = tc[keep].values[:, o]
    M = M / M.max(0, keepdims=True)
    im = ax.imshow(M.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, 2 * np.pi, 0, len(keep)])
    ax.set_xlabel("head direction (rad)"); ax.set_ylabel("cell (sorted)")
    ax.set_title("normalised tuning curves", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{SES}: head-direction tuning during waking exploration", y=0.98)
    path = f"{outdir}/fig02_hd_tuning.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- fig 3 ----
def fig03(R, outdir="figures"):
    res, st, keep, pref, SES = _setup(R)
    w = res["wake"]["real"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    e, sel = w["emb"], w["sel"]
    sc = axes[0].scatter(e[:, 0], e[:, 1], c=w["true"][w["mask"]][sel], cmap="hsv", s=5)
    axes[0].set_title("Isomap of wake population vectors\n(colour = measured head direction)",
                      fontsize=10)
    axes[0].set_xlabel("Isomap 1"); axes[0].set_ylabel("Isomap 2")
    plt.colorbar(sc, ax=axes[0], label="HD (rad)", fraction=.046)

    cc, off, sgn = hc.best_circ_align(w["ring"]["angle"], w["true"][w["mask"]][sel])
    axes[1].scatter(np.mod(sgn * w["ring"]["angle"] + off, hc.TWOPI),
                    w["true"][w["mask"]][sel], s=4, alpha=.25, color="#1b6ca8")
    axes[1].set_xlabel("manifold angle (aligned)"); axes[1].set_ylabel("measured HD (rad)")
    axes[1].set_title(f"manifold coordinate = head direction\ncircular r = {cc:.2f}", fontsize=10)

    axes[2].hist(np.degrees(np.abs(w["err"])), bins=60, color="#1b6ca8")
    axes[2].axvline(np.degrees(np.median(np.abs(w["err"]))), color="k", ls="--")
    axes[2].set_xlabel("|decoding error| (deg)"); axes[2].set_ylabel("bins")
    axes[2].set_title(f"Bayesian decoding of HD from spikes\nmedian error "
                      f"{np.degrees(np.median(np.abs(w['err']))):.0f}°", fontsize=10)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        L = PCA(n_components=4).fit(w["Y"]).components_
    lang = np.mod(np.arctan2(L[1], L[0]), hc.TWOPI)
    sc = axes[3].scatter(L[0], L[1], c=pref, cmap="hsv", s=26)
    axes[3].axhline(0, color="k", lw=.5); axes[3].axvline(0, color="k", lw=.5)
    axes[3].set_xlabel("PC1 loading"); axes[3].set_ylabel("PC2 loading")
    plt.colorbar(sc, ax=axes[3], label="preferred direction (rad)", fraction=.046)
    axes[3].set_title("each cell's (PC1, PC2) loading sits at its\npreferred direction: "
                      f"circular r = {hc.best_circ_align(lang, pref)[0]:.2f}", fontsize=10)
    fig.suptitle(f"{SES}: during waking, the population lives on a ring parameterised by head direction",
                 y=1.02)
    fig.tight_layout()
    path = f"{outdir}/fig03_wake_ring.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- fig 4 ----
def fig04(R, outdir="figures"):
    res, st, keep, pref, SES = _setup(R)
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.4), layout="constrained")
    for i, s in enumerate(["wake", "rem", "nrem"]):
        for j, cond in enumerate(["real", "shuffle"]):
            a = axes[j, i]
            r = res[s][cond]
            e = r["emb"]
            sc = a.scatter(e[:, 0], e[:, 1], c=r["dv"][r["sel"]], cmap="hsv", s=5)
            a.set_title(f"{STATE_LAB[s]} — {'data' if cond=='real' else 'shuffled'}\n"
                        f"ring score {r['ring']['ring_score']:.2f}, "
                        f"centre-filling {r['ring']['hole']:.2f}", fontsize=10)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_ylabel("data" if cond == "real" else "per-cell shuffle", fontsize=11)
    fig.colorbar(sc, ax=axes[:, 2], label="decoded head direction (rad)",
                 location="bottom", shrink=.6, pad=.02)

    a = axes[0, 3]
    x = np.arange(3)
    real = [res[s]["real"]["ring"]["ring_score"] for s in ["wake", "rem", "nrem"]]
    nl = [R["null"][s][:, 0] for s in ["wake", "rem", "nrem"]]
    a.bar(x - .18, real, .36, color=[STATE_COL[s] for s in ["wake", "rem", "nrem"]], label="data")
    a.bar(x + .18, [n.mean() for n in nl], .36, yerr=[n.std() for n in nl],
          color="#bdc3c7", label="shuffle (n=5)")
    a.set_xticks(x); a.set_xticklabels(["wake", "REM", "NREM"]); a.set_ylabel("ring score")
    a.legend(fontsize=8); a.set_title("annular geometry", fontsize=10)

    a = axes[1, 3]
    real = [res[s]["real"]["split"]["cc"] for s in ["wake", "rem", "nrem"]]
    nl = [R["null"][s][:, 2] for s in ["wake", "rem", "nrem"]]
    a.bar(x - .18, real, .36, color=[STATE_COL[s] for s in ["wake", "rem", "nrem"]])
    a.bar(x + .18, [n.mean() for n in nl], .36, yerr=[n.std() for n in nl], color="#bdc3c7")
    a.set_xticks(x); a.set_xticklabels(["wake", "REM", "NREM"])
    a.set_ylabel("split-half decoder agreement (circ. r)")
    a.set_title("two disjoint halves of the population\nreport the same angle", fontsize=10)
    fig.suptitle(f"{SES}: the ring survives in sleep and is destroyed by shuffling cell-cell "
                 f"coordination")
    path = f"{outdir}/fig04_sleep_ring.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- fig 5 ----
def fig05(R, outdir="figures"):
    res, st, keep, pref, SES = _setup(R)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for i, s in enumerate(["wake", "rem", "nrem"]):
        a = axes[i]
        for cond, col, mk in [("real", STATE_COL[s], "o"), ("shuffle", "#bdc3c7", "^")]:
            dg = res[s][cond]["ph"]["dgm"]
            a.scatter(dg[:, 0], dg[:, 1], s=22, c=col, marker=mk, alpha=.75,
                      label="data" if cond == "real" else "shuffle")
        lim = max(a.get_xlim()[1], a.get_ylim()[1])
        a.plot([0, lim], [0, lim], "k-", lw=.8)
        a.set_xlabel("birth"); a.set_ylabel("death")
        a.set_title(f"{STATE_LAB[s]}: H1 persistence\nlongest lifetime "
                    f"{res[s]['real']['ph']['top']:.2f} "
                    f"(shuffle {res[s]['shuffle']['ph']['top']:.2f})", fontsize=10)
        a.legend(fontsize=8)
    a = axes[3]
    x = np.arange(3)
    real = [res[s]["real"]["ph"]["top"] for s in ["wake", "rem", "nrem"]]
    nl = [R["null"][s][:, 1] for s in ["wake", "rem", "nrem"]]
    a.bar(x - .18, real, .36, color=[STATE_COL[s] for s in ["wake", "rem", "nrem"]], label="data")
    a.bar(x + .18, [n.mean() for n in nl], .36, yerr=[n.std() for n in nl],
          color="#bdc3c7", label="shuffle (n=5)")
    a.set_xticks(x); a.set_xticklabels(["wake", "REM", "NREM"])
    a.set_ylabel("longest H1 lifetime"); a.legend(fontsize=8)
    a.set_title("one persistent hole in every state", fontsize=10)
    fig.suptitle(f"{SES}: persistent homology of the population activity "
                 f"(H1 = loops)", y=1.02)
    fig.tight_layout()
    path = f"{outdir}/fig05_topology.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- fig 6 ----
def fig06(R, outdir="figures"):
    res, st, keep, pref, SES = _setup(R)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    o = np.argsort(pref)
    for i, s in enumerate(["wake", "rem", "nrem"]):
        r = res[s]["real"]
        _, off, sgn = hc.best_circ_align(r["ipref"], pref)
        M = r["itc"][:, o]
        M = (M - M.min(0)) / (M.max(0) - M.min(0) + 1e-9)
        # rotate the internal coordinate onto the wake frame for display
        idx = np.argsort(np.mod(sgn * r["itc_idx"] + off, hc.TWOPI))
        a = axes[0, i]
        im = a.imshow(M[idx].T, aspect="auto", origin="lower", cmap="viridis",
                      extent=[0, hc.TWOPI, 0, len(keep)])
        a.set_title(f"{STATE_LAB[s]}: firing vs *internal* ring angle", fontsize=10)
        a.set_xlabel("internal manifold angle (aligned, rad)")
        if i == 0:
            a.set_ylabel("cell, sorted by wake preferred direction")
        a = axes[1, i]
        a.scatter(np.mod(sgn * r["ipref"] + off, hc.TWOPI), pref, s=16, color=STATE_COL[s])
        a.set_xlabel("preferred angle on the internal ring")
        a.set_ylabel("wake preferred direction" if i == 0 else "")
        a.set_title(f"circular r = {r['cc_pref']:.2f}", fontsize=10)
    fig.suptitle(f"{SES}: the ordering of cells around the internally generated ring is the "
                 f"waking head-direction map", y=1.0)
    fig.tight_layout()
    path = f"{outdir}/fig06_internal_tuning.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- fig 7 ----
def fig07(R, outdir="figures"):
    res, st, keep, pref, SES = _setup(R)
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(3, 3, figure=fig, hspace=.6, wspace=.3, height_ratios=[1, 1, 1.1])
    for i, s in enumerate(["wake", "rem", "nrem"]):
        r = res[s]["real"]
        a = fig.add_subplot(gs[i, :2])
        t = r["prob_t"]
        # pick the longest continuous stretch and show 40 s of it
        d = np.diff(t); brk = np.where(d > 3 * hc.EMBED["bin_size"])[0]
        seg = np.split(np.arange(len(t)), brk + 1)
        seg = max(seg, key=len)
        span = 20 if s == "nrem" else 40
        seg = seg[:int(span / hc.EMBED["bin_size"])]
        P = r["prob"][seg]
        a.imshow(P.T, aspect="auto", origin="lower", cmap="magma",
                 extent=[t[seg][0], t[seg][-1], 0, hc.TWOPI])
        a.plot(t[seg], r["dec"][seg], ".", ms=2.5, color="#2ecc71")
        if s == "wake":
            tr = r["true"][seg].copy()
            tr[np.abs(np.diff(tr, prepend=tr[0])) > np.pi] = np.nan   # hide 2pi wraps
            a.plot(r["prob_t"][seg], tr, "-", lw=1.2, color="w", alpha=.8)
        a.set_ylabel("HD (rad)")
        a.set_title(f"{STATE_LAB[s]}: decoded posterior (green = peak"
                    f"{', white = measured HD' if s=='wake' else ''})", fontsize=10)
        if i == 2:
            a.set_xlabel("time (s)")

    a = fig.add_subplot(gs[0, 2])
    grid = np.arange(0, 740, 5)
    for s in ["wake", "rem", "nrem"]:
        v = np.sort(res[s]["real"]["speed"])
        a.plot(grid, np.searchsorted(v, grid) / len(v), color=STATE_COL[s], lw=1.8,
               label=f"{STATE_LAB[s]} (median {np.median(v):.0f}°/s)")
    v = np.sort(res["nrem"]["shuffle"]["speed"])
    a.plot(grid, np.searchsorted(v, grid) / len(v), color="k", ls=":", lw=1.5,
           label=f"shuffle (median {np.median(v):.0f}°/s)")
    a.set_xlabel("|angular velocity| of the bump (deg/s)")
    a.set_ylabel("cumulative fraction of bins")
    a.legend(fontsize=7, loc="lower right"); a.set_title("the bump moves in small steps", fontsize=10)

    a = fig.add_subplot(gs[1, 2])
    x = np.arange(3)
    a.bar(x - .18, [np.degrees(res[s]["real"]["split"]["median_abs_err"])
                    for s in ["wake", "rem", "nrem"]], .36,
          color=[STATE_COL[s] for s in ["wake", "rem", "nrem"]], label="data")
    a.bar(x + .18, [np.degrees(res[s]["shuffle"]["split"]["median_abs_err"])
                    for s in ["wake", "rem", "nrem"]], .36, color="#bdc3c7", label="shuffle")
    a.set_xticks(x); a.set_xticklabels(["wake", "REM", "NREM"])
    a.set_ylabel("median disagreement (deg)"); a.legend(fontsize=8)
    a.set_title("split-half decoder disagreement", fontsize=10)

    a = fig.add_subplot(gs[2, 2])
    r = res["rem"]["real"]["split"]
    a.scatter(r["a"], r["b"], s=5, alpha=.3, color=STATE_COL["rem"])
    a.set_xlabel("HD decoded from half A (rad)"); a.set_ylabel("half B (rad)")
    a.set_title(f"REM: independent halves agree\ncircular r = {r['cc']:.2f}", fontsize=10)
    fig.suptitle(f"{SES}: a single, sharp, continuously moving activity bump in sleep", y=.99)
    path = f"{outdir}/fig07_bump_dynamics.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- fig 8 ----
def fig08(R, outdir="figures"):
    res, st, keep, pref, SES = _setup(R)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    o = np.argsort(pref)
    for i, s in enumerate(["wake", "rem", "nrem"]):
        C = res[s]["real"]["C"][np.ix_(o, o)]
        v = np.nanpercentile(np.abs(C - np.diag(np.diag(C))), 99)
        im = axes[i].imshow(C, cmap="RdBu_r", vmin=-v, vmax=v)
        axes[i].set_title(f"{STATE_LAB[s]}: pairwise correlations\n(cells sorted by preferred "
                          f"direction)", fontsize=10)
        axes[i].set_xlabel("cell"); axes[i].set_ylabel("cell")
        plt.colorbar(im, ax=axes[i], fraction=.046)
    a = axes[3]
    for s in ["wake", "rem", "nrem"]:
        a.plot(np.degrees(res[s]["real"]["offs"]), res[s]["real"]["corroff"], "o-",
               color=STATE_COL[s], label=STATE_LAB[s], ms=4)
    a.plot(np.degrees(res["nrem"]["shuffle"]["offs"]), res["nrem"]["shuffle"]["corroff"], "k:",
           label="shuffle")
    a.axhline(0, color="k", lw=.6)
    a.set_xlabel("difference in preferred direction (deg)")
    a.set_ylabel("pairwise correlation")
    a.legend(fontsize=8)
    a.set_title("correlation structure is inherited\nfrom the waking map", fontsize=10)
    fig.suptitle(f"{SES}: cell-cell coordination during sleep follows waking preferred directions",
                 y=1.02)
    fig.tight_layout()
    path = f"{outdir}/fig08_pairwise_correlation.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- fig 9 ----
def fig09(R, outdir="figures"):
    res, st, keep, pref, SES = _setup(R)
    tc = R["tc"]
    rng = np.random.default_rng(0)
    conds = []
    w = res["wake"]["real"]
    conds.append(("wake\nmeasured HD", w["true"][w["mask"]][w["sel"]], w["counts"][w["sel"]], "wake"))
    for s in ["wake", "rem", "nrem"]:
        r = res[s]["real"]
        _, off, sgn = hc.best_circ_align(r["ipref"], pref)
        conds.append((f"{STATE_LAB[s]}\ninternal",
                      np.mod(sgn * r["ring"]["angle"] + off, hc.TWOPI), r["counts"][r["sel"]], s))
    pr2, nulls, cols = [], [], []
    for lab, ang, cnt, s in conds:
        pr2.append(hg.cv_pseudo_r2(ang, cnt))
        nulls.append(hg.cv_pseudo_r2(hg.shift_null(ang, rng), cnt))
        cols.append(STATE_COL[s])

    fig = plt.figure(figsize=(15, 4.6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.4, 1], wspace=.32)
    a = fig.add_subplot(gs[0, 0])
    for i, (p, n, c) in enumerate(zip(pr2, nulls, cols)):
        a.boxplot([p], positions=[i - .18], widths=.32, patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor=c, alpha=.8), medianprops=dict(color="k"))
        a.boxplot([n], positions=[i + .18], widths=.32, patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor="#bdc3c7"), medianprops=dict(color="k"))
    a.axhline(0, color="k", lw=.6)
    a.set_xticks(range(len(conds)))
    a.set_xticklabels([c[0].replace(" (arena)", "").replace(" sleep", "") for c in conds],
                      fontsize=8)
    a.set_ylabel("cross-validated pseudo-$R^2$ per cell")
    a.set_title("GLM: the internal ring angle predicts\nspiking in sleep (grey = shifted null)",
                fontsize=10)

    a = fig.add_subplot(gs[0, 2])
    a.scatter(pr2[0], pr2[2], s=18, color=STATE_COL["rem"], label="REM")
    a.scatter(pr2[0], pr2[3], s=18, color=STATE_COL["nrem"], label="NREM")
    lim = [min(0, min(pr2[0])), max(pr2[0]) * 1.05]
    a.plot(lim, lim, "k-", lw=.8); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("pseudo-$R^2$, wake (measured HD)")
    a.set_ylabel("pseudo-$R^2$, sleep (internal angle)")
    a.legend(fontsize=8); a.set_title("cell by cell", fontsize=10)

    gs2 = gs[0, 1].subgridspec(2, 3, hspace=.55, wspace=.35)
    strong = keep[st.loc[keep, "mvl"].values >= np.median(st.loc[keep, "mvl"].values)]
    ex = strong[np.argsort(pref[np.isin(keep, strong)])][np.linspace(0, len(strong) - 1, 6).astype(int)]
    idx = {u: i for i, u in enumerate(keep)}
    curves = {}
    for s in ["rem", "nrem"]:
        r = res[s]["real"]
        _, off, sgn = hc.best_circ_align(r["ipref"], pref)
        curves[s] = hg.glm_tuning(np.mod(sgn * r["ring"]["angle"] + off, hc.TWOPI),
                                  r["counts"][r["sel"]], bin_size=hc.EMBED["bin_size"])
    for i, u in enumerate(ex):
        ax = fig.add_subplot(gs2[i // 3, i % 3])
        ax.plot(tc.index.values, tc[u].values / tc[u].values.max(), color="k", lw=1.4,
                label="wake (measured HD)")
        for s in ["rem", "nrem"]:
            g, cur = curves[s]
            ax.plot(g, cur[:, idx[u]] / cur[:, idx[u]].max(), color=STATE_COL[s], lw=1.2,
                    label=f"{s.upper()} GLM (internal)")
        ax.set_xticks([0, np.pi, hc.TWOPI]); ax.set_xticklabels(["0", "π", "2π"], fontsize=7)
        ax.set_yticks([]); ax.set_title(f"#{u}", fontsize=8)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")
    fig.text(0.44, 1.0, "GLM tuning curves from the sleep-internal angle vs waking tuning",
             ha="center", fontsize=10)
    fig.suptitle(f"{SES}: a Poisson GLM driven by the internally generated angle reproduces "
                 f"waking tuning", y=1.08)
    path = f"{outdir}/fig09_glm_internal_coordinate.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def fig10(df, outdir="figures"):
    """Cross-session summary from `cross_session_metrics.csv`."""
    states = ["wake", "rem", "nrem"]
    real = df[df["cond"] == "real"]
    shuf = df[df["cond"] != "real"]
    metrics = [("ring_score", "ring score (annular geometry)"),
               ("h1_top", "longest H1 lifetime (topology)"),
               ("split_cc", "split-half decoder agreement"),
               ("cc_internal_pref", "internal ring order vs\nwaking preferred direction (circ. r)")]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), layout="constrained")
    for k, (col, lab) in enumerate(metrics):
        a = axes.flat[k]
        for i, s in enumerate(states):
            r = real[real["state"] == s].set_index("session")[col]
            sh = shuf[shuf["state"] == s].set_index("session")[col]
            common = r.index.intersection(sh.index)
            for ses in common:
                a.plot([i - .18, i + .18], [r[ses], sh[ses]], "-", color="#cccccc", lw=.7, zorder=1)
            jr = np.random.default_rng(0).uniform(-.05, .05, len(r))
            js = np.random.default_rng(1).uniform(-.05, .05, len(sh))
            a.scatter(i - .18 + jr, r.values, s=20, color=STATE_COL[s], zorder=3,
                      label="data" if i == 0 else None)
            a.scatter(i + .18 + js, sh.values, s=20, color="#7f8c8d", zorder=3,
                      label="shuffle" if i == 0 else None)
            a.plot([i - .32, i - .04], [np.median(r), np.median(r)], "k-", lw=2, zorder=4)
            a.plot([i + .04, i + .32], [np.median(sh), np.median(sh)], "k-", lw=2, zorder=4)
        a.set_xticks(range(3)); a.set_xticklabels(["wake", "REM", "NREM"])
        a.set_ylabel(lab, fontsize=9)
        a.axhline(0, color="k", lw=.5)
        if k == 0:
            a.legend(fontsize=8, loc="lower left")

    a = axes.flat[4]
    for s in ["rem", "nrem"]:
        r = real[real["state"] == s]
        a.scatter(r["n_cells"], r["split_cc"], s=24, color=STATE_COL[s], label=STATE_LAB[s])
    a.axhline(0, color="k", lw=.5)
    a.set_xlabel("number of head-direction cells"); a.set_ylabel("split-half decoder agreement")
    a.legend(fontsize=8)
    a.set_title("moment-to-moment coherence needs a\nlarge enough population, especially in NREM",
                fontsize=9)

    a = axes.flat[5]
    for i, s in enumerate(states):
        r = real[real["state"] == s]["med_speed"]
        sh = shuf[shuf["state"] == s]["med_speed"]
        a.scatter(i - .18 + np.random.default_rng(0).uniform(-.05, .05, len(r)), r.values,
                  s=20, color=STATE_COL[s])
        a.scatter(i + .18 + np.random.default_rng(1).uniform(-.05, .05, len(sh)), sh.values,
                  s=20, color="#7f8c8d")
        a.plot([i - .32, i - .04], [np.median(r)] * 2, "k-", lw=2)
        a.plot([i + .04, i + .32], [np.median(sh)] * 2, "k-", lw=2)
    a.set_xticks(range(3)); a.set_xticklabels(["wake", "REM", "NREM"])
    a.set_ylabel("median |angular velocity| of the bump (deg/s)", fontsize=9)
    a.set_title("the bump drifts slowly in REM and\nfast in NREM, never randomly", fontsize=9)
    fig.suptitle(f"Cross-session summary: {real['session'].nunique()} sessions, "
                 f"{real['subject'].nunique()} mice (DANDI 000939); grey = per-cell shuffle")
    path = f"{outdir}/fig10_cross_session.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
