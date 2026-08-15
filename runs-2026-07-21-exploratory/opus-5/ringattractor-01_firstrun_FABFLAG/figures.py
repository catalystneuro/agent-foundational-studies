"""Publication-style figures for the head-direction ring-attractor analysis."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import analysis as A
import ring_lib as R

STATE_COLORS = {"wake": "#2a9d8f", "rem": "#e76f51", "sws": "#4361ee"}
STATE_LABELS = {"wake": "Wake (exploration)", "rem": "REM sleep", "sws": "non-REM sleep"}

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
def fig_overview(S, fname="fig01_session_overview.png"):
    """Raw behaviour, hypnogram and population activity for one session."""
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.4], hspace=0.55, wspace=0.3)

    states = S["states"]
    cmap = {"Awake": "#2a9d8f", "Non-REM": "#4361ee", "REM": "#e76f51"}
    ax = fig.add_subplot(gs[0, :])
    for _, r in states.iterrows():
        ax.axvspan(r.start_time / 60, r.stop_time / 60, color=cmap[r.label], alpha=0.85, lw=0)
    ax.set_xlim(0, states.stop_time.max() / 60)
    ax.set_yticks([])
    ax.set_xlabel("time (min)")
    ax.set_title(f"{S['name']}: scored brain state over the whole recording")
    handles = [plt.Line2D([], [], color=c, lw=6, label=k) for k, c in cmap.items()]
    ax.legend(handles=handles, ncol=3, loc="upper right", frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, :])
    sp = S["speed"]
    ax.plot(np.asarray(sp.index.values) / 60, np.asarray(sp.values), lw=0.3, color="0.3")
    for s, e in zip(S["epochs"]["explore"].start, S["epochs"]["explore"].end):
        ax.axvspan(s / 60, e / 60, color="#2a9d8f", alpha=0.25, lw=0)
    ax.set_ylabel("speed (cm/s)")
    ax.set_xlabel("time (min)")
    ax.set_ylim(0, np.nanpercentile(np.asarray(sp.values), 99.5))
    ax.set_xlim(0, states.stop_time.max() / 60)
    ax.set_title("locomotion speed; shading marks the exploration epochs used for tuning curves")

    ep = S["epochs"]["explore"]
    ax = fig.add_subplot(gs[2, 0])
    pos = S["position"].restrict(ep)
    ax.plot(np.asarray(pos["x"]), np.asarray(pos["y"]), lw=0.2, color="0.3")
    ax.set_aspect("equal")
    ax.set_title("path in the open field")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")

    ax = fig.add_subplot(gs[2, 1])
    t0 = ep.start[3]
    win = nap.IntervalSet(t0, t0 + 60)
    a = S["angle"].restrict(win)
    ax.plot(np.asarray(a.index.values) - t0, np.degrees(np.asarray(a.values)), ".", ms=1)
    ax.set_ylabel("head direction (deg)")
    ax.set_xlabel("time (s)")
    ax.set_title("reconstructed head direction")

    ax = fig.add_subplot(gs[2, 2])
    ax.hist(np.degrees(np.asarray(S["angle"].restrict(ep).values)), bins=36, color="0.5")
    ax.set_xlabel("head direction (deg)")
    ax.set_ylabel("samples")
    ax.set_title("directional occupancy")

    ax = fig.add_subplot(gs[3, :])
    spk = S["spikes"]
    win = nap.IntervalSet(t0, t0 + 20)
    for i, k in enumerate(spk.keys()):
        ts = np.asarray(spk[k].restrict(win).index.values) - t0
        ax.plot(ts, np.full_like(ts, i), "|", ms=3, color="k", mew=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("unit")
    ax.set_title("raw spike raster, all recorded units (20 s of exploration)")
    fig.savefig(fname)
    return fig


# ---------------------------------------------------------------------------
def fig_tuning(sel, fname="fig02_hd_tuning.png"):
    tc, mvl, pd_, rate = sel["tc"], sel["mvl"], sel["pd"], sel["rate"]
    mask = sel["mask"]
    order = np.argsort(-mvl)
    show = [i for i in order if mask[i]][:18]

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(4, 6, hspace=0.75, wspace=0.5)
    th = tc.index.values
    for j, i in enumerate(show):
        ax = fig.add_subplot(gs[j // 6, j % 6], projection="polar")
        u = tc.columns[i]
        y = tc[u].values
        ax.plot(np.append(th, th[0]), np.append(y, y[0]), color="#264653", lw=1.2)
        ax.fill(np.append(th, th[0]), np.append(y, y[0]), color="#2a9d8f", alpha=0.3)
        ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"r={mvl[i]:.2f}\n{y.max():.0f} Hz", fontsize=7, pad=10)
    fig.text(0.5, 0.985, "Wake head-direction tuning curves (18 strongest HD cells)",
             ha="center", fontsize=11)

    ax = fig.add_subplot(gs[3, 0:2])
    ax.hist(mvl, bins=25, color="0.7", label="observed")
    ax.hist(mvl[mask], bins=25, color="#2a9d8f", label="HD cells")
    ax.axvline(np.median(sel["null_mvl"]), color="r", ls="--", lw=1,
               label="median of\ntime-shift null")
    ax.set_xlabel("mean vector length")
    ax.set_ylabel("units")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"{mask.sum()}/{len(mvl)} units classified as HD")

    ax = fig.add_subplot(gs[3, 2:4], projection="polar")
    ax.hist(pd_[mask], bins=18, color="#2a9d8f")
    ax.set_yticklabels([])
    ax.set_title("preferred directions\ntile the circle", fontsize=9, pad=15)

    ax = fig.add_subplot(gs[3, 4:6])
    ax.scatter(rate, mvl, s=12, c=np.where(mask, "#2a9d8f", "0.7"))
    ax.set_xscale("log")
    ax.set_xlabel("mean rate during exploration (Hz)")
    ax.set_ylabel("mean vector length")
    ax.set_title("tuning strength vs firing rate")
    fig.savefig(fname)
    return fig


# ---------------------------------------------------------------------------
def _bump_panel(ax, S, sel, ep, t0, dur, bin_size, smooth, show_true=False, cbar=False):
    ids, pdh = sel["ids"], sel["pd_hd"]
    spk = S["spikes"][ids]
    tch = sel["tc"][ids]
    order = np.argsort(pdh)
    pd_sorted = np.degrees(np.mod(pdh[order], 2 * np.pi))
    win = nap.IntervalSet(t0, t0 + dur)
    Rr = R.binned_rates(spk, win, bin_size, smooth)
    X = np.asarray(Rr.values)[:, order]
    X = X / np.maximum(X.max(0, keepdims=True), 1e-9)
    im = ax.imshow(X.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, dur, 0, len(ids)], interpolation="nearest", vmin=0, vmax=1)
    dec, _ = nap.decode_1d(tuning_curves=tch, group=spk, ep=win,
                           bin_size=bin_size, feature=None)
    grid = np.sort(pd_sorted)
    y = np.interp(np.degrees(np.mod(np.asarray(dec.values), 2 * np.pi)), grid,
                  np.arange(len(ids)))
    ax.plot(np.asarray(dec.index.values) - t0, y, ".", color="w", ms=2.5,
            label="decoded direction")
    if show_true:
        tr = dec.value_from(S["angle"])
        yt = np.interp(np.degrees(np.mod(np.asarray(tr.values), 2 * np.pi)), grid,
                       np.arange(len(ids)))
        ax.plot(np.asarray(tr.index.values) - t0, yt, ".", color="#00e5ff", ms=2.5,
                label="measured head direction")
    ticks = np.linspace(0, len(ids) - 1, 5).astype(int)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{grid[i]:.0f}" for i in ticks])
    ax.set_ylabel("preferred direction (deg)")
    ax.set_xlabel("time (s)")
    if cbar:
        plt.colorbar(im, ax=ax, pad=0.01, label="norm. rate")
    return im


def fig_bump(S, sel, fname="fig03_bump_raster.png"):
    ep = S["epochs"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5))
    w = ep["explore"]
    i = np.argmax(w.end - w.start)
    _bump_panel(axes[0], S, sel, w, w.start[i], 20, 0.1, 1.5, show_true=True, cbar=True)
    axes[0].legend(loc="upper right", fontsize=7, markerscale=3, framealpha=0.4)
    axes[0].set_title("Wake exploration: the activity bump tracks the measured head direction")

    r = ep["rem"]
    i = np.argmax(r.end - r.start)
    _bump_panel(axes[1], S, sel, r, r.start[i] + 40, 20, 0.1, 1.5, cbar=True)
    axes[1].set_title("REM sleep: the same bump sweeps the ring with no sensory input")

    s = ep["sws"]
    i = np.argmax(s.end - s.start)
    _bump_panel(axes[2], S, sel, s, s.start[i] + 100, 20, 0.025, 2.0, cbar=True)
    axes[2].set_title("non-REM sleep: a coherent bump persists and moves faster")
    fig.tight_layout()
    fig.savefig(fname)
    return fig


# ---------------------------------------------------------------------------
def fig_pairwise(res, fname="fig04_pairwise_correlations.png"):
    pdh = res["sel"]["pd_hd"]
    order = np.argsort(pdh)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    iu = np.triu_indices(len(pdh), 1)
    for j, st in enumerate(A.STATES):
        d = res["states"][st]
        C = d["corr_matrix"][np.ix_(order, order)]
        ax = axes[0, j]
        v = np.nanpercentile(np.abs(C[np.triu_indices(len(pdh), 1)]), 98)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"{STATE_LABELS[st]}\n({A.BINS[st]*1000:.0f} ms bins)")
        ax.set_xlabel("cell (sorted by PD)")
        ax.set_ylabel("cell (sorted by PD)")
        plt.colorbar(im, ax=ax, fraction=0.046, label="corr")

        ax = axes[1, j]
        ax.scatter(np.degrees(d["dpd"]), d["cvals"], s=5, alpha=0.25, color="0.5")
        ctr, m, s = d["profile"]
        ax.errorbar(np.degrees(ctr), m, s, color=STATE_COLORS[st], lw=2, marker="o", ms=4)
        ax.axhline(0, ls="--", lw=0.7, color="k")
        ax.set_xlabel("|Δ preferred direction| (deg)")
        ax.set_ylabel("pairwise correlation")
        ax.set_title(f"r with cos(ΔPD) = {d['r_cos']:.2f}")

    for j, st in enumerate(("rem", "sws")):
        ax = axes[2, j]
        x = res["states"]["wake"]["corr_matrix"][iu]
        y = res["states"][st]["corr_matrix"][iu]
        ax.scatter(x, y, s=6, alpha=0.4, color=STATE_COLORS[st])
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.set_xlabel("wake pairwise correlation")
        ax.set_ylabel(f"{STATE_LABELS[st]} correlation")
        ax.set_title(f"r = {res['corr_similarity']['wake_vs_' + st]:.2f}")
    axes[2, 2].axis("off")
    fig.suptitle("Correlation structure of the HD population is inherited by sleep", y=1.0)
    fig.tight_layout()
    fig.savefig(fname)
    return fig


# ---------------------------------------------------------------------------
def fig_manifold(res, fname="fig05_manifold.png"):
    fig, axes = plt.subplots(3, 3, figsize=(13, 11.5))
    pdh = res["sel"]["pd_hd"]
    for j, st in enumerate(A.STATES):
        d = res["states"][st]
        emb, theta = d["emb_iso"], d["theta"]
        ax = axes[0, j]
        ax.scatter(emb[:, 0], emb[:, 1], c=np.mod(theta, 2 * np.pi), cmap="hsv", s=4)
        ax.set_aspect("equal")
        ax.set_title(f"{STATE_LABELS[st]}\nIsomap of population vectors")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")

        ax = axes[1, j]
        al = d["align"]
        pred = np.mod(al["sign"] * pdh + al["offset"], 2 * np.pi)
        ax.scatter(np.degrees(np.mod(pdh, 2 * np.pi)),
                   np.degrees(np.mod(d["pd_int"], 2 * np.pi)),
                   s=25, color=STATE_COLORS[st])
        o = np.argsort(np.degrees(np.mod(pdh, 2 * np.pi)))
        ax.plot(np.degrees(np.mod(pdh, 2 * np.pi))[o], np.degrees(pred)[o], ".",
                color="k", ms=3, label="rigid fit")
        ax.set_xlabel("wake preferred direction (deg)")
        ax.set_ylabel("internal ring position (deg)")
        ax.set_title(f"R = {al['R']:.2f}  (shuffle {d['align_shuffle'].mean():.2f})")
        ax.legend(fontsize=7, frameon=False)

        ax = axes[2, j]
        ax.hist(d["align_shuffle"], bins=8, color="0.7", label="time-shift shuffle")
        ax.axvline(al["R"], color=STATE_COLORS[st], lw=2, label="observed")
        ax.set_xlim(0, 1)
        ax.set_xlabel("alignment R")
        ax.set_ylabel("count")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("An unsupervised ring coordinate recovers each cell's wake preferred direction",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(fname)
    return fig


# ---------------------------------------------------------------------------
def fig_coherence(res, fname="fig06_bump_coherence.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sts = A.STATES
    x = np.arange(len(sts))

    ax = axes[0]
    obs = [res["states"][s]["split_half"].mean() for s in sts]
    err = [res["states"][s]["split_half"].std() for s in sts]
    nul = [res["states"][s]["split_half_shuffle"].mean() for s in sts]
    nerr = [res["states"][s]["split_half_shuffle"].std() for s in sts]
    ax.bar(x - 0.18, obs, 0.35, yerr=err, color=[STATE_COLORS[s] for s in sts], label="observed")
    ax.bar(x + 0.18, nul, 0.35, yerr=nerr, color="0.75", label="cell-identity shuffle")
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABELS[s].split(" ")[0] for s in sts])
    ax.set_ylabel("split-half agreement (R)")
    ax.set_title("Two disjoint halves of the population\ndecode the same direction")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for s in sts:
        c = res["states"][s]["conc"]
        ax.hist(c, bins=50, range=(0, 1), density=True, histtype="step", lw=1.6,
                color=STATE_COLORS[s], label=f"{STATE_LABELS[s]} ({np.mean(c):.2f})")
    ax.set_xlabel("posterior concentration (resultant length)")
    ax.set_ylabel("density")
    ax.set_title("The decoded posterior is a single sharp bump")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    for s in sts:
        av = np.degrees(res["states"][s]["ang_vel"])
        ax.hist(av, bins=120, range=(-900, 900), density=True, histtype="step", lw=1.6,
                color=STATE_COLORS[s], label=f"{STATE_LABELS[s]}")
    avt = res["states"]["wake"].get("ang_vel_true")
    if avt is not None:
        ax.hist(np.degrees(avt), bins=120, range=(-900, 900), density=True,
                histtype="step", lw=1.2, ls="--", color="k", label="measured head AV")
    ax.set_yscale("log")
    ax.set_xlabel("angular velocity of the bump (deg/s)")
    ax.set_ylabel("density")
    ax.set_title(f"Bump motion is continuous\n({A.DRIFT_BIN*1000:.0f} ms bins, matched)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(fname)
    return fig


# ---------------------------------------------------------------------------
def fig_multisession(allres, fname="fig07_multisession_summary.png"):
    ok = [r for r in allres if r.get("ok")]
    names = [r["name"] for r in ok]
    sts = A.STATES
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    ax = axes[0, 0]
    w = 0.26
    for i, s in enumerate(sts):
        ax.bar(np.arange(len(ok)) + (i - 1) * w, [r["states"][s]["r_cos"] for r in ok],
               w, color=STATE_COLORS[s], label=STATE_LABELS[s])
    ax.set_xticks(np.arange(len(ok)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("r(pairwise corr, cos ΔPD)")
    ax.set_title("Cosine correlation structure")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[0, 1]
    for i, s in enumerate(sts):
        ax.bar(np.arange(len(ok)) + (i - 1) * w, [r["states"][s]["align_R"] for r in ok],
               w, color=STATE_COLORS[s])
        ax.plot(np.arange(len(ok)) + (i - 1) * w,
                [r["states"][s]["align_shuffle"].mean() for r in ok], "k_", ms=8)
    ax.set_xticks(np.arange(len(ok)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("internal-ring alignment R")
    ax.set_title("Ring coordinate vs wake PDs\n(black ticks: shuffle)")

    ax = axes[0, 2]
    for i, s in enumerate(sts):
        ax.bar(np.arange(len(ok)) + (i - 1) * w,
               [r["states"][s]["split_half"].mean() for r in ok], w, color=STATE_COLORS[s])
        ax.plot(np.arange(len(ok)) + (i - 1) * w,
                [r["states"][s]["split_half_shuffle"].mean() for r in ok], "k_", ms=8)
    ax.set_xticks(np.arange(len(ok)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("split-half agreement R")
    ax.set_title("Bump coherence\n(black ticks: shuffle)")

    ax = axes[1, 0]
    for i, s in enumerate(sts):
        y = [r["corr_similarity"]["wake_vs_" + s] for r in ok] if s != "wake" else None
        if y is None:
            continue
        ax.bar(np.arange(len(ok)) + (i - 1.5) * w, y, w, color=STATE_COLORS[s],
               label=f"wake vs {s}")
    ax.set_xticks(np.arange(len(ok)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("r between correlation matrices")
    ax.set_title("Wake correlation structure preserved in sleep")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    for i, s in enumerate(sts):
        vals = [np.degrees(r["states"][s]["drift_speed"]) for r in ok]
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.12, 0.12, len(vals)),
                   vals, color=STATE_COLORS[s], s=25)
        ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, "k-", lw=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([STATE_LABELS[s].split(" ")[0] for s in sts])
    ax.set_ylabel("median |bump angular velocity| (deg/s)")
    ax.set_title("Drift speed of the internal bump")

    ax = axes[1, 2]
    txt = [f"{len(ok)} sessions, {sum(r['n_hd'] for r in ok)} HD cells "
           f"of {sum(r['n_units'] for r in ok)} units", ""]
    for s in sts:
        txt.append(f"{STATE_LABELS[s]}:")
        txt.append(f"   r(cos ΔPD)      {np.mean([r['states'][s]['r_cos'] for r in ok]):.2f}")
        txt.append(f"   alignment R     {np.mean([r['states'][s]['align_R'] for r in ok]):.2f}"
                   f"  (shuf {np.mean([r['states'][s]['align_shuffle'].mean() for r in ok]):.2f})")
        txt.append(f"   split-half R    {np.mean([r['states'][s]['split_half'].mean() for r in ok]):.2f}"
                   f"  (shuf {np.mean([r['states'][s]['split_half_shuffle'].mean() for r in ok]):.2f})")
        txt.append("")
    ax.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=8)
    ax.axis("off")
    fig.suptitle("Ring structure across sessions and animals", y=1.0)
    fig.tight_layout()
    fig.savefig(fname)
    return fig


# ---------------------------------------------------------------------------
def fig_dimensionality(S, sel, res, fname="fig08_dimensionality.png"):
    """Eigenvalue spectrum of the HD-cell correlation matrix per brain state.

    A population whose activity lives on a ring with cosine-like tuning has a
    correlation matrix dominated by exactly two eigenvalues (the cosine and sine
    harmonics of the ring coordinate). Independent circular time shifts of each
    spike train give the matched null spectrum.
    """
    ids = sel["ids"]
    spk = S["spikes"][ids]
    n = len(ids)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    frac2 = {}
    for j, st in enumerate(A.STATES):
        ep = A.epoch_of(S, st)
        bs, sm = A.BINS[st], A.SMOOTH[st]
        ev = np.sort(np.linalg.eigvalsh(res["states"][st]["corr_matrix"]))[::-1]
        nulls = []
        for k in range(5):
            sh = R.shuffle_population(spk, ep, seed=200 + k)
            Cn = R.pairwise_corr(sh, ep, bs, sm)
            nulls.append(np.sort(np.linalg.eigvalsh(Cn))[::-1])
        nulls = np.array(nulls)
        ax = axes[j]
        ax.plot(np.arange(1, n + 1), ev / n * 100, "o-", ms=4,
                color=STATE_COLORS[st], label="observed")
        ax.fill_between(np.arange(1, n + 1), nulls.min(0) / n * 100,
                        nulls.max(0) / n * 100, color="0.75",
                        label="time-shift shuffle")
        ax.set_xlabel("component")
        ax.set_ylabel("variance explained (%)")
        frac2[st] = ev[:2].sum() / ev.sum()
        ax.set_title(f"{STATE_LABELS[st]}\nfirst 2 components: {100*frac2[st]:.0f}% "
                     f"(shuffle {100*nulls[:, :2].sum(1).mean()/nulls.sum(1).mean():.0f}%)")
        ax.legend(fontsize=7, frameon=False)
        ax.set_xlim(0.5, min(n, 20) + 0.5)
    fig.suptitle("Two dimensions dominate the population covariance in every state, "
                 "as expected for a ring", y=1.02)
    fig.tight_layout()
    fig.savefig(fname)
    return fig, frac2
