"""Figure generation for the head-direction ring-attractor analysis."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from hdcells import ANG
from ringanalysis import manifold_input, ring_embedding, ring_stats, shift_shuffle
from ringanalysis import taubin_center, split_half_decode, circ_diff, circ_corr
from ringanalysis import bayesian_decode

TWO_PI = 2 * np.pi
STATE_COLORS = {"wake": "#1b6ca8", "REM": "#c0392b", "nREM": "#7d3c98"}
STATE_LABEL = {"wake": "Wake", "REM": "REM sleep", "nREM": "non-REM sleep"}
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})


def _hd_raster(ax, spikes, units, pref, ep_window, t_origin=0.0):
    order = np.argsort(pref)
    cmap = plt.get_cmap("hsv")
    for row, i in enumerate(order):
        st = spikes[units[i]].get(ep_window[0], ep_window[1]).index.values - t_origin
        ax.plot(st, np.full_like(st, row), "|", ms=3, color=cmap(pref[i] / TWO_PI), mew=0.8)
    ax.set_ylim(-1, len(order))
    ax.set_xlim(ep_window[0] - t_origin, ep_window[1] - t_origin)


def smooth_decode(counts, tc, bin_size, win=3):
    """Poisson MAP decoding from counts pooled over `win` bins, for display."""
    from scipy.ndimage import uniform_filter1d
    c = uniform_filter1d(np.asarray(counts, float), win, axis=0) * win
    return bayesian_decode(c, tc, bin_size * win)[0]


# --------------------------------------------------------------- figure 1 ---
def _best_wake_window(hd, wake, dur=60.0):
    """Pick the window of `dur` seconds in which the head direction moves the most."""
    best, best_disp = None, -1
    for a, b in zip(wake.start, wake.end):
        if b - a < dur:
            continue
        for t0 in np.arange(a, b - dur, dur):
            h = hd.get(t0, t0 + dur).values
            if len(h) < 100:
                continue
            disp = 1 - np.abs(np.exp(1j * h).mean())
            if disp > best_disp:
                best_disp, best = disp, (t0, t0 + dur)
    return best


def fig_overview(res, s, fname):
    hd, spikes, ep = s["hd"], s["spikes"], s["epochs"]
    units, pref = res["hd_units"], res["pref"]
    fig = plt.figure(figsize=(12.5, 8.5))
    gs = GridSpec(3, 2, height_ratios=[0.55, 1.4, 1.4], hspace=0.55, wspace=0.25)

    # hypnogram
    ax = fig.add_subplot(gs[0, :])
    for lab, c, y in [("Awake", "#1b6ca8", 2), ("REM", "#c0392b", 1), ("Non-REM", "#7d3c98", 0)]:
        e = ep[lab]
        for a, b in zip(e.start, e.end):
            ax.add_patch(plt.Rectangle((a / 60, y - 0.35), (b - a) / 60, 0.7, color=c))
    ax.set_ylim(-0.6, 2.6); ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["non-REM", "REM", "awake"])
    ax.set_xlim(0, max(ep["Non-REM"].end.max(), ep["Awake"].end.max()) / 60)
    ax.set_xlabel("time in session (min)")
    ax.set_title(f"{res['sid']}  ({res['subject']}): brain-state segmentation, "
                 f"{res['n_units']} units, {len(units)} head-direction cells")

    # wake: raster over the most head-mobile minute, with the measured direction below
    t0, t1 = _best_wake_window(hd, ep["wake_tracked"])
    ax = fig.add_subplot(gs[1, 0])
    _hd_raster(ax, spikes, units, pref, (t0, t1), t0)
    ax.set_ylabel("HD cell (sorted by\npreferred direction)")
    ax.set_title("Wake: the bump follows the head")
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax)
    h = hd.get(t0, t1)
    ax2.plot(h.index.values - t0, np.degrees(h.values), ".", ms=1.5, color="k")
    ax2.set_ylabel("measured head\ndirection (deg)"); ax2.set_xlabel("time (s)")
    ax2.set_ylim(0, 360); ax2.set_yticks([0, 180, 360])

    # sleep: raster with the decoded internal heading on top
    for row, (st, dur) in enumerate([("REM", 60.0), ("nREM", 25.0)]):
        r = res[st]
        t, dec = r["t"], smooth_decode(r["counts"], res["tc_hd"], res["bin"])
        eps = _episodes(t, res["bin"], min_bins=int(dur / res["bin"]))
        i0, i1 = max(eps, key=lambda e: e[1] - e[0])
        i1 = min(i1, i0 + int(dur / res["bin"]))
        ax = fig.add_subplot(gs[row + 1, 1])
        _hd_raster(ax, spikes, units, pref, (t[i0], t[i1 - 1]), t[i0])
        ax.plot(t[i0:i1] - t[i0], dec[i0:i1] / TWO_PI * len(units), ".", ms=3, color="k",
                label="internal heading decoded from these spikes")
        ax.set_title(f"{STATE_LABEL[st]}: same ensemble, animal asleep and still",
                     color=STATE_COLORS[st])
        ax.set_ylabel("HD cell (sorted)")
        ax.legend(fontsize=7, markerscale=2.5, loc="lower left",
                  bbox_to_anchor=(0, 1.0), frameon=False)
        if row == 1:
            ax.set_xlabel("time (s)")
    fig.savefig(fname)
    plt.close(fig)


# --------------------------------------------------------------- figure 2 ---
def fig_tuning(res, fname, n_show=24):
    tc = res["tc_hd"]; pref = res["pref"]; df = res["screen"]
    order = np.argsort(pref)
    n = min(n_show, tc.shape[1])
    ncol = 6
    nrow = int(np.ceil(n / ncol)) + 1
    fig = plt.figure(figsize=(13, 2.4 * nrow))
    gs = GridSpec(nrow, ncol, hspace=0.75, wspace=0.45)
    a = np.append(ANG, ANG[0])
    for k in range(n):
        i = order[int(round(k * (tc.shape[1] - 1) / max(n - 1, 1)))]
        ax = fig.add_subplot(gs[k // ncol, k % ncol], projection="polar")
        v = np.append(tc[:, i], tc[0, i])
        ax.plot(a, v, color=plt.get_cmap("hsv")(pref[i] / TWO_PI), lw=1.3)
        ax.fill(a, v, color=plt.get_cmap("hsv")(pref[i] / TWO_PI), alpha=0.25)
        ax.set_xticks(np.arange(0, TWO_PI, np.pi / 2))
        ax.set_xticklabels(["0", "", "180", ""], fontsize=6)
        ax.set_yticklabels([])
        ax.set_title(f"unit {res['hd_units'][i]}  ({v.max():.0f} Hz)", fontsize=7, pad=10)

    r = nrow - 1
    ax = fig.add_subplot(gs[r, 0:2], projection="polar")
    ax.hist(pref, bins=18, range=(0, TWO_PI), color="#444", alpha=0.8)
    ax.set_title("preferred directions\ntile the circle", fontsize=9, pad=26)
    ax.set_yticklabels([])

    ax = fig.add_subplot(gs[r, 2:4])
    ax.scatter(df["mvl_null99"], df["mvl"], s=16,
               c=["#c0392b" if h else "#bbb" for h in df["is_hd"]])
    lim = [0, max(np.nanmax(df["mvl"]), np.nanmax(df["mvl_null99"])) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("99th pct of circular-shift null"); ax.set_ylabel("mean vector length")
    ax.set_title("HD selectivity vs shuffle null", fontsize=9)

    ax = fig.add_subplot(gs[r, 4:6])
    ax.scatter(df["stability"], df["mvl"], s=16,
               c=["#c0392b" if h else "#bbb" for h in df["is_hd"]])
    ax.axvline(0.5, ls="--", lw=0.8, c="k"); ax.axhline(0.25, ls="--", lw=0.8, c="k")
    ax.set_xlabel("split-half tuning-curve correlation"); ax.set_ylabel("mean vector length")
    ax.set_title("tuning stability within wake", fontsize=9)
    fig.suptitle(f"{res['sid']}: head-direction tuning during wake "
                 f"({len(pref)} HD cells of {res['n_units']} units)", y=1.0)
    fig.savefig(fname)
    plt.close(fig)


# --------------------------------------------------------------- figure 3 ---
def fig_correlations(res, fname):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    order = res["order"]
    for j, st in enumerate(["wake", "REM", "nREM"]):
        r = res[st]
        c = r["cmat"][np.ix_(order, order)].copy()
        np.fill_diagonal(c, np.nan)
        v = np.nanpercentile(np.abs(c), 98)
        ax = axes[0, j]
        im = ax.imshow(c, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"{STATE_LABEL[st]}\npairwise correlations", color=STATE_COLORS[st])
        ax.set_xlabel("HD cell (sorted by preferred direction)")
        if j == 0:
            ax.set_ylabel("HD cell (sorted)")
        plt.colorbar(im, ax=ax, fraction=0.046, label="r")

        ax = axes[1, j]
        d, rr = r["corr_d"], r["corr_r"]
        ax.scatter(np.degrees(d), rr, s=8, alpha=0.35, color=STATE_COLORS[st])
        b = np.linspace(0, np.pi, 13)
        idx = np.digitize(d, b) - 1
        m = np.array([np.mean(rr[idx == k]) if (idx == k).sum() else np.nan
                      for k in range(len(b) - 1)])
        bc = np.degrees((b[:-1] + b[1:]) / 2)
        ax.plot(bc, m, "o-", color="k", ms=4, lw=1.5, label="binned mean")
        xx = np.linspace(0, np.pi, 200)
        ax.plot(np.degrees(xx), r["cos_a"] + r["cos_b"] * np.cos(xx), "--", color="k", lw=1.2,
                label=f"cosine fit, $R^2$={r['cos_r2']:.2f}")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel("|Δ preferred direction| (deg, from wake)")
        if j == 0:
            ax.set_ylabel("correlation of binned firing (100 ms)")
        ax.set_title(f"shuffle null $R^2$ = {r['null_cos_r2'].mean():.3f}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xticks([0, 45, 90, 135, 180])
    fig.suptitle(f"{res['sid']}: the wake correlation structure of the HD ensemble "
                 "is preserved in both sleep states", y=0.98)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# --------------------------------------------------------------- figure 4 ---
def fig_manifold(res, fname):
    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.35)
    for j, st in enumerate(["wake", "REM", "nREM"]):
        r = res[st]
        emb = r["emb"]
        col = (r["true_hd_emb"] if st == "wake"
               else smooth_decode(r["counts"], res["tc_hd"], res["bin"])[r["emb_keep"]][r["emb_sel"]])
        ax = fig.add_subplot(gs[0, j])
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=col, cmap="hsv", s=5, vmin=0, vmax=TWO_PI)
        ax.set_title(f"{STATE_LABEL[st]}\nhollowness = {r['hollowness']:.2f} "
                     f"(null {r['null_hollowness'].mean():.2f})", color=STATE_COLORS[st])
        ax.set_xlabel("Isomap 1"); ax.set_ylabel("Isomap 2" if j == 0 else "")
        ax.set_aspect("equal")
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, ticks=[0, np.pi, TWO_PI])
        cb.ax.set_yticklabels(["0", "180", "360"])
        cb.set_label("measured HD (deg)" if st == "wake"
                     else "decoded internal HD (deg, 300 ms)", fontsize=8)

    r = res["nREM"]
    ax = fig.add_subplot(gs[1, 0])
    _, mxs = manifold_input(shift_shuffle(r["counts"], np.random.default_rng(1)))
    _, semb = ring_embedding(mxs, max_points=2500, seed=1)
    ax.scatter(semb[:, 0], semb[:, 1], s=5, color="#999")
    ax.set_aspect("equal"); ax.set_xlabel("Isomap 1"); ax.set_ylabel("Isomap 2")
    ax.set_title(f"non-REM, time-shifted control\nhollowness = "
                 f"{ring_stats(semb)['hollowness']:.2f}")

    ax = fig.add_subplot(gs[1, 1])
    if "ctrl" in r:
        ax.scatter(r["ctrl"]["emb"][:, 0], r["ctrl"]["emb"][:, 1], s=5, color="#999")
        ax.set_title(f"non-REM, non-HD units (n={r['ctrl']['n']})\n"
                     f"hollowness = {r['ctrl']['hollowness']:.2f}")
    ax.set_aspect("equal"); ax.set_xlabel("Isomap 1"); ax.set_ylabel("Isomap 2")

    ax = fig.add_subplot(gs[1, 2])
    for st in ["wake", "REM", "nREM"]:
        e = res[st]["emb"]
        rad = np.hypot(*(e - taubin_center(e)).T)
        ax.hist(rad / rad.mean(), bins=60, range=(0, 2.2), histtype="step", density=True,
                color=STATE_COLORS[st], label=STATE_LABEL[st], lw=1.4)
    rad = np.hypot(*(semb - taubin_center(semb)).T)
    ax.hist(rad / rad.mean(), bins=60, range=(0, 2.2), histtype="step", density=True,
            color="#999", label="non-REM, time-shifted", lw=1.4, ls="--")
    ax.set_xlabel("radius in embedding / mean radius")
    ax.set_ylabel("density")
    ax.set_title("a ring is hollow, a blob is not")
    ax.legend(fontsize=7)
    fig.suptitle(f"{res['sid']}: population states lie on a closed one-dimensional ring "
                 "in every brain state", y=0.98)
    fig.savefig(fname)
    plt.close(fig)


# --------------------------------------------------------------- figure 5 ---
def fig_dimensionality(res, fname):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    ax = axes[0]
    for st in ["wake", "REM", "nREM"]:
        sp = res[st]["pca_spectrum"]
        ax.plot(np.arange(1, len(sp) + 1), sp * 100, "o-", ms=4,
                color=STATE_COLORS[st], label=STATE_LABEL[st])
    ctrl_spec = res["nREM"].get("ctrl", {}).get("pca_spectrum")
    if ctrl_spec is not None:
        ax.plot(np.arange(1, len(ctrl_spec) + 1), ctrl_spec * 100, "o--", ms=4,
                color="#999", label="non-HD units (nREM)")
    ax.set_xlabel("principal component"); ax.set_ylabel("variance explained (%)")
    ax.set_title("PCA spectrum (200 ms smoothed)"); ax.legend(fontsize=7)

    ax = axes[1]
    vals = [res[st]["pca_spectrum"][1] / res[st]["pca_spectrum"][0]
            for st in ["wake", "REM", "nREM"]]
    ax.bar(np.arange(3), vals, color=[STATE_COLORS[s] for s in ["wake", "REM", "nREM"]])
    if ctrl_spec is not None:
        ax.bar(3, ctrl_spec[1] / ctrl_spec[0], color="#999")
    ax.axhline(1.0, ls="--", lw=0.8, color="k")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(["wake", "REM", "nREM", "non-HD\n(nREM)"])
    ax.set_ylabel("PC2 variance / PC1 variance")
    ax.set_title("a ring spans a plane, so its two\nleading components are equal")

    ax = axes[2]
    for i, st in enumerate(["wake", "REM", "nREM"]):
        r = res[st]
        ax.bar(i - 0.18, r["hollowness"], 0.34, color=STATE_COLORS[st])
        ax.bar(i + 0.18, r["null_hollowness"].mean(), 0.34, color="#ccc",
               yerr=r["null_hollowness"].std(), error_kw=dict(lw=0.8))
    ax.set_xticks(range(3)); ax.set_xticklabels(["wake", "REM", "nREM"])
    ax.set_ylabel("hollowness  (mean r / SD r)")
    ax.set_title("ring topology vs time-shift null\n(grey = null)")

    ax = axes[3]
    for i, st in enumerate(["wake", "REM", "nREM"]):
        r = res[st]
        ax.bar(i - 0.18, r["ev1d"], 0.34, color=STATE_COLORS[st])
        ax.bar(i + 0.18, r["null_ev1d"].mean(), 0.34, color="#ccc",
               yerr=r["null_ev1d"].std(), error_kw=dict(lw=0.8))
    ax.set_xticks(range(3)); ax.set_xticklabels(["wake", "REM", "nREM"])
    ax.set_ylabel("cross-validated mean $R^2$")
    ax.set_title("held-out neurons explained by the\nangle decoded from the others")
    fig.suptitle(f"{res['sid']}: the variance of the HD ensemble concentrates in an "
                 "equal-variance plane that holds the ring", y=1.04)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# --------------------------------------------------------------- figure 6 ---
def fig_coherence(res, fname, seed=0):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    ax = axes[0]
    for i, st in enumerate(["wake", "REM", "nREM"]):
        r = res[st]
        ax.bar(i - 0.18, r["split_coh"], 0.34, color=STATE_COLORS[st])
        ax.bar(i + 0.18, r["null_coh"].mean(), 0.34, color="#ccc",
               yerr=r["null_coh"].std(), error_kw=dict(lw=0.8))
    ax.set_xticks(range(3)); ax.set_xticklabels(["wake", "REM", "nREM"])
    ax.set_ylabel("resultant length of the angular difference")
    ax.set_ylim(0, 1)
    ax.set_title("split-ensemble decoder agreement\n(grey = time-shift null)")

    for j, st in enumerate(["REM", "nREM"]):
        r = res[st]
        da, db = split_half_decode(r["counts"], res["tc_hd"], res["bin"],
                                   np.random.default_rng(seed))
        ax = axes[1 + j]
        ax.hist2d(np.degrees(da), np.degrees(db), bins=48, range=[[0, 360], [0, 360]],
                  cmap="magma", norm="log")
        ax.set_xlabel("heading decoded from ensemble half A (deg)")
        ax.set_ylabel("half B (deg)")
        ax.set_title(f"{STATE_LABEL[st]}\nagreement = {r['split_coh']:.2f}",
                     color=STATE_COLORS[st])
        ax.set_xticks([0, 180, 360]); ax.set_yticks([0, 180, 360])

    ax = axes[3]
    k = int(round(1.0 / res["bin"]))
    for st in ["wake", "REM", "nREM"]:
        r = res[st]
        d = smooth_decode(r["counts"], res["tc_hd"], res["bin"])
        t = r["t"]
        ok = (t[k:] - t[:-k]) < 1.5
        v = np.degrees(np.abs(circ_diff(d[k:], d[:-k]))[ok])
        ax.hist(v, bins=np.linspace(0, 180, 60), histtype="step", density=True,
                color=STATE_COLORS[st], lw=1.4,
                label=f"{STATE_LABEL[st]} (median {np.median(v):.0f}°/s)")
    ax.set_xlim(0, 180)
    ax.set_xlabel("|angular displacement| of the internal heading over 1 s (deg)")
    ax.set_ylabel("density"); ax.legend(fontsize=7)
    ax.set_title("the internal heading moves smoothly,\nfaster in non-REM than in REM")
    fig.suptitle(f"{res['sid']}: the ring coordinate is internally coherent during sleep",
                 y=1.04)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# --------------------------------------------------------------- figure 7 ---
def fig_decode_validation(res, fname):
    r = res["wake"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axes[0]
    t, dec, true = r["t"], r["decoded"], r["true_hd"]
    # show the 90 s stretch in which the animal actually turned its head the most
    w = 900
    best, best_disp = (0, w), -1.0
    for a, b in _episodes(t, res["bin"], min_bins=w):
        for i0 in range(a, b - w, w):
            h = true[i0:i0 + w]; h = h[np.isfinite(h)]
            if len(h) < w // 2:
                continue
            disp = 1 - np.abs(np.exp(1j * h).mean())
            if disp > best_disp:
                best_disp, best = disp, (i0, i0 + w)
    i0, i1 = best
    ax.plot(t[i0:i1] - t[i0], np.degrees(true[i0:i1]), ".", ms=2, color="k", label="measured HD")
    ax.plot(t[i0:i1] - t[i0], np.degrees(dec[i0:i1]), ".", ms=2, color="#1b6ca8",
            label="decoded from HD cells")
    ax.set_xlabel("time (s)"); ax.set_ylabel("head direction (deg)")
    ax.set_yticks([0, 180, 360]); ax.legend(fontsize=7, markerscale=4)
    ax.set_title("wake: decoder validation")

    ax = axes[1]
    ax.hist(np.degrees(r["decode_err"]), bins=60, color="#1b6ca8")
    ax.axvline(np.degrees(np.median(r["decode_err"])), color="k", ls="--")
    ax.set_xlabel("|decoding error| (deg)"); ax.set_ylabel("bins")
    ax.set_title(f"median error {np.degrees(np.median(r['decode_err'])):.0f}°, "
                 f"circ. r = {r['decode_circ_corr']:.2f}")

    ax = axes[2]
    for i, st in enumerate(["wake", "REM", "nREM"]):
        ax.bar(i - 0.18, res[st]["map_conc"], 0.34, color=STATE_COLORS[st])
        ax.bar(i + 0.18, res[st]["null_map_conc"], 0.34, color="#ccc")
    ax.set_ylabel("resultant length of the residual")
    ax.set_xticks(range(3)); ax.set_xticklabels(["wake", "REM", "nREM"])
    ax.set_ylim(0, 1)
    ax.set_title("how sharply the ring angle fixes the\ndecoded heading (grey = shuffle)")
    fig.suptitle(f"{res['sid']}: the ring coordinate recovered without any behavioural "
                 "reference matches the wake heading code", y=1.04)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# --------------------------------------------------------------- figure 8 ---
def fig_multisession(summary, fname):
    import pandas as pd
    df = pd.DataFrame(summary)
    metrics = [("cos_r2", "cosine fit $R^2$\n(corr. vs Δpreferred dir.)"),
               ("hollowness", "ring hollowness\n(mean r / SD r)"),
               ("ev1d", "held-out neurons explained by\none angular coordinate ($R^2$)"),
               ("split_coh", "split-ensemble decoder\nagreement (resultant)"),
               ("map_conc", "ring angle fixes the\ndecoded heading (resultant)")]
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.8))
    states = ["wake", "REM", "nREM"]
    for ax, (m, lab) in zip(axes, metrics):
        for i, st in enumerate(states):
            v = df[f"{m}_{st}"].values
            nv = df[f"null_{m}_{st}"].values
            ax.plot(np.full_like(v, i - 0.16) + np.random.uniform(-.04, .04, len(v)), v,
                    "o", ms=5, color=STATE_COLORS[st], alpha=0.85)
            ax.plot(np.full_like(nv, i + 0.16) + np.random.uniform(-.04, .04, len(nv)), nv,
                    "o", ms=5, color="#bbb")
            ax.hlines(np.mean(v), i - 0.30, i - 0.02, color=STATE_COLORS[st], lw=2)
            ax.hlines(np.mean(nv), i + 0.02, i + 0.30, color="#888", lw=2)
        ax.set_xticks(range(3)); ax.set_xticklabels(states)
        ax.set_title(lab, fontsize=9)
    axes[0].set_ylabel("value  (coloured = data, grey = null)")
    fig.suptitle(f"Across {len(df)} sessions from {df.subject.nunique()} mice: "
                 "the one-dimensional ring is present in wake, REM and non-REM", y=1.03)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return df


# --------------------------------------------------------------- figure 9 ---
def _episodes(t, bin_size, min_bins=300):
    b = np.concatenate([[0], np.where(np.diff(t) > 1.5 * bin_size)[0] + 1, [len(t)]])
    return [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] - b[i] >= min_bins]


def fig_sleep_vs_behavior(res, s, fname, nearest_hd=None):
    """The decoded internal heading travels the ring while the animal's head does not."""
    hd = s["hd"]
    bs = res["bin"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    meas = {st: nearest_hd(hd, res[st]["t"], bs) for st in ["wake", "REM", "nREM"]}
    stats = {}

    for row, st in enumerate(["REM", "nREM"]):
        r = res[st]
        t, dec, m = r["t"], r["decoded"], meas[st]
        eps = _episodes(t, bs)
        i0, i1 = max(eps, key=lambda e: e[1] - e[0])
        i1 = min(i1, i0 + 1200)
        ax = axes[row, 0]
        ax.plot(t[i0:i1] - t[i0], np.degrees(m[i0:i1]), ".", ms=2, color="#999",
                label="measured head direction")
        ax.plot(t[i0:i1] - t[i0], np.degrees(dec[i0:i1]), ".", ms=2,
                color=STATE_COLORS[st], label="decoded internal heading")
        ax.set_ylim(0, 360); ax.set_yticks([0, 180, 360])
        ax.set_ylabel("direction (deg)")
        ax.set_title(f"{STATE_LABEL[st]}: the head is still, the internal heading is not",
                     color=STATE_COLORS[st])
        ax.legend(fontsize=7, markerscale=4, loc="upper right")
        if row == 1:
            ax.set_xlabel("time within sleep episode (s)")

    # per-episode angular concentration of the two signals
    ax = axes[0, 1]
    for i, st in enumerate(["wake", "REM", "nREM"]):
        r = res[st]
        Rm, Rd = [], []
        for a, b in _episodes(r["t"], bs):
            mm = meas[st][a:b]; mm = mm[np.isfinite(mm)]
            if len(mm) < 100:
                continue
            Rm.append(np.abs(np.exp(1j * mm).mean()))
            Rd.append(np.abs(np.exp(1j * r["decoded"][a:b]).mean()))
        stats[f"R_head_{st}"] = float(np.median(Rm))
        stats[f"R_decoded_{st}"] = float(np.median(Rd))
        stats[f"n_episodes_{st}"] = len(Rm)
        jit = np.random.uniform(-0.06, 0.06, len(Rm))
        ax.plot(i - 0.18 + jit, Rm, "o", ms=4, color="#999", alpha=0.6)
        ax.plot(i + 0.18 + jit, Rd, "o", ms=4, color=STATE_COLORS[st], alpha=0.6)
        ax.hlines(np.median(Rm), i - 0.33, i - 0.03, color="k", lw=2)
        ax.hlines(np.median(Rd), i + 0.03, i + 0.33, color="k", lw=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["wake", "REM", "nREM"])
    ax.set_ylabel("concentration within episode (resultant length)")
    ax.set_ylim(0, 1.05)
    ax.set_title("grey: measured head direction   colour: decoded internal heading\n"
                 "in sleep the head is fixed while the internal heading sweeps the ring",
                 fontsize=9)

    # median angular speed over a 1 s window, head vs internal
    ax = axes[1, 1]
    w = 1.0; k = int(round(w / bs))
    for i, st in enumerate(["wake", "REM", "nREM"]):
        r = res[st]
        t, dec, m = r["t"], r["decoded"], meas[st]
        ok = ((t[k:] - t[:-k]) < 1.5 * w) & np.isfinite(m[k:]) & np.isfinite(m[:-k])
        vh = np.degrees(np.median(np.abs(circ_diff(m[k:], m[:-k]))[ok] / w))
        vd = np.degrees(np.median(np.abs(circ_diff(dec[k:], dec[:-k]))[ok] / w))
        stats[f"head_speed_{st}"] = float(vh)
        stats[f"internal_speed_{st}"] = float(vd)
        ax.bar(i - 0.18, vh, 0.34, color="#999")
        ax.bar(i + 0.18, vd, 0.34, color=STATE_COLORS[st])
    ax.set_xticks(range(3)); ax.set_xticklabels(["wake", "REM", "nREM"])
    ax.set_ylabel("median |angular speed| over 1 s (deg/s)")
    ax.set_title("grey: real head movement   colour: internal heading", fontsize=9)

    fig.suptitle(f"{res['sid']}: during sleep the ring state is driven internally, "
                 "not by the animal's head", y=1.0)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return stats
