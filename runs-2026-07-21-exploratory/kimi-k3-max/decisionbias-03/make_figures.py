"""Aggregate results across sessions and produce all figures."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import norm, combine_pvalues

from ibl_common import ASSET_IDS, load_session

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

KEYS = list(ASSET_IDS.keys())
C_LEFT, C_RIGHT = "#2166ac", "#b2182b"
BLOCK_COLORS = {0.2: "#b2182b", 0.5: "#7b7b7b", 0.8: "#2166ac"}


def load_results():
    return {k: np.load(f"results_{k}.npz", allow_pickle=True) for k in KEYS}


# ---------------------------------------------------------------- fig 1: task + behavior
def fig1(res):
    fig = plt.figure(figsize=(11, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.32)

    # A: trial timeline schematic
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    ax.set_xlim(-1.15, 1.75)
    ax.set_ylim(-1.35, 1.25)
    ax.text(-1.15, 1.18, "IBL contrast-detection task", fontsize=10, weight="bold",
            va="top")
    ax.text(-1.15, 1.02, "grating appears left or right (or 0% contrast);\n"
                         "mouse turns wheel to report side;\n"
                         "block prior P(left) in {0.2, 0.5, 0.8}",
            fontsize=9, va="top")
    # timeline
    y_tl = -0.2
    ax.annotate("", xy=(1.7, y_tl), xytext=(-1.1, y_tl),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    ax.add_patch(Rectangle((-0.4, y_tl - 0.06), 0.4, 0.12, facecolor="#fdae61",
                           edgecolor="none", alpha=0.85, zorder=3))
    ax.plot([-0.2, -0.2], [y_tl + 0.12, y_tl + 0.44], color="#a6611a", lw=0.8)
    ax.text(-0.2, y_tl + 0.48, "pre-stimulus\ndecoding window", ha="center",
            va="bottom", fontsize=8.5, color="#a6611a")
    # staggered event labels to avoid overlap
    for x, xt, lab, dy in [(-0.4, -0.4, "quiescence\n(wheel held still)", -0.14),
                           (0.0, 0.0, "stimulus onset\n+ go cue", -0.58),
                           (1.15, 1.15, "feedback", -0.14),
                           (0.15, 0.15, "movement\n(median 150 ms)", 0.14)]:
        ax.plot([xt, xt], [y_tl - 0.06, y_tl + 0.06], color="k", lw=1.2)
        va = "top" if dy < 0 else "bottom"
        ax.text(x, y_tl + dy, lab, ha="center", va=va, fontsize=8.5)
    ax.set_title("A  Task and decoding window", loc="left", fontsize=10, weight="bold")

    # B: psychometric curves pooled over sessions
    ax = fig.add_subplot(gs[1])
    for p in [0.2, 0.5, 0.8]:
        # pool trials across sessions: psych_vals rows are (p_left, n)
        sc_all, pl_all, n_all = [], [], []
        for k in KEYS:
            pk, pv = res[k]["psych_keys"], res[k]["psych_vals"]
            m = pk[:, 0] == p
            sc_all += list(pk[m, 1])
            pl_all += list(pv[m, 0] * pv[m, 1])
            n_all += list(pv[m, 1])
        d = pd.DataFrame({"sc": sc_all, "plw": pl_all, "n": n_all})
        g = d.groupby("sc").apply(lambda r: r.plw.sum() / r.n.sum(), include_groups=False)
        ax.plot(g.index, g.values, "o-", color=BLOCK_COLORS[p], ms=4, lw=1.5,
                label=f"P(left) = {p}")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("signed contrast (left - right)")
    ax.set_ylabel("P(choose left)")
    ax.legend(fontsize=8, frameon=False, title="block prior", title_fontsize=8)
    ax.set_title("B  Behavioral bias from block prior", loc="left", fontsize=10, weight="bold")

    # C: block prior across trials (example session)
    ax = fig.add_subplot(gs[2])
    r = res[KEYS[0]]
    # reconstruct probabilityLeft per valid trial from block run ids is lossy; use y/blocks only for illustration of block structure
    blocks = r["blocks"]
    ax.plot(np.arange(len(blocks)), blocks, lw=0.8, color="0.3")
    ax.set_xlabel(f"valid trial number (session {KEYS[0]})")
    ax.set_ylabel("block index (prior switches)")
    ax.set_title("C  Block structure within a session", loc="left", fontsize=10, weight="bold")

    fig.savefig("fig1_task_behavior.png", bbox_inches="tight")
    plt.close(fig)
    print("fig1 saved")


# ---------------------------------------------------------------- fig 2: example neural data
def peth(spikes, events, window=(-1.0, 1.0), bin_size=0.01, sigma=0.05):
    """Smoothed peri-event firing rate. Returns (t, rate).

    Bins are computed over an extended window (+-4 sigma) and cropped after
    convolution so the edges are not attenuated.
    """
    pad = 4 * sigma
    bins = np.arange(window[0] - pad, window[1] + pad + bin_size, bin_size)
    counts = np.zeros(len(bins) - 1)
    for st in spikes:
        for e in events:
            rel = st[(st >= e + window[0] - pad) & (st < e + window[1] + pad)] - e
            counts += np.histogram(rel, bins=bins)[0]
    rate = counts / (len(events) * bin_size)
    kw = int(4 * sigma / bin_size)
    x = np.arange(-kw, kw + 1) * bin_size
    kern = np.exp(-0.5 * (x / sigma) ** 2)
    kern /= kern.sum()
    rate = np.convolve(rate, kern, mode="same")
    t = bins[:-1] + bin_size / 2
    m = (t >= window[0]) & (t <= window[1])
    return t[m], rate[m]


def fig2(res, example_key):
    trials, spike_list, units = load_session(ASSET_IDS[example_key])
    good = units.ks2_label.values == "good"
    good_idx = np.where(good)[0]
    r = res[example_key]
    unit_auc = r["unit_auc"]
    regions = r["regions"].astype(str)

    choice = trials.choice.values
    stim_on = trials.stimOn_times.values
    first_mov = trials.firstMovement_times.values
    valid = (choice != 0) & ~np.isnan(stim_on) & (np.isnan(first_mov) | (first_mov >= stim_on))
    tr = trials[valid]
    y = (tr.choice.values == 1).astype(int)
    stim = tr.stimOn_times.values

    fig = plt.figure(figsize=(11, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.32)

    # A: population raster over 40 s
    ax = fig.add_subplot(gs[0])
    t_mid = stim[len(stim) // 2]
    t0, t1 = t_mid - 20, t_mid + 20
    show_units = np.arange(0, len(good_idx), max(1, len(good_idx) // 120))
    order = np.argsort(regions[show_units])
    for row, ui in enumerate(show_units[order]):
        st = spike_list[good_idx[ui]]
        seg = st[(st >= t0) & (st < t1)]
        ax.plot(seg, np.full_like(seg, row), "|", color="0.25", ms=1.5, mew=0.4)
    ev = stim[(stim >= t0) & (stim < t1)]
    ey = y[(stim >= t0) & (stim < t1)]
    for e, c in zip(ev, ey):
        ax.axvline(e, color=C_LEFT if c == 1 else C_RIGHT, alpha=0.5, lw=1.0)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("time in session (s)")
    ax.set_ylabel("unit (sorted by region)")
    ax.set_title(f"A  Population spiking with stimulus onsets colored by upcoming choice "
                 f"(session {example_key})", loc="left", fontsize=10, weight="bold")

    # B: PETHs of top choice-selective units (header strip + 4 panels)
    bot = gs[1].subgridspec(2, 1, height_ratios=[0.07, 1.0], hspace=0.0)
    axh = fig.add_subplot(bot[0])
    axh.axis("off")
    axh.set_title("B  Example pre-stimulus choice-selective units", loc="left",
                  fontsize=10, weight="bold")
    sub = bot[1].subgridspec(1, 4, wspace=0.4)
    top = np.argsort(-np.abs(unit_auc - 0.5))[:4]
    for j, ui in enumerate(top):
        ax = fig.add_subplot(sub[j])
        st = [spike_list[good_idx[ui]]]
        t, rL = peth(st, stim[y == 1])
        _, rR = peth(st, stim[y == 0])
        ax.plot(t, rL, color=C_LEFT, lw=1.4, label="left choice")
        ax.plot(t, rR, color=C_RIGHT, lw=1.4, label="right choice")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.axvspan(-0.4, 0, color="#fdae61", alpha=0.3)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("time from stimOn (s)", fontsize=9)
        if j == 0:
            ax.set_ylabel("firing rate (Hz)")
        ax.set_title(f"unit {ui} ({regions[ui]}), pre-stim AUC={unit_auc[ui]:.2f}",
                     fontsize=8)
        if j == 3:
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.savefig("fig2_example_neural.png", bbox_inches="tight")
    plt.close(fig)
    print("fig2 saved")


# ---------------------------------------------------------------- fig 3: time-resolved decoding
def fig3(res):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    all_null = np.concatenate([res[k]["null_global"] for k in KEYS])
    lo, hi = np.percentile(all_null, [2.5, 97.5])
    ax.axhspan(lo, hi, color="0.85", zorder=0, label="shuffle null 95%")
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    curves = []
    for k in KEYS:
        c = res[k]["centers"]
        b = res[k]["tr_decode"][:, 0]
        ax.plot(c, b, color="0.6", lw=1.0, alpha=0.8)
        curves.append(b)
    curves = np.array(curves)
    ax.plot(c, curves.mean(axis=0), color="#b2182b", lw=2.4, label="mean across sessions")
    ax.axvline(0, color="k", lw=1.0, ls="--")
    ax.axvspan(-0.4, 0, color="#fdae61", alpha=0.35)
    ax.text(-0.2, 0.66, "pre-stimulus\nwindow", ha="center",
            fontsize=8.5, color="#a6611a")
    ax.set_xlabel("time from stimulus onset (s)  [250 ms window center]")
    ax.set_ylabel("balanced accuracy (choice decoding)")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_title("Upcoming choice is decodable before stimulus onset", fontsize=11)
    fig.savefig("fig3_timecourse.png", bbox_inches="tight")
    plt.close(fig)
    print("fig3 saved")


# ---------------------------------------------------------------- fig 4: pre-stim decoding vs nulls
def fig4(res):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    baccs = [res[k]["bacc"] for k in KEYS]
    ps_g = [res[k]["p_global"] for k in KEYS]
    ps_b = [res[k]["p_block"] for k in KEYS]

    for ax, null_key, ps, title in [
        (axes[0], "null_global", ps_g, "vs global trial-shuffle null"),
        (axes[1], "null_block", ps_b, "vs within-block shuffle null"),
    ]:
        for i, k in enumerate(KEYS):
            null = res[k][null_key]
            parts = ax.violinplot([null], positions=[i], widths=0.62, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("0.8")
                pc.set_edgecolor("none")
            ax.scatter([i], [np.mean(null)], color="0.45", s=12, zorder=3)
            ax.scatter([i], [res[k]["bacc"]], color="#b2182b", s=46, zorder=4,
                       marker="D" if ps[i] < 0.05 else "o")
            ax.text(i, res[k]["bacc"] + 0.012, f"p={ps[i]:.3g}", ha="center", fontsize=8)
        ax.axhline(0.5, color="k", lw=0.7, ls=":")
        ax.set_xticks(range(len(KEYS)))
        ax.set_xticklabels(KEYS)
        ax.set_xlabel("session")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("balanced accuracy (pre-stimulus, -0.4 to 0 s)")
    # Stouffer combined p across sessions (global null)
    z = sum(norm.isf(np.clip(p, 1e-16, 1)) for p in ps_g) / np.sqrt(len(ps_g))
    p_comb = norm.sf(z)
    fig.suptitle(f"Pre-stimulus choice decoding, 4 sessions "
                 f"(diamond = p<0.05; Stouffer combined p = {p_comb:.2g})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("fig4_prestim_decoding.png", bbox_inches="tight")
    plt.close(fig)
    print("fig4 saved")
    return p_comb


# ---------------------------------------------------------------- fig 5: single units
def fig5(res):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    aucs = np.concatenate([res[k]["unit_auc"] for k in KEYS])
    thr = np.percentile(np.concatenate([res[k]["null_max"] for k in KEYS]), 95)
    ax.hist(aucs, bins=60, color="0.55", edgecolor="none")
    for s in [-1, 1]:
        ax.axvline(0.5 + s * thr, color="#b2182b", lw=1.2, ls="--")
    frac = np.mean(np.abs(aucs - 0.5) > thr)
    frac_p05 = [res[k]["frac_p05"].item() for k in KEYS]
    null_frac = np.concatenate([res[k]["null_frac_p05"] for k in KEYS])
    nlo, nhi = np.percentile(null_frac, [2.5, 97.5])
    ax.set_xlabel("pre-stimulus choice AUC (single units, pooled)")
    ax.set_ylabel("number of units")
    ax.set_title(
        f"A  {frac*100:.1f}% of units exceed family-wise threshold (|AUC-0.5|>{thr:.3f})\n"
        f"uncorrected p<0.05: {100*np.mean(frac_p05):.1f}% of units "
        f"(null 95%: {100*nlo:.1f}-{100*nhi:.1f}%)", fontsize=9.5)

    ax = axes[1]
    rows = []
    for k in KEYS:
        reg = res[k]["regions"].astype(str)
        up = res[k]["unit_p"]
        for region in np.unique(reg):
            if region == "unknown":
                continue
            m = reg == region
            if m.sum() >= 15:
                rows.append((region, m.sum(), np.mean(up[m] < 0.05)))
    d = pd.DataFrame(rows, columns=["region", "n", "frac"])
    g = d.groupby("region").apply(lambda r: np.average(r.frac, weights=r.n),
                                  include_groups=False).sort_values(ascending=False)
    g = g.head(12)
    ax.barh(range(len(g)), g.values, color="#4d7fb8")
    ax.axvline(0.05, color="k", lw=0.8, ls=":", label="chance (5%)")
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{r}" for r in g.index], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of units choice-selective pre-stimulus (p<0.05)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("B  Selective units by brain region (regions with >=15 units)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("fig5_single_units.png", bbox_inches="tight")
    plt.close(fig)
    print("fig5 saved")


# ---------------------------------------------------------------- fig 6: bias controls
def fig6(res):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, bkey, nkey, nully, pkey, title in [
        (axes[0], "bacc0", "n_zero", "null0", "p0",
         "0%-contrast trials only (pure bias)"),
        (axes[1], "baccu", "n_unb", "nullu", "pu",
         "unbiased block (P(left)=0.5) only"),
    ]:
        for i, k in enumerate(KEYS):
            null = res[k][nully]
            b = res[k][bkey].item()
            p = res[k][pkey].item()
            n = res[k][nkey].item()
            if np.isnan(b):
                continue
            parts = ax.violinplot([null], positions=[i], widths=0.62, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("0.8")
                pc.set_edgecolor("none")
            ax.scatter([i], [b], color="#b2182b", s=46, zorder=4,
                       marker="D" if p < 0.05 else "o")
            ax.text(i, b + 0.012, f"p={p:.3g}\nn={n}", ha="center", fontsize=8)
        ax.axhline(0.5, color="k", lw=0.7, ls=":")
        ax.set_xticks(range(len(KEYS)))
        ax.set_xticklabels(KEYS)
        ax.set_xlabel("session")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("balanced accuracy (pre-stimulus)")
    fig.suptitle("Pre-stimulus choice decoding in bias-dominated trial subsets "
                 "(diamond = p<0.05)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("fig6_bias_controls.png", bbox_inches="tight")
    plt.close(fig)
    print("fig6 saved")


if __name__ == "__main__":
    res = load_results()
    # example session: strongest pre-stim decoding among sessions with region labels
    annotated = [k for k in KEYS if (res[k]["regions"].astype(str) != "unknown").any()]
    example_key = max(annotated, key=lambda k: res[k]["bacc"].item())
    print("example session:", example_key)
    fig1(res)
    fig2(res, example_key)
    fig3(res)
    p_comb = fig4(res)
    fig5(res)
    fig6(res)
    print("Stouffer combined p (global null):", p_comb)
