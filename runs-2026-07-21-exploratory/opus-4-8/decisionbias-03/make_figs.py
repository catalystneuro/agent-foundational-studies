"""Figure generation for the pre-stimulus decision-bias decoding analysis."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd; np.seterr(all="ignore")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from analyze import load, CENTERS, EDGES, PRESTIM, SESSIONS, FIG

BLU, RED, GRY = "#2c6fbb", "#c0392b", "#7f8c8d"


def fig_raw(name="NYU-11"):
    """Validation: peri-onset population activity aligned to stimulus onset, split by block."""
    T, d = load(name)
    rate = T / np.diff(EDGES)[None, None, :]           # Hz per bin
    blk02 = d.prob_left.values == 0.2                  # right-heavy block
    blk08 = d.prob_left.values == 0.8                  # left-heavy block
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) trial-averaged firing heatmap, units sorted by pre-stim rate
    m = rate.mean(0)                                    # units x bins
    order = np.argsort(m[:, PRESTIM].mean(1))
    im = ax[0].imshow(m[order], aspect="auto", cmap="magma",
                      extent=[CENTERS[0], CENTERS[-1], 0, m.shape[0]],
                      vmax=np.percentile(m, 99))
    ax[0].axvline(0, color="w", ls="--", lw=1)
    ax[0].axvspan(-0.4, 0, color="cyan", alpha=0.15)
    ax[0].set(xlabel="Time from stimulus onset (s)", ylabel="Unit (sorted)",
              title=f"{name}: trial-averaged firing")
    plt.colorbar(im, ax=ax[0], label="Hz", fraction=0.046)
    # (b) mean population rate split by block
    for mask, c, lab in [(blk02, RED, "P(left)=0.2 (right-biased)"),
                         (blk08, BLU, "P(left)=0.8 (left-biased)")]:
        mu = rate[mask].mean((0, 1)); se = rate[mask].mean(1).std(0) / np.sqrt(mask.sum())
        ax[1].plot(CENTERS, mu, color=c, label=lab)
        ax[1].fill_between(CENTERS, mu - se, mu + se, color=c, alpha=0.2)
    ax[1].axvline(0, color="k", ls="--", lw=1, label="stimulus onset")
    ax[1].axvspan(-0.4, 0, color="cyan", alpha=0.15, label="pre-stim window")
    ax[1].set(xlabel="Time from stimulus onset (s)", ylabel="Mean population rate (Hz)",
              title="Population rate by block")
    ax[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_raw_validation.png", bbox_inches="tight"); plt.close(fig)


def fig_decode(res):
    """Per-session pre-stimulus decoding of upcoming choice and upcoming bias (block)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(res))
    for a, key, lab, col, loc in [(ax[0], "choice", "upcoming choice", BLU, "upper left"),
                                  (ax[1], "block", "upcoming bias (block prior)", RED, "lower right")]:
        a.bar(x, res[f"{key}_auc"], color=col, alpha=0.85, width=0.6)
        a.plot(x, res[f"{key}_null95"], "k_", ms=18, mew=2, label="shuffle null (95th pct)")
        a.axhline(0.5, color=GRY, ls=":", lw=1)
        for i, p in enumerate(res[f"{key}_p"]):
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            a.text(i, res[f"{key}_auc"].iloc[i] + 0.015, star, ha="center", fontsize=9)
        a.set(xticks=x, ylim=(0.45, 0.93), ylabel="Cross-validated AUC",
              title=f"Decoding {lab}\nfrom pre-stimulus activity (-400 to 0 ms)")
        a.set_xticklabels(res.session, rotation=30, ha="right")
        a.legend(fontsize=8, loc=loc)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_prestim_decoding.png", bbox_inches="tight"); plt.close(fig)


def fig_temporal(choice_curve, block_curve):
    """Sliding-window decoding vs time relative to onset (mean +/- SEM across sessions)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for curve, c, lab in [(choice_curve, BLU, "upcoming choice"),
                          (block_curve, RED, "upcoming bias (block prior)")]:
        mu = np.nanmean(curve, 0); se = np.nanstd(curve, 0) / np.sqrt(curve.shape[0])
        ax.plot(CENTERS, mu, "-o", color=c, ms=3, label=lab)
        ax.fill_between(CENTERS, mu - se, mu + se, color=c, alpha=0.2)
    ax.axhline(0.5, color=GRY, ls=":", lw=1, label="chance")
    ax.axvline(0, color="k", ls="--", lw=1.2, label="stimulus onset")
    ax.axvspan(-0.4, 0, color="cyan", alpha=0.15, label="pre-stim window")
    ax.set(xlabel="Time of 100-ms decoding window rel. to onset (s)",
           ylabel="Cross-validated AUC (mean $\\pm$ SEM, 6 sessions)",
           title="Decision bias is present in neural activity\nbefore stimulus onset")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_temporal_decoding.png", bbox_inches="tight"); plt.close(fig)


def fig_behavior_link(prestim_proba):
    """Behavioral bias by block, and neural pre-stim prediction of choice on zero-contrast trials."""
    # Pool trials across sessions
    frames = []
    for name, (d, p_c, blk) in prestim_proba.items():
        dd = d.copy(); dd["p_choice"] = p_c; dd["session"] = name
        frames.append(dd)
    D = pd.concat(frames, ignore_index=True)
    # signed contrast: +right(-)/... use side to sign. clockwise=1.
    sign = np.where(D.side.values == "left", -1.0, 1.0)
    D["scon"] = sign * D.contrast.values

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # (a) psychometric: P(clockwise) vs signed contrast, by block
    for pl, c, lab in [(0.2, RED, "P(left)=0.2"), (0.5, GRY, "0.5"), (0.8, BLU, "P(left)=0.8")]:
        sub = D[D.prob_left == pl]
        xs = np.sort(sub.scon.unique())
        ys = [sub[sub.scon == x].y.mean() for x in xs]
        ns = [(sub.scon == x).sum() for x in xs]
        ax[0].plot(xs, ys, "-o", color=c, label=lab, ms=4)
    ax[0].axhline(0.5, color="k", ls=":", lw=0.8); ax[0].axvline(0, color="k", ls=":", lw=0.8)
    ax[0].set(xlabel="Signed contrast (%)  (- left stim, + right stim)",
              ylabel="P(clockwise choice)", title="(a) Behavioral bias by block")
    ax[0].legend(fontsize=8)
    # (b) zero-contrast choice bias by block (pure bias, no sensory evidence)
    z = D[D.contrast == 0]
    blocks = [0.2, 0.5, 0.8]; cols = [RED, GRY, BLU]
    means = [z[z.prob_left == b].y.mean() for b in blocks]
    ns = [(z.prob_left == b).sum() for b in blocks]
    sems = [np.sqrt(m * (1 - m) / n) for m, n in zip(means, ns)]
    ax[1].bar(range(3), means, yerr=sems, color=cols, alpha=0.85, capsize=4)
    ax[1].axhline(0.5, color="k", ls=":", lw=1)
    for i, (m, n) in enumerate(zip(means, ns)):
        ax[1].text(i, m + 0.03, f"n={n}", ha="center", fontsize=8)
    ax[1].set(xticks=range(3), xticklabels=["0.2", "0.5", "0.8"], ylim=(0, 1),
              xlabel="Block prior P(left)", ylabel="P(clockwise) | zero contrast",
              title="(b) Bias at zero contrast\n(choice = pure internal bias)")
    # (c) neural pre-stim decoder predicts choice on zero-contrast trials (pooled)
    zc = D[D.contrast == 0]
    from sklearn.metrics import roc_auc_score
    auc0 = roc_auc_score(zc.y.values, zc.p_choice.values)
    # bin decoder output into terciles, show P(clockwise)
    q = pd.qcut(zc.p_choice, 3, labels=["low", "mid", "high"])
    gm = zc.groupby(q).y.mean(); gn = zc.groupby(q).y.size()
    ax[2].bar(range(3), gm.values, color=[ "#95a5a6","#5d6d7e","#212f3d"], alpha=0.9)
    ax[2].axhline(0.5, color="k", ls=":", lw=1)
    for i, n in enumerate(gn.values):
        ax[2].text(i, gm.values[i] + 0.03, f"n={n}", ha="center", fontsize=8)
    ax[2].set(xticks=range(3), xticklabels=["low", "mid", "high"], ylim=(0, 1),
              xlabel="Pre-stim decoder output (tercile)", ylabel="P(clockwise choice)",
              title=f"(c) Pre-stim activity predicts\nzero-contrast choice (AUC={auc0:.2f})")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_behavior_link.png", bbox_inches="tight"); plt.close(fig)
    return auc0


def make_figures(res, prestim_proba, choice_curve, block_curve):
    fig_raw()
    fig_decode(res)
    fig_temporal(choice_curve, block_curve)
    auc0 = fig_behavior_link(prestim_proba)
    print("figures written; pooled zero-contrast choice AUC =", round(auc0, 3))
