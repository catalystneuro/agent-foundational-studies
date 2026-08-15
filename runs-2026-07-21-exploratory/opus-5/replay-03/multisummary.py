"""Aggregate the per-session replay results and draw the cross-session figure."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, wilcoxon

CACHE, FIG = "cache", "figures"
KEYS = ("POSTEpoch", "PREEpoch", "control")


def _sig_table(a, b):
    return [[a.significant.sum(), (~a.significant).sum()],
            [b.significant.sum(), (~b.significant).sum()]]


def summarize(summary_csv=f"{CACHE}/multi_session_summary.csv"):
    S = pd.read_csv(summary_csv)
    pooled = {k: pd.concat([pd.read_csv(f"{CACHE}/events_{s}_{k}.csv") for s in S.session],
                           keys=list(S.session))
              for k in KEYS}
    post, pre, ctrl = pooled["POSTEpoch"], pooled["PREEpoch"], pooled["control"]

    stats = {}
    stats["chi2_post_ctrl"], stats["p_post_ctrl"] = chi2_contingency(_sig_table(post, ctrl))[:2]
    stats["chi2_pre_ctrl"], stats["p_pre_ctrl"] = chi2_contingency(_sig_table(pre, ctrl))[:2]
    stats["chi2_post_pre"], stats["p_post_pre"] = chi2_contingency(_sig_table(post, pre))[:2]
    stats["p_u_post_ctrl"] = mannwhitneyu(post.r.abs(), ctrl.r.abs(),
                                          alternative="greater")[1]
    stats["p_u_pre_ctrl"] = mannwhitneyu(pre.r.abs(), ctrl.r.abs(),
                                         alternative="greater")[1]
    stats["p_u_post_pre"] = mannwhitneyu(post.r.abs(), pre.r.abs(),
                                         alternative="greater")[1]
    stats["w_post_ctrl"], stats["p_w_post_ctrl"] = wilcoxon(
        S.frac_sig_POST, S.frac_sig_ctrl, alternative="greater")

    print(S[["session", "n_place", "n_events_POST", "frac_sig_POST", "frac_sig_PREE",
             "frac_sig_ctrl", "median_speed_mps"]].to_string(index=False))
    print(f"\npooled ({len(S)} sessions): POST {100 * post.significant.mean():.1f}%, "
          f"PRE {100 * pre.significant.mean():.1f}%, "
          f"cell-ID control {100 * ctrl.significant.mean():.1f}%")
    print(f"  prevalence POST vs control : chi2 = {stats['chi2_post_ctrl']:.1f}, "
          f"p = {stats['p_post_ctrl']:.2g}")
    print(f"  prevalence PRE  vs control : chi2 = {stats['chi2_pre_ctrl']:.1f}, "
          f"p = {stats['p_pre_ctrl']:.2g}")
    print(f"  |r|        POST vs control : Mann-Whitney p = {stats['p_u_post_ctrl']:.2g}")
    print(f"  |r|        PRE  vs control : Mann-Whitney p = {stats['p_u_pre_ctrl']:.2g}")
    print(f"  per-session POST > control : Wilcoxon W = {stats['w_post_ctrl']:.1f}, "
          f"p = {stats['p_w_post_ctrl']:.3f} (n = {len(S)})")
    return S, pooled, stats


def figure(S, pooled, stats, path=f"{FIG}/07_multi_session.png"):
    post, pre, ctrl = pooled["POSTEpoch"], pooled["PREEpoch"], pooled["control"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))

    ax = axes[0]
    x = np.arange(len(S))
    wid = 0.27
    for k, (col, lab, c) in enumerate([("frac_sig_ctrl", "cell-ID shuffled", "0.6"),
                                       ("frac_sig_PREE", "PRE sleep", "tab:blue"),
                                       ("frac_sig_POST", "POST sleep", "tab:red")]):
        ax.bar(x + (k - 1) * wid, 100 * S[col], wid, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("-", "\n") for s in S.session], fontsize=8)
    ax.set_ylabel("significant replay events (%)")
    ax.set_title(f"Replay prevalence per session\npooled POST vs control "
                 f"p = {stats['p_post_ctrl']:.1e}", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[1]
    bins = np.linspace(0, 1, 26)
    ax.hist(ctrl.r.abs(), bins=bins, density=True, histtype="step", color="k", lw=2,
            label=f"cell-ID shuffled (n={len(ctrl)})")
    ax.hist(pre.r.abs(), bins=bins, density=True, alpha=0.5, color="tab:blue",
            label=f"PRE (n={len(pre)})")
    ax.hist(post.r.abs(), bins=bins, density=True, alpha=0.5, color="tab:red",
            label=f"POST (n={len(post)})")
    ax.set_xlabel("|weighted correlation|")
    ax.set_ylabel("density")
    ax.set_title(f"Pooled sequence scores ({len(S)} sessions)\n"
                 f"POST > control p = {stats['p_u_post_ctrl']:.1e}; "
                 f"PRE > control p = {stats['p_u_pre_ctrl']:.2f}", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[2]
    sig = post[post.significant]
    ax.hist(sig.slope.abs(), bins=np.linspace(0, 30, 31), color="tab:red", alpha=0.8)
    ax.axvline(sig.slope.abs().median(), color="k", ls="--",
               label=f"median {sig.slope.abs().median():.1f} m/s")
    ax.set_xlabel("replay speed |slope| (m/s)")
    ax.set_ylabel("events")
    ax.set_title(f"Pooled replay speed\n{(sig.slope > 0).mean() * 100:.0f}% forward, "
                 f"{(sig.slope < 0).mean() * 100:.0f}% reverse", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[3]
    excess = 100 * (S.frac_sig_POST - S.frac_sig_ctrl)
    ax.scatter(S.n_place, excess, s=60, c="tab:red", zorder=3)
    for _, r in S.iterrows():
        ax.annotate(r.session.split("-")[0], (r.n_place,
                    100 * (r.frac_sig_POST - r.frac_sig_ctrl)),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("place cells in the decoding ensemble")
    ax.set_ylabel("POST minus control (percentage points)")
    ax.set_title("Effect size scales with ensemble size\n"
                 f"Spearman rho = {S.n_place.corr(excess, method='spearman'):.2f} "
                 f"(n = {len(S)} sessions)", fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("wrote", path)
    return fig
