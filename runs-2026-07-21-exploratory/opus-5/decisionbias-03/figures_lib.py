"""Figure construction for the pre-stimulus prior-decoding analysis."""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

import analysis_lib as al

BLUE, RED, GREY = "#2c6fbb", "#c8452f", "#8a8a8a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9.5, "axes.titlesize": 10})


def _save(fig, name):
    fig.savefig(name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def cache_for(res):
    return pd.read_pickle(f"cache/{res['subject']}_{res['eid'][:8]}.pkl")


# --------------------------------------------------------------------------- #
def fig_behavior(R):
    fig = plt.figure(figsize=(13.5, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.33)
    r0 = R[0]
    d0 = r0["trials"]

    ax = fig.add_subplot(gs[0, :2])
    ax.step(np.arange(len(d0)), np.where(d0["block_right"] == 1, 0.2, 0.8),
            where="post", color="k", lw=1.3)
    lc = d0["gabor_stimulus_contrast"].values <= 6.25
    ax.plot(np.where(lc)[0], np.where(d0["choice_right"].values[lc] == 1, 0.13, 0.87),
            "|", color=RED, ms=5, alpha=0.7)
    ax.set_ylim(0.05, 0.95)
    ax.text(1.005, 0.87, "chose left", transform=ax.get_yaxis_transform(),
            color=RED, fontsize=8, va="center")
    ax.text(1.005, 0.13, "chose right", transform=ax.get_yaxis_transform(),
            color=RED, fontsize=8, va="center")
    ax.set_xlabel("trial within session")
    ax.set_ylabel("block prior  P(stim left)")
    ax.set_title(f"Hidden block structure and choices on weak stimuli "
                 f"({r0['subject']}, {r0['n_blocks']} blocks; red ticks = choices)")

    allc = pd.concat([r["trials"].assign(sess=i) for i, r in enumerate(R)])
    ax = fig.add_subplot(gs[0, 2])
    for br, col, lab in [(0, BLUE, "P(left)=0.8"), (1, RED, "P(left)=0.2")]:
        s = allc[allc["block_right"] == br]
        g = s.groupby("signed_contrast")["choice_right"].agg(["mean", "count", "sem"])
        ax.errorbar(g.index, g["mean"], yerr=g["sem"], fmt="o-", color=col,
                    label=lab, ms=4, lw=1.4, capsize=2)
    ax.axhline(0.5, ls=":", c="k", lw=0.8)
    ax.set_xlabel("signed contrast (%)")
    ax.set_ylabel("P(choose right)")
    ax.set_title("Pooled psychometric curves")
    ax.legend(fontsize=8, frameon=False)

    bias = []
    for r in R:
        d = r["trials"]
        m = d["gabor_stimulus_contrast"] <= 6.25
        bias.append(d.loc[m & (d["block_right"] == 1), "choice_right"].mean()
                    - d.loc[m & (d["block_right"] == 0), "choice_right"].mean())
    bias = np.array(bias)
    ax = fig.add_subplot(gs[1, 0])
    o = np.argsort(-bias)
    ax.bar(np.arange(len(bias)), bias[o], color=[RED if x > 0 else GREY for x in bias[o]])
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(np.arange(len(bias)))
    ax.set_xticklabels([R[i]["subject"] for i in o], rotation=90, fontsize=7)
    ax.set_ylabel(r"$\Delta$P(choose right)")
    p = stats.wilcoxon(bias)[1]
    ax.set_title(f"Behavioural bias at $\\leq$6.25% contrast\n"
                 f"median {np.median(bias):+.3f}, Wilcoxon p={p:.2g}", fontsize=9)

    bins = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 200)]
    ax = fig.add_subplot(gs[1, 1])
    ys, es = [], []
    for lo, hi in bins:
        s = allc[(allc["trial_in_block"] >= lo) & (allc["trial_in_block"] < hi)
                 & (allc["gabor_stimulus_contrast"] <= 6.25)]
        a = s.loc[s["block_right"] == 1, "choice_right"]
        b = s.loc[s["block_right"] == 0, "choice_right"]
        ys.append(a.mean() - b.mean())
        es.append(float(np.hypot(a.sem(), b.sem())))
    ax.errorbar(np.arange(len(bins)), ys, yerr=es, fmt="o-", color=RED, capsize=3)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in bins], fontsize=8)
    ax.set_xlabel("trials since block switch")
    ax.set_ylabel(r"$\Delta$P(choose right)")
    ax.set_title("Bias builds up within a block")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    conv = np.mean([r["conv"] for r in R])
    ax.text(0, 1,
            "DANDI:000409  IBL Brain Wide Map\n"
            f"{len(R)} sessions / {len(set(r['subject'] for r in R))} mice\n"
            f"{sum(r['n_units'] for r in R)} units, "
            f"{sum(r['n_trials'] for r in R)} analysed trials\n"
            f"choice-convention check: {conv*100:.1f}% correct\n"
            f"on >=50% contrast trials\n\n"
            "Task: two-alternative visual contrast\n"
            "detection reported by turning a wheel.\n"
            "After the first ~90 trials the stimulus side\n"
            "is drawn from hidden blocks of 20-100 trials\n"
            "with P(left) = 0.8 or 0.2.  Mice track this\n"
            "prior; it biases choices when the stimulus\n"
            "is weak, and it is the 'decision bias' we try\n"
            "to read out from spiking BEFORE the stimulus.",
            va="top", fontsize=8.6, family="monospace")
    _save(fig, "fig01_task_and_behavior.png")
    return bias


# --------------------------------------------------------------------------- #
def fig_raw(R, which=0):
    r = R[which]
    c = cache_for(r)
    d = r["trials"]
    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.26,
                          height_ratios=[1.0, 1.25, 1.0])

    # (a) raw spiking
    ax = fig.add_subplot(gs[0, :])
    t0 = c["raw_t0"]
    for i, (u, s) in enumerate(c["raw_spikes"].items()):
        ax.plot(s, np.full_like(s, i), "|", ms=2.5, color="k", alpha=0.55)
    for st in d["stim_on"]:
        if t0 < st < t0 + 26:
            ax.axvspan(st - 0.4, st, color=BLUE, alpha=0.18, lw=0)
            ax.axvline(st, color=RED, lw=1.1)
    ax.set_xlim(t0, t0 + 26)
    ax.set_xlabel("session time (s)")
    ax.set_ylabel("unit")
    ax.set_title(f"{r['subject']}: raw spiking of 60 units. "
                 "Red = stimulus onset, blue = 400 ms pre-stimulus window")

    # (b) most block-selective unit
    # Restrict the example to units with a reasonable pre-stimulus count, so the
    # raster is readable; the extreme AUCs are otherwise dominated by rare units.
    aucs = r["aucs"]
    keep = r["peri_keep"]                       # map back into the unfiltered cache
    rate = c["X_pre"][:, keep].mean(0)
    elig = rate >= np.percentile(rate, 70)
    score = np.where(elig, np.abs(aucs - 0.5), -1)
    u = int(np.nanargmax(score))
    peri = c["peri"][keep[u]]
    order = np.argsort(d["block_right"].values, kind="stable")
    ax = fig.add_subplot(gs[1, 0])
    for row, ti in enumerate(order):
        col = RED if d["block_right"].iloc[ti] == 1 else BLUE
        s = peri[ti]
        ax.plot(s, np.full_like(s, row), "|", ms=1.8, color=col, alpha=0.75)
    ax.axvline(0, color="k", lw=1)
    ax.axvspan(-0.4, 0, color="k", alpha=0.07, lw=0)
    ax.set_xlim(-1.0, 0.3)
    ax.set_xlabel("time from stimulus onset (s)")
    ax.set_ylabel("trial (sorted by block)")
    ax.set_title("Most block-selective unit among the 30% with the\n"
                 f"highest pre-stimulus rate, AUC={aucs[u]:.2f}\n"
                 f"{r['regions'][u][:44]}", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    edges = np.arange(-1.0, 0.301, 0.05)
    for br, col, lab in [(0, BLUE, "P(left)=0.8"), (1, RED, "P(left)=0.2")]:
        sel = np.where(d["block_right"].values == br)[0]
        h = np.sum([np.histogram(peri[i], bins=edges)[0] for i in sel], 0)
        ax.step(edges[:-1], h / len(sel) / 0.05, where="post", color=col, label=lab)
    ax.axvline(0, color="k", lw=1)
    ax.axvspan(-0.4, 0, color="k", alpha=0.07, lw=0)
    ax.set_xlabel("time from stimulus onset (s)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("Same unit, PSTH split by block", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    # (c) wheel quiescence
    tt, prof = c["wheel_profile"]
    ax = fig.add_subplot(gs[2, 0])
    for br, col, lab in [(0, BLUE, "P(left)=0.8"), (1, RED, "P(left)=0.2")]:
        sel = d["block_right"].values == br
        m = np.nanmean(prof[sel], 0)
        se = np.nanstd(prof[sel], 0) / np.sqrt(sel.sum())
        ax.plot(tt, m, color=col, label=lab)
        ax.fill_between(tt, m - se, m + se, color=col, alpha=0.25, lw=0)
    ax.axvline(0, color="k", lw=1)
    ax.axvspan(-0.4, 0, color="k", alpha=0.07, lw=0)
    ax.set_xlabel("time from stimulus onset (s)")
    ax.set_ylabel("|wheel velocity| (rad/s)")
    ax.set_title("Wheel is held still before stimulus onset\n"
                 "(the task enforces a quiescence period)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    # (d) pre-stimulus population matrix
    ax = fig.add_subplot(gs[2, 1])
    Z = al.highpass(c["X_pre"][:, r["peri_keep"]], r["w"])
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    uorder = np.argsort(aucs)          # units sorted by block preference
    im = ax.imshow(Z[np.ix_(order, uorder)].T, aspect="auto", cmap="RdBu_r",
                   vmin=-1.5, vmax=1.5, interpolation="nearest")
    ax.axvline(int((d["block_right"] == 0).sum()), color="k", lw=1.5)
    ax.set_xlabel("trial (sorted by block)")
    ax.set_ylabel("unit (sorted by block AUC)")
    ax.set_title(f"Pre-stimulus counts, drift-removed and z-scored\n"
                 f"(w={r['w']} trials); line splits the two blocks", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save(fig, "fig02_raw_data.png")


# --------------------------------------------------------------------------- #
def fig_single_units(R):
    aucs = np.concatenate([r["aucs"] for r in R])
    ps = np.concatenate([r["auc_p"] for r in R])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))

    ax = axes[0]
    b = np.arange(0.2, 0.811, 0.02)
    ax.hist(aucs, bins=b, color=GREY, edgecolor="w", label="all units")
    ax.hist(aucs[ps < 0.05], bins=b, color=RED, edgecolor="w",
            label="p<0.05 vs pseudo-session")
    ax.axvline(0.5, c="k", ls=":")
    ax.set_xlabel("AUC: pre-stimulus rate discriminates block")
    ax.set_ylabel("units")
    k, n = int((ps < 0.05).sum()), len(ps)
    pb = stats.binomtest(k, n, 0.05, alternative="greater").pvalue
    ax.set_title(f"{k/n*100:.1f}% of {n} units block-modulated\n"
                 f"(5% expected by chance, binomial p={pb:.2g})", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    per = [(r["subject"], (r["auc_p"] < 0.05).mean(), r["n_units"]) for r in R]
    per.sort(key=lambda x: -x[1])
    ax.bar(np.arange(len(per)), [x[1] for x in per], color=BLUE)
    ax.axhline(0.05, c=RED, ls="--", label="chance (5%)")
    ax.set_xticks(np.arange(len(per)))
    ax.set_xticklabels([f"{s} ({n})" for s, _, n in per], rotation=90, fontsize=7)
    ax.set_ylabel("fraction of units with p<0.05")
    ax.set_title("Block-modulated units per session\n(unit count in brackets)",
                 fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    x = np.concatenate([[r["acc"]] * r["n_units"] for r in R])
    fr = np.array([(r["auc_p"] < 0.05).mean() for r in R])
    acc = np.array([r["acc"] for r in R])
    ax.scatter(fr, acc, c=BLUE, s=34)
    rho, pv = stats.spearmanr(fr, acc)
    ax.set_xlabel("fraction of block-modulated units")
    ax.set_ylabel("population decoding accuracy")
    ax.set_title(f"Single-unit and population effects agree\n"
                 f"Spearman rho={rho:.2f}, p={pv:.3f}", fontsize=9.5)
    fig.tight_layout()
    _save(fig, "fig03_single_unit_selectivity.png")


# --------------------------------------------------------------------------- #
def fig_decoding(R):
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.36, width_ratios=[1.7, 1, 1])
    accs = np.array([r["acc"] for r in R])

    ax = fig.add_subplot(gs[0, 0])
    o = np.argsort(-accs)
    for k, i in enumerate(o):
        r = R[i]
        lo, hi = np.percentile(r["null"], [2.5, 97.5])
        ax.plot([k, k], [lo, hi], color=GREY, lw=6, solid_capstyle="butt", alpha=0.55)
        ax.plot(k, np.median(r["null"]), "_", color="k", ms=10)
        ax.plot(k, r["acc"], "o", color=RED if r["p"] < 0.05 else BLUE, ms=7.5)
    ax.axhline(0.5, c="k", ls=":", lw=0.8)
    ax.set_xticks(np.arange(len(R)))
    ax.set_xticklabels([f"{R[i]['subject']} ({R[i]['n_units']}u)" for i in o],
                       rotation=90, fontsize=7)
    ax.set_ylabel("balanced accuracy")
    nsig = int(sum(r["p"] < 0.05 for r in R))
    ax.set_title("Block prior decoded from the 400 ms before stimulus onset\n"
                 f"red = p<0.05 vs pseudo-session null ({nsig}/{len(R)}); "
                 "grey bar = null 95% interval", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    z = np.array([(r["acc"] - r["null"].mean()) / r["null"].std() for r in R])
    ax.hist(z, bins=10, color=BLUE, edgecolor="w")
    ax.axvline(0, c="k", ls=":")
    pw = stats.wilcoxon(z)[1]
    ax.set_xlabel("z relative to pseudo-session null")
    ax.set_ylabel("sessions")
    ax.set_title(f"Consistent across sessions\nmedian z={np.median(z):.2f}, "
                 f"Wilcoxon p={pw:.2g}", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 2])
    pooled = np.mean([r["null"] for r in R], axis=0)
    ax.hist(pooled, bins=25, color=GREY, edgecolor="w", label="pseudo-session null")
    ax.axvline(accs.mean(), color=RED, lw=2.3, label="observed")
    pp = (np.sum(pooled >= accs.mean()) + 1) / (len(pooled) + 1)
    ax.set_xlabel("session-mean balanced accuracy")
    ax.set_ylabel("pseudo-sessions")
    ax.set_title(f"Pooled test\nobserved {accs.mean():.3f}, p={pp:.2g}", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)
    _save(fig, "fig04_block_decoding.png")
    return accs, z, pp


# --------------------------------------------------------------------------- #
def fig_time_resolved(R):
    centers = R[0]["centers"]
    A = np.array([r["tr_acc"] for r in R])
    N = np.array([r["tr_null"] for r in R])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))

    ax = axes[0]
    m, se = A.mean(0), A.std(0) / np.sqrt(len(A))
    nullm = N.mean(0)
    ax.fill_between(centers, np.percentile(nullm, 2.5, 0),
                    np.percentile(nullm, 97.5, 0), color=GREY, alpha=0.45, lw=0,
                    label="pseudo-session null (95%)")
    ax.plot(centers, m, "o-", color=RED, label="observed", ms=4)
    ax.fill_between(centers, m - se, m + se, color=RED, alpha=0.22, lw=0)
    ax.axvline(0, c="k", lw=1)
    ax.axvspan(-0.4, 0, color=BLUE, alpha=0.10, lw=0)
    ax.set_xlabel("window centre relative to stimulus onset (s)")
    ax.set_ylabel("balanced accuracy (session mean)")
    ax.set_title("The prior is readable throughout the pre-stimulus period\n"
                 "200 ms windows, whole blocks held out", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    pv = np.array([(np.sum(nullm[:, j] >= m[j]) + 1) / (nullm.shape[0] + 1)
                   for j in range(len(centers))])
    ax.semilogy(centers, pv, "o-", color=BLUE, ms=4)
    ax.axhline(0.05, c=RED, ls="--", label="p=0.05")
    ax.axvline(0, c="k", lw=1)
    ax.axvspan(-0.4, 0, color=BLUE, alpha=0.10, lw=0)
    ax.set_xlabel("window centre relative to stimulus onset (s)")
    ax.set_ylabel("p vs pseudo-session null")
    ax.set_title("Per-window significance", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, "fig05_time_resolved.png")
    return centers, m, pv


# --------------------------------------------------------------------------- #
def fig_controls(R, G, WS, CS, params):
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    V = np.array([[r["stack"]["acc_neu"], r["stack"]["acc_hist"],
                   r["stack"]["acc_stack"]] for r in R])
    for row in V:
        ax.plot(range(3), row, "-", color=GREY, alpha=0.5, lw=0.9)
    ax.plot(range(3), V.mean(0), "o-", color=RED, lw=2.2, ms=8)
    ax.axhline(0.5, c="k", ls=":", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["pre-stimulus\nspikes", "previous-trial\nbehaviour",
                        "stacked"], fontsize=8)
    ax.set_ylabel("balanced accuracy")
    beta = np.array([r["stack"]["beta_neural"] for r in R])
    p1 = stats.wilcoxon(beta)[1]
    ax.set_title("The previous trial predicts the block better than\n"
                 "the neural window, but the two are not redundant\n"
                 f"(stacked neural weight {int((beta>0).sum())}/{len(beta)} "
                 f"positive, p={p1:.2g})", fontsize=8.6)

    ax = fig.add_subplot(gs[0, 1])
    am = np.array([r["acc_matched"] for r in R])
    ac = np.array([r["acc"] for r in R])
    pm = np.array([r["p_matched"] for r in R])
    ok = np.isfinite(am)
    lim = [min(ac.min(), np.nanmin(am)) - 0.03, max(ac.max(), np.nanmax(am)) + 0.03]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.axhline(0.5, c=GREY, lw=0.6)
    ax.axvline(0.5, c=GREY, lw=0.6)
    ax.scatter(ac[ok], am[ok], c=[RED if q < 0.05 else BLUE for q in pm[ok]], s=34)
    ax.set_xlabel("accuracy, all trials")
    ax.set_ylabel("accuracy, history-matched trials")
    ax.set_title("Matched on previous choice, reward\nand stimulus side; "
                 f"{int((pm[ok]<0.05).sum())}/{int(ok.sum())} sessions p<0.05",
                 fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    names = ["wheel_speed", "pupil", "motion_energy"]
    pos = 0
    for k in names:
        e = []
        for r in R:
            v = r["beh"].get(k)
            if v is None:
                continue
            y = r["trials"]["block_right"].values
            m = np.isfinite(v)
            if m.sum() < 50 or np.nanstd(v[m]) == 0:
                continue
            e.append((v[m & (y == 1)].mean() - v[m & (y == 0)].mean())
                     / np.nanstd(v[m]))
        e = np.array(e)
        if not len(e):
            continue
        ax.scatter(np.full(len(e), pos) + np.random.default_rng(0)
                   .uniform(-.09, .09, len(e)), e, s=24, color=GREY)
        ax.plot([pos - .22, pos + .22], [e.mean()] * 2, color=RED, lw=2.6)
        lab = (f"p={stats.wilcoxon(e)[1]:.2f}\nn={len(e)}" if len(e) > 5
               else f"n={len(e)}")
        ax.annotate(lab, (pos, e.max()),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=7.5)
        pos += 1
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(range(pos))
    ax.set_xticklabels([n.replace("_", "\n") for n in names[:pos]], fontsize=8)
    ax.set_ylabel("block difference (Cohen's d)")
    ax.set_title("Pre-stimulus behaviour barely\ndiffers between blocks", fontsize=9.5)
    ax.margins(y=0.22)

    ax = fig.add_subplot(gs[1, 0])
    bins = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 200)]
    curves = []
    for r in R:
        d, pr = r["trials"], r["prob"]
        sgn = np.where(d["block_right"].values == 1, 1, -1)
        ev = sgn * (pr - 0.5)
        tb = d["trial_in_block"].values
        curves.append([np.nanmean(ev[(tb >= lo) & (tb < hi)]) for lo, hi in bins])
    curves = np.array(curves)
    m, se = np.nanmean(curves, 0), np.nanstd(curves, 0) / np.sqrt(len(curves))
    ax.errorbar(np.arange(len(bins)), m, yerr=se, fmt="o-", color=RED, capsize=3)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in bins], fontsize=8)
    ax.set_xlabel("trials since block switch")
    ax.set_ylabel("decoder evidence for the true block")
    ax.set_title("Decoded prior strengthens within a block,\n"
                 "mirroring the behavioural bias", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    M = G.mean(0)
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(CS)))
    ax.set_xticklabels([f"{c:g}" for c in CS])
    ax.set_yticks(range(len(WS)))
    ax.set_yticklabels([str(w) for w in WS])
    ax.set_xlabel("ridge C")
    ax.set_ylabel("drift-removal width (trials)")
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            ax.text(b, a, f"{M[a,b]:.2f}", ha="center", va="center",
                    color="w", fontsize=7.5)
    ax.set_title("Session-mean accuracy over the\nhyper-parameter grid", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    chosen = ", ".join(sorted({f"w={w},C={c:g}" for w, c in params}))
    ax.text(0, 1,
            "Controls applied\n\n"
            "1. Whole blocks (in left/right pairs) held\n"
            "   out in cross-validation.\n"
            "2. Slow drift removed with a moving-average\n"
            "   high-pass over trials; without it the\n"
            "   decoder inverts on held-out blocks.\n"
            "3. Null = pseudo-sessions built from this\n"
            "   session's own block lengths, so the label\n"
            "   autocorrelation matches exactly.\n"
            "4. Hyper-parameters chosen leave-one-\n"
            "   session-out, never on the tested session.\n"
            "5. Trials with wheel movement inside the\n"
            "   window dropped.\n"
            "6. Compared against a previous-trial\n"
            "   behavioural-history decoder, and repeated\n"
            "   on history-matched trials.\n"
            "7. Pre-stimulus wheel, pupil and face motion\n"
            "   checked for block differences.\n\n"
            f"chosen: {chosen}",
            va="top", fontsize=8.2, family="monospace")
    _save(fig, "fig06_controls.png")


# --------------------------------------------------------------------------- #
def fig_choice(R):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    ok = [r for r in R if np.isfinite(r["ch"]["acc"])]
    a = np.array([r["ch"]["acc"] for r in ok])
    p = np.array([r["ch"]["p"] for r in ok])
    wi = np.array([r["ch"]["within"] for r in ok])

    ax = axes[0]
    o = np.argsort(-a)
    ax.bar(np.arange(len(a)), a[o] - 0.5, bottom=0.5,
           color=[RED if p[i] < 0.05 else BLUE for i in o])
    ax.axhline(0.5, c="k", lw=0.9)
    ax.set_xticks(np.arange(len(a)))
    ax.set_xticklabels([ok[i]["subject"] for i in o], rotation=90, fontsize=7)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Upcoming choice on weak-stimulus trials\n"
                 f"decoded before onset ({int((p<0.05).sum())}/{len(a)} p<0.05 vs "
                 "shift null)", fontsize=9.5)

    ax = axes[1]
    ax.scatter(a, wi, c=BLUE, s=34)
    lim = [min(a.min(), wi.min()) - .03, max(a.max(), wi.max()) + .03]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.axhline(0.5, c=GREY, lw=0.7)
    ax.axvline(0.5, c=GREY, lw=0.7)
    pw = stats.wilcoxon(wi - 0.5)[1]
    ax.set_xlabel("accuracy, all weak-stimulus trials")
    ax.set_ylabel("accuracy computed within block")
    sig = "still above chance" if pw < 0.05 else "no longer above chance"
    ax.set_title("Choice prediction recomputed within block:\n"
                 f"{sig} (mean {wi.mean():.3f}, p={pw:.2g})", fontsize=9)

    ax = axes[2]
    coefs, subs = [], []
    for r in R:
        d = r["trials"]
        lc = (d["gabor_stimulus_contrast"] <= 12.5).values
        y = d["choice_right"].values[lc]
        z = r["prob"][lc]
        b = d["block_right"].values[lc]
        m = np.isfinite(y) & np.isfinite(z)
        if m.sum() < 60 or len(np.unique(y[m])) < 2:
            continue
        Xd = np.column_stack([stats.zscore(z[m]), b[m] - b[m].mean()])
        clf = LogisticRegression(max_iter=2000).fit(Xd, y[m])
        coefs.append(clf.coef_[0])
        subs.append(r["subject"])
    coefs = np.array(coefs)
    rng = np.random.default_rng(0)
    for j, (lab, col) in enumerate([("decoded prior\n(neural)", RED),
                                    ("block label\n(task)", BLUE)]):
        v = coefs[:, j]
        ax.scatter(np.full(len(v), j) + rng.uniform(-.09, .09, len(v)),
                   v, color=GREY, s=26)
        ax.plot([j - .22, j + .22], [v.mean()] * 2, color=col, lw=3)
        ax.annotate(f"p={stats.wilcoxon(v)[1]:.2g}", (j, v.max()),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["decoded prior\n(neural)", "block label\n(task)"], fontsize=8)
    ax.set_ylabel("logistic weight on P(choose right)")
    ax.set_title("Does the decoded prior predict choice\n"
                 "beyond the true block label?", fontsize=9.5)
    ax.margins(y=0.2)
    fig.tight_layout()
    _save(fig, "fig07_choice_decoding.png")
    return a, wi, coefs


# --------------------------------------------------------------------------- #
GROUPS = {
    "frontal cortex": ["Secondary motor", "Anterior cingulate", "Prelimbic",
                       "Infralimbic", "Orbital area", "Primary motor",
                       "Frontal pole"],
    "sensory cortex": ["Primary visual", "visual area", "Visual area",
                       "somatosensory", "Auditory area", "Retrosplenial",
                       "Temporal association", "Posterior parietal"],
    "striatum / pallidum": ["Caudoputamen", "Nucleus accumbens", "Striatum",
                            "Olfactory tubercle", "pallidum", "Globus pallidus"],
    "thalamus": ["thalamus", "Thalamus", "geniculate", "Lateral posterior nucleus",
                 "Posterior complex", "Reticular nucleus", "Medial habenula"],
    "midbrain": ["Superior colliculus", "Midbrain", "Periaqueductal",
                 "Substantia nigra", "Ventral tegmental", "pretectal",
                 "Inferior colliculus", "optic tract"],
    "hippocampal formation": ["Field CA", "Dentate gyrus", "Subiculum",
                              "Entorhinal", "Ammon"],
    "amygdala": ["amygdal"],
}


def group_of(name):
    for g, keys in GROUPS.items():
        if any(k in name for k in keys):
            return g
    return "other"


def fig_regions(R):
    rows = []
    for r in R:
        for u in range(len(r["regions"])):
            rows.append((group_of(r["regions"][u]), r["aucs"][u], r["auc_p"][u],
                         r["subject"]))
    df = pd.DataFrame(rows, columns=["group", "auc", "p", "subject"])
    g = df.groupby("group").agg(n=("auc", "size"),
                                nses=("subject", "nunique"),
                                frac=("p", lambda x: (x < 0.05).mean()),
                                dev=("auc", lambda x: np.nanmean(np.abs(x - 0.5))))
    g = g[g["n"] >= 40].sort_values("frac", ascending=False)
    # Binomial CI on the fraction of significant units.
    lo, hi = [], []
    for f, n in zip(g["frac"], g["n"]):
        ci = stats.binomtest(int(round(f * n)), int(n)).proportion_ci()
        lo.append(f - ci.low)
        hi.append(ci.high - f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    ax.barh(np.arange(len(g)), g["frac"], color=BLUE,
            xerr=[lo, hi], error_kw=dict(ecolor="k", lw=1))
    ax.axvline(0.05, c=RED, ls="--", label="chance (5%)")
    ax.set_yticks(np.arange(len(g)))
    ax.set_yticklabels([f"{i}\n(n={int(n)} units, {int(s)} sessions)"
                        for i, n, s in zip(g.index, g["n"], g["nses"])], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of units whose pre-stimulus rate tracks the block")
    ax.set_title("Where the prior appears before the stimulus")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    ax.barh(np.arange(len(g)), g["dev"], color=GREY)
    ax.set_yticks(np.arange(len(g)))
    ax.set_yticklabels(g.index, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("mean |AUC - 0.5|")
    ax.set_title("Mean single-unit effect size")
    fig.tight_layout()
    _save(fig, "fig08_regions.png")
    return g
