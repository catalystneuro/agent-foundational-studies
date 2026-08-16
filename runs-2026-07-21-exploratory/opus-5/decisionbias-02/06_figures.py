"""All figures for the pre-stimulus decision-bias analysis."""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

import ibl_common as ic

mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False,
})

C_LEFT, C_RIGHT, C_NEURAL, C_NULL = "#3B6FB6", "#C0504D", "#2E8B57", "#9a9a9a"
FIG = "."
RES = "results"
DATA = "session_data"


def all_trials():
    frames = []
    for f in sorted(glob.glob(f"{DATA}/*_trials.csv")):
        t = pd.read_csv(f)
        t["session"] = os.path.basename(f).replace("_trials.csv", "")
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Fig 1: the task produces a decision bias
# --------------------------------------------------------------------------

def fig_behaviour(tr):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    ax = axes[0]
    for blk, col, lab in [(0, C_LEFT, "left block  P(left)=0.8"),
                          (1, C_RIGHT, "right block  P(left)=0.2")]:
        g = tr[tr.block_right == blk].groupby("signed_contrast").choice_right
        m, n = g.mean(), g.count()
        se = np.sqrt(m * (1 - m) / n)
        ax.errorbar(m.index * 100, m, yerr=se, marker="o", ms=4, color=col, label=lab, lw=1.5)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("signed contrast (%)   negative = left")
    ax.set_ylabel("P(choose right)")
    ax.set_title(f"Prior blocks shift the psychometric curve\n({tr.session.nunique()} sessions, "
                 f"{len(tr):,} trials)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    low = tr[tr.abs_contrast <= 0.0625]
    per = low.groupby(["session", "block_right"]).choice_right.mean().unstack()
    bias = (per[1] - per[0]).sort_values()
    ax.bar(range(len(bias)), bias, color=[C_NEURAL if b > 0 else C_NULL for b in bias])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("session (sorted)")
    ax.set_ylabel(r"$\Delta$P(right):  right block $-$ left block")
    ax.set_title("Behavioural bias per session\n(trials with |contrast| $\\leq$ 6.25 %)")
    t, p = stats.ttest_1samp(bias.dropna(), 0)
    ax.text(0.03, 0.95, f"mean {bias.mean():.2f}\nt={t:.1f}, p={p:.1e}",
            transform=ax.transAxes, va="top", fontsize=8)

    ax = axes[2]
    s = tr[tr.session == tr.session.iloc[0]].reset_index(drop=True)
    ax.step(np.arange(len(s)), 1 - s.probability_left, where="mid", color="k", lw=1,
            label="P(stimulus on right)")
    run = s.choice_right.rolling(15, center=True, min_periods=5).mean()
    ax.plot(run, color=C_NEURAL, lw=1.5, label="P(choose right), 15-trial average")
    ax.set_xlabel("trial")
    ax.set_ylabel("probability")
    ax.set_title("Uncued block structure and the\nanimal's tracking of it (example session)")
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(f"{FIG}/fig01_task_and_behavioural_bias.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig 2: raw neural and behavioural data around stimulus onset (pynapple)
# --------------------------------------------------------------------------

def fig_raw(asset_id, session_id):
    import pynapple as nap

    nwbfile = ic.open_nwb(asset_id)
    trials = ic.valid_trials(ic.get_trials(nwbfile))
    units = ic.get_units(nwbfile)

    spikes = nap.TsGroup(
        {int(i): nap.Ts(np.asarray(nwbfile.units["spike_times"][i])) for i in units.index[:400]},
        region=units["region"].to_numpy()[:400],
    )
    stim = nap.Ts(trials["stim_on"].to_numpy())
    block = trials["block_right"].to_numpy()

    wheel = nwbfile.processing["wheel"]["WheelPosition"]
    wheel_tsd = nap.Tsd(t=np.asarray(wheel.timestamps[:]), d=np.asarray(wheel.data[:]))

    fig = plt.figure(figsize=(11, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.45, wspace=0.3)

    # (a) 20 s of raw population activity with trial events
    ax = fig.add_subplot(gs[0, :2])
    t0 = trials["stim_on"].iloc[20] - 6
    ep = nap.IntervalSet(start=t0, end=t0 + 24)
    sub = spikes.restrict(ep)
    for row, k in enumerate(list(sub.keys())[:120]):
        t = sub[k].t
        ax.plot(t, np.full_like(t, row), "|", ms=2, color="k", alpha=0.55, mew=0.5)
    for s in trials["stim_on"]:
        if ep.start[0] < s < ep.end[0]:
            ax.axvline(s, color=C_RIGHT, lw=1.2)
            ax.axvspan(s - 0.4, s, color=C_NEURAL, alpha=0.18, lw=0)
    ax.set_xlim(ep.start[0], ep.end[0])
    ax.set_ylabel("unit")
    ax.set_xlabel("time (s)")
    ax.set_title("Raw spiking, example session:  red = Gabor onset,  "
                 "green = 400 ms pre-stimulus decoding window", loc="left", fontsize=9)

    # (b) wheel position over the same stretch: the quiescence requirement means
    # the wheel is still in the decoding window
    ax = fig.add_subplot(gs[0, 2])
    w = wheel_tsd.restrict(ep)
    ax.plot(w.t, w.d - np.median(w.d), color="k", lw=1)
    for s in trials["stim_on"]:
        if ep.start[0] < s < ep.end[0]:
            ax.axvline(s, color=C_RIGHT, lw=1.2)
            ax.axvspan(s - 0.4, s, color=C_NEURAL, alpha=0.18, lw=0)
    ax.set_xlim(ep.start[0], ep.end[0])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("wheel position (rad)")
    ax.set_title("Wheel position", fontsize=9)

    # Pick an illustrative unit: the one whose pre-stimulus rate separates the
    # two blocks most strongly in this session.  This is for display only; every
    # quantitative result below uses all units and cross-validation.
    stim_on = trials["stim_on"].to_numpy()
    pre = ic.window_counts([spikes[k].t for k in spikes.keys()], stim_on, ic.PRE_WINDOW)
    tstat = np.array([
        stats.ttest_ind(pre[block == 1, j], pre[block == 0, j], equal_var=False).statistic
        for j in range(pre.shape[1])
    ])
    key = list(spikes.keys())[int(np.nanargmax(np.abs(tstat)))]

    # (c) peri-event raster for that unit, trials grouped by block
    ax = fig.add_subplot(gs[1, 0])
    row = 0
    peths = {}
    for blk, col, lab in [(0, C_LEFT, "left block"), (1, C_RIGHT, "right block")]:
        ev = nap.Ts(stim_on[block == blk])
        peth = nap.compute_perievent(spikes[key], ev, window=(-1.0, 0.5))
        peths[blk] = peth
        for k in peth.keys():
            t = peth[k].t
            ax.plot(t, np.full_like(t, row), "|", ms=2.5, color=col, mew=0.6)
            row += 1
    ax.axvline(0, color="k", lw=1)
    ax.axvspan(-0.4, 0, color=C_NEURAL, alpha=0.15, lw=0)
    ax.set_xlabel("time from Gabor onset (s)")
    ax.set_ylabel("trial (grouped by block)")
    ax.set_title(f"Example unit ({units.loc[key, 'region']})")

    # (d) its rate around onset, split by block
    ax = fig.add_subplot(gs[1, 1])
    edges = np.arange(-1.0, 0.51, 0.05)
    c = edges[:-1] + 0.025
    for blk, col, lab in [(0, C_LEFT, "left block"), (1, C_RIGHT, "right block")]:
        peth = peths[blk]
        rates = np.stack([np.histogram(peth[k].t, bins=edges)[0] / 0.05 for k in peth.keys()])
        m, se = rates.mean(0), rates.std(0) / np.sqrt(len(rates))
        ax.plot(c, m, color=col, lw=1.5, label=lab)
        ax.fill_between(c, m - se, m + se, color=col, alpha=0.25, lw=0)
    ax.axvline(0, color="k", lw=1)
    ax.axvspan(-0.4, 0, color=C_NEURAL, alpha=0.15, lw=0)
    ax.set_xlabel("time from Gabor onset (s)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("Same unit, mean rate by block")
    ax.legend(fontsize=8)

    # (e) how block-separated the pre-stimulus rate is across all units
    ax = fig.add_subplot(gs[1, 2])
    rng = np.random.default_rng(0)
    shuf = []
    for j in range(pre.shape[1]):
        p = rng.permutation(block)
        shuf.append(stats.ttest_ind(pre[p == 1, j], pre[p == 0, j], equal_var=False).statistic)
    shuf = np.array(shuf)
    bins = np.linspace(-6, 6, 41)
    ax.hist(shuf[np.isfinite(shuf)], bins=bins, color=C_NULL, alpha=0.6, density=True,
            label="trial-shuffled")
    ax.hist(tstat[np.isfinite(tstat)], bins=bins, histtype="step", color=C_NEURAL, lw=2,
            density=True, label="observed")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("t statistic, right block $-$ left block")
    ax.set_ylabel("density")
    ax.set_title("Pre-stimulus rates differ between\nblocks across the population")
    ax.legend(fontsize=8)

    fig.savefig(f"{FIG}/fig02_raw_activity_and_alignment.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig 3: time-resolved decoding of the block
# --------------------------------------------------------------------------

def fig_time_resolved():
    t = pd.read_csv(f"{RES}/time_resolved.csv")
    piv = t.pivot(index="center", columns="session", values="auc")
    null = t.pivot(index="center", columns="session", values="null_mean")
    c = piv.index.to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    for s in piv.columns:
        ax.plot(c, piv[s], color=C_NEURAL, alpha=0.22, lw=0.9)
    m = piv.mean(1)
    se = piv.std(1) / np.sqrt(piv.shape[1])
    ax.plot(c, m, color=C_NEURAL, lw=2.5, label="mean across sessions")
    ax.fill_between(c, m - se, m + se, color=C_NEURAL, alpha=0.3, lw=0)
    nm = null.mean(1)
    ax.plot(c, nm, color=C_NULL, lw=2, ls="--", label="pseudo-session null")
    ax.axvline(0, color="k", lw=1)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.axvspan(-1.2, 0, color="k", alpha=0.05, lw=0)
    ax.set_xlim(c.min(), c.max())
    ax.set_xlabel("centre of 200 ms window, relative to Gabor onset (s)")
    ax.set_ylabel("cross-validated AUC for block")
    ax.set_title("Block identity is decodable before the stimulus")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    z = t.pivot(index="center", columns="session", values="z")
    m, se = z.mean(1), z.std(1) / np.sqrt(z.shape[1])
    ax.plot(c, m, color=C_NEURAL, lw=2)
    ax.fill_between(c, m - se, m + se, color=C_NEURAL, alpha=0.3, lw=0)
    ax.axvline(0, color="k", lw=1)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.axhline(1.96, color=C_NULL, lw=0.8, ls="--")
    ax.set_xlim(c.min(), c.max())
    ax.set_xlabel("centre of 200 ms window, relative to Gabor onset (s)")
    ax.set_ylabel("z vs pseudo-session null")
    ax.set_title("Effect size relative to the null")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig03_time_resolved_block_decoding.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig 4: pre-stimulus block decoding per session and per region
# --------------------------------------------------------------------------

def fig_block_decoding():
    b = pd.read_csv(f"{RES}/block_decoding.csv")
    allpre = b[(b.region == "ALL") & (b.window == "pre")].sort_values("auc")
    early = b[(b.region == "ALL") & (b.window == "early")].set_index("session")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    ax = axes[0]
    y = np.arange(len(allpre))
    ax.barh(y, allpre.auc - 0.5, left=0.5, color=C_NEURAL, height=0.65,
            label="pre-stimulus $-$400 to 0 ms")
    ax.plot(allpre.null_mean, y, "|", color="k", ms=9, mew=1.4, label="null mean")
    for i, (_, r) in enumerate(allpre.iterrows()):
        ax.plot([r.null_mean - 1.96 * r.null_sd, r.null_mean + 1.96 * r.null_sd], [i, i],
                color="k", lw=1)
    ax.axvline(0.5, color="k", lw=0.8, ls=":")
    ax.set_yticks([])
    ax.set_xlabel("cross-validated AUC")
    ax.set_ylabel("session")
    sig = (allpre.p < 0.05).sum()
    grp = pd.read_csv(f"{RES}/group_tests.csv").set_index("analysis").loc["block_pre"]
    ax.set_title(f"Block decoding, all units.  {sig}/{len(allpre)} sessions p<0.05;\n"
                 f"group mean {grp.mean_auc:.3f} vs null {grp.null_mean:.3f}, p={grp.p:.3f}")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    ax = axes[1]
    ax.scatter(early.loc[allpre.session].auc, allpre.auc, c=C_NEURAL, s=28)
    lims = [0.4, max(allpre.auc.max(), early.auc.max()) + 0.03]
    ax.plot(lims, lims, color="k", lw=0.8, ls=":")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("AUC, $-$1000 to $-$600 ms")
    ax.set_ylabel("AUC, $-$400 to 0 ms")
    ax.set_title("The signal is not confined to the\nlast 400 ms before onset")

    ax = axes[2]
    reg = b[(b.region != "ALL")].groupby("region").agg(
        auc=("auc", "mean"), n=("auc", "size"),
        Z=("z", lambda x: np.nansum(x) / np.sqrt(np.isfinite(x).sum())))
    reg = reg[reg.n >= 3].sort_values("auc")
    ax.barh(np.arange(len(reg)), reg.auc - 0.5, left=0.5, color=C_NEURAL, height=0.65)
    ax.set_yticks(np.arange(len(reg)))
    ax.set_yticklabels([f"{i}  (n={int(r.n)})" for i, r in reg.iterrows()], fontsize=8)
    lo = min(0.47, reg.auc.min() - 0.01)
    hi = reg.auc.max() + 0.055
    for i, r in enumerate(reg.itertuples()):
        ax.text(hi - 0.004, i, f"Z={r.Z:.1f}", va="center", ha="right", fontsize=7)
    ax.axvline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("mean cross-validated AUC")
    ax.set_title("Pre-stimulus block decoding by region\n(session $\\times$ region populations)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig04_prestimulus_block_decoding.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig 5: decoding the upcoming choice on low-evidence trials
# --------------------------------------------------------------------------

def fig_choice_decoding():
    c = pd.read_csv(f"{RES}/choice_decoding.csv").sort_values("auc")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    ax = axes[0]
    y = np.arange(len(c))
    ax.barh(y, c.auc - 0.5, left=0.5, color=C_RIGHT, height=0.65)
    ax.plot(c.null_mean, y, "|", color="k", ms=9, mew=1.4, label="shift null mean")
    ax.axvline(0.5, color="k", lw=0.8, ls=":")
    ax.set_yticks([])
    ax.set_ylabel("session")
    ax.set_xlabel("cross-validated AUC")
    ax.set_title("Upcoming choice on |contrast| $\\leq$ 6.25 %\ndecoded from pre-stimulus activity")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    ax.hist(c.z, bins=np.arange(-3, 9, 0.75), color=C_RIGHT, alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(1.96, color=C_NULL, ls="--", lw=1)
    ax.set_xlabel("z vs circular-shift null")
    ax.set_ylabel("sessions")
    Z = c.z.sum() / np.sqrt(len(c))
    ax.set_title(f"Combined across sessions:\nStouffer Z = {Z:.1f}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig05_upcoming_choice_decoding.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig 6: controls
# --------------------------------------------------------------------------

def fig_controls(tr):
    ctl = pd.read_csv(f"{RES}/controls.csv")
    neg = pd.read_csv(f"{RES}/negative_control.csv")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))

    ax = axes[0]
    cols = ["auc_movement", "auc_history", "auc_neural", "auc_history_neural"]
    labs = ["movement +\narousal", "behavioural\nhistory", "pre-stimulus\nspiking",
            "history +\nspiking"]
    for i, cname in enumerate(cols):
        v = ctl[cname]
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).normal(0, 0.06, len(v)),
                   v, s=18, color=C_NEURAL, alpha=0.7)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color="k", lw=2)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("cross-validated AUC for block")
    ax.set_title("What predicts the block from the\npre-stimulus window?")

    ax = axes[1]
    ax.scatter(ctl.delta_null_mean, ctl.delta, color=C_NEURAL, s=28)
    lim = [min(ctl.delta.min(), ctl.delta_null_mean.min()) - 0.01,
           max(ctl.delta.max(), ctl.delta_null_mean.max()) + 0.01]
    ax.plot(lim, lim, color="k", ls=":", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$\Delta$AUC expected under the null")
    ax.set_ylabel(r"$\Delta$AUC from adding spiking")
    Z = ctl.z_delta.sum() / np.sqrt(len(ctl))
    ax.set_title(f"Neural read-out added to a behavioural-\nhistory model (combined Z={Z:.1f})")

    ax = axes[2]
    for v, lab, col in [("wheel_abs_vel", "wheel speed", C_LEFT),
                        ("motion_energy", "motion energy", C_RIGHT),
                        ("pupil", "pupil", C_NEURAL)]:
        g = tr.groupby(["session", "block_right"])[v].mean().unstack()
        d = ((g[1] - g[0]) / tr.groupby("session")[v].std()).dropna()
        ax.scatter(np.full(len(d), lab), d, s=18, color=col, alpha=0.75)
        ax.plot([lab, lab], [d.mean()] * 2, marker="_", ms=26, color="k", mew=2, ls="none")
        t, p = stats.ttest_1samp(d, 0)
        ax.annotate(f"p={p:.2f}", (lab, ax.get_ylim()[1]), fontsize=7, ha="center",
                    va="bottom", annotation_clip=False)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("right block $-$ left block (SD units)")
    ax.set_title("Pre-stimulus movement and arousal\ndo not differ between blocks")

    ax = axes[3]
    b = pd.read_csv(f"{RES}/block_decoding.csv")
    allpre = b[(b.region == "ALL") & (b.window == "pre")]
    for i, (v, lab, col) in enumerate([(allpre.auc, "block\n(the bias)", C_NEURAL),
                                       (neg.auc, "upcoming contrast\n(unknowable)", C_NULL)]):
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).normal(0, 0.06, len(v)),
                   v, s=20, color=col, alpha=0.8)
        ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color="k", lw=2)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["block\n(the bias)", "upcoming contrast\n(unknowable)"],
                                              fontsize=8)
    ax.set_ylabel("cross-validated AUC")
    ax.set_title(f"Negative control: contrast is drawn\nindependently and is not decodable\n"
                 f"(mean AUC {neg.auc.mean():.3f})")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig06_controls.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig 7: the pre-stimulus read-out tracks the bias trial by trial
# --------------------------------------------------------------------------

def fig_trialwise(psycho):
    tw = pd.read_csv(f"{RES}/trialwise_bias.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))

    ax = axes[0]
    o = tw.sort_values("beta_neural")
    y = np.arange(len(o))
    ax.errorbar(o.beta_neural, y, xerr=1.96 * o.se_neural, fmt="o", ms=4,
                color=C_NEURAL, lw=1)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks([])
    ax.set_ylabel("session")
    ax.set_xlabel("logistic weight on pre-stimulus neural read-out")
    t, p = stats.ttest_1samp(tw.beta_neural, 0)
    ax.set_title("Choice on low-evidence trials, given\nblock, previous choice and contrast\n"
                 f"t={t:.2f}, p={p:.3g}")

    ax = axes[1]
    for lab, col in [("low", C_LEFT), ("high", C_RIGHT)]:
        g = psycho[psycho.neural_group == lab].groupby("signed_contrast").choice_right
        m, n = g.mean(), g.count()
        ax.errorbar(m.index * 100, m, yerr=np.sqrt(m * (1 - m) / n), marker="o", ms=4,
                    color=col, lw=1.5,
                    label=f"pre-stimulus read-out: {lab}est tercile")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xlabel("signed contrast (%)")
    ax.set_ylabel("P(choose right)")
    ax.set_title("Psychometric curve split by the\npre-stimulus neural read-out\n(block identity held fixed)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig07_trialwise_bias_readout.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig 8: single-neuron encoding (NeMoS GLM)
# --------------------------------------------------------------------------

def fig_glm():
    real = pd.read_csv(f"{RES}/glm_block_coefs.csv")
    null = pd.read_csv(f"{RES}/glm_block_coefs_null.csv")
    thr = np.percentile(np.abs(null.beta), 97.5)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    ax = axes[0]
    bins = np.linspace(-0.5, 0.5, 61)
    ax.hist(null.beta, bins=bins, density=True, color=C_NULL, alpha=0.6,
            label="pseudo-session null")
    ax.hist(real.beta, bins=bins, density=True, histtype="step", color=C_NEURAL, lw=2,
            label="observed")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("GLM weight on block identity (drift- and history-corrected)")
    ax.set_ylabel("density")
    frac = (np.abs(real.beta) > thr).mean()
    ax.set_title(f"Pre-stimulus rate depends on the block\nin {frac:.0%} of units "
                 "(2.5 % expected)")
    ax.legend(fontsize=8)

    ax = axes[1]
    real["sig"] = np.abs(real.beta) > thr
    g = real.groupby("region").sig.agg(["mean", "count"])
    g = g[g["count"] >= 50].sort_values("mean")
    ax.barh(np.arange(len(g)), g["mean"], color=C_NEURAL, height=0.65)
    ax.axvline(0.025, color=C_NULL, ls="--", lw=1.2, label="null expectation")
    ax.set_yticks(np.arange(len(g)))
    ax.set_yticklabels([f"{i}  (n={int(r['count'])})" for i, r in g.iterrows()], fontsize=8)
    ax.set_xlabel("fraction of units with block-dependent pre-stimulus rate")
    ax.set_title("Only a few regions exceed the null\nexpectation, and not by much")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig08_glm_single_unit_encoding.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig 9: brain-behaviour correspondence
# --------------------------------------------------------------------------

def fig_brain_behaviour(tr):
    b = pd.read_csv(f"{RES}/block_decoding.csv")
    allpre = b[(b.region == "ALL") & (b.window == "pre")].set_index("session")
    low = tr[tr.abs_contrast <= 0.0625]
    per = low.groupby(["session", "block_right"]).choice_right.mean().unstack()
    bias = (per[1] - per[0]).reindex(allpre.index)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(bias, allpre.auc, s=32, color=C_NEURAL)
    ok = bias.notna() & allpre.auc.notna()
    r, p = stats.pearsonr(bias[ok], allpre.auc[ok])
    m, c = np.polyfit(bias[ok], allpre.auc[ok], 1)
    xs = np.linspace(bias.min(), bias.max(), 10)
    ax.plot(xs, m * xs + c, color="k", lw=1.2, ls="--")
    ax.set_xlabel(r"behavioural bias, $\Delta$P(right)")
    ax.set_ylabel("pre-stimulus block decoding AUC")
    ax.set_title(f"Sessions with a stronger behavioural bias\nhave a stronger pre-stimulus "
                 f"signal\nr={r:.2f}, p={p:.3f}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig09_brain_behaviour_correlation.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    tr = all_trials()
    fig_behaviour(tr)
    sel = pd.read_csv("selected_sessions.csv")
    fig_raw(sel.asset_id.iloc[10], sel.session_id.iloc[10])
    fig_time_resolved()
    fig_block_decoding()
    fig_choice_decoding()
    fig_controls(tr)
    fig_trialwise(pd.read_csv(f"{RES}/psycho_by_readout.csv"))
    fig_glm()
    fig_brain_behaviour(tr)
    print("figures written")
