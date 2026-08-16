"""Figure 5: split-half validation of tuning and population decoding of tone identity."""

import numpy as np

import plotting as P
from plotting import khz, plt

C = np.load("results_crossval_000986.npz", allow_pickle=True)
D = np.load("results_000986.npz", allow_pickle=True)
UF = C["ufreq"]
n_f = len(UF)
TUNED = D["p_anova"] < 0.01


def main():
    fig = plt.figure(figsize=(12, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # ---- split-half tuning curve, BF from half A, response from half B
    ax = fig.add_subplot(gs[0, 0])
    oct_axis = np.arange(-(n_f - 1), n_f) * 1.0
    for lbl, sel, col in [("all units", slice(None), "0.5"),
                          ("frequency-tuned", TUNED, "C3")]:
        cv = C["cv_curve"][sel]
        m = np.nanmean(cv, 0)
        s = np.nanstd(cv, 0) / np.sqrt(np.sum(~np.isnan(cv), 0))
        n = cv.shape[0]
        ax.fill_between(oct_axis, m - s, m + s, color=col, alpha=0.3)
        ax.plot(oct_axis, m, "o-", color=col, ms=4, label=f"{lbl} (n={n})")
    ax.axhline(0, color="0.7", lw=0.8, ls="--")
    ax.set(xlabel="octaves from best frequency\n(BF estimated on held-out trials)",
           ylabel="evoked rate (Hz)", title="Split-half tuning curve")
    ax.legend(fontsize=7.5)

    # ---- best frequency agreement between halves
    ax = fig.add_subplot(gs[0, 1])
    conf = np.zeros((n_f, n_f))
    m = TUNED
    for a, b in zip(C["bf_a"][m], C["bf_b"][m]):
        conf[a, b] += 1
    conf = conf / conf.sum(1, keepdims=True)
    im = ax.imshow(conf, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(n_f), [khz(f) for f in UF])
    ax.set_yticks(range(n_f), [khz(f) for f in UF])
    ax.set(xlabel="BF, second half (kHz)", ylabel="BF, first half (kHz)",
           title=f"BF reproducibility\n{np.mean(C['bf_a'][m] == C['bf_b'][m]):.0%} identical "
                 f"(chance {1/n_f:.0%})")
    plt.colorbar(im, ax=ax, label="fraction")

    # ---- decoding confusion matrix
    ax = fig.add_subplot(gs[0, 2])
    conf = C["conf"].mean(0)
    im = ax.imshow(conf, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(n_f), [khz(f) for f in UF])
    ax.set_yticks(range(n_f), [khz(f) for f in UF])
    ax.set(xlabel="decoded tone (kHz)", ylabel="presented tone (kHz)",
           title=f"Single-trial decoding\nmean accuracy {C['acc'].mean():.0%} "
                 f"(chance {1/n_f:.0%})")
    plt.colorbar(im, ax=ax, label="fraction of trials")
    for i in range(n_f):
        for j in range(n_f):
            ax.text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="w" if conf[i, j] < 0.6 else "k")

    # ---- accuracy per session
    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(C["n_units"])
    ax.plot(C["n_units"][order], C["acc"][order], "o", color="C0")
    ax.axhline(1 / n_f, color="r", ls="--", lw=0.8, label="chance")
    ax.set(xlabel="# simultaneously recorded units", ylabel="decoding accuracy",
           ylim=(0, 1), title="Decoding accuracy per session")
    ax.legend(fontsize=7)

    # ---- accuracy vs population size
    ax = fig.add_subplot(gs[1, 1])
    ax.errorbar(C["sizes"], C["acc_by_size"], yerr=C["acc_by_size_sem"],
                marker="o", color="C0", capsize=3)
    ax.axhline(1 / n_f, color="r", ls="--", lw=0.8, label="chance")
    ax.set_xscale("log")
    ax.set(xlabel="# units in decoder", ylabel="decoding accuracy", ylim=(0, 1),
           title="Accuracy vs population size")
    ax.legend(fontsize=7)
    for x, y, n in zip(C["sizes"], C["acc_by_size"], C["n_sessions_by_size"]):
        ax.text(x, min(y + 0.06, 0.97), f"{n} ses", ha="center", fontsize=5.5, color="0.4")

    # ---- single-unit information: how well one unit alone identifies the tone
    ax = fig.add_subplot(gs[1, 2])
    d = C["cv_curve"]
    peak = np.nanmax(d, 1)
    flank = np.nanmean(np.where(np.abs(np.arange(-(n_f - 1), n_f))[None, :] >= 2, d, np.nan), 1)
    sel = np.isfinite(peak) & np.isfinite(flank)
    ax.scatter(peak[sel & ~TUNED], flank[sel & ~TUNED], s=4, color="0.75",
               label="not tuned")
    ax.scatter(peak[sel & TUNED], flank[sel & TUNED], s=4, color="C3", label="tuned")
    lim = np.nanpercentile(np.abs(peak[sel]), 99.5)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.set(xlim=(-lim * 0.2, lim), ylim=(-lim * 0.4, lim),
           xlabel="evoked rate at BF (Hz)",
           ylabel="evoked rate ≥ 2 octaves away (Hz)",
           title="Response at BF vs off BF\n(both from held-out trials)")
    ax.legend(fontsize=7)

    fig.savefig("fig5_crossvalidation_decoding.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
    print(f"split-half BF agreement (tuned units): "
          f"{np.mean(C['bf_a'][TUNED] == C['bf_b'][TUNED]):.3f}")
    print(f"decoding accuracy: {C['acc'].mean():.3f} +/- {C['acc'].std():.3f}")
    print("saved fig5_crossvalidation_decoding.png")
