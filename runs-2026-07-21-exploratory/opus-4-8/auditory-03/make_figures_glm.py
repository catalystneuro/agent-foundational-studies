"""Figure 6: NeMoS Poisson GLM with frequency-specific stimulus filters."""

import numpy as np

import plotting as P
from plotting import khz, plt

G = np.load("results_glm_000986.npz", allow_pickle=True)
D = np.load("results_000986.npz", allow_pickle=True)
UF = G["ufreq"]
COL = P.freq_colors(UF)
SESSION = str(G["session"])


def glm_tuning():
    """GLM-predicted evoked rate in the same 10-60 ms window used for spike counts.

    Taking the peak of the filter instead would floor the model tuning curve at
    zero, since a filter maximum is never negative, and suppressive responses
    would not be comparable to the measured ones.
    """
    intercept = np.log(G["base_rate"] * G["bin_size"])
    m = (G["kernel_t"] >= 0.010) & (G["kernel_t"] < 0.060)
    pred = np.exp(intercept[:, None, None] + G["filters"][:, m, :]).mean(1) / G["bin_size"]
    return pred - G["base_rate"][:, None]


def main():
    r2_full = 1 - G["ll_full"] / G["ll_null"]
    r2_blind = 1 - G["ll_blind"] / G["ll_null"]
    bits = (G["ll_full"] - G["ll_blind"]) / np.maximum(G["n_spikes"], 1) / np.log(2)

    # match GLM units to the window-count analysis for the same session
    sess_mask = D["session"] == SESSION
    order = {int(u): i for i, u in enumerate(D["unit_ids"][sess_mask])}
    emp = D["evoked_rate"][sess_mask]
    idx = np.array([order[int(u)] for u in G["unit_ids"]])
    emp = emp[idx]
    tuned = (D["p_anova"][sess_mask])[idx] < 0.01

    fig = plt.figure(figsize=(12, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # ---- example fitted filters: the units whose fitted tuning is most selective
    tg = glm_tuning()
    selectivity = np.where(tg.max(1) > 5, 1 - tg.min(1) / tg.max(1), -np.inf)
    show = np.argsort(np.where(tuned, selectivity, -np.inf))[::-1][:2]
    for n, i in enumerate(show):
        ax = fig.add_subplot(gs[n, 0])
        for k in range(len(UF)):
            ax.plot(G["kernel_t"] * 1000, G["filters"][i, :, k], color=COL[k], lw=1.4,
                    label=f"{khz(UF[k])} kHz")
        ax.axhline(0, color="0.7", lw=0.8, ls="--")
        ax.set(ylabel="filter (log rate)",
               title=f"unit {G['unit_ids'][i]}: fitted stimulus filters")
        if n == 0:
            ax.legend(fontsize=6.5, ncol=2)
        if n == 1:
            ax.set_xlabel("time from tone onset (ms)")

    # ---- GLM tuning vs empirical tuning, same two units
    ax = fig.add_subplot(gs[0, 1])
    for n, i in enumerate(show):
        ax.plot(np.log2(UF), tg[i], "o-", color=f"C{n}",
                label=f"unit {G['unit_ids'][i]} GLM")
        ax.plot(np.log2(UF), emp[i], "s--", color=f"C{n}", alpha=0.5, ms=4,
                label=f"unit {G['unit_ids'][i]} spike counts")
    ax.set_xticks(np.log2(UF), [khz(f) for f in UF])
    ax.set(xlabel="tone frequency (kHz)", ylabel="evoked rate (Hz)",
           title="Model-based vs measured tuning")
    ax.legend(fontsize=6.5)

    # ---- agreement across all fitted units
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(emp.ravel(), tg.ravel(), s=5, color="0.4", alpha=0.5)
    lim = np.percentile(np.abs(emp), 99.5)
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=0.8)
    r = np.corrcoef(emp.ravel(), tg.ravel())[0, 1]
    ax.set(xlim=(-lim * 0.3, lim), ylim=(-lim * 0.3, lim),
           xlabel="measured evoked rate (Hz)", ylabel="GLM predicted evoked rate (Hz)",
           title=f"All units × all tones (r = {r:.2f})")

    # ---- held-out likelihood comparison
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(r2_blind[~tuned], r2_full[~tuned], s=6, color="0.75", label="not tuned")
    ax.scatter(r2_blind[tuned], r2_full[tuned], s=6, color="C3", label="tuned")
    lo = min(r2_blind.min(), r2_full.min())
    hi = max(r2_blind.max(), r2_full.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set(xlabel="pseudo-$R^2$, frequency-blind model",
           ylabel="pseudo-$R^2$, frequency-specific model",
           title="Held-out prediction of spiking")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    ax.hist(bits, bins=40, color="0.4")
    ax.axvline(0, color="r", ls="--", lw=0.8)
    ax.set(xlabel="log-likelihood gain (bits/spike)", ylabel="# units",
           title=f"Gain from knowing the frequency\nmedian {np.median(bits):.3f} bits/spike, "
                 f"{np.mean(G['ll_full'] > G['ll_blind']):.0%} of units improve")

    fig.suptitle(f"Poisson GLM (NeMoS) of tone responses, "
                 f"{SESSION.split('/')[-1].replace('_behavior.nwb', '')}")
    fig.savefig("fig6_glm_nemos.png")
    plt.close(fig)
    print("saved fig6_glm_nemos.png")


if __name__ == "__main__":
    main()
