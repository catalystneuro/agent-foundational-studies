"""Figure: the block-shift null for instantaneous-velocity-direction tuning."""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CACHE = "./cache"


def plot_shuffle_control(obs_maze, null_maze, p_maze, obs_rtt, null_rtt, p_rtt,
                         fname="fig07_shuffle_control.png"):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    panels = [("MC_Maze (Jenkins, 115 units)", obs_maze, null_maze, p_maze, "#2b6cb0"),
              ("MC_RTT (Indy, 94 units)", obs_rtt, null_rtt, p_rtt, "#c05621")]

    for col, (title, obs, null, p_emp, colr) in enumerate(panels):
        r2 = obs["r2"].values
        order = np.argsort(r2)
        ax = axes[0, col]
        hi = np.percentile(null, 99.9, axis=0)[order]
        lo = null.min(axis=0)[order]
        x = np.arange(len(r2))
        ax.fill_between(x, lo, hi, color="0.7", label="block-shift null (min to 99.9th pct)")
        ax.plot(x, r2[order], ".", ms=5, color=colr, label="observed")
        sig = p_emp[order] < 0.05
        ax.plot(x[~sig], r2[order][~sig], "x", ms=6, color="0.35", label="not significant")
        ax.set_yscale("log")
        ax.set_xlabel("unit (sorted by observed $R^2$)")
        ax.set_ylabel("cosine fit $R^2$")
        ax.set_title(title, pad=8)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

        # how far above its own null does each unit sit?
        ratio = r2 / np.percentile(null, 99.9, axis=0)
        ax = axes[1, col]
        bins = np.logspace(np.log10(min(ratio.min(), 0.5) * 0.8),
                           np.log10(ratio.max() * 1.3), 34)
        ax.hist(ratio, bins=bins, color=colr, alpha=0.85)
        ax.axvline(1.0, color="k", ls="--", lw=1.5)
        ax.text(1.15, ax.get_ylim()[1] * 0.92, "null 99.9th pct", fontsize=8, va="top")
        ax.set_xscale("log")
        ax.set_xlabel("observed $R^2$ / that unit's null 99.9th percentile")
        ax.set_ylabel("units")
        n_nom = (obs["p"] < 0.01).sum()
        med = np.median(ratio)
        ax.set_title("nominal F test: %d/%d   block-shift null: %d/%d   (median %.0fx null)"
                     % (n_nom, len(obs), (p_emp < 0.05).sum(), len(obs), med), fontsize=10, pad=8)

    fig.suptitle("Velocity-direction tuning survives a null that preserves the autocorrelation "
                 "of both signals", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


if __name__ == "__main__":
    with open(f"{CACHE}/shuffle_control.pkl", "rb") as fh:
        d = pickle.load(fh)
    print("wrote", plot_shuffle_control(d["obs_maze"], d["null_maze"], d["p_maze"],
                                        d["obs_rtt"], d["null_rtt"], d["p_rtt"]))
