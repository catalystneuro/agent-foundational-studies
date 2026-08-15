"""Replay figures: example decoded trajectories + population statistics."""
import numpy as np
import matplotlib.pyplot as plt

res = np.load("replay_results.npz", allow_pickle=True)
ev = np.load("replay_events.npz", allow_pickle=True)
events = list(ev["events"])
cpos = res["cpos"]
rr, pv, nact, null_r = res["r"], res["pval"], res["n_active"], res["null_r"]
n_sig = int((pv < 0.05).sum())

# ---------- Figure 1: example replay events ----------
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for k, ax in enumerate(axes.flat):
    if k >= len(events):
        ax.axis("off"); continue
    e = events[k]
    post = e["post"]              # (n_bins, n_pos)
    tc = (e["tc"] - e["t0"]) * 1000   # ms from window start
    extent = [tc[0], tc[-1], cpos[0], cpos[-1]]
    ax.imshow(post.T, aspect="auto", origin="lower", extent=extent,
              cmap="hot", vmin=0, vmax=np.percentile(post, 99))
    # MAP estimate + fitted line
    mapx = cpos[np.argmax(post, 1)]
    ax.plot(tc, mapx, "o", color="cyan", ms=4, alpha=0.8)
    A = np.polyfit(tc, mapx, 1)
    ax.plot(tc, np.polyval(A, tc), "-", color="white", lw=1.5)
    ax.set_title(f"r={e['r']:.2f}, p={e['pval']:.3f}\n{e['n_active']} cells",
                 fontsize=9)
    if k % 4 == 0:
        ax.set_ylabel("decoded position (m)")
    if k >= 4:
        ax.set_xlabel("time in ripple (ms)")
fig.suptitle("Hippocampal replay: Bayesian-decoded position sweeps during POST-sleep ripples",
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig("fig_replay_events.png", dpi=130, bbox_inches="tight")
print("saved fig_replay_events.png")

# ---------- Figure 2: population replay statistics ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

# (A) observed vs shuffle |r| distributions
ax[0].hist(np.abs(rr), bins=30, density=True, alpha=0.7, color="C3", label="observed ripples")
ax[0].hist(np.abs(null_r), bins=30, density=True, alpha=0.5, color="grey", label="cell-ID shuffle")
ax[0].axvline(np.median(np.abs(rr)), color="C3", ls="--", lw=1)
ax[0].axvline(np.median(np.abs(null_r)), color="k", ls="--", lw=1)
ax[0].set(xlabel="|weighted corr| (trajectory linearity)", ylabel="density",
          title="A. Replay sequence score vs shuffle")
ax[0].legend(fontsize=9)

# (B) per-event p-value distribution
ax[1].hist(pv, bins=20, color="C0", alpha=0.8)
ax[1].axvline(0.05, color="C3", ls="--", label="p=0.05")
frac = 100 * n_sig / len(pv)
ax[1].set(xlabel="shuffle p-value", ylabel="ripple count",
          title=f"B. Significant replay: {n_sig}/{len(pv)} ({frac:.0f}%)")
ax[1].legend(fontsize=9)

# (C) |r| vs number of active cells
sig = pv < 0.05
ax[2].scatter(nact[~sig], np.abs(rr)[~sig], s=8, color="grey", alpha=0.4, label="n.s.")
ax[2].scatter(nact[sig], np.abs(rr)[sig], s=10, color="C3", alpha=0.6, label="p<0.05")
ax[2].set(xlabel="active place cells in ripple", ylabel="|weighted corr|",
          title="C. Replay strength vs recruitment")
ax[2].legend(fontsize=9)

# KS-like effect statement
from scipy.stats import mannwhitneyu
u, pu = mannwhitneyu(np.abs(rr), np.abs(null_r), alternative="greater")
fig.suptitle(f"Population replay statistics — observed |r| > shuffle "
             f"(Mann-Whitney p={pu:.1e})", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("fig_replay_stats.png", dpi=130, bbox_inches="tight")
print("saved fig_replay_stats.png")
print(f"observed median |r| {np.median(np.abs(rr)):.3f} vs shuffle {np.median(np.abs(null_r)):.3f}; MWU p={pu:.2e}")
