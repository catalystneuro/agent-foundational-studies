"""Step 7: replay figures — example decoded trajectories + population statistics."""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

with open("replay.pkl", "rb") as f:
    rp = pickle.load(f)
with open("place_fields.pkl", "rb") as f:
    pf = pickle.load(f)
centers = pf["centers"]
events = rp["events"]

pre = [o for o in events if o["epoch"] == "PRE"]
post = [o for o in events if o["epoch"] == "POST"]
sig_post = sorted([o for o in post if o["pval"] < 0.05], key=lambda o: -abs(o["score"]))

fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.1, 1], hspace=0.5, wspace=0.35)

# --- top two rows: 6 example decoded events (3 forward, 3 reverse) ---
fwd = [o for o in sig_post if o["score"] > 0][:3]
rev = [o for o in sig_post if o["score"] < 0][:3]
for col, o in enumerate(fwd):
    ax = fig.add_subplot(gs[0, col])
    P = o["post_prob"]
    T = P.shape[0]
    ax.imshow(P.T, aspect="auto", origin="lower", cmap="hot",
              extent=[0, T * 20, 0, 1.6])
    # weighted-mean trajectory
    wmean = (P * centers[None, :]).sum(1) / P.sum(1)
    ax.plot(np.arange(T) * 20 + 10, wmean, color="cyan", lw=1.2)
    ax.set_title(f"forward, score={o['score']:.2f}, p={o['pval']:.3f}", fontsize=8)
    if col == 0:
        ax.set_ylabel("decoded position (m)")
    ax.set_xlabel("time in SWR (ms)")
for col, o in enumerate(rev):
    ax = fig.add_subplot(gs[1, col])
    P = o["post_prob"]
    T = P.shape[0]
    ax.imshow(P.T, aspect="auto", origin="lower", cmap="hot",
              extent=[0, T * 20, 0, 1.6])
    wmean = (P * centers[None, :]).sum(1) / P.sum(1)
    ax.plot(np.arange(T) * 20 + 10, wmean, color="cyan", lw=1.2)
    ax.set_title(f"reverse, score={o['score']:.2f}, p={o['pval']:.3f}", fontsize=8)
    if col == 0:
        ax.set_ylabel("decoded position (m)")
    ax.set_xlabel("time in SWR (ms)")

# --- bottom left: score distributions ---
ax = fig.add_subplot(gs[2, 0])
bins = np.linspace(0, 1, 31)
ax.hist(np.abs([o["score"] for o in pre]), bins=bins, alpha=0.7, density=True,
        label=f"PRE (n={len(pre)})", color="0.5")
ax.hist(np.abs([o["score"] for o in post]), bins=bins, alpha=0.7, density=True,
        label=f"POST (n={len(post)})", color="tab:red")
ax.set_xlabel("|weighted correlation|")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"|score| distributions (MW p={rp['p_mw']:.1e})")

# --- bottom middle: significant fractions ---
ax = fig.add_subplot(gs[2, 1])
fracs = [100 * np.mean([o["pval"] < 0.05 for o in g]) for g in (pre, post)]
bars = ax.bar(["PRE", "POST"], fracs, color=["0.5", "tab:red"], width=0.55)
ax.axhline(5, color="k", ls="--", lw=0.8, label="chance (5%)")
for b, f, g in zip(bars, fracs, (pre, post)):
    n_sig = sum(o["pval"] < 0.05 for o in g)
    ax.text(b.get_x() + b.get_width() / 2, f + 0.4, f"{n_sig}/{len(g)}",
            ha="center", fontsize=8)
ax.set_ylabel("% events significant (p<0.05)")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"Replay above chance after maze (Fisher p={rp['p_fisher']:.1e})", fontsize=8)
ax.set_ylim(0, max(fracs) * 1.25)

# --- bottom right: forward/reverse among significant POST ---
ax = fig.add_subplot(gs[2, 2])
n_fwd = sum(1 for o in sig_post if o["score"] > 0)
n_rev = sum(1 for o in sig_post if o["score"] < 0)
ax.bar(["forward", "reverse"], [n_fwd, n_rev], color=["tab:blue", "tab:orange"],
       width=0.55)
ax.set_ylabel("significant POST events")
ax.set_title(f"Trajectory direction (binomial p={rp['p_binom']:.2f})", fontsize=8)
fig.savefig("fig4_replay.png", dpi=150)
plt.close(fig)
print("fig4 saved")
