"""Generate all figures from results.pkl."""
import pickle, glob, os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, RED, GREY = "#2c6fbb", "#c0392b", "#7f8c8d"

# aggregate per-session results (res_0.pkl, res_1.pkl, ...) in index order
files = sorted(glob.glob("res_*.pkl"), key=lambda f: int(f.split("_")[1].split(".")[0]))
results = [pickle.load(open(f, "rb")) for f in files]
pickle.dump(results, open("results.pkl", "wb"))
print(f"loaded {len(results)} sessions")
r0 = results[0]  # prototype session for detail panels


def sess_label(r):
    return r["path"].split("/")[-1].split("_ses-")[0].replace("sub-", "")


# ---------------------------------------------------------------------------
# Figure 1: behavioral decision bias (psychometric shift by block)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
ax = axes[0]
c = r0["contrasts"]
ax.plot(c, r0["psych"][0.8], "-o", color=BLUE, label="left block (P$_L$=0.8)")
ax.plot(c, r0["psych"][0.2], "-o", color=RED, label="right block (P$_L$=0.2)")
ax.axhline(0.5, ls=":", color=GREY); ax.axvline(0, ls=":", color=GREY)
ax.set_xlabel("signed contrast (%)  (neg = left stim)")
ax.set_ylabel("P(choose right)")
ax.set_title(f"Choice is biased by the block prior\n{sess_label(r0)}")
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
labels = [sess_label(r) for r in results]
b08 = [r["zero_bias"][0.8] for r in results]
b02 = [r["zero_bias"][0.2] for r in results]
x = np.arange(len(results))
ax.bar(x - 0.19, b08, 0.36, color=BLUE, label="left block")
ax.bar(x + 0.19, b02, 0.36, color=RED, label="right block")
ax.axhline(0.5, ls=":", color=GREY)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
ax.set_ylabel("P(choose right) at 0% contrast")
ax.set_title("Prior-driven bias on zero-contrast trials\n(choice reflects bias, not stimulus)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig("fig1_behavioral_bias.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: pre-stimulus population activity structured by upcoming block
# Reload session-0 trials (cheap, no spike data) to recover per-trial block label.
# ---------------------------------------------------------------------------
import json, pipeline as P
sess0 = json.load(open("sessions.json"))[0]
nwbfile0, _ = P.load_session(sess0["url"])
df0 = P.trials_frame(nwbfile0)
df0 = df0[~np.isnan(df0["gabor_stimulus_onset_time"].values)].reset_index(drop=True)
pl0 = df0["probability_left"].values
X = r0["X_prestim"]
assert len(pl0) == X.shape[0]
z = (X - X.mean(0)) / (X.std(0) + 1e-9)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
# (a) trial-ordered z heatmap with block bands overlaid
ax = axes[0]
im = ax.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
# shade left-block (0.8) trial ranges lightly along the top
top = z.shape[1]
for i in range(len(pl0)):
    if pl0[i] == 0.8:
        ax.axvspan(i - 0.5, i + 0.5, ymin=0.985, ymax=1.0, color=BLUE, lw=0)
    elif pl0[i] == 0.2:
        ax.axvspan(i - 0.5, i + 0.5, ymin=0.985, ymax=1.0, color=RED, lw=0)
ax.set_xlabel("trial (time order)"); ax.set_ylabel("unit")
ax.set_title(f"Pre-stimulus spike counts (z / unit)\n{sess_label(r0)}  bar: blue=left, red=right block")
plt.colorbar(im, ax=ax, fraction=0.046, label="z")

# (b) per-unit mean pre-stim z, left block vs right block; sort by difference
ax = axes[1]
m08 = z[pl0 == 0.8].mean(0)
m02 = z[pl0 == 0.2].mean(0)
order = np.argsort(m08 - m02)
ax.plot(m08[order], np.arange(len(order)), ".", color=BLUE, ms=4, label="left block (0.8)")
ax.plot(m02[order], np.arange(len(order)), ".", color=RED, ms=4, label="right block (0.2)")
ax.axvline(0, ls=":", color="k")
ax.set_xlabel("mean pre-stimulus activity (z)")
ax.set_ylabel("unit (sorted by block difference)")
ax.set_title("Many units shift their pre-stimulus rate\nwith the upcoming block")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig("fig2_prestim_activity.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: block-prior decoding vs drift-aware null (per session)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
ax = axes[0]
nb = r0["block"]
ax.hist(nb["null"], bins=25, color=GREY, alpha=0.8, edgecolor="k",
        label="circular-shift null")
ax.axvline(nb["auc"], color=RED, lw=2.5,
           label=f"real AUC={nb['auc']:.3f}\n(p={nb['p']:.3f})")
ax.axvline(0.5, ls=":", color="k", label="chance")
ax.set_xlabel("cross-validated AUC"); ax.set_ylabel("# shuffles")
ax.set_title(f"Pre-stimulus block decoding vs drift null\n{sess_label(r0)}")
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
x = np.arange(len(results))
real = [r["block"]["auc"] for r in results]
nmean = [r["block"]["null"].mean() for r in results]
np95 = [np.percentile(r["block"]["null"], 95) for r in results]
ax.bar(x, real, 0.55, color=BLUE, label="real (pre-stim)")
ax.plot(x, nmean, "o", color="k", label="null mean")
ax.vlines(x, nmean, np95, color="k")
for i, r in enumerate(results):
    star = "*" if r["block"]["auc"] > np.percentile(r["block"]["null"], 95) else ""
    ax.text(i, real[i] + 0.01, star, ha="center", fontsize=14)
ax.axhline(0.5, ls=":", color=GREY)
ax.set_xticks(x); ax.set_xticklabels([sess_label(r) for r in results], rotation=35, ha="right", fontsize=7)
ax.set_ylabel("AUC"); ax.set_ylim(0.45, max(real) + 0.08)
ax.set_title("Block decoding across sessions\n(bar > null 95th pct marked *)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("fig3_block_decoding.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4: decoding time course around stimulus onset
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
wins = r0["timecourse"]["windows"]
centers = np.array([(a + b) / 2 for a, b in wins])
# average across sessions
auc_stack = np.stack([r["timecourse"]["auc"] for r in results])
null_stack = np.stack([r["timecourse"]["null_mean"] for r in results])
ax.plot(centers, auc_stack.mean(0), "-o", color=BLUE, lw=2, label="real (mean of 5 sessions)")
ax.fill_between(centers, auc_stack.mean(0) - auc_stack.std(0) / np.sqrt(len(results)),
                auc_stack.mean(0) + auc_stack.std(0) / np.sqrt(len(results)),
                color=BLUE, alpha=0.2)
ax.plot(centers, null_stack.mean(0), "-o", color=GREY, lw=2, label="drift null (mean)")
ax.axvline(0, color=RED, ls="--", lw=1.5, label="stimulus onset")
ax.axhline(0.5, ls=":", color="k")
ax.set_xlabel("time relative to stimulus onset (s)")
ax.set_ylabel("block-decoding AUC")
ax.set_title("Block prior is decodable BEFORE stimulus onset\nand rises further after (sliding 200 ms windows)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig("fig4_decoding_timecourse.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5: upcoming-choice decoding on low-contrast (bias-driven) trials
# ---------------------------------------------------------------------------
have = [r for r in results if r.get("choice")]
fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(have))
real = [r["choice"]["auc"] for r in have]
nmean = [r["choice"]["null"].mean() for r in have]
np95 = [np.percentile(r["choice"]["null"], 95) for r in have]
ax.bar(x, real, 0.55, color="#8e44ad", label="real (pre-stim)")
ax.plot(x, nmean, "o", color="k", label="shuffle null mean")
ax.vlines(x, nmean, np95, color="k")
for i, r in enumerate(have):
    star = "*" if r["choice"]["auc"] > np.percentile(r["choice"]["null"], 95) else ""
    ax.text(i, real[i] + 0.01, star, ha="center", fontsize=14)
ax.axhline(0.5, ls=":", color=GREY)
ax.set_xticks(x); ax.set_xticklabels([sess_label(r) for r in have], rotation=35, ha="right", fontsize=7)
ax.set_ylabel("AUC (predict upcoming choice)")
ax.set_ylim(0.45, max(real) + 0.08)
ax.set_title("Upcoming choice on low-contrast trials decoded from\npre-stimulus activity (choice here is driven by bias)")
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig("fig5_choice_decoding.png", bbox_inches="tight")
plt.close(fig)

print("figures written")
for r in results:
    print(sess_label(r), "block AUC %.3f p=%.3f" % (r["block"]["auc"], r["block"]["p"]),
          ("choice AUC %.3f p=%.3f" % (r["choice"]["auc"], r["choice"]["p"])) if r.get("choice") else "no-choice")
