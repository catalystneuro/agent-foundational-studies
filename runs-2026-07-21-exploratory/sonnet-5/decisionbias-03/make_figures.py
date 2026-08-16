"""Generate all figures for the pre-stimulus decision-bias decoding analysis."""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

from ibl_bias_lib import (
    ASSET_IDS,
    open_session,
    load_trials,
    load_good_units_spiketrain,
    build_prestim_features,
)

with open("results.pkl", "rb") as fh:
    results = pickle.load(fh)

SESSIONS = list(ASSET_IDS.keys())
COLORS = plt.cm.tab10(np.linspace(0, 1, len(SESSIONS)))

# ---------------------------------------------------------------------------
# Figure 1: raw data overview for an example session (spike raster + block structure)
# ---------------------------------------------------------------------------
EXAMPLE_SESSION = "session_1"
f = open_session(ASSET_IDS[EXAMPLE_SESSION])
trials_ex = load_trials(f)
tsgroup_ex, _ = load_good_units_spiketrain(f)
feat_ex, valid_ex = build_prestim_features(tsgroup_ex, trials_ex, window=0.4)
trials_ex_v = trials_ex.loc[valid_ex].reset_index(drop=True)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

# spike raster spanning trials 60-160, first 40 units (this window crosses two block
# transitions: 0.5 -> 0.2 -> 0.8, so the bias panel below is informative)
TRIAL_START, TRIAL_END = 60, 160
t_start = trials_ex["start_time"].iloc[TRIAL_START]
t_end = trials_ex["stop_time"].iloc[TRIAL_END - 1]
n_units_plot = 40
for i, (uid, ts) in enumerate(list(tsgroup_ex.items())[:n_units_plot]):
    spk = ts.t
    spk = spk[(spk >= t_start) & (spk <= t_end)]
    axes[0].vlines(spk, i, i + 0.9, color="k", linewidth=0.4)
for k in range(TRIAL_START, TRIAL_END):
    axes[0].axvline(trials_ex["stimOn_times"].iloc[k], color="tab:red", alpha=0.3, linewidth=0.8)
axes[0].set_ylabel("Unit # (first 40 good units)")
axes[0].set_title(f"{EXAMPLE_SESSION}: example raw spike raster, trials {TRIAL_START}-{TRIAL_END}\n"
                   "(red lines = stimulus onset)")
axes[0].set_xlim(t_start, t_end)

axes[1].plot(trials_ex["start_time"], trials_ex["probabilityLeft"], drawstyle="steps-post", color="tab:blue")
axes[1].set_ylabel("P(stim=left)\n(block prior)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylim(0.1, 0.9)
axes[1].set_xlim(t_start, t_end)

plt.tight_layout()
plt.savefig("figures/fig1_raw_data_overview.png", dpi=150)
plt.close()
print("saved fig1")

# ---------------------------------------------------------------------------
# Figure 1b: block structure across the whole session (all trials)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 2.5))
ax.plot(trials_ex["probabilityLeft"], drawstyle="steps-post", color="tab:blue")
ax.set_xlabel("Trial #")
ax.set_ylabel("P(stim=left)")
ax.set_title(f"{EXAMPLE_SESSION}: block structure across the full session ({len(trials_ex)} trials)")
ax.set_ylim(0.1, 0.9)
plt.tight_layout()
plt.savefig("figures/fig1b_block_structure.png", dpi=150)
plt.close()
print("saved fig1b")

# ---------------------------------------------------------------------------
# Figure 2: behavioral bias on zero-contrast trials (psychometric check)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
x = np.arange(len(SESSIONS))
width = 0.35
low = [results[s]["p_left_lowblock"] for s in SESSIONS]
high = [results[s]["p_left_highblock"] for s in SESSIONS]
ax.bar(x - width / 2, low, width, label="block P(left)=0.2", color="tab:orange")
ax.bar(x + width / 2, high, width, label="block P(left)=0.8", color="tab:green")
ax.set_xticks(x)
ax.set_xticklabels(SESSIONS, rotation=20)
ax.set_ylabel("P(choose left | zero contrast)")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax.set_title("Behavioral choice bias on zero-contrast (no sensory evidence) trials")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig2_behavioral_bias.png", dpi=150)
plt.close()
print("saved fig2")

# ---------------------------------------------------------------------------
# Figure 3: pre-stimulus population activity split by upcoming block bias
# ---------------------------------------------------------------------------
mask_bias_ex = trials_ex_v["probabilityLeft"].isin([0.2, 0.8]).values
y_bias_ex = (trials_ex_v["probabilityLeft"].values[mask_bias_ex] == 0.8).astype(int)
X_bias_ex = feat_ex.values[mask_bias_ex]
rate_ex = X_bias_ex / 0.4  # Hz

mean_rate_low = rate_ex[y_bias_ex == 0].mean(axis=0)
mean_rate_high = rate_ex[y_bias_ex == 1].mean(axis=0)
diff = mean_rate_high - mean_rate_low
order = np.argsort(diff)[::-1]

n_show = 25
top_units = np.concatenate([order[:n_show], order[-n_show:]])

fig, axes = plt.subplots(1, 2, figsize=(11, 6))
axes[0].barh(np.arange(len(top_units)), diff[top_units], color=["tab:green" if d > 0 else "tab:orange" for d in diff[top_units]])
axes[0].set_yticks([])
axes[0].axvline(0, color="k", linewidth=0.8)
axes[0].set_xlabel("Firing rate diff (Hz): P(left)=0.8 minus P(left)=0.2 block")
axes[0].set_title(f"{EXAMPLE_SESSION}: pre-stim (0.4s) rate modulation\nby upcoming block, per unit (top/bottom {n_show})")

sc = axes[1].scatter(mean_rate_low, mean_rate_high, c=diff, cmap="RdYlGn", s=18, vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
lim = max(mean_rate_low.max(), mean_rate_high.max()) * 1.05
axes[1].plot([0, lim], [0, lim], "k--", linewidth=0.8)
axes[1].set_xlabel("Mean pre-stim rate (Hz), block P(left)=0.2")
axes[1].set_ylabel("Mean pre-stim rate (Hz), block P(left)=0.8")
axes[1].set_title("Per-unit pre-stimulus rate by block")
plt.colorbar(sc, ax=axes[1], label="rate diff (Hz)")
plt.tight_layout()
plt.savefig("figures/fig3_prestim_population_activity.png", dpi=150)
plt.close()
print("saved fig3")

# ---------------------------------------------------------------------------
# Figure 4: bias decoding performance (real vs null), per session + ROC
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 4a: real CV AUC vs null distribution per session
ax = axes[0]
for i, s in enumerate(SESSIONS):
    r = results[s]
    null = r["null_aucs_bias"]
    ax.scatter([i] * len(null), null, color="gray", alpha=0.4, s=15, zorder=1)
    ax.errorbar(i, r["aucs_bias"].mean(), yerr=r["aucs_bias"].std(), fmt="o", color=COLORS[i],
                markersize=10, capsize=4, zorder=3, label=f"{s} (p={r['pval_bias']:.1e})")
ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
ax.set_xticks(range(len(SESSIONS)))
ax.set_xticklabels(SESSIONS, rotation=20)
ax.set_ylabel("Decoding AUC")
ax.set_title("Bias (block prior) decode:\ncolored = real CV AUC, gray = shuffled null")
ax.legend(fontsize=7, loc="lower right")

# 4b: pooled ROC curves (out-of-fold predictions) per session
ax = axes[1]
for i, s in enumerate(SESSIONS):
    r = results[s]
    fpr, tpr, _ = roc_curve(r["y_bias"], r["oof_bias"])
    auc = roc_auc_score(r["y_bias"], r["oof_bias"])
    ax.plot(fpr, tpr, color=COLORS[i], label=f"{s} (AUC={auc:.2f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Bias decode ROC\n(pooled out-of-fold predictions)")
ax.legend(fontsize=8, loc="lower right")

# 4c: pooled distribution of real vs null AUC across sessions
ax = axes[2]
real_aucs = np.concatenate([results[s]["aucs_bias"] for s in SESSIONS])
null_aucs = np.concatenate([results[s]["null_aucs_bias"] for s in SESSIONS])
bins = np.linspace(0.2, 0.9, 30)
ax.hist(null_aucs, bins=bins, alpha=0.6, color="gray", label="shuffled null (pooled)", density=True)
ax.hist(real_aucs, bins=bins, alpha=0.6, color="tab:blue", label="real (pooled CV folds)", density=True)
ax.axvline(0.5, color="k", linestyle="--", linewidth=1)
ax.set_xlabel("AUC")
ax.set_ylabel("Density")
ax.set_title("Pooled across all 4 sessions")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig4_bias_decoding.png", dpi=150)
plt.close()
print("saved fig4")

# ---------------------------------------------------------------------------
# Figure 5: choice decoding on zero-contrast (no-evidence) trials
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
for i, s in enumerate(SESSIONS):
    r = results[s]
    null = r["null_aucs_choice"]
    ax.scatter([i] * len(null), null, color="gray", alpha=0.4, s=15, zorder=1)
    ax.errorbar(i, r["aucs_choice"].mean(), yerr=r["aucs_choice"].std(), fmt="o", color=COLORS[i],
                markersize=10, capsize=4, zorder=3, label=f"{s} (p={r['pval_choice']:.2g}, n={r['n_choice_trials']})")
ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
ax.set_xticks(range(len(SESSIONS)))
ax.set_xticklabels(SESSIONS, rotation=20)
ax.set_ylabel("Decoding AUC")
ax.set_title("Upcoming choice decode on zero-contrast\n(no sensory evidence) trials")
ax.legend(fontsize=7, loc="lower right")

ax = axes[1]
for i, s in enumerate(SESSIONS):
    r = results[s]
    fpr, tpr, _ = roc_curve(r["y_choice"], r["oof_choice"])
    auc = roc_auc_score(r["y_choice"], r["oof_choice"])
    ax.plot(fpr, tpr, color=COLORS[i], label=f"{s} (AUC={auc:.2f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Choice decode ROC (zero-contrast trials)")
ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("figures/fig5_choice_decoding_zero_contrast.png", dpi=150)
plt.close()
print("saved fig5")

# ---------------------------------------------------------------------------
# Figure 6: summary across sessions - trials/units + pooled significance table as text figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.axis("off")
rows = [["Session", "N trials", "N good units", "Bias AUC (p-value)", "Choice AUC (p-value, n)"]]
for s in SESSIONS:
    r = results[s]
    rows.append([
        s,
        str(r["n_trials"]),
        str(r["n_units"]),
        f"{r['aucs_bias'].mean():.3f} (p={r['pval_bias']:.1e})",
        f"{r['aucs_choice'].mean():.3f} (p={r['pval_choice']:.2g}, n={r['n_choice_trials']})",
    ])
table = ax.table(cellText=rows, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)
for j in range(5):
    table[0, j].set_facecolor("#dddddd")
ax.set_title("Summary: pre-stimulus decoding results per session", pad=20)
plt.tight_layout()
plt.savefig("figures/fig6_summary_table.png", dpi=150)
plt.close()
print("saved fig6")

print("\nAll figures saved to figures/")
