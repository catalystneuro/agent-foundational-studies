# %% [markdown]
# # Decoding an Upcoming Decision Bias from Pre-Stimulus Neural Activity
#
# This notebook demonstrates that an animal's upcoming *decision bias* can be decoded
# from population spiking activity recorded **before stimulus onset**, in a mouse
# perceptual decision-making task.
#
# **Dataset**: DANDI Archive [Dandiset 000149](https://dandiarchive.org/dandiset/000149)
# ("IBL ephys data"), from the International Brain Laboratory. Each session contains
# spike-sorted units (Neuropixels) recorded across multiple brain regions during the
# IBL visual contrast discrimination task, together with a `trials` table encoding
# stimulus contrast, block prior, choice and reward outcome.
#
# **Task structure**: on each trial, mice must report whether a grating stimulus
# appears on the left or right screen. Critically, the prior probability that the
# stimulus will appear on the left, `probabilityLeft`, is fixed within blocks of
# ~20-100 trials and switches between 0.2, 0.5 and 0.8. This block prior is the
# animal's **decision bias**: a purely internal, non-sensory expectation about the
# upcoming trial that the animal must learn and track, and which is known to shape
# choice most strongly when the visual stimulus itself carries no information
# (zero-contrast trials).
#
# Every trial has an enforced ~0.4-0.7 s "quiescence period" immediately before
# stimulus onset, during which the wheel must be still. We decode the upcoming block
# bias (and, on zero-contrast trials, the upcoming choice) from population spike
# counts in a fixed 0.4 s window ending at `stimOn_times`, i.e. entirely before any
# stimulus information is available.
#
# **Approach**: for each of the 4 available IBL sessions, we
# 1. bulk-load spike times and build good-unit population spike counts in the
#    pre-stimulus window per trial,
# 2. decode the upcoming block identity (`probabilityLeft` = 0.2 vs 0.8) using a
#    cross-validated NeMoS Bernoulli GLM (logistic regression) on population counts,
# 3. test significance against a label-shuffled null via an exact Mann-Whitney U
#    test on cross-validated out-of-fold predictions,
# 4. additionally decode the animal's upcoming *choice* from the same pre-stimulus
#    window, restricted to zero-contrast trials where no sensory evidence is present,
#    to link the neural bias signal directly to behavior.

# %% [markdown]
# ## Setup

# %%
import pickle
import time

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
    cv_decode_bernoulli,
    oof_auc_pvalue,
    permutation_null_auc,
)

ALPHA = 1.0          # Ridge regularization strength for the NeMoS Bernoulli GLM decoders
WINDOW = 0.4          # pre-stimulus window duration (s), ending at stimOn_times
N_SPLITS = 5          # cross-validation folds
N_PERM_ILLUSTRATIVE = 15  # permutations for the illustrative null-distribution plot only
SESSIONS = list(ASSET_IDS.keys())

print("DANDI 000149 sessions:", ASSET_IDS)

# %% [markdown]
# ## Inspect a single session
#
# We start by loading one session (`session_1`) with LINDI streaming access and
# inspecting the trial structure and spike data before running the full analysis.

# %%
f_ex = open_session(ASSET_IDS["session_1"])
trials_ex = load_trials(f_ex)
tsgroup_ex, _ = load_good_units_spiketrain(f_ex)
print(f"session_1: {trials_ex.shape[0]} trials, {len(tsgroup_ex)} good units "
      f"(label == 1.0, i.e. passing all IBL QC criteria)")
print(trials_ex[["stimOn_times", "choice", "probabilityLeft", "contrastLeft", "contrastRight"]].head())

feat_ex, valid_ex = build_prestim_features(tsgroup_ex, trials_ex, window=WINDOW)
trials_ex_v = trials_ex.loc[valid_ex].reset_index(drop=True)
print(f"\nPre-stimulus ({WINDOW}s) spike-count feature matrix: {feat_ex.shape}")

# minimum gap between trial start and stimulus onset, to confirm the 0.4s window
# never reaches back into the previous trial
gap = (trials_ex["stimOn_times"] - trials_ex["start_time"]).values
print(f"min(stimOn - trial start) = {np.nanmin(gap):.3f}s "
      f"(> {WINDOW}s window for all trials: {np.all(gap >= WINDOW)})")

# %% [markdown]
# ### Raw data visualization
#
# Example spike raster for the first 60 trials (first 40 good units), with stimulus
# onset times marked, and the block-prior (`probabilityLeft`) time course.

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

# trials 60-160 cross two block transitions (0.5 -> 0.2 -> 0.8), so the bias panel
# below is informative (an earlier window would sit entirely in the neutral block)
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
axes[0].set_title(f"session_1: example raw spike raster, trials {TRIAL_START}-{TRIAL_END}\n"
                   "(red lines = stimulus onset)")
axes[0].set_xlim(t_start, t_end)

axes[1].plot(trials_ex["start_time"], trials_ex["probabilityLeft"], drawstyle="steps-post", color="tab:blue")
axes[1].set_ylabel("P(stim=left)\n(block prior)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylim(0.1, 0.9)
axes[1].set_xlim(t_start, t_end)

plt.tight_layout()
plt.savefig("figures/fig1_raw_data_overview.png", dpi=150)
plt.show()
print("saved figures/fig1_raw_data_overview.png")

# %%
fig, ax = plt.subplots(figsize=(11, 2.5))
ax.plot(trials_ex["probabilityLeft"], drawstyle="steps-post", color="tab:blue")
ax.set_xlabel("Trial #")
ax.set_ylabel("P(stim=left)")
ax.set_title(f"session_1: block structure across the full session ({len(trials_ex)} trials)")
ax.set_ylim(0.1, 0.9)
plt.tight_layout()
plt.savefig("figures/fig1b_block_structure.png", dpi=150)
plt.show()
print("saved figures/fig1b_block_structure.png")

# %% [markdown]
# ## Run the analysis across all 4 sessions
#
# For each session we:
# - build the pre-stimulus (0.4 s) population spike-count feature matrix,
# - check the *behavioral* signature of the bias (choice bias on zero-contrast trials),
# - decode the upcoming block identity from pre-stimulus population activity with a
#   cross-validated NeMoS Bernoulli GLM,
# - decode the upcoming choice from pre-stimulus activity, restricted to zero-contrast
#   trials (no sensory evidence available).
#
# Significance is assessed with an exact Mann-Whitney U test on cross-validated
# out-of-fold predicted probabilities (equivalent to testing AUC != 0.5); a small
# label-permutation null is also generated purely for the illustrative null-distribution
# figure below.

# %%
results = {}

for sess_name, asset_id in ASSET_IDS.items():
    print(f"\n=== {sess_name} ({asset_id}) ===")
    t0 = time.time()
    f = open_session(asset_id)
    trials = load_trials(f)
    tsgroup, good_idx = load_good_units_spiketrain(f)
    feat, valid = build_prestim_features(tsgroup, trials, window=WINDOW)
    trials_v = trials.loc[valid].reset_index(drop=True)
    assert len(trials_v) == len(feat)
    print(f"  loaded: {trials.shape[0]} trials, {len(tsgroup)} good units, {time.time()-t0:.1f}s")

    zc = trials_v["zero_contrast"].values & trials_v["probabilityLeft"].isin([0.2, 0.8]).values
    zc_choice_left = (trials_v["choice"].values == -1).astype(float)
    p_left_lowblock = zc_choice_left[zc & (trials_v["probabilityLeft"].values == 0.2)].mean()
    p_left_highblock = zc_choice_left[zc & (trials_v["probabilityLeft"].values == 0.8)].mean()
    n_zc = zc.sum()
    print(f"  zero-contrast trials: {n_zc}, P(left|probL=0.2)={p_left_lowblock:.3f}, "
          f"P(left|probL=0.8)={p_left_highblock:.3f}")

    mask_bias = trials_v["probabilityLeft"].isin([0.2, 0.8]).values
    X_bias = feat.values[mask_bias]
    y_bias = (trials_v["probabilityLeft"].values[mask_bias] == 0.8).astype(float)
    aucs_bias, accs_bias, oof_bias = cv_decode_bernoulli(X_bias, y_bias, n_splits=N_SPLITS, alpha=ALPHA)
    pval_bias = oof_auc_pvalue(y_bias, oof_bias)
    null_aucs_bias = permutation_null_auc(X_bias, y_bias, n_perm=N_PERM_ILLUSTRATIVE, alpha=ALPHA)
    print(f"  BIAS decode: real AUC={aucs_bias.mean():.3f}+-{aucs_bias.std():.3f}, p={pval_bias:.4g}")

    mask_choice = zc
    X_choice = feat.values[mask_choice]
    y_choice = zc_choice_left[mask_choice]
    n_splits_choice = min(N_SPLITS, int(min(np.sum(y_choice == 0), np.sum(y_choice == 1))))
    aucs_choice, accs_choice, oof_choice = cv_decode_bernoulli(
        X_choice, y_choice, n_splits=n_splits_choice, alpha=ALPHA
    )
    pval_choice = oof_auc_pvalue(y_choice, oof_choice)
    null_aucs_choice = permutation_null_auc(X_choice, y_choice, n_perm=N_PERM_ILLUSTRATIVE, alpha=ALPHA)
    print(f"  CHOICE(zero-contrast) decode: real AUC={aucs_choice.mean():.3f}+-{aucs_choice.std():.3f}, "
          f"p={pval_choice:.4g}, n={mask_choice.sum()}")

    results[sess_name] = dict(
        asset_id=asset_id, n_trials=trials.shape[0], n_units=len(tsgroup), n_zero_contrast=int(n_zc),
        p_left_lowblock=p_left_lowblock, p_left_highblock=p_left_highblock,
        aucs_bias=aucs_bias, pval_bias=pval_bias, null_aucs_bias=null_aucs_bias,
        n_bias_trials=int(mask_bias.sum()), y_bias=y_bias, oof_bias=oof_bias,
        aucs_choice=aucs_choice, pval_choice=pval_choice, null_aucs_choice=null_aucs_choice,
        n_choice_trials=int(mask_choice.sum()), y_choice=y_choice, oof_choice=oof_choice,
    )

with open("results.pkl", "wb") as fh:
    pickle.dump(results, fh)
print("\nAll sessions done.")

# %% [markdown]
# ## Pre-stimulus population activity split by upcoming block bias
#
# For the example session, we compare mean pre-stimulus firing rate per unit between
# trials from the two biased blocks (`probabilityLeft` = 0.2 vs 0.8). Many units show
# systematic rate differences before the stimulus even appears.

# %%
mask_bias_ex = trials_ex_v["probabilityLeft"].isin([0.2, 0.8]).values
y_bias_ex = (trials_ex_v["probabilityLeft"].values[mask_bias_ex] == 0.8).astype(int)
X_bias_ex = feat_ex.values[mask_bias_ex]
rate_ex = X_bias_ex / WINDOW

mean_rate_low = rate_ex[y_bias_ex == 0].mean(axis=0)
mean_rate_high = rate_ex[y_bias_ex == 1].mean(axis=0)
diff = mean_rate_high - mean_rate_low
order = np.argsort(diff)[::-1]
n_show = 25
top_units = np.concatenate([order[:n_show], order[-n_show:]])

fig, axes = plt.subplots(1, 2, figsize=(11, 6))
axes[0].barh(np.arange(len(top_units)), diff[top_units],
             color=["tab:green" if d > 0 else "tab:orange" for d in diff[top_units]])
axes[0].set_yticks([])
axes[0].axvline(0, color="k", linewidth=0.8)
axes[0].set_xlabel("Firing rate diff (Hz): P(left)=0.8 minus P(left)=0.2 block")
axes[0].set_title(f"session_1: pre-stim (0.4s) rate modulation\nby upcoming block, per unit (top/bottom {n_show})")

sc = axes[1].scatter(mean_rate_low, mean_rate_high, c=diff, cmap="RdYlGn", s=18,
                      vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
lim = max(mean_rate_low.max(), mean_rate_high.max()) * 1.05
axes[1].plot([0, lim], [0, lim], "k--", linewidth=0.8)
axes[1].set_xlabel("Mean pre-stim rate (Hz), block P(left)=0.2")
axes[1].set_ylabel("Mean pre-stim rate (Hz), block P(left)=0.8")
axes[1].set_title("Per-unit pre-stimulus rate by block")
plt.colorbar(sc, ax=axes[1], label="rate diff (Hz)")
plt.tight_layout()
plt.savefig("figures/fig3_prestim_population_activity.png", dpi=150)
plt.show()
print("saved figures/fig3_prestim_population_activity.png")

# %% [markdown]
# ## Behavioral signature of the bias
#
# As a positive control, we confirm the behavioral effect of the block prior: on
# zero-contrast trials (no visual evidence), the probability of choosing left should
# track the block prior directly.

# %%
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
plt.show()
print("saved figures/fig2_behavioral_bias.png")

# %% [markdown]
# ## Decoding the upcoming block bias from pre-stimulus population activity
#
# Left: real cross-validated AUC per session (colored) against a label-shuffled null
# (gray). Middle: ROC curves from pooled out-of-fold predictions per session. Right:
# pooled distribution of real vs. null AUC across all sessions and CV folds.

# %%
COLORS = plt.cm.tab10(np.linspace(0, 1, len(SESSIONS)))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

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
plt.show()
print("saved figures/fig4_bias_decoding.png")

# %% [markdown]
# ## Linking the neural bias signal to behavior: decoding upcoming choice on
# zero-contrast trials
#
# If the pre-stimulus bias signal reflects a genuine decision-relevant internal
# state, it should also predict the animal's upcoming choice specifically when no
# sensory evidence is available to justify that choice (zero-contrast trials).

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
for i, s in enumerate(SESSIONS):
    r = results[s]
    null = r["null_aucs_choice"]
    ax.scatter([i] * len(null), null, color="gray", alpha=0.4, s=15, zorder=1)
    ax.errorbar(i, r["aucs_choice"].mean(), yerr=r["aucs_choice"].std(), fmt="o", color=COLORS[i],
                markersize=10, capsize=4, zorder=3,
                label=f"{s} (p={r['pval_choice']:.2g}, n={r['n_choice_trials']})")
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
plt.show()
print("saved figures/fig5_choice_decoding_zero_contrast.png")

# %% [markdown]
# ## Summary table

# %%
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.axis("off")
rows = [["Session", "N trials", "N good units", "Bias AUC (p-value)", "Choice AUC (p-value, n)"]]
for s in SESSIONS:
    r = results[s]
    rows.append([
        s, str(r["n_trials"]), str(r["n_units"]),
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
plt.show()
print("saved figures/fig6_summary_table.png")

for s in SESSIONS:
    r = results[s]
    print(f"{s}: bias AUC={r['aucs_bias'].mean():.3f} (p={r['pval_bias']:.2e}), "
          f"choice AUC={r['aucs_choice'].mean():.3f} (p={r['pval_choice']:.2g}, n={r['n_choice_trials']})")

# %% [markdown]
# ## Conclusion
#
# Across all 4 IBL sessions in Dandiset 000149, population spiking activity in a
# 0.4 s window strictly *before* stimulus onset significantly decodes the upcoming
# block bias (`probabilityLeft`), well above a label-shuffled chance level, using a
# cross-validated NeMoS Bernoulli GLM. This pre-stimulus neural signal is behaviorally
# relevant: on zero-contrast trials, where the animal has no sensory evidence to guide
# its choice, the same pre-stimulus population activity carries information about the
# choice the animal is about to make, mirroring the purely internally-driven behavioral
# choice bias measured directly from the trial data. Together these results demonstrate
# that an upcoming decision bias is encoded in neural activity before the decision-
# relevant stimulus is even presented.
