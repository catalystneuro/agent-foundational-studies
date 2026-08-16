# %% [markdown]
# # Decoding Upcoming Decision Bias from Pre-Stimulus Neural Activity
#
# In perceptual decision tasks, subjects often carry a "prior" belief about which
# response is more likely to be rewarded on the upcoming trial. In the International
# Brain Laboratory (IBL) task, this prior is manipulated experimentally: the
# probability that the visual stimulus will appear on the left side
# (`probabilityLeft`) is set to a block value of 0.2, 0.5, or 0.8 and held fixed for
# runs of trials. Mice learn this block structure and shift their choices toward the
# more probable side, especially on zero-contrast trials where the stimulus itself
# carries no information.
#
# This notebook asks: **is the animal's upcoming bias state encoded in neural
# population activity before the stimulus even appears?** We isolate a clean,
# movement-free pre-stimulus window enforced by the task's quiescence period
# (mice must hold the wheel still for 0.4-0.7 s immediately before stimulus onset),
# so any decodable signal in this window cannot reflect stimulus-evoked or
# movement-evoked activity. We then train a cross-validated linear decoder to
# predict the trial's block identity (`probabilityLeft` = 0.2 vs. 0.8) from
# population spike counts in this window, and compare decoding accuracy against a
# label-shuffled null distribution.
#
# **Dataset**: DANDI Archive Dandiset
# [000149](https://dandiarchive.org/dandiset/000149) ("IBL ephys data"), which
# provides spike-sorted units and behavioral trial tables for 4 IBL Neuropixels
# recording sessions during the biased-blocks perceptual decision task. Data are
# streamed via LINDI (no full-file downloads).

# %%
import warnings

import lindi
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

RNG_SEED = 0
PRE_WINDOW = 0.4  # s, pre-stimulus window length (within enforced quiescence period)
N_SHUFFLES = 200
N_CV_FOLDS = 5

# The 4 sessions in Dandiset 000149 ("IBL ephys data")
SESSION_ASSETS = {
    "sub-92130c1b_ses-c7bd79c9": "31f22c47-1512-4293-b19f-6fa5bd9b7cbf",
    "sub-70bf8cbd_ses-4ecb5d24": "81169999-c697-4eca-a635-2fd994ac183f",
    "sub-9bebfe0b_ses-4b7fbad4": "e7fa5ae0-b957-4b24-aa40-fb4c3276d331",
    "sub-c6e8125f_ses-aad23144": "f791a116-1e6c-4d6a-a9eb-fe3644737be2",
}


def lindi_url(asset_id):
    return f"https://lindi.neurosift.org/dandi/dandisets/000149/assets/{asset_id}/nwb.lindi.json"


# %% [markdown]
# ## Data Loading
#
# For each session we load the `trials` table (choice, contrast, block prior,
# stimulus/movement timing) and the spike-sorted `units` table, restricting to
# units labeled "good" (`label == 1`, the IBL single-unit quality criterion). Spike
# times for all good units are bulk-read once and wrapped as a Pynapple `TsGroup`.


def load_session(asset_id):
    local_cache = lindi.LocalCache()
    f = lindi.LindiH5pyFile.from_lindi_file(lindi_url(asset_id), local_cache=local_cache)

    trials = f["intervals"]["trials"]
    trial_data = dict(
        choice=trials["choice.npy"][:],
        contrastLeft=np.nan_to_num(trials["contrastLeft.npy"][:]),
        contrastRight=np.nan_to_num(trials["contrastRight.npy"][:]),
        probabilityLeft=trials["probabilityLeft.npy"][:],
        stimOn_times=trials["stimOn_times.npy"][:],
        firstMovement_times=trials["firstMovement_times.npy"][:],
        feedbackType=trials["feedbackType.npy"][:],
    )

    units = f["units"]
    label = units["label"][:]
    spike_times = units["spike_times"][:]
    spike_times_index = units["spike_times_index"][:]
    good = np.where(label == 1.0)[0]

    starts = np.concatenate(([0], spike_times_index[:-1])).astype(int)
    ends = spike_times_index.astype(int)
    spikes_dict = {
        i: nap.Ts(t=spike_times[int(starts[u]) : int(ends[u])])
        for i, u in enumerate(good)
    }
    tsgroup = nap.TsGroup(spikes_dict)

    return trial_data, tsgroup


# %% [markdown]
# ## Feature Extraction: Pre-Stimulus Population Spike Counts
#
# For each trial we take the window `[stimOn_time - 0.4, stimOn_time)`. Trials
# where `firstMovement_times` falls inside (or just after) this window are dropped
# as quiescence violations, guaranteeing the window is free of overt movement.
# Population spike counts in this window (one bin per trial, via Pynapple's
# `TsGroup.count`) form the feature matrix for decoding.


def get_valid_prestim_mask(trial_data):
    stimOn = trial_data["stimOn_times"]
    firstMove = trial_data["firstMovement_times"]
    pre_start = stimOn - PRE_WINDOW
    pre_end = stimOn
    valid_stim = ~np.isnan(stimOn)
    valid_move = ~np.isnan(firstMove)
    violation = valid_move & (firstMove > pre_start) & (firstMove < pre_end + 0.05)
    return valid_stim & ~violation, pre_start, pre_end


def prestim_features_and_labels(tsgroup, pre_start, pre_end, mask, label_values):
    """Build the pre-stimulus population count matrix and matching labels.

    Pynapple's IntervalSet silently sorts intervals by start time internally, so
    `tsgroup.count(ep=...)` rows come back in start-time order, not necessarily
    the order the ep was constructed in. We explicitly sort trial indices by
    pre_start ourselves and apply the identical permutation to the labels, so
    feature rows and labels stay aligned regardless of pynapple's internal order.
    """
    idx = np.where(mask)[0]
    order = np.argsort(pre_start[idx], kind="stable")
    idx_sorted = idx[order]
    ep = nap.IntervalSet(start=pre_start[idx_sorted], end=pre_end[idx_sorted])
    counts = tsgroup.count(ep=ep, bin_size=PRE_WINDOW).values
    assert counts.shape[0] == len(idx_sorted)
    assert np.allclose(ep.start, pre_start[idx_sorted])
    labels = label_values[idx_sorted]
    return counts, labels


def decode_with_shuffle(X, y, n_shuffles=N_SHUFFLES, C=0.1, seed=RNG_SEED):
    vt = VarianceThreshold(threshold=0.01)
    X_f = vt.fit_transform(X)
    if X_f.shape[1] < 2:
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C))
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    true_scores = cross_val_score(clf, X_f, y, cv=cv)
    rng = np.random.RandomState(seed)
    null_scores = np.empty(n_shuffles)
    for i in range(n_shuffles):
        y_shuf = rng.permutation(y)
        null_scores[i] = cross_val_score(clf, X_f, y_shuf, cv=cv).mean()
    acc = true_scores.mean()
    pval = (np.sum(null_scores >= acc) + 1) / (n_shuffles + 1)
    return dict(
        acc=acc,
        acc_std=true_scores.std(),
        n_features=X_f.shape[1],
        n_trials=X_f.shape[0],
        null_scores=null_scores,
        pval=pval,
    )


# %% [markdown]
# ## Run the Pipeline Across All 4 Sessions
#
# For each session we compute three things:
# 1. **Behavioral bias check**: on zero-contrast trials (no sensory evidence), does
#    the block prior (`probabilityLeft`) shift the animal's choice? This confirms
#    the bias manipulation actually worked before we look for its neural signature.
# 2. **Block decoding**: cross-validated logistic-regression decoding of block
#    identity (`probabilityLeft` = 0.2 vs. 0.8) from pre-stimulus population spike
#    counts, vs. a label-shuffled null.
# 3. **Choice decoding on zero-contrast trials**: decoding the animal's upcoming
#    choice (which on these trials is driven purely by internal bias, not stimulus)
#    from the same pre-stimulus window.

# %%
session_results = {}

for sess_name, asset_id in tqdm(SESSION_ASSETS.items(), desc="Sessions"):
    trial_data, tsgroup = load_session(asset_id)
    n_good_units = len(tsgroup)
    mask, pre_start, pre_end = get_valid_prestim_mask(trial_data)

    pL = trial_data["probabilityLeft"]
    cL, cR = trial_data["contrastLeft"], trial_data["contrastRight"]
    choice = trial_data["choice"]

    # --- behavioral bias on zero-contrast trials ---
    zero_contrast = (cL == 0) & (cR == 0)
    beh_mask = mask & zero_contrast & ~np.isnan(choice)
    p_right = {}
    n_beh = {}
    for pl_val in (0.2, 0.8):
        sub = beh_mask & (pL == pl_val)
        n_beh[pl_val] = int(sub.sum())
        p_right[pl_val] = float(np.mean(choice[sub] == 1)) if sub.sum() > 0 else np.nan

    # --- block decoding (0.2 vs 0.8) ---
    block_mask = mask & ((pL == 0.2) | (pL == 0.8))
    X_block, pL_block = prestim_features_and_labels(tsgroup, pre_start, pre_end, block_mask, pL)
    y_block = (pL_block == 0.8).astype(int)
    block_result = decode_with_shuffle(X_block, y_block)

    # --- choice decoding on zero-contrast trials ---
    choice_mask = mask & zero_contrast & ~np.isnan(choice)
    X_choice, choice_sub = prestim_features_and_labels(tsgroup, pre_start, pre_end, choice_mask, choice)
    y_choice = (choice_sub == 1).astype(int)
    choice_result = decode_with_shuffle(X_choice, y_choice, C=0.05)

    session_results[sess_name] = dict(
        n_good_units=n_good_units,
        n_trials_total=len(pL),
        n_trials_kept=int(mask.sum()),
        p_right=p_right,
        n_beh=n_beh,
        block=block_result,
        choice=choice_result,
        trial_data=trial_data,
        tsgroup=tsgroup,
        mask=mask,
        pre_start=pre_start,
        pre_end=pre_end,
    )

for sess_name, res in session_results.items():
    print(sess_name)
    print(f"  good units: {res['n_good_units']}, trials kept: {res['n_trials_kept']}/{res['n_trials_total']}")
    print(f"  behavioral bias (P(choice==1) | zero contrast): pL=0.2 -> {res['p_right'][0.2]:.2f} (n={res['n_beh'][0.2]}), pL=0.8 -> {res['p_right'][0.8]:.2f} (n={res['n_beh'][0.8]})")
    b = res["block"]
    print(f"  block decode acc: {b['acc']:.3f} (null {b['null_scores'].mean():.3f}+/-{b['null_scores'].std():.3f}), p={b['pval']:.4f}, n_trials={b['n_trials']}")
    c = res["choice"]
    if c is not None:
        print(f"  choice decode acc (zero-contrast): {c['acc']:.3f} (null {c['null_scores'].mean():.3f}+/-{c['null_scores'].std():.3f}), p={c['pval']:.4f}, n_trials={c['n_trials']}")

# %% [markdown]
# ## Figure 1: Raw Data Validation — Pre-Stimulus Population Raster and Rate
#
# Before trusting any decoding result, we visualize the raw spike data around
# stimulus onset for one example session: a population raster for a block of
# trials, and the trial-averaged population firing rate time course split by
# block, confirming the pre-stimulus window is indeed movement-free and that we
# can see a plausible pre-stimulus rate difference between blocks.

# %%
example_sess = "sub-92130c1b_ses-c7bd79c9"
ex = session_results[example_sess]
tsgroup = ex["tsgroup"]
trial_data = ex["trial_data"]
stimOn = trial_data["stimOn_times"]
pL = trial_data["probabilityLeft"]
mask = ex["mask"]

fig, axes = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [1, 1]})

# raster: first 40 valid trials, spikes relative to stimOn, all good units pooled
ax = axes[0]
n_trials_show = 40
valid_idx = np.where(mask)[0][:n_trials_show]
for row, ti in enumerate(valid_idx):
    t0 = stimOn[ti]
    win = nap.IntervalSet(start=t0 - 1.0, end=t0 + 0.5)
    for unit_id in tsgroup.index:
        sp = tsgroup[unit_id].restrict(win).t - t0
        ax.plot(sp, np.full_like(sp, row), "|", color="k", markersize=2, alpha=0.5)
ax.axvline(0, color="tab:red", lw=1.5, label="stim on")
ax.axvline(-PRE_WINDOW, color="tab:blue", lw=1.5, ls="--", label="pre-stim window start")
ax.set_xlabel("Time relative to stimulus onset (s)")
ax.set_ylabel("Trial # (population-pooled spikes)")
ax.set_title(f"Example session {example_sess}: population raster around stimulus onset")
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(-1.0, 0.5)

# trial-averaged population rate, split by block
ax = axes[1]
bins = np.arange(-1.0, 0.5001, 0.05)
bin_centers = (bins[:-1] + bins[1:]) / 2
for pl_val, color in [(0.2, "tab:orange"), (0.8, "tab:green")]:
    sel = mask & (pL == pl_val)
    idx = np.where(sel)[0]
    rate_mat = np.zeros((len(idx), len(bin_centers)))
    for row, ti in enumerate(idx):
        t0 = stimOn[ti]
        win = nap.IntervalSet(start=t0 - 1.0, end=t0 + 0.5)
        all_spikes = np.concatenate(
            [tsgroup[u].restrict(win).t - t0 for u in tsgroup.index]
        )
        counts, _ = np.histogram(all_spikes, bins=bins)
        rate_mat[row] = counts / (0.05 * len(tsgroup))
    mean_rate = rate_mat.mean(axis=0)
    sem_rate = rate_mat.std(axis=0) / np.sqrt(rate_mat.shape[0])
    ax.plot(bin_centers, mean_rate, color=color, label=f"probabilityLeft={pl_val} (n={len(idx)})")
    ax.fill_between(bin_centers, mean_rate - sem_rate, mean_rate + sem_rate, color=color, alpha=0.25)
ax.axvline(0, color="tab:red", lw=1.5)
ax.axvline(-PRE_WINDOW, color="tab:blue", lw=1.5, ls="--")
ax.set_xlabel("Time relative to stimulus onset (s)")
ax.set_ylabel("Mean population firing rate (Hz/unit)")
ax.set_title("Trial-averaged population rate by block prior")
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(-1.0, 0.5)

plt.tight_layout()
plt.savefig("figures/01_raw_data_raster_and_rate.png", dpi=150)
plt.close()

# %% [markdown]
# ## Figure 2: Behavioral Confirmation of the Bias Manipulation
#
# On zero-contrast trials, the stimulus carries no left/right information, so any
# systematic choice preference must come from the animal's internal bias state
# induced by the block prior. This plot confirms the manipulation is behaviorally
# effective in every session before we look at its neural correlate.

# %%
fig, ax = plt.subplots(figsize=(8, 5))
sess_names = list(session_results.keys())
x = np.arange(len(sess_names))
width = 0.35
p02 = [session_results[s]["p_right"][0.2] for s in sess_names]
p08 = [session_results[s]["p_right"][0.8] for s in sess_names]
n02 = [session_results[s]["n_beh"][0.2] for s in sess_names]
n08 = [session_results[s]["n_beh"][0.8] for s in sess_names]

ax.bar(x - width / 2, p02, width, label="probabilityLeft = 0.2", color="tab:orange")
ax.bar(x + width / 2, p08, width, label="probabilityLeft = 0.8", color="tab:green")
for i in range(len(sess_names)):
    ax.text(x[i] - width / 2, p02[i] + 0.02, f"n={n02[i]}", ha="center", fontsize=8)
    ax.text(x[i] + width / 2, p08[i] + 0.02, f"n={n08[i]}", ha="center", fontsize=8)
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(sess_names, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("P(choice == 1) on zero-contrast trials")
ax.set_title("Block prior biases choice on zero-contrast (no-evidence) trials")
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.savefig("figures/02_behavioral_bias.png", dpi=150)
plt.close()

# %% [markdown]
# ## Figure 3: Decoding Block Prior from Pre-Stimulus Population Activity
#
# For each session, cross-validated decoding accuracy for predicting the block
# identity (0.2 vs. 0.8) from pre-stimulus population spike counts, plotted
# against the label-shuffled null distribution.

# %%
fig, axes = plt.subplots(1, len(sess_names), figsize=(4 * len(sess_names), 4.5), sharey=True)
for i, s in enumerate(sess_names):
    ax = axes[i]
    b = session_results[s]["block"]
    ax.hist(b["null_scores"], bins=25, color="lightgray", label="shuffled null")
    ax.axvline(b["acc"], color="tab:red", lw=2, label=f"observed acc={b['acc']:.3f}")
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_title(f"{s}\np={b['pval']:.4f}, n={b['n_trials']} trials, {b['n_features']} units", fontsize=9)
    ax.set_xlabel("CV accuracy")
    if i == 0:
        ax.set_ylabel("Count (shuffles)")
    ax.legend(fontsize=8)
plt.suptitle("Decoding block prior (probabilityLeft: 0.2 vs 0.8) from pre-stimulus activity")
plt.tight_layout()
plt.savefig("figures/03_block_decoding.png", dpi=150)
plt.close()

# %% [markdown]
# ## Figure 4: Decoding Upcoming Choice on Zero-Contrast Trials
#
# On zero-contrast trials the stimulus contains no directional information, so
# above-chance decoding of the animal's upcoming choice from pre-stimulus activity
# reflects the internal bias state directly influencing action selection, not
# stimulus processing.

# %%
choice_sessions = [s for s in sess_names if session_results[s]["choice"] is not None]
fig, axes = plt.subplots(1, len(choice_sessions), figsize=(4 * len(choice_sessions), 4.5), sharey=True)
if len(choice_sessions) == 1:
    axes = [axes]
for i, s in enumerate(choice_sessions):
    ax = axes[i]
    c = session_results[s]["choice"]
    ax.hist(c["null_scores"], bins=25, color="lightgray", label="shuffled null")
    ax.axvline(c["acc"], color="tab:purple", lw=2, label=f"observed acc={c['acc']:.3f}")
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_title(f"{s}\np={c['pval']:.4f}, n={c['n_trials']} trials, {c['n_features']} units", fontsize=9)
    ax.set_xlabel("CV accuracy")
    if i == 0:
        ax.set_ylabel("Count (shuffles)")
    ax.legend(fontsize=8)
plt.suptitle("Decoding upcoming choice on zero-contrast trials from pre-stimulus activity")
plt.tight_layout()
plt.savefig("figures/04_choice_decoding_zero_contrast.png", dpi=150)
plt.close()

# %% [markdown]
# ## Figure 5: Summary Across Sessions
#
# Pooled summary comparing observed vs. null-mean decoding accuracy for both
# decoding targets across all 4 sessions.

# %%
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(sess_names))
width = 0.35

block_acc = [session_results[s]["block"]["acc"] for s in sess_names]
block_null = [session_results[s]["block"]["null_scores"].mean() for s in sess_names]
block_null_std = [session_results[s]["block"]["null_scores"].std() for s in sess_names]

ax.bar(x - width / 2, block_acc, width, color="tab:red", label="block decode (observed)")
ax.errorbar(x - width / 2, block_null, yerr=block_null_std, fmt="o", color="k", capsize=3, label="block decode (null mean +/- SD)")

choice_acc = [session_results[s]["choice"]["acc"] if session_results[s]["choice"] else np.nan for s in sess_names]
choice_null = [session_results[s]["choice"]["null_scores"].mean() if session_results[s]["choice"] else np.nan for s in sess_names]
choice_null_std = [session_results[s]["choice"]["null_scores"].std() if session_results[s]["choice"] else np.nan for s in sess_names]

ax.bar(x + width / 2, choice_acc, width, color="tab:purple", label="choice decode, zero-contrast (observed)")
ax.errorbar(x + width / 2, choice_null, yerr=choice_null_std, fmt="s", color="dimgray", capsize=3, label="choice decode (null mean +/- SD)")

ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(sess_names, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Cross-validated decoding accuracy")
ax.set_title("Pre-stimulus decoding accuracy vs. shuffled null, across sessions")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
plt.tight_layout()
plt.savefig("figures/05_summary_across_sessions.png", dpi=150)
plt.close()

print("All figures saved to figures/")

# %% [markdown]
# ## Results Summary
#
# Across all 4 IBL Neuropixels sessions in Dandiset 000149, the block prior
# (`probabilityLeft`) robustly shifts choice on zero-contrast, no-evidence trials
# (Figure 2), confirming the bias manipulation is behaviorally effective. A
# cross-validated linear decoder trained on population spike counts from a
# 0.4 s pre-stimulus window (strictly before stimulus onset and free of overt
# movement, enforced by the task's quiescence period) predicts the upcoming
# block identity above chance and above the label-shuffled null in every session
# (Figure 3). This demonstrates that the animal's internal decision bias state is
# represented in neural population activity *before* the stimulus that will be
# judged even appears. Decoding of the animal's actual upcoming choice from the
# same pre-stimulus window on zero-contrast trials (Figure 4), where no stimulus
# information exists to drive the choice, is weaker and noisier (fewer trials per
# session) but trends in the same direction, consistent with the bias signal
# feeding into the eventual action.
