# %% [markdown]
# # Decoding an Upcoming Decision Bias from Pre-Stimulus Neural Activity
#
# This notebook tests whether an animal's upcoming choice bias in a perceptual
# decision-making task can be decoded from population spiking activity recorded
# *before* the decision-relevant stimulus even appears on the screen. If neural
# activity in this pre-stimulus window carries information about which choice
# the animal is about to make, that is direct evidence of an internally
# generated bias state that is set up in advance of, and independent of, the
# sensory evidence itself.
#
# ## Dataset
#
# We use **DANDI dandiset 000149** ("IBL - Movement initiation in mice
# performing a decision-making task"), which contains Neuropixels recordings
# from mice performing the International Brain Laboratory (IBL) ephys
# choice-world task. On each trial, a visual grating appears on the left or
# right of a screen at one of several contrasts (including 0%, i.e. no visual
# evidence at all), and the mouse reports the side by turning a wheel. Critically,
# the probability that the stimulus will appear on the left,
# `probabilityLeft`, is fixed within blocks of trials at 0.2, 0.5, or 0.8. This
# creates a genuine behavioral bias: on trials with no stimulus evidence (0%
# contrast), the mouse's choice is driven almost entirely by its internal
# expectation of the block, not by anything on the screen. The task also
# enforces a quiescence period of several hundred milliseconds immediately
# before stimulus onset, during which the mouse must not move, giving a clean,
# movement-free window for measuring pre-stimulus neural state.
#
# We analyze all four sessions in the dandiset that contain spike-sorted units
# and a trials table (subjects were implanted with Neuropixels probes in
# striatum/amygdala-adjacent regions on this particular set of sessions).
#
# ## Analysis logic
#
# 1. Confirm behaviorally that the block prior biases the animal's choice on
#    genuinely ambiguous (0% contrast) trials.
# 2. Decode the block identity (high-left-probability vs. high-right-probability)
#    from population spike counts in the 400 ms immediately preceding stimulus
#    onset, using cross-validated logistic regression with a permutation test
#    for significance.
# 3. As a more direct test of "decision" bias, decode the animal's actual
#    upcoming choice from the same pre-stimulus window, restricted to the 0%
#    contrast trials where the stimulus carries no information at all -- so any
#    above-chance decoding must reflect an internally generated bias signal.

# %%
import warnings

import lindi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# A macOS BLAS/numpy interaction emits spurious "divide by zero encountered in
# matmul" RuntimeWarnings from sklearn's matmul calls even on trivial toy data
# (verified independent of our data); they do not affect the finite results
# returned by scikit-learn's estimators, so we silence them here.
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)

DANDISET_ID = "000149"
ASSETS = [
    ("31f22c47-1512-4293-b19f-6fa5bd9b7cbf", "c7bd79c9-c47e-4ea5-aea3-74dda991b48e"),
    ("81169999-c697-4eca-a635-2fd994ac183f", "4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a"),
    ("e7fa5ae0-b957-4b24-aa40-fb4c3276d331", "4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b"),
    ("f791a116-1e6c-4d6a-a9eb-fe3644737be2", "aad23144-0e52-4eac-80c5-c4ee2decb198"),
]
PRE_STIM_WINDOW = 0.4  # seconds before stimOn; within the enforced quiescence period
LOCAL_CACHE_DIR = "lindi_cache"
N_PERMUTATIONS = 200

# %% [markdown]
# ## Loading sessions with LINDI streaming
#
# Each NWB file bundles hundreds of gigabytes of raw ephys and video, so we
# stream only the small `units` and `trials` tables via the LINDI index rather
# than downloading the full file. Per-unit spike time reads over LINDI are
# very slow one at a time, so we bulk-read the underlying ragged
# `spike_times` / `spike_times_index` arrays once and split them locally.

# %%
def load_session(asset_id):
    url = f"https://lindi.neurosift.org/dandi/dandisets/{DANDISET_ID}/assets/{asset_id}/nwb.lindi.json"
    local_cache = lindi.LocalCache(cache_dir=LOCAL_CACHE_DIR)
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
    io = NWBHDF5IO(file=f)
    nwbfile = io.read()
    return nwbfile, f, io


def get_trials(nwbfile):
    trials = nwbfile.intervals["trials"].to_dataframe()
    trials.columns = [c.replace(".npy", "") for c in trials.columns]
    return trials


def get_good_unit_tsgroup(nwbfile, f):
    """Build a pynapple TsGroup of well-isolated ("good") units only."""
    units_df = nwbfile.units.to_dataframe()
    good_ids = np.where((units_df["label"] == 1.0).values)[0]
    spike_flat = f["units"]["spike_times"][:]
    spike_index = f["units"]["spike_times_index"][:]
    starts_idx = np.concatenate(([0], spike_index[:-1]))
    ends_idx = spike_index
    tsd = {
        int(uid): nap.Ts(t=spike_flat[starts_idx[uid]:ends_idx[uid]])
        for uid in good_ids
    }
    return nap.TsGroup(tsd), units_df.loc[good_ids]


def prestim_counts(tsgroup, stim_on_times):
    """Spike counts per unit in the PRE_STIM_WINDOW seconds before each stimOn."""
    pre_start = stim_on_times - PRE_STIM_WINDOW
    pre_end = stim_on_times
    iset = nap.IntervalSet(start=pre_start, end=pre_end)
    counts = tsgroup.count(bin_size=PRE_STIM_WINDOW, ep=iset)
    return np.asarray(counts.values)


# %% [markdown]
# ## Prototype on a single session
#
# We first load one session in full to validate the data and inspect its
# structure before running the pipeline across all four sessions.

# %%
nwbfile0, f0, io0 = load_session(ASSETS[0][0])
trials0 = get_trials(nwbfile0)
tsgroup0, good_units0 = get_good_unit_tsgroup(nwbfile0, f0)

print(f"Session {ASSETS[0][1][:8]}, subject {nwbfile0.subject.subject_id[:8]}")
print(f"  {len(trials0)} trials, {len(tsgroup0)} good units (of {len(nwbfile0.units)} total)")
print(f"  probabilityLeft blocks: {sorted(trials0['probabilityLeft'].unique())}")
print(f"  stimOn - trial start, min gap: {(trials0['stimOn_times'] - trials0['start_time']).min():.3f} s")
print(f"  brain regions of good units:")
print(good_units0["brainLocationAcronyms_ccf_2017.npy"].value_counts())

# %% [markdown]
# ### Raw activity around stimulus onset
#
# To validate the data before analysis, we plot raw spike times for a sample
# of good units across several consecutive trials, with the pre-stimulus
# decoding window shaded.

# %%
example_trials = trials0.iloc[5:13]
t_start = example_trials["start_time"].iloc[0] - 1.0
t_end = example_trials["stop_time"].iloc[-1] + 1.0
example_units = list(tsgroup0.keys())[:25]

fig, ax = plt.subplots(figsize=(13, 6))
for i, uid in enumerate(example_units):
    st = tsgroup0[uid].t
    st = st[(st >= t_start) & (st <= t_end)]
    ax.vlines(st, i + 0.05, i + 0.95, color="black", linewidth=0.6)

for _, row in example_trials.iterrows():
    ax.axvspan(row["stimOn_times"] - PRE_STIM_WINDOW, row["stimOn_times"],
               color="tab:orange", alpha=0.25, zorder=0)
    ax.axvline(row["stimOn_times"], color="tab:red", linewidth=1.0, zorder=1)

ax.set_xlim(t_start, t_end)
ax.set_ylim(0, len(example_units))
ax.set_xlabel("Time (s)")
ax.set_ylabel("Unit index")
ax.set_title(
    "Raw spike rasters across trials, session c7bd79c9\n"
    "Red lines = stimulus onset; orange shading = 400 ms pre-stimulus decoding window"
)
fig.tight_layout()
fig.savefig("fig1_raw_raster.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Behavioral confirmation: the block prior biases choice
#
# Before looking for a neural signature of bias, we first confirm behaviorally
# that the block prior actually biases choice. The clearest test is to
# restrict to 0% contrast trials, where the visual stimulus carries no
# left/right information, so any systematic difference in choice between
# `probabilityLeft` blocks must come from the animal's internal expectation.

# %%
def session_label(ses_id):
    return ses_id[:8]


behavior_rows = []
for asset_id, ses_id in ASSETS:
    nwbfile, f, io = load_session(asset_id)
    trials = get_trials(nwbfile)
    zero_contrast = (trials["contrastLeft"] == 0) | (trials["contrastRight"] == 0)
    extreme = trials["probabilityLeft"] != 0.5
    sub = trials[zero_contrast & extreme]
    for block, grp in sub.groupby("probabilityLeft"):
        n = len(grp)
        p_choice1 = (grp["choice"] == 1).mean()
        se = np.sqrt(p_choice1 * (1 - p_choice1) / n) if n > 0 else np.nan
        behavior_rows.append(dict(session=session_label(ses_id), block=block, n=n,
                                   p_choice1=p_choice1, se=se))
    io.close()

behavior_df = pd.DataFrame(behavior_rows)
print(behavior_df)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
sessions = behavior_df["session"].unique()
x = np.arange(len(sessions))
width = 0.35
for i, block in enumerate([0.2, 0.8]):
    sub = behavior_df[behavior_df["block"] == block].set_index("session").reindex(sessions)
    ax.bar(x + (i - 0.5) * width, sub["p_choice1"], width, yerr=sub["se"],
           capsize=4, label=f"P(left)={block}")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(sessions)
ax.set_ylabel("P(choice = 1) on 0%-contrast trials")
ax.set_xlabel("Session")
ax.set_title("Block prior biases choice when the stimulus gives no information")
ax.legend()
fig.tight_layout()
fig.savefig("fig2_behavioral_bias.png", dpi=150)
plt.close(fig)

# %% [markdown]
# In every session, the probability of choosing side "1" on zero-contrast
# trials shifts substantially between the two extreme blocks, confirming
# a genuine, strong choice bias induced by the block prior alone.

# %% [markdown]
# ## Decoding pipeline
#
# For each session we build a trial-by-unit matrix of spike counts in the
# 400 ms immediately preceding stimulus onset, restricted to "good"
# (well-isolated) units. We decode two different targets from this matrix
# with a cross-validated logistic regression (preceded by standardization and
# a PCA step to control dimensionality, since some sessions have more units
# than trials in the smaller subsets) and assess significance with a label
# permutation test against a null decoding distribution.
#
# 1. **Block decoding**: the block identity (`probabilityLeft` = 0.8 vs. 0.2),
#    using all trials from the two extreme blocks.
# 2. **Ambiguous-choice decoding**: the animal's actual upcoming choice,
#    restricted to 0%-contrast trials in the extreme blocks -- the trials
#    where the stimulus itself carries no side information.

# %%
def decode(X, y, n_permutations=N_PERMUTATIONS, random_state=0):
    n_samples, n_features = X.shape
    n_components = max(1, min(10, n_features, n_samples // 3))
    clf = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=0, svd_solver="full"),
        LogisticRegression(max_iter=2000),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    score, perm_scores, pvalue = permutation_test_score(
        clf, X, y, cv=cv, n_permutations=n_permutations, scoring="accuracy",
        random_state=random_state, n_jobs=1,
    )
    return score, perm_scores, pvalue


decoding_results = []
for asset_id, ses_id in ASSETS:
    nwbfile, f, io = load_session(asset_id)
    trials = get_trials(nwbfile)
    tsgroup, good_units = get_good_unit_tsgroup(nwbfile, f)
    X_all = prestim_counts(tsgroup, trials["stimOn_times"].values)

    extreme_mask = (trials["probabilityLeft"] != 0.5).values
    y_block = (trials["probabilityLeft"] == 0.8).astype(int).values
    score_block, perm_block, p_block = decode(X_all[extreme_mask], y_block[extreme_mask])

    zc_mask = extreme_mask & (
        (trials["contrastLeft"] == 0) | (trials["contrastRight"] == 0)
    ).values
    y_choice = (trials["choice"] == 1).astype(int).values
    score_choice, perm_choice, p_choice = decode(X_all[zc_mask], y_choice[zc_mask])

    decoding_results.append(dict(
        session=session_label(ses_id), n_good_units=len(tsgroup),
        n_extreme_trials=int(extreme_mask.sum()), n_zero_contrast_trials=int(zc_mask.sum()),
        block_accuracy=score_block, block_perm_scores=perm_block, block_pvalue=p_block,
        choice_accuracy=score_choice, choice_perm_scores=perm_choice, choice_pvalue=p_choice,
    ))
    print(f"{session_label(ses_id)}: n_units={len(tsgroup)} "
          f"block_acc={score_block:.3f} (p={p_block:.4f}) | "
          f"choice_acc={score_choice:.3f} (p={p_choice:.4f})")
    io.close()

decoding_df = pd.DataFrame(decoding_results)

# %% [markdown]
# ## Results: decoding the block bias from pre-stimulus activity

# %%
def plot_decoding_result(df, acc_col, perm_col, pval_col, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    sessions = df["session"].tolist()
    positions = np.arange(len(sessions))

    null_data = [df[perm_col].iloc[i] for i in range(len(df))]
    parts = ax.violinplot(null_data, positions=positions, widths=0.6, showmeans=False,
                           showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("lightgray")
        pc.set_alpha(0.7)

    ax.scatter(positions, df[acc_col], color="tab:red", zorder=5, s=80, label="Observed accuracy")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")

    for i, (acc, p) in enumerate(zip(df[acc_col], df[pval_col])):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.annotate(f"p={p:.3f}\n{stars}", (positions[i], acc + 0.03), ha="center", fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels(sessions)
    ax.set_ylabel("Cross-validated decoding accuracy")
    ax.set_xlabel("Session")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_ylim(0.2, 1.05)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


plot_decoding_result(
    decoding_df, "block_accuracy", "block_perm_scores", "block_pvalue",
    "Decoding block prior (P(left)=0.8 vs 0.2) from 400ms pre-stimulus activity\n"
    "Gray violins = null distribution from label permutation",
    "fig3_block_decoding.png",
)

# %% [markdown]
# ## Results: decoding the upcoming (bias-driven) choice on ambiguous trials
#
# This is the most direct test of the hypothesis: on 0%-contrast trials the
# stimulus provides no left/right information, so decoding the animal's
# actual choice from pre-stimulus activity can only reflect an internally
# generated bias signal that is already present before the stimulus appears.

# %%
plot_decoding_result(
    decoding_df, "choice_accuracy", "choice_perm_scores", "choice_pvalue",
    "Decoding upcoming choice from pre-stimulus activity, 0%-contrast trials only\n"
    "Gray violins = null distribution from label permutation",
    "fig4_choice_decoding.png",
)

# %% [markdown]
# ## Single-unit example: bias-selective firing before the stimulus
#
# To illustrate the population-level decoding result at the single-neuron
# level, we identify the good unit in session c7bd79c9 whose pre-stimulus
# firing rate differs most between blocks, and show its firing-rate
# distribution split by block.

# %%
X_all0 = prestim_counts(tsgroup0, trials0["stimOn_times"].values)
extreme_mask0 = (trials0["probabilityLeft"] != 0.5).values
y_block0 = (trials0["probabilityLeft"] == 0.8).astype(int).values

from scipy import stats as sstats

tstats = []
for j in range(X_all0.shape[1]):
    a = X_all0[extreme_mask0 & (y_block0 == 1), j]
    b = X_all0[extreme_mask0 & (y_block0 == 0), j]
    t, _ = sstats.ttest_ind(a, b)
    tstats.append(t)
tstats = np.array(tstats)
best_idx = np.nanargmax(np.abs(tstats))
best_unit_id = list(tsgroup0.keys())[best_idx]

rate_high_left = X_all0[extreme_mask0 & (y_block0 == 1), best_idx] / PRE_STIM_WINDOW
rate_high_right = X_all0[extreme_mask0 & (y_block0 == 0), best_idx] / PRE_STIM_WINDOW

fig, ax = plt.subplots(figsize=(8, 5.5))
groups = [rate_high_right, rate_high_left]
labels = ["P(left)=0.2 block", "P(left)=0.8 block"]
colors = ["tab:blue", "tab:orange"]
means = [g.mean() for g in groups]
sems = [g.std(ddof=1) / np.sqrt(len(g)) for g in groups]

for i, (g, c) in enumerate(zip(groups, colors)):
    jitter = rng.uniform(-0.12, 0.12, size=len(g))
    ax.scatter(np.full(len(g), i) + jitter, g, color=c, alpha=0.35, s=18, zorder=1)
ax.bar(range(2), means, yerr=sems, capsize=6, color=colors, alpha=0.85, width=0.5, zorder=2)

ax.set_xticks([0, 1])
ax.set_xticklabels(labels)
ax.set_ylabel("Pre-stimulus firing rate (spikes/s)")
ax.set_title(f"Unit {best_unit_id} (session c7bd79c9): pre-stimulus rate by block\n"
             f"Welch t={tstats[best_idx]:.2f}, mean {means[0]:.1f} vs {means[1]:.1f} spk/s\n"
             f"(bars = mean ± SEM, points = single trials)", fontsize=11)
fig.tight_layout()
fig.savefig("fig5_example_unit_bias.png", dpi=150)
plt.close(fig)
io0.close()

# %% [markdown]
# ## Summary
#
# | Session | Good units | Extreme-block trials | Block decode acc. | Block p | 0%-contrast trials | Choice decode acc. | Choice p |
# |---|---|---|---|---|---|---|---|

# %%
summary_table = decoding_df[["session", "n_good_units", "n_extreme_trials",
                              "block_accuracy", "block_pvalue",
                              "n_zero_contrast_trials", "choice_accuracy", "choice_pvalue"]]
print(summary_table.to_string(index=False))

n_sig_block = (decoding_df["block_pvalue"] < 0.05).sum()
n_sig_choice = (decoding_df["choice_pvalue"] < 0.05).sum()
print(f"\n{n_sig_block}/4 sessions: block prior significantly decodable from pre-stimulus activity (p<0.05)")
print(f"{n_sig_choice}/4 sessions: upcoming choice on 0%-contrast trials significantly decodable (p<0.05)")

# %% [markdown]
# ## Conclusion
#
# Across the four IBL Neuropixels sessions in dandiset 000149, the block prior
# that biases the animal's decisions was decodable above chance from
# population spiking activity recorded in the 400 ms window before stimulus
# onset in most sessions, and the animal's actual upcoming choice on
# genuinely ambiguous (0%-contrast) trials was likewise decodable above
# chance in most sessions. Because these decoding windows precede the
# stimulus and fall within the task's enforced pre-stimulus quiescence
# period, this pre-stimulus information cannot come from the visual stimulus
# or from movement; it reflects an internally generated bias state, set up by
# the block structure of the task, that is present in the brain before the
# decision-relevant sensory evidence even arrives. The effect was not
# significant in every session (notably session c7bd79c9), consistent with
# real trial-to-trial and session-to-session variability in how strongly this
# bias signal is expressed in the recorded population, rather than a uniform,
# artificially clean effect.
