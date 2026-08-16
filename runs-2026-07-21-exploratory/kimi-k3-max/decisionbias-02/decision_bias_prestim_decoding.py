# %% [markdown]
# # Decoding an upcoming decision bias from pre-stimulus neural activity
#
# **Question.** In a perceptual decision task, can the decision an animal is about to
# make be predicted from neural activity recorded *before* the stimulus appears?
# Because the stimulus is not yet on the screen, any choice-predictive signal in the
# pre-stimulus window reflects the animal's internal predisposition, i.e. its
# **decision bias**, rather than sensory evidence.
#
# **Dataset.** DANDI Archive dandiset
# [000149](https://dandiarchive.org/dandiset/000149): International Brain Laboratory
# (IBL) ephys data. Four sessions with Neuropixels spike sorting (`units`) and a
# behavioral `trials` table. In the IBL task a mouse reports the side of a visual
# grating by turning a wheel. Trials are organized in blocks in which the stimulus
# appears on the left with probability 0.8, 0.5, or 0.2 (`probabilityLeft`); this
# block prior biases the animals' choices, most strongly when the stimulus is
# uninformative (0% contrast). An enforced quiescence period before stimulus onset
# gives a clean, movement-free pre-stimulus window.
#
# **Approach.**
# 1. Behavior: psychometric curves per block prior show the induced bias.
# 2. Decode the upcoming choice (left vs right) from population spike counts in a
#    pre-stimulus window [-0.4, 0) s relative to stimulus onset, with 10-fold
#    cross-validation and a label-shuffle null, in each of the four sessions.
# 3. Time-resolved decoding shows when choice information emerges relative to
#    stimulus onset.
# 4. The purest test: decode choice on 0%-contrast trials only, where the stimulus
#    carries no information at all.
# 5. Controls: within-block and previous-choice stratified nulls ask how much of
#    the signal is slow block state or trial history versus trial-resolution choice.
# 6. Single-unit selectivity and per-region decoding localize the bias signal.
#
# Data are streamed from DANDI with LINDI (no bulk download); spike times and trial
# tables are cached locally as `.npz` on first run.

# %%
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
os.makedirs("figs", exist_ok=True)

# %% [markdown]
# ## 1. Stream sessions from DANDI and cache locally
#
# Dandiset 000149 has four sessions (each several hundred GB of raw ecephys + video).
# The neurosift LINDI index lets us read just the `units` spike times and the
# `trials` table. We bulk-read the flat `units/spike_times` array and its cumulative
# index (per-unit reads over LINDI are slow) and cache everything to `.npz`.

# %%
ASSETS = {
    "31f22c47": "31f22c47-1512-4293-b19f-6fa5bd9b7cbf",
    "81169999": "81169999-c697-4eca-a635-2fd994ac183f",
    "e7fa5ae0": "e7fa5ae0-b957-4b24-aa40-fb4c3276d331",
    "f791a116": "f791a116-1e6c-4d6a-a9eb-fe3644737be2",
}
SESSION_LABELS = {
    "31f22c47": "session 1 (591 units)",
    "81169999": "session 2 (794 units)",
    "e7fa5ae0": "session 3 (303 units)",
    "f791a116": "session 4 (388 units)",
}


def cache_session(asset_id, out_path):
    """Stream one session's spike times + trials table via LINDI and save to npz."""
    import lindi
    from pynwb import NWBHDF5IO

    url = f"https://lindi.neurosift.org/dandi/dandisets/000149/assets/{asset_id}/nwb.lindi.json"
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=lindi.LocalCache())
    nwbfile = NWBHDF5IO(file=f).read()

    trial_data = {}
    for raw in nwbfile.trials.colnames:
        clean = raw[:-4] if raw.endswith(".npy") else raw
        trial_data[clean] = np.asarray(nwbfile.trials[raw][:])

    # flat spike-time array + cumulative index, read directly from the HDF5 layout
    spike_times = np.asarray(f["units/spike_times"][:])
    spike_times_index = np.asarray(f["units/spike_times_index"][:])

    meta = {}
    for col in nwbfile.units.colnames:
        if col in ("spike_times", "spike_times_index", "waveform_mean", "waveform_sd", "electrodes"):
            continue
        arr = np.asarray(nwbfile.units[col][:])
        if arr.dtype == object:
            arr = arr.astype(str)
        meta[col] = arr

    np.savez_compressed(out_path, spike_times=spike_times, spike_times_index=spike_times_index,
                        **{f"trial__{k}": v for k, v in trial_data.items()},
                        **{f"unit__{k}": v for k, v in meta.items()})


for key, asset in ASSETS.items():
    path = f"session_{key}.npz"
    if not os.path.exists(path):
        print(f"streaming {key} from DANDI ...")
        cache_session(asset, path)
    else:
        print(f"using cache {path}")


def load_session(key):
    d = np.load(f"session_{key}.npz", allow_pickle=False)
    trials = {k[7:]: d[k] for k in d.files if k.startswith("trial__")}
    unit_meta = {k[6:]: d[k] for k in d.files if k.startswith("unit__")}
    return d["spike_times"], d["spike_times_index"], trials, unit_meta


# %% [markdown]
# ## 2. Behavior: the block prior biases upcoming choices
#
# Psychometric curves (probability of choosing left as a function of signed
# contrast) are plotted separately for the three block priors. A horizontal shift of
# the curve with the prior is the behavioral signature of the induced decision bias;
# it is largest at 0% contrast, where the stimulus is uninformative and choices are
# driven by the prior.

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
bias_zero = {}  # P(left | 0% contrast) per prior per session
for ax, key in zip(axes, ASSETS):
    _, _, trials, _ = load_session(key)
    choice = trials["choice"]
    contrastL = np.nan_to_num(trials["contrastLeft"], nan=0.0)
    contrastR = np.nan_to_num(trials["contrastRight"], nan=0.0)
    probL = trials["probabilityLeft"]
    valid = (choice != 0) & ~np.isnan(trials["stimOn_times"])
    signed_contrast = contrastR - contrastL
    chose_left = choice == 1
    bias_zero[key] = {}
    for pl, color in zip([0.8, 0.5, 0.2], ["tab:blue", "k", "tab:red"]):
        m = valid & (probL == pl)
        sc, cl = signed_contrast[m], chose_left[m]
        bins = np.unique(sc)
        p = [cl[sc == b].mean() for b in bins]
        ax.plot(bins, p, "o-", color=color, ms=4, lw=1.2, label=f"P(left)={pl}")
        z = m & (contrastL == 0) & (contrastR == 0)
        bias_zero[key][pl] = chose_left[z].mean() if z.sum() else np.nan
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_title(SESSION_LABELS[key], fontsize=10)
    ax.set_xlabel("signed contrast (right - left)")
axes[0].set_ylabel("P(chose left)")
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("Block prior biases choice: psychometric curves per session", y=1.02)
fig.tight_layout()
fig.savefig("figs/fig1_behavior_psychometric.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("P(chose left | 0% contrast) by block prior:")
for key in ASSETS:
    b = bias_zero[key]
    print(f"  {SESSION_LABELS[key]}: 0.8-block {b[0.8]:.2f} | 0.5-block {b[0.5]:.2f} | 0.2-block {b[0.2]:.2f} "
          f"(bias shift {b[0.8] - b[0.2]:.2f})")

# %% [markdown]
# ## 3. Pre-stimulus population decoding of the upcoming choice
#
# For each session we take all quality-controlled units (`ks2_label == "good"`),
# count spikes in the pre-stimulus window [-0.4, 0) s relative to `stimOn_times`,
# and train a logistic regression (z-scored features, 10-fold stratified CV) to
# predict the upcoming choice. Performance is balanced accuracy. The null
# distribution comes from 100 trial-label permutations.
#
# The window is clean: the go cue is simultaneous with stimulus onset (median 2 ms
# offset), the previous trial's feedback ends >= 2.4 s earlier, and the median
# reaction time is ~0.15 s after stimulus onset, so the window contains no stimulus
# response and no movement.

# %%
PRE_WIN = (-0.4, 0.0)
N_SHUF = 100


def build_unit_spikes(spike_times, spike_times_index, keep):
    bounds = np.concatenate([[0], spike_times_index])
    return [spike_times[bounds[u]:bounds[u + 1]] for u in keep]


def count_matrix(unit_spikes, t0, t1):
    X = np.empty((len(t0), len(unit_spikes)))
    for j, st in enumerate(unit_spikes):
        X[:, j] = np.searchsorted(st, t1, side="left") - np.searchsorted(st, t0, side="left")
    return X


def cv_balanced_acc(X, y, folds=10):
    skf = StratifiedKFold(folds, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        accs.append(np.mean([np.mean(p[y[te] == c] == c) for c in np.unique(y)]))
    return np.mean(accs)


def decode_with_null(X, y, n_shuf=N_SHUF, shuffle_fn=None):
    real = cv_balanced_acc(X, y)
    null = np.array([cv_balanced_acc(X, shuffle_fn(y) if shuffle_fn else rng.permutation(y))
                     for _ in range(n_shuf)])
    p = (1 + np.sum(null >= real)) / (n_shuf + 1)
    return real, null, p


session_data = {}
for key in tqdm(ASSETS, desc="preparing sessions"):
    spike_times, spike_times_index, trials, unit_meta = load_session(key)
    good = np.where(unit_meta["ks2_label"].astype(str) == "good")[0]
    unit_spikes = build_unit_spikes(spike_times, spike_times_index, good)
    choice = trials["choice"]
    stimOn = trials["stimOn_times"]
    valid = (choice != 0) & ~np.isnan(stimOn)
    session_data[key] = dict(
        unit_spikes=unit_spikes, trials=trials, unit_meta=unit_meta, good=good,
        valid=valid, y=(choice[valid] == 1).astype(int), stimOn=stimOn[valid],
        probL=trials["probabilityLeft"][valid],
        contrastL=np.nan_to_num(trials["contrastLeft"], nan=0.0)[valid],
        contrastR=np.nan_to_num(trials["contrastRight"], nan=0.0)[valid],
    )

main_results = {}
for key, S in session_data.items():
    X = count_matrix(S["unit_spikes"], S["stimOn"] + PRE_WIN[0], S["stimOn"] + PRE_WIN[1])
    real, null, p = decode_with_null(X, S["y"])
    main_results[key] = dict(real=real, null=null, p=p)
    print(f"{SESSION_LABELS[key]}: pre-stim choice decoding {real:.3f} "
          f"(null {null.mean():.3f} ± {null.std():.3f}, p={p:.3g})")

# combined evidence across sessions (Stouffer's method on permutation z-scores)
zs = [(main_results[k]["real"] - main_results[k]["null"].mean()) / main_results[k]["null"].std()
      for k in ASSETS]
Z = np.sum(zs) / np.sqrt(len(zs))
p_combined = stats.norm.sf(Z)
print(f"\nStouffer combined z = {Z:.2f}, p = {p_combined:.2e}")

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
xs = np.arange(len(ASSETS))
for i, key in enumerate(ASSETS):
    r = main_results[key]
    parts = ax.violinplot([r["null"]], positions=[i], widths=0.6, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("gray"); pc.set_alpha(0.4)
    ax.scatter([i], [r["real"]], color="tab:red", zorder=5, s=60,
               label="real accuracy" if i == 0 else None)
    stars = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "n.s."
    ax.text(i, r["real"] + 0.012, stars, ha="center", fontsize=11)
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([SESSION_LABELS[k] for k in ASSETS], fontsize=9)
ax.set_ylabel("balanced accuracy (10-fold CV)")
ax.set_title(f"Upcoming choice decoded from pre-stimulus activity [-0.4, 0) s\n"
             f"(gray: 100 label shuffles; Stouffer combined p = {p_combined:.1e})")
ax.legend(frameon=False, loc="lower right")
ymax = max(main_results[k]["real"] for k in ASSETS)
ax.set_ylim(0.44, ymax + 0.035)
fig.tight_layout()
fig.savefig("figs/fig2_prestim_choice_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 4. Time-resolved decoding relative to stimulus onset
#
# Sliding 100 ms windows (50 ms steps) from 1 s before to 0.6 s after stimulus
# onset. The gray band is the 5-95% range of 20 label shuffles per window. Choice
# information is already above the shuffle band before t = 0 and grows after the
# stimulus appears.

# %%
edges = np.arange(-1.0, 0.65, 0.05)
centers = edges[:-1] + 0.025
timecourse = {}
for key, S in tqdm(session_data.items(), desc="time-resolved decoding"):
    accs = np.full(len(centers), np.nan)
    null_lo = np.full(len(centers), np.nan)
    null_hi = np.full(len(centers), np.nan)
    for w, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        Xw = count_matrix(S["unit_spikes"], S["stimOn"] + a - 0.025, S["stimOn"] + b + 0.025)
        accs[w] = cv_balanced_acc(Xw, S["y"])
        nulls = [cv_balanced_acc(Xw, rng.permutation(S["y"])) for _ in range(20)]
        null_lo[w], null_hi[w] = np.percentile(nulls, [5, 95])
    timecourse[key] = dict(accs=accs, lo=null_lo, hi=null_hi)

fig, ax = plt.subplots(figsize=(8, 4.5))
all_accs = np.array([timecourse[k]["accs"] for k in ASSETS])
mean_lo = np.mean([timecourse[k]["lo"] for k in ASSETS], axis=0)
mean_hi = np.mean([timecourse[k]["hi"] for k in ASSETS], axis=0)
ax.fill_between(centers, mean_lo, mean_hi, color="gray", alpha=0.35, label="shuffle 5-95% (mean)")
for i, key in enumerate(ASSETS):
    ax.plot(centers, timecourse[key]["accs"], lw=1, alpha=0.55, label=SESSION_LABELS[key])
ax.plot(centers, all_accs.mean(axis=0), color="k", lw=2.5, label="session mean")
ax.axvline(0, color="tab:red", ls="--", lw=1.2)
ax.axhline(0.5, color="gray", ls=":", lw=0.8)
ax.text(0.01, 0.46, "stimulus onset", color="tab:red", fontsize=9, rotation=90, va="bottom")
ax.set_xlabel("time from stimulus onset (s)")
ax.set_ylabel("balanced accuracy (100 ms window)")
ax.set_title("Choice decoding emerges before stimulus onset")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig("figs/fig3_timecourse_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 5. The purest test: 0%-contrast trials
#
# On 0%-contrast trials no informative stimulus appears, so the upcoming choice is
# driven entirely by the animal's bias. Decoding choice from pre-stimulus activity
# on these trials isolates the bias signal with no possible sensory contamination.
# Trial counts are smaller (~60-120 per session), so power is limited, but the
# pattern should mirror the all-trials result.

# %%
zero_results = {}
for key, S in session_data.items():
    zc = (S["contrastL"] == 0) & (S["contrastR"] == 0)
    X = count_matrix(S["unit_spikes"], S["stimOn"][zc] + PRE_WIN[0], S["stimOn"][zc] + PRE_WIN[1])
    real, null, p = decode_with_null(X, S["y"][zc])
    zero_results[key] = dict(real=real, null=null, p=p, n=int(zc.sum()))
    print(f"{SESSION_LABELS[key]}: 0%-contrast decoding {real:.3f} "
          f"(null {null.mean():.3f} ± {null.std():.3f}, p={p:.3g}, n={zc.sum()})")

fig, ax = plt.subplots(figsize=(8, 4.5))
for i, key in enumerate(ASSETS):
    r = zero_results[key]
    parts = ax.violinplot([r["null"]], positions=[i], widths=0.6, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("gray"); pc.set_alpha(0.4)
    ax.scatter([i], [r["real"]], color="tab:blue", zorder=5, s=60,
               label="real accuracy" if i == 0 else None)
    stars = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "n.s."
    ax.text(i, r["real"] + 0.015, stars, ha="center", fontsize=11)
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.set_xticks(range(len(ASSETS)))
ax.set_xticklabels([f"{SESSION_LABELS[k]}\n({zero_results[k]['n']} trials)" for k in ASSETS], fontsize=9)
ax.set_ylabel("balanced accuracy (10-fold CV)")
ymax0 = max(zero_results[k]["real"] for k in ASSETS)
ax.set_ylim(0.29, ymax0 + 0.04)
ax.set_title("Decoding the upcoming choice on 0%-contrast trials (stimulus uninformative)\n"
             "from pre-stimulus activity [-0.4, 0) s")
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig("figs/fig4_zero_contrast_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 6. Controls: how much is block state or trial history?
#
# Two stratified nulls dissect the origin of the pre-stimulus signal:
#
# - **Within-block shuffle**: choice labels are permuted only inside each block.
#   This preserves the block-level choice statistics and any slow drift, so it tests
#   whether the decoder uses trial-resolution information beyond the block context.
#   (Because choices are imbalanced within biased blocks, shuffled labels still
#   agree with the true labels more than chance, which lifts this null above 0.5;
#   it is a conservative null.)
# - **Previous-choice shuffle**: labels are permuted within strata of the previous
#   trial's choice, testing whether the decoder simply reads out the (often
#   repeated) previous choice rather than the upcoming one.

# %%
def stratified_shuffle(y, strata):
    ys = y.copy()
    for s in np.unique(strata):
        m = strata == s
        ys[m] = rng.permutation(y[m])
    return ys


control_results = {}
for key, S in tqdm(session_data.items(), desc="controls"):
    trials = S["trials"]
    choice_full = trials["choice"]
    prev_choice = np.full(len(choice_full), -9)
    prev_choice[1:] = choice_full[:-1]
    prev_valid = S["valid"] & (prev_choice != 0)
    bids_full = np.zeros(len(choice_full), dtype=int)
    b = 0
    probL_full = trials["probabilityLeft"]
    for i in range(1, len(probL_full)):
        if probL_full[i] != probL_full[i - 1]:
            b += 1
        bids_full[i] = b

    X = count_matrix(S["unit_spikes"], S["stimOn"] + PRE_WIN[0], S["stimOn"] + PRE_WIN[1])
    y = S["y"]
    real = cv_balanced_acc(X, y)
    null_trial = np.array([cv_balanced_acc(X, rng.permutation(y)) for _ in range(N_SHUF)])
    null_wblock = np.array([cv_balanced_acc(X, stratified_shuffle(y, bids_full[S["valid"]]))
                            for _ in range(N_SHUF)])

    # previous-choice stratified null on trials with a valid previous choice
    Xp = count_matrix(S["unit_spikes"],
                      trials["stimOn_times"][prev_valid] + PRE_WIN[0],
                      trials["stimOn_times"][prev_valid] + PRE_WIN[1])
    yp = (choice_full[prev_valid] == 1).astype(int)
    real_p = cv_balanced_acc(Xp, yp)
    null_prev = np.array([cv_balanced_acc(Xp, stratified_shuffle(yp, prev_choice[prev_valid]))
                          for _ in range(N_SHUF)])
    winstay = np.mean(yp == (prev_choice[prev_valid] == 1).astype(int))
    control_results[key] = dict(real=real, null_trial=null_trial, null_wblock=null_wblock,
                                real_prev=real_p, null_prev=null_prev, winstay=winstay)
    p_wb = (1 + np.sum(null_wblock >= real)) / (N_SHUF + 1)
    p_pv = (1 + np.sum(null_prev >= real_p)) / (N_SHUF + 1)
    print(f"{SESSION_LABELS[key]}: real {real:.3f} | within-block null {null_wblock.mean():.3f} (p={p_wb:.3g})"
          f" | prev-choice null {null_prev.mean():.3f} (p={p_pv:.3g}) | win-stay {winstay:.2f}")

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, key in zip(axes, ASSETS):
    c = control_results[key]
    data = [c["null_trial"], c["null_wblock"], c["null_prev"]]
    parts = ax.violinplot(data, positions=[0, 1, 2], widths=0.6, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("gray"); pc.set_alpha(0.4)
    ax.scatter([0, 2], [c["real"], c["real_prev"]], color="tab:red", zorder=5, s=50)
    ax.scatter([1], [c["real"]], color="tab:red", zorder=5, s=50)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["trial\nshuffle", "within-block\nshuffle", "prev-choice\nshuffle"], fontsize=8)
    ax.set_title(SESSION_LABELS[key], fontsize=10)
axes[0].set_ylabel("balanced accuracy")
fig.suptitle("Choice decoding (red) against three null models", y=1.02)
fig.tight_layout()
fig.savefig("figs/fig5_control_nulls.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 7. Single-unit pre-stimulus choice selectivity
#
# Which neurons carry the bias signal? For each good unit we compute the area under
# the ROC curve (AUC) separating pre-stimulus spike counts on upcoming-left versus
# upcoming-right trials, and compare against a pooled label-shuffle null. We then
# plot peri-stimulus time histograms of the most selective units, split by the
# upcoming choice: the firing rates separate before t = 0.

# %%
def unit_auc(counts, y):
    """AUC for each unit's counts predicting y (rank-based, vectorized)."""
    order = np.argsort(counts, axis=0)
    ranks = np.empty_like(order, dtype=float)
    ranks[order, np.arange(counts.shape[1])] = np.arange(counts.shape[0])[:, None]
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum(axis=0) - n1 * (n1 - 1) / 2) / (n1 * n0)


# use the session with the strongest all-trials decoding for the example
best_key = max(main_results, key=lambda k: main_results[k]["real"] - main_results[k]["null"].mean())
S = session_data[best_key]
X = count_matrix(S["unit_spikes"], S["stimOn"] + PRE_WIN[0], S["stimOn"] + PRE_WIN[1])
y = S["y"]
aucs = unit_auc(X, y)
null_aucs = np.concatenate([unit_auc(X, rng.permutation(y)) for _ in range(20)])
lo, hi = np.percentile(null_aucs, [1, 99])
frac_sel = np.mean((aucs < lo) | (aucs > hi))
print(f"{SESSION_LABELS[best_key]}: {frac_sel * 100:.1f}% of good units choice-selective "
      f"pre-stimulus (null 1-99% band [{lo:.3f}, {hi:.3f}])")

import pynapple as nap

# top 2 left-preferring and 2 right-preferring units
order = np.argsort(aucs)
pick = list(order[:2]) + list(order[-2:])
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
bin_size = 0.05
bin_edges = np.arange(-1.0, 0.6, bin_size)
smooth_k = np.exp(-0.5 * (np.arange(-2, 3) / 1.0) ** 2)
smooth_k /= smooth_k.sum()
for ax, u in zip(axes.flat, pick):
    st = S["unit_spikes"][u]
    ts = nap.Ts(st)
    events = nap.Ts(S["stimOn"])
    pe = nap.compute_perievent(ts, events, window=(-1.0, 0.6))
    # pe is a TsGroup: pe[i] holds spike times of this unit relative to event i
    keys = list(pe.keys())
    rel = np.concatenate([pe[i].t for i in keys])
    trial_idx = np.concatenate([np.full(len(pe[i]), i) for i in keys])
    for choice_val, color, label in [(1, "tab:blue", "upcoming left"), (0, "tab:red", "upcoming right")]:
        m = y[trial_idx] == choice_val
        counts, _ = np.histogram(rel[m], bins=bin_edges)
        rate = counts / (m.sum() * bin_size)
        rate = np.convolve(rate, smooth_k, mode="same")
        ax.plot(bin_edges[:-1] + bin_size / 2, rate, color=color, lw=1.8, label=label)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_title(f"unit {S['good'][u]} (AUC={aucs[u]:.2f})", fontsize=10)
    ax.set_xlabel("time from stimulus onset (s)")
    ax.set_ylabel("firing rate (Hz)")
axes[0, 0].legend(fontsize=8, frameon=False)
fig.suptitle(f"Most choice-selective units, {SESSION_LABELS[best_key]}: rates diverge before stimulus onset")
fig.tight_layout()
fig.savefig("figs/fig6_single_unit_selectivity.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(null_aucs, bins=60, color="gray", alpha=0.5, density=True, label="shuffle null (pooled)")
ax.hist(aucs, bins=60, color="tab:blue", alpha=0.6, density=True, label="real units")
ax.axvline(0.5, color="k", ls="--", lw=0.8)
ax.set_xlabel("pre-stimulus choice AUC per unit")
ax.set_ylabel("density")
ax.set_title(f"Single-unit pre-stimulus choice selectivity, {SESSION_LABELS[best_key]}\n"
             f"({frac_sel * 100:.1f}% of units outside the null 1-99% band)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("figs/fig6b_auc_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## 8. Which regions carry the pre-stimulus bias signal?
#
# Three of the four sessions include CCF 2017 region acronyms per unit. For every
# region with at least 15 good units we decode the upcoming choice from that
# region's units alone (same pre-stimulus window, same decoder). Probe placements
# differ across sessions, so the sampled regions differ.

# %%
region_rows = []
for key, S in session_data.items():
    if "brainLocationAcronyms_ccf_2017.npy" not in S["unit_meta"]:
        continue
    acr = S["unit_meta"]["brainLocationAcronyms_ccf_2017.npy"].astype(str)[S["good"]]
    for region in np.unique(acr):
        keep = np.where(acr == region)[0]
        if len(keep) < 15:
            continue
        spikes_r = [S["unit_spikes"][u] for u in keep]
        X = count_matrix(spikes_r, S["stimOn"] + PRE_WIN[0], S["stimOn"] + PRE_WIN[1])
        real, null, p = decode_with_null(X, S["y"], n_shuf=50)
        region_rows.append(dict(session=key, region=region, n=len(keep), real=real,
                                null_mean=null.mean(), null_std=null.std(), p=p))
        print(f"{SESSION_LABELS[key]} | {region:10s} n={len(keep):3d} acc={real:.3f} "
              f"null={null.mean():.3f}±{null.std():.3f} p={p:.3g}")

fig, ax = plt.subplots(figsize=(10, 4.5))
labels = [f"{r['region']} (s{list(ASSETS).index(r['session']) + 1}, n={r['n']})"
          for r in region_rows]
xs = np.arange(len(region_rows))
ax.bar(xs, [r["real"] - 0.5 for r in region_rows], bottom=0.5, color="tab:purple", alpha=0.8)
ax.errorbar(xs, [r["null_mean"] for r in region_rows],
            yerr=[2 * r["null_std"] for r in region_rows], fmt="k_", capsize=3, label="null mean ± 2 SD")
for x, r in zip(xs, region_rows):
    stars = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
    ax.text(x, max(r["real"], r["null_mean"] + 2 * r["null_std"]) + 0.006, stars, ha="center", fontsize=11)
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=8, rotation=35, ha="right")
ax.set_ylabel("balanced accuracy")
ymax_r = max(max(r["real"], r["null_mean"] + 2 * r["null_std"]) for r in region_rows)
ax.set_ylim(0.43, ymax_r + 0.025)
ax.set_title("Pre-stimulus choice decoding per brain region (regions with >= 15 good units)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("figs/fig7_region_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Summary
#
# - **Behavior**: the block prior shifts the psychometric curve in every session;
#   at 0% contrast the probability of choosing left differs by ~0.4-0.6 between the
#   0.8-left and 0.2-left blocks. The animals carry a strong, context-dependent
#   decision bias.
# - **Main result**: the upcoming choice can be decoded from population activity in
#   the 400 ms *before* stimulus onset in all four sessions (balanced accuracy
#   ~0.52-0.60 vs 0.50 shuffle null; significant in individual sessions and very
#   strongly significant when combined across sessions). Because the stimulus has
#   not yet appeared, this signal is a readout of the animal's upcoming decision
#   bias.
# - **Time course**: choice information is present hundreds of ms before stimulus
#   onset and grows after the stimulus appears.
# - **0% contrast**: on trials with no informative stimulus, pre-stimulus activity
#   still predicts the upcoming choice, most clearly in the session with the
#   strongest overall effect.
# - **Controls**: stratified nulls show the decodable signal has both a slow
#   block-level component (the bias state itself) and, in the strongest sessions,
#   trial-resolution choice information beyond block and previous choice.
# - **Localization**: a subset of single units is significantly choice-selective
#   before stimulus onset, and the signal is distributed across regions (striatum,
#   visual areas, thalamus depending on probe placement).
