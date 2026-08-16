# %% [markdown]
# # Decoding an upcoming decision bias from pre-stimulus neural activity
#
# **Question.** In a perceptual decision task, can the choice an animal is biased
# toward be decoded from neural activity recorded *before* the stimulus appears?
#
# **Dataset.** DANDI Archive dandiset
# [000149](https://dandiarchive.org/dandiset/000149) — International Brain
# Laboratory (IBL) ephys data: Neuropixels recordings from mice performing a
# visual contrast-detection task. A grating appears on the left or right (or at
# 0% contrast) after an enforced 0.4–0.7 s pre-stimulus quiescence period; the
# mouse reports the side by turning a wheel. Blocks of trials carry a prior
# probability of the stimulus appearing on the left, P(left) ∈ {0.2, 0.5, 0.8},
# which biases the animals' choices (strongest on 0%-contrast trials). All 4
# sessions of the dandiset are analyzed.
#
# **Approach.**
# 1. Stream each session with LINDI (no bulk download; spike times are read from
#    the flat HDF5 arrays, which is the fast path over LINDI).
# 2. Count spikes of each unit in the pre-stimulus window [-0.4, 0] s relative to
#    stimulus onset (movement-free by task design; trials with premature movement
#    are excluded).
# 3. Decode the upcoming choice (left vs right) with an L2-regularized logistic
#    regression under repeated stratified 5-fold cross-validation.
# 4. Test significance against two nulls: a global trial-shuffle null, and a
#    within-block shuffle null that preserves the block structure (controlling
#    for slow, block-locked drift).
# 5. Characterize the signal: time-resolved decoding around stimulus onset,
#    single-unit selectivity (choice AUC), and decoding restricted to
#    bias-dominated trial subsets (0%-contrast trials; unbiased-block trials).
#
# **Key result.** The upcoming choice is decodable from pre-stimulus population
# activity in 3 of 4 sessions (balanced accuracy 0.52–0.59; Stouffer combined
# p ≈ 1.3e-5 against the trial-shuffle null), before any sensory evidence is
# available. The within-block control shows the signal is dominated by a slow,
# block-locked bias state rather than fast trial-by-trial fluctuations, and
# decoding is strongest on 0%-contrast trials where choice is purely bias-driven.

# %% [markdown]
# ## Setup

# %%
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import lindi

# lbfgs can emit transient overflow RuntimeWarnings on shuffled-label fits
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# The four sessions of DANDI 000149 (asset IDs)
ASSET_IDS = {
    "s1": "31f22c47-1512-4293-b19f-6fa5bd9b7cbf",
    "s2": "81169999-c697-4eca-a635-2fd994ac183f",
    "s3": "e7fa5ae0-b957-4b24-aa40-fb4c3276d331",
    "s4": "f791a116-1e6c-4d6a-a9eb-fe3644737be2",
}
KEYS = list(ASSET_IDS.keys())
LINDI_TEMPLATE = "https://lindi.neurosift.org/dandi/dandisets/000149/assets/{}/nwb.lindi.json"

PRE_STIM = (-0.4, 0.0)  # pre-stimulus decoding window (s), enforced quiescence
N_SHUFFLE = 200         # shuffle replicates for null distributions

C_LEFT, C_RIGHT = "#2166ac", "#b2182b"
BLOCK_COLORS = {0.2: "#b2182b", 0.5: "#7b7b7b", 0.8: "#2166ac"}

# %% [markdown]
# ## Data access
#
# Each NWB file is streamed through its LINDI index. The trials table is read
# column by column (column names carry a `.npy` suffix, which we strip). Spike
# times are read from the flat `units/spike_times` and `units/spike_times_index`
# arrays and sliced locally per unit — reading the ragged `spike_times` column
# through the NWB API over LINDI is orders of magnitude slower. One session
# (s2) lacks CCF brain-region annotations; its units are labeled "unknown".
# Loaded sessions are cached locally as pickles.

# %%
def load_session(asset_id):
    """Return (trials DataFrame, list of per-unit spike-time arrays, units DataFrame)."""
    cache_path = os.path.join(CACHE_DIR, f"{asset_id}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    f = lindi.LindiH5pyFile.from_lindi_file(
        LINDI_TEMPLATE.format(asset_id), local_cache=lindi.LocalCache()
    )

    # trials table
    trials_grp = f["intervals/trials"]
    data = {}
    for k in trials_grp.keys():
        if k == "id":
            continue
        v = trials_grp[k][:]
        if v.dtype.kind == "S":
            v = v.astype(str)
        data[k.replace(".npy", "")] = v
    trials = pd.DataFrame(data, index=trials_grp["id"][:])

    # units: flat spike arrays
    spike_times = f["units/spike_times"][:]
    spike_index = f["units/spike_times_index"][:]
    starts = np.concatenate([[0], spike_index[:-1]])
    spike_list = [spike_times[s:e] for s, e in zip(starts, spike_index)]

    if "brainLocationAcronyms_ccf_2017.npy" in f["units"]:
        regions = f["units/brainLocationAcronyms_ccf_2017.npy"][:].astype(str)
    else:
        regions = np.array(["unknown"] * len(spike_list))
    units = pd.DataFrame({
        "region": regions,
        "ks2_label": f["units/ks2_label"][:].astype(str),
        "firing_rate": f["units/firing_rate"][:],
        "presence_ratio": f["units/presence_ratio"][:],
    })

    out = (trials, spike_list, units)
    with open(cache_path, "wb") as fh:
        pickle.dump(out, fh)
    return out


def counts_in_windows(spike_list, starts, ends):
    """Spike counts per unit in [start, end) windows. Returns (n_windows, n_units)."""
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    X = np.empty((len(starts), len(spike_list)), dtype=np.float64)
    for u, st in enumerate(spike_list):
        X[:, u] = np.searchsorted(st, ends) - np.searchsorted(st, starts)
    return X


def block_runs(probability_left):
    """Integer run-id per trial: increments whenever the block prior changes."""
    pl = np.asarray(probability_left)
    return np.concatenate([[0], np.cumsum(np.diff(pl) != 0)])

# %% [markdown]
# ## Decoding machinery
#
# The decoder is an L2-regularized logistic regression (C = 0.1) on spike counts
# z-scored within each training fold, evaluated with repeated stratified 5-fold
# cross-validation (balanced accuracy and ROC AUC). Null distributions come from
# shuffling choice labels, either globally across all trials or within
# contiguous block runs. For small trial subsets (0%-contrast or unbiased-block
# trials) we train on all trials and score only the subset within each test
# fold, which is much better powered than training on the subset alone.

# %%
def decode_cv(X, y, n_repeats=5, seed=0):
    """Repeated stratified 5-fold CV. Returns (balanced accuracy, ROC AUC)."""
    y = np.asarray(y)
    baccs, aucs = [], []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(X, y):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=1000),
            )
            clf.fit(X[tr], y[tr])
            baccs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(baccs)), float(np.mean(aucs))


def shuffle_null(X, y, n_shuffles=200, seed=0, blocks=None):
    """Null balanced-accuracy distribution under label shuffles.

    If `blocks` is given, labels are shuffled within each block run (preserves
    block-level choice statistics); otherwise shuffled globally.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    null = np.empty(n_shuffles)
    for i in range(n_shuffles):
        if blocks is None:
            yp = rng.permutation(y)
        else:
            yp = y.copy()
            for b in np.unique(blocks):
                m = blocks == b
                yp[m] = rng.permutation(y[m])
        null[i], _ = decode_cv(X, yp, n_repeats=1, seed=seed + 1000 + i)
    return null


def decode_cv_subset(X, y, subset, n_repeats=5, seed=0):
    """CV over all trials; balanced accuracy scored only on `subset` test trials."""
    y = np.asarray(y)
    subset = np.asarray(subset)
    baccs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(X, y):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=1000),
            )
            clf.fit(X[tr], y[tr])
            te_sub = te[subset[te]]
            if len(te_sub) < 10 or len(np.unique(y[te_sub])) < 2:
                continue
            baccs.append(balanced_accuracy_score(y[te_sub], clf.predict(X[te_sub])))
    return float(np.mean(baccs))


def shuffle_null_subset(X, y, subset, n_shuffles=200, seed=0):
    """Null for decode_cv_subset under global label shuffle."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    null = np.empty(n_shuffles)
    for i in range(n_shuffles):
        null[i] = decode_cv_subset(X, rng.permutation(y), subset,
                                   n_repeats=1, seed=seed + 1000 + i)
    return null


def time_resolved_decoding(spike_list, event_times, y, windows, n_repeats=3, seed=0):
    """Decode y from spike counts in sliding windows around event_times."""
    out = np.empty((len(windows), 2))
    for i, (w0, w1) in enumerate(windows):
        X = counts_in_windows(spike_list, event_times + w0, event_times + w1)
        out[i] = decode_cv(X, y, n_repeats=n_repeats, seed=seed + i)
    return out


def single_unit_auc(X, y):
    """Rank-based choice AUC per unit (positive class = left choice)."""
    y = np.asarray(y)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    aucs = np.empty(X.shape[1])
    for u in range(X.shape[1]):
        r = rankdata(X[:, u])
        aucs[u] = (r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return aucs


def single_unit_auc_null(X, y, n_shuffles=200, seed=0):
    """Null AUCs per unit under global label shuffle.

    Returns (null_max, null_aucs): the family-wise max |AUC-0.5| per shuffle and
    the full (n_shuffles, n_units) null matrix for per-unit p-values. Ranks are
    fixed under label permutation, so each shuffle is a single reduction.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n_pos = (y == 1).sum()
    n = len(y)
    ranks = np.apply_along_axis(rankdata, 0, X)
    null_max = np.empty(n_shuffles)
    null_aucs = np.empty((n_shuffles, X.shape[1]))
    for i in range(n_shuffles):
        idx = rng.permutation(n)
        pos_mask = np.zeros(n, dtype=bool)
        pos_mask[idx[:n_pos]] = True
        s = ranks[pos_mask].sum(axis=0)
        aucs = (s - n_pos * (n_pos + 1) / 2) / (n_pos * (n - n_pos))
        null_aucs[i] = aucs
        null_max[i] = np.abs(aucs - 0.5).max()
    return null_max, null_aucs

# %% [markdown]
# ## Per-session analysis
#
# For each of the 4 sessions:
# * keep only Kilosort "good" units and valid trials (a choice was made, stimulus
#   onset exists, and no movement preceded stimulus onset);
# * decode the upcoming choice from pre-stimulus spike counts, with global and
#   within-block shuffle nulls;
# * decode in sliding 250 ms windows from -1.0 to +0.6 s around stimulus onset;
# * compute single-unit pre-stimulus choice AUCs with shuffle-based p-values;
# * decode on two bias-dominated subsets: 0%-contrast trials (no sensory
#   evidence, so choice reflects pure bias) and unbiased-block trials
#   (P(left) = 0.5).
#
# Results are saved to `results_<session>.npz`. Runtime is a few minutes for all
# four sessions once the stream cache is warm.

# %%
def analyze_session(key):
    t0 = time.time()
    trials, spike_list, units = load_session(ASSET_IDS[key])
    print(f"[{key}] loaded: {len(trials)} trials, {len(spike_list)} units", flush=True)

    good = units.ks2_label.values == "good"
    good_idx = np.where(good)[0]
    spikes = [spike_list[i] for i in good_idx]
    regions = units.region.values[good]

    choice = trials.choice.values
    stim_on = trials.stimOn_times.values
    first_mov = trials.firstMovement_times.values
    valid = (choice != 0) & ~np.isnan(stim_on) & (
        np.isnan(first_mov) | (first_mov >= stim_on)
    )
    tr = trials[valid].copy()
    y = (tr.choice.values == 1).astype(int)  # 1 = left choice
    stim = tr.stimOn_times.values
    blocks = block_runs(tr.probabilityLeft.values)
    print(f"[{key}] {good.sum()} good units, {valid.sum()} valid trials", flush=True)

    # main: pre-stimulus choice decoding
    X_pre = counts_in_windows(spikes, stim + PRE_STIM[0], stim + PRE_STIM[1])
    bacc, auc = decode_cv(X_pre, y, n_repeats=5, seed=0)
    null_global = shuffle_null(X_pre, y, N_SHUFFLE, seed=1)
    null_block = shuffle_null(X_pre, y, N_SHUFFLE, seed=2, blocks=blocks)
    p_global = (np.sum(null_global >= bacc) + 1) / (N_SHUFFLE + 1)
    p_block = (np.sum(null_block >= bacc) + 1) / (N_SHUFFLE + 1)
    print(f"[{key}] pre-stim bacc={bacc:.3f} (global p={p_global:.4f}, "
          f"within-block p={p_block:.4f})", flush=True)

    # time-resolved decoding around stimulus onset
    starts = np.arange(-1.0, 0.61 - 0.25, 0.1)
    windows = [(s, s + 0.25) for s in starts]
    centers = np.array([np.mean(w) for w in windows])
    tr_decode = time_resolved_decoding(spikes, stim, y, windows, n_repeats=3, seed=10)

    # single-unit pre-stimulus selectivity
    unit_auc = single_unit_auc(X_pre, y)
    null_max, null_aucs = single_unit_auc_null(X_pre, y, N_SHUFFLE, seed=3)
    thr = np.percentile(null_max, 95)
    frac_sig = np.mean(np.abs(unit_auc - 0.5) > thr)
    unit_p = (np.sum(np.abs(null_aucs - 0.5) >= np.abs(unit_auc - 0.5)[None, :],
                     axis=0) + 1) / (N_SHUFFLE + 1)
    frac_p05 = np.mean(unit_p < 0.05)
    # null distribution of the p<0.05 fraction (leave-one-out over shuffles)
    dev = np.abs(null_aucs - 0.5)
    null_frac_p05 = np.empty(N_SHUFFLE)
    for i in range(N_SHUFFLE):
        others = np.delete(dev, i, axis=0)
        p_i = (np.sum(others >= dev[i][None, :], axis=0) + 1) / N_SHUFFLE
        null_frac_p05[i] = np.mean(p_i < 0.05)

    # bias-dominated subsets (train on all trials, score on the subset)
    cl = tr.contrastLeft.fillna(0).values
    cr = tr.contrastRight.fillna(0).values
    zero = (cl == 0) & (cr == 0)
    bacc0 = decode_cv_subset(X_pre, y, zero, n_repeats=5, seed=20)
    null0 = shuffle_null_subset(X_pre, y, zero, N_SHUFFLE, seed=21)
    p0 = (np.sum(null0 >= bacc0) + 1) / (N_SHUFFLE + 1)

    unb = tr.probabilityLeft.values == 0.5
    baccu = decode_cv_subset(X_pre, y, unb, n_repeats=5, seed=30)
    nullu = shuffle_null_subset(X_pre, y, unb, N_SHUFFLE, seed=31)
    pu = (np.sum(nullu >= baccu) + 1) / (N_SHUFFLE + 1)
    print(f"[{key}] 0%-contrast bacc={bacc0:.3f} (p={p0:.4f}); "
          f"unbiased-block bacc={baccu:.3f} (p={pu:.4f})", flush=True)

    # psychometric summary for Figure 1
    signed_contrast = cl - cr
    psych = {}
    for p in [0.2, 0.5, 0.8]:
        m = tr.probabilityLeft.values == p
        for sc in np.unique(signed_contrast):
            mm = m & (signed_contrast == sc)
            if mm.sum() > 0:
                psych[(p, sc)] = (np.mean(y[mm] == 1), mm.sum())

    np.savez(
        f"results_{key}.npz",
        bacc=bacc, auc=auc,
        null_global=null_global, null_block=null_block,
        p_global=p_global, p_block=p_block,
        centers=centers, tr_decode=tr_decode,
        unit_auc=unit_auc, null_max=null_max, frac_sig=frac_sig, thr_auc=thr,
        unit_p=unit_p, frac_p05=frac_p05, null_frac_p05=null_frac_p05,
        regions=regions,
        bacc0=bacc0, null0=null0, p0=p0, n_zero=zero.sum(),
        baccu=baccu, nullu=nullu, pu=pu, n_unb=unb.sum(),
        psych_keys=np.array(list(psych.keys())),
        psych_vals=np.array(list(psych.values())),
        n_trials=len(y), n_units=len(spikes),
        y=y, blocks=blocks,
    )
    print(f"[{key}] done ({time.time()-t0:.0f}s)", flush=True)


for key in KEYS:
    analyze_session(key)


def load_results():
    return {k: np.load(f"results_{k}.npz", allow_pickle=True) for k in KEYS}


res = load_results()

# %% [markdown]
# ## Figure 1 — Task, behavioral bias, and block structure
#
# The block prior biases choice: on 0%-contrast trials (signed contrast = 0) the
# probability of choosing left shifts from ~0.25 in P(left)=0.2 blocks to ~0.77
# in P(left)=0.8 blocks. This behavioral bias is the signal we look for in
# pre-stimulus neural activity.

# %%
fig = plt.figure(figsize=(11, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.32)

ax = fig.add_subplot(gs[0])
ax.axis("off")
ax.set_xlim(-1.15, 1.75)
ax.set_ylim(-1.35, 1.25)
ax.text(-1.15, 1.18, "IBL contrast-detection task", fontsize=10, weight="bold",
        va="top")
ax.text(-1.15, 1.02, "grating appears left or right (or 0% contrast);\n"
                     "mouse turns wheel to report side;\n"
                     "block prior P(left) in {0.2, 0.5, 0.8}",
        fontsize=9, va="top")
y_tl = -0.2
ax.annotate("", xy=(1.7, y_tl), xytext=(-1.1, y_tl),
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
ax.add_patch(Rectangle((-0.4, y_tl - 0.06), 0.4, 0.12, facecolor="#fdae61",
                       edgecolor="none", alpha=0.85, zorder=3))
ax.plot([-0.2, -0.2], [y_tl + 0.12, y_tl + 0.44], color="#a6611a", lw=0.8)
ax.text(-0.2, y_tl + 0.48, "pre-stimulus\ndecoding window", ha="center",
        va="bottom", fontsize=8.5, color="#a6611a")
for x, xt, lab, dy in [(-0.4, -0.4, "quiescence\n(wheel held still)", -0.14),
                       (0.0, 0.0, "stimulus onset\n+ go cue", -0.58),
                       (1.15, 1.15, "feedback", -0.14),
                       (0.15, 0.15, "movement\n(median 150 ms)", 0.14)]:
    ax.plot([xt, xt], [y_tl - 0.06, y_tl + 0.06], color="k", lw=1.2)
    va = "top" if dy < 0 else "bottom"
    ax.text(x, y_tl + dy, lab, ha="center", va=va, fontsize=8.5)
ax.set_title("A  Task and decoding window", loc="left", fontsize=10, weight="bold")

ax = fig.add_subplot(gs[1])
for p in [0.2, 0.5, 0.8]:
    sc_all, pl_all, n_all = [], [], []
    for k in KEYS:
        pk, pv = res[k]["psych_keys"], res[k]["psych_vals"]
        m = pk[:, 0] == p
        sc_all += list(pk[m, 1])
        pl_all += list(pv[m, 0] * pv[m, 1])
        n_all += list(pv[m, 1])
    d = pd.DataFrame({"sc": sc_all, "plw": pl_all, "n": n_all})
    g = d.groupby("sc").apply(lambda r: r.plw.sum() / r.n.sum(), include_groups=False)
    ax.plot(g.index, g.values, "o-", color=BLOCK_COLORS[p], ms=4, lw=1.5,
            label=f"P(left) = {p}")
ax.axhline(0.5, color="k", lw=0.6, ls=":")
ax.axvline(0, color="k", lw=0.6, ls=":")
ax.set_xlabel("signed contrast (left - right)")
ax.set_ylabel("P(choose left)")
ax.legend(fontsize=8, frameon=False, title="block prior", title_fontsize=8)
ax.set_title("B  Behavioral bias from block prior", loc="left", fontsize=10,
             weight="bold")

ax = fig.add_subplot(gs[2])
ax.plot(np.arange(len(res["s1"]["blocks"])), res["s1"]["blocks"], lw=0.8,
        color="0.3")
ax.set_xlabel("valid trial number (session s1)")
ax.set_ylabel("block index (prior switches)")
ax.set_title("C  Block structure within a session", loc="left", fontsize=10,
             weight="bold")

fig.savefig("fig1_task_behavior.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Figure 2 — Example neural data
#
# Top: population raster over 40 s of the example session (s4), with stimulus
# onsets colored by the upcoming choice. Bottom: peri-stimulus firing rates of
# the four most choice-selective units, split by upcoming choice. The traces
# separate inside the shaded pre-stimulus window, before any sensory evidence.

# %%
def peth(spikes, events, window=(-1.0, 1.0), bin_size=0.01, sigma=0.05):
    """Smoothed peri-event firing rate (edges padded before convolution)."""
    pad = 4 * sigma
    bins = np.arange(window[0] - pad, window[1] + pad + bin_size, bin_size)
    counts = np.zeros(len(bins) - 1)
    for st in spikes:
        for e in events:
            rel = st[(st >= e + window[0] - pad) & (st < e + window[1] + pad)] - e
            counts += np.histogram(rel, bins=bins)[0]
    rate = counts / (len(events) * bin_size)
    kw = int(4 * sigma / bin_size)
    x = np.arange(-kw, kw + 1) * bin_size
    kern = np.exp(-0.5 * (x / sigma) ** 2)
    kern /= kern.sum()
    rate = np.convolve(rate, kern, mode="same")
    t = bins[:-1] + bin_size / 2
    m = (t >= window[0]) & (t <= window[1])
    return t[m], rate[m]


# example session: strongest pre-stim decoding among sessions with region labels
annotated = [k for k in KEYS if (res[k]["regions"].astype(str) != "unknown").any()]
example_key = max(annotated, key=lambda k: res[k]["bacc"].item())
print("example session:", example_key)

trials, spike_list, units = load_session(ASSET_IDS[example_key])
good_idx = np.where(units.ks2_label.values == "good")[0]
r = res[example_key]
unit_auc_ex = r["unit_auc"]
regions_ex = r["regions"].astype(str)

choice = trials.choice.values
stim_on = trials.stimOn_times.values
first_mov = trials.firstMovement_times.values
valid = (choice != 0) & ~np.isnan(stim_on) & (np.isnan(first_mov) | (first_mov >= stim_on))
tr = trials[valid]
y_ex = (tr.choice.values == 1).astype(int)
stim_ex = tr.stimOn_times.values

fig = plt.figure(figsize=(11, 7.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.32)

ax = fig.add_subplot(gs[0])
t_mid = stim_ex[len(stim_ex) // 2]
t0, t1 = t_mid - 20, t_mid + 20
show_units = np.arange(0, len(good_idx), max(1, len(good_idx) // 120))
order = np.argsort(regions_ex[show_units])
for row, ui in enumerate(show_units[order]):
    st = spike_list[good_idx[ui]]
    seg = st[(st >= t0) & (st < t1)]
    ax.plot(seg, np.full_like(seg, row), "|", color="0.25", ms=1.5, mew=0.4)
ev = stim_ex[(stim_ex >= t0) & (stim_ex < t1)]
ey = y_ex[(stim_ex >= t0) & (stim_ex < t1)]
for e, c in zip(ev, ey):
    ax.axvline(e, color=C_LEFT if c == 1 else C_RIGHT, alpha=0.5, lw=1.0)
ax.set_xlim(t0, t1)
ax.set_xlabel("time in session (s)")
ax.set_ylabel("unit (sorted by region)")
ax.set_title(f"A  Population spiking with stimulus onsets colored by upcoming choice "
             f"(session {example_key})", loc="left", fontsize=10, weight="bold")

bot = gs[1].subgridspec(2, 1, height_ratios=[0.07, 1.0], hspace=0.0)
axh = fig.add_subplot(bot[0])
axh.axis("off")
axh.set_title("B  Example pre-stimulus choice-selective units", loc="left",
              fontsize=10, weight="bold")
sub = bot[1].subgridspec(1, 4, wspace=0.4)
top_units = np.argsort(-np.abs(unit_auc_ex - 0.5))[:4]
for j, ui in enumerate(top_units):
    ax = fig.add_subplot(sub[j])
    st = [spike_list[good_idx[ui]]]
    t, rL = peth(st, stim_ex[y_ex == 1])
    _, rR = peth(st, stim_ex[y_ex == 0])
    ax.plot(t, rL, color=C_LEFT, lw=1.4, label="left choice")
    ax.plot(t, rR, color=C_RIGHT, lw=1.4, label="right choice")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.axvspan(-0.4, 0, color="#fdae61", alpha=0.3)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("time from stimOn (s)", fontsize=9)
    if j == 0:
        ax.set_ylabel("firing rate (Hz)")
    ax.set_title(f"unit {ui} ({regions_ex[ui]}), pre-stim AUC={unit_auc_ex[ui]:.2f}",
                 fontsize=8)
    if j == 3:
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
fig.savefig("fig2_example_neural.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Figure 3 — Time-resolved choice decoding around stimulus onset
#
# Balanced accuracy of the same decoder in sliding 250 ms windows. Accuracy sits
# above the shuffle null throughout the pre-stimulus period and rises steeply
# after stimulus onset, when sensory evidence becomes available. The null band
# is the 95% range of the pooled pre-stimulus global shuffle nulls (the null is
# time-independent because shuffled labels destroy trial structure at every
# time point).

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.2))
all_null = np.concatenate([res[k]["null_global"] for k in KEYS])
lo, hi = np.percentile(all_null, [2.5, 97.5])
ax.axhspan(lo, hi, color="0.85", zorder=0, label="shuffle null 95%")
ax.axhline(0.5, color="k", lw=0.7, ls=":")
curves = []
for k in KEYS:
    c = res[k]["centers"]
    b = res[k]["tr_decode"][:, 0]
    ax.plot(c, b, color="0.6", lw=1.0, alpha=0.8)
    curves.append(b)
curves = np.array(curves)
ax.plot(c, curves.mean(axis=0), color="#b2182b", lw=2.4,
        label="mean across sessions")
ax.axvline(0, color="k", lw=1.0, ls="--")
ax.axvspan(-0.4, 0, color="#fdae61", alpha=0.35)
ax.text(-0.2, 0.66, "pre-stimulus\nwindow", ha="center", fontsize=8.5,
        color="#a6611a")
ax.set_xlabel("time from stimulus onset (s)  [250 ms window center]")
ax.set_ylabel("balanced accuracy (choice decoding)")
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.set_title("Upcoming choice is decodable before stimulus onset", fontsize=11)
fig.savefig("fig3_timecourse.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Figure 4 — Pre-stimulus choice decoding against two nulls
#
# Left: against the global trial-shuffle null, pre-stimulus decoding is
# significant in 3 of 4 sessions (Stouffer combined p ≈ 1.3e-5). Right: against
# a within-block shuffle null, which preserves block-level choice statistics,
# the effect is strongly attenuated. The within-block null itself sits above
# chance because block-locked slow drift makes even block-preserving label
# permutations partially decodable — evidence that the pre-stimulus signal is
# dominated by a slow bias state rather than fast trial-resolution fluctuations.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
ps_g = [res[k]["p_global"].item() for k in KEYS]
ps_b = [res[k]["p_block"].item() for k in KEYS]

for ax, null_key, ps, title in [
    (axes[0], "null_global", ps_g, "vs global trial-shuffle null"),
    (axes[1], "null_block", ps_b, "vs within-block shuffle null"),
]:
    for i, k in enumerate(KEYS):
        null = res[k][null_key]
        parts = ax.violinplot([null], positions=[i], widths=0.62, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("0.8")
            pc.set_edgecolor("none")
        ax.scatter([i], [np.mean(null)], color="0.45", s=12, zorder=3)
        b = res[k]["bacc"].item()
        ax.scatter([i], [b], color="#b2182b", s=46, zorder=4,
                   marker="D" if ps[i] < 0.05 else "o")
        ax.text(i, b + 0.012, f"p={ps[i]:.3g}", ha="center", fontsize=8)
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels(KEYS)
    ax.set_xlabel("session")
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel("balanced accuracy (pre-stimulus, -0.4 to 0 s)")
z = sum(norm.isf(np.clip(p, 1e-16, 1)) for p in ps_g) / np.sqrt(len(ps_g))
p_comb = norm.sf(z)
fig.suptitle(f"Pre-stimulus choice decoding, 4 sessions "
             f"(diamond = p<0.05; Stouffer combined p = {p_comb:.2g})", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig4_prestim_decoding.png", bbox_inches="tight")
plt.close(fig)
print("Stouffer combined p (global null):", p_comb)

# %% [markdown]
# ## Figure 5 — Single-unit pre-stimulus selectivity
#
# Choice AUCs of all good units in the pre-stimulus window. A family-wise
# threshold (95th percentile of the across-unit max |AUC - 0.5| under shuffles)
# is conservative; at an uncorrected p < 0.05, ~10% of units are selective
# against a null expectation of 5% (null 95% band from leave-one-out shuffle
# replicates). Selective units are distributed across visual, thalamic, and
# midbrain regions rather than concentrated in one area.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

ax = axes[0]
aucs = np.concatenate([res[k]["unit_auc"] for k in KEYS])
thr = np.percentile(np.concatenate([res[k]["null_max"] for k in KEYS]), 95)
ax.hist(aucs, bins=60, color="0.55", edgecolor="none")
for s in [-1, 1]:
    ax.axvline(0.5 + s * thr, color="#b2182b", lw=1.2, ls="--")
frac = np.mean(np.abs(aucs - 0.5) > thr)
frac_p05 = [res[k]["frac_p05"].item() for k in KEYS]
null_frac = np.concatenate([res[k]["null_frac_p05"] for k in KEYS])
nlo, nhi = np.percentile(null_frac, [2.5, 97.5])
ax.set_xlabel("pre-stimulus choice AUC (single units, pooled)")
ax.set_ylabel("number of units")
ax.set_title(
    f"A  {frac*100:.1f}% of units exceed family-wise threshold (|AUC-0.5|>{thr:.3f})\n"
    f"uncorrected p<0.05: {100*np.mean(frac_p05):.1f}% of units "
    f"(null 95%: {100*nlo:.1f}-{100*nhi:.1f}%)", fontsize=9.5)

ax = axes[1]
rows = []
for k in KEYS:
    reg = res[k]["regions"].astype(str)
    up = res[k]["unit_p"]
    for region in np.unique(reg):
        if region == "unknown":
            continue
        m = reg == region
        if m.sum() >= 15:
            rows.append((region, m.sum(), np.mean(up[m] < 0.05)))
d = pd.DataFrame(rows, columns=["region", "n", "frac"])
g = d.groupby("region").apply(lambda r: np.average(r.frac, weights=r.n),
                              include_groups=False).sort_values(ascending=False)
g = g.head(12)
ax.barh(range(len(g)), g.values, color="#4d7fb8")
ax.axvline(0.05, color="k", lw=0.8, ls=":", label="chance (5%)")
ax.set_yticks(range(len(g)))
ax.set_yticklabels([f"{rg}" for rg in g.index], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("fraction of units choice-selective pre-stimulus (p<0.05)")
ax.legend(fontsize=8, frameon=False)
ax.set_title("B  Selective units by brain region (regions with >=15 units)",
             fontsize=10)
fig.tight_layout()
fig.savefig("fig5_single_units.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Figure 6 — Decoding in bias-dominated trial subsets
#
# Left: scoring the decoder only on 0%-contrast trials, where the stimulus
# carries no evidence and choice is purely bias-driven. Pre-stimulus decoding is
# strongest here (session s4: balanced accuracy 0.66, p = 0.01; Stouffer
# combined across sessions p ≈ 0.006). Right: scoring only on unbiased-block
# (P(left) = 0.5) trials. Little signal survives, consistent with the bias
# signal being carried by the block-locked slow state rather than
# trial-resolution fluctuations.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
for ax, bkey, nkey, nully, pkey, title in [
    (axes[0], "bacc0", "n_zero", "null0", "p0",
     "0%-contrast trials only (pure bias)"),
    (axes[1], "baccu", "n_unb", "nullu", "pu",
     "unbiased block (P(left)=0.5) only"),
]:
    for i, k in enumerate(KEYS):
        null = res[k][nully]
        b = res[k][bkey].item()
        p = res[k][pkey].item()
        n = res[k][nkey].item()
        parts = ax.violinplot([null], positions=[i], widths=0.62, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("0.8")
            pc.set_edgecolor("none")
        ax.scatter([i], [b], color="#b2182b", s=46, zorder=4,
                   marker="D" if p < 0.05 else "o")
        ax.text(i, b + 0.012, f"p={p:.3g}\nn={n}", ha="center", fontsize=8)
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels(KEYS)
    ax.set_xlabel("session")
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel("balanced accuracy (pre-stimulus)")
fig.suptitle("Pre-stimulus choice decoding in bias-dominated trial subsets "
             "(diamond = p<0.05)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig6_bias_controls.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Summary
#
# * Upcoming choice is decodable from pre-stimulus population activity in 3 of 4
#   IBL sessions (balanced accuracy 0.52–0.59; Stouffer combined p ≈ 1.3e-5
#   against a global trial-shuffle null), during a movement-free window before
#   any sensory evidence.
# * The signal is a slow, block-locked bias state: it is attenuated by a
#   within-block shuffle control, is strongest on 0%-contrast trials where
#   choice is purely bias-driven (combined p ≈ 0.006), and is weak within
#   unbiased blocks.
# * About 10% of single units are individually choice-selective before stimulus
#   onset (uncorrected p < 0.05, null expectation 5%), distributed across
#   visual, thalamic, and midbrain regions.

# %%
rows = []
for k in KEYS:
    rows.append({
        "session": k,
        "n_trials": res[k]["n_trials"].item(),
        "n_units": res[k]["n_units"].item(),
        "prestim_bacc": round(res[k]["bacc"].item(), 3),
        "prestim_auc": round(res[k]["auc"].item(), 3),
        "p_global": res[k]["p_global"].item(),
        "p_within_block": res[k]["p_block"].item(),
        "frac_units_p05": round(res[k]["frac_p05"].item(), 3),
        "zero_contrast_bacc": round(res[k]["bacc0"].item(), 3),
        "zero_contrast_p": res[k]["p0"].item(),
        "unbiased_bacc": round(res[k]["baccu"].item(), 3),
        "unbiased_p": res[k]["pu"].item(),
    })
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv("summary_stats.csv", index=False)

z0 = sum(norm.isf(np.clip(res[k]["p0"].item(), 1e-16, 1)) for k in KEYS) / np.sqrt(len(KEYS))
print(f"\nStouffer combined p, pre-stimulus choice decoding (all trials): {p_comb:.2g}")
print(f"Stouffer combined p, 0%-contrast trials: {norm.sf(z0):.2g}")
