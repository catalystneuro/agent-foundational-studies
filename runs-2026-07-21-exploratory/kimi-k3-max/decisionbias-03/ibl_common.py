"""Shared helpers for IBL pre-stimulus decision-bias decoding (DANDI 000149)."""
import os
import pickle
import warnings

import numpy as np

# lbfgs can emit transient overflow RuntimeWarnings on shuffled-label fits
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
import lindi
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

ASSET_IDS = {
    "s1": "31f22c47-1512-4293-b19f-6fa5bd9b7cbf",
    "s2": "81169999-c697-4eca-a635-2fd994ac183f",
    "s3": "e7fa5ae0-b957-4b24-aa40-fb4c3276d331",
    "s4": "f791a116-1e6c-4d6a-a9eb-fe3644737be2",
}

LINDI_TEMPLATE = "https://lindi.neurosift.org/dandi/dandisets/000149/assets/{}/nwb.lindi.json"


def load_session(asset_id):
    """Load trials table, per-unit spike times, and unit metadata for one session.

    Streams via LINDI; reads spikes from the flat HDF5 arrays (fast) rather than
    the ragged units column (slow/broken over LINDI). Caches to a local pickle.
    Returns (trials DataFrame, list of spike-time arrays, units DataFrame).
    """
    cache_path = os.path.join(CACHE_DIR, f"{asset_id}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    f = lindi.LindiH5pyFile.from_lindi_file(
        LINDI_TEMPLATE.format(asset_id), local_cache=lindi.LocalCache()
    )

    # --- trials table (strip .npy suffixes) ---
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

    # --- units: flat spike arrays + metadata ---
    spike_times = f["units/spike_times"][:]
    spike_index = f["units/spike_times_index"][:]
    starts = np.concatenate([[0], spike_index[:-1]])
    spike_list = [spike_times[s:e] for s, e in zip(starts, spike_index)]

    # brain-region annotations are missing in some sessions (e.g. asset s2)
    if "brainLocationAcronyms_ccf_2017.npy" in f["units"]:
        regions = f["units/brainLocationAcronyms_ccf_2017.npy"][:].astype(str)
    else:
        regions = np.array(["unknown"] * len(spike_list))
    units = pd.DataFrame(
        {
            "region": regions,
            "ks2_label": f["units/ks2_label"][:].astype(str),
            "firing_rate": f["units/firing_rate"][:],
            "presence_ratio": f["units/presence_ratio"][:],
        }
    )

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


def decode_cv(X, y, n_repeats=5, seed=0):
    """Repeated stratified 5-fold CV with a scaled logistic regression.

    Returns (balanced accuracy, ROC AUC) averaged over repeats.
    """
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
            pred = clf.predict(X[te])
            baccs.append(balanced_accuracy_score(y[te], pred))
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(baccs)), float(np.mean(aucs))


def shuffle_null(X, y, n_shuffles=200, seed=0, blocks=None):
    """Null distribution of balanced accuracy under label shuffles.

    If blocks is given, labels are shuffled within each block run (controls for
    slow drift locked to block structure); otherwise shuffled globally.
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


def time_resolved_decoding(spike_list, event_times, y, windows, n_repeats=3, seed=0):
    """Decode y from spike counts in sliding windows around event_times.

    windows: array of (start, end) offsets relative to the events.
    Returns (n_windows, 2) array of (balanced acc, AUC).
    """
    out = np.empty((len(windows), 2))
    for i, (w0, w1) in enumerate(windows):
        X = counts_in_windows(spike_list, event_times + w0, event_times + w1)
        out[i] = decode_cv(X, y, n_repeats=n_repeats, seed=seed + i)
    return out


def single_unit_auc(X, y):
    """Rank-based AUC (choice probability) per unit. Positive class = y==1."""
    y = np.asarray(y)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    aucs = np.empty(X.shape[1])
    for u in range(X.shape[1]):
        r = rankdata(X[:, u])
        aucs[u] = (r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return aucs


def decode_cv_subset(X, y, subset, n_repeats=5, seed=0):
    """CV over all trials; balanced accuracy scored only on `subset` test trials.

    Training always uses all trials in the train fold, which gives much better
    power for small trial subsets (e.g. 0%-contrast trials) than training on
    the subset alone.
    """
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
        yp = rng.permutation(y)
        null[i] = decode_cv_subset(X, yp, subset, n_repeats=1, seed=seed + 1000 + i)
    return null


def single_unit_auc_null(X, y, n_shuffles=200, seed=0):
    """Null AUC values per unit under global label shuffle.

    Returns (null_max, null_aucs): the family-wise max |AUC-0.5| per shuffle,
    and the full (n_shuffles, n_units) null AUC matrix for per-unit p-values.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n_pos = (y == 1).sum()
    n = len(y)
    ranks = np.apply_along_axis(rankdata, 0, X)  # (n_trials, n_units), fixed under shuffle
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
