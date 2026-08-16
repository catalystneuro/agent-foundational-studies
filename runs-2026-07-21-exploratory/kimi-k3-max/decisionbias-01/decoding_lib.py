# Shared library: IBL pre-stimulus choice/bias decoding
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score


def load_session(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    trials = pd.DataFrame({k[len("trial_"):]: d[k] for k in d.files if k.startswith("trial_")})
    st = d["spike_times"]
    sti = d["spike_times_index"]
    ks2 = d["ks2"]
    regions = d["regions"]
    # per-unit spike arrays
    bounds = np.concatenate([[0], sti])
    unit_spikes = [st[bounds[i]:bounds[i + 1]] for i in range(len(sti))]
    return trials, unit_spikes, ks2, regions


def select_units(unit_spikes, ks2, min_rate_hz=0.5, good_only=True):
    rates = np.array([len(s) / (s[-1] - s[0]) if len(s) > 1 else 0.0 for s in unit_spikes])
    mask = rates >= min_rate_hz
    if good_only:
        mask &= ks2 == "good"
    return mask, rates


def count_in_windows(unit_spikes, starts, ends):
    """Spike counts per unit per window. starts/ends: (n_trials,) arrays.
    Returns (n_trials, n_units) count matrix."""
    cols = []
    for sp in unit_spikes:
        cols.append(np.searchsorted(sp, ends, side="right") - np.searchsorted(sp, starts, side="left"))
    return np.stack(cols, axis=1).astype(float)


def make_decoder(C=1.0, max_iter=3000):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=max_iter),
    )


def cv_accuracy(X, y, n_repeats=10, n_splits=5, seed0=0, C=1.0):
    """Mean balanced accuracy over repeated stratified CV."""
    y = np.asarray(y)
    accs = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed0 + r)
        for tr, te in skf.split(X, y):
            clf = make_decoder(C=C)
            clf.fit(X[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs)), float(np.std(accs) / np.sqrt(len(accs)))


def shuffle_null(X, y, n_shuffles=200, n_splits=5, seed0=1000, C=1.0):
    """Null distribution of CV balanced accuracy under label permutation."""
    rng = np.random.default_rng(seed0)
    y = np.asarray(y)
    null = np.empty(n_shuffles)
    for i in range(n_shuffles):
        yp = rng.permutation(y)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed0 + i)
        accs = []
        for tr, te in skf.split(X, yp):
            clf = make_decoder(C=C)
            clf.fit(X[tr], yp[tr])
            accs.append(balanced_accuracy_score(yp[te], clf.predict(X[te])))
        null[i] = np.mean(accs)
    return null


def sliding_window_decoding(unit_spikes, event_times, y, centers, width=0.2,
                            n_repeats=5, n_shuffles=50, C=1.0, seed0=0):
    """Decode y from spike counts in windows centered at event_times + centers."""
    acc = np.full(len(centers), np.nan)
    acc_sem = np.full(len(centers), np.nan)
    null_mean = np.full(len(centers), np.nan)
    null_hi = np.full(len(centers), np.nan)
    for i, c in enumerate(centers):
        X = count_in_windows(unit_spikes, event_times + c - width / 2,
                             event_times + c + width / 2)
        a, s = cv_accuracy(X, y, n_repeats=n_repeats, C=C, seed0=seed0 + i)
        null = shuffle_null(X, y, n_shuffles=n_shuffles, C=C, seed0=seed0 + 5000 + i)
        acc[i], acc_sem[i] = a, s
        null_mean[i] = null.mean()
        null_hi[i] = np.percentile(null, 95)
    return acc, acc_sem, null_mean, null_hi


def unit_auroc(counts, y):
    """Per-unit auROC of spike count vs binary y (Mann-Whitney formulation),
    with correct average-rank handling of ties."""
    y = np.asarray(y)
    vals = np.asarray(counts, dtype=float)
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(len(vals))
    ranks[order] = np.arange(1, len(vals) + 1)
    sv = vals[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int(np.sum(y == 1))
    n_neg = len(y) - n_pos
    r_pos = ranks[y == 1].sum()
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
