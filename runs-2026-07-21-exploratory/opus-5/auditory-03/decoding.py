"""Decode which tone was played from the single-trial population response."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import EVOKED_WIN, FREQS, window_counts


def population_features(spikes, trials):
    """(n_trials, n_units) spike counts in the evoked window."""
    return window_counts(spikes, trials.start_time.values, EVOKED_WIN)


def decode_frequency(X, y, n_splits=5, seed=0, shuffle_label=False):
    """Cross-validated multinomial logistic decoding of tone frequency.

    Returns (accuracy, confusion) with confusion normalised per true class.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if shuffle_label:
        y = rng.permutation(y)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=0.05))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.empty_like(y)
    for tr, te in cv.split(X, y):
        pred[te] = clf.fit(X[tr], y[tr]).predict(X[te])
    conf = np.zeros((len(FREQS), len(FREQS)))
    for i, f in enumerate(FREQS):
        for j, g in enumerate(FREQS):
            conf[i, j] = np.sum((y == f) & (pred == g))
    conf = conf / conf.sum(axis=1, keepdims=True)
    return float(np.mean(pred == y)), conf


def accuracy_vs_units(X, y, sizes, n_rep=5, seed=0):
    """Decoding accuracy as a function of the number of randomly drawn units."""
    rng = np.random.default_rng(seed)
    out = np.full((len(sizes), n_rep), np.nan)
    for i, n in enumerate(sizes):
        if n > X.shape[1]:
            continue
        for r in range(n_rep):
            cols = rng.choice(X.shape[1], n, replace=False)
            out[i, r] = decode_frequency(X[:, cols], y, n_splits=3,
                                         seed=100 * r + i)[0]
    return out
