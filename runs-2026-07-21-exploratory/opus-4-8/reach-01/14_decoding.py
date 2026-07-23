"""Population decoding: continuous hand velocity, and reach direction from delay activity."""
import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

def lagged_population(counts, trial, lags_bins):
    """Stack population counts at several lags; NaN where the lag leaves the trial."""
    n, u = counts.shape
    out = np.full((n, u * len(lags_bins)), np.nan)
    idx = np.arange(n)
    for j, L in enumerate(lags_bins):
        src = idx - L
        ok = (src >= 0) & (src < n)
        ok[ok] &= trial[src[ok]] == trial[idx[ok]]
        out[np.ix_(ok, np.arange(u) + j * u)] = counts[src[ok]]
    return out

def decode_velocity(counts, trial, vel, lags_bins, n_folds=5):
    X = lagged_population(counts, trial, lags_bins)
    good = ~np.isnan(X).any(1)
    X, Y, g = X[good], vel[good], trial[good]
    pred = np.full_like(Y, np.nan)
    gkf = GroupKFold(n_splits=n_folds)
    for tr, te in gkf.split(X, Y, groups=g):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(0, 5, 12)).fit(sc.transform(X[tr]), Y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    r2 = 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return pred, Y, g, r2, good

def decode_direction(rates, labels, groups, n_folds=5):
    """Multinomial logistic decoding of the reach-direction bin from per-trial rates."""
    pred = np.empty(len(labels), dtype=int)
    for tr, te in GroupKFold(n_splits=n_folds).split(rates, labels, groups=groups):
        sc = StandardScaler().fit(rates[tr])
        m = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(rates[tr]), labels[tr])
        pred[te] = m.predict(sc.transform(rates[te]))
    return pred
