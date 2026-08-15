"""Cross-validated linear decoding with autocorrelation-aware null models.

Two things make decoding of the IBL prior block statistically delicate, and both
are handled here:

1. The target is strongly autocorrelated in trial order (blocks last 20-100
   trials).  Random K-fold splits let a decoder exploit slow drift in the neural
   data to "predict" the block, so all cross-validation here uses *contiguous*
   folds in trial order.
2. Even with contiguous folds, slow non-stationarity inflates chance
   performance.  Significance is therefore assessed against a null that
   preserves the temporal structure of the target: either pseudo-sessions drawn
   from the IBL block-generating process, or circular shifts of the labels.
"""

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

# Rare units with almost no spikes produce large z-scores and overflow warnings
# inside the solver; the fits themselves converge fine.
np.seterr(over="ignore", invalid="ignore", divide="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.metrics import roc_auc_score, log_loss

# Fixed regularisation, applied identically to real and null data, which keeps
# the permutation test valid without the cost of nested hyper-parameter search.
C_DEFAULT = 0.1


def contiguous_folds(n, k=5):
    """Indices of k contiguous folds in trial order."""
    return np.array_split(np.arange(n), k)


def oof_predict(X, y, k=5, C=C_DEFAULT):
    """Out-of-fold P(y=1) from L2 logistic regression on z-scored features."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    p = np.full(len(y), np.nan)
    for fold in contiguous_folds(len(y), k):
        train = np.setdiff1d(np.arange(len(y)), fold)
        if len(np.unique(y[train])) < 2:
            continue
        mu, sd = X[train].mean(0), X[train].std(0)
        # Near-constant units would otherwise produce enormous z-scores.
        sd[sd < 1e-3] = 1.0
        clf = LogisticRegression(
            penalty="l2", C=C, max_iter=2000, class_weight="balanced", solver="lbfgs"
        )
        clf.fit((X[train] - mu) / sd, y[train])
        p[fold] = clf.predict_proba((X[fold] - mu) / sd)[:, 1]
    return p


def cv_auc(X, y, k=5, C=C_DEFAULT):
    """Mean within-fold AUC.

    Pooling out-of-fold probabilities across folds and scoring them together
    would mix decision values from models fit on different data, which adds
    variance and pushes the null off 0.5.  Scoring each fold separately and
    averaging avoids that.
    """
    y = np.asarray(y, dtype=int)
    p = oof_predict(X, y, k=k, C=C)
    aucs, weights = [], []
    for fold in contiguous_folds(len(y), k):
        yf, pf = y[fold], p[fold]
        ok = ~np.isnan(pf)
        if len(np.unique(yf[ok])) < 2:
            continue
        aucs.append(roc_auc_score(yf[ok], pf[ok]))
        weights.append(ok.sum())
    if not aucs:
        return np.nan
    return float(np.average(aucs, weights=weights))


def cv_logloss(X, y, k=5, C=C_DEFAULT):
    p = oof_predict(X, y, k=k, C=C)
    ok = ~np.isnan(p)
    return log_loss(y[ok], np.clip(p[ok], 1e-6, 1 - 1e-6), labels=[0, 1])


# --------------------------------------------------------------------------
# Null models
# --------------------------------------------------------------------------

def circular_shifts(n, n_shift, min_shift=60, rng=None):
    """Random circular shift offsets that are at least ``min_shift`` trials.

    The floor is capped at a fifth of the series so that short trial subsets
    (for example, the zero-contrast trials of one session) still yield a usable
    number of distinct surrogates.
    """
    rng = np.random.default_rng(rng)
    min_shift = min(min_shift, max(5, n // 5))
    lo, hi = min_shift, n - min_shift
    pool = np.arange(lo, hi)
    return rng.choice(pool, size=min(n_shift, len(pool)), replace=len(pool) < n_shift)


def pseudo_blocks(n, rng=None, mean_len=60, lo=20, hi=100):
    """A surrogate block sequence from the IBL block-generating process.

    Block lengths are drawn from an exponential with mean 60 truncated to
    [20, 100] trials and sides alternate deterministically, which is how the
    IBL task engine schedules the 0.8/0.2 prior blocks.
    """
    rng = np.random.default_rng(rng)
    side = rng.integers(0, 2)
    out = []
    while len(out) < n:
        L = int(np.clip(rng.exponential(mean_len), lo, hi))
        out += [side] * L
        side = 1 - side
    return np.array(out[:n])


def null_aucs(X, y, n_null=200, kind="pseudo", k=5, C=C_DEFAULT, rng=0):
    """Null distribution of cross-validated AUC under a structure-preserving null."""
    rng = np.random.default_rng(rng)
    n = len(y)
    out = []
    if kind == "pseudo":
        for _ in range(n_null):
            yn = pseudo_blocks(n, rng=rng)
            if len(np.unique(yn)) < 2:
                continue
            out.append(cv_auc(X, yn, k=k, C=C))
    else:
        for s in circular_shifts(n, n_null, rng=rng):
            out.append(cv_auc(X, np.roll(y, s), k=k, C=C))
    return np.array([a for a in out if np.isfinite(a)])


def perm_p(real, null):
    """One-sided permutation p-value with the standard +1 correction."""
    null = np.asarray(null)
    return (1.0 + np.sum(null >= real)) / (1.0 + len(null))


def null_z(real, null):
    null = np.asarray(null)
    sd = null.std()
    return (real - null.mean()) / sd if sd > 0 else np.nan


def stouffer(zs):
    """Combine independent z-scores."""
    zs = np.asarray([z for z in zs if np.isfinite(z)])
    return zs.sum() / np.sqrt(len(zs)) if len(zs) else np.nan
