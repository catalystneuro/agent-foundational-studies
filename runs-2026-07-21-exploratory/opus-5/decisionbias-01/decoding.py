"""Cross-validated decoders and their null distributions.

Design choices fixed a priori and applied identically to real and surrogate data:
  * features   : spike counts in a time window, z-scored and projected onto the first
                 `N_COMP` PCs.  Both the z-scoring statistics and the PCA rotation are
                 estimated on the training fold only.
  * classifier : L2-regularised logistic regression, C = 0.05.  Fixed; never tuned
                 against the outcome, and identical for real and surrogate labels.
  * folds      : 5 *contiguous* folds over the trial sequence.  Random k-fold would let a
                 slowly drifting firing rate leak the (slowly drifting) block identity
                 from train to test; contiguous folds do not.
  * score      : AUC computed once over the pooled out-of-fold decision values.

Because the fold geometry and the feature matrix do not depend on the labels, the
expensive part (z-scoring + PCA per fold) is computed once and reused for the hundreds of
surrogate label vectors.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

C_FIXED = 0.05
N_FOLDS = 5
N_COMP = 50


def contiguous_folds(n, k=N_FOLDS):
    edges = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k)]


def _clean(X):
    return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def prep_folds(X, folds, n_comp=N_COMP, covariates=None):
    """Pre-compute the label-independent part of each fold: z-score, PCA, optional
    residualisation of the PCs on nuisance covariates (fit on training trials only)."""
    X = _clean(X)
    cov = _clean(covariates) if covariates is not None else None
    prepped = []
    for te in folds:
        tr = np.setdiff1d(np.arange(len(X)), te)
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        k = int(min(n_comp, len(tr) - 1, X.shape[1]))
        # PCA via SVD of the centred training block (Ztr is already ~centred)
        c = Ztr.mean(0)
        U, S, Vt = np.linalg.svd(Ztr - c, full_matrices=False)
        V = Vt[:k].T
        Ptr, Pte = (Ztr - c) @ V, (Zte - c) @ V
        if cov is not None:
            A = np.column_stack([np.ones(len(tr)), cov[tr]])
            B = np.column_stack([np.ones(len(te)), cov[te]])
            beta, *_ = np.linalg.lstsq(A, Ptr, rcond=None)
            Ptr, Pte = Ptr - A @ beta, Pte - B @ beta
        prepped.append((tr, te, Ptr, Pte))
    return prepped


def _zscore(v):
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v - v.mean()


def cv_decode_prepped(prepped, y, n, C=C_FIXED):
    oof = np.full(n, np.nan)
    for tr, te, Ptr, Pte in prepped:
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(C=C, penalty="l2", max_iter=1000, solver="lbfgs")
        clf.fit(Ptr, y[tr])
        # z-scored within fold so that decision values from different folds are on a
        # comparable scale before they are pooled into a single AUC
        oof[te] = _zscore(clf.decision_function(Pte))
    ok = ~np.isnan(oof)
    auc = roc_auc_score(y[ok], oof[ok]) if len(np.unique(y[ok])) == 2 else np.nan
    return auc, oof


def cv_decode(X, y, folds=None, n_comp=N_COMP, covariates=None, return_scores=False):
    folds = folds if folds is not None else contiguous_folds(len(y))
    prepped = prep_folds(X, folds, n_comp=n_comp, covariates=covariates)
    auc, oof = cv_decode_prepped(prepped, y, len(y))
    return (auc, oof) if return_scores else auc


def train_test_decode(Xtr, ytr, Xte, n_comp=N_COMP, C=C_FIXED, cov_tr=None, cov_te=None):
    """Train on one trial set, score a disjoint one.

    If nuisance covariates are supplied they are regressed out of every feature, with the
    regression weights estimated on the training trials only.
    """
    Xtr, Xte = _clean(Xtr), _clean(Xte)
    if cov_tr is not None:
        A = np.column_stack([np.ones(len(Xtr)), _clean(cov_tr)])
        B = np.column_stack([np.ones(len(Xte)), _clean(cov_te)])
        beta, *_ = np.linalg.lstsq(A, Xtr, rcond=None)
        Xtr, Xte = Xtr - A @ beta, Xte - B @ beta
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    c = Ztr.mean(0)
    k = int(min(n_comp, len(Xtr) - 1, Xtr.shape[1]))
    _, _, Vt = np.linalg.svd(Ztr - c, full_matrices=False)
    V = Vt[:k].T
    clf = LogisticRegression(C=C, penalty="l2", max_iter=1000, solver="lbfgs")
    clf.fit((Ztr - c) @ V, ytr)
    return clf.decision_function((Zte - c) @ V)


def null_p_value(observed, null_samples):
    """One-sided permutation p with the standard +1 correction."""
    null_samples = np.asarray(null_samples, dtype=float)
    null_samples = null_samples[~np.isnan(null_samples)]
    if len(null_samples) == 0 or np.isnan(observed):
        return np.nan
    return (1 + np.sum(null_samples >= observed)) / (1 + len(null_samples))


def crossdecode_subset(X, y_train_label, test_mask, folds, covariates=None, n_comp=N_COMP):
    """Train a decoder on `y_train_label` using only trials *outside* `test_mask`, and only
    from training folds, then score the `test_mask` trials of the held-out fold.

    Two forms of leakage are avoided: the model never sees a test trial's label (the test
    trials are excluded from training entirely) and, because the folds are contiguous in
    time, it also never sees a trial from the same stretch of the session.  Without the
    second constraint a decoder can ride slow drift in firing rate and appear to predict a
    slowly varying behavioural variable it has learned nothing about.
    """
    X = _clean(X)
    n = len(X)
    test_idx = np.where(test_mask)[0]
    out = np.full(len(test_idx), np.nan)
    for te in folds:
        in_fold = np.zeros(n, dtype=bool)
        in_fold[te] = True
        tr_sel = (~test_mask) & (~in_fold)
        te_sel = test_mask & in_fold
        if te_sel.sum() < 3 or len(np.unique(y_train_label[tr_sel])) < 2:
            continue
        kw = {}
        if covariates is not None:
            cov = _clean(covariates)
            kw = dict(cov_tr=cov[tr_sel], cov_te=cov[te_sel])
        s = train_test_decode(X[tr_sel], y_train_label[tr_sel], X[te_sel],
                              n_comp=n_comp, **kw)
        out[np.searchsorted(test_idx, np.where(te_sel)[0])] = _zscore(s)
    return out
