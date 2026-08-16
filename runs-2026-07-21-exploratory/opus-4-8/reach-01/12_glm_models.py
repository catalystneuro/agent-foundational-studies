"""Poisson GLM encoding models (NeMoS): what about the reach does M1/PMd encode?"""
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np, nemos as nmo
from scipy.special import gammaln

def deviance_explained(y, rate, rate_null):
    """Per-unit fraction of Poisson deviance explained relative to a constant-rate model."""
    def ll(r):
        r = np.clip(r, 1e-9, None)
        return (y * np.log(r) - r - gammaln(y + 1)).sum(0)
    ll_sat = ll(np.clip(y, 1e-9, None))
    return 1 - (ll_sat - ll(rate)) / (ll_sat - ll(rate_null))

def make_bases():
    ang = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, label='direction')
    spd = nmo.basis.MSplineEval(n_basis_funcs=5, label='speed')
    px = nmo.basis.BSplineEval(n_basis_funcs=5, label='pos_x')
    py = nmo.basis.BSplineEval(n_basis_funcs=5, label='pos_y')
    return {
        'speed only':          (spd, ('speed',)),
        'direction only':      (ang, ('angle',)),
        'direction x speed':   (ang * spd, ('angle', 'speed')),
        'position':            (px * py, ('x', 'y')),
        'dir x speed + position': (ang * spd + px * py, ('angle', 'speed', 'x', 'y')),
    }

def design(basis, names, feats):
    """M-spline/B-spline Eval bases are density-normalised, so their output scale is ~1/range.
    Rescaling every feature to [0, 1] first keeps all blocks on a comparable scale, which
    matters once a Ridge penalty is applied."""
    def scale(v):
        lo, hi = np.min(v), np.max(v)
        return v if hi - lo == 0 else (v - lo) / (hi - lo)
    vals = [feats[n] if n == 'angle' else scale(feats[n]) for n in names]
    return np.asarray(basis.compute_features(*vals))

def fit_cv(X, Y, groups, n_folds=5, reg=1e-3):
    """Group (trial-wise) cross-validated PopulationGLM; returns held-out deviance explained."""
    ug = np.unique(groups)
    fold_of = {g: i % n_folds for i, g in enumerate(ug)}
    fold = np.array([fold_of[g] for g in groups])
    de = np.full((n_folds, Y.shape[1]), np.nan)
    from tqdm import tqdm
    for k in tqdm(range(n_folds), desc='CV folds', leave=False):
        tr, te = fold != k, fold == k
        m = nmo.glm.PopulationGLM(regularizer='Ridge', regularizer_strength=reg,
                                  solver_name='LBFGS', solver_kwargs={'tol': 1e-7, 'maxiter': 300})
        m.fit(X[tr], Y[tr])
        pred = np.asarray(m.predict(X[te]))
        de[k] = deviance_explained(Y[te], pred, np.tile(Y[tr].mean(0), (te.sum(), 1)))
    return de.mean(0)
