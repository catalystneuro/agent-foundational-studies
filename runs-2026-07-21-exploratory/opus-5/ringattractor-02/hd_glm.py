"""GLM test: does the internally generated ring coordinate predict spiking during sleep?

A Poisson GLM with a cyclic B-spline basis over an angular covariate is fitted with
NeMoS and cross-validated.  During waking the covariate can be the measured head
direction; during sleep the only covariate available is the angle the population
itself defines on its manifold.
"""
import numpy as np
from scipy.special import gammaln
import nemos as nmo

TWOPI = 2 * np.pi


def _poisson_ll(y, lam):
    lam = np.clip(lam, 1e-9, None)
    return (y * np.log(lam) - lam - gammaln(y + 1)).sum(0)


def cv_pseudo_r2(angle, counts, n_basis=8, n_folds=5, reg=1e-4, seed=0):
    """Cross-validated McFadden pseudo-R^2 per cell for an angular GLM."""
    basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=n_basis, label="angle")
    X = np.asarray(basis.compute_features(np.mod(angle, TWOPI)))
    y = np.asarray(counts, dtype=float)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % n_folds
    num = np.zeros(y.shape[1])
    den = np.zeros(y.shape[1])
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        model = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=reg,
                                      solver_name="LBFGS",
                                      solver_kwargs={"maxiter": 2000, "tol": 1e-8}).fit(X[tr], y[tr])
        lam = np.asarray(model.predict(X[te]))
        num += _poisson_ll(y[te], lam)
        den += _poisson_ll(y[te], np.tile(y[tr].mean(0), (te.sum(), 1)))
    return 1 - num / den


def glm_tuning(angle, counts, n_basis=8, reg=1e-4, n_grid=120, bin_size=0.2):
    """Fit the angular GLM on all data and return predicted tuning curves (Hz)."""
    basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=n_basis, label="angle")
    X = np.asarray(basis.compute_features(np.mod(angle, TWOPI)))
    model = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=reg,
                                  solver_name="LBFGS",
                                  solver_kwargs={"maxiter": 2000, "tol": 1e-8}
                                  ).fit(X, np.asarray(counts, dtype=float))
    grid = np.linspace(0, TWOPI, n_grid, endpoint=False)
    Xg = np.asarray(basis.compute_features(grid))
    return grid, np.asarray(model.predict(Xg)) / bin_size


def shift_null(angle, rng):
    """Circularly shift the covariate relative to the spike counts."""
    k = rng.integers(len(angle) // 4, 3 * len(angle) // 4)
    return np.roll(angle, k)
