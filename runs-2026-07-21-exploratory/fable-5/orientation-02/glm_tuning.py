"""Poisson GLM comparison of orientation- vs direction-based encoding (NeMoS).

Two encoding models are fit to per-trial spike counts:

  direction model    smooth function of drift direction, periodic over 360 deg
  orientation model  smooth function of the grating axis, periodic over 180 deg

If a unit is orientation selective but not direction selective, the 180 deg
model predicts held-out trials as well as the 360 deg model despite using half
the parameters. The comparison is done with cross-validated Poisson pseudo-R^2.
"""

import jax
import numpy as np
import nemos as nmo

jax.config.update("jax_enable_x64", True)
from sklearn.model_selection import KFold
from tqdm import tqdm

DIR_BASIS = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, bounds=(0, 360))
ORI_BASIS = nmo.basis.CyclicBSplineEval(n_basis_funcs=4, bounds=(0, 180))


def _poisson_ll(y, rate):
    from scipy.special import gammaln

    rate = np.maximum(rate, 1e-9)
    return np.sum(y * np.log(rate) - rate - gammaln(y + 1))


def _ll_per_unit(Y, rate):
    from scipy.special import gammaln

    rate = np.maximum(rate, 1e-9)
    return np.sum(Y * np.log(rate) - rate - gammaln(Y + 1), axis=0)


def cv_pseudo_r2_population(X, Y, n_splits=5, seed=0, desc="GLM"):
    """Cross-validated Poisson pseudo-R^2 per unit, fitting all units jointly.

    `PopulationGLM` without a feature mask is equivalent to independent
    per-neuron GLMs, but runs as a single JAX-compiled optimisation.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_units = Y.shape[1]
    ll_model = np.zeros(n_units)
    ll_null = np.zeros(n_units)
    ll_sat = np.zeros(n_units)
    for tr, te in tqdm(list(kf.split(X)), desc=desc):
        glm = nmo.glm.PopulationGLM(solver_name="LBFGS")
        glm.fit(X[tr], Y[tr])
        ll_model += _ll_per_unit(Y[te], np.asarray(glm.predict(X[te])))
        null = np.broadcast_to(np.maximum(Y[tr].mean(0), 1e-9), (len(te), n_units))
        ll_null += _ll_per_unit(Y[te], null)
        ll_sat += _ll_per_unit(Y[te], np.maximum(Y[te], 1e-9))
    denom = ll_sat - ll_null
    return np.where(denom > 0, (ll_model - ll_null) / np.where(denom > 0, denom, 1), np.nan)


def fit_encoding_models(counts, directions, n_splits=5):
    """Per-unit cross-validated pseudo-R^2 for the 360 deg and 180 deg models."""
    x_dir = np.asarray(directions, dtype=float)[:, None]
    X_dir = np.asarray(DIR_BASIS.compute_features(x_dir))
    X_ori = np.asarray(ORI_BASIS.compute_features(x_dir % 180.0))
    Y = counts.T.astype(float)
    return {
        "dir": cv_pseudo_r2_population(X_dir, Y, n_splits, desc="GLM direction (360 deg)"),
        "ori": cv_pseudo_r2_population(X_ori, Y, n_splits, desc="GLM orientation (180 deg)"),
    }


def fitted_curves(counts_unit, directions, grid=np.arange(0, 360, 2.0)):
    """Full-data fits of both encoding models, evaluated on a fine grid.

    Returns (grid, direction-model counts/trial, orientation-model counts/trial).
    """
    x = np.asarray(directions, float)[:, None]
    y = counts_unit.astype(float)
    out = []
    for basis, xx, gg in (
        (DIR_BASIS, x, grid[:, None]),
        (ORI_BASIS, x % 180.0, grid[:, None] % 180.0),
    ):
        glm = nmo.glm.GLM(solver_name="LBFGS")
        glm.fit(np.asarray(basis.compute_features(xx)), y)
        out.append(np.asarray(glm.predict(np.asarray(basis.compute_features(gg)))))
    return grid, out[0], out[1]
