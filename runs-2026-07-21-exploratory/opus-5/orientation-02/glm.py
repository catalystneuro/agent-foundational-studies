"""
Poisson GLM encoding models (NeMoS) for single-trial spike counts during
drifting gratings.

Four nested models are compared by 5-fold cross-validated Poisson
log-likelihood, fit for every unit of a session simultaneously with
`nmo.glm.PopulationGLM` (the design matrix is shared across units):

  M0  intercept only
  M1  running speed only            (B-spline basis)
  M2  drift direction only          (cyclic B-spline basis over 0-360 deg)
  M3  drift direction + running speed

M3 - M1 isolates the contribution of grating direction *over and above* the
animal's locomotor state, which strongly co-varies with cortical firing rate in
this dataset.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import nemos as nmo
from sklearn.model_selection import KFold

DIRECTIONS = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
SOLVER = dict(solver_name="LBFGS", regularizer="Ridge", regularizer_strength=1e-3,
              solver_kwargs=dict(maxiter=2000, tol=1e-8))


def build_design(directions_deg, speed, n_dir_basis=6, n_speed_basis=4):
    """Feature matrices for the direction and running-speed terms."""
    dir_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=n_dir_basis, bounds=(0.0, 360.0))
    X_dir = np.asarray(dir_basis.compute_features(np.asarray(directions_deg, dtype=float)))

    s = np.asarray(speed, dtype=float)
    s = np.clip(s, np.nanpercentile(s, 1), np.nanpercentile(s, 99))
    spd_basis = nmo.basis.BSplineEval(n_basis_funcs=n_speed_basis,
                                      bounds=(float(s.min()), float(s.max())))
    X_spd = np.asarray(spd_basis.compute_features(s))
    return X_dir, X_spd, dir_basis


def _poisson_ll(y, rate):
    """Mean Poisson log-likelihood per trial (dropping the constant log y! term)."""
    rate = np.clip(np.asarray(rate), 1e-8, None)
    return np.mean(y * np.log(rate) - rate, axis=0)


def _cv_loglik_population(X, Y, n_splits=5, seed=0):
    """Held-out per-unit Poisson log-likelihood. X=None gives the intercept-only model."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    lls = []
    for tr, te in kf.split(Y):
        if X is None:
            mu = np.maximum(Y[tr].mean(0), 1e-8)
            lls.append(_poisson_ll(Y[te], np.broadcast_to(mu, Y[te].shape)))
        else:
            model = nmo.glm.PopulationGLM(**SOLVER)
            model.fit(X[tr], Y[tr])
            lls.append(_poisson_ll(Y[te], np.asarray(model.predict(X[te]))))
    return np.mean(np.stack(lls), axis=0)


def fit_session(res, speed, seed=0):
    """Cross-validated log-likelihoods of the four nested models, all units at once."""
    Y = np.rint(res["dg_rates"] * 2.0)  # rates -> counts over the 2 s presentation
    X_dir, X_spd, dir_basis = build_design(res["dg_labels"], speed)

    ll_null = _cv_loglik_population(None, Y, seed=seed)
    ll_spd = _cv_loglik_population(X_spd, Y, seed=seed)
    ll_dir = _cv_loglik_population(X_dir, Y, seed=seed)
    ll_full = _cv_loglik_population(np.hstack([X_dir, X_spd]), Y, seed=seed)

    df = pd.DataFrame(
        dict(
            unit_id=res["units"].index.values,
            area=res["units"]["area"].values,
            session_id=res["session_id"],
            mean_count=Y.mean(0),
            ll_null=ll_null,
            ll_speed=ll_spd,
            ll_dir=ll_dir,
            ll_full=ll_full,
            d_dir=ll_dir - ll_null,
            d_speed=ll_spd - ll_null,
            d_dir_given_speed=ll_full - ll_spd,
        )
    )
    return df, dir_basis


def predicted_tuning(Y, X_dir, dir_basis, n_grid=181):
    """Fit the direction-only population GLM on all trials; return smooth tuning curves."""
    model = nmo.glm.PopulationGLM(**SOLVER)
    model.fit(X_dir, np.rint(np.asarray(Y, dtype=float)))
    grid = np.linspace(0, 360, n_grid)
    Xg = np.asarray(dir_basis.compute_features(grid))
    rate = np.asarray(model.predict(Xg)) / 2.0  # counts per 2 s -> Hz
    return grid, rate
