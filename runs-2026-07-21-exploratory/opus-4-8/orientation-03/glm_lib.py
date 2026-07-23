"""NeMoS Poisson-GLM model comparison: untuned vs orientation vs direction.

For each unit we model the spike count on trial *i* as Poisson with rate

    log lambda_i = b + w . f(theta_i)

where ``f`` is a cyclic B-spline basis over the grating angle.  Three nested
models are compared by 5-fold cross-validated Poisson log-likelihood:

* ``constant``    - intercept only, no stimulus dependence
* ``orientation`` - the basis is 180 deg periodic, so a grating and its 180 deg
  counterpart are forced to predict the same rate
* ``direction``   - the basis is 360 deg periodic and can separate the two

Cross-validation supplies the complexity penalty, so "direction beats
orientation" means the extra flexibility actually pays for itself on held-out
trials.  A unit is called orientation selective when ``orientation`` beats
``constant``, and direction selective when ``direction`` also beats
``orientation``.

All units are fit together with :class:`nemos.glm.PopulationGLM` (one set of
coefficients per neuron, shared design matrix), which is orders of magnitude
faster than looping a single-neuron GLM over hundreds of units.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import nemos as nmo
from scipy.special import gammaln
from sklearn.model_selection import KFold

N_ORI_BASIS = 5
N_DIR_BASIS = 8
# A small ridge penalty is needed for a finite optimum: units that are silent at
# some directions otherwise drive their coefficients towards -inf and LBFGS never
# converges.  1e-2 is the smallest value at which every session converges.
RIDGE = 1e-2


def _basis(model_name):
    if model_name == "orientation":
        return nmo.basis.CyclicBSplineEval(n_basis_funcs=N_ORI_BASIS, bounds=(0.0, 180.0)), 180.0
    return nmo.basis.CyclicBSplineEval(n_basis_funcs=N_DIR_BASIS, bounds=(0.0, 360.0)), 360.0


def design_matrix(angles_deg, model_name):
    basis, period = _basis(model_name)
    return np.asarray(basis.compute_features(np.asarray(angles_deg, dtype=float) % period))


def poisson_loglik(y, rate):
    """Per-observation Poisson log-likelihood, elementwise."""
    rate = np.clip(rate, 1e-10, None)
    return y * np.log(rate) - rate - gammaln(y + 1.0)


def _fit_population(X, y, reg_strength):
    model = nmo.glm.PopulationGLM(
        observation_model="Poisson",
        regularizer="Ridge",
        regularizer_strength=reg_strength,
        solver_name="LBFGS",
        solver_kwargs=dict(maxiter=2000, tol=1e-8),
    )
    model.fit(X, y)
    return model


def cross_validated_loglik(counts, angles_deg, n_splits=5, seed=0, reg_strength=RIDGE):
    """Held-out Poisson log-likelihood per trial, for each model and each unit.

    Parameters
    ----------
    counts : (n_trials, n_units) integer spike counts
    angles_deg : (n_trials,) grating direction in degrees

    Returns
    -------
    dict mapping model name -> (n_units,) mean held-out log-likelihood in nats/trial
    """
    counts = np.asarray(counts, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    designs = {m: design_matrix(angles_deg, m) for m in ("orientation", "direction")}
    out = {m: np.zeros(counts.shape[1]) for m in ("constant", "orientation", "direction")}

    for train, test in kf.split(counts):
        # constant model: the MLE rate is just the training mean
        mu = counts[train].mean(axis=0)
        out["constant"] += poisson_loglik(counts[test], mu[None, :]).mean(axis=0)
        for name, X in designs.items():
            model = _fit_population(X[train], counts[train], reg_strength)
            rate = np.asarray(model.predict(X[test]))
            out[name] += poisson_loglik(counts[test], rate).mean(axis=0)

    for k in out:
        out[k] /= n_splits
    return out


def classify(ll):
    """Label each unit from the cross-validated log-likelihoods."""
    return np.where(
        ll["orientation"] > ll["constant"],
        np.where(ll["direction"] > ll["orientation"], "direction", "orientation"),
        "untuned",
    )


def fitted_curves(counts, angles_deg, model_name="direction", grid=None, reg_strength=RIDGE):
    """Fit on all trials and return the predicted count per trial over a fine grid."""
    if grid is None:
        grid = np.linspace(0, 360, 361)
    X = design_matrix(angles_deg, model_name)
    Xg = design_matrix(grid, model_name)
    model = _fit_population(X, np.asarray(counts, dtype=float), reg_strength)
    return grid, np.asarray(model.predict(Xg))
