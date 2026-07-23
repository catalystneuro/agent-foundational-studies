"""NeMoS Poisson encoding GLM: how much of each unit's spike count does orientation explain?

Single-trial spike counts are modelled as Poisson with a rate that depends either on
temporal frequency alone (null model) or on temporal frequency and orientation
(full model). Orientation enters through a cyclic B-spline basis evaluated at
2 x direction, which makes the basis 180 deg periodic and therefore blind to
direction: the full model can only improve on the null by capturing orientation
tuning, not direction tuning.

The quantity reported per unit is the gain in held-out (5-fold cross-validated)
Poisson log-likelihood, expressed as a McFadden pseudo-R2. Unlike gOSI this is
computed on trials the model never saw, so it cannot be inflated by overfitting
the tuning curve to noise.

All units of a session are fitted together with nmo.glm.PopulationGLM, which shares
one design matrix across units and is far faster than fitting units one at a time.
"""
import jax
import nemos as nmo
import numpy as np

# The default float32 makes LBFGS stop early on these designs; the fits are small,
# so the extra precision costs nothing and removes the convergence warnings.
jax.config.update("jax_enable_x64", True)
import pandas as pd
from scipy.special import gammaln
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from analyze_helpers import responsive

R = responsive(pd.read_pickle("units.pkl"))
WINDOW = 1.95            # the counting window used during extraction, in seconds
ori_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=6, label="orientation")


def poisson_ll(y, mu):
    """Per-column Poisson log-likelihood; y and mu are (n_trials, n_units)."""
    mu = np.clip(mu, 1e-8, None)
    return np.sum(y * np.log(mu) - mu - gammaln(y + 1), axis=0)


def build_design(dg_ori, dg_tf):
    # Orientation: cyclic B-spline on the doubled angle, so the basis has period 180 deg.
    ang = ((2 * dg_ori) % 360) / 360.0
    x_ori = np.asarray(ori_basis.compute_features(ang))
    # Temporal frequency: one-hot, first level absorbed into the intercept.
    tfs = np.array(sorted(np.unique(dg_tf)))
    x_tf = np.stack([(dg_tf == f).astype(float) for f in tfs[1:]], axis=1)
    return x_tf, np.hstack([x_tf, x_ori])


rows = []
for ses in tqdm(sorted(R.session.unique()), desc="GLM"):
    d = np.load(f"extracted/{ses}.npz", allow_pickle=True)
    rates = d["dg_rates"].astype(float)
    sel = R[R.session == ses]
    idx_map = {int(u): i for i, u in enumerate(d["unit_ids"])}
    take = np.array([idx_map[int(u)] for u in sel.unit_id])

    # PopulationGLM needs one design matrix for all units, so trials invalidated on
    # any contributing probe are dropped rather than NaN-masked per unit.
    ok = np.isfinite(rates[take]).all(axis=0)
    Y = np.rint(rates[np.ix_(take, np.where(ok)[0])] * WINDOW).T   # (n_trials, n_units)
    dg_ori, dg_tf = d["dg_ori"][ok], d["dg_tf"][ok]
    x_null, x_full = build_design(dg_ori, dg_tf)

    gains = np.zeros((5, Y.shape[1]))
    base = np.zeros((5, Y.shape[1]))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for k, (tr, te) in enumerate(cv.split(x_full, dg_ori)):
        ll = {}
        for name, X in (("null", x_null), ("full", x_full)):
            m = nmo.glm.PopulationGLM(
                regularizer="Ridge", regularizer_strength=1e-3, solver_name="LBFGS",
                solver_kwargs=dict(maxiter=2000, tol=1e-10)).fit(X[tr], Y[tr])
            ll[name] = poisson_ll(Y[te], np.asarray(m.predict(X[te])))
        mu0 = np.tile(np.clip(Y[tr].mean(axis=0), 1e-8, None), (len(te), 1))
        ll0 = poisson_ll(Y[te], mu0)
        ll_sat = poisson_ll(Y[te], np.clip(Y[te], 1e-8, None))
        denom = ll_sat - ll0
        denom[denom <= 0] = np.nan
        gains[k] = (ll["full"] - ll["null"]) / denom
        base[k] = (ll["null"] - ll0) / denom

    out = sel.copy()
    out["dR2_orientation"] = np.nanmean(gains, axis=0)
    out["R2_tf"] = np.nanmean(base, axis=0)
    rows.append(out[["session", "unit_id", "area", "region", "gosi", "gosi_corrected",
                     "p_perm", "peak_evoked", "dR2_orientation", "R2_tf"]])

glm = pd.concat(rows, ignore_index=True)
glm.to_csv("glm_results.csv", index=False)
print(glm.groupby("area")["dR2_orientation"].agg(["count", "median", "mean"]).to_string())
