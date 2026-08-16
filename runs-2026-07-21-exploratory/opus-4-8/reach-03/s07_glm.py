"""Poisson GLM (NeMoS): how much of motor-cortical firing is explained by reach
direction, by speed, and by their interaction?

Nested models on binned spike counts during movement, cross-validated over trials.
  null   : mean rate
  dir    : cyclic B-spline over movement direction
  speed  : B-spline over speed
  add    : direction + speed (separable)
  full   : direction x speed (interaction -> speed-dependent gain)
"""
import numpy as np, pandas as pd, os
np.seterr(all='ignore')
import jax
jax.config.update('jax_enable_x64', True)
import pynapple as nap
import nemos as nmo
from tqdm import tqdm
from s01_load import load_all, to_pynapple
from s04_velocity_tuning import movement_epochs, shifted_vel

CACHE = "cache"
BIN = 0.05
SPD_MIN = 50.0
NFOLD = 5
REG = 1e-3          # ridge strength; the same value is used for every nested model


def build_design(spk, vel, K, lag):
    ep = movement_epochs(K)
    cnt = spk.count(BIN, ep=ep)
    tb = cnt.index.values
    v = shifted_vel(vel, lag).interpolate(nap.Ts(tb))
    V = np.asarray(v.values)
    spd = np.hypot(V[:, 0], V[:, 1])
    ang = np.arctan2(V[:, 1], V[:, 0])
    keep = np.isfinite(spd) & (spd > SPD_MIN)
    # trial id for each retained bin (for grouped CV)
    tid = np.searchsorted(np.asarray(ep.start), tb, side='right') - 1
    return (np.asarray(cnt.values)[keep].astype(float), ang[keep],
            np.clip(spd[keep], SPD_MIN, 1200.0), tid[keep])


def poisson_pr2(y, mu):
    """Per-neuron deviance-based pseudo-R^2 against the mean-rate model."""
    mu = np.clip(mu, 1e-9, None)
    ybar = y.mean(0, keepdims=True)
    dev = 2 * np.sum(np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu), 0)
    dev0 = 2 * np.sum(np.where(y > 0, y * np.log(y / ybar), 0.0) - (y - ybar), 0)
    return 1 - dev / dev0


def make_bases():
    b_dir = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, label="direction")
    b_spd = nmo.basis.BSplineEval(n_basis_funcs=5, label="speed")
    return {
        "dir":   (b_dir, ("ang",)),
        "speed": (b_spd, ("spd",)),
        "add":   (b_dir + b_spd, ("ang", "spd")),
        "full":  (b_dir * b_spd, ("ang", "spd")),
    }


if __name__ == "__main__":
    d = load_all(); spk, vel, pos = to_pynapple(d)
    K = pd.read_pickle(f"{CACHE}/kinematics.pkl")
    LAG = float(np.load(f"{CACHE}/best_lag.npy")[0])
    y_all, ang, spd, tid = build_design(spk, vel, K, LAG)
    keep_u = y_all.sum(0) >= 200          # need enough spikes to fit a Poisson GLM
    y = y_all[:, keep_u]
    np.save(f"{CACHE}/glm_units.npy", np.where(keep_u)[0])
    print(f"design: {y.shape[0]} bins x {y.shape[1]}/{y_all.shape[1]} units "
          f"(units with >=200 movement spikes), lag {LAG*1000:.0f} ms, bin {BIN*1000:.0f} ms")

    inputs = dict(ang=ang, spd=spd)
    bases = make_bases()
    rng = np.random.default_rng(0)
    trials = np.unique(tid)
    fold_of_trial = rng.permutation(len(trials)) % NFOLD
    fold = fold_of_trial[np.searchsorted(trials, tid)]

    scores = {k: np.zeros((NFOLD, y.shape[1])) for k in bases}
    coefs = {}
    for name, (basis, args) in bases.items():
        X = basis.compute_features(*[inputs[a] for a in args])
        X = np.asarray(X)
        for f in tqdm(range(NFOLD), desc=f"GLM [{name}] ({X.shape[1]} features)"):
            tr, te = fold != f, fold == f
            glm = nmo.glm.PopulationGLM(
                solver_name="LBFGS",
                regularizer=nmo.regularizer.Ridge(),
                regularizer_strength=REG,
                solver_kwargs=dict(tol=1e-7, maxiter=600),
            ).fit(X[tr], y[tr])
            scores[name][f] = poisson_pr2(y[te], np.asarray(glm.predict(X[te])))
            if f == 0:
                coefs[name] = (np.asarray(glm.coef_), np.asarray(glm.intercept_))
    np.savez(f"{CACHE}/glm_scores.npz", **{k: v for k, v in scores.items()})
    np.savez(f"{CACHE}/glm_coefs.npz", **{f"{k}_coef": v[0] for k, v in coefs.items()},
             **{f"{k}_int": v[1] for k, v in coefs.items()})

    S = {k: v.mean(0) for k, v in scores.items()}
    df = pd.read_pickle(f"{CACHE}/dir_tuning.pkl")
    for k in bases:
        col = np.full(len(df), np.nan); col[keep_u] = S[k]
        df[f"pr2_{k}"] = col
    df.to_pickle(f"{CACHE}/dir_tuning.pkl")
    print(f"\ncross-validated Poisson pseudo-R^2 (median over {y.shape[1]} units)")
    for k in ["dir", "speed", "add", "full"]:
        print(f"  {k:6s} {np.median(S[k]):.4f}   (mean {np.mean(S[k]):.4f})")
    from scipy.stats import wilcoxon
    print("\nfull vs add :", wilcoxon(S['full'], S['add']))
    print("add vs dir  :", wilcoxon(S['add'], S['dir']))
    print(f"units better with speed in the model (add > dir): {np.mean(S['add'] > S['dir']):.1%}")
    print(f"units better with interaction (full > add):      {np.mean(S['full'] > S['add']):.1%}")
