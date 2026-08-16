"""Poisson GLM comparison: is the movement signal direction, speed, or velocity?

All models are fit only on bins in which the hand is actually moving, so that
"speed" cannot win simply by separating movement from rest.
"""
import jax

jax.config.update("jax_enable_x64", True)  # LBFGS does not converge in float32 here

import numpy as np
import pynapple as nap
import nemos as nmo
import mcmaze_io as mio
from tqdm import tqdm

nap.nap_config.suppress_conversion_warnings = True

BIN = 0.02
MOVE_THRESH = 100.0
N_FOLDS = 5

nwbfile, nwb = mio.open_nwb()
spikes = nwb["units"]
trials_ep = nwb["trials"]
hand_pos, hand_vel = mio.load_kinematics(nwbfile)
LAG = float(np.load("trial_data.npz")["best_lag"])

vel = nap.TsdFrame(t=hand_vel.t - LAG, d=hand_vel.values, columns=["vx", "vy"]).restrict(trials_ep)
counts = spikes.count(BIN, ep=trials_ep)
V = vel.interpolate(counts, ep=counts.time_support)
sp = np.hypot(V["vx"].values, V["vy"].values)
th = np.arctan2(V["vy"].values, V["vx"].values)

move = sp > MOVE_THRESH
print(f"moving bins: {move.sum()} / {len(move)} ({100*move.mean():.1f}%)")

Y = counts.values[move].astype(float)
vx, vy, s, thm = V["vx"].values[move], V["vy"].values[move], sp[move], th[move]
n_units = Y.shape[1]

# Contiguous 5-fold split (blocks of time, not shuffled bins, to avoid leakage).
# Every fold is given exactly the same number of bins: JAX recompiles the whole
# LBFGS loop whenever an input shape changes, which otherwise dominates runtime.
N_BLOCKS = N_FOLDS * 20
blk = len(Y) // N_BLOCKS
n_use = blk * N_BLOCKS
Y, vx, vy, s, thm = Y[:n_use], vx[:n_use], vy[:n_use], s[:n_use], thm[:n_use]
fold = (np.arange(n_use) // blk) % N_FOLDS
print(f"using {n_use} bins, {blk} per block, fold sizes {np.bincount(fold)}")

# Standardise regressors so the ridge penalty treats them comparably.
def z(a):
    return (a - a.mean()) / a.std()


SPEED_SCALE = 500.0  # mm/s, for interpretable coefficients
designs = {
    "speed only": np.column_stack([z(s)]),
    "direction only": np.column_stack([np.cos(thm), np.sin(thm)]),
    "direction + speed": np.column_stack([np.cos(thm), np.sin(thm), z(s)]),
    "velocity (vx, vy)": np.column_stack([vx / SPEED_SCALE, vy / SPEED_SCALE]),
    "velocity + speed": np.column_stack([vx / SPEED_SCALE, vy / SPEED_SCALE, z(s)]),
}
# nonparametric 2D velocity field as an upper bound
basis2d = (nmo.basis.RaisedCosineLinearEval(n_basis_funcs=6)
           * nmo.basis.RaisedCosineLinearEval(n_basis_funcs=6))
designs["2D velocity basis"] = basis2d.compute_features(
    np.clip(vx, -800, 800) / 800.0, np.clip(vy, -800, 800) / 800.0)
print({k: v.shape for k, v in designs.items()})


def poisson_ll(y, rate):
    """Per-unit Poisson log-likelihood, dropping the constant log(y!) term."""
    rate = np.clip(rate, 1e-10, None)
    return (y * np.log(rate) - rate).sum(0)


results = {}
null_ll = np.zeros(n_units)
for f in range(N_FOLDS):
    tr, te = fold != f, fold == f
    mu = Y[tr].mean(0)
    null_ll += poisson_ll(Y[te], np.tile(mu, (te.sum(), 1)))

for name, X in tqdm(designs.items(), desc="models"):
    ll = np.zeros(n_units)
    coefs = []
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        model = nmo.glm.PopulationGLM(
            regularizer="Ridge", regularizer_strength=1e-4, solver_name="LBFGS",
            solver_kwargs={"tol": 1e-8, "maxiter": 500})
        model.fit(X[tr], Y[tr])
        ll += poisson_ll(Y[te], np.asarray(model.predict(X[te])))
        coefs.append(np.asarray(model.coef_))
    results[name] = dict(ll=ll, coef=np.mean(coefs, axis=0))
    nspk = Y.sum(0)
    bits = (ll - null_ll) / (nspk * np.log(2))
    print(f"  {name:22s} mean bits/spike = {np.nanmean(bits):+.4f}")

np.savez("glm_results.npz",
         names=np.array(list(results)),
         ll=np.array([results[k]["ll"] for k in results]),
         null_ll=null_ll, nspk=Y.sum(0),
         coef_vel=results["velocity (vx, vy)"]["coef"],
         coef_dir=results["direction only"]["coef"],
         coef_velspeed=results["velocity + speed"]["coef"],
         unit_ids=np.array(list(spikes.keys())), speed_scale=SPEED_SCALE)
print("saved glm_results.npz")
