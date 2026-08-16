"""Add the Moran & Schwartz multiplicative model: direction with a speed-dependent gain.

log(rate) = a + (b + c*|v|) . u,  where u is the unit vector of movement direction.
This is the form the tuning-curve analysis actually suggests, unlike exp(b.v), whose
rate grows exponentially with speed.
"""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pynapple as nap
import nemos as nmo
import mcmaze_io as mio
from tqdm import tqdm

nap.nap_config.suppress_conversion_warnings = True
BIN, MOVE_THRESH, N_FOLDS = 0.02, 100.0, 5

nwbfile, nwb = mio.open_nwb()
spikes = nwb["units"]
trials_ep = nwb["trials"]
hand_pos, hand_vel = mio.load_kinematics(nwbfile)
LAG = float(np.load("trial_data.npz")["best_lag"])

vel = nap.TsdFrame(t=hand_vel.t - LAG, d=hand_vel.values, columns=["vx", "vy"]).restrict(trials_ep)
counts = spikes.count(BIN, ep=trials_ep)
V = vel.interpolate(counts, ep=counts.time_support)
sp = np.hypot(*V.values.T)
th = np.arctan2(V["vy"].values, V["vx"].values)
move = sp > MOVE_THRESH
Y = counts.values[move].astype(float)
s, thm = sp[move], th[move]
n_units = Y.shape[1]

N_BLOCKS = N_FOLDS * 20
blk = len(Y) // N_BLOCKS
n_use = blk * N_BLOCKS
Y, s, thm = Y[:n_use], s[:n_use], thm[:n_use]
fold = (np.arange(n_use) // blk) % N_FOLDS

zs = (s - s.mean()) / s.std()
X = np.column_stack([np.cos(thm), np.sin(thm), zs,
                     zs * np.cos(thm), zs * np.sin(thm)])
print("design:", X.shape)


def poisson_ll(y, lam):
    return (y * np.log(np.clip(lam, 1e-10, None)) - lam).sum(0)


ll = np.zeros(n_units)
null_ll = np.zeros(n_units)
coefs = []
for f in tqdm(range(N_FOLDS), desc="folds"):
    tr, te = fold != f, fold == f
    null_ll += poisson_ll(Y[te], np.tile(Y[tr].mean(0), (te.sum(), 1)))
    m = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=1e-4,
                              solver_name="LBFGS",
                              solver_kwargs={"tol": 1e-8, "maxiter": 500})
    m.fit(X[tr], Y[tr])
    ll += poisson_ll(Y[te], np.asarray(m.predict(X[te])))
    coefs.append(np.asarray(m.coef_))

g = np.load("glm_results.npz")
nspk = Y.sum(0)
keep = nspk > 200
bits = (ll - null_ll) / (nspk * np.log(2))
print(f"direction x speed gain: median {np.median(bits[keep]):+.4f} bits/spike, "
      f"mean {bits[keep].mean():+.4f}")
old = (g["ll"] - g["null_ll"]) / (g["nspk"] * np.log(2))
for i, n in enumerate([str(x) for x in g["names"]]):
    print(f"  vs {n:22s} median {np.median(old[i][keep]):+.4f}")

np.savez("glm_results_all.npz",
         names=np.array([str(x) for x in g["names"]] + ["direction x speed gain"]),
         ll=np.vstack([g["ll"], ll]), null_ll=g["null_ll"], nspk=g["nspk"],
         coef_vel=g["coef_vel"], coef_dir=g["coef_dir"],
         coef_gain=np.mean(coefs, axis=0), unit_ids=g["unit_ids"])
print("saved glm_results_all.npz")
