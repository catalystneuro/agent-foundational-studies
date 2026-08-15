"""Velocity tuning in motor cortex.

Moves from the trial-level "reach direction" description to a continuous
instantaneous-velocity description: at every 20 ms of every reach, does the
firing rate depend on the direction and the magnitude of the hand velocity?

Model comparison is done with Poisson GLMs (NeMoS), scored by cross-validated
McFadden pseudo-R^2 against a constant-rate model.
"""

import jax

jax.config.update("jax_enable_x64", True)  # LBFGS does not converge in float32 here

import jax.numpy as jnp
import numpy as np
import pynapple as nap
import nemos as nmo
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold
from tqdm import tqdm

import mc_maze_io as mio

BIN = 0.02              # s, analysis bin for the GLMs
LAG_BIN = 0.01          # s, finer bin for the lag scan
WIN = (-0.20, 0.60)     # s relative to movement onset
LAGS = np.arange(-0.20, 0.301, 0.02)   # neural lead (>0 = spikes precede kinematics)

d = mio.load_mc_maze()
spikes, kin, trials = d["spikes"], d["kin"], d["trials"]
n_units = len(spikes)
onset = trials["move_onset_time"].values
print(f"{n_units} units, {len(onset)} reaches (all maze versions)")

# raw kinematic arrays for fast interpolation at arbitrary lags
kt = np.asarray(kin.index)
kv = kin.values  # x, y, vx, vy
assert np.isfinite(kv).all(), "unexpected NaNs in kinematics"
speed_all = np.hypot(kv[:, 2], kv[:, 3])
print("speed percentiles (mm/s):", np.percentile(speed_all, [50, 90, 99, 100]).round(0))


def kin_tensor(bin_size, lag):
    """(n_trials, n_bins, 4) hand kinematics at bin centres shifted by `lag`."""
    tc = WIN[0] + bin_size * (np.arange(round((WIN[1] - WIN[0]) / bin_size)) + 0.5)
    tt = onset[:, None] + tc[None, :] + lag
    out = np.stack([np.interp(tt, kt, kv[:, j]) for j in range(4)], axis=-1)
    return out, tc


def count_tensor(bin_size):
    """(n_units, n_trials, n_bins) spike counts."""
    ep = nap.IntervalSet(start=onset + WIN[0], end=onset + WIN[1])
    return nap.build_tensor(spikes, ep, bin_size=bin_size)


# ----------------------------------------------------------------- 1. optimal lag
# Linear (Gaussian) fit of lightly smoothed rate on [vx, vy, speed]; the point is
# the *shape* of R^2 vs lag, which is robust to the choice of smoothing.
#
# The scan is run twice. Restricted to bins where the hand is moving it measures
# the lead of the motor command over the arm. Run over every bin in the window it
# also picks up the instructed-delay period, where preparatory activity predicts a
# velocity that is still hundreds of milliseconds away, which pushes the apparent
# lead much higher. The moving-bin optimum is the one used for the GLMs, because
# those are fit on moving bins.
cnt_lag = count_tensor(LAG_BIN)
Y_all = gaussian_filter1d(cnt_lag / LAG_BIN, 3.0, axis=-1,
                          mode="nearest").reshape(n_units, -1).T
_, tc_lag = kin_tensor(LAG_BIN, 0.0)
K0, _ = kin_tensor(LAG_BIN, 0.0)
mv = (np.hypot(K0[:, :, 2], K0[:, :, 3]).ravel() > 50.0)

r2_by_lag = np.empty((n_units, len(LAGS)))       # moving bins only
r2_by_lag_all = np.empty((n_units, len(LAGS)))   # every bin in the window
for i, lag in enumerate(tqdm(LAGS, desc="lag scan")):
    K, _ = kin_tensor(LAG_BIN, lag)
    v = K[:, :, 2:].reshape(-1, 2)
    X_full = np.column_stack([v, np.hypot(v[:, 0], v[:, 1])])
    for out, sel in ((r2_by_lag, mv), (r2_by_lag_all, slice(None))):
        X = X_full[sel] - X_full[sel].mean(0)
        Y = Y_all[sel] - Y_all[sel].mean(0)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        out[:, i] = 1 - ((Y - X @ beta) ** 2).sum(0) / (Y ** 2).sum(0)

opt_lag = LAGS[np.argmax(r2_by_lag.mean(0))]
best_lag = LAGS[np.argmax(r2_by_lag, axis=1)]
print(f"neural lead over the hand, moving bins: population optimum "
      f"{1000*opt_lag:.0f} ms, median single unit {1000*np.median(best_lag):.0f} ms")
print(f"  (including delay-period bins the optimum shifts to "
      f"{1000*LAGS[np.argmax(r2_by_lag_all.mean(0))]:.0f} ms)")
del cnt_lag, Y_all

# ----------------------------------------------------------------- 2. GLM design
cnt = count_tensor(BIN)                                  # (n_units, n_trials, n_bins)
K, tc = kin_tensor(BIN, opt_lag)
n_trials, n_bins = K.shape[:2]
vx, vy = K[:, :, 2].ravel(), K[:, :, 3].ravel()
speed = np.hypot(vx, vy)
theta = np.arctan2(vy, vx)
counts = cnt.reshape(n_units, -1).T.astype(float)        # (N, n_units)

# only model bins where the hand is actually moving: direction is undefined at rest
moving = speed > 50.0
print(f"{moving.sum()}/{len(moving)} bins with speed > 50 mm/s "
      f"({100*moving.mean():.0f}%)")
vx, vy, speed, theta, counts = (a[moving] for a in (vx, vy, speed, theta, counts))
trial_id = np.repeat(np.arange(n_trials), n_bins)[moving]

# NeMoS bases: direction is periodic, speed is not
dir_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, label="direction")
spd_basis = nmo.basis.BSplineEval(n_basis_funcs=6, label="speed")
sp_clip = np.clip(speed, 0, np.percentile(speed, 99.5))

X_dir = dir_basis.compute_features(theta)
X_spd = spd_basis.compute_features(sp_clip)
X_add = np.column_stack([X_dir, X_spd])                     # direction + speed
X_mul = (dir_basis * spd_basis).compute_features(theta, sp_clip)  # direction x speed
X_lin = np.column_stack([vx, vy]) / 500.0                   # classic linear velocity

MODELS = {
    "speed only": X_spd,
    "direction only": X_dir,
    "linear velocity ($v_x, v_y$)": X_lin,
    "direction + speed": X_add,
    "direction $\\times$ speed": X_mul,
}
for k, v in MODELS.items():
    print(f"  {k}: {v.shape[1]} features")


def cv_pseudo_r2(X, y, n_splits=5):
    """Cross-validated McFadden pseudo-R^2 per unit (contiguous trial blocks)."""
    out = np.zeros((n_splits, y.shape[1]))
    kf = KFold(n_splits=n_splits, shuffle=False)
    for f, (tr_idx, te_idx) in enumerate(kf.split(np.arange(n_trials))):
        tr = np.isin(trial_id, tr_idx)
        te = ~tr
        glm = nmo.glm.PopulationGLM(
            regularizer="Ridge", regularizer_strength=1e-4,
            solver_name="LBFGS", solver_kwargs={"tol": 1e-6, "maxiter": 1000},
        )
        glm.fit(X[tr], y[tr])
        # aggregate over samples only, so the score stays per neuron
        out[f] = glm.score(X[te], y[te], score_type="pseudo-r2-McFadden",
                           aggregate_sample_scores=lambda a: jnp.mean(a, axis=0))
    return out.mean(0)


cv_scores = {}
for name, X in MODELS.items():
    cv_scores[name] = cv_pseudo_r2(np.asarray(X, dtype=float), counts)
    print(f"cv pseudo-R2 {name:32s} median {np.median(cv_scores[name]):.4f}")

# full model fit on all data, used for the tuning-surface plots
glm_full = nmo.glm.PopulationGLM(
    regularizer="Ridge", regularizer_strength=1e-4,
    solver_name="LBFGS", solver_kwargs={"tol": 1e-6, "maxiter": 1000},
)
glm_full.fit(np.asarray(X_mul, dtype=float), counts)

# model-predicted rate on a (direction x speed) grid
n_g = 60
g_dir = np.linspace(-np.pi, np.pi, n_g, endpoint=False)
g_spd = np.linspace(sp_clip.min(), sp_clip.max(), n_g)
GD, GS = np.meshgrid(g_dir, g_spd, indexing="ij")
X_grid = np.asarray((dir_basis * spd_basis).compute_features(GD.ravel(), GS.ravel()),
                    dtype=float)
surf = np.asarray(glm_full.predict(X_grid)).reshape(n_g, n_g, n_units) / BIN

# empirical firing-rate map in the (direction, speed) plane
n_db, n_sb = 16, 8
db = np.clip(((theta + np.pi) / (2 * np.pi) * n_db).astype(int), 0, n_db - 1)
sedges = np.percentile(speed, np.linspace(0, 99.5, n_sb + 1))  # matches the GLM grid
sb = np.clip(np.searchsorted(sedges, speed, side="right") - 1, 0, n_sb - 1)
emp = np.full((n_units, n_db, n_sb), np.nan)
occ = np.zeros((n_db, n_sb), int)
for i in range(n_db):
    for j in range(n_sb):
        m = (db == i) & (sb == j)
        occ[i, j] = m.sum()
        if m.sum() > 50:
            emp[:, i, j] = counts[m].mean(0) / BIN

# Preferred direction from the GLM: vector average of the speed-averaged direction
# profile, which is the same definition as the PD of the trial-level cosine fit and
# far less noisy than the arg-max of the surface.
prof = surf.mean(axis=1)                      # (n_dir_grid, n_units)
pd_glm = np.angle((np.exp(1j * g_dir)[:, None] * (prof - prof.mean(0))).sum(0))
pref_idx = np.argmin(np.abs(np.angle(np.exp(1j * (g_dir[:, None] - pd_glm)))), axis=0)

# speed gain: slope of rate vs speed in the preferred vs anti-preferred direction
gain = np.empty((n_units, 2))
for u in range(n_units):
    for k, idx in enumerate([pref_idx[u], (pref_idx[u] + n_g // 2) % n_g]):
        gain[u, k] = np.polyfit(g_spd, surf[idx, :, u], 1)[0] * 100  # Hz per 100 mm/s

# how much velocity signal a unit has at all, used to gate the PD comparison
vel_signal = r2_by_lag.max(1)

np.savez(
    "results_velocity.npz",
    LAGS=LAGS, r2_by_lag=r2_by_lag, r2_by_lag_all=r2_by_lag_all,
    best_lag=best_lag, opt_lag=opt_lag, opt_lag_all=LAGS[np.argmax(r2_by_lag_all.mean(0))],
    g_dir=g_dir, g_spd=g_spd, surf=surf, emp=emp, occ=occ,
    sedges=sedges, pd_glm=pd_glm, pref_idx=pref_idx, gain=gain, vel_signal=vel_signal,
    **{f"cv_{k}": v for k, v in cv_scores.items()},
)
print("saved results_velocity.npz")
