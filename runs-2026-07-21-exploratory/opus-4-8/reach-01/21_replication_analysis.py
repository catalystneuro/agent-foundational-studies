"""Velocity tuning, position/velocity confound, GLM and decoding on MC_RTT, with MC_Maze alongside."""
import pickle

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from tqdm.auto import tqdm

import nemos as nmo
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CACHE = "./cache"
BIN = 0.02


def cosine_fit(rates, angles):
    X = np.column_stack([np.ones_like(angles), np.cos(angles), np.sin(angles)])
    beta, *_ = np.linalg.lstsq(X, rates, rcond=None)
    ss_res = ((rates - X @ beta) ** 2).sum(0)
    ss_tot = ((rates - rates.mean(0)) ** 2).sum(0)
    n = len(angles)
    fstat = (ss_tot - ss_res) / 2 / (ss_res / (n - 3))
    return pd.DataFrame(dict(b0=beta[0], pd=np.arctan2(beta[2], beta[1]),
                             mod_depth=np.hypot(beta[1], beta[2]),
                             r2=1 - ss_res / ss_tot, f=fstat, p=stats.f.sf(fstat, 2, n - 3)))


def shift_by_lag(M, lag_bins, blockkey):
    n = len(M["t"])
    src = np.arange(n) + lag_bins
    ok = (src >= 0) & (src < n)
    ok[ok] &= M[blockkey][src[ok]] == M[blockkey][np.arange(n)[ok]]
    return np.arange(n)[ok], src[ok]


def deviance_explained(y, rate, rate_null):
    def ll(r):
        r = np.clip(r, 1e-9, None)
        return (y * np.log(r) - r - gammaln(y + 1)).sum(0)
    ll_sat = ll(np.clip(y, 1e-9, None))
    return 1 - (ll_sat - ll(rate)) / (ll_sat - ll(rate_null))


def unit_scale(v):
    lo, hi = np.min(v), np.max(v)
    return v if hi - lo == 0 else (v - lo) / (hi - lo)


def fit_cv(X, Y, groups, n_folds=5, reg=1e-3):
    ug = np.unique(groups)
    fold_of = {g: i % n_folds for i, g in enumerate(ug)}
    fold = np.array([fold_of[g] for g in groups])
    de = np.full((n_folds, Y.shape[1]), np.nan)
    for k in tqdm(range(n_folds), desc="CV folds", leave=False):
        tr_i, te_i = fold != k, fold == k
        m = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=reg,
                                  solver_name="LBFGS", solver_kwargs={"tol": 1e-7, "maxiter": 300})
        m.fit(X[tr_i], Y[tr_i])
        de[k] = deviance_explained(Y[te_i], np.asarray(m.predict(X[te_i])),
                                   np.tile(Y[tr_i].mean(0), (te_i.sum(), 1)))
    return de.mean(0)


def position_predicts_velocity(M, groups, n_folds=5):
    """Cross-validated R^2 of predicting (vx, vy) from a spline expansion of (x, y).

    This is the size of the position/velocity confound: if the task geometry makes velocity a
    deterministic function of position, a position-only encoding model can masquerade as a
    velocity model.
    """
    bx = nmo.basis.BSplineEval(n_basis_funcs=6, label="x")
    by = nmo.basis.BSplineEval(n_basis_funcs=6, label="y")
    X = np.asarray((bx * by).compute_features(unit_scale(M["pos"][:, 0]), unit_scale(M["pos"][:, 1])))
    Y = M["vel"]
    pred = np.full_like(Y, np.nan)
    for tr_i, te_i in GroupKFold(n_splits=n_folds).split(X, Y, groups=groups):
        sc = StandardScaler().fit(X[tr_i])
        m = RidgeCV(alphas=np.logspace(-2, 4, 10)).fit(sc.transform(X[tr_i]), Y[tr_i])
        pred[te_i] = m.predict(sc.transform(X[te_i]))
    return 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


# ------------------------------------------------------------------ load both datasets
with open(f"{CACHE}/rtt_binned.pkl", "rb") as fh:
    R = pickle.load(fh)
Mr, keep_r = R["M"], R["keep"]
with open(f"{CACHE}/binned.pkl", "rb") as fh:
    Mz = pickle.load(fh)
Mz["block"] = Mz["trial"]

out = {}

# ------------------------------------------------------------------ 1. lag sweep
LAGS = np.arange(-0.40, 0.52, 0.02)
md_by_lag = []
for lag in tqdm(LAGS, desc="RTT lag sweep"):
    i_n, i_k = shift_by_lag(Mr, int(round(lag / BIN)), "seg")
    mv = Mr["speed"][i_k] > 100
    md_by_lag.append(cosine_fit(Mr["counts"][i_n][mv] / BIN, Mr["vel_angle"][i_k][mv])["mod_depth"].values)
md_by_lag = np.array(md_by_lag)
BEST_LAG = LAGS[int(np.argmax(np.median(md_by_lag, 1)))]
print("MC_RTT: velocity-direction tuning peaks at a neural lead of %d ms" % round(BEST_LAG * 1000))

# ------------------------------------------------------------------ 2. tuning at best lag
i_n, i_k = shift_by_lag(Mr, int(round(BEST_LAG / BIN)), "seg")
mv = Mr["speed"][i_k] > 100
C, A, S = Mr["counts"][i_n][mv], Mr["vel_angle"][i_k][mv], Mr["speed"][i_k][mv]
vel_fit = cosine_fit(C / BIN, A)
print("MC_RTT: %d/%d units tuned for instantaneous velocity direction (p<0.01), median depth %.2f Hz"
      % ((vel_fit["p"] < 0.01).sum(), len(vel_fit), vel_fit["mod_depth"].median()))

nb = 18
vedges = np.linspace(-np.pi, np.pi, nb + 1)
vctr = 0.5 * (vedges[:-1] + vedges[1:])
ib = np.clip(np.digitize(A, vedges) - 1, 0, nb - 1)
tc = np.stack([[C[ib == b, u].mean() / BIN for b in range(nb)] for u in range(C.shape[1])])
tc_se = np.stack([[C[ib == b, u].std() / np.sqrt((ib == b).sum()) / BIN for b in range(nb)]
                  for u in range(C.shape[1])])

# speed tuning within each unit's preferred direction
qs = np.linspace(50.0, np.percentile(Mr["speed"][i_k], 99), 10)
sctr = np.r_[25.0, 0.5 * (qs[:-1] + qs[1:])]
idx = np.digitize(Mr["speed"][i_k], qs)
spd_tun = np.full((len(keep_r), 10), np.nan)
for u in range(len(keep_r)):
    near = (np.abs(np.angle(np.exp(1j * (Mr["vel_angle"][i_k] - vel_fit["pd"].values[u]))))
            < np.pi / 4) | (Mr["speed"][i_k] < 50.0)
    for b in range(10):
        m = near & (idx == b)
        if m.sum() > 30:
            spd_tun[u, b] = Mr["counts"][i_n][m, u].mean() / BIN
print("MC_RTT: population mean rate in the PD %.1f Hz at rest -> %.1f Hz fastest"
      % (np.nanmean(spd_tun[:, 0]), np.nanmean(spd_tun[:, -1])))

# ------------------------------------------------------------------ 3. the confound, both datasets
conf = {}
for nm, MM, gk in [("MC_Maze (center-out)", Mz, "trial"), ("MC_RTT (random target)", Mr, "block")]:
    conf[nm] = position_predicts_velocity(MM, MM[gk])
    print("%-24s cross-validated R2 of velocity predicted from position: vx=%.3f vy=%.3f"
          % (nm, conf[nm][0], conf[nm][1]))

# ------------------------------------------------------------------ 4. GLM comparison on MC_RTT
glm_mask = Mr["speed"][i_k] > 50.0
Y_glm = Mr["counts"][i_n][glm_mask].astype(float)
groups = Mr["block"][i_n][glm_mask]
feat = dict(angle=Mr["vel_angle"][i_k][glm_mask],
            speed=unit_scale(Mr["speed"][i_k][glm_mask]),
            x=unit_scale(Mr["pos"][i_k][glm_mask, 0]),
            y=unit_scale(Mr["pos"][i_k][glm_mask, 1]))
b_ang = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, label="direction")
b_spd = nmo.basis.MSplineEval(n_basis_funcs=5, label="speed")
b_px = nmo.basis.BSplineEval(n_basis_funcs=5, label="pos_x")
b_py = nmo.basis.BSplineEval(n_basis_funcs=5, label="pos_y")
MODELS = {"speed only": (b_spd, ("speed",)),
          "direction only": (b_ang, ("angle",)),
          "direction x speed": (b_ang * b_spd, ("angle", "speed")),
          "position": (b_px * b_py, ("x", "y")),
          "dir x speed + position": (b_ang * b_spd + b_px * b_py, ("angle", "speed", "x", "y"))}
glm_scores = {}
for name, (basis, argnames) in MODELS.items():
    X = np.asarray(basis.compute_features(*[feat[a] for a in argnames]))
    glm_scores[name] = fit_cv(X, Y_glm, groups)
    print("MC_RTT %-24s median held-out deviance explained = %.4f"
          % (name, np.median(glm_scores[name])))

# ------------------------------------------------------------------ 5. velocity decoding on MC_RTT
def lagged_population(counts, block, lags_bins):
    n, u = counts.shape
    outm = np.full((n, u * len(lags_bins)), np.nan)
    idx = np.arange(n)
    for j, L in enumerate(lags_bins):
        src = idx - L
        okk = (src >= 0) & (src < n)
        okk[okk] &= block[src[okk]] == block[idx[okk]]
        outm[np.ix_(okk, np.arange(u) + j * u)] = counts[src[okk]]
    return outm


X_dec = lagged_population(Mr["counts"].astype(np.float32), Mr["seg"], np.arange(0, 12, 2))
good = ~np.isnan(X_dec).any(1)
X_dec, Y_dec, g_dec = X_dec[good], Mr["vel"][good], Mr["block"][good]
pred = np.full_like(Y_dec, np.nan)
for tr_i, te_i in tqdm(list(GroupKFold(n_splits=5).split(X_dec, Y_dec, groups=g_dec)),
                       desc="RTT decoding folds"):
    sc = StandardScaler().fit(X_dec[tr_i])
    m = RidgeCV(alphas=np.logspace(0, 5, 12)).fit(sc.transform(X_dec[tr_i]), Y_dec[tr_i])
    pred[te_i] = m.predict(sc.transform(X_dec[te_i]))
r2_vel = 1 - ((Y_dec - pred) ** 2).sum(0) / ((Y_dec - Y_dec.mean(0)) ** 2).sum(0)
ang_err = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(Y_dec[:, 1], Y_dec[:, 0])
                                                  - np.arctan2(pred[:, 1], pred[:, 0]))))))
fast = np.hypot(*Y_dec.T) > 100
print("MC_RTT velocity decoding: R2(vx)=%.3f R2(vy)=%.3f median direction error %.1f deg"
      % (r2_vel[0], r2_vel[1], np.median(ang_err[fast])))

with open(f"{CACHE}/rtt_results.pkl", "wb") as fh:
    pickle.dump(dict(LAGS=LAGS, md_by_lag=md_by_lag, BEST_LAG=BEST_LAG, vel_fit=vel_fit,
                     vctr=vctr, tc=tc, tc_se=tc_se, sctr=sctr, spd_tun=spd_tun, conf=conf,
                     glm_scores=glm_scores, r2_vel=r2_vel, ang_err=ang_err, fast=fast,
                     Y_dec=Y_dec, pred=pred, g_dec=g_dec, keep=keep_r,
                     i_n=i_n, i_k=i_k), fh)
print("wrote rtt_results.pkl")
