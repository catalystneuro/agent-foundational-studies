"""Poisson GLM encoding: spike counts from lagged hand kinematics (nemos PopulationGLM)."""
import numpy as np, pickle
import jax
jax.config.update("jax_enable_x64", True)
import nemos as nmo
from scipy.special import gammaln
from tqdm import tqdm

with open('data/velocity_stats.pkl','rb') as f:
    V = pickle.load(f)
with open('data/direction_stats.pkl','rb') as f:
    D = pickle.load(f)
res = D['res']

counts = V['counts']            # (n_bins, n_units)
vx, vy, spd = V['vx'], V['vy'], V['spd']
trial_of_bin = V['trial_of_bin']; BIN = V['BIN']; bin_times = V['bin_times']
n_bins, n_units = counts.shape

SHIFT = 5          # bins (250 ms): feed kinematics shifted earlier so causal conv captures neural lead
WIN = 10           # conv window (bins); tap k sees kinematics at (SHIFT-1-k) bins -> lags +200..-250 ms
NB = 4             # basis funcs per channel

def shift_sig(x, k):
    y = np.empty_like(x); y[:-k] = x[k:]; y[-k:] = x[-k:]
    return y

vxs, vys, spds = shift_sig(vx, SHIFT), shift_sig(vy, SHIFT), shift_sig(spd, SHIFT)
basis = nmo.basis.RaisedCosineLinearConv(n_basis_funcs=NB, window_size=WIN)
X_vx = np.asarray(basis.compute_features(vxs))
X_vy = np.asarray(basis.compute_features(vys))
X_sp = np.asarray(basis.compute_features(spds))
X_vel = np.concatenate([X_vx, X_vy], axis=1)
X_full = np.concatenate([X_vx, X_vy, X_sp], axis=1)

# valid bins: conv window fully inside trial, shift not crossing trial end
_, widx = np.unique(trial_of_bin, return_index=True)
within = np.arange(n_bins) - widx[trial_of_bin]
tlen = np.bincount(trial_of_bin)[trial_of_bin]
valid = (within >= WIN) & (within < tlen - SHIFT) & ~np.isnan(X_full).any(1)
print(f"valid bins: {valid.sum()}/{n_bins}")

# 5 folds over trial blocks
trials_all = np.unique(trial_of_bin)
# movement-epoch mask: bins within [-0.1, 0.5] s of move onset
with open('data/session_cache.pkl','rb') as f:
    C = pickle.load(f)
mo = C['trials']['move_onset']
trel = bin_times - mo[trial_of_bin]
move_mask = (trel >= -0.1) & (trel <= 0.5)

folds = np.array_split(trials_all, 5)
models = {'speed': X_sp, 'velocity': X_vel, 'velocity+speed': X_full}

def poisson_ll(y, mu):
    mu = np.clip(mu, 1e-9, None)
    return (y*np.log(mu) - mu - gammaln(y+1)).mean(0)   # per unit

test_ll = {name: np.zeros((5, n_units)) for name in models}
test_ll_move = {name: np.zeros((5, n_units)) for name in models}
for fi, te_trials in enumerate(folds):
    te = valid & np.isin(trial_of_bin, te_trials)
    trm = valid & ~np.isin(trial_of_bin, te_trials)
    for name, X in models.items():
        m = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=0.01,
                                  solver_name="LBFGS", solver_kwargs={"tol":1e-10, "maxiter":300})
        m.fit(X[trm], counts[trm])
        mu = np.asarray(m.predict(X[te]))
        test_ll[name][fi] = poisson_ll(counts[te], mu)
        tem = te & move_mask
        test_ll_move[name][fi] = poisson_ll(counts[tem], np.asarray(m.predict(X[tem])))
    print(f"fold {fi} done", flush=True)

mean_ll = {k: v.mean(0) for k, v in test_ll.items()}   # per unit
mean_ll_move = {k: v.mean(0) for k, v in test_ll_move.items()}
sig = res['anova_p_move'] < 0.01
print("\nmedian test LL (bits-ish, log-lik per bin):")
for k in models: print(f"  {k}: {np.median(mean_ll[k]):.4f}")
beat = mean_ll['velocity'] > mean_ll['speed']
print(f"units where velocity beats speed-only: {beat.sum()}/{n_units} (sig-tuned: {(beat&sig).sum()}/{sig.sum()})")
beat2 = mean_ll['velocity+speed'] > mean_ll['velocity']
print(f"units where velocity+speed beats velocity: {beat2.sum()}/{n_units}")
print("\nmovement-epoch scoring:")
for k in models: print(f"  {k}: {np.median(mean_ll_move[k]):.4f}")
beatm = mean_ll_move['velocity'] > mean_ll_move['speed']
print(f"velocity beats speed-only (movement bins): {beatm.sum()}/{n_units} (tuned: {(beatm&sig).sum()}/{sig.sum()})")
beatm2 = mean_ll_move['velocity+speed'] > mean_ll_move['velocity']
print(f"velocity+speed beats velocity (movement bins): {beatm2.sum()}/{n_units}")

# --- example unit filters from full-data fit ---
m_full = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=0.01,
                               solver_name="LBFGS", solver_kwargs={"tol":1e-10, "maxiter":300})
m_full.fit(X_vel[valid], counts[valid])
_, kernels = basis.evaluate_on_grid(WIN)          # (WIN, NB)
coef = np.asarray(m_full.coef_)                    # (2*NB, n_units)
fx = kernels @ coef[:NB]                           # (WIN, n_units) filter for vx
fy = kernels @ coef[NB:2*NB]
lag_axis = (SHIFT - 1 - np.arange(WIN)) * BIN * 1000   # ms, + = neural leads (empirically verified conv orientation)
norm = np.hypot(fx, fy)
peak_k = np.argmax(norm, axis=0)
pd_glm = (np.degrees(np.arctan2(fy[peak_k, np.arange(n_units)], fx[peak_k, np.arange(n_units)])) + 360) % 360
pref_lag = lag_axis[peak_k]

with open('data/glm_stats.pkl','wb') as f:
    pickle.dump(dict(test_ll=test_ll, mean_ll=mean_ll, test_ll_move=test_ll_move, mean_ll_move=mean_ll_move, fx=fx, fy=fy, lag_axis=lag_axis,
                     pd_glm=pd_glm, pref_lag=pref_lag, valid=valid, models=list(models)), f)
print("saved glm_stats.pkl")
