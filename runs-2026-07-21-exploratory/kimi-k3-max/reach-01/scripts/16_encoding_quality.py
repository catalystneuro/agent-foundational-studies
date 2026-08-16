"""Per-unit encoding quality of the velocity+speed GLM on held-out trials:
Pearson r between predicted and actual smoothed rate on movement-epoch test bins.
Adds r_enc_move (and r_enc_all) to data/glm_stats.pkl."""
import numpy as np, pickle
import jax
jax.config.update("jax_enable_x64", True)
import nemos as nmo
from scipy.ndimage import gaussian_filter1d

with open('data/velocity_stats.pkl','rb') as f:
    V = pickle.load(f)
with open('data/session_cache.pkl','rb') as f:
    C = pickle.load(f)

counts = V['counts']
vx, vy, spd = V['vx'], V['vy'], V['spd']
trial_of_bin = V['trial_of_bin']; BIN = V['BIN']; bin_times = V['bin_times']
n_bins, n_units = counts.shape

SHIFT, WIN, NB = 5, 10, 4   # same window as the main GLM analysis

def shift_sig(x, k):
    y = np.empty_like(x); y[:-k] = x[k:]; y[-k:] = x[-k:]
    return y

basis = nmo.basis.RaisedCosineLinearConv(n_basis_funcs=NB, window_size=WIN)
X_full = np.concatenate([np.asarray(basis.compute_features(shift_sig(s, SHIFT)))
                         for s in (vx, vy, spd)], axis=1)

_, widx = np.unique(trial_of_bin, return_index=True)
within = np.arange(n_bins) - widx[trial_of_bin]
tlen = np.bincount(trial_of_bin)[trial_of_bin]
valid = (within >= WIN) & (within < tlen - SHIFT) & ~np.isnan(X_full).any(1)

mo = C['trials']['move_onset']
trel = bin_times - mo[trial_of_bin]
move_mask = (trel >= -0.1) & (trel <= 0.5)

folds = np.array_split(np.unique(trial_of_bin), 5)
r_enc_move = np.full(n_units, np.nan)
r_enc_all = np.full(n_units, np.nan)
for fi, te_trials in enumerate(folds):
    te = valid & np.isin(trial_of_bin, te_trials)
    trm = valid & ~np.isin(trial_of_bin, te_trials)
    m = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=0.01,
                              solver_name="LBFGS", solver_kwargs={"tol":1e-10, "maxiter":300})
    m.fit(X_full[trm], counts[trm])
    pred = np.asarray(m.predict(X_full[te])) / BIN      # predicted rate (sp/s)
    act = counts[te] / BIN
    # smooth along time within contiguous test segments (trial blocks are contiguous)
    pred_s = gaussian_filter1d(pred, 2, axis=0)
    act_s = gaussian_filter1d(act, 2, axis=0)
    tm = move_mask[te]
    for u in range(n_units):
        if act_s[tm, u].std() > 0:
            r = np.corrcoef(act_s[tm, u], pred_s[tm, u])[0, 1]
            r_enc_move[u] = r if np.isnan(r_enc_move[u]) else (r_enc_move[u]*fi + r)/(fi+1)
        if act_s[:, u].std() > 0:
            r = np.corrcoef(act_s[:, u], pred_s[:, u])[0, 1]
            r_enc_all[u] = r if np.isnan(r_enc_all[u]) else (r_enc_all[u]*fi + r)/(fi+1)
    print(f"fold {fi} done", flush=True)

with open('data/glm_stats.pkl','rb') as f:
    G = pickle.load(f)
G['r_enc_move'] = r_enc_move
G['r_enc_all'] = r_enc_all
with open('data/glm_stats.pkl','wb') as f:
    pickle.dump(G, f)
print(f"median encoding r (movement bins): {np.nanmedian(r_enc_move):.3f}")
print(f"median encoding r (all bins): {np.nanmedian(r_enc_all):.3f}")
print("saved glm_stats.pkl with encoding quality")
