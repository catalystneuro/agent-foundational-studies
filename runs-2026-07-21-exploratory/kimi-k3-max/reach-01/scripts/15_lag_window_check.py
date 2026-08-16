"""Diagnostic: is the GLM lag window wide enough? Refit full-data velocity GLM with a
much wider window (SHIFT=8, WIN=20 -> lags +350..-600 ms) and inspect peak-lag distribution
plus example filter shapes. Edge pile-up that moves interior => window was too narrow."""
import numpy as np, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import nemos as nmo

with open('data/velocity_stats.pkl','rb') as f:
    V = pickle.load(f)
with open('data/direction_stats.pkl','rb') as f:
    D = pickle.load(f)
res = D['res']

counts = V['counts']
vx, vy = V['vx'], V['vy']
trial_of_bin = V['trial_of_bin']; BIN = V['BIN']
n_bins, n_units = counts.shape

SHIFT, WIN, NB = 8, 20, 6

def shift_sig(x, k):
    y = np.empty_like(x); y[:-k] = x[k:]; y[-k:] = x[-k:]
    return y

basis = nmo.basis.RaisedCosineLinearConv(n_basis_funcs=NB, window_size=WIN)
X_vx = np.asarray(basis.compute_features(shift_sig(vx, SHIFT)))
X_vy = np.asarray(basis.compute_features(shift_sig(vy, SHIFT)))
X_sp = np.asarray(basis.compute_features(shift_sig(V['spd'], SHIFT)))
X_vel = np.concatenate([X_vx, X_vy], axis=1)
X_full = np.concatenate([X_vx, X_vy, X_sp], axis=1)

_, widx = np.unique(trial_of_bin, return_index=True)
within = np.arange(n_bins) - widx[trial_of_bin]
tlen = np.bincount(trial_of_bin)[trial_of_bin]
valid = (within >= WIN) & (within < tlen - SHIFT) & ~np.isnan(X_full).any(1)
print(f"valid bins: {valid.sum()}/{n_bins}", flush=True)

# full model: speed features absorb the speed-profile dynamics, leaving cleaner
# directional velocity filters
m = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=0.01,
                          solver_name="LBFGS", solver_kwargs={"tol":1e-10, "maxiter":300})
m.fit(X_full[valid], counts[valid])
print("fit done", flush=True)

_, kernels = basis.evaluate_on_grid(WIN)
coef = np.asarray(m.coef_)
fx = kernels @ coef[:NB]
fy = kernels @ coef[NB:2*NB]
lag_axis = (SHIFT - 1 - np.arange(WIN)) * BIN * 1000
norm = np.hypot(fx, fy)
peak_k = np.argmax(norm, axis=0)
pref_lag = lag_axis[peak_k]
sig = res['anova_p_move'] < 0.01

edge = (peak_k == 0) | (peak_k == WIN-1)
print(f"peak at window edge: {edge.sum()}/{n_units} ({edge[sig].sum()}/{sig.sum()} tuned)")
print("peak lag percentiles (tuned):",
      np.percentile(pref_lag[sig], [10, 25, 50, 75, 90]))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
ax.hist(pref_lag[sig], bins=np.arange(lag_axis.min(), lag_axis.max()+50, 50),
        color="tab:purple", alpha=0.8)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("GLM preferred lag (ms; + = neural leads)"); ax.set_ylabel("units")
ax.set_title(f"A  Peak lags, wide window (+350..-600 ms)\nmedian {np.median(pref_lag[sig]):.0f} ms, "
             f"edge {edge[sig].sum()}/{sig.sum()}", loc="left", fontsize=10)

# example filter norms: 6 tuned units with strong filters + the panel-A example unit
ax = axes[0, 1]
strong = np.argsort(norm.max(0))[-8:]
for u in strong:
    ax.plot(lag_axis, norm[:, u]/norm[:, u].max(), lw=1, alpha=0.8, label=f"u{u}")
ax.axvline(0, color="gray", ls=":", lw=0.8)
ax.set_xlabel("lag (ms)"); ax.set_ylabel("filter norm (self-normalized)")
ax.legend(fontsize=7, ncol=2)
ax.set_title("B  Filter norms, 8 strongest units", loc="left", fontsize=10)

# peak lag vs tuning-curve-based cross-correlation lag for comparison
ax = axes[1, 0]
pd_glm = (np.degrees(np.arctan2(fy[peak_k, np.arange(n_units)],
                                fx[peak_k, np.arange(n_units)])) + 360) % 360
okg = sig & (norm.max(0) > np.percentile(norm.max(0), 25))
d_glm = (res["pd_move"][okg] - pd_glm[okg] + 180) % 360 - 180
ax.scatter(res["pd_move"][okg], pd_glm[okg], s=12, alpha=0.6, c="tab:green")
ax.plot([0, 360], [0, 360], "k--", lw=0.8)
ax.set_xlabel("PD from tuning curve (deg)"); ax.set_ylabel("PD from GLM filters (deg)")
ax.set_title(f"C  GLM vs tuning-curve PD, wide window\n(n={okg.sum()}, med |Δ|={np.median(np.abs(d_glm)):.0f}°)",
             loc="left", fontsize=10)

ax = axes[1, 1]
with open('data/session_cache.pkl','rb') as f:
    C = pickle.load(f)
grp = np.array(C['units']['group'])
for g, c in [("M1", "tab:red"), ("PMd", "tab:blue")]:
    msk = sig & (grp == g)
    ax.hist(pref_lag[msk], bins=np.arange(lag_axis.min(), lag_axis.max()+50, 50),
            alpha=0.6, color=c, label=f"{g} (n={msk.sum()})")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("GLM preferred lag (ms)"); ax.set_ylabel("units"); ax.legend(fontsize=9)
ax.set_title("D  Peak lags by area", loc="left", fontsize=10)

fig.suptitle("fig08  Lag-window diagnostic: velocity+speed model, SHIFT=8, WIN=20, NB=6", fontsize=13)
fig.tight_layout()
fig.savefig("figures/fig08_lag_window_check.png", dpi=150, bbox_inches="tight")
print("saved figures/fig08_lag_window_check.png")

with open('data/lag_window_check_full.pkl','wb') as f:
    pickle.dump(dict(fx=fx, fy=fy, lag_axis=lag_axis, pref_lag=pref_lag, pd_glm=pd_glm,
                     norm=norm, valid=valid), f)
