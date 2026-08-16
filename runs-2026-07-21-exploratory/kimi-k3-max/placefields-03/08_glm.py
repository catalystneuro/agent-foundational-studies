"""GLM-based position tuning with nemos: Poisson GLM + B-spline basis over position.

Fits one GLM per place cell, compares GLM-predicted tuning curves to empirical
rate maps, and validates peak positions across the population.
"""
import h5py
import remfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pynapple as nap
import jax
jax.config.update('jax_enable_x64', True)
import nemos as nmo
from pynwb import NWBHDF5IO
from tqdm import tqdm

# ---------------- reload results ----------------
data = np.load('placefield_results.npz', allow_pickle=True)
unit_keys = data['unit_keys']
BIN_CENTERS = data['bin_centers']
results = data['results'][0]

si_all = np.array([results[k]['spatial_info'] for k in unit_keys])
p_all = np.array([results[k]['shuffle_p'] for k in unit_keys])
rate_all = np.array([results[k]['mean_rate_track'] for k in unit_keys])
peak_all = np.array([np.nanmax(results[k]['rate_map']) for k in unit_keys])
exc = np.array([results[k]['cell_type'] == 'excitatory' for k in unit_keys])
is_place = exc & (p_all < 0.05) & (rate_all > 0.1) & (peak_all > 1.0) & ~np.isnan(p_all)
place_keys = [k for k, m in zip(unit_keys, is_place) if m]

# ---------------- reload nwb ----------------
with open('/tmp/achilles_url.txt') as f:
    s3_url = f.read().strip()
disk_cache = remfile.DiskCache('/tmp/remfile_cache_achilles')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
units = nwb['units']
maze = nwb['epochs'][1]
lin = nwb['1.6mLinearMazeLinearizedTimeSeries'].restrict(maze)
lin_t = lin.t
lin_v = lin.values[:, 0]
valid = ~np.isnan(lin_v)
dt = np.median(np.diff(lin_t))

pos_valid = lin_v[valid]
t_valid = lin_t[valid]

# ---------------- design matrix: B-spline over position ----------------
N_BASIS = 12
basis = nmo.basis.BSplineEval(n_basis_funcs=N_BASIS)
X_all = basis.compute_features(lin_v)  # (n_samples, n_basis); NaN rows where lin is NaN
X = np.asarray(X_all[valid])
print("design matrix:", X.shape)

# evaluate basis on a position grid for tuning curves
grid = np.linspace(0, 1.6, 200)
X_grid = np.asarray(basis.compute_features(grid))

def cohen_pseudo_r2(model, X, y):
    """Deviance-based (Cohen) pseudo-R^2 for a fitted Poisson GLM."""
    mu = np.clip(np.asarray(model.predict(X)), 1e-12, None)
    ll_model = np.sum(y * np.log(mu) - mu)
    lam0 = max(y.mean(), 1e-12)
    ll_null = np.sum(y * np.log(lam0) - lam0)
    y_pos = y[y > 0]
    ll_sat = np.sum(y_pos * np.log(y_pos) - y_pos)
    denom = ll_sat - ll_null
    if abs(denom) < 1e-9:
        return np.nan
    return float(1 - (ll_sat - ll_model) / denom)

# ---------------- fit GLMs for all place cells ----------------
glm_curves = {}
glm_peak = {}
emp_peak = {}
glm_score = {}

for k in tqdm(place_keys, desc="GLM fits"):
    st = units[k].restrict(maze).t
    # spike count per position sample bin
    counts, _ = np.histogram(st, bins=np.append(lin_t, lin_t[-1] + dt))
    y = counts[valid].astype(float)
    model = nmo.glm.GLM(solver_name="LBFGS", solver_kwargs=dict(tol=1e-10, maxiter=1000))
    model.fit(X, y)
    # predicted tuning curve in Hz
    rate_grid = np.exp(model.intercept_ + X_grid @ model.coef_) / dt
    glm_curves[k] = rate_grid
    glm_peak[k] = grid[np.argmax(rate_grid)]
    emp_peak[k] = BIN_CENTERS[np.nanargmax(results[k]['rate_map'])]
    glm_score[k] = cohen_pseudo_r2(model, X, y)

np.savez_compressed('glm_results.npz',
                    place_keys=np.array(place_keys),
                    grid=grid,
                    curves=np.array([glm_curves[k] for k in place_keys]),
                    glm_peak=np.array([glm_peak[k] for k in place_keys]),
                    emp_peak=np.array([emp_peak[k] for k in place_keys]),
                    glm_score=np.array([glm_score[k] for k in place_keys]))

# ---------------- Fig 7: GLM vs empirical for 6 examples ----------------
si_place = {k: results[k]['spatial_info'] for k in place_keys}
peak_place = {k: emp_peak[k] for k in place_keys}
top_by_si = sorted(place_keys, key=lambda k: -si_place[k])
chosen = []
for k in top_by_si:
    if all(abs(peak_place[k] - peak_place[c]) > 0.2 for c in chosen):
        chosen.append(k)
    if len(chosen) == 6:
        break
chosen = sorted(chosen, key=lambda k: peak_place[k])

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, k in zip(axes.flat, chosen):
    rm = results[k]['rate_map']
    ax.fill_between(BIN_CENTERS, rm, color='gray', alpha=0.4, label='empirical')
    ax.plot(grid, glm_curves[k], color='crimson', lw=2, label='GLM (Poisson, spline)')
    ax.set_xlim(0, 1.6)
    ax.set_title(f"unit {k}  (pseudo-R$^2$={glm_score[k]:.2f})", fontsize=10)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Firing rate (Hz)')
axes.flat[0].legend(fontsize=8)
plt.suptitle('GLM-predicted vs empirical position tuning', fontsize=12)
plt.tight_layout()
plt.savefig('fig7_glm_tuning.png', dpi=150)
plt.close()
print("saved fig7_glm_tuning.png")

# ---------------- Fig 8: population validation ----------------
gp = np.array([glm_peak[k] for k in place_keys])
ep = np.array([emp_peak[k] for k in place_keys])
gs = np.array([glm_score[k] for k in place_keys])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
ax.scatter(ep, gp, s=15, color='darkblue', alpha=0.7)
ax.plot([0, 1.6], [0, 1.6], 'k--', lw=1)
ax.set_xlabel('Empirical rate-map peak (m)')
ax.set_ylabel('GLM tuning-curve peak (m)')
r = np.corrcoef(ep, gp)[0, 1]
ax.set_title(f'A  Peak position agreement (r={r:.3f})', loc='left', fontsize=11)

ax = axes[1]
ax.hist(gs[~np.isnan(gs)], bins=30, color='slategray')
ax.set_xlabel('GLM pseudo-R$^2$')
ax.set_ylabel('Place cells')
ax.set_title('B  GLM goodness of fit', loc='left', fontsize=11)
plt.tight_layout()
plt.savefig('fig8_glm_validation.png', dpi=150)
plt.close()
print("saved fig8_glm_validation.png")
print("peak correlation: %.3f" % r)
print("median pseudo-R2: %.3f (n=%d finite of %d)" % (np.nanmedian(gs), np.isfinite(gs).sum(), len(gs)))
