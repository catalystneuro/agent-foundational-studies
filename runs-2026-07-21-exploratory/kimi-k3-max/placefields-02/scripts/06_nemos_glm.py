"""Figure 5: Poisson GLM encoding models (nemos) for place cells.

Position -> raised-cosine basis -> Poisson GLM, fit on run-bout time bins
(25.6 ms). Compare GLM-predicted spatial rate maps to binned tuning curves;
pseudo-R2 computed manually from Poisson log-likelihoods.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import nemos as nmo
import sys
sys.path.insert(0, "scripts")
from pf_common import load_session, compute_bouts, bout_axis
from tqdm import tqdm

d = np.load("cache/place_fields.npz")
keys = d["keys"]; cell_type = d["cell_type"]; rates = d["rates"]
is_place = d["is_place"]; si = d["si_real"]; centers = d["centers"]
exc = cell_type == "excitatory"

nwb, _ = load_session()
units = nwb["units"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
t = lin.t
x = np.asarray(lin.values).ravel()
bouts, dt = compute_bouts(t, x)
tau, offsets, T_total = bout_axis(t, x, bouts)

# ---- design matrix: position samples inside bouts ----
in_bout = ~np.isnan(tau) & ~np.isnan(x)
t_b = t[in_bout]
x_b = x[in_bout]
n_bins = len(t_b)
print(f"{n_bins} run-bout time bins of {dt*1000:.0f} ms")

basis = nmo.basis.RaisedCosineLinearEval(n_basis_funcs=12, bounds=(0.0, 1.6))
X = np.asarray(basis.compute_features(x_b))
print("design matrix:", X.shape)

# ---- spike counts per time bin for all units ----
key_list = sorted(units.keys())
counts = np.zeros((n_bins, len(key_list)), dtype=float)
for j, k in enumerate(key_list):
    st = units[k].t
    st = st[(st >= t_b[0]) & (st <= t_b[-1])]
    ib = np.searchsorted(t_b, st, side="right") - 1
    ok = (ib >= 0) & (np.abs(t_b[np.clip(ib, 0, n_bins-1)] - st) < dt)
    np.add.at(counts[:, j], ib[ok], 1)

def poisson_ll(count, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = count * np.log(mu) - mu  # log(count!) cancels in comparisons
    return np.nansum(ll)

pc_idx = np.where(is_place & exc)[0]
key_pos = {k: j for j, k in enumerate(key_list)}

glm_maps = np.full((len(pc_idx), 50), np.nan)
pseudo_r2 = np.zeros(len(pc_idx))
grid = np.linspace(0, 1.6, 50)
X_grid = np.asarray(basis.compute_features(grid))

for ii, ci in enumerate(tqdm(pc_idx, desc="GLM fits")):
    y = counts[:, key_pos[keys[ci]]]
    model = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1e-4,
                        solver_name="LBFGS", solver_kwargs=dict(maxiter=5000))
    model.fit(X, y)
    mu = model.predict(X)              # counts per bin
    mu_grid = model.predict(X_grid) / dt   # Hz
    glm_maps[ii] = mu_grid
    # pseudo-R2 (McFadden-style, vs intercept-only null)
    lam0 = np.full_like(y, y.mean())
    ll_m = poisson_ll(y, np.clip(mu, 1e-12, None))
    ll_0 = poisson_ll(y, np.clip(lam0, 1e-12, None))
    pseudo_r2[ii] = 1 - ll_m / ll_0

print(f"median pseudo-R2 across {len(pc_idx)} place cells: {np.median(pseudo_r2):.3f}")

# agreement between GLM map and binned tuning curve
corr = np.array([
    np.corrcoef(np.nan_to_num(glm_maps[ii]), np.nan_to_num(rates[ci]))[0, 1]
    for ii, ci in enumerate(pc_idx)
])
print(f"median GLM-vs-binned map correlation: {np.median(corr):.3f}")

np.savez("cache/glm_results.npz", glm_maps=glm_maps, pseudo_r2=pseudo_r2,
         corr=corr, pc_keys=keys[pc_idx], grid=grid)

# ---- figure ----
# 4 example cells spanning pseudo-R2 range (high ones)
ex_order = np.argsort(-pseudo_r2)[:4]
fig, axes = plt.subplots(2, 4, figsize=(13, 6),
                         gridspec_kw=dict(hspace=0.5, wspace=0.3,
                                          left=0.06, right=0.97, top=0.88, bottom=0.12))
for col, ei in enumerate(ex_order):
    ci = pc_idx[ei]
    ax = axes[0, col]
    ax.plot(centers, rates[ci], color="k", lw=1.5, label="binned tuning curve")
    ax.plot(grid, glm_maps[ei], color="tab:red", lw=1.5, ls="--", label="Poisson GLM")
    ax.set_title(f"unit {keys[ci]}  (pseudo-$R^2$={pseudo_r2[ei]:.2f})",
                 fontsize=10, loc="left")
    ax.set_xlim(0, 1.6)
    ax.tick_params(labelsize=8)
    if col == 0:
        ax.set_ylabel("firing rate (Hz)", fontsize=9)
        ax.legend(frameon=False, fontsize=8)

# bottom row: population summaries
ax = axes[1, 0]
ax.hist(pseudo_r2, bins=np.linspace(0, 1, 30), color="tab:red", alpha=0.8)
ax.axvline(np.median(pseudo_r2), color="k", ls="--", lw=1,
           label=f"median={np.median(pseudo_r2):.2f}")
ax.set_xlabel("pseudo-$R^2$ (vs intercept-only)", fontsize=9)
ax.set_ylabel("place cells", fontsize=9)
ax.set_title("GLM goodness of fit", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8)
ax.tick_params(labelsize=8)

ax = axes[1, 1]
ax.hist(corr, bins=np.linspace(-1, 1, 30), color="0.4")
ax.axvline(np.median(corr), color="k", ls="--", lw=1,
           label=f"median={np.median(corr):.2f}")
ax.set_xlabel("corr(GLM map, binned map)", fontsize=9)
ax.set_ylabel("place cells", fontsize=9)
ax.set_title("GLM vs binned agreement", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8)
ax.tick_params(labelsize=8)

ax = axes[1, 2]
ax.scatter(si[pc_idx], pseudo_r2, s=14, alpha=0.7, color="tab:blue")
ax.set_xlabel("Skaggs SI (bits/spike)", fontsize=9)
ax.set_ylabel("pseudo-$R^2$", fontsize=9)
ax.set_title("SI vs GLM fit", fontsize=10, loc="left")
ax.tick_params(labelsize=8)

# basis functions
ax = axes[1, 3]
bg = np.asarray(basis.compute_features(np.linspace(0, 1.6, 200)))
ax.plot(np.linspace(0, 1.6, 200), bg, lw=1)
ax.set_xlabel("track position (m)", fontsize=9)
ax.set_title("12 raised-cosine basis functions", fontsize=10, loc="left")
ax.tick_params(labelsize=8)

for col in range(4):
    axes[0, col].set_xlabel("track position (m)", fontsize=9)

fig.suptitle("Poisson GLM encoding of position (nemos) — excitatory place cells", fontsize=12)
fig.savefig("figures/fig5_nemos_glm.png", dpi=150)
print("saved figures/fig5_nemos_glm.png")
