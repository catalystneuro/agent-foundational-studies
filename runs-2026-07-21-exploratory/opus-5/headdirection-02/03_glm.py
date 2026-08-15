"""Poisson GLM of head-direction tuning (NeMoS), with a spike-history control."""

import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm

import hd_analysis as ha
import hd_io

SESSION = "A3707"
BIN = 0.02          # s
N_BASIS = 10        # cyclic B-splines over [0, 2*pi)

assets = hd_io.list_assets("000939")
path, aid, _ = [a for a in assets if SESSION in a[0]][0]
nwb, nwbfile, io = hd_io.open_session(aid, dandiset="000939")
eps = ha.get_epochs(nwbfile)
units = nwb["units"]
ep = eps["wake_square"]
hd = ha.clean_head_direction(nwb["head-direction"], ep)
stats = pd.read_csv("stats_example_session.csv", index_col=0)

counts = units.count(BIN, ep=ep)
angle = hd.interpolate(counts, ep=ep)
t = counts.t

basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=N_BASIS, label="head_direction")
X_hd = np.asarray(basis.compute_features(angle))
hist_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=5, window_size=int(0.25 / BIN),
                                           label="spike_history")

mid = ep.start[0] + ep.tot_length() / 2
is_train = t < mid
valid_hd = ~np.isnan(angle.values) & ~np.isnan(X_hd).any(1)

# basis evaluated on a dense grid, to read the GLM tuning curve back out
grid = np.linspace(0, 2 * np.pi, 180, endpoint=False)
X_grid = np.asarray(basis.compute_features(nap.Tsd(t=np.arange(len(grid)) * 1.0, d=grid)))


def fit_score(X, y, mask):
    tr = mask & is_train
    te = mask & ~is_train
    m = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge", regularizer_strength=1e-5)
    m.fit(X[tr], y[tr])
    r2 = float(m.score(X[te], y[te], score_type="pseudo-r2-McFadden"))
    return m, r2


rows, glm_tuning = [], {}
for uid in tqdm(units.index, desc="GLM per unit"):
    cu = counts.loc[uid]
    y = np.asarray(cu).astype(float)
    X_hist = np.asarray(hist_basis.compute_features(cu))
    valid = valid_hd & ~np.isnan(X_hist).any(1)

    m_hd, r2_hd = fit_score(X_hd, y, valid)
    _, r2_hist = fit_score(X_hist, y, valid)
    _, r2_both = fit_score(np.concatenate([X_hd, X_hist], 1), y, valid)

    glm_tuning[uid] = np.asarray(m_hd.predict(X_grid)) / BIN
    rows.append(dict(unit=uid, r2_hd=r2_hd, r2_hist=r2_hist, r2_both=r2_both))

glm = pd.DataFrame(rows).set_index("unit")
glm = glm.join(stats[["mvl", "pref_dir", "hd_cell", "mean_rate", "is_fs"]])
glm.to_csv("glm_example_session.csv")
print(glm.groupby("hd_cell")[["r2_hd", "r2_hist", "r2_both"]].median())

# ------------------------------------------------------------------ figure 6
tc = ha.tuning_curves(units, hd, ep, nb_bins=60)
hd_ids = list(stats.index[stats.hd_cell])
# three good fits with well-separated preferred directions
ex = []
best = glm.loc[hd_ids].sort_values("r2_hd", ascending=False)
for target in np.radians([60, 180, 300]):
    cand = [i for i in best.index[:40] if i not in ex]
    ex.append(min(cand, key=lambda i: abs(ha.circ_diff(glm.pref_dir[i], target))))
ex += [glm[~glm.hd_cell].sort_values("mean_rate").index[-1]]

fig = plt.figure(figsize=(14, 7.5))
gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.4)

for k, uid in enumerate(ex):
    ax = fig.add_subplot(gs[0, k])
    ax.plot(np.degrees(tc.index.values), tc[uid].values, color="0.6", lw=1,
            label="empirical")
    ax.plot(np.degrees(grid), glm_tuning[uid], color="crimson", lw=2, label="GLM")
    ax.set(xlabel="head direction (deg)", ylabel="rate (Hz)" if k == 0 else "",
           title=f"unit {uid}  {'HD' if glm.hd_cell[uid] else 'non-HD'}\n"
                 f"pseudo-$R^2$ = {glm.r2_hd[uid]:.3f}", xticks=[0, 180, 360])
    if k == 0:
        ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 0])
ax.scatter(glm.mvl, glm.r2_hd, s=16, c=np.where(glm.hd_cell, "crimson", "0.6"))
ax.set(xlabel="mean vector length", ylabel="held-out pseudo-$R^2$ (HD model)",
       title="Directional tuning explains spiking")

ax = fig.add_subplot(gs[1, 1])
lim = [-0.2, 0.55]
n_off = int(((glm.r2_hist < lim[0]) | (glm.r2_hd < lim[0])).sum())
ax.scatter(glm.r2_hist, glm.r2_hd, s=16, c=np.where(glm.hd_cell, "crimson", "0.6"))
ax.plot(lim, lim, "k--", lw=0.8)
ax.set(xlabel="pseudo-$R^2$, spike history only", ylabel="pseudo-$R^2$, head direction only",
       title=f"Head direction beats spike history\nfor HD cells ({n_off} units off scale)",
       xlim=lim, ylim=lim)

ax = fig.add_subplot(gs[1, 2])
ax.scatter(glm.r2_hd, glm.r2_both, s=16, c=np.where(glm.hd_cell, "crimson", "0.6"))
ax.plot(lim, lim, "k--", lw=0.8)
ax.set(xlabel="pseudo-$R^2$, head direction", ylabel="pseudo-$R^2$, direction + history",
       title="Adding spike history helps only a little",
       xlim=lim, ylim=lim)

ax = fig.add_subplot(gs[1, 3])
w = np.array([(glm_tuning[u] >= glm_tuning[u].max() / 2).mean() * 360 for u in hd_ids])
ax.hist(w, bins=20, color="#1b6ca8")
ax.set(xlabel="tuning width at half maximum (deg)", ylabel="HD cells",
       title=f"Tuning width (median {np.median(w):.0f}$\\degree$)")

fig.suptitle("Poisson GLM with a cyclic B-spline basis over head direction (NeMoS)",
             y=0.98)
fig.savefig("fig06_glm.png", dpi=150, bbox_inches="tight")
print("wrote fig06")
