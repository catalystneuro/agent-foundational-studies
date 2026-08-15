"""Poisson GLM description of the place code, using NeMoS.

The occupancy-normalized rate map is a nonparametric estimate. Fitting a Poisson
GLM with a spline basis over position gives a smooth, regularized version of the
same tuning and, more usefully, lets us ask how much of a place cell's spiking is
explained by position alone versus position plus running speed, running direction,
and the cell's own recent spike history. Model comparison is by cross-validated
McFadden pseudo-R^2 on held-out traversals.
"""
import pickle
import warnings

warnings.simplefilter("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm

import pf_lib as pf

SESSION = "Achilles_10252013"
BIN = 0.025  # s, GLM time bin
N_POS_BASIS = 12
N_SPEED_BASIS = 5
HIST_WINDOW_S = 0.2
COL = {"right": "#1f77b4", "left": "#d62728"}

s = pf.prepare(SESSION, pf.session_urls()[SESSION])
pos, L, eps, pyr = s["pos"], s["track_length"], s["eps"], s["pyr"]
vel = s["vel"]
run = eps["run"]

rng = np.random.default_rng(0)
mets = {d: pf.direction_metrics(pyr, pos, eps[d], L, rng=rng) for d in pf.DIRS}
pc_mask = (
    mets["right"]["metrics"].is_place_cell.values | mets["left"]["metrics"].is_place_cell.values
)
pc_ids = np.asarray(pyr.index)[pc_mask]
pc = pyr[list(pc_ids)]
print(f"fitting GLMs for {len(pc)} place cells, {BIN * 1000:.0f} ms bins")

# ------------------------------------------------------------ design matrix
counts = pc.count(BIN, ep=run)  # TsdFrame (time x unit)
pos_b = pos.interpolate(counts, ep=run)
vel_b = vel.interpolate(counts, ep=run)
speed_b = np.abs(np.asarray(vel_b.values))
dir_b = (np.asarray(vel_b.values) > 0).astype(float)
print(f"{counts.shape[0]} time bins over {run.tot_length():.0f} s of running")

pos_basis = nmo.basis.BSplineEval(n_basis_funcs=N_POS_BASIS, label="position")
pos_basis.set_input_shape(1)
speed_basis = nmo.basis.BSplineEval(n_basis_funcs=N_SPEED_BASIS, label="speed")
speed_basis.set_input_shape(1)
hist_basis = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=5, window_size=int(HIST_WINDOW_S / BIN), label="history"
)

# position features, scaled to [0, 1] so the spline domain is fixed
xpos = np.clip(np.asarray(pos_b.values) / L, 0, 1)
Xpos = pos_basis.compute_features(xpos)
Xspeed = speed_basis.compute_features(np.clip(speed_b / 1.5, 0, 1))
# Direction-specific place fields: the same spline basis, gated by direction.
Xposdir = np.hstack([Xpos * dir_b[:, None], Xpos * (1 - dir_b)[:, None]])

MODELS = {
    "position": lambda uid: Xpos,
    "position x direction": lambda uid: Xposdir,
    "position + speed": lambda uid: np.hstack([Xpos, Xspeed]),
    "position x direction\n+ speed": lambda uid: np.hstack([Xposdir, Xspeed]),
    "position x direction\n+ speed + history": lambda uid: np.hstack(
        [Xposdir, Xspeed, np.asarray(hist_basis.compute_features(counts.loc[uid]))]
    ),
}

# Two-fold cross-validation over traversals, split within direction so that both
# folds contain runs in both directions.
def half(sl):
    a = nap.IntervalSet(start=eps["right"].start[sl], end=eps["right"].end[sl])
    b = nap.IntervalSet(start=eps["left"].start[sl], end=eps["left"].end[sl])
    return a.union(b)


fa, fb = half(slice(None, None, 2)), half(slice(1, None, 2))
tvec = np.asarray(counts.t)


def in_ep(ep):
    m = np.zeros(len(tvec), bool)
    st, en = np.asarray(ep.start), np.asarray(ep.end)
    for a, b in zip(st, en):
        m |= (tvec >= a) & (tvec <= b)
    return m


masks = [(in_ep(fa), in_ep(fb)), (in_ep(fb), in_ep(fa))]

scores = {k: [] for k in MODELS}
pred_curves = {}
for uid in tqdm(pc_ids, desc="units"):
    y = np.asarray(counts.loc[uid].values, dtype=float)
    for mname, build in MODELS.items():
        X = build(uid)
        ok = np.isfinite(X).all(axis=1)
        vals = []
        for tr, te in masks:
            a, b = tr & ok, te & ok
            model = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                                regularizer_strength=1e-4)
            model.fit(X[a], y[a])
            vals.append(float(model.score(X[b], y[b], score_type="pseudo-r2-McFadden")))
        scores[mname].append(np.mean(vals))
    # full-data fit of the direction-gated position model, for the tuning curves
    X = Xposdir
    ok = np.isfinite(X).all(axis=1)
    model = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge", regularizer_strength=1e-4)
    model.fit(X[ok], y[ok])
    grid = np.linspace(0, 1, 200)
    Bg = pos_basis.compute_features(grid)
    curves = {}
    for k, d in enumerate(("right", "left")):
        Xg = np.zeros((len(grid), Xposdir.shape[1]))
        Xg[:, k * N_POS_BASIS : (k + 1) * N_POS_BASIS] = Bg
        curves[d] = np.asarray(model.predict(Xg)) / BIN
    pred_curves[uid] = curves

sc = pd.DataFrame(scores, index=pc_ids)
print("\ncross-validated McFadden pseudo-R^2 (mean over units):")
for k in MODELS:
    print(f"  {k.replace(chr(10), ' '):45s} {sc[k].mean():.4f}  "
          f"(median {sc[k].median():.4f})")

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(16, 9.5))
gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.1, 1.2], hspace=0.55, wspace=0.35)

# GLM tuning curves against the empirical rate maps, 8 example cells
tcs = {d: mets[d]["tc_smooth"] for d in pf.DIRS}
centers = np.asarray(tcs["right"].index)
best = sc["position x direction"].sort_values(ascending=False).index[:8]
for j, uid in enumerate(best):
    ax = fig.add_subplot(gs[j // 4, j % 4])
    for d in pf.DIRS:
        ax.plot(centers, tcs[d][uid].values, color=COL[d], lw=1.0, alpha=0.5)
        ax.plot(np.linspace(0, L, 200), pred_curves[uid][d], color=COL[d], lw=2.0)
    ax.set_xlim(0, L)
    ax.set_ylim(0, None)
    ax.set_title(f"unit {uid}  pseudo-$R^2$ = {sc['position x direction'][uid]:.3f}", fontsize=9)
    if j % 4 == 0:
        ax.set_ylabel("firing rate (Hz)")
    if j >= 4:
        ax.set_xlabel("position (m)")
fig.text(0.5, 0.935,
         "Poisson GLM with a 12-element B-spline basis over position (thick) reproduces the "
         "occupancy-normalized rate map (thin)",
         ha="center", fontsize=11)

ax = fig.add_subplot(gs[2, 0:2])
names = list(MODELS)
data = [sc[k].values for k in names]
bp = ax.boxplot(data, labels=[n.replace("\n", " ") for n in names], showfliers=False,
                patch_artist=True, widths=0.6)
for patch in bp["boxes"]:
    patch.set_facecolor("#4c72b0")
for k, v in enumerate(data):
    ax.plot(np.full(len(v), k + 1) + rng.normal(0, 0.06, len(v)), v, ".",
            color="k", ms=3, alpha=0.4)
ax.set_ylabel("cross-validated pseudo-$R^2$")
ax.tick_params(axis="x", rotation=25, labelsize=8)
for lab in ax.get_xticklabels():
    lab.set_ha("right")
ax.set_title(f"Model comparison across {len(pc)} place cells", fontsize=11)

ax = fig.add_subplot(gs[2, 2])
ax.scatter(sc["position"], sc["position x direction"], s=14, c="k", alpha=0.6,
           edgecolors="none")
lim = [0, max(sc["position x direction"].max(), sc["position"].max()) * 1.05]
ax.plot(lim, lim, "r--", lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("pseudo-$R^2$, position only")
ax.set_ylabel("pseudo-$R^2$, position x direction")
frac = 100 * np.mean(sc["position x direction"] > sc["position"])
ax.set_title(f"Direction helps in {frac:.0f}% of cells", fontsize=11)

ax = fig.add_subplot(gs[2, 3])
# position basis functions, for reference
gridx, bvals = pos_basis.evaluate_on_grid(200)
ax.plot(gridx * L, bvals, lw=1.2)
ax.set_xlabel("position (m)")
ax.set_ylabel("basis value")
ax.set_title(f"{N_POS_BASIS} B-spline basis functions", fontsize=11)

fig.savefig("fig09_glm.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig09 done")

sc.to_csv("glm_scores.csv")
with open("glm.pkl", "wb") as fh:
    pickle.dump({"scores": sc, "pred_curves": pred_curves, "L": L}, fh)
s["io"].close()
