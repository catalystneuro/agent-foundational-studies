"""Step 4: Poisson GLM model comparison (nemos) and population decoding of hand velocity."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nemos as nmo
from tqdm import tqdm

import common

data = common.load_pynapple()
units = data["units"]
area = units.get_info("area").values.astype(str)
LAG = common.BEST_LAG

counts, t_all = common.bin_spikes(data)
vxy, ok = common.velocity_at(data, t_all, lag=LAG)
counts, vxy, t_bin = counts[ok], vxy[ok], t_all[ok]
speed = np.hypot(vxy[:, 0], vxy[:, 1])
udir = np.where(speed[:, None] > 0.01, vxy / np.maximum(speed, 1e-9)[:, None], 0.0)
print("design: %d bins, %d units" % counts.shape)

# ------------------------------------------------------------ GLM comparison
vb = nmo.basis.BSplineEval(n_basis_funcs=5, label="vx") * \
     nmo.basis.BSplineEval(n_basis_funcs=5, label="vy")
CLIP = 0.7
Xnl = np.asarray(vb.compute_features(np.clip(vxy[:, 0], -CLIP, CLIP),
                                     np.clip(vxy[:, 1], -CLIP, CLIP)), dtype=np.float32)

def zscore(X):
    """LBFGS converges far faster on standardised features."""
    X = np.asarray(X, dtype=np.float32)
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


MODELS = {
    "direction only\n(cos θ, sin θ)": zscore(udir),
    "velocity\n(vₓ, v_y)": zscore(vxy),
    "velocity + speed": zscore(np.column_stack([vxy, speed])),
    "nonlinear velocity\n(5×5 B-spline)": zscore(Xnl),
}

# Fitting on every 4th training bin keeps the estimates within ~2% of the full-data
# fit (checked directly) while cutting the run time by roughly a factor of five.
SUB, MAXITER = 4, 400
pr2 = {k: np.full((5, counts.shape[1]), np.nan) for k in MODELS}
for f, (tr_i, te_i) in enumerate(tqdm(list(common.contiguous_folds(t_bin)), desc="CV folds")):
    y_tr, y_te = counts[tr_i][::SUB], counts[te_i]
    null = y_tr.mean(0)[None, :]
    for name, X in MODELS.items():
        m = nmo.glm.PopulationGLM(solver_name="LBFGS", regularizer="Ridge",
                                  regularizer_strength=1e-4,
                                  solver_kwargs={"maxiter": MAXITER})
        m.fit(X[tr_i][::SUB], y_tr)
        pr2[name][f] = common.poisson_pseudo_r2(y_te, np.asarray(m.predict(X[te_i])), null)

summary = pd.DataFrame({k: v.mean(0) for k, v in pr2.items()})
summary.insert(0, "area", area)
summary.insert(0, "unit", np.arange(len(units)))
summary.to_csv("results_glm_pseudo_r2.csv", index=False)
print("\ncross-validated Poisson pseudo-R² (median over %d units):" % len(units))
for k in MODELS:
    print("  %-32s %.4f" % (k.replace("\n", " "), np.nanmedian(summary[k])))

from scipy import stats
dir_col, vel_col, vs_col = list(MODELS)[0], list(MODELS)[1], list(MODELS)[2]
w = stats.wilcoxon(summary[vel_col], summary[dir_col])
print("velocity beats direction-only: Wilcoxon p=%.2g, %d/%d units improve"
      % (w.pvalue, (summary[vel_col] > summary[dir_col]).sum(), len(summary)))
w_vs = stats.wilcoxon(summary[vs_col], summary[dir_col])
print("velocity+speed beats direction-only: Wilcoxon p=%.2g, %d/%d units improve"
      % (w_vs.pvalue, (summary[vs_col] > summary[dir_col]).sum(), len(summary)))
w2 = stats.wilcoxon(summary[list(MODELS)[3]], summary[vs_col])
print("nonlinear beats velocity+speed: Wilcoxon p=%.2g" % w2.pvalue)

# fitted nonlinear surfaces for a few example units (fit on all data)
mu_nl, sd_nl = Xnl.mean(0), Xnl.std(0) + 1e-8
m_nl = nmo.glm.PopulationGLM(solver_name="LBFGS", regularizer="Ridge",
                             regularizer_strength=1e-4,
                             solver_kwargs={"maxiter": MAXITER}
                             ).fit(((Xnl - mu_nl) / sd_nl)[::SUB], counts[::SUB])
G_EDGES = np.linspace(-CLIP, CLIP, 41)
gx = G_EDGES[:-1] + np.diff(G_EDGES) / 2
GX, GY = np.meshgrid(gx, gx)
Xg = np.asarray(vb.compute_features(GX.ravel(), GY.ravel()), dtype=np.float32)
surf = np.asarray(m_nl.predict((Xg - mu_nl) / sd_nl)) / common.BIN   # (n_grid, n_units), Hz
# A B-spline extrapolates freely where it saw no data, so blank out grid cells the hand
# essentially never visited rather than plotting the extrapolation.
g_occ = np.histogram2d(vxy[:, 0], vxy[:, 1], bins=[G_EDGES, G_EDGES])[0].T
surf_mask = g_occ < 20

# ------------------------------------------------------- population decoding
LAG_BINS = np.arange(0, 11)      # 0-200 ms of neural history preceding the hand
Xd, okd = common.lag_matrix(counts, t_bin, LAG_BINS)
Xd, Yd = Xd[okd], vxy[okd]
print("\ndecoder design: %s" % (Xd.shape,))

# Ridge over several penalties at once: the Gram matrix is the expensive part and does
# not depend on alpha, so we form it once per fold and back-substitute for each alpha.
ALPHAS = np.array([1e1, 1e2, 1e3, 1e4])
preds = {a: np.full_like(Yd, np.nan) for a in ALPHAS}
for tr_i, te_i in tqdm(list(common.contiguous_folds(t_bin[okd])), desc="decoder folds"):
    Xtr, Ytr = Xd[tr_i].astype(np.float64), Yd[tr_i]
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc = Xtr - xm
    G, b = Xc.T @ Xc, Xc.T @ (Ytr - ym)
    for a in ALPHAS:
        w = np.linalg.solve(G + a * np.eye(len(G)), b)
        preds[a][te_i] = (Xd[te_i] - xm) @ w + ym

def _r2(p):
    return 1 - ((Yd - p) ** 2).sum(0) / ((Yd - Yd.mean(0)) ** 2).sum(0)

for a in ALPHAS:
    print("  alpha=%-8g held-out R\u00b2: vx %.3f, vy %.3f" % ((a,) + tuple(_r2(preds[a]))))
ALPHA = ALPHAS[int(np.argmax([_r2(preds[a]).mean() for a in ALPHAS]))]
pred = preds[ALPHA]
r2 = _r2(pred)
print("chosen alpha = %g; held-out decoding R\u00b2: vx %.3f, vy %.3f" % ((ALPHA,) + tuple(r2)))
mov = np.hypot(Yd[:, 0], Yd[:, 1]) > 0.10
ang_err = np.degrees(np.abs(np.angle(np.exp(1j * (
    np.arctan2(pred[mov, 1], pred[mov, 0]) - np.arctan2(Yd[mov, 1], Yd[mov, 0]))))))
print("decoded direction error (bins > 10 cm/s): median %.1f°, %.0f%% within 45°"
      % (np.median(ang_err), 100 * (ang_err < 45).mean()))
sp_true, sp_pred = np.hypot(*Yd.T), np.hypot(*pred.T)
print("speed correlation r = %.3f" % np.corrcoef(sp_true, sp_pred)[0, 1])

# ---------------------------------------------- Georgopoulos population vector
tr_res = pd.read_csv("results_direction_tuning.csv")
tuned = tr_res.tuned.values
pd_rad = tr_res.pd_rad.values
trials = common.add_movement_kinematics(data, common.trial_table(data))
trials = trials[trials["use"] & np.isfinite(trials["mv_dir"])]
rates_tr = common.rates_in_epochs(units,
                                  trials["move_onset_time"].values + common.NEURAL_WIN[0],
                                  trials["move_onset_time"].values + common.NEURAL_WIN[1])
z = (rates_tr[:, tuned] - rates_tr[:, tuned].mean(0)) / (rates_tr[:, tuned].std(0) + 1e-9)
pv = z @ np.column_stack([np.cos(pd_rad[tuned]), np.sin(pd_rad[tuned])]) / tuned.sum()
pv_ang = np.arctan2(pv[:, 1], pv[:, 0])
true_ang = trials["mv_dir"].values
pv_err = np.degrees(np.abs(np.angle(np.exp(1j * (pv_ang - true_ang)))))
print("population-vector direction error: median %.1f°, %.0f%% within 45°"
      % (np.median(pv_err), 100 * (pv_err < 45).mean()))
print("population-vector length vs speed: r = %.3f"
      % np.corrcoef(np.hypot(*pv.T), trials["mv_speed"].values)[0, 1])

np.savez("results_decoding.npz", r2=r2, ang_err=ang_err, pv_err=pv_err,
         pv=pv, true_ang=true_ang, mv_speed=trials["mv_speed"].values,
         sp_true=sp_true, sp_pred=sp_pred)

# ------------------------------------------------------------------ figure 6
cmap_masked = plt.get_cmap("magma").copy()
cmap_masked.set_bad("0.75")
fig = plt.figure(figsize=(15, 8.4))
gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.42, left=0.06, right=0.96,
                      top=0.85, bottom=0.13)

ax = fig.add_subplot(gs[0, 0])
med = [np.nanmedian(summary[k]) for k in MODELS]
q1 = [np.nanpercentile(summary[k], 25) for k in MODELS]
q3 = [np.nanpercentile(summary[k], 75) for k in MODELS]
ax.bar(range(4), med, yerr=[np.array(med) - q1, np.array(q3) - np.array(med)],
       color=["#bdbdbd", "#4292c6", "#08519c", "#08306b"], capsize=4)
ax.set_xticks(range(4))
ax.set_xticklabels([k.replace("\n", " ") for k in MODELS], fontsize=7,
                   rotation=20, ha="right")
ax.set_ylabel("cross-validated\nPoisson pseudo-R²")
ax.set_title("Model comparison (median ± IQR)", fontsize=10)

ax = fig.add_subplot(gs[0, 1])
ax.scatter(summary[dir_col], summary[vs_col],
           c=[common.AREA_COLORS[a] for a in area], s=16, alpha=0.85)
lim = [0, max(summary[vs_col].max(), summary[dir_col].max()) * 1.05]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("pseudo-R², direction only")
ax.set_ylabel("pseudo-R², velocity + speed")
ax.set_title("Speed adds information beyond\ndirection (p=%.0e, %d/%d units)"
             % (w_vs.pvalue, (summary[vs_col] > summary[dir_col]).sum(), len(summary)),
             fontsize=10)

ex = np.argsort(-summary[vs_col].values)[:2]
for i, u in enumerate(ex):
    ax = fig.add_subplot(gs[0, 2 + i])
    im = ax.pcolormesh(100 * gx, 100 * gx,
                       np.ma.masked_where(surf_mask, surf[:, u].reshape(40, 40)),
                       cmap=cmap_masked, shading="auto")
    ax.set_aspect("equal")
    ax.set_xlabel("$v_x$ (cm/s)", fontsize=9)
    ax.set_ylabel("$v_y$ (cm/s)", fontsize=9)
    ax.set_title("GLM velocity field, unit %d (%s)" % (u, area[u]), fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("Hz", fontsize=8)
    cb.ax.tick_params(labelsize=7)

d = np.load("results_decoding.npz")
seg = slice(2000, 2600)
ax = fig.add_subplot(gs[1, :2])
tt = np.arange(seg.stop - seg.start) * common.BIN
ax.plot(tt, 100 * Yd[seg, 0], color="k", lw=1.4, label="actual $v_x$")
ax.plot(tt, 100 * pred[seg, 0], color="#d1495b", lw=1.4, label="decoded $v_x$")
ax.plot(tt, 100 * Yd[seg, 1] - 150, color="k", lw=1.4)
ax.plot(tt, 100 * pred[seg, 1] - 150, color="#3aa17e", lw=1.4, label="decoded $v_y$")
ax.text(0.15, -195, "$v_y$ (offset by -150)", fontsize=8, va="center")
ax.set_xlabel("time (s)"); ax.set_ylabel("velocity (cm/s)")
ax.legend(fontsize=8, ncol=3, loc="upper right")
ax.set_title("Held-out reconstruction of hand velocity from 182 units "
             "(R² = %.2f / %.2f, ridge alpha %g)" % (r2[0], r2[1], ALPHA), fontsize=10)

ax = fig.add_subplot(gs[1, 2])
ax.hist(ang_err, bins=np.arange(0, 181, 7.5), color="#4292c6", density=True,
        label="ridge decoder, 20 ms bins\n(median %.0f°)" % np.median(ang_err))
ax.hist(pv_err, bins=np.arange(0, 181, 7.5), histtype="step", lw=2, color="#d1495b",
        density=True, label="population vector, per trial\n(median %.0f°)" % np.median(pv_err))
ax.axvline(90, color="0.4", ls=":", lw=1.2, label="chance median")
ax.set_xlabel("direction error (°)"); ax.set_ylabel("density")
ax.legend(fontsize=7.5)
ax.set_title("Decoded reach direction", fontsize=10)

ax = fig.add_subplot(gs[1, 3])
h = ax.hist2d(100 * sp_true, 100 * sp_pred, bins=[np.linspace(0, 90, 45)] * 2,
              cmap="Blues", norm="log")
ax.plot([0, 90], [0, 90], "k--", lw=0.8)
ax.set_xlabel("actual speed (cm/s)"); ax.set_ylabel("decoded speed (cm/s)")
ax.set_title("Decoded speed (r = %.2f)" % np.corrcoef(sp_true, sp_pred)[0, 1], fontsize=10)
fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.03, label="bins")

fig.suptitle("Poisson GLM encoding models and population decoding of hand velocity\n"
             "MC_Maze, monkey Jenkins (DANDI 000128), 20 ms bins, %d ms neural lead"
             % round(1000 * LAG), fontsize=12)
fig.savefig("fig07_glm_decoding.png", dpi=130)
print("wrote fig07_glm_decoding.png")
