"""Nested Poisson GLMs (NeMoS): is motor-cortical activity tuned to direction, speed, or velocity?"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap, nemos as nmo
from tqdm import tqdm
from reach_lib import open_nwb, obs_intervals_set, load_units

BIN2, SPD_TH = 0.05, 100.0
Vr = np.load("_res_velocity.npz"); LAG = float(Vr["best_lag"])
sig = np.load("_res_direction.npz")["sig"]

nwbfile = open_nwb("MC_Maze"); ep = obs_intervals_set(nwbfile); units = load_units(nwbfile, ep)
t = np.load("_cache_t.npy"); vel = np.load("_cache_vel.npy")
counts = units.count(BIN2, ep)
V = nap.TsdFrame(t=t - LAG, d=vel, columns=["vx", "vy"]).restrict(ep).bin_average(BIN2, ep).values
speed = np.hypot(V[:, 0], V[:, 1])
m = np.isfinite(V).all(1) & (speed > SPD_TH)
Y = np.asarray(counts.values[m], dtype=float)
Vm, sp = V[m], speed[m]
th = np.arctan2(Vm[:, 1], Vm[:, 0])
blocks = (np.searchsorted(ep.start, counts.t, side="right") // 60)[m]
print("GLM data: %d bins x %d units, lag %+.0f ms" % (Y.shape[0], Y.shape[1], LAG * 1000))

# ---- design matrices ----
# B-spline bases return NaN outside their bounds, so inputs are clipped into range.
SC = 500.0                                        # scale velocities to O(1) for conditioning
SLO, SHI = SPD_TH, float(np.percentile(sp, 99.5))
VLO, VHI = -800.0, 800.0
Bs = nmo.basis.BSplineEval(6, bounds=(SLO, SHI)).compute_features(np.clip(sp, SLO, SHI))
vb = nmo.basis.BSplineEval(5, bounds=(VLO, VHI)) * nmo.basis.BSplineEval(5, bounds=(VLO, VHI))
Bv2 = vb.compute_features(np.clip(Vm[:, 0], VLO, VHI), np.clip(Vm[:, 1], VLO, VHI))
Bd = np.column_stack([np.cos(th), np.sin(th)])
designs = {
    "direction only":       Bd,
    "speed only":           Bs,
    "direction + speed":    np.column_stack([Bd, Bs]),
    "velocity, linear":     Vm / SC,
    "velocity, 2D spline":  Bv2,
}
for k, X in designs.items():
    print("  %-22s %d features, NaNs %d" % (k, X.shape[1], np.isnan(X).sum()))


def poisson_dev(y, lam):
    """Poisson deviance per unit (y, lam in counts/bin)."""
    lam = np.maximum(lam, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / lam), 0.0)
    return 2 * (term - (y - lam)).sum(0)


def cv_pseudo_r2(X, Y, blk, n_fold=5):
    """5-fold cross-validated McFadden pseudo-R^2 per unit; folds are blocks of trials."""
    f = blk % n_fold
    pred = np.empty_like(Y)
    for k in range(n_fold):
        tr, te = f != k, f == k
        g = nmo.glm.PopulationGLM(observation_model="Poisson", regularizer="Ridge",
                                  regularizer_strength=1e-4, solver_name="LBFGS",
                                  solver_kwargs={"tol": 1e-10, "maxiter": 400})
        g.fit(X[tr], Y[tr])
        pred[te] = np.asarray(g.predict(X[te]))
    null = np.repeat(Y.mean(0, keepdims=True), len(Y), 0)
    return 1 - poisson_dev(Y, pred) / poisson_dev(Y, null), pred


res, preds = {}, {}
for k, X in tqdm(designs.items(), desc="GLMs", mininterval=5):
    res[k], preds[k] = cv_pseudo_r2(np.asarray(X, dtype=float), Y, blocks)
    print("  %-22s cross-validated pseudo-R2: median %.4f over tuned units (%.4f over all)"
          % (k, np.median(res[k][sig]), np.median(res[k])))

kv, kd = "velocity, 2D spline", "direction only"
win = (res[kv] > res[kd])[sig]
print("velocity beats direction-only in %d/%d tuned units (%.0f%%)" % (win.sum(), sig.sum(), 100 * win.mean()))
print("direction+speed (additive) beats direction only in %.0f%% of tuned units"
      % (100 * (res["direction + speed"] > res[kd])[sig].mean()))
print("NOTE: the linear velocity model fits worse under the exponential link, because it makes\n"
      "      rate grow exponentially rather than linearly with speed; the 2D spline is the\n"
      "      general velocity model and is the fair comparison against direction alone.")

np.savez("_res_glm.npz", **{("m_" + k.replace(" ", "_")): v for k, v in res.items()},
         names=np.array(list(res.keys())), lag=LAG)

# ---------------- figure 5 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw=dict(wspace=0.32))
names = list(res.keys())
ax = axes[0]
data = [res[k][sig] for k in names]
bp = ax.boxplot(data, tick_labels=[n.replace(" ", "\n") for n in names], showfliers=False, patch_artist=True)
for p in bp["boxes"]: p.set_facecolor("lightsteelblue")
ax.set(ylabel="cross-validated pseudo-$R^2$",
       title="Nested Poisson GLMs (NeMoS)\n%d directionally tuned units" % sig.sum())
ax.tick_params(axis="x", labelsize=8)
ax.text(0.02, 0.98, "the linear velocity model is handicapped by the\nexponential link (it makes rate grow exponentially\nwith speed); the 2D spline is the general version",
        transform=ax.transAxes, fontsize=6.5, va="top", color="0.35")

ax = axes[1]
ax.scatter(res[kd][sig], res[kv][sig], s=16, color="steelblue")
lim = [0, max(res[kd][sig].max(), res[kv][sig].max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1); ax.set(xlim=lim, ylim=lim)
ax.set(xlabel="pseudo-$R^2$, direction only", ylabel="pseudo-$R^2$, velocity (2D spline)",
       title="A general velocity model beats direction alone\n%.0f%% of units above the diagonal" % (100 * win.mean()))

ax = axes[2]
gain_frac = (res[kv][sig] - res[kd][sig]) / np.maximum(res[kv][sig], 1e-9)
ax.hist(100 * gain_frac, bins=np.linspace(-50, 100, 31), color="steelblue", edgecolor="w")
ax.axvline(0, color="k", lw=1)
ax.axvline(100 * np.median(gain_frac), color="crimson", lw=2)
ax.set(xlabel="gain from modelling speed\n(% of velocity-model pseudo-$R^2$)", ylabel="units",
       title="Median improvement %.0f%%" % (100 * np.median(gain_frac)))
fig.suptitle("Direction alone is not enough: firing rate encodes hand velocity (direction $\\times$ speed)", y=1.02, fontsize=13)
fig.savefig("fig05_glm_model_comparison.png", dpi=140, bbox_inches="tight")
print("saved fig05")
