"""Poisson GLM encoding models (NeMoS) and single-trial velocity decoding.

Three nested descriptions of what a unit encodes are compared on held-out
trials, all with the same 20 ms bins and the same neural lead time:

    direction   cos and sin of the instantaneous movement direction
    velocity    a linear function of (vx, vy), i.e. direction scaled by speed
    full 2D     a 6 x 6 B-spline surface over the (vx, vy) plane

Then the population is used in the other direction: hand velocity is
reconstructed from the spike counts of all units on trials the decoder has
never seen.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nemos as nmo
import pynapple as nap
from sklearn.linear_model import Ridge
from tqdm import tqdm

import analysis as an
import dandi_io as dio

WIN = (0.0, 0.5)
LEAD = 0.08          # s; population-optimal neural lead from script 03
SPEED_MIN = 50.0     # mm/s; bins slower than this carry no direction

sess = dio.load_session("maze_full")
spikes, pos, vel, support = dio.to_pynapple(sess)
tr = sess["trials"]
keep = an.quality_units(spikes, dio.trial_intervals(sess), min_rate=0.5)
uid = np.flatnonzero(keep)
spikes = spikes[list(uid)]

idx = np.flatnonzero(np.isfinite(tr["move_onset_time"]) & tr["success"].astype(bool))
ntr = len(idx)
ep = an.movement_epochs(sess, idx, WIN)
counts = spikes.count(an.BIN, ep=ep)
mask, V = an.sample_velocity(vel, counts, LEAD)
assert mask.all(), "every maze bin should have a velocity sample"
Y = np.asarray(counts.values, float)
per = Y.shape[0] // ntr
speed = np.linalg.norm(V, axis=1)
theta = np.arctan2(V[:, 1], V[:, 0])
print(f"{ntr} trials x {per} bins, {Y.shape[1]} units, "
      f"{(speed > SPEED_MIN).mean()*100:.0f}% of bins above {SPEED_MIN:.0f} mm/s")

# Held-out trials, not held-out bins: bins within a trial are correlated.
rng = np.random.default_rng(0)
trial_fold = rng.permutation(ntr) % 5
bin_fold = np.repeat(trial_fold, per)
mov = speed > SPEED_MIN

# ------------------------------------------------------------ feature sets
vscale = np.percentile(speed, 99)
Vn = V / vscale
basis2d = (nmo.basis.BSplineEval(n_basis_funcs=6, label="vx") *
           nmo.basis.BSplineEval(n_basis_funcs=6, label="vy"))
X_full = np.asarray(basis2d.compute_features(Vn[:, 0], Vn[:, 1]))
FEATURES = {
    "direction": np.column_stack([np.cos(theta), np.sin(theta)]),
    "velocity": Vn,
    "2D spline": X_full,
}
print("feature dimensionalities:", {k: x.shape[1] for k, x in FEATURES.items()})


def poisson_pseudo_r2(y, mu, y_train_mean):
    """McFadden pseudo-R^2 against a constant-rate null, per unit."""
    eps = 1e-9
    ll = y * np.log(mu + eps) - mu
    ll0 = y * np.log(y_train_mean + eps) - y_train_mean
    lls = y * np.log(y + eps) - y            # saturated model
    return 1 - (lls - ll).sum(0) / (lls - ll0).sum(0)


pseudo = {}
for name, X in FEATURES.items():
    pr2 = np.zeros((5, Y.shape[1]))
    for f in tqdm(range(5), desc=f"GLM {name}"):
        te = (bin_fold == f) & mov
        trn = (bin_fold != f) & mov
        model = nmo.glm.PopulationGLM(observation_model="Poisson",
                                      regularizer="Ridge",
                                      regularizer_strength=1e-3,
                                      solver_name="LBFGS")
        model.fit(X[trn], Y[trn])
        mu = np.asarray(model.predict(X[te]))
        pr2[f] = poisson_pseudo_r2(Y[te], mu, Y[trn].mean(0))
    pseudo[name] = pr2.mean(0)
    print(f"{name:10s} cross-validated pseudo-R2: median {np.median(pseudo[name]):.4f}, "
          f"90th pct {np.percentile(pseudo[name], 90):.4f}")

gain_speed = pseudo["velocity"] - pseudo["direction"]
gain_nonlin = pseudo["2D spline"] - pseudo["velocity"]
print(f"speed term helps {(gain_speed > 0).sum()}/{len(gain_speed)} units, "
      f"nonlinearity helps a further {(gain_nonlin > 0).sum()}/{len(gain_nonlin)}")

# --------------------------------------------------- population decoding
# Reconstruct hand velocity from the smoothed population rate on held-out
# trials.  This is the same tuning seen from the other side.
rate = an.smooth_rate(counts, sd=0.03)
dec_pred = np.empty_like(V)
for f in range(5):
    te = bin_fold == f
    reg = Ridge(alpha=1.0).fit(rate[~te], V[~te])
    dec_pred[te] = reg.predict(rate[te])
dec_r2 = 1 - ((V - dec_pred) ** 2).sum(0) / ((V - V.mean(0)) ** 2).sum(0)
dec_corr = [np.corrcoef(V[:, k], dec_pred[:, k])[0, 1] for k in range(2)]
print(f"single-trial decoding of held-out trials: R2 vx {dec_r2[0]:.3f}, "
      f"vy {dec_r2[1]:.3f}; r = {dec_corr[0]:.3f}, {dec_corr[1]:.3f}")

# how many units are needed?
sizes = [1, 2, 5, 10, 20, 40, 80, len(uid)]
curve = np.zeros((len(sizes), 5))
for i, n in enumerate(tqdm(sizes, desc="decoder size")):
    for rep in range(5):
        sub = np.random.default_rng(rep).choice(len(uid), n, replace=False)
        pr = np.empty_like(V)
        for f in range(5):
            te = bin_fold == f
            reg = Ridge(alpha=1.0).fit(rate[~te][:, sub], V[~te])
            pr[te] = reg.predict(rate[te][:, sub])
        curve[i, rep] = np.mean(1 - ((V - pr) ** 2).sum(0) / ((V - V.mean(0)) ** 2).sum(0))
print("decoding R2 vs population size:",
      dict(zip(sizes, curve.mean(1).round(3))))

np.savez("results_glm.npz", pseudo_dir=pseudo["direction"],
         pseudo_vel=pseudo["velocity"], pseudo_2d=pseudo["2D spline"],
         dec_r2=dec_r2, dec_pred=dec_pred, V=V, sizes=sizes, curve=curve, uid=uid)

# ---------------------------------------------------------------- figure 4
plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.45, top=0.87, bottom=0.08)

ax = fig.add_subplot(gs[0, 0])
data = [pseudo[k] for k in FEATURES]
parts = ax.violinplot(data, showmedians=True)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(list(FEATURES), fontsize=9)
ax.set(ylabel="cross-validated pseudo-$R^2$",
       title="Poisson GLM: what the spikes encode")
for i, d in enumerate(data):
    ax.text(i + 1, np.median(d), f" {np.median(d):.3f}", fontsize=8, va="bottom")

ax = fig.add_subplot(gs[0, 1])
ax.scatter(pseudo["direction"], pseudo["velocity"], s=14, c="steelblue",
           label="velocity vs direction")
ax.scatter(pseudo["direction"], pseudo["2D spline"], s=14, c="crimson",
           label="2D spline vs direction")
m = max(np.max(pseudo["2D spline"]), np.max(pseudo["direction"])) * 1.05
ax.plot([0, m], [0, m], "k:", lw=1)
ax.set(xlim=(0, m), ylim=(0, m), xlabel="direction-only pseudo-$R^2$",
       ylabel="richer model pseudo-$R^2$", title="Speed and curvature add fit")
ax.legend(fontsize=8, loc="upper left")

# GLM tuning surfaces predicted by the 2D spline model, for two example units
model = nmo.glm.PopulationGLM(observation_model="Poisson", regularizer="Ridge",
                              regularizer_strength=1e-3, solver_name="LBFGS")
model.fit(X_full[mov], Y[mov])
grid = np.linspace(-1, 1, 40)
gx, gy = np.meshgrid(grid, grid)
Xg = np.asarray(basis2d.compute_features(gx.ravel(), gy.ravel()))
mu_grid = np.asarray(model.predict(Xg)) / an.BIN
best = np.argsort(-pseudo["2D spline"])[:2]
for c, u in enumerate(best):
    ax = fig.add_subplot(gs[0, 2] if c == 0 else gs[1, 0])
    im = ax.pcolormesh(gx * vscale, gy * vscale, mu_grid[:, u].reshape(gx.shape),
                       cmap="magma", shading="auto")
    ax.set(xlabel="$v_x$ (mm/s)", ylabel="$v_y$ (mm/s)", aspect="equal",
           title=f"GLM rate surface, unit {uid[u]}\npseudo-$R^2$ = {pseudo['2D spline'][u]:.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("rate (Hz)", size=8)

ax = fig.add_subplot(gs[1, 1])
win_bins = np.arange(per * 6)          # six consecutive held-out trials
t_ms = np.arange(len(win_bins)) * an.BIN
for k, (lab, col) in enumerate([("$v_x$", "tab:blue"), ("$v_y$", "tab:orange")]):
    ax.plot(t_ms, V[win_bins, k], color=col, lw=1.5, label=f"{lab} measured")
    ax.plot(t_ms, dec_pred[win_bins, k], color=col, lw=1.2, ls="--",
            label=f"{lab} decoded")
for b in range(1, 6):
    ax.axvline(b * per * an.BIN, color="0.8", lw=0.8)
ax.legend(fontsize=7, ncol=2)
ax.set(xlabel="time (s), six consecutive trials", ylabel="velocity (mm/s)",
       title=f"Held-out decoding  ($R^2$ = {dec_r2[0]:.2f}, {dec_r2[1]:.2f})")

ax = fig.add_subplot(gs[1, 2])
ax.errorbar(sizes, curve.mean(1), yerr=curve.std(1), marker="o", color="crimson")
ax.set_xscale("log")
ax.set(xlabel="units in the decoder", ylabel="decoding $R^2$",
       title="Velocity is a distributed code")
ax.grid(alpha=0.3)

fig.suptitle("Encoding and decoding of reach velocity (MC_Maze, NeMoS Poisson GLMs, "
             f"{an.BIN*1000:.0f} ms bins, {LEAD*1000:.0f} ms neural lead)\n"
             "all scores are cross-validated on held-out trials",
             fontsize=12, y=0.965)
fig.savefig("fig04_glm_decoding.png", dpi=150, bbox_inches="tight")
print("wrote fig04_glm_decoding.png")
