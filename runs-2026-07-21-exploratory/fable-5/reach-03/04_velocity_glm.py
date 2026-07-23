"""Stage 4: Poisson GLMs (NeMoS) separating direction tuning from speed tuning."""

import jax
jax.config.update("jax_enable_x64", True)  # float32 leaves the Poisson fits short of tolerance

import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap
from tqdm import tqdm

import reachlib as rl

BIN = 0.02
SPEED_MIN = 50.0
N_FOLDS = 5

d = rl.load_mc_maze()
units, hand_vel, epochs, areas = d["units"], d["hand_vel"], d["epochs"], d["areas"]
vt = np.load("cache/velocity_tuning.npz")
LAG = float(vt["pop_lag"])

counts = units.count(BIN, ep=epochs)
tb = np.asarray(counts.t)
tq = tb + LAG
i = np.searchsorted(epochs.start, tq, side="right") - 1
inside = (i >= 0) & (tq <= epochs.end[np.clip(i, 0, None)])
vx = np.interp(tq, hand_vel.t, hand_vel.values[:, 0])
vy = np.interp(tq, hand_vel.t, hand_vel.values[:, 1])
speed = np.hypot(vx, vy)
keep = inside & (speed > SPEED_MIN)
keep_u = vt["keep_u"]          # units with enough spikes during movement
uidx = vt["uidx"]

theta = np.arctan2(vy[keep], vx[keep])
# Speed is converted to m/s so that the spline features are O(1); in mm/s the
# basis values are ~1e-3 and the penalised fit is badly conditioned.
sp = speed[keep] / 1000.0
Y = np.asarray(counts.values[keep][:, keep_u], dtype=float)
trial_id = np.searchsorted(epochs.start, tb[keep], side="right") - 1
print("GLM samples: %d bins, %d units, lag %.0f ms" % (Y.shape[0], Y.shape[1], 1000 * LAG))

dir_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, label="direction")
speed_basis = nmo.basis.BSplineEval(n_basis_funcs=5, label="speed")

designs = {
    "direction": (dir_basis, (theta,)),
    "speed": (speed_basis, (sp,)),
    "direction + speed": (dir_basis + speed_basis, (theta, sp)),
    "direction x speed": (dir_basis * speed_basis, (theta, sp)),
}
X = {name: np.asarray(b.compute_features(*args)) for name, (b, args) in designs.items()}
for k, v in X.items():
    print("  %-20s %d features" % (k, v.shape[1]))


def poisson_deviance(y, mu):
    """Per-column Poisson deviance, with the y*log(y/mu) term defined at y=0."""
    mu = np.clip(mu, 1e-9, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (term - (y - mu)).sum(0)


# Contiguous blocks of trials form the folds, so training and test bins never
# come from the same reach.
fold = np.floor(N_FOLDS * trial_id / (trial_id.max() + 1)).astype(int)
pr2 = {name: np.zeros((N_FOLDS, Y.shape[1])) for name in X}
models = {}
for name in X:
    for f in tqdm(range(N_FOLDS), desc="GLM %s" % name):
        tr, te = fold != f, fold == f
        glm = nmo.glm.PopulationGLM(
            regularizer="Ridge", regularizer_strength=1e-4, solver_name="LBFGS",
            solver_kwargs={"tol": 1e-8, "maxiter": 1500},
        )
        glm.fit(X[name][tr], Y[tr])
        mu = np.asarray(glm.predict(X[name][te]))
        null = Y[tr].mean(0)[None, :] * np.ones((te.sum(), 1))
        pr2[name][f] = 1 - poisson_deviance(Y[te], mu) / poisson_deviance(Y[te], null)
        if f == 0:
            models[name] = glm

pr2_mean = {k: v.mean(0) for k, v in pr2.items()}
print("\ncross-validated pseudo-R^2 (median over %d units)" % Y.shape[1])
for k, v in pr2_mean.items():
    print("  %-20s %.4f" % (k, np.median(v)))
gain = pr2_mean["direction x speed"] - pr2_mean["direction"]
print("direction x speed beats direction alone in %d/%d units (median gain %.4f)"
      % ((gain > 0).sum(), gain.size, np.median(gain)))
np.savez("cache/glm.npz", **{("pr2_" + k.replace(" ", "_")): v for k, v in pr2_mean.items()},
         uidx=uidx)

# --- Figure 7: model comparison and GLM tuning surfaces -------------------
# The surfaces are evaluated only over the well-sampled speed range; a tensor
# product basis extrapolates unreliably into the sparse tails.
grid_th = np.linspace(-np.pi, np.pi, 64)
grid_sp = np.linspace(np.percentile(sp, 2), np.percentile(sp, 95), 40)
TH, SP = np.meshgrid(grid_th, grid_sp, indexing="ij")
Xg = np.asarray((dir_basis * speed_basis).compute_features(TH.ravel(), SP.ravel()))
rate_grid = (np.asarray(models["direction x speed"].predict(Xg)) / BIN).reshape(
    grid_th.size, grid_sp.size, -1)

# Same example units as the empirical velocity fields, so the two can be compared.
beta_lin = vt["beta_lin"]
examples_glm = np.argsort(-np.hypot(beta_lin[1], beta_lin[2]))[:4]
Yn = Y.shape[1]

np.savez("cache/glm_surfaces.npz", rate_grid=rate_grid, grid_th=grid_th, grid_sp=grid_sp,
         examples=examples_glm, **{("pr2_" + k.replace(" ", "_")): v for k, v in pr2_mean.items()})

# Directional modulation depth of the fitted surfaces, as a function of speed.
depth_sp = rate_grid.max(0) - rate_grid.min(0)          # (n_speed, n_units)
depth_norm = depth_sp / depth_sp.max(0, keepdims=True)

fig = plt.figure(figsize=(14, 7.6))
gs = fig.add_gridspec(1, 4, top=0.855, bottom=0.635, wspace=0.42)
gs_p = fig.add_gridspec(1, 4, top=0.475, bottom=0.05, wspace=0.45)

ax = fig.add_subplot(gs[0])
order_m = ["speed", "direction", "direction + speed", "direction x speed"]
ax.boxplot([pr2_mean[k] for k in order_m],
           tick_labels=["speed", "dir", "dir+sp", "dir×sp"], showfliers=False)
ax.set_ylabel("cross-validated pseudo-$R^2$")
ax.set_title("nested model comparison\n(%d units)" % Yn, fontsize=10)

ax = fig.add_subplot(gs[1])
ax.scatter(pr2_mean["direction"], pr2_mean["direction x speed"], s=16, color="tab:blue")
lim = max(pr2_mean["direction x speed"].max(), pr2_mean["direction"].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=0.8)
ax.set_xlabel("pseudo-$R^2$, direction only")
ax.set_ylabel("pseudo-$R^2$, direction × speed")
ax.set_title("adding speed improves\n%d of %d units" % ((gain > 0).sum(), gain.size),
             fontsize=10)

ax = fig.add_subplot(gs[2])
ax.plot(1000 * grid_sp, np.median(depth_norm, 1), color="k", lw=2.5)
lo, hi = np.percentile(depth_norm, [25, 75], axis=1)
ax.fill_between(1000 * grid_sp, lo, hi, color="0.8")
ax.set_xlabel("speed (mm/s)")
ax.set_ylabel("directional depth / its peak")
pk = 1000 * grid_sp[np.median(depth_norm, 1).argmax()]
ax.set_title("direction tuning deepens with\nspeed, peaking near %d mm/s" % pk, fontsize=10)

ax = fig.add_subplot(gs[3])
u = examples_glm[0]
for j, s_val in enumerate(np.linspace(grid_sp[0], grid_sp[-1], 5)):
    k = np.argmin(np.abs(grid_sp - s_val))
    ax.plot(np.degrees(grid_th), rate_grid[:, k, u], lw=1.6,
            color=plt.get_cmap("plasma")(j / 4), label="%.0f" % (1000 * s_val))
ax.set_xlabel("direction (deg)")
ax.set_ylabel("GLM rate (Hz)")
ax.set_title("unit %d: tuning amplitude\ngrows, then saturates" % uidx[u], fontsize=10)
ax.legend(fontsize=7, title="speed (mm/s)", title_fontsize=7)

for col, u in enumerate(examples_glm):
    ax = fig.add_subplot(gs_p[col], projection="polar")
    th_e = np.linspace(-np.pi, np.pi, grid_th.size + 1)
    sp_e = np.linspace(grid_sp[0], grid_sp[-1], grid_sp.size + 1)
    T, S = np.meshgrid(th_e, sp_e, indexing="ij")
    pc = ax.pcolormesh(T, S, rate_grid[:, :, u], cmap="magma", shading="flat")
    ax.set_thetagrids([0, 90, 180, 270], labels=["0°", "", "180°", ""])
    ticks = np.round(np.linspace(grid_sp[0], grid_sp[-1], 4)[1:], 2)
    ax.set_rticks(ticks)
    ax.set_yticklabels(["%.0f" % (1000 * t) for t in ticks])
    ax.tick_params(labelsize=6)
    ax.set_rlabel_position(45)
    ax.set_title("unit %d\npseudo-$R^2$ = %.3f" % (uidx[u], pr2_mean["direction x speed"][u]),
                 fontsize=9, pad=16)
    plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.14, label="Hz" if col == 3 else "")

fig.text(0.5, 0.545, "GLM firing-rate surfaces for the same units shown in figure 5: "
                     "direction (angle) × speed (radius, mm/s)", ha="center", fontsize=11)
fig.suptitle("Poisson GLM: direction and speed are separable but interacting",
             fontsize=13, y=0.945)
fig.savefig("fig07_glm_direction_speed.png", dpi=150)
plt.close(fig)
