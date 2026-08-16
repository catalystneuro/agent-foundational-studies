"""Prototype NeMoS GLM: Poisson encoding model of HD tuning (Mouse17-130128)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import nemos as nmo

from hd_utils import (
    SESSIONS, load_session, compute_head_direction, get_wake_epochs,
    tuning_curves_hd, classify_hd_cells,
)

name = "Mouse17-130128"
nwb, io = load_session(SESSIONS[name])
units = nwb["units"]
hd = compute_head_direction(nwb)
wake = get_wake_epochs(nwb, "Awake")

res = np.load("cache/prototype_results.npz", allow_pickle=True)
keys = list(res["keys"])
is_hd = res["is_hd"]
hd_keys = [k for k, h in zip(keys, is_hd) if h]
print("HD cells:", hd_keys)

rates_emp, centers, occupancy = tuning_curves_hd(units, hd, wake, bins=60)

# ------------------------------------------------------------- design matrix
bin_size = 0.05  # s
hd_group = units[hd_keys]
count = hd_group.count(bin_size, ep=wake)
print("count shape:", count.shape, "bins:", count.shape[0])

# circular-safe interpolation of HD onto count bin times via sin/cos
valid = ~np.isnan(hd.values)
t_v, a_v = hd.t[valid], hd.values[valid]
ct = count.t
cos_i = np.interp(ct, t_v, np.cos(a_v))
sin_i = np.interp(ct, t_v, np.sin(a_v))
hd_binned = np.arctan2(sin_i, cos_i) % (2 * np.pi)
hd_tsd = nap.Tsd(t=ct, d=hd_binned, time_support=count.time_support)

basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, order=3, bounds=(0, 2 * np.pi))
X = basis.compute_features(hd_tsd)
print("X shape:", X.shape)

model = nmo.glm.PopulationGLM(
    observation_model="Poisson",
    regularizer="Ridge",
    regularizer_strength=1e-5,
    solver_name="LBFGS",
    solver_kwargs={"tol": 1e-12, "maxiter": 5000},
)
model.fit(X, count)
print("coef shape:", model.coef_.shape, "intercept:", model.intercept_.shape)
score = model.score(X, count)
print("score (mean pseudo-R2 or ll):", score)

# GLM-implied tuning curves: predict on an HD grid
grid = np.linspace(0, 2 * np.pi, 60, endpoint=False)
Xg = basis.compute_features(grid)
rate_grid = model.predict(Xg) / bin_size  # Hz

# predicted rate in time for a snippet
pred_rate = model.predict(X) / bin_size

# ------------------------------------------------------------- figure
n_hd = len(hd_keys)
fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
for i, k in enumerate(hd_keys):
    ax = axes.flat[i]
    emp = rates_emp.sel(unit=k).values
    ax.plot(np.rad2deg(centers), emp, "k-", lw=1.5, label="empirical")
    ax.plot(np.rad2deg(grid), rate_grid[:, i], "r-", lw=1.5, label="GLM")
    ax.set_title(f"unit {k}", fontsize=10)
    ax.set_xlabel("HD (deg)")
    if i % 4 == 0:
        ax.set_ylabel("Rate (Hz)")
    if i == 0:
        ax.legend(fontsize=8)
axes.flat[n_hd].axis("off")
# snippet of predicted vs smoothed actual rate
ax = axes.flat[-1]
t0, t1 = 17900.0, 17930.0
sl = (ct >= t0) & (ct <= t1)
k = hd_keys[1]
i = hd_keys.index(k)
actual = count.values[sl, i] / bin_size
# smooth actual with 0.5 s boxcar
kern = np.ones(10) / 10
ax.plot(ct[sl], np.convolve(actual, kern, mode="same"), "k-", lw=1, label="actual (smoothed)")
ax.plot(ct[sl], pred_rate[sl, i], "r-", lw=1, label="GLM prediction")
ax.set_title(f"unit {k}: rate snippet", fontsize=10)
ax.set_xlabel("Time (s)")
ax.legend(fontsize=8)
fig.suptitle(f"{name}: Poisson GLM with cyclic B-spline HD basis")
fig.savefig("figures/05_glm_tuning.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/05_glm_tuning.png")
io.close()
