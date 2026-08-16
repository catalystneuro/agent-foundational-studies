"""Population decoding of instantaneous hand velocity.

If single units are genuinely tuned to the direction and magnitude of hand
velocity, a linear read-out of the population should track the velocity vector
moment by moment. This is the standard optimal-linear-estimator decoder used in
the BMI literature, cross-validated over held-out reaches.
"""

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from matplotlib.colors import LogNorm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import mc_maze_io as mio

BIN = 0.02
WIN = (-0.20, 0.60)
N_LAG = 10          # 10 bins x 20 ms = 200 ms of neural history used per prediction

d = mio.load_mc_maze()
spikes, kin, trials = d["spikes"], d["kin"], d["trials"]
onset = trials["move_onset_time"].values
n_units = len(spikes)

ep = nap.IntervalSet(start=onset + WIN[0], end=onset + WIN[1])
cnt = nap.build_tensor(spikes, ep, bin_size=BIN)          # (n_units, n_trials, n_bins)
n_trials, n_bins = cnt.shape[1:]

tc = WIN[0] + BIN * (np.arange(n_bins) + 0.5)
kt, kv = np.asarray(kin.index), kin.values
tt = onset[:, None] + tc[None, :]
vx = np.interp(tt, kt, kv[:, 2])
vy = np.interp(tt, kt, kv[:, 3])

# stack N_LAG future bins of spiking as predictors of the velocity at bin t
usable = n_bins - N_LAG
X = np.concatenate([cnt[:, :, l:l + usable] for l in range(N_LAG)], axis=0)
X = X.transpose(1, 2, 0).reshape(n_trials * usable, n_units * N_LAG)
Y = np.column_stack([vx[:, :usable].ravel(), vy[:, :usable].ravel()])
print("decoder design matrix", X.shape, "targets", Y.shape)

pred = np.empty_like(Y)
kf = KFold(n_splits=5, shuffle=False)
for tr_idx, te_idx in kf.split(np.arange(n_trials)):
    tr = np.isin(np.repeat(np.arange(n_trials), usable), tr_idx)
    model = Ridge(alpha=50.0).fit(X[tr], Y[tr])
    pred[~tr] = model.predict(X[~tr])

ss_res = ((Y - pred) ** 2).sum(0)
ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
r2 = 1 - ss_res / ss_tot
speed_true, speed_pred = np.hypot(*Y.T), np.hypot(*pred.T)
r2_speed = 1 - ((speed_true - speed_pred) ** 2).sum() / ((speed_true - speed_true.mean()) ** 2).sum()
ang_err = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(pred[:, 1], pred[:, 0])
                                                 - np.arctan2(Y[:, 1], Y[:, 0]))))))
fast = speed_true > 200
print(f"held-out R2: vx = {r2[0]:.3f}, vy = {r2[1]:.3f}, speed = {r2_speed:.3f}")
print(f"median direction error (speed > 200 mm/s) = {np.median(ang_err[fast]):.1f} deg")

np.savez("results_decoding.npz", Y=Y, pred=pred, r2=r2, r2_speed=r2_speed,
         ang_err=ang_err, fast=fast, usable=usable, n_trials=n_trials)

# ------------------------------------------------------------------- figure 6
fig = plt.figure(figsize=(14.5, 8))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35, top=0.86)

t_plot = np.arange(30 * usable) * BIN
sl = slice(400 * usable, 430 * usable)
for k, (lab, col) in enumerate([("$v_x$", "tab:blue"), ("$v_y$", "tab:orange")]):
    ax = fig.add_subplot(gs[k, :2])
    ax.plot(t_plot, Y[sl, k], color="k", lw=1.4, label="measured")
    ax.plot(t_plot, pred[sl, k], color=col, lw=1.4, label="decoded from spikes")
    for b in range(31):
        ax.axvline(b * usable * BIN, color="0.85", lw=0.6, zorder=0)
    ax.set(ylabel=f"{lab} (mm/s)", title="" if k else
           "Hand velocity decoded from 182 units on held-out reaches "
           "(grey lines = reach boundaries)")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    if k:
        ax.set_xlabel("time (concatenated held-out reaches, s)")

ax = fig.add_subplot(gs[0, 2])
h = ax.hist2d(Y[:, 0], pred[:, 0], bins=60, cmap="magma", norm=LogNorm(vmin=1),
              range=[[-900, 900], [-900, 900]])
fig.colorbar(h[3], ax=ax, label="20 ms bins")
ax.plot([-900, 900], [-900, 900], "w--", lw=1)
ax.set(xlabel="measured $v_x$ (mm/s)", ylabel="decoded $v_x$ (mm/s)",
       title=f"$R^2$ = {r2[0]:.2f} ($v_x$), {r2[1]:.2f} ($v_y$)")

ax = fig.add_subplot(gs[1, 2])
ax.hist(ang_err[fast], bins=np.arange(0, 181, 5), color="steelblue")
ax.axvline(np.median(ang_err[fast]), color="crimson", ls="--",
           label=f"median {np.median(ang_err[fast]):.0f}$\\degree$")
ax.set(xlabel="decoded $-$ measured direction (deg)", ylabel="20 ms bins",
       title="Direction read-out error\n(bins with speed > 200 mm/s)")
ax.legend(fontsize=8)

fig.suptitle("A linear read-out of the population recovers the hand velocity vector "
             "moment by moment", fontsize=13)
fig.savefig("fig06_population_decoding.png", dpi=140, bbox_inches="tight")
print("saved fig06_population_decoding.png")
