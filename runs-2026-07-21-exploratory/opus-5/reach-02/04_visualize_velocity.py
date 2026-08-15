"""Figures for the continuous velocity-tuning analysis (reads results_velocity.npz)."""

import matplotlib.pyplot as plt
import numpy as np

R = np.load("results_velocity.npz", allow_pickle=True)
D = np.load("results_direction.npz")

g_dir, g_spd, surf, emp = R["g_dir"], R["g_spd"], R["surf"], R["emp"]
sedges, occ = R["sedges"], R["occ"]
LAGS, r2_by_lag, best_lag, opt_lag = R["LAGS"], R["r2_by_lag"], R["best_lag"], R["opt_lag"]
r2_by_lag_all, opt_lag_all = R["r2_by_lag_all"], R["opt_lag_all"]
pd_glm, pref_idx, gain, vel_signal = R["pd_glm"], R["pref_idx"], R["gain"], R["vel_signal"]
n_units, n_g = surf.shape[2], surf.shape[0]

ORDER = ["speed only", "direction only", "linear velocity ($v_x, v_y$)",
         "direction + speed", "direction $\\times$ speed"]
cv = {k: R[f"cv_{k}"] for k in ORDER}
cv_dir, cv_full = cv["direction only"], cv["direction $\\times$ speed"]

active = D["move_rates"].mean(1) > 0.5   # Hz; the rest are effectively silent
tuned = active & (D["perm_p"] < 0.01)    # direction tuned in the trial-level analysis
has_vel = active & (vel_signal > 0.02)   # has instantaneous velocity signal at all
keep = tuned & has_vel
# Example units for display: well fit by the velocity GLM, but also fast enough that
# the measured rate map is not mostly empty (pseudo-R^2 alone favours sparse units).
bright = tuned & (D["move_rates"].mean(1) > 10.0)
ex = np.argsort(-(cv_full * bright))[:3]
dir_edges = np.linspace(-180, 180, emp.shape[1] + 1)

# ------------------------------------------------------------------ figure 4
fig = plt.figure(figsize=(14.5, 9.5))
gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.38, top=0.86, bottom=0.07)

for j, u in enumerate(ex):
    ax = fig.add_subplot(gs[0, j])
    im = ax.pcolormesh(dir_edges, sedges, emp[u].T, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Hz" if j == 2 else None)
    ax.set(xlabel="hand direction (deg)", title=f"unit {u} — measured",
           xticks=[-180, -90, 0, 90, 180])
    if j == 0:
        ax.set_ylabel("hand speed (mm/s)")

    ax = fig.add_subplot(gs[1, j])
    im = ax.pcolormesh(np.degrees(g_dir), g_spd, surf[:, :, u].T, cmap="viridis",
                       shading="gouraud")
    fig.colorbar(im, ax=ax, label="Hz" if j == 2 else None)
    ax.set(xlabel="hand direction (deg)", title=f"unit {u} — Poisson GLM",
           xticks=[-180, -90, 0, 90, 180])
    if j == 0:
        ax.set_ylabel("hand speed (mm/s)")

ax = fig.add_subplot(gs[2, 0])
ax.plot(1000 * LAGS, r2_by_lag[has_vel].T, color="k", alpha=0.10, lw=0.6)
ax.plot(1000 * LAGS, r2_by_lag.mean(0), color="crimson", lw=2.4, label="moving bins")
ax.plot(1000 * LAGS, r2_by_lag_all.mean(0), color="darkorange", lw=2.0, ls="--",
        label="all bins (incl. delay)")
ax.axvline(1000 * opt_lag, color="crimson", ls=":", lw=1)
ax.axvline(1000 * opt_lag_all, color="darkorange", ls=":", lw=1)
ax.set(xlabel="neural lead (ms)", ylabel="$R^2$ (linear velocity model)",
       title=f"Spiking leads the hand by {1000*opt_lag:.0f} ms\n"
             f"({1000*opt_lag_all:.0f} ms if the delay period is included)")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 1])
ax.hist(1000 * best_lag[has_vel], bins=np.arange(-210, 311, 20), color="steelblue")
ax.axvline(1000 * np.median(best_lag[has_vel]), color="crimson", ls="--",
           label=f"median {1000*np.median(best_lag[has_vel]):.0f} ms")
ax.set(xlabel="best lead per unit (ms)", ylabel="units",
       title=f"Single-unit optimal lead\n({has_vel.sum()} units with $R^2$ > 0.02)")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 2])
anti_idx = (pref_idx + n_g // 2) % n_g
norm = np.maximum(surf.max(axis=(0, 1)), 1e-6)[:, None]
pref = np.stack([surf[pref_idx[u], :, u] for u in range(n_units)]) / norm
anti = np.stack([surf[anti_idx[u], :, u] for u in range(n_units)]) / norm
for arr, col, lab in [(pref, "crimson", "preferred direction"),
                      (anti, "steelblue", "anti-preferred")]:
    m = arr[has_vel].mean(0)
    se = arr[has_vel].std(0) / np.sqrt(has_vel.sum())
    ax.plot(g_spd, m, color=col, lw=2.2, label=lab)
    ax.fill_between(g_spd, m - se, m + se, color=col, alpha=0.25)
ax.set(xlabel="hand speed (mm/s)", ylabel="firing rate (norm. to peak)",
       title="Rate grows with speed only in\nthe preferred direction")
ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Instantaneous velocity tuning in motor cortex (DANDI 000128, MC_Maze)\n"
             "top two rows: firing rate over the (direction, speed) plane, measured and "
             "GLM-predicted, for three example units", fontsize=12.5)
fig.savefig("fig04_velocity_tuning.png", dpi=140, bbox_inches="tight")
print("saved fig04_velocity_tuning.png")

# ------------------------------------------------------------------ figure 5
fig = plt.figure(figsize=(14.5, 8))
gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.36, top=0.86)

ax = fig.add_subplot(gs[0, 0])
med = [np.median(cv[k][active]) for k in ORDER]
bars = ax.bar(range(len(ORDER)), med,
              color=["0.6", "steelblue", "seagreen", "goldenrod", "crimson"])
for b, m in zip(bars, med):
    ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}", ha="center", va="bottom",
            fontsize=8)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, rotation=35, ha="right", fontsize=8)
ax.set(ylabel="median CV pseudo-$R^2$",
       title="Poisson GLM model comparison\n(5-fold CV, held-out reaches)")

ax = fig.add_subplot(gs[0, 1])
ax.scatter(cv_dir[active], cv_full[active], s=14, c="k", alpha=0.55)
lim = [min(cv_dir[active].min(), 0),
       max(cv_dir[active].max(), cv_full[active].max()) * 1.05]
ax.plot(lim, lim, "r--", lw=0.9)
frac = (cv_full[tuned] > cv_dir[tuned]).mean()
ax.set(xlabel="direction only", ylabel="direction $\\times$ speed", xlim=lim, ylim=lim,
       title=f"Adding speed helps {100*frac:.0f}% of\ndirection-tuned units")

ax = fig.add_subplot(gs[0, 2])
bins_g = np.linspace(-1.5, 3, 40)
ax.hist(gain[has_vel, 0], bins=bins_g, color="crimson", alpha=0.65,
        label="preferred direction")
ax.hist(gain[has_vel, 1], bins=bins_g, color="steelblue", alpha=0.65,
        label="anti-preferred")
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.set(xlabel="speed gain (Hz per 100 mm/s)", ylabel="units",
       title=f"Speed gain: median {np.median(gain[has_vel,0]):+.2f} vs "
             f"{np.median(gain[has_vel,1]):+.2f} Hz/100 mm/s")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 0])
delta = np.angle(np.exp(1j * (pd_glm - D["pd_move"])))
circ_r = np.abs(np.exp(1j * delta[keep]).mean())
ax.scatter(np.degrees(D["pd_move"][active]) % 360, np.degrees(pd_glm[active]) % 360,
           s=10, c="0.75", label="all active units")
ax.scatter(np.degrees(D["pd_move"][keep]) % 360, np.degrees(pd_glm[keep]) % 360,
           s=16, c="k", alpha=0.75, label=f"n = {keep.sum()} with velocity signal")
ax.plot([0, 360], [0, 360], "r--", lw=0.9)
ax.set(xlabel="PD from trial-level cosine fit (deg)", ylabel="PD from velocity GLM (deg)",
       xticks=[0, 90, 180, 270, 360], yticks=[0, 90, 180, 270, 360],
       title=f"Preferred directions agree\n(circular $r$ = {circ_r:.2f})")
ax.legend(fontsize=7, loc="lower right")

ax = fig.add_subplot(gs[1, 1])
ax.hist(np.degrees(delta[keep]), bins=np.arange(-180, 181, 15), color="steelblue")
ax.set(xlabel="PD(velocity GLM) $-$ PD(trial cosine) (deg)", ylabel="units",
       xticks=[-180, -90, 0, 90, 180],
       title=f"median |difference| = "
             f"{np.median(np.abs(np.degrees(delta[keep]))):.0f}$\\degree$")

ax = fig.add_subplot(gs[1, 2])
im = ax.pcolormesh(dir_edges, sedges, occ.T, cmap="magma")
fig.colorbar(im, ax=ax, label="20 ms bins")
ax.set(xlabel="hand direction (deg)", ylabel="hand speed (mm/s)",
       xticks=[-180, -90, 0, 90, 180],
       title="Sampling of the velocity plane\n(all directions and speeds visited)")

fig.suptitle("Direction and speed are separate, jointly necessary determinants of "
             "motor-cortex firing", fontsize=13)
fig.savefig("fig05_velocity_model_comparison.png", dpi=140, bbox_inches="tight")
print("saved fig05_velocity_model_comparison.png")
