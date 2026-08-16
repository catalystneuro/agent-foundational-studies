"""Reach-direction tuning in motor cortex: cosine tuning curves and population statistics."""

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import mc_maze_io as mio
import reach_lib as rl

MOVE_WIN = (0.05, 0.35)   # movement epoch, relative to movement onset
PLAN_WIN = (-0.25, -0.05)  # late delay period, relative to movement onset

d = mio.load_mc_maze()
spikes, kin, trials = d["spikes"], d["kin"], d["trials"]

straight = trials[trials.trial_version == 0].reset_index(drop=True)
onset = straight["move_onset_time"].values

# reach direction measured from the hand itself (onset -> +400 ms)
p0 = nap.Ts(onset).value_from(kin)[["x", "y"]].values
p1 = nap.Ts(onset + 0.4).value_from(kin)[["x", "y"]].values
disp = p1 - p0
theta = np.arctan2(disp[:, 1], disp[:, 0])
dbin, centres = rl.direction_bins(theta, 8)
print("trials per direction bin:", np.bincount(dbin, minlength=8))

move_rates = rl.trial_rates(spikes, onset, MOVE_WIN)
plan_rates = rl.trial_rates(spikes, onset, PLAN_WIN)
base_rates = rl.trial_rates(spikes, straight["target_on_time"].values, (-0.25, -0.05))
print("move_rates", move_rates.shape, "mean", move_rates.mean())

fit_move = rl.cosine_fit(move_rates, theta)
fit_plan = rl.cosine_fit(plan_rates, theta)
perm_p, depth = rl.permutation_p(move_rates, theta, n_perm=500)
perm_p_plan, _ = rl.permutation_p(plan_rates, theta, n_perm=500)

# cosine fit on the 8 condition means as well: this is the classical
# Georgopoulos measure and is not limited by single-trial spiking noise
occupied = np.array([b for b in range(8) if (dbin == b).sum() > 5])
cond_mu = np.stack([move_rates[:, dbin == b].mean(1) for b in occupied], axis=1)
fit_cond = rl.cosine_fit(cond_mu, centres[occupied])

n = len(spikes)
sig = perm_p < 0.01
print(f"units with significant direction tuning during movement: {sig.sum()}/{n} "
      f"({100*sig.mean():.0f}%)")
print(f"units with significant direction tuning during the delay: "
      f"{(perm_p_plan < 0.01).sum()}/{n}")
print(f"median single-trial cosine R2 (movement) = {np.median(fit_move['r2']):.3f}")
print(f"median condition-mean cosine R2 = {np.median(fit_cond['r2'][sig]):.3f} (tuned units)")
print(f"median modulation depth = {np.median(fit_move['depth']):.1f} Hz, "
      f"tuned units {np.median(fit_move['depth'][sig]):.1f} Hz")

np.savez(
    "results_direction.npz",
    theta=theta, dbin=dbin, centres=centres, onset=onset,
    move_rates=move_rates, plan_rates=plan_rates, base_rates=base_rates,
    pd_move=fit_move["pd"], depth_move=fit_move["depth"], r2_move=fit_move["r2"],
    b0_move=fit_move["b0"], perm_p=perm_p,
    pd_plan=fit_plan["pd"], depth_plan=fit_plan["depth"], r2_plan=fit_plan["r2"],
    perm_p_plan=perm_p_plan,
    r2_cond=fit_cond['r2'], occupied=occupied, cond_mu=cond_mu,
)

# ---------------------------------------------------------------- example units
rate_psth, tpsth = rl.psth(spikes, onset, (-0.4, 0.6), bin_size=0.01, sigma_bins=2.0)
ex = np.argsort(-fit_move["depth"] * (fit_cond["r2"] > 0.7))[:3]
print("example units", ex, "depths", fit_move["depth"][ex])

cols = plt.get_cmap("hsv")(np.linspace(0, 1, 9)[:8])
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5),
                         subplot_kw=dict(), gridspec_kw=dict(hspace=0.5, wspace=0.35))
for j, u in enumerate(ex):
    ax = axes[0, j]
    for b in occupied:
        m = rate_psth[u][dbin == b].mean(0)
        ax.plot(tpsth, m, color=cols[b], lw=1.4,
                label=f"{int(np.degrees(centres[b]))}$\\degree$")
    ax.axvline(0, ls="--", c="k", lw=0.8)
    ax.set(xlabel="time from movement onset (s)", title=f"unit {u}")
    if j == 0:
        ax.set_ylabel("firing rate (Hz)")
    if j == 2:
        ax.legend(title="reach direction", fontsize=7, title_fontsize=7,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    axp = fig.add_subplot(2, 3, 4 + j, projection="polar")
    axes[1, j].remove()
    mu = np.array([move_rates[u][dbin == b].mean() for b in occupied])
    se = np.array([move_rates[u][dbin == b].std() / np.sqrt((dbin == b).sum())
                   for b in occupied])
    th = np.append(centres[occupied], centres[occupied][0])
    axp.errorbar(th, np.append(mu, mu[0]), yerr=np.append(se, se[0]),
                 color="k", lw=1.5, marker="o", ms=4)
    grid = np.linspace(0, 2 * np.pi, 200)
    axp.plot(grid, fit_move["b0"][u] + fit_move["depth"][u] * np.cos(grid - fit_move["pd"][u]),
             color="crimson", lw=1.5)
    axp.set_title(f"PD = {np.degrees(fit_move['pd'][u]) % 360:.0f}$\\degree$,  "
                  f"$R^2_{{cond}}$ = {fit_cond['r2'][u]:.2f}", pad=30, fontsize=10)
    axp.tick_params(labelsize=7)
    axp.set_rlabel_position(157)

fig.suptitle("Directional tuning of single motor-cortex units (MC_Maze, straight reaches)\n"
             "top: PSTHs by reach direction (colour = direction); "
             "bottom: mean movement-epoch rate (black) with cosine fit (red)", fontsize=12)
fig.savefig("fig02_example_direction_tuning.png", dpi=140, bbox_inches="tight")
print("saved fig02_example_direction_tuning.png")

# ---------------------------------------------------------------- population
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.42, top=0.86)

ax = fig.add_subplot(gs[0, 0], projection="polar")
ax.hist(fit_move["pd"][sig], bins=np.linspace(-np.pi, np.pi, 25), color="steelblue")
ax.set_title(f"Preferred directions\n(n = {sig.sum()} tuned units)", pad=30, fontsize=10)
ax.set_rlabel_position(157)
ax.tick_params(labelsize=7)

ax = fig.add_subplot(gs[0, 1])
ax.hist(fit_move["depth"], bins=30, color="steelblue", label="movement")
ax.hist(fit_plan["depth"], bins=30, color="darkorange", alpha=0.6, label="delay")
ax.set(xlabel="cosine modulation depth (Hz)", ylabel="units",
       title="Depth of directional modulation")
ax.legend()

ax = fig.add_subplot(gs[0, 2])
ax.scatter(fit_plan["depth"], fit_move["depth"], s=12, c="k", alpha=0.6)
lim = max(fit_move["depth"].max(), fit_plan["depth"].max()) * 1.05
ax.plot([0, lim], [0, lim], "r--", lw=0.8)
ax.set(xlabel="delay-period depth (Hz)", ylabel="movement depth (Hz)",
       title="Directional tuning is already\npresent before movement onset")

ax = fig.add_subplot(gs[1, 0])
ax.hist(fit_cond["r2"][sig], bins=30, color="steelblue")
ax.axvline(np.median(fit_cond["r2"][sig]), color="r", ls="--")
ax.set(xlabel="$R^2$ of cosine fit to condition means", ylabel="units",
       title=f"Cosine model captures directional tuning\n"
             f"median $R^2$ = {np.median(fit_cond['r2'][sig]):.2f} (tuned units)")

ax = fig.add_subplot(gs[1, 1])
ax.hist(perm_p, bins=np.linspace(0, 1, 41), color="steelblue")
ax.axvline(0.01, color="r", ls="--")
ax.set(xlabel="permutation p (direction shuffled)", ylabel="units",
       title=f"{100*sig.mean():.0f}% of units significantly\ndirection tuned (p < 0.01)")

# time-resolved strength of directional tuning
ax = fig.add_subplot(gs[1, 2])
win_t = tpsth
depth_t = np.empty((len(spikes), len(win_t)))
for k in range(len(win_t)):
    depth_t[:, k] = rl.cosine_fit(rate_psth[:, :, k], theta)["depth"]
m = depth_t[sig].mean(0)
se = depth_t[sig].std(0) / np.sqrt(sig.sum())
ax.plot(win_t, m, color="crimson", lw=1.8, label="directional modulation depth")
ax.fill_between(win_t, m - se, m + se, color="crimson", alpha=0.25)
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.set(xlabel="time from movement onset (s)", ylabel="mean depth (Hz)",
       title="Directional tuning ramps up before\nmovement and peaks near peak speed")
ax2 = ax.twinx()
speed = nap.Tsd(t=kin.index, d=np.hypot(kin["vx"].values, kin["vy"].values))
sp_t = np.nanmean(nap.build_tensor(speed, nap.IntervalSet(onset - 0.4, onset + 0.6)), 0)
ax2.plot(np.arange(len(sp_t)) / 1000 - 0.4, sp_t, color="gray", lw=1.2, label="hand speed")
ax2.set_ylabel("hand speed (mm/s)", color="gray")
ax2.tick_params(axis="y", colors="gray")
lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
np.save("results_depth_time.npy", depth_t)

fig.suptitle("Population statistics of reach-direction tuning (182 units, 789 straight reaches)",
             fontsize=13, y=0.99)
fig.savefig("fig03_population_direction_tuning.png", dpi=140, bbox_inches="tight")
print("saved fig03_population_direction_tuning.png")
