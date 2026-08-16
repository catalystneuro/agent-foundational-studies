"""Direction tuning of motor cortical units during barrier-free center-out reaches."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from scipy import stats

import analysis as an
import dandi_io as dio

sess = dio.load_session("maze_full")
spikes, pos, vel, support = dio.to_pynapple(sess)
tr = sess["trials"]

idx = an.straight_trials(sess)
theta, dist = an.reach_direction(sess, idx)
on = tr["move_onset_time"][idx]
print(f"{len(idx)} straight reaches, displacement {dist.mean():.0f} +- {dist.std():.0f} mm")

keep = an.quality_units(spikes, dio.trial_intervals(sess), min_rate=0.5)
uid = np.flatnonzero(keep)
print(f"{keep.sum()} of {len(spikes)} units pass the 0.5 Hz rate criterion")

# movement epoch rates
rates_mov, _ = an.epoch_rates(spikes, on + an.MOVE_WIN[0], on + an.MOVE_WIN[1])
rates_mov = rates_mov[:, keep]
fit_mov = an.fit_cosine_tuning(rates_mov, theta)

# planning epoch: 150 ms after target onset to the go cue, long-delay trials only
delay = tr["go_cue_time"][idx] - tr["target_on_time"][idx]
plan_ok = delay > 0.4
rates_plan, _ = an.epoch_rates(spikes, tr["target_on_time"][idx][plan_ok] + 0.15,
                               tr["go_cue_time"][idx][plan_ok])
rates_plan = rates_plan[:, keep]
fit_plan = an.fit_cosine_tuning(rates_plan, theta[plan_ok])

sig_mov = fit_mov["p_perm"] < 0.01
sig_plan = fit_plan["p_perm"] < 0.01
print(f"movement epoch: {sig_mov.sum()}/{len(sig_mov)} units direction tuned (perm p<0.01), "
      f"median R2 = {np.nanmedian(fit_mov['r2'][sig_mov]):.3f}, "
      f"max R2 = {np.nanmax(fit_mov['r2']):.3f}")
print(f"planning epoch ({plan_ok.sum()} trials): {sig_plan.sum()}/{len(sig_plan)} tuned, "
      f"median R2 = {np.nanmedian(fit_plan['r2'][sig_plan]):.3f}")
pd_sd = an.bootstrap_pd(rates_mov, theta)
print(f"bootstrap PD s.d. (tuned units): median {np.degrees(np.median(pd_sd[sig_mov])):.1f} deg")

# Is the population of preferred directions uniform around the circle?
z = np.exp(1j * fit_mov["pd"][sig_mov])
R = np.abs(z.mean())
rayleigh_p = np.exp(np.sqrt(1 + 4 * sig_mov.sum() + 4 * (sig_mov.sum() ** 2) *
                            (1 - R ** 2)) - (1 + 2 * sig_mov.sum()))
print(f"PD distribution: resultant length R = {R:.3f}, Rayleigh p = {rayleigh_p:.3g}")

# Do preferred directions carry over from planning to execution?
both = sig_mov & sig_plan
dpd = np.angle(np.exp(1j * (fit_plan["pd"] - fit_mov["pd"])))[both]
Rd = np.abs(np.exp(1j * dpd).mean())
nb = both.sum()
p_d = np.exp(np.sqrt(1 + 4 * nb + 4 * (nb ** 2) * (1 - Rd ** 2)) - (1 + 2 * nb))
print(f"planning vs movement PD difference: n={nb}, mean {np.degrees(np.angle(np.exp(1j*dpd).mean())):.0f} deg, "
      f"R = {Rd:.3f}, Rayleigh p = {p_d:.3g}, "
      f"{(np.abs(np.degrees(dpd)) < 45).mean()*100:.0f}% within 45 deg")

# Split-half control: how reproducible is a movement-epoch PD within the same
# epoch?  If odd/even halves agree far better than planning does with movement,
# the planning-to-movement change cannot be attributed to estimation noise.
half = np.arange(len(theta)) % 2 == 0
fa = an.fit_cosine_tuning(rates_mov[half], theta[half], n_perm=1)
fb = an.fit_cosine_tuning(rates_mov[~half], theta[~half], n_perm=1)
dpd_split = np.angle(np.exp(1j * (fa["pd"] - fb["pd"])))[sig_mov]
Rs = np.abs(np.exp(1j * dpd_split).mean())
print(f"split-half movement PD: R = {Rs:.3f}, "
      f"{(np.abs(np.degrees(dpd_split)) < 45).mean()*100:.0f}% within 45 deg")

np.savez("results_direction.npz", idx=idx, theta=theta, rates_mov=rates_mov, uid=uid,
         dpd=dpd, dpd_split=dpd_split,
         pd=fit_mov["pd"], md=fit_mov["md"], b0=fit_mov["b0"], r2=fit_mov["r2"],
         p_perm=fit_mov["p_perm"], pd_sd=pd_sd,
         pd_plan=fit_plan["pd"], md_plan=fit_plan["md"], r2_plan=fit_plan["r2"],
         p_perm_plan=fit_plan["p_perm"], plan_ok=plan_ok)

# ---------------------------------------------------------------- figure 2
mu, se, centres = an.tuning_curve_by_sector(rates_mov, theta, 8)
examples = np.argsort(-fit_mov["md"] * sig_mov)[:4]

plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 4, hspace=0.70, wspace=0.42, top=0.88, bottom=0.06)

bins = np.arange(-0.4, 0.65, 0.02)
dbin = an.direction_bins(theta, 8)
cmap = plt.get_cmap("hsv")
for c, u in enumerate(examples):
    ax = fig.add_subplot(gs[0, c])
    st = spikes[uid[u]].index.values
    for k in range(8):
        tt = on[dbin == k]
        if len(tt) < 5:
            continue
        h, _ = np.histogram((st[None, :] - tt[:, None]).ravel(), bins)
        ax.plot(bins[:-1] + 0.01, h / len(tt) / 0.02, color=cmap(k / 8), lw=1.2,
                label=f"{k*45}$\\degree$")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_title(f"unit {uid[u]}", fontsize=10)
    ax.set_xlabel("time from onset (s)", fontsize=8)
    if c == 0:
        ax.set_ylabel("firing rate (Hz)", fontsize=9)
        ax.legend(fontsize=6, ncol=2, frameon=False, title="reach direction",
                  title_fontsize=6)
    ax.tick_params(labelsize=8)
fig.suptitle("Reach-direction tuning in macaque motor cortex "
             "(MC_Maze, monkey Jenkins, 789 barrier-free reaches)\n"
             "top: peri-movement PSTHs by direction   |   "
             "middle: polar tuning curves (mean $\\pm$ s.e.m. per 45$\\degree$ sector) "
             "with the fitted cosine   |   bottom: population summary",
             fontsize=12, y=0.975)

for c, u in enumerate(examples):
    ax = fig.add_subplot(gs[1, c], projection="polar")
    ang = np.append(centres, centres[0])
    m = np.append(mu[:, u], mu[0, u])
    s = np.append(se[:, u], se[0, u])
    # the monkey never reached straight downward, so that sector stays empty
    ax.errorbar(ang, m, yerr=s, fmt="o-", ms=3, lw=1.2, color="k")
    fine = np.linspace(0, 2 * np.pi, 200)
    ax.plot(fine, np.clip(fit_mov["b0"][u] + fit_mov["md"][u] * np.cos(fine - fit_mov["pd"][u]), 0, None),
            color="crimson", lw=1.5)
    ax.set_title(f"PD {np.degrees(fit_mov['pd'][u]) % 360:.0f}$\\degree$   "
                 f"$R^2$ = {fit_mov['r2'][u]:.2f}", fontsize=9, pad=22)
    ax.tick_params(axis="x", labelsize=7, pad=0)
    rmax = np.nanmax(m + s)
    ax.set_yticks([round(rmax / 2), round(rmax)])
    ax.tick_params(axis="y", labelsize=6)
    ax.set_rlabel_position(250)

ax = fig.add_subplot(gs[2, 0])
b = np.linspace(0, 0.65, 27)
ax.hist([fit_mov["r2"][~sig_mov], fit_mov["r2"][sig_mov]], bins=b, stacked=True,
        color=["0.75", "crimson"],
        label=[f"n.s. (n={(~sig_mov).sum()})", f"tuned (n={sig_mov.sum()})"])
ax.set(xlabel="cosine-fit $R^2$ (single trials)", ylabel="units",
       title="Direction tuning strength")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 1], projection="polar")
ax.hist(np.mod(fit_mov["pd"][sig_mov], 2 * np.pi), bins=np.linspace(0, 2 * np.pi, 25),
        color="crimson", alpha=0.85)
ax.set_title(f"Preferred directions\nR = {R:.2f} (Rayleigh p = {rayleigh_p:.2f})",
             fontsize=10, pad=18)
ax.tick_params(axis="x", labelsize=7, pad=0)
ax.tick_params(axis="y", labelsize=6)
ax.set_rlabel_position(250)

ax = fig.add_subplot(gs[2, 2])
ax.scatter(fit_mov["b0"], fit_mov["md"], s=14, c=np.where(sig_mov, "crimson", "0.7"))
lim = [0, max(fit_mov["b0"].max(), fit_mov["md"].max()) * 1.05]
ax.plot(lim, lim, "k:", lw=0.8)
ax.set(xlabel="baseline rate $b_0$ (Hz)", ylabel="modulation depth (Hz)",
       title="Modulation depth vs mean rate")

ax = fig.add_subplot(gs[2, 3])
eb = np.linspace(-180, 180, 25)
ax.hist(np.degrees(dpd_split), bins=eb, color="0.4", histtype="step", lw=1.8,
        density=True, label=f"movement, split half (R={Rs:.2f})")
ax.hist(np.degrees(dpd), bins=eb, color="steelblue", alpha=0.8, density=True,
        label=f"planning vs movement (R={Rd:.2f})")
ax.set(xlabel="PD difference (deg)", ylabel="density",
       title="PD is stable within the movement\nepoch but not across epochs")
ax.set_xticks([-180, -90, 0, 90, 180])
ax.legend(fontsize=7)

fig.savefig("fig02_direction_tuning.png", dpi=150, bbox_inches="tight")
print("wrote fig02_direction_tuning.png")
