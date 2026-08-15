"""Step 2: trial-based reach-direction tuning and its dependence on speed."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import common

rng = np.random.default_rng(0)
data = common.load_pynapple()
units = data["units"]
tr = common.trial_table(data)
tr = common.add_movement_kinematics(data, tr)
use = tr[tr["use"] & np.isfinite(tr["mv_dir"])].copy()
print("trials used: %d" % len(use))

# spike counts in the peri-movement window (neural window leads the kinematics)
starts = use["move_onset_time"].values + common.NEURAL_WIN[0]
stops = use["move_onset_time"].values + common.NEURAL_WIN[1]
rates = common.rates_in_epochs(units, starts, stops)      # (n_trials, n_units)
print("rates matrix", rates.shape, "mean %.2f Hz" % rates.mean())

theta = use["mv_dir"].values
sp = use["mv_speed"].values
area = units.get_info("area").values.astype(str)

# ------------------------------------------------------- cosine tuning fits
fit = common.fit_cosine(rates, theta)
n_perm = 1000
perm_md = np.zeros((n_perm, rates.shape[1]))
for i in tqdm(range(n_perm), desc="permutations"):
    perm_md[i] = common.fit_cosine(rates, rng.permutation(theta))["mod_depth"]
pval = (perm_md >= fit["mod_depth"][None, :]).mean(0)

res = pd.DataFrame(dict(
    unit=np.arange(rates.shape[1]), area=area,
    mean_rate=rates.mean(0), b0=fit["b0"], mod_depth=fit["mod_depth"],
    pd_rad=fit["pd"], pd_deg=np.degrees(fit["pd"]) % 360,
    r2=fit["r2"], p=pval,
))
res["tuned"] = res["p"] < 0.01
res.to_csv("results_direction_tuning.csv", index=False)
print(res.groupby("area")[["tuned"]].agg(["sum", "size", "mean"]))
print("overall tuned: %d/%d (%.0f%%)" % (res.tuned.sum(), len(res), 100 * res.tuned.mean()))
print("median mod depth (tuned): %.2f Hz; median R2 %.2f"
      % (res.loc[res.tuned, "mod_depth"].median(), res.loc[res.tuned, "r2"].median()))

# ------------------------------------- does tuning amplitude scale with speed?
q = np.quantile(sp, [1 / 3, 2 / 3])
terc = np.digitize(sp, q)
terc_md, terc_pd, terc_b0 = [], [], []
for k in range(3):
    m = terc == k
    f = common.fit_cosine(rates[m], theta[m])
    terc_md.append(f["mod_depth"]); terc_pd.append(f["pd"]); terc_b0.append(f["b0"])
terc_md, terc_pd, terc_b0 = np.array(terc_md), np.array(terc_pd), np.array(terc_b0)
speed_terc = np.array([sp[terc == k].mean() for k in range(3)])
print("speed terciles (m/s):", speed_terc.round(3))
tu = res.tuned.values
print("modulation depth by tercile (tuned units, median Hz):", np.median(terc_md[:, tu], 1).round(2))
print("baseline b0 by tercile (tuned units, median Hz):", np.median(terc_b0[:, tu], 1).round(2))
from scipy import stats
w = stats.wilcoxon(terc_md[2, tu], terc_md[0, tu])
print("Wilcoxon fast vs slow modulation depth: W=%.0f p=%.2g" % (w.statistic, w.pvalue))
pd_shift = np.degrees(np.abs(np.angle(np.exp(1j * (terc_pd[2, tu] - terc_pd[0, tu])))))
print("median |PD(fast) - PD(slow)| = %.1f deg" % np.median(pd_shift))

np.savez("results_speed_scaling.npz", terc_md=terc_md, terc_b0=terc_b0, terc_pd=terc_pd,
         speed_terc=speed_terc, tuned=tu, theta=theta, sp=sp, rates=rates,
         pd_rad=res.pd_rad.values, area=area)

# ------------------------------------------------------------------ figures
DIR_EDGES = np.linspace(-np.pi, np.pi, 13)
DIR_CENT = DIR_EDGES[:-1] + np.diff(DIR_EDGES) / 2
dbin = np.digitize(theta, DIR_EDGES) - 1


def dir_curve(r, mask=None):
    m = np.ones(len(r), bool) if mask is None else mask
    mu = np.array([r[m & (dbin == i)].mean() for i in range(12)])
    se = np.array([r[m & (dbin == i)].std() / max(np.sqrt((m & (dbin == i)).sum()), 1)
                   for i in range(12)])
    return mu, se


ex = res.loc[res.tuned].sort_values("mod_depth", ascending=False).unit.values[:6]

fig = plt.figure(figsize=(15, 6.4))
gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.85], hspace=0.30, wspace=0.5,
                      left=0.05, right=0.97, top=0.82, bottom=0.10)
for i, u in enumerate(ex):
    ax = fig.add_subplot(gs[0, i], projection="polar")
    mu, se = dir_curve(rates[:, u])
    th = np.append(DIR_CENT, DIR_CENT[0])
    ax.errorbar(th, np.append(mu, mu[0]), yerr=np.append(se, se[0]),
                color=common.AREA_COLORS[area[u]], lw=1.6, capsize=2)
    fx = np.linspace(-np.pi, np.pi, 200)
    ax.plot(fx, res.b0[u] + res.mod_depth[u] * np.cos(fx - res.pd_rad[u]),
            "k--", lw=1.1)
    ax.set_title("unit %d (%s)\nPD %.0f°  R²=%.2f" % (u, area[u], res.pd_deg[u], res.r2[u]),
                 fontsize=9, pad=22)
    ax.tick_params(labelsize=7)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_rlabel_position(160)

for i, u in enumerate(ex):
    ax = fig.add_subplot(gs[1, i])
    for k, (c, lbl) in enumerate(zip(["#9ecae1", "#4292c6", "#08519c"],
                                     ["slow", "mid", "fast"])):
        mu, se = dir_curve(rates[:, u], terc == k)
        d = np.degrees(np.angle(np.exp(1j * (DIR_CENT - res.pd_rad[u]))))
        o = np.argsort(d)
        ax.errorbar(d[o], mu[o], yerr=se[o], color=c, lw=1.3, capsize=1.5,
                    label="%s (%.0f cm/s)" % (lbl, 100 * speed_terc[k]))
    ax.set_xlabel("angle from PD (°)", fontsize=8)
    if i == 0:
        ax.set_ylabel("firing rate (Hz)")
    ax.tick_params(labelsize=7)
    ax.set_xticks([-180, -90, 0, 90, 180])
    if i == 5:
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
fig.suptitle("Reach-direction tuning of M1/PMd units, MC_Maze (DANDI 000128)\n"
             "top: polar tuning curves (dashed = cosine fit)  |  bottom: same units split by "
             "movement speed", fontsize=12)
fig.savefig("fig03_example_tuning.png", dpi=130)
print("wrote fig03_example_tuning.png")


fig, axs = plt.subplots(2, 3, figsize=(14, 8.5))
fig.subplots_adjust(hspace=0.42, wspace=0.32, left=0.07, right=0.97, top=0.88, bottom=0.09)

ax = axs[0, 0]
bins = np.linspace(0, np.ceil(res.mod_depth.max()), 30)
for a in ["M1", "PMd"]:
    ax.hist(res.loc[res.area == a, "mod_depth"], bins=bins, alpha=0.6,
            color=common.AREA_COLORS[a], label=a)
ax.set_xlabel("cosine modulation depth (Hz)"); ax.set_ylabel("units")
ax.legend(fontsize=8); ax.set_title("Depth of directional modulation")

ax = axs[0, 1]
ax.scatter(res.mean_rate, res.mod_depth, c=[common.AREA_COLORS[a] for a in res.area],
           s=16, alpha=0.8)
ax.plot([0, res.mean_rate.max()], [0, res.mean_rate.max()], "k--", lw=0.8,
        label="modulation = mean rate")
ax.set_xlabel("mean peri-movement rate (Hz)"); ax.set_ylabel("modulation depth (Hz)")
ax.legend(fontsize=8); ax.set_title("Modulation vs. overall rate")

ax = fig.add_subplot(2, 3, 3, projection="polar")
axs[0, 2].remove()
for a in ["M1", "PMd"]:
    m = (res.area == a) & res.tuned
    h, e = np.histogram(res.loc[m, "pd_rad"], bins=np.linspace(-np.pi, np.pi, 19))
    ax.bar(e[:-1] + np.diff(e) / 2, h, width=np.diff(e), alpha=0.6,
           color=common.AREA_COLORS[a], label=a)
ax.set_title("Preferred directions\n(significantly tuned units)", fontsize=10, pad=18)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.22, 1.12))

ax = axs[1, 0]
for k in range(3):
    ax.hist(terc_md[k, tu], bins=np.linspace(0, terc_md[:, tu].max(), 25),
            histtype="step", lw=1.8, color=["#9ecae1", "#4292c6", "#08519c"][k],
            label="%.0f cm/s" % (100 * speed_terc[k]))
ax.set_xlabel("modulation depth (Hz)"); ax.set_ylabel("tuned units")
ax.legend(fontsize=8, title="mean speed", title_fontsize=8)
ax.set_title("Directional modulation grows with speed")

ax = axs[1, 1]
ax.plot(100 * speed_terc, terc_md[:, tu], color="0.85", lw=0.5, alpha=0.6)
ax.errorbar(100 * speed_terc, np.median(terc_md[:, tu], 1),
            yerr=stats.median_abs_deviation(terc_md[:, tu], axis=1),
            color="k", lw=2.5, marker="o", capsize=4, label="modulation depth")
ax.errorbar(100 * speed_terc, np.median(terc_b0[:, tu], 1),
            yerr=stats.median_abs_deviation(terc_b0[:, tu], axis=1),
            color="#d1495b", lw=2.5, marker="s", capsize=4, label="baseline b₀")
ax.set_ylim(0, 6)
ax.set_xlabel("mean movement speed (cm/s)"); ax.set_ylabel("Hz  (median ± MAD)")
ax.legend(fontsize=8); ax.set_title("Gain, not offset, scales with speed\n"
                                    "(Wilcoxon fast vs slow p=%.1g)" % w.pvalue, fontsize=10)

ax = axs[1, 2]
ax.hist(pd_shift, bins=np.linspace(0, 180, 25), color="0.4")
ax.axvline(np.median(pd_shift), color="#d1495b", lw=2,
           label="median %.0f°" % np.median(pd_shift))
ax.axvline(90, color="k", ls=":", lw=1.5, label="median if PDs were unrelated")
ax.set_xlabel("|PD(fast) − PD(slow)| (°)"); ax.set_ylabel("tuned units")
ax.legend(fontsize=7.5)
ax.set_title("Preferred direction is largely\npreserved across speeds", fontsize=10)

fig.suptitle("Population summary: %d/%d units directionally tuned (permutation p<0.01), "
             "MC_Maze monkey Jenkins" % (res.tuned.sum(), len(res)), fontsize=12)
fig.savefig("fig04_population_direction.png", dpi=130)
print("wrote fig04_population_direction.png")
