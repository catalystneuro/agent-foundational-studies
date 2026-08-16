"""Step 3: continuous velocity tuning - 2D velocity rate maps, speed gain, lead time."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

import common

data = common.load_pynapple()
units = data["units"]
area = units.get_info("area").values.astype(str)

# ------------------------------------------------ how far does the neural lead?
all_counts, t_all = common.bin_spikes(data)
LAGS = np.arange(-0.20, 0.32, 0.02)
lag_r2 = np.zeros((len(LAGS), len(units)))
for i, lag in enumerate(tqdm(LAGS, desc="lag scan")):
    vxy, ok = common.velocity_at(data, t_all, lag=lag)
    lag_r2[i] = common.cosine_speed_fit(all_counts[ok], vxy[ok])["r2"]
best_lag_unit = LAGS[np.argmax(lag_r2, 0)]
pop_lag = LAGS[np.argmax(lag_r2.mean(1))]
print("population-optimal lag: %.0f ms (neural leads hand)" % (1000 * pop_lag))
print("median per-unit optimal lag: %.0f ms" % (1000 * np.median(best_lag_unit)))

# ------------------------------------------------------ fit at the best lag
vxy, ok = common.velocity_at(data, t_all, lag=pop_lag)
counts, vxy, t_bin = all_counts[ok], vxy[ok], t_all[ok]
speed = np.hypot(vxy[:, 0], vxy[:, 1])
print("bins: %d (%.0f s), speed 95th pct %.2f m/s" % (len(t_bin), len(t_bin) * common.BIN,
                                                      np.percentile(speed, 95)))

fit = common.cosine_speed_fit(counts, vxy)
res = pd.DataFrame(dict(unit=np.arange(len(units)), area=area,
                        b0=fit["b0"], bx=fit["bx"], by=fit["by"],
                        gain=fit["gain"], pd_rad=fit["pd"],
                        pd_deg=np.degrees(fit["pd"]) % 360, r2=fit["r2"],
                        best_lag=best_lag_unit))
res.to_csv("results_velocity_model.csv", index=False)
print("velocity-model R2: median %.3f, 90th pct %.3f" % (res.r2.median(), res.r2.quantile(.9)))
print("velocity gain (Hz per m/s): median %.1f" % res.gain.median())

# consistency with the trial-based preferred directions
trial_res = pd.read_csv("results_direction_tuning.csv")
d = np.angle(np.exp(1j * (res.pd_rad.values - trial_res.pd_rad.values)))
m = trial_res.tuned.values
print("|PD(continuous) - PD(trial)| median %.1f deg over %d tuned units"
      % (np.degrees(np.median(np.abs(d[m]))), m.sum()))

# --------------------------------------------- 2D velocity rate maps + speed curves
V_EDGES = np.linspace(-0.6, 0.6, 25)
vb = [np.digitize(vxy[:, k], V_EDGES) - 1 for k in range(2)]
inb = (vb[0] >= 0) & (vb[0] < 24) & (vb[1] >= 0) & (vb[1] < 24)
flat = vb[1][inb] * 24 + vb[0][inb]
occ = np.bincount(flat, minlength=576).astype(float)


def velocity_map(u, min_occ=20):
    s = np.bincount(flat, weights=counts[inb, u], minlength=576)
    r = np.where(occ >= min_occ, s / np.maximum(occ, 1) / common.BIN, np.nan)
    return r.reshape(24, 24)


S_EDGES = np.array([0.0, 0.05, 0.12, 0.22, 0.35, 0.5, 0.7, 1.2])
S_CENT = (S_EDGES[:-1] + S_EDGES[1:]) / 2
sb = np.digitize(speed, S_EDGES) - 1


def speed_curves(u):
    """Rate vs speed separately for bins moving toward / away from the unit's PD."""
    ang = np.abs(np.angle(np.exp(1j * (np.arctan2(vxy[:, 1], vxy[:, 0]) - res.pd_rad[u]))))
    out = {}
    for lbl, sel in [("toward PD", ang < np.pi / 4), ("away from PD", ang > 3 * np.pi / 4)]:
        mu = np.full(len(S_CENT), np.nan)
        se = np.full(len(S_CENT), np.nan)
        for i in range(len(S_CENT)):
            m2 = sel & (sb == i)
            if m2.sum() > 30:
                mu[i] = counts[m2, u].mean() / common.BIN
                se[i] = counts[m2, u].std() / common.BIN / np.sqrt(m2.sum())
        out[lbl] = (mu, se)
    return out


ex = res.sort_values("r2", ascending=False).unit.values[:4]
fig = plt.figure(figsize=(15, 8.2))
gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.34, left=0.06, right=0.93,
                      top=0.85, bottom=0.09)
cmap = plt.get_cmap("magma").copy()
cmap.set_bad("0.75")
for i, u in enumerate(ex):
    ax = fig.add_subplot(gs[0, i])
    rm = velocity_map(u)
    im = ax.imshow(rm, origin="lower", cmap=cmap,
                   extent=[100 * V_EDGES[0], 100 * V_EDGES[-1]] * 2, aspect="equal")
    ax.arrow(0, 0, 45 * np.cos(res.pd_rad[u]), 45 * np.sin(res.pd_rad[u]),
             color="w", width=2.5, head_width=9, length_includes_head=True)
    ax.set_xlabel("hand $v_x$ (cm/s)", fontsize=9)
    if i == 0:
        ax.set_ylabel("hand $v_y$ (cm/s)")
    ax.set_title("unit %d (%s)\nR²=%.2f, gain %.0f Hz per m/s"
                 % (u, area[u], res.r2[u], res.gain[u]), fontsize=9)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Hz", fontsize=8)

for i, u in enumerate(ex):
    ax = fig.add_subplot(gs[1, i])
    for lbl, c in [("toward PD", "#08519c"), ("away from PD", "#bdbdbd")]:
        mu, se = speed_curves(u)[lbl]
        ax.errorbar(100 * S_CENT, mu, yerr=se, marker="o", ms=4, lw=1.6, color=c,
                    capsize=2.5, label=lbl)
    ax.set_xlabel("hand speed (cm/s)", fontsize=9)
    if i == 0:
        ax.set_ylabel("firing rate (Hz)")
    ax.tick_params(labelsize=8)
    if i == 3:
        ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Firing rate as a function of the hand velocity vector "
             "(20 ms bins, neural activity shifted %d ms earlier)\n"
             "top: rate map over ($v_x$, $v_y$), arrow = fitted preferred direction   |   "
             "bottom: speed tuning along vs. against the preferred direction"
             % round(1000 * pop_lag), fontsize=12)
fig.savefig("fig05_velocity_maps.png", dpi=130)
print("wrote fig05_velocity_maps.png")

# ------------------------------------------------------------ population panel
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
fig.subplots_adjust(wspace=0.48, left=0.06, right=0.98, top=0.82, bottom=0.15)

ax = axs[0]
tuned40 = np.argsort(-lag_r2.max(0))[:40]      # units the velocity model actually fits
norm = lag_r2[:, tuned40] / np.maximum(lag_r2.max(0)[tuned40], 1e-9)
m_, s_ = norm.mean(1), norm.std(1) / np.sqrt(norm.shape[1])
ax.fill_between(1000 * LAGS, m_ - s_, m_ + s_, color="#4292c6", alpha=0.3)
ax.plot(1000 * LAGS, m_, color="#08519c", lw=2.5, label="40 best-fit units (mean \u00b1 SEM)")
pop = lag_r2.mean(1)
ax.plot(1000 * LAGS, pop / pop.max(), "k", lw=1.8, label="all %d units" % len(units))
ax.axvline(1000 * pop_lag, color="#d1495b", ls="--", lw=1.5,
           label="peak %d ms" % round(1000 * pop_lag))
ax.axvline(0, color="0.5", lw=0.8)
axh = ax.twinx()
axh.hist(1000 * best_lag_unit[tuned40], bins=1000 * np.append(LAGS - 0.01, LAGS[-1] + 0.01),
         color="0.8", zorder=0)
axh.set_ylabel("best-fit units\nwith this peak lag", fontsize=8, color="0.5")
axh.tick_params(labelsize=8, colors="0.5")
axh.set_ylim(0, 26)
ax.set_zorder(axh.get_zorder() + 1); ax.patch.set_visible(False)
ax.set_xlabel("neural lead (ms)   [spikes shifted earlier →]")
ax.set_ylabel("velocity-model R² (normalised)")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("Activity leads the hand by ~%d ms" % round(1000 * pop_lag))

ax = axs[1]
ax.scatter(np.degrees(trial_res.pd_rad) % 360, res.pd_deg,
           c=["#d0d0d0" if not t else common.AREA_COLORS[a]
              for t, a in zip(trial_res.tuned, res.area)], s=18)
ax.plot([0, 360], [0, 360], "k--", lw=0.8)
ax.set_xlabel("preferred direction, trial-averaged (°)")
ax.set_ylabel("preferred direction,\ncontinuous velocity model (°)")
ax.set_title("The two analyses agree\n(median offset %.0f°)"
             % np.degrees(np.median(np.abs(d[m]))), fontsize=10)
ax.set_xticks([0, 90, 180, 270, 360]); ax.set_yticks([0, 90, 180, 270, 360])

ax = axs[2]
for a in ["M1", "PMd"]:
    ax.hist(res.loc[res.area == a, "gain"], bins=np.linspace(0, res.gain.max(), 28),
            alpha=0.6, color=common.AREA_COLORS[a], label=a)
ax.set_xlabel("velocity gain |b| (Hz per m/s)")
ax.set_ylabel("units")
ax.legend(fontsize=8)
ax.set_title("Speed sensitivity across the population\n(median %.0f Hz per m/s)"
             % res.gain.median(), fontsize=10)
fig.suptitle("Continuous velocity encoding across %d M1/PMd units" % len(units), fontsize=12)
fig.savefig("fig06_velocity_population.png", dpi=130)
print("wrote fig06_velocity_population.png")

np.savez("results_lag_scan.npz", LAGS=LAGS, lag_r2=lag_r2, pop_lag=pop_lag)
