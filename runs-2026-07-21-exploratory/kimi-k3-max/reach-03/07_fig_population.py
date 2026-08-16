# Figures: population direction tuning summary + speed/velocity tuning
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("analysis_results.pkl", "rb") as f:
    R = pickle.load(f)
tun_m, tun_d = R["tun_move"], R["tun_delay"]
sig_m, sig_d = R["sig_move"], R["sig_delay"]
n_units = len(sig_m)

peth = np.load("peth_population.npz")
t_ax, pop_pref, pop_anti = peth["t_ax"], peth["pop_pref"], peth["pop_anti"]

fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

# (a) polar histogram of preferred directions (movement-tuned units)
axa = fig.add_subplot(gs[0, 0], projection="polar")
pd_m = tun_m["pd"][sig_m]
bins = np.linspace(-np.pi, np.pi, 17)
counts, _ = np.histogram(pd_m, bins=bins)
theta = (bins[:-1] + bins[1:]) / 2
axa.bar(theta, counts, width=np.diff(bins), color="steelblue", edgecolor="k", lw=0.5)
axa.set_theta_zero_location("E"); axa.set_theta_direction(1)
axa.set_title(f"Preferred directions, movement epoch\n(n={sig_m.sum()} tuned units)", fontsize=10, pad=14)
axa.tick_params(labelsize=8)

# circular correlation (Fisher-Lee)
def circ_corr(a, b):
    sa = np.sin(a[:, None] - a[None, :]); sb = np.sin(b[:, None] - b[None, :])
    iu = np.triu_indices(len(a), 1)
    num = (sa[iu] * sb[iu]).sum()
    den = np.sqrt((sa[iu] ** 2).sum() * (sb[iu] ** 2).sum())
    return num / den

# (b) delay vs movement PD for units tuned in both epochs
axb = fig.add_subplot(gs[0, 1])
both = sig_m & sig_d
d = np.angle(np.exp(1j * (tun_m["pd"][both] - tun_d["pd"][both])))
axb.hist(np.degrees(d), bins=np.arange(-180, 181, 20), color="seagreen", edgecolor="k", lw=0.5)
axb.axvline(0, color="k", ls="--", lw=1)
axb.set_xlabel("PD(movement) − PD(delay) (deg)")
axb.set_ylabel("units")
r_cc = circ_corr(tun_d["pd"][both], tun_m["pd"][both])
axb.set_title(f"PD stability across epochs (n={both.sum()} tuned in both)\ncircular corr = {r_cc:.2f}", fontsize=10)

# (c) modulation depth distributions
axc = fig.add_subplot(gs[0, 2])
axc.hist(tun_m["mod_depth"][sig_m], bins=20, alpha=0.7, color="steelblue",
         label=f"movement (n={sig_m.sum()})")
axc.hist(tun_d["mod_depth"][sig_d], bins=20, alpha=0.7, color="darkorange",
         label=f"delay (n={sig_d.sum()})")
axc.set_xlabel("modulation depth (cosine amp / baseline)")
axc.set_ylabel("units")
axc.set_title("Direction tuning strength", fontsize=10)
axc.legend(fontsize=8)

# (d) fraction tuned
axd = fig.add_subplot(gs[1, 0])
fracs = [sig_d.mean(), sig_m.mean(), (sig_d & sig_m).mean()]
axd.bar(["delay", "movement", "both"], [f * 100 for f in fracs],
        color=["darkorange", "steelblue", "slategray"], edgecolor="k", lw=0.5)
for i, f in enumerate(fracs):
    axd.text(i, f * 100 + 1, f"{f*100:.0f}%", ha="center", fontsize=10)
axd.set_ylabel("% of 182 units")
axd.set_ylim(0, 90)
axd.set_title("Fraction significantly direction-tuned\n(ANOVA & shuffle p<0.01)", fontsize=10)

# (e) population PETH preferred vs anti-preferred
axe = fig.add_subplot(gs[1, 1])
m_p, s_p = pop_pref.mean(axis=0), pop_pref.std(axis=0) / np.sqrt(len(pop_pref))
m_a, s_a = pop_anti.mean(axis=0), pop_anti.std(axis=0) / np.sqrt(len(pop_anti))
axe.plot(t_ax, m_p, color="crimson", lw=1.5, label="preferred dir. (±45°)")
axe.fill_between(t_ax, m_p - s_p, m_p + s_p, color="crimson", alpha=0.25)
axe.plot(t_ax, m_a, color="navy", lw=1.5, label="anti-preferred dir. (±45°)")
axe.fill_between(t_ax, m_a - s_a, m_a + s_a, color="navy", alpha=0.25)
axe.axvline(0, color="k", ls="--", lw=1)
axe.set_xlabel("time from move onset (s)")
axe.set_ylabel("firing rate (Hz)")
axe.set_title(f"Population PETH (n={len(pop_pref)} tuned units, mean ± sem)", fontsize=10)
axe.legend(fontsize=8)

# (f) cosine fit R2 distribution
axf = fig.add_subplot(gs[1, 2])
axf.hist(tun_m["r2"][sig_m], bins=20, color="steelblue", edgecolor="k", lw=0.5)
axf.set_xlabel("cosine fit R² (single-trial rates)")
axf.set_ylabel("units")
axf.set_title(f"Cosine fit quality, movement-tuned units\nmedian R²={np.median(tun_m['r2'][sig_m]):.2f}", fontsize=10)

fig.suptitle("Population summary: reach-direction tuning in macaque M1/PMd (MC_Maze, DANDI 000128)", fontsize=12)
plt.savefig("fig5_population_direction.png", dpi=150, bbox_inches="tight")
print("saved fig5_population_direction.png")

# ---------------- speed tuning figure ----------------
tc_speed = R["tc_speed"]  # (n_units, 12) xarray or ndarray
tc_speed = np.asarray(tc_speed)
speed_bins = np.linspace(0, 1, tc_speed.shape[1])  # placeholder; real centers below if available
try:
    speed_centers = R["tc_speed"].speed.values
except Exception:
    speed_centers = None

xcorr, lags = R["xcorr"], R["lags"]
peak_r, peak_lag = R["peak_r"], R["peak_lag"]

fig2, ax2 = plt.subplots(2, 2, figsize=(11, 8))
# (a) example speed tuning curves (top modulated units by speed range)
rng_speed = tc_speed.max(axis=1) - tc_speed.min(axis=1)
ex = np.argsort(-rng_speed)[:4]
x = speed_centers if speed_centers is not None else np.arange(tc_speed.shape[1])
for u in ex:
    ax2[0, 0].plot(x, tc_speed[u], lw=1.5, label=f"unit {u}")
ax2[0, 0].set_xlabel("hand speed" + ("" if speed_centers is not None else " bin"))
ax2[0, 0].set_ylabel("firing rate (Hz)")
ax2[0, 0].set_title("Example speed tuning curves (movement epochs)", fontsize=10)
ax2[0, 0].legend(fontsize=8)

# (b) population normalized speed tuning
zn = (tc_speed - tc_speed.mean(axis=1, keepdims=True)) / np.maximum(tc_speed.std(axis=1, keepdims=True), 1e-9)
ax2[0, 1].plot(x, zn.mean(axis=0), color="k", lw=2)
ax2[0, 1].fill_between(x, zn.mean(axis=0) - zn.std(axis=0) / np.sqrt(n_units),
                       zn.mean(axis=0) + zn.std(axis=0) / np.sqrt(n_units), color="gray", alpha=0.4)
ax2[0, 1].set_xlabel("hand speed" + ("" if speed_centers is not None else " bin"))
ax2[0, 1].set_ylabel("z-scored rate")
ax2[0, 1].set_title(f"Population speed tuning (n={n_units}, mean ± sem)", fontsize=10)

# (c) peak correlation distribution
ax2[1, 0].hist(peak_r, bins=25, color="teal", edgecolor="k", lw=0.5)
ax2[1, 0].axvline(np.median(peak_r), color="k", ls="--",
                  label=f"median r={np.median(peak_r):.2f}")
ax2[1, 0].set_xlabel("peak spike-rate × speed correlation")
ax2[1, 0].set_ylabel("units")
ax2[1, 0].set_title("Speed modulation of firing rate", fontsize=10)
ax2[1, 0].legend(fontsize=8)

# (d) peak lag distribution
ax2[1, 1].hist(peak_lag, bins=np.arange(-300, 301, 20), color="purple", edgecolor="k", lw=0.5, alpha=0.7)
ax2[1, 1].axvline(0, color="k", ls="-", lw=1)
ax2[1, 1].axvline(np.median(peak_lag), color="k", ls="--",
                  label=f"median={np.median(peak_lag):.0f} ms")
ax2[1, 1].set_xlabel("lag of peak correlation (ms; >0 = spikes lead speed)")
ax2[1, 1].set_ylabel("units")
ax2[1, 1].set_title("Neural activity leads hand speed", fontsize=10)
ax2[1, 1].legend(fontsize=8)

fig2.suptitle("Velocity / speed tuning in M1/PMd during reaching", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("fig6_speed_tuning.png", dpi=150)
print("saved fig6_speed_tuning.png")
print("speed_centers:", speed_centers)
