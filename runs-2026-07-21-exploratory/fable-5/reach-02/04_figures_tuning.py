import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter


def smooth_nan(M, valid, sigma=0.8):
    """Gaussian-smooth a rate map, ignoring bins with too little occupancy."""
    A = np.where(valid, M, 0.0)
    num = gaussian_filter(A, sigma, mode="nearest")
    den = gaussian_filter(valid.astype(float), sigma, mode="nearest")
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    return np.where(valid, out, np.nan)

D = np.load("_res_direction.npz"); Vr = np.load("_res_velocity.npz")
PEAK_SPD_MS = 125.0   # measured in 02: mean hand speed peaks 125 ms after movement onset
sig = D["sig"]; pd_m = D["move_pd"]; depth = D["depth"] if "depth" in D else D["move_depth"]
r2b = D["r2_binned"]; centers = D["centers"]; keep = D["keep"]; kb = np.where(keep)[0]
psth_dir = D["psth_dir"]; tax = D["tax"]

# ============================ figure 3: population direction tuning ============================
fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

ax = fig.add_subplot(gs[0, 0], projection="polar")
ax.hist(pd_m[sig], bins=np.linspace(-np.pi, np.pi, 25), color="steelblue", edgecolor="w", lw=0.5)
ax.set_rlabel_position(285); ax.tick_params(labelsize=8)
ax.set_title("Preferred directions\n(%d tuned units)" % sig.sum(), pad=30, fontsize=11)

ax = fig.add_subplot(gs[0, 1])
ax.hist(depth[sig], bins=25, color="steelblue", edgecolor="w")
ax.axvline(np.median(depth[sig]), color="crimson", lw=2)
ax.set(xlabel="modulation depth (Hz)", ylabel="units",
       title="Depth of directional modulation\nmedian %.1f Hz" % np.median(depth[sig]))

ax = fig.add_subplot(gs[0, 2])
ax.hist(r2b[sig], bins=np.linspace(0, 1, 26), color="steelblue", edgecolor="w")
ax.axvline(np.median(r2b[sig]), color="crimson", lw=2)
ax.set(xlabel="$R^2$ of cosine fit (direction-binned means)", ylabel="units",
       title="Cosine tuning is a good description\nmedian $R^2$ = %.2f" % np.median(r2b[sig]))

ax = fig.add_subplot(gs[1, 0])
ctr = (D["offsets"] + 0.15) * 1000                       # centre of the 300 ms window
ax.plot(ctr, D["scan"], "o-", color="k")
bc = (D["best_offset"] + 0.15) * 1000
ax.axvline(bc, color="crimson", ls="--", label="best window centre (%+.0f ms)" % bc)
ax.axvline(PEAK_SPD_MS, color="steelblue", ls=":", lw=2, label="peak hand speed (%+.0f ms)" % PEAK_SPD_MS)
ax.legend(fontsize=8)
ax.set(xlabel="centre of the 300 ms rate window\nrelative to movement onset (ms)",
       ylabel="mean modulation depth (Hz)",
       title="Direction signal peaks %d ms\nbefore peak hand speed" % (PEAK_SPD_MS - bc))

ax = fig.add_subplot(gs[1, 1])
dl = np.degrees(np.angle(np.exp(1j * (D["delay_pd"] - pd_m))))
both = sig & (D["delay_p"] < 0.01)
a, b_ = pd_m[both], D["delay_pd"][both]
am, bm = np.angle(np.mean(np.exp(1j * a))), np.angle(np.mean(np.exp(1j * b_)))
rcc = (np.sum(np.sin(a - am) * np.sin(b_ - bm))
       / np.sqrt(np.sum(np.sin(a - am) ** 2) * np.sum(np.sin(b_ - bm) ** 2)))
ax.scatter(np.degrees(a), np.degrees(b_), s=14, color="steelblue")
ax.plot([-180, 180], [-180, 180], "k--", lw=1)
ax.set(xlabel="preferred direction, movement (deg)", ylabel="preferred direction, delay (deg)",
       xticks=[-180, -90, 0, 90, 180], yticks=[-180, -90, 0, 90, 180],
       title="Delay-period PDs do NOT predict movement PDs\n"
             "%d units tuned in both; circular $r$ = %.2f, %.0f%% within 45$\\degree$ (chance 25%%)"
             % (both.sum(), rcc, 100 * (np.abs(dl[both]) < 45).mean()))
ax.title.set_fontsize(10)

ax = fig.add_subplot(gs[1, 2])
# each unit's PSTH aligned to its own preferred direction, normalised
pref_bin = np.array([kb[np.argmin(np.abs(np.angle(np.exp(1j * (centers[kb] - p)))))] for p in pd_m])
anti_bin = np.array([kb[np.argmin(np.abs(np.angle(np.exp(1j * (centers[kb] - p - np.pi)))))] for p in pd_m])
uu = np.where(sig)[0]
diff = np.stack([psth_dir[pref_bin[u], :, u] - psth_dir[anti_bin[u], :, u] for u in uu])
diff = diff / np.abs(diff).max(1, keepdims=True)
order = np.argsort(np.argmax(diff, 1))
im = ax.imshow(diff[order], aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1),
               extent=[tax[0], tax[-1], len(uu), 0], interpolation="nearest")
ax.axvline(0, color="k", ls="--", lw=1)
ax.set(xlabel="time from movement onset (s)", ylabel="unit (sorted by peak time)",
       title="Preferred minus anti-preferred rate\n(each unit normalised)")
fig.colorbar(im, ax=ax, label="normalised rate difference")
fig.suptitle("Population summary of reach-direction tuning (DANDI:000128 MC_Maze, monkey Jenkins, %d units)" % len(sig), y=0.965, fontsize=13)
fig.savefig("fig03_direction_population.png", dpi=140, bbox_inches="tight")
print("saved fig03")

# ============================ figure 4: velocity tuning ============================
lags = Vr["lags"] * 1000
tc2d = Vr["tc2d"]; xe, ye = Vr["xe"], Vr["ye"]; occ = Vr["occ"]
pd_v = Vr["pd_vel"]; pop_grid = Vr["pop_grid"]; scent = Vr["scent"]; acent = Vr["acent"]

fig = plt.figure(figsize=(15.5, 10))
gs = fig.add_gridspec(3, 4, hspace=0.66, wspace=0.45, height_ratios=[1, 1, 1])

# Both curves are normalised to their own peak: the absolute scales differ by 200x
# (a population decode explains far more variance than one 50 ms Poisson spike count),
# but they peak at the same lag, which is the point.
ax = fig.add_subplot(gs[0, :2])
pop_c = Vr["r2_lag_pop"]; su_c = np.nanmedian(Vr["r2_lag"][:, sig], 1)
ax.plot(lags, pop_c / pop_c.max(), "o-", color="crimson",
        label="population decode of hand velocity (peak $R^2$ = %.2f)" % pop_c.max())
ax.plot(lags, su_c / su_c.max(), "s-", color="steelblue", ms=4,
        label="single unit, linear velocity model (peak median $R^2$ = %.4f)" % su_c.max())
ax.axvline(Vr["best_lag"] * 1000, color="crimson", ls="--")
ax.axvline(0, color="0.6", lw=1)
ax.set(xlabel="lag (ms):  positive = neural activity leads the hand",
       ylabel="$R^2$ (normalised to peak)",
       title="Motor cortex leads the hand by ~%d ms" % (Vr["best_lag"] * 1000))
ax.legend(fontsize=8, loc="lower center")

ax = fig.add_subplot(gs[0, 2])
ax.hist(np.degrees(Vr["dpd"][sig]), bins=np.linspace(-180, 180, 37), color="steelblue", edgecolor="w")
ax.set(xlabel="PD(velocity model) - PD(trial cosine fit)  (deg)", ylabel="units", xticks=[-180, -90, 0, 90, 180],
       title="Two independent estimates of PD agree\nmedian |difference| %.0f$\\degree$" % np.degrees(np.median(np.abs(Vr["dpd"][sig]))))

ax = fig.add_subplot(gs[0, 3])
ax.hist(Vr["gain"][sig] * 500, bins=25, color="steelblue", edgecolor="w")
ax.set(xlabel="rate change at 500 mm/s along PD (Hz)", ylabel="units",
       title="Velocity gain\nmedian %.1f Hz" % np.median(Vr["gain"][sig] * 500))

# same four example units as figure 2, so the trial-based and continuous views can be compared
ex = np.argsort(np.where(sig, D["move_depth"], -1))[::-1][:4]
for j, ui in enumerate(ex):
    ax = fig.add_subplot(gs[1, j])
    M = smooth_nan(tc2d[ui].T, occ.T >= 25)
    im = ax.imshow(M, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="viridis", interpolation="nearest")
    L = 0.75 * xe[-1]
    ax.arrow(0, 0, L * np.cos(pd_v[ui]), L * np.sin(pd_v[ui]), color="w", width=18, length_includes_head=True)
    ax.set(xlabel="$v_x$ (mm/s)", ylabel="$v_y$ (mm/s)" if j == 0 else "", title="unit %d (arrow = PD)" % ui)
    fig.colorbar(im, ax=ax, label="Hz" if j == 3 else "")

ax = fig.add_subplot(gs[2, 0:2])
im = ax.imshow(pop_grid, origin="lower", aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(1.0),
               extent=[np.degrees(acent[0]) - 30, np.degrees(acent[-1]) + 30, scent[0] - 50, scent[-1] + 125])
ax.set(xlabel="hand direction relative to each unit's PD (deg)", ylabel="hand speed (mm/s)",
       xticks=[-150, -90, -30, 30, 90, 150],
       title="Population firing rate depends on direction AND speed\n(mean over %d tuned units, normalised to each unit's mean rate)" % sig.sum())
fig.colorbar(im, ax=ax, label="rate / mean rate")

ax = fig.add_subplot(gs[2, 2:])
ipd, iap = int(Vr["ipd"]), int(Vr["iap"])
iorth = int(np.argmin(np.abs(np.abs(acent) - np.pi / 2)))
g = Vr["grid"][..., sig] / np.nanmean(Vr["grid"][..., sig], (0, 1))
for i, (lab, c) in [(ipd, ("toward PD", "crimson")), (iorth, ("orthogonal to PD", "0.4")), (iap, ("away from PD", "steelblue"))]:
    mu = np.nanmean(g[:, i, :], 1); se = np.nanstd(g[:, i, :], 1) / np.sqrt(sig.sum())
    ax.errorbar(scent, mu, yerr=se, marker="o", lw=2, color=c, label=lab, capsize=3)
ax.axhline(1, color="k", lw=0.8, ls=":")
ax.set(xlabel="hand speed (mm/s)", ylabel="rate / mean rate",
       title="Speed scales the directional signal\n(the signature of velocity, not direction, tuning)")
ax.legend(fontsize=9)
fig.suptitle("Hand-velocity tuning in macaque motor cortex (DANDI:000128 MC_Maze, %.0f ms bins, movement periods)" % (Vr["bin2"] * 1000),
             y=0.965, fontsize=13)
fig.savefig("fig04_velocity_tuning.png", dpi=140, bbox_inches="tight")
print("saved fig04")
