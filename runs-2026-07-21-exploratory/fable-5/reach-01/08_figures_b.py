"""Figures 4-7: polar tuning gallery, population direction summary, speed tuning, 2D fields."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter
from scipy import stats

mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

fits = np.load("tuning_fits.npz")
vt = np.load("velocity_tuning.npz")
td = np.load("trial_data.npz")
tc, centers, b0, b1, pdir, r2, pval = (fits["tc"], fits["centers"], fits["b0"],
                                       fits["b1"], fits["pdir"], fits["r2"], fits["pval"])
tc_sem, unit_ids = fits["tc_sem"], fits["unit_ids"]
sig = pval < 0.01
n_units = len(unit_ids)

# ==================================================== Figure 4: polar tuning gallery
order = np.argsort(-r2)[:12]
fig, axes = plt.subplots(3, 4, figsize=(12, 11.5), subplot_kw={"projection": "polar"})
fig.subplots_adjust(hspace=0.6, wspace=0.35, top=0.9)
for ax, j in zip(axes.ravel(), order):
    th = np.r_[centers, centers[0]]
    v = np.r_[tc[:, j], tc[0, j]]
    e = np.r_[tc_sem[:, j], tc_sem[0, j]]
    ax.fill_between(th, v - e, v + e, color="C0", alpha=0.25)
    ax.plot(th, v, "o-", color="C0", ms=3, lw=1.2)
    fine = np.linspace(0, 2 * np.pi, 200)
    ax.plot(fine, np.maximum(b0[j] + b1[j] * np.cos(fine - pdir[j]), 0),
            color="crimson", lw=1.4)
    ax.plot([pdir[j], pdir[j]], [0, ax.get_ylim()[1]], color="crimson", ls=":", lw=1)
    ax.set_title(f"unit {unit_ids[j]}\nPD {np.degrees(pdir[j]):.0f}°, $R^2$={r2[j]:.2f}",
                 fontsize=8, pad=20)
    ax.tick_params(labelsize=7)
    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
    rmax = np.ceil((tc[:, j] + tc_sem[:, j]).max())
    ax.set_rticks([rmax / 2, rmax])
    ax.set_rlabel_position(135)
fig.suptitle("Cosine direction tuning in single units (blue = measured mean ± SEM, "
             "red = cosine fit)", fontweight="bold", y=0.96)
fig.savefig("fig04_polar_gallery.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig04 done")

# ======================================== Figure 5: population summary of direction tuning
# PD from the continuous (velocity-based) tuning curves, for cross-validation
dir_bins = vt["dir_bins"]
tcd = vt["tc_dir"]                       # (units, 24)
vec = (tcd * np.exp(1j * dir_bins)[None, :]).sum(1) / tcd.sum(1)
pd_cont = np.angle(vec) % (2 * np.pi)
mvl = np.abs(vec)                        # mean vector length = tuning sharpness

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
ax = axes[0]
fig.delaxes(ax)
ax = fig.add_subplot(1, 4, 1, projection="polar")
h, e = np.histogram(pdir[sig], bins=18, range=(0, 2 * np.pi))
ax.bar((e[:-1] + e[1:]) / 2, h, width=2 * np.pi / 18, color="C0", alpha=0.85)
ax.set_title(f"A   Preferred directions\n({sig.sum()} tuned units)", loc="left",
             fontweight="bold", pad=24, fontsize=10)
ax.set_yticklabels([])
# Rayleigh test for uniformity of PDs
Rbar = np.abs(np.exp(1j * pdir[sig]).mean())
z = sig.sum() * Rbar ** 2
p_ray = np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * sig.sum()))
print(f"Rayleigh test on PDs: R={Rbar:.3f}, p={p_ray:.3g}")

ax = axes[1]
ax.hist([b1[sig], b1[~sig]], bins=25, stacked=True, color=["C0", "0.75"],
        label=[f"tuned (n={sig.sum()})", f"not tuned (n={(~sig).sum()})"])
ax.set_xlabel("directional modulation depth $b_1$ (Hz)")
ax.set_ylabel("units")
ax.legend(frameon=False, fontsize=8)
ax.set_title("B   Modulation depth", loc="left", fontweight="bold", fontsize=10)

ax = axes[2]
ax.hist(r2, bins=25, color="0.4")
ax.set_xlabel("single-trial cosine fit $R^2$")
ax.set_ylabel("units")
ax.set_title(f"C   Fit quality (median {np.median(r2):.2f})", loc="left",
             fontweight="bold", fontsize=10)

ax = axes[3]
d = np.degrees((pd_cont - pdir + np.pi) % (2 * np.pi) - np.pi)
ax.scatter(np.degrees(pdir[sig]), np.degrees(pd_cont[sig]), s=14, c=b1[sig],
           cmap="viridis")
ax.plot([0, 360], [0, 360], "k--", lw=0.8)
ax.set_xlabel("PD from trial-averaged rates (°)")
ax.set_ylabel("PD from continuous velocity (°)")
ax.set_title(f"D   Two independent PD estimates agree\n(median |diff| = "
             f"{np.median(np.abs(d[sig])):.1f}°)", loc="left", fontweight="bold",
             fontsize=10)
fig.tight_layout()
fig.savefig("fig05_population_direction.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig05 done; median PD difference = %.1f deg" % np.median(np.abs(d[sig])))

# ================================================== Figure 6: speed / velocity gain
tcs, sbins = vt["tc_speed"], vt["speed_bins"]
tcds = vt["tc_dir_by_speed"]             # (3 terciles, units, 16)
dbins16 = vt["dir_bins16"]
sp_labels = [str(s) for s in vt["sp_labels"]]


def cosfit_curve(theta, r):
    X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    return beta[0], np.hypot(beta[1], beta[2]), np.arctan2(beta[2], beta[1]) % (2 * np.pi)


# align each unit's direction tuning curve to its own preferred direction
aligned = np.zeros_like(tcds)
amp = np.zeros((3, n_units))
base = np.zeros((3, n_units))
pd_sp = np.zeros((3, n_units))
for k in range(3):
    for j in range(n_units):
        b0k, b1k, pdk = cosfit_curve(dbins16, tcds[k, j])
        amp[k, j], base[k, j], pd_sp[k, j] = b1k, b0k, pdk
        shift = int(np.round((pdir[j]) / (2 * np.pi) * 16))
        aligned[k, j] = np.roll(tcds[k, j], -shift + 8)

sd = np.load("speed_direction.npz")
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
axes = {}
ax = fig.add_subplot(gs[0, 0])
top = np.argsort(-b1)[:6]
for j in top:
    ax.plot(sbins, tcs[j], lw=1.3, label=f"u{unit_ids[j]}")
ax.set_xlabel("hand speed (mm/s)")
ax.set_ylabel("firing rate (Hz)")
ax.legend(fontsize=7, frameon=False, ncol=2)
ax.set_title("A   Speed tuning, example units", loc="left", fontweight="bold", fontsize=10)

# rate vs speed, split by direction relative to each unit's preferred direction
ax = fig.add_subplot(gs[0, 1])
cols3 = ["#08519c", "#807dba", "#cb181d"]
for ci, name in enumerate([str(x) for x in sd["cat_names"]]):
    y = sd["prof_n"][ci][:, sig]
    m, s = np.nanmean(y, 1), stats.sem(y, 1, nan_policy="omit")
    ax.plot(sd["sp_c"], m, "o-", color=cols3[ci], label=name, ms=4)
    ax.fill_between(sd["sp_c"], m - s, m + s, color=cols3[ci], alpha=0.25)
ax.set_xlabel("hand speed (mm/s)")
ax.set_ylabel("rate / unit mean rate")
ax.legend(frameon=False, fontsize=8)
ax.set_title("B   Speed raises the rate only for movements\ntoward the preferred direction",
             loc="left", fontweight="bold", fontsize=10)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(sd["slopes"][2][sig], sd["slopes"][0][sig], s=12, color="0.35")
lim = np.nanpercentile(np.abs(sd["slopes"][:, sig]), 99)
ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
ax.axhline(0, color="0.7", lw=0.6)
ax.axvline(0, color="0.7", lw=0.6)
ax.set_xlabel("slope, away from PD (Hz per 100 mm/s)")
ax.set_ylabel("slope, toward PD")
ax.set_title(f"C   Sign of the speed effect flips with direction\n"
             f"(Wilcoxon p = {float(sd['wilcoxon_p']):.1e})", loc="left",
             fontweight="bold", fontsize=10)

ax = fig.add_subplot(gs[1, 0])
xrel = np.degrees(np.linspace(-np.pi, np.pi, 17)[:-1] + np.pi / 16)
cols = ["#9ecae1", "#4292c6", "#08519c"]
for k in range(3):
    m = np.nanmean(aligned[k][sig], 0)
    s = stats.sem(aligned[k][sig], 0, nan_policy="omit")
    ax.plot(xrel, m, color=cols[k], label=sp_labels[k].split(" (")[0])
    ax.fill_between(xrel, m - s, m + s, color=cols[k], alpha=0.3)
ax.set_xlabel("direction relative to preferred (°)")
ax.set_ylabel("firing rate (Hz)")
ax.legend(frameon=False, fontsize=8, title="speed", title_fontsize=8)
ax.set_title("D   Tuning depth grows with speed", loc="left", fontweight="bold", fontsize=10)

ax = fig.add_subplot(gs[1, 1])
for k in range(3):
    ax.scatter(np.full(sig.sum(), k) + np.random.default_rng(k).normal(0, .06, sig.sum()),
               amp[k][sig], s=6, color=cols[k], alpha=0.5)
ax.plot(range(3), [np.median(amp[k][sig]) for k in range(3)], "o-", color="crimson",
        label="median")
ax.set_xticks(range(3))
ax.set_xticklabels([l.split(" (")[0] for l in sp_labels], fontsize=8)
ax.set_ylabel("cosine amplitude $b_1$ (Hz)")
ax.set_yscale("log")
ax.legend(frameon=False, fontsize=8)
t_res = stats.wilcoxon(amp[2][sig], amp[0][sig])
ax.set_title(f"E   Per-unit gain increase\n(Wilcoxon p = {t_res.pvalue:.1e})", loc="left",
             fontweight="bold", fontsize=10)
print("median b1 per speed tercile:", [float(np.median(amp[k][sig])) for k in range(3)],
      "Wilcoxon p =", t_res.pvalue)

# population-average firing rate over PD-relative polar velocity space
ax = fig.add_subplot(gs[1, 2], projection="polar")
pm = np.nanmean(sd["pop_map_n"][:, :, sig], axis=2)
dt_edges, sp_edges = sd["dt_edges"], sd["sp_edges"]
TH, RR = np.meshgrid(dt_edges, sp_edges[:len(sd["sp_c"]) + 1])
pc = ax.pcolormesh(TH, RR, pm, cmap="magma", shading="auto")
plt.colorbar(pc, ax=ax, label="rate / unit mean rate", pad=0.12, shrink=0.8)
ax.set_theta_zero_location("E")
ax.set_xticks(np.radians([0, 90, 180, 270]))
ax.set_xticklabels(["PD", "+90°", "anti-PD", "-90°"], fontsize=8)
ax.set_rlabel_position(160)
ax.set_rticks([500, 1000])
ax.tick_params(labelsize=7, colors="0.3")
for lbl in ax.get_yticklabels():
    lbl.set_color("white")
    lbl.set_fontsize(7)
ax.set_title("F   Population velocity field\n(radius = speed, angle = direction re. PD)",
             loc="left", fontweight="bold", fontsize=10, pad=22)
fig.savefig("fig06_speed_tuning.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig06 done")

# =================================================== Figure 7: 2D velocity tuning fields
tcv, occ = vt["tc_vel"], vt["occ_vel"]
vxb, vyb = vt["vx_bins"], vt["vy_bins"]
mask = occ < 2.0                       # seconds of occupancy required per bin
print("2D bins retained:", (~mask).sum(), "of", mask.size)
sel = np.argsort(-b1)[:8]
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, j in zip(axes.ravel(), sel):
    f = tcv[j].copy()
    f[mask] = np.nan
    fs = gaussian_filter(np.nan_to_num(f), 0.9) / np.maximum(
        gaussian_filter((~mask).astype(float), 0.9), 1e-6)
    fs[mask] = np.nan
    vmax = np.nanpercentile(fs, 99)
    im = ax.pcolormesh(vxb, vyb, fs.T, cmap="magma", vmin=0, vmax=vmax)
    ax.plot([0, 600 * np.cos(pdir[j])], [0, 600 * np.sin(pdir[j])], color="cyan", lw=2)
    ax.set_aspect("equal")
    ax.set_title(f"unit {unit_ids[j]}", fontsize=9)
    plt.colorbar(im, ax=ax, label="Hz", fraction=0.046)
    ax.set_xlabel("$v_x$ (mm/s)")
    ax.set_ylabel("$v_y$ (mm/s)")
fig.suptitle("Firing rate over the 2D hand-velocity plane (cyan = preferred direction "
             "from the trial-based cosine fit)", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig07_velocity_fields.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig07 done")

np.savez("population_summary.npz", pd_cont=pd_cont, mvl=mvl, amp_by_speed=amp,
         base_by_speed=base, aligned=aligned, sig=sig, pd_diff=d)
