"""Figures 8-9: GLM model comparison and population decoding."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

g = np.load("glm_results.npz")
fits = np.load("tuning_fits.npz")
names = [str(x) for x in g["names"]]
ll, null_ll, nspk = g["ll"], g["null_ll"], g["nspk"]
bits = (ll - null_ll) / (nspk * np.log(2))          # (n_models, n_units)
keep = nspk > 200                                   # units with enough spikes to score
print("units scored:", keep.sum(), "of", len(nspk))
for i, n in enumerate(names):
    print(f"  {n:22s} median {np.median(bits[i][keep]):+.4f} bits/spike, "
          f"{100*np.mean(bits[i][keep] > 0):.0f}% of units improved")

# ================================================= Figure 8: GLM model comparison
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
ax = axes[0]
order = np.argsort([np.median(bits[i][keep]) for i in range(len(names))])
pos = np.arange(len(names))
for k, i in enumerate(order):
    v = bits[i][keep]
    ax.scatter(np.full(len(v), k) + np.random.default_rng(k).normal(0, .07, len(v)),
               v, s=5, color="0.6", alpha=0.5)
    ax.plot([k - .3, k + .3], [np.median(v)] * 2, color="crimson", lw=2.5)
ax.set_xticks(pos)
ax.set_xticklabels([names[i] for i in order], rotation=30, ha="right", fontsize=8)
ax.axhline(0, color="k", lw=0.7)
ax.set_ylabel("cross-validated bits / spike\n(vs. constant-rate model)")
ax.set_title("A   Velocity beats direction or speed alone", loc="left", fontweight="bold",
             fontsize=10)

def sc(ax, ia, ib, label):
    x, y = bits[ia][keep], bits[ib][keep]
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.scatter(x, y, s=12, color="0.35")
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel(f"{names[ia]} (bits/spike)")
    ax.set_ylabel(f"{names[ib]} (bits/spike)")
    w = stats.wilcoxon(y, x)
    ax.set_title(f"{label}\n{100*np.mean(y > x):.0f}% of units above the line, "
                 f"p = {w.pvalue:.1e}", loc="left", fontweight="bold", fontsize=10)

sc(axes[1], names.index("direction only"), names.index("velocity (vx, vy)"),
   "B   Adding speed scaling to direction")
sc(axes[2], names.index("velocity (vx, vy)"), names.index("velocity + speed"),
   "C   Adding a direction-independent speed term")
sc(axes[3], names.index("velocity + speed"), names.index("2D velocity basis"),
   "D   Nonparametric 2D velocity field")
fig.tight_layout()
fig.savefig("fig08_glm_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig08 done")

# preferred directions from the GLM velocity coefficients vs. the trial-based fit
cv = g["coef_vel"]                                  # (2, n_units)
pd_glm = np.arctan2(cv[1], cv[0]) % (2 * np.pi)
pdir = fits["pdir"]
sig = fits["pval"] < 0.01
dpd = np.degrees((pd_glm - pdir + np.pi) % (2 * np.pi) - np.pi)
print(f"median |PD_glm - PD_trial| = {np.median(np.abs(dpd[sig & keep])):.1f} deg")

# ======================================================== Figure 9: decoding
d = np.load("decoding_results.npz")
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
ax.scatter(np.degrees(d["reach_dir"]), np.degrees(d["dec_angle"]), s=4, alpha=0.25,
           color="C0")
ax.plot([0, 360], [0, 360], "k--", lw=0.8)
ax.set_xlabel("actual reach direction (°)")
ax.set_ylabel("population-vector direction (°)")
ax.set_title("A   Decoding single reaches from 155 units", loc="left", fontweight="bold",
             fontsize=10)

ax = fig.add_subplot(gs[0, 1])
err = d["err"]
ax.hist(err, bins=60, color="0.45")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("population-vector error (°)")
ax.set_ylabel("trials")
ax.set_title(f"B   Median |error| = {np.median(np.abs(err)):.1f}°, "
             f"{100*np.mean(np.abs(err)<45):.0f}% within 45°", loc="left",
             fontweight="bold", fontsize=10)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(np.degrees(d["reach_dir"]), np.abs(err), s=4, alpha=0.2, color="0.4")
bins = np.linspace(0, 2 * np.pi, 13)
bidx = np.digitize(d["reach_dir"], bins) - 1
med = [np.median(np.abs(err)[bidx == k]) for k in range(12)]
ax.plot(np.degrees((bins[:-1] + bins[1:]) / 2), med, "o-", color="crimson")
ax.set_xlabel("actual reach direction (°)")
ax.set_ylabel("|error| (°)")
ax.set_ylim(0, 180)
ax.set_title("C   Error is similar across directions", loc="left", fontweight="bold",
             fontsize=10)

# continuous decoding traces
ax = fig.add_subplot(gs[1, :2])
t, Vt, Vp = d["t"], d["Vtrue"], d["pred"]
m = (t > 300) & (t < 325)
ax.plot(t[m], Vt[m, 0], color="k", lw=1.2, label="actual $v_x$")
ax.plot(t[m], Vp[m, 0], color="C3", lw=1.2, label="decoded $v_x$")
ax.plot(t[m], Vt[m, 1] - 1600, color="k", lw=1.2, ls="--", label="actual $v_y$")
ax.plot(t[m], Vp[m, 1] - 1600, color="C0", lw=1.2, label="decoded $v_y$")
ax.set_xlabel("time (s)")
ax.set_ylabel("hand velocity (mm/s)\n($v_y$ offset for display)")
ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper right")
ax.set_title(f"D   Cross-validated linear decoding of hand velocity "
             f"($R^2$ = {d['r2'][0]:.2f}, {d['r2'][1]:.2f})", loc="left",
             fontweight="bold", fontsize=10)

ax = fig.add_subplot(gs[1, 2])
mv = d["mv"]
hb = ax.hexbin(d["sp_true"][mv], d["sp_pred"][mv], gridsize=40, cmap="magma", bins="log",
               extent=(0, 1300, 0, 1300))
ax.plot([0, 1300], [0, 1300], "w--", lw=0.9)
plt.colorbar(hb, ax=ax, label="20 ms bins")
ax.set_xlabel("actual speed (mm/s)")
ax.set_ylabel("decoded speed (mm/s)")
ax.set_title(f"E   Speed is decodable too (r = {float(d['r_speed']):.2f})", loc="left",
             fontweight="bold", fontsize=10)
fig.savefig("fig09_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig09 done")

np.savez("glm_summary.npz", bits=bits, keep=keep, pd_glm=pd_glm, dpd=dpd, names=g["names"])
