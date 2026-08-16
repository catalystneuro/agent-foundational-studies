"""Theta phase precession of CA1 place cells, single session of DANDI:000044."""

import numpy as np
import scipy.stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import theta_lib as tl

res = tl.analyze_session(tl.SESSIONS[0])
fields = res["fields"]
pos, theta_phase, laps = res["position"], res["theta_phase"], res["laps"]

slopes = np.array([f["slope"] for f in fields])
rhos = np.array([f["rho"] for f in fields])
pvals = np.array([f["p"] for f in fields])
sig = (pvals < 0.05) & (slopes < 0)

# ---------------------------------------------------------------------------------
# Shuffle control: break the spike-position / spike-phase pairing within each field
# ---------------------------------------------------------------------------------
rng = np.random.default_rng(1)
null_rho, null_slope = [], []
for f in tqdm(fields, desc="phase shuffles"):
    for _ in range(20):
        r = tl.circ_lin_regress(f["x"], rng.permutation(f["phase"]), n_slopes=501)
        null_rho.append(r["rho"])
        null_slope.append(r["slope"])
null_rho = np.array(null_rho)
null_slope = np.array(null_slope)
ok = np.isfinite(null_rho)
null_rho, null_slope = null_rho[ok], null_slope[ok]

print("observed: median rho %.3f, %.0f%% of fields significant and negative"
      % (np.median(rhos), 100 * sig.mean()))
print("shuffled: median rho %.3f, %.1f%% of shuffles at or below the observed median"
      % (np.median(null_rho), 100 * (null_rho <= np.median(rhos)).mean()))
print("observed vs shuffled slope: Mann-Whitney p = %.2g"
      % scipy.stats.mannwhitneyu(slopes, null_slope).pvalue)

# ---------------------------------------------------------------------------------
# Figure 5: example cells
# ---------------------------------------------------------------------------------
order = np.argsort(rhos)  # strongest precession first
examples = [fields[i] for i in order[:6]]

fig, axes = plt.subplots(2, 6, figsize=(19, 6.5), height_ratios=[1, 2.2])
for j, f in enumerate(examples):
    k = list(res["pyr"].index).index(f["unit"])
    m = res["maps"][f["direction"]]
    axes[0, j].plot(m["centers"], m["rates"][k], "k")
    axes[0, j].axvspan(f["lo"], f["hi"], color="tab:orange", alpha=0.25)
    axes[0, j].set_xlim(res["track_range"])
    axes[0, j].set_title(
        "unit %d, %sward\n%.1f Hz peak, %.0f cm field" % (f["unit"], f["direction"], f["peak_rate"], f["width"]),
        fontsize=9,
    )
    if j == 0:
        axes[0, j].set_ylabel("Rate (Hz)")
    axes[0, j].set_xlabel("Position (cm)", fontsize=8)

    ax = axes[1, j]
    ph = np.degrees(f["phase"])
    ax.plot(f["x"], ph, ".", ms=3, color="0.35", alpha=0.6)
    ax.plot(f["x"], ph + 360, ".", ms=3, color="0.35", alpha=0.6)
    xx = np.linspace(0, 1, 100)
    yy = np.degrees(f["slope"] * xx + f["phi0"])
    for shift in (-360, 0, 360, 720):
        ax.plot(xx, (yy % 360) + shift, "r-", lw=2)
    ax.set_ylim(0, 720)
    ax.set_xlim(0, 1)
    ax.set_yticks([0, 180, 360, 540, 720])
    ax.set_xlabel("Position in field")
    if j == 0:
        ax.set_ylabel("Theta phase (deg)")
    ptxt = "p<1e-16" if f["p"] < 1e-16 else "p=%.1g" % f["p"]
    ax.set_title(
        r"slope %.0f$\degree$/field, $\rho$=%.2f, %s" % (np.degrees(f["slope"]), f["rho"], ptxt),
        fontsize=9,
    )

fig.suptitle(
    "Theta phase precession in single CA1 place fields (%s): spikes fire at progressively "
    "earlier theta phases as the rat crosses the field" % res["session"],
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig05_precession_examples.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------------
# Figure 6: population summary
# ---------------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3)

# (a) pooled phase-position density
ax = fig.add_subplot(gs[0, 0])
X = np.concatenate([f["x"] for f in np.array(fields, dtype=object)[sig]])
P = np.degrees(np.concatenate([f["phase"] for f in np.array(fields, dtype=object)[sig]]))
H, xe, ye = np.histogram2d(
    np.r_[X, X], np.r_[P, P + 360], bins=[20, 40], range=[[0, 1], [0, 720]]
)
H = H / H.sum(axis=1, keepdims=True)
im = ax.pcolormesh(xe, ye, H.T, cmap="magma")
ax.set_xlabel("Normalized position in field")
ax.set_ylabel("Theta phase (deg)")
ax.set_yticks([0, 180, 360, 540, 720])
ax.set_title("Pooled spike density\n(%d significant fields, %d spikes)" % (sig.sum(), len(X)))
plt.colorbar(im, ax=ax, label="P(phase | position)")

# (b) circular mean phase vs position
ax = fig.add_subplot(gs[0, 1])
edges = np.linspace(0, 1, 11)
means, sems = [], []
for a, b in zip(edges[:-1], edges[1:]):
    per_field = []
    for f in np.array(fields, dtype=object)[sig]:
        m = (f["x"] >= a) & (f["x"] < b)
        if m.sum() >= 5:
            per_field.append(np.angle(np.exp(1j * f["phase"][m]).mean()))
    per_field = np.unwrap(np.array(per_field))
    means.append(np.degrees(np.angle(np.exp(1j * np.array(per_field)).mean())))
    sems.append(np.degrees(np.std(per_field) / np.sqrt(len(per_field))))
means = np.unwrap(np.radians(np.array(means)))
means = np.degrees(means)
means = means - means[0] + np.mod(means[0], 360)
ctr = (edges[:-1] + edges[1:]) / 2
ax.errorbar(ctr, means, yerr=sems, fmt="o-", color="tab:blue", capsize=3)
ax.errorbar(ctr, np.array(means) + 360, yerr=sems, fmt="o-", color="tab:blue", capsize=3, alpha=0.5)
ax.set_xlabel("Normalized position in field")
ax.set_ylabel("Mean theta phase (deg)")
ax.set_title("Population phase advance\n(mean $\\pm$ s.e.m. across fields)")

# (c) slope distribution vs shuffle
ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(-720, 720, 49)
ax.hist(np.degrees(null_slope), bins=bins, density=True, color="0.75", label="phase-shuffled")
ax.hist(np.degrees(slopes), bins=bins, density=True, histtype="step", lw=2, color="tab:red", label="observed")
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Phase-position slope (deg per field traversal)")
ax.set_ylabel("Density")
ax.set_title("Slopes are negative\n(median %.0f$\\degree$ per field)" % np.degrees(np.median(slopes)))
ax.legend(fontsize=9)

# (d) rho distribution vs shuffle
ax = fig.add_subplot(gs[1, 0])
bins = np.linspace(-0.8, 0.8, 41)
ax.hist(null_rho, bins=bins, density=True, color="0.75", label="phase-shuffled")
ax.hist(rhos, bins=bins, density=True, histtype="step", lw=2, color="tab:red", label="observed")
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Circular-linear correlation $\\rho$")
ax.set_ylabel("Density")
ax.set_title("Phase-position correlation")
ax.legend(fontsize=9)

# (e) slope vs field width
ax = fig.add_subplot(gs[1, 1])
w = np.array([f["width"] for f in fields])
ax.scatter(w[~sig], np.degrees(slopes[~sig]), s=18, color="0.7", label="n.s.")
ax.scatter(w[sig], np.degrees(slopes[sig]), s=18, color="tab:red", label="p<0.05")
ax.axhline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Field width (cm)")
ax.set_ylabel("Slope (deg per field traversal)")
ax.set_title("Precession slope vs. field size")
ax.legend(fontsize=9)

# (f) slope in deg/cm
ax = fig.add_subplot(gs[1, 2])
per_cm = np.degrees(slopes) / w
ax.hist(per_cm, bins=25, color="tab:blue")
ax.axvline(0, color="k", ls="--")
ax.axvline(np.median(per_cm), color="tab:red", lw=2)
ax.set_xlabel("Slope (deg/cm)")
ax.set_ylabel("Number of fields")
ax.set_title("Phase advance per cm\n(median %.1f deg/cm)" % np.median(per_cm))

fig.suptitle(
    "Population summary of theta phase precession, %s (%d CA1 place fields)"
    % (res["session"], len(fields)),
    fontsize=13,
)
fig.savefig("fig06_precession_population.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------------
# Figure 7: single-pass structure for the best example cell
# ---------------------------------------------------------------------------------
f = fields[int(order[0])]
ep = res["dir_ep"][f["direction"]]
lap_ep = res["laps"][res["laps"].direction == f["direction"]]
unit = res["pyr"][f["unit"]]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
n_shown, n_passes = 0, 12
for i, (a, b) in enumerate(zip(lap_ep.start, lap_ep.end)):
    st = unit.get(a, b)
    if len(st) < 4:
        continue
    p = res["position"].restrict(ep).interpolate(st).values
    ph = np.degrees(st.value_from(theta_phase).values)
    m = (p >= f["lo"]) & (p <= f["hi"]) & np.isfinite(p)
    if m.sum() < 4:
        continue
    # distance travelled into the field, so the x axis always runs entry -> exit
    dist = (p[m] - f["lo"]) if f["direction"] == "right" else (f["hi"] - p[m])
    color = plt.cm.viridis(n_shown / n_passes)
    axes[0].plot(dist, ph[m], "o", ms=6, color=color, alpha=0.85)
    axes[0].plot(dist, ph[m] + 360, "o", ms=6, color=color, alpha=0.85)
    n_shown += 1
    if n_shown >= n_passes:
        break
axes[0].set_ylim(0, 720)
axes[0].set_yticks([0, 180, 360, 540, 720])
axes[0].set_xlabel("Distance into field (cm)")
axes[0].set_ylabel("Theta phase (deg)")
axes[0].set_title(
    "Unit %d (%sward): %d individual passes\n(colour = pass number)"
    % (f["unit"], f["direction"], n_shown)
)

axes[1].plot(f["x"], np.degrees(f["phase"]), ".", ms=4, color="0.35")
axes[1].plot(f["x"], np.degrees(f["phase"]) + 360, ".", ms=4, color="0.35")
xx = np.linspace(0, 1, 100)
yy = np.degrees(f["slope"] * xx + f["phi0"]) % 360
for shift in (-360, 0, 360, 720):
    axes[1].plot(xx, yy + shift, "r-", lw=2)
axes[1].set_ylim(0, 720)
axes[1].set_yticks([0, 180, 360, 540, 720])
axes[1].set_xlabel("Normalized position in field")
axes[1].set_ylabel("Theta phase (deg)")
axes[1].set_title("All passes pooled: %.0f$\\degree$ advance, $\\rho$=%.2f" % (np.degrees(f["slope"]), f["rho"]))

fig.tight_layout()
fig.savefig("fig07_single_passes.png", dpi=150)
plt.close(fig)
print("done")
