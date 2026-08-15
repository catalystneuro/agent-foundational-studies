"""Stage 6: theta phase precession, an independent check that the phase
assignment is behaviourally meaningful and not a filtering artefact."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from tqdm import tqdm

import theta_utils as tu

SESSION = "Achilles-10252013"
BEST_CH = 117
NBINS_POS = 40
MIN_FIELD_SPIKES = 60

h = tu.open_session(tu.SESSIONS[SESSION])
meta = tu.load_metadata(h)
units = meta["units"]
maze = meta["maze_epoch"]

pos = meta["position"].restrict(maze)
# tracking has short dropouts; keep only valid samples for the spatial analysis
x_raw = pos["x"]
valid = np.isfinite(x_raw.d)
x = nap.Tsd(t=x_raw.t[valid], d=x_raw.d[valid])
print(f"position: {valid.mean()*100:.0f}% of samples tracked")
speed = tu.compute_speed(pos, rate=meta["position_rate"])
run = speed.threshold(10.0).time_support.drop_short_intervals(1.0)

# split running into rightward / leftward traversals
# smooth the derivative over ~0.5 s so that tracking jitter does not chop the
# traversals into fragments
w = int(0.5 * meta["position_rate"]) | 1
kern = np.ones(w) / w
dx = np.convolve(np.gradient(x.d, x.t), kern, mode="same")
direction = nap.Tsd(t=x.t, d=np.sign(dx))
runs = {
    "rightward": direction.threshold(0.5).time_support.intersect(run).drop_short_intervals(1.0),
    "leftward": direction.threshold(-0.5, "below").time_support.intersect(run).drop_short_intervals(1.0),
}
print({k: f"{v.tot_length():.0f}s" for k, v in runs.items()})

phase, amp, raw, kept = tu.phase_by_interval(h, BEST_CH, run, max_total=1500.0)
lookup = tu.PhaseLookup(phase)


def circ_lin_corr(phi, xnorm, n_perm=500, rng=None):
    """Kempter et al. (2012) circular-linear correlation with a permutation p."""
    rng = rng or np.random.default_rng(0)
    slopes = np.linspace(-3, 3, 601)  # cycles per normalised field width

    def best_fit(ph, xs):
        R = np.abs(np.mean(np.exp(1j * (ph - 2 * np.pi * slopes[:, None] * xs)), axis=1))
        k = int(np.argmax(R))
        return slopes[k], np.angle(np.mean(np.exp(1j * (ph - 2 * np.pi * slopes[k] * xs))))

    a, phi0 = best_fit(phi, xnorm)
    theta_hat = np.mod(2 * np.pi * a * xnorm, 2 * np.pi)
    # circular-circular correlation between observed and predicted phase
    def cc(p1, p2):
        s1 = np.sin(p1 - tu.circ_mean(p1)); s2 = np.sin(p2 - tu.circ_mean(p2))
        return np.sum(s1 * s2) / np.sqrt(np.sum(s1**2) * np.sum(s2**2))
    rho = cc(phi, theta_hat)
    null = np.empty(n_perm)
    for i in range(n_perm):
        xs = rng.permutation(xnorm)
        aa, _ = best_fit(phi, xs)
        null[i] = np.abs(cc(phi, np.mod(2 * np.pi * aa * xs, 2 * np.pi)))
    p = (np.sum(null >= np.abs(rho)) + 1) / (n_perm + 1)
    return a, phi0, rho, p


rng = np.random.default_rng(3)
rows, fields = [], {}
for dname, ep in runs.items():
    tc = nap.compute_1d_tuning_curves(units, x, nb_bins=NBINS_POS, ep=ep)
    centers = tc.index.values
    for uid in tqdm(units.index, desc=f"place fields {dname}", leave=False):
        if units.cell_type.loc[uid] != "excitatory":
            continue
        curve = tc[uid].values
        if np.nanmax(curve) < 2.0:
            continue
        peak = int(np.nanargmax(curve))
        thr = 0.3 * curve[peak]
        lo = peak
        while lo > 0 and curve[lo - 1] >= thr:
            lo -= 1
        hi = peak
        while hi < len(curve) - 1 and curve[hi + 1] >= thr:
            hi += 1
        if hi - lo < 3:
            continue
        x0, x1 = centers[lo], centers[hi]
        in_field = nap.Tsd(t=x.t, d=((x.d >= x0) & (x.d <= x1)).astype(float))
        fep = in_field.threshold(0.5).time_support.intersect(ep)
        st = units[uid].restrict(fep).t
        ph = lookup(st)
        if len(ph) < MIN_FIELD_SPIKES:
            continue
        xs = np.interp(st, x.t, x.d)
        xn = (xs - x0) / (x1 - x0)
        if dname == "leftward":
            xn = 1 - xn
        keep = np.isfinite(xn) & (xn >= 0) & (xn <= 1)
        xn = xn[keep]
        ph2 = lookup(st[keep])
        if len(ph2) < MIN_FIELD_SPIKES or len(ph2) != len(xn):
            continue
        a, phi0, rho, p = circ_lin_corr(ph2, xn, rng=rng)
        rows.append(dict(unit=int(uid), direction=dname, n=len(ph2), slope=a,
                         phi0=phi0, rho=rho, p=p, peak_rate=curve[peak],
                         field=(x0, x1)))
        fields[(int(uid), dname)] = (xn, ph2)

pre = pd.DataFrame(rows)
pre["sig"] = tu.benjamini_hochberg(pre["p"].values)
pre.drop(columns=["field"]).to_csv("phase_precession.csv", index=False)
neg = pre.sig & (pre.slope < 0)
print(f"\nplace fields analysed: {len(pre)}")
print(f"significant circular-linear correlation (FDR q<0.05): {pre.sig.sum()} "
      f"({100*pre.sig.mean():.0f}%)")
print(f"  of which negative slope (phase precession): {neg.sum()} "
      f"({100*neg.sum()/max(pre.sig.sum(),1):.0f}% of significant)")
print(f"median slope of significant fields: {pre.loc[pre.sig,'slope'].median():.2f} "
      "theta cycles per field traversal")

# ------------------------------------------------------------------- figure
best = pre[pre.sig & (pre.slope < 0)].nlargest(4, "n")
fig = plt.figure(figsize=(14.5, 7))
gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.42)
for k, (_, r) in enumerate(best.iterrows()):
    ax = fig.add_subplot(gs[k // 2, k % 2])
    xn, ph = fields[(int(r.unit), r.direction)]
    ax.scatter(np.concatenate([xn, xn]),
               np.degrees(np.concatenate([ph, ph + 2 * np.pi])), s=6,
               color="C0", alpha=0.55)
    xx = np.linspace(0, 1, 100)
    for off in (0, 360, 720):
        ax.plot(xx, np.degrees(2 * np.pi * r.slope * xx + r.phi0) + off,
                color="C3", lw=1.6)
    ax.set_ylim(0, 720); ax.set_xlim(0, 1)
    ax.set_yticks(np.arange(0, 721, 180))
    ax.set_title(f"unit {int(r.unit)}, {r.direction}, n={int(r.n)}\n"
                 f"slope={r.slope:.2f} cyc/field\n"
                 f"rho={r.rho:.2f}, p={r.p:.3f}", fontsize=8.5)
    if k >= 2:
        ax.set_xlabel("normalised position in field")
    ax.set_ylabel("theta phase (deg)")

ax = fig.add_subplot(gs[0, 2])
ax.hist(pre.loc[pre.sig, "slope"], bins=np.arange(-3, 3.01, 0.25), color="C0")
ax.axvline(0, color="k", ls="--")
ax.set_xlabel("slope (theta cycles per field traversal)")
ax.set_ylabel("fields"); ax.set_title("precession slopes (significant fields)")

ax = fig.add_subplot(gs[0, 3])
ax.hist(pre["rho"], bins=20, color="0.6", label="all fields")
ax.hist(pre.loc[pre.sig, "rho"], bins=20, color="C0", label="significant")
ax.set_xlabel("circular-linear correlation"); ax.set_ylabel("fields")
ax.legend(fontsize=8); ax.set_title("correlation strength")

ax = fig.add_subplot(gs[1, 2:])
allxn = np.concatenate([fields[(int(r.unit), r.direction)][0]
                        for _, r in pre[pre.sig & (pre.slope < 0)].iterrows()])
allph = np.concatenate([fields[(int(r.unit), r.direction)][1]
                        for _, r in pre[pre.sig & (pre.slope < 0)].iterrows()])
H, xe, ye = np.histogram2d(allxn, np.degrees(allph), bins=[20, 18],
                           range=[[0, 1], [0, 360]])
H = H / H.sum(axis=1, keepdims=True)
ax.imshow(np.tile(H.T, (2, 1)), aspect="auto", origin="lower", cmap="magma",
          extent=[0, 1, 0, 720], interpolation="bilinear")
ax.set_xlabel("normalised position in field")
ax.set_ylabel("theta phase (deg)")
ax.set_yticks(np.arange(0, 721, 180))
ax.set_title(f"pooled spike density, {int(neg.sum())} precessing fields")
for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"{SESSION}: theta phase precession on the linear track", y=0.98)
fig.savefig("fig09_phase_precession.png", dpi=150, bbox_inches="tight")
print("saved fig09_phase_precession.png")
