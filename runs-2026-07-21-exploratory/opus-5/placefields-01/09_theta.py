"""Theta phase precession: the temporal side of the place code.

Beyond firing at a preferred location, a CA1 place cell fires at a systematically
earlier phase of the ongoing theta rhythm as the animal advances through the
field.  This script pulls the CA1 LFP for the maze epoch, extracts theta phase,
and relates spike phase to the animal's position within each place field.
"""
import pickle
import warnings

warnings.simplefilter("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy.signal import hilbert

import pf_lib as pf

SESSION = "Achilles_10252013"
THETA_BAND = (6.0, 10.0)
COL = {"right": "#1f77b4", "left": "#d62728"}

s = pf.prepare(SESSION, pf.session_urls()[SESSION])
pos, L, eps, pyr, maze = s["pos"], s["track_length"], s["eps"], s["pyr"], s["maze"]
nwbfile = s["nwbfile"]

# ------------------------------------------------------- pick an LFP channel
# The file has 128 channels at 1250 Hz; only the maze epoch of a few candidate
# channels is streamed, and the one with the strongest theta is kept.
es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
fs_lfp = float(es.rate)
i0 = int((float(maze.start[0]) - es.starting_time) * fs_lfp)
i1 = int((float(maze.end[0]) - es.starting_time) * fs_lfp)
elec = nwbfile.electrodes.to_dataframe()
cands = [int(g.index[len(g) // 2]) for _, g in elec.groupby("group_name")]
print(f"LFP {es.data.shape} at {fs_lfp} Hz; testing channels {cands}")

t_lfp = es.starting_time + np.arange(i0, i1) / fs_lfp
best, best_ratio = None, -np.inf
for ch in cands:
    x = np.asarray(es.data[i0:i1, ch], dtype=float) * es.conversion
    sig = nap.Tsd(t=t_lfp, d=x)
    psd = nap.compute_mean_power_spectral_density(sig, 1.0, fs=fs_lfp, ep=eps["run"])
    f = np.asarray(psd.index)
    p = np.abs(np.asarray(psd.values).ravel()) ** 2
    theta = p[(f >= THETA_BAND[0]) & (f <= THETA_BAND[1])].mean()
    delta = p[(f >= 1) & (f <= 4)].mean()
    ratio = theta / delta
    print(f"  channel {ch:3d} ({elec.group_name[ch]}): theta/delta = {ratio:.2f}")
    if ratio > best_ratio:
        best, best_ratio, best_x = ch, ratio, x
print(f"using channel {best} (theta/delta = {best_ratio:.2f})")

lfp = nap.Tsd(t=t_lfp, d=best_x)
theta_tsd = nap.apply_bandpass_filter(lfp, THETA_BAND, fs=fs_lfp)
analytic = hilbert(np.asarray(theta_tsd.values))
phase = nap.Tsd(t=t_lfp, d=np.mod(np.angle(analytic), 2 * np.pi))
amp = nap.Tsd(t=t_lfp, d=np.abs(analytic))

# ------------------------------------------------------------ precession
import os, pickle as _pk

rng = np.random.default_rng(0)
if os.path.exists("single_session.pkl"):
    _d = _pk.load(open("single_session.pkl", "rb"))
    mets, tcs = _d["metrics"], _d["tc"]
    print("reusing place-cell metrics from single_session.pkl")
else:
    mets, tcs = {}, {}
    for d in pf.DIRS:
        out = pf.direction_metrics(pyr, pos, eps[d], L, rng=rng)
        mets[d] = out["metrics"]
        tcs[d] = out["tc_smooth"]
centers = np.asarray(tcs["right"].index)


def circ_lin_corr(x, ph, n_slopes=1601):
    """Circular-linear regression and correlation, following Kempter et al. (2012).

    The slope is the value of `a` that maximizes the mean resultant length of
    exp(i(phase - a*x)); rho is then the circular correlation between the
    observed phases and the phases predicted by that fit. Returns
    (rho, slope in rad per unit x, phase offset).
    """
    if len(x) < 20:
        return np.nan, np.nan, np.nan
    slopes = np.linspace(-8 * np.pi, 8 * np.pi, n_slopes)
    # |mean(exp(i(ph - a x)))| for every candidate slope, evaluated as a matrix
    # product of the unit phasors against exp(-i a x).
    z = np.exp(1j * ph)
    R = np.abs(np.exp(-1j * np.outer(slopes, x)) @ z) / len(x)
    slope = float(slopes[int(np.argmax(R))])
    phi0 = float(np.angle(np.exp(1j * (ph - slope * x)).mean()))

    theta = np.mod(slope * x, 2 * np.pi)
    ph_bar = np.angle(np.exp(1j * ph).mean())
    th_bar = np.angle(np.exp(1j * theta).mean())
    sp, st = np.sin(ph - ph_bar), np.sin(theta - th_bar)
    den = np.sqrt(np.sum(sp**2) * np.sum(st**2))
    rho = float(np.sum(sp * st) / den) if den > 0 else np.nan
    return rho, slope, phi0


records = []
pooled_x, pooled_ph = [], []
for d in pf.DIRS:
    m = mets[d]
    for uid in pyr.index:
        if not m.is_place_cell[uid]:
            continue
        pk, w = m.peak_pos[uid], m.width[uid]
        lo, hi = pk - w, pk + w  # a window of +/- one full width around the peak
        st = np.asarray(pyr[uid].restrict(eps[d]).t)
        if len(st) == 0:
            continue
        sp = np.interp(st, pos.t, pos.values)
        inside = (sp >= lo) & (sp <= hi)
        st, sp = st[inside], sp[inside]
        if len(st) < 40:
            continue
        ph = np.interp(st, phase.t, np.unwrap(phase.values))
        ph = np.mod(ph, 2 * np.pi)
        xnorm = (sp - lo) / (hi - lo)  # 0 at field entry, 1 at field exit
        if d == "left":
            xnorm = 1 - xnorm  # travel direction, not track direction
        rho, slope, phi0 = circ_lin_corr(xnorm, ph)
        records.append(
            {"unit": uid, "direction": d, "n_spikes": len(st), "rho": rho,
             "slope_cycles": slope / (2 * np.pi), "peak_pos": pk, "width": w}
        )
        pooled_x.append(xnorm)
        pooled_ph.append(ph)

prec = pd.DataFrame(records)
pooled_x = np.concatenate(pooled_x)
pooled_ph = np.concatenate(pooled_ph)
rho_all, slope_all, phi0_all = circ_lin_corr(pooled_x, pooled_ph)

# permutation test on the pooled slope: shuffle the position labels
nperm = 200
null_rho = np.zeros(nperm)
for k in range(nperm):
    null_rho[k] = circ_lin_corr(rng.permutation(pooled_x), pooled_ph, n_slopes=401)[0]
p_perm = (np.sum(null_rho <= rho_all) + 1) / (nperm + 1)

print(f"\n{len(prec)} place-field passes analysed, {len(pooled_x)} in-field spikes")
print(f"pooled circular-linear rho = {rho_all:.3f}, "
      f"slope = {slope_all / (2 * np.pi):.2f} theta cycles per field, "
      f"permutation p = {p_perm:.4f}")
print(f"per-field slope: median {prec.slope_cycles.median():.2f} cycles, "
      f"{100 * (prec.slope_cycles < 0).mean():.0f}% negative (precession)")

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.3, 1.1], hspace=0.45, wspace=0.3)

# raw LFP with the theta filter
ax = fig.add_subplot(gs[0, :])
w = nap.IntervalSet(start=float(eps["right"].start[6]), end=float(eps["right"].end[6]))
lr, tr = lfp.restrict(w), theta_tsd.restrict(w)
ax.plot(lr.t, lr.values * 1e3, color="0.6", lw=0.8, label="raw LFP")
ax.plot(tr.t, tr.values * 1e3, color="k", lw=1.5, label=f"{THETA_BAND[0]:.0f}-{THETA_BAND[1]:.0f} Hz")
ax2 = ax.twinx()
pr = phase.restrict(w)
ax2.plot(pr.t, pr.values, color="#e377c2", lw=0.7, alpha=0.8)
ax2.set_ylabel("theta phase (rad)", color="#e377c2")
ax2.tick_params(axis="y", colors="#e377c2")
ax.set_xlabel("time (s)")
ax.set_ylabel("LFP (mV)")
ax.set_title(f"{SESSION}: CA1 LFP on channel {best} during one traversal, theta is prominent "
             "while the animal runs", fontsize=11)
ax.legend(fontsize=8, loc="upper right")

# power spectrum during running vs during immobility in the maze epoch
ax = fig.add_subplot(gs[1, 0])
still = maze.set_diff(eps["run"])
for ep, lab, c in [(eps["run"], "running", "crimson"), (still, "immobile", "0.5")]:
    psd = nap.compute_mean_power_spectral_density(lfp, 1.0, fs=fs_lfp, ep=ep)
    f = np.asarray(psd.index)
    p = np.abs(np.asarray(psd.values).ravel()) ** 2
    m = (f > 0.5) & (f < 60)
    ax.semilogy(f[m], p[m], color=c, lw=1.2, label=lab)
ax.axvspan(*THETA_BAND, color="gold", alpha=0.25, lw=0)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("power (a.u.)")
ax.set_title("LFP power spectrum", fontsize=11)
ax.legend(fontsize=8)

# three example fields
best3 = prec.reindex(prec.rho.abs().sort_values(ascending=False).index).head(3)
for k, (_, row) in enumerate(best3.iterrows()):
    ax = fig.add_subplot(gs[1, 1 + k] if k < 2 else gs[2, 0])
    uid, d = int(row.unit), row.direction
    pk, wdt = row.peak_pos, row.width
    lo, hi = pk - wdt, pk + wdt
    st = np.asarray(pyr[uid].restrict(eps[d]).t)
    sp = np.interp(st, pos.t, pos.values)
    ins = (sp >= lo) & (sp <= hi)
    st, sp = st[ins], sp[ins]
    ph = np.mod(np.interp(st, phase.t, np.unwrap(phase.values)), 2 * np.pi)
    xn = (sp - lo) / (hi - lo)
    if d == "left":
        xn = 1 - xn
    for rep in (0, 1):
        ax.plot(xn, ph + rep * 2 * np.pi, ".", color=COL[d], ms=3, alpha=0.7)
    _, sl, p0 = circ_lin_corr(xn, ph)
    xx = np.linspace(0, 1, 200)
    for rep in (0, 1, 2):
        yy = np.mod(sl * xx + p0, 2 * np.pi) + rep * 2 * np.pi
        yy[np.abs(np.diff(yy, prepend=yy[0])) > np.pi] = np.nan
        ax.plot(xx, yy, "k-", lw=1.5)
    ax.set_xlabel("position within field (entry -> exit)")
    ax.set_ylabel("theta phase (rad)")
    ax.set_ylim(0, 4 * np.pi)
    ax.set_yticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
    ax.set_yticklabels(["0", "$\\pi$", "$2\\pi$", "$3\\pi$", "$4\\pi$"])
    ax.set_title(f"unit {uid}, {d}ward\n"
                 f"rho = {row.rho:.2f}, {row.slope_cycles:.2f} cycles/field", fontsize=10)

# pooled phase-position density
ax = fig.add_subplot(gs[2, 1])
H, xe, ye = np.histogram2d(pooled_x, pooled_ph, [np.linspace(0, 1, 21),
                                                 np.linspace(0, 2 * np.pi, 25)])
H = H / H.sum(axis=1, keepdims=True)
H2 = np.vstack([H.T, H.T])
im = ax.imshow(H2, aspect="auto", origin="lower", extent=[0, 1, 0, 4 * np.pi], cmap="magma")
xx = np.linspace(0, 1, 100)
for rep in (0, 1, 2):
    ax.plot(xx, np.mod(slope_all * xx + phi0_all, 2 * np.pi) + rep * 2 * np.pi, "c-", lw=2)
ax.set_ylim(0, 4 * np.pi)
ax.set_yticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
ax.set_yticklabels(["0", "$\\pi$", "$2\\pi$", "$3\\pi$", "$4\\pi$"])
ax.set_xlabel("position within field (entry -> exit)")
ax.set_ylabel("theta phase (rad)")
ax.set_title(f"All {len(prec)} fields pooled ({len(pooled_x)} spikes)\n"
             f"rho = {rho_all:.2f}, slope = {slope_all / (2 * np.pi):.2f} cycles, "
             f"p = {p_perm:.3f}", fontsize=10)
plt.colorbar(im, ax=ax, label="P(phase | position)", fraction=0.046)

# per-field slope distribution
ax = fig.add_subplot(gs[2, 2])
ax.hist(prec.slope_cycles, np.linspace(-3, 3, 37), color="#4c72b0", edgecolor="w")
ax.axvline(0, color="k", lw=1)
ax.axvline(prec.slope_cycles.median(), color="crimson", ls="--", lw=1.5)
ax.set_xlabel("phase-position slope (theta cycles per field)")
ax.set_ylabel("number of fields")
ax.set_title(f"Slopes are negative in {100 * (prec.slope_cycles < 0).mean():.0f}% of fields\n"
             f"(median {prec.slope_cycles.median():.2f} cycles)", fontsize=10)

fig.suptitle(f"{SESSION}: theta phase precession within CA1 place fields", fontsize=13)
fig.savefig("fig08_theta_precession.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig08 done")

with open("theta.pkl", "wb") as fh:
    pickle.dump(
        {"prec": prec, "rho_all": rho_all, "slope_all": slope_all, "p_perm": p_perm,
         "channel": best, "theta_delta": best_ratio}, fh
    )
s["io"].close()
