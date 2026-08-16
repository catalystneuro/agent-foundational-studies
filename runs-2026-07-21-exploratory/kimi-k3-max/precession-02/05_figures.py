# 05_figures.py — precession figures: examples, single traversals, population stats, pooled
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from achilles_loader import open_nwb

TWO_PI = 2 * np.pi
C = np.load("cache_preproc.npz")
F = np.load("cache_fields.npz")
P = np.load("cache_precession.npz")
rec = json.load(open("precession_results.json"))
pos_t, lin_v = C["pos_t"], C["lin"]
bouts = np.column_stack([C["bout_starts"], C["bout_ends"]])
bout_dir = C["bout_dir"]
maze_start = float(C["maze_start"])
lfp_t, theta_phase = C["lfp_t"], C["theta_phase"]
LFP_FS = 1.0 / np.median(np.diff(lfp_t))
cos_ph, sin_ph = np.cos(theta_phase), np.sin(theta_phase)

nwbfile, nwb, h5 = open_nwb()
units = nwb["units"]

SLOPES = np.linspace(-6 * np.pi, 6 * np.pi, 4321)

def theta_phase_at(t):
    x = (t - maze_start) * LFP_FS
    i0 = np.clip(np.floor(x).astype(int), 0, len(theta_phase) - 2)
    frac = x - np.floor(x)
    c = cos_ph[i0] * (1 - frac) + cos_ph[i0 + 1] * frac
    s = sin_ph[i0] * (1 - frac) + sin_ph[i0 + 1] * frac
    return np.mod(np.arctan2(s, c), TWO_PI)

def spikes_lin_pos(spk_t):
    idx = np.clip(np.searchsorted(pos_t, spk_t) - 1, 0, len(pos_t) - 1)
    return lin_v[idx]

def in_bouts(t, d_bouts):
    starts, ends = d_bouts[:, 0], d_bouts[:, 1]
    idx = np.searchsorted(starts, t, side="right") - 1
    inb = np.zeros(len(t), dtype=bool)
    ok = idx >= 0
    inb[ok] = t[ok] <= ends[idx[ok]]
    return inb

def fit_line(xn, ph):
    z = np.exp(1j * (ph[:, None] - xn[:, None] * SLOPES[None, :]))
    R = np.abs(z.mean(axis=0))
    m = SLOPES[int(np.argmax(R))]
    phi0 = np.angle(np.mean(np.exp(1j * (ph - m * xn))))
    return m, phi0

# ---------- fig 3: example cells ----------
# pick 6 significant-negative cells: strong correlations with plenty of spikes
neg = [r for r in rec if r["p_val"] < 0.05 and r["signed_r"] < 0 and r["n_spikes"] >= 100]
neg.sort(key=lambda r: r["signed_r"])
picks = [neg[i] for i in [0, 3, 6, 9, 12, 15] if i < len(neg)][:6]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, r in zip(axes.ravel(), picks):
    u, d = r["uid"], r["d"]
    xn, ph = P[f"spk_{u}_{d}"].T
    m, phi0 = fit_line(xn, ph)
    # plot duplicated over two cycles
    ax.scatter(xn, ph / TWO_PI, s=4, c="k", alpha=0.5)
    ax.scatter(xn, ph / TWO_PI + 1, s=4, c="k", alpha=0.5)
    xs = np.linspace(0, 1, 50)
    yfit = (phi0 + m * xs) / TWO_PI
    yfit_wrapped = np.mod(yfit, 1.0)
    # plot fit in both cycles, splitting at wraps
    for off in (0, 1):
        y = yfit_wrapped + off
        jumps = np.where(np.abs(np.diff(y)) > 0.5)[0]
        y_seg = y.copy()
        y_seg[jumps] = np.nan
        ax.plot(xs, y_seg, color="tab:red", lw=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(["0", "180", "360", "540", "720"])
    ax.set_title(f"unit {u}, dir {'+' if d > 0 else '-'}: r={r['signed_r']:.2f}, "
                 f"p={r['p_val']:.1e}, slope={r['slope_cycles']:.2f} cyc/field", fontsize=10)
    ax.set_xlabel("field progress (0=entry, 1=exit)")
    ax.set_ylabel("theta phase (deg)")
fig.suptitle("Theta phase precession: example place cells (spikes duplicated over 2 cycles)", y=0.995)
fig.tight_layout()
fig.savefig("fig03_example_precession.png", dpi=150)
print("saved fig03_example_precession.png")

# ---------- fig 4: single traversals for a clean showcase cell ----------
# choose a cell with many traversals and a strongly negative per-traversal mean
cands = [r for r in rec if r["p_val"] < 0.05 and r["signed_r"] < 0
         and r["trav_n"] >= 10 and np.isfinite(r["trav_mean"])]
cands.sort(key=lambda r: r["trav_mean"])
best = cands[0]
u, d = best["uid"], best["d"]
lo, hi = F[f"field_{u}_{d}"][:2]
spk_t = np.asarray(units[u].t)
d_bouts = bouts[bout_dir == d]
mask = in_bouts(spk_t, d_bouts)
st = spk_t[mask]
sp = spikes_lin_pos(st)
ok = np.isfinite(sp) & (sp >= lo) & (sp <= hi)
st, sp = st[ok], sp[ok]
ph = theta_phase_at(st)
xn_all = (sp - lo) / (hi - lo) if d == 1 else (hi - sp) / (hi - lo)

# collect traversals with >= 5 spikes
travs = []
for s, e in d_bouts:
    m = (st >= s) & (st <= e)
    if m.sum() >= 5 and xn_all[m].max() - xn_all[m].min() > 0.4:
        travs.append((s, xn_all[m], ph[m]))
travs = travs[:6]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, (s, xn_t, ph_t) in zip(axes.ravel(), travs):
    ax.scatter(xn_t, ph_t / TWO_PI, s=30, c="k")
    ax.scatter(xn_t, ph_t / TWO_PI + 1, s=30, c="k")
    if len(xn_t) >= 4:
        m, phi0 = fit_line(xn_t, ph_t)
        xs = np.linspace(xn_t.min(), xn_t.max(), 30)
        yfit = np.mod((phi0 + m * xs) / TWO_PI, 1.0)
        for off in (0, 1):
            y = yfit + off
            yseg = y.copy()
            yseg[np.where(np.abs(np.diff(y)) > 0.5)[0]] = np.nan
            ax.plot(xs, yseg, color="tab:red", lw=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(["0", "180", "360", "540", "720"])
    ax.set_title(f"traversal at t={s:.0f} s ({len(xn_t)} spikes)", fontsize=10)
    ax.set_xlabel("field progress")
    ax.set_ylabel("theta phase (deg)")
fig.suptitle(f"Single-traversal precession: unit {u}, dir {'+' if d > 0 else '-'}", y=0.995)
fig.tight_layout()
fig.savefig("fig04_single_traversals.png", dpi=150)
print("saved fig04_single_traversals.png")

# ---------- fig 5: population stats ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax = axes[0, 0]
sr = np.array([r["signed_r"] for r in rec])
pv = np.array([r["p_val"] for r in rec])
ax.hist(sr[pv < 0.05], bins=25, alpha=0.8, label=f"p<0.05 (n={np.sum(pv < 0.05)})", color="tab:blue")
ax.hist(sr[pv >= 0.05], bins=25, alpha=0.6, label=f"n.s. (n={np.sum(pv >= 0.05)})", color="gray")
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.median(sr), color="tab:red", ls="--", lw=1,
           label=f"median {np.median(sr):.2f}")
ax.set_xlabel("signed circular-linear correlation")
ax.set_ylabel("cell-directions")
ax.set_title("Phase-position correlation (pooled spikes)")
ax.legend(fontsize=9)

ax = axes[0, 1]
ax.scatter(sr, -np.log10(pv), c=np.where(pv < 0.05, "tab:blue", "gray"), s=18, alpha=0.7)
ax.axhline(-np.log10(0.05), color="k", ls="--", lw=0.8)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("signed circular-linear correlation")
ax.set_ylabel("-log10 p (phase-permutation shuffle)")
n_sig_neg = np.sum((pv < 0.05) & (sr < 0))
n_sig_pos = np.sum((pv < 0.05) & (sr > 0))
ax.set_title(f"{n_sig_neg} precessing vs {n_sig_pos} positive (of {len(rec)})")

ax = axes[1, 0]
sl = np.array([r["slope_cycles"] for r in rec])
ax.hist(sl[pv < 0.05], bins=25, alpha=0.8, color="tab:blue", label="p<0.05")
ax.hist(sl[pv >= 0.05], bins=25, alpha=0.6, color="gray", label="n.s.")
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.median(sl), color="tab:red", ls="--", lw=1, label=f"median {np.median(sl):.2f}")
ax.set_xlabel("precession slope (theta cycles per field)")
ax.set_ylabel("cell-directions")
ax.set_title("Precession slope")
ax.legend(fontsize=9)

ax = axes[1, 1]
tm = np.array([r["trav_mean"] for r in rec if np.isfinite(r["trav_mean"])])
tn = np.array([r["trav_n"] for r in rec if np.isfinite(r["trav_mean"])])
ax.hist(tm, bins=25, color="tab:green", alpha=0.8)
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.median(tm), color="tab:red", ls="--", lw=1, label=f"median {np.median(tm):.2f}")
ax.set_xlabel("mean per-traversal signed correlation")
ax.set_ylabel("cell-directions")
ax.set_title(f"Single-pass precession ({int(tn.sum())} traversals);\n"
             f"{np.mean(tm < 0):.0%} of {len(tm)} cell-directions negative", fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("fig05_population_stats.png", dpi=150)
print("saved fig05_population_stats.png")

# ---------- fig 6b: raw demonstration on one traversal ----------
# pick the traversal of the showcase cell with the most spikes
trav_spikes = []
for s, e in d_bouts:
    m = (st >= s) & (st <= e)
    if m.sum() >= 8:
        trav_spikes.append((s, e, m.sum()))
trav_spikes.sort(key=lambda x: -x[2])
s0, e0, _ = trav_spikes[0]
pad = 0.35
m_lfp = (lfp_t >= s0 - pad) & (lfp_t <= e0 + pad)
m_pos = (pos_t >= s0 - pad) & (pos_t <= e0 + pad)
m_spk = (st >= s0 - pad) & (st <= e0 + pad)

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                         gridspec_kw=dict(height_ratios=[2.2, 1]))
ax = axes[0]
ax.plot(lfp_t[m_lfp], C["lfp_filt"][m_lfp], color="tab:purple", lw=1.2,
        label="theta-filtered LFP (6-12 Hz)")
for t_spk in st[m_spk]:
    ax.axvline(t_spk, color="k", alpha=0.55, lw=0.9)
ax.set_ylabel("LFP (uV)")
ax.legend(loc="upper right", fontsize=9)
ax.set_title(f"Unit {u} spikes (vertical lines) against theta during one field traversal "
             f"(t = {s0:.2f}-{e0:.2f} s)")

ax = axes[1]
ax.plot(pos_t[m_pos], lin_v[m_pos], color="tab:green", lw=1.5)
for t_spk, p_spk in zip(st[m_spk], sp[m_spk]):
    ax.plot(t_spk, p_spk, "k|", ms=8)
ax.axhspan(lo, hi, color="tab:red", alpha=0.12, label="place field")
ax.set_ylabel("linearized pos (m)")
ax.set_xlabel("time (s)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("fig07_raw_traversal.png", dpi=150)
print("saved fig07_raw_traversal.png")

# ---------- fig 6: pooled population ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, subset, ttl in zip(
        axes,
        [[r for r in rec if r["p_val"] < 0.05 and r["signed_r"] < 0], rec],
        ["significant precessing cells", "all place cells"]):
    xs_all, ph_all = [], []
    for r in subset:
        xn, ph = P[f"spk_{r['uid']}_{r['d']}"].T
        xs_all.append(xn)
        ph_all.append(ph)
    xs_all = np.concatenate(xs_all)
    ph_all = np.concatenate(ph_all) / TWO_PI
    H, xe, ye = np.histogram2d(xs_all, ph_all % 1.0, bins=[40, 36], range=[[0, 1], [0, 1]])
    H2 = np.vstack([H, H])  # two cycles
    ax.imshow(H2.T, origin="lower", aspect="auto", extent=[0, 1, 0, 2], cmap="magma")
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(["0", "180", "360", "540", "720"])
    ax.set_xlabel("field progress (0=entry, 1=exit)")
    ax.set_ylabel("theta phase (deg)")
    ax.set_title(f"Pooled spikes, {ttl} (n={len(subset)}, {len(xs_all)} spikes)")
fig.tight_layout()
fig.savefig("fig06_pooled_population.png", dpi=150)
print("saved fig06_pooled_population.png")
