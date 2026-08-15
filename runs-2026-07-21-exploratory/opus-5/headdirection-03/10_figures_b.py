"""Figures 6-9: decoding, sleep correlation structure, GLM, cross-animal summary."""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import hd_lib

R = pickle.load(open("results/all_sessions.pkl", "rb"))
EX = "sub-A3701_ses-191119"
r = R[EX]
C = r["centers"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
HDC, OTH = "#c0392b", "#7f8c8d"

d = hd_lib.load_session(EX)
hd, units, sq = d["hd"], d["units"], d["epochs"]["wake_square"]
hd_idx = np.where(r["is_hd"])[0]
order = hd_idx[np.argsort(r["pref"][hd_idx])]

# ============================= fig 6: population bump + Bayesian decoding ===
mid = sq.start[0] + (sq.end[0] - sq.start[0]) / 2
e1, e2 = nap.IntervalSet(sq.start[0], mid), nap.IntervalSet(mid, sq.end[0])
f1 = hd_lib.FastTuning(hd, e1, 60)
tc1 = pd.DataFrame(index=C, data={int(c): f1.curve(units[int(c)].t) for c in hd_idx})
sub = units[[int(c) for c in hd_idx]]
dec, prob = nap.decode_1d(tuning_curves=tc1, group=sub, ep=e2, bin_size=0.2)
true = hd.restrict(e2).interpolate(dec)
err = np.degrees(np.abs(hd_lib.angdiff(dec.values, true.values)))

fig = plt.figure(figsize=(13, 7.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], width_ratios=[2.2, 1, 1],
                      hspace=0.42, wspace=0.32)

t0 = e2.start[0] + 400
win = nap.IntervalSet(t0, t0 + 30)
ax = fig.add_subplot(gs[0, :])
cnt = sub.count(0.1, win)
z = np.asarray(cnt.values, dtype=float)[:, np.argsort(r["pref"][hd_idx])]
zz = z / np.maximum(z.max(0, keepdims=True), 1)
ax.imshow(zz.T, aspect="auto", origin="lower", cmap="magma",
          extent=[0, 30, 0, 360], interpolation="nearest")
h = hd.restrict(win)
ax.plot(h.t - t0, np.degrees(h.values), ".", color="w", ms=2.5, label="true head direction")
dd = dec.restrict(win)
ax.plot(dd.t - t0, np.degrees(dd.values), ".", color="#2ecc71", ms=4,
        label="decoded (Bayesian, 200 ms)")
ax.set_ylabel("preferred direction of cell (deg)\n/ head direction (deg)")
ax.set_xlabel("time (s)"); ax.set_yticks([0, 90, 180, 270, 360])
ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
ax.set_title("%s: population activity forms a single bump that tracks the animal's "
             "head direction" % EX, fontsize=10)

ax = fig.add_subplot(gs[1, 0])
sl = ~np.isnan(true.values)
ax.plot(np.degrees(true.values[sl]), np.degrees(dec.values[sl]), ".", ms=1.2,
        color=HDC, alpha=0.25)
ax.set_xlabel("true head direction (deg)"); ax.set_ylabel("decoded (deg)")
ax.set_xticks([0, 180, 360]); ax.set_yticks([0, 180, 360])
ax.set_title("held-out decoding\n(tuning curves from the other half)", fontsize=9)

ax = fig.add_subplot(gs[1, 1])
alle = np.concatenate([x["decode_err"] for x in R.values()])
ax.hist(alle, bins=np.arange(0, 181, 3), color=HDC, alpha=0.85)
ax.axvline(np.median(alle), color="k", ls="--", lw=1.2,
           label="median %.1f°" % np.median(alle))
ax.set_xlabel("|decoding error| (deg)"); ax.set_ylabel("200 ms bins")
ax.legend(fontsize=8, frameon=False)
ax.set_title("all animals (chance median = 90°)", fontsize=9)

ax = fig.add_subplot(gs[1, 2])
dn = {}
for x in R.values():
    for n, e in x["decode_vs_n"]:
        dn.setdefault(n, []).append(e)
ns = sorted(dn)
med = [np.median(dn[n]) for n in ns]
ax.plot(ns, med, "o-", color=HDC)
for n in ns:
    ax.plot([n] * len(dn[n]), dn[n], ".", color=OTH, ms=3, alpha=0.6)
ax.axhline(90, color="k", ls=":", lw=1, label="chance")
ax.set_xscale("log"); ax.set_xlabel("number of HD cells used")
ax.set_ylabel("median |error| (deg)"); ax.legend(fontsize=8, frameon=False)
ax.set_xticks(ns); ax.set_xticklabels(ns)
ax.set_title("decoding improves with ensemble size", fontsize=9)
fig.savefig("fig06_decoding.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("fig06")

# ================================ fig 7: correlation structure across states ==
def binned_offset_profile(x, key, nbin=12):
    v = x["pw"][key]
    if v is None:
        return None
    o = x["pw_offsets"]
    good = np.isfinite(v)
    edges = np.linspace(0, np.pi, nbin + 1)
    idx = np.clip(np.digitize(o[good], edges) - 1, 0, nbin - 1)
    return np.array([v[good][idx == k].mean() if (idx == k).any() else np.nan
                     for k in range(nbin)]), 0.5 * (edges[:-1] + edges[1:])


fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ax = axes[0]
cols = {"wake": "#2c3e50", "REM": "#e67e22", "nREM": "#2980b9"}
for key in ["wake", "REM", "nREM"]:
    prof = [binned_offset_profile(x, key) for x in R.values()]
    prof = [p for p in prof if p is not None]
    Y = np.array([p[0] for p in prof]); xx = np.degrees(prof[0][1])
    m, s = np.nanmean(Y, 0), np.nanstd(Y, 0) / np.sqrt(len(Y))
    ax.plot(xx, m, "o-", color=cols[key], label="%s (n=%d)" % (key, len(Y)), ms=4)
    ax.fill_between(xx, m - s, m + s, color=cols[key], alpha=0.2)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("|difference in preferred direction| (deg)")
ax.set_ylabel("mean pairwise spike-count correlation")
ax.legend(fontsize=8, frameon=False)
ax.set_title("HD-cell correlation structure by brain state", fontsize=9)

ax = axes[1]
x0, y0 = r["pw"]["wake"], r["pw"]["REM"]
g = np.isfinite(x0) & np.isfinite(y0)
sc = ax.scatter(x0[g], y0[g], s=4, c=np.degrees(r["pw_offsets"][g]), cmap="viridis", alpha=0.6)
ax.set_xlabel("pairwise correlation, wake"); ax.set_ylabel("pairwise correlation, REM sleep")
lim = [min(x0[g].min(), y0[g].min()), max(x0[g].max(), y0[g].max())]
ax.plot(lim, lim, "k--", lw=0.8)
plt.colorbar(sc, ax=ax, label="|Δ preferred dir| (deg)", fraction=0.045)
ax.set_title("%s: r = %.2f" % (EX, np.corrcoef(x0[g], y0[g])[0, 1]), fontsize=9)

ax = axes[2]
vals = {}
for key in ["REM", "nREM"]:
    v = []
    for x in R.values():
        if x["pw"][key] is None:
            continue
        a, b = x["pw"]["wake"], x["pw"][key]
        g = np.isfinite(a) & np.isfinite(b)
        v.append(np.corrcoef(a[g], b[g])[0, 1])
    vals[key] = np.array(v)
pos = [0, 1]
ax.bar(pos, [vals["REM"].mean(), vals["nREM"].mean()],
       color=[cols["REM"], cols["nREM"]], alpha=0.7, width=0.6)
for p, key in zip(pos, ["REM", "nREM"]):
    ax.plot(np.full(len(vals[key]), p) + np.linspace(-0.12, 0.12, len(vals[key])),
            vals[key], "ko", ms=4)
ax.set_xticks(pos); ax.set_xticklabels(["wake vs REM\n(n=%d)" % len(vals["REM"]),
                                        "wake vs nREM\n(n=%d)" % len(vals["nREM"])])
ax.set_ylabel("correlation of the pairwise\ncorrelation matrices")
ax.set_ylim(0, 1)
ax.set_title("the ring structure persists in sleep,\nwithout any sensory input", fontsize=9)
fig.tight_layout()
fig.savefig("fig07_sleep_structure.png", dpi=140)
plt.close(fig)
print("fig07")

# ============================================================== fig 8: GLM ===
G = pickle.load(open("results/glm.pkl", "rb"))
g = G[EX]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ax = axes[0]
picks = hd_idx[np.argsort(-r["mvl"][hd_idx])][:4]
for k, u in enumerate(picks):
    ax.plot(np.degrees(C), r["tc"][u], color="C%d" % k, lw=1, alpha=0.5)
    ax.plot(np.degrees(g["grid"]), g["pred_tc"][u], color="C%d" % k, lw=2,
            label="unit %d (pseudo-R²=%.2f)" % (u, g["pseudo_r2"][u]))
ax.set_xlabel("head direction (deg)"); ax.set_ylabel("firing rate (Hz)")
ax.set_xticks([0, 90, 180, 270, 360]); ax.legend(fontsize=7, frameon=False)
ax.set_title("GLM fit (thick) vs empirical tuning (thin)", fontsize=9)

ax = axes[1]
allr2 = np.concatenate([G[s]["pseudo_r2"] for s in R])
allh = np.concatenate([R[s]["is_hd"] for s in R])
allm = np.concatenate([R[s]["mvl"] for s in R])
ax.scatter(allm[~allh], allr2[~allh], s=6, c=OTH, label="not HD", alpha=0.6)
ax.scatter(allm[allh], allr2[allh], s=6, c=HDC, label="HD cell", alpha=0.6)
ax.set_ylim(-0.05, max(0.6, allr2.max() * 1.05))
ax.set_xlabel("mean vector length")
ax.set_ylabel("held-out pseudo-R² (McFadden)")
ax.legend(fontsize=8, frameon=False)
ax.set_title("GLM: head direction alone predicts held-out spiking\n(Spearman ρ = %.2f)"
             % pd.Series(allm).corr(pd.Series(allr2), method="spearman"), fontsize=9)

ax = axes[2]
ax.hist(allr2[~allh], bins=np.linspace(-0.05, 0.6, 40), color=OTH, alpha=0.75,
        label="not HD (median %.3f)" % np.median(allr2[~allh]))
ax.hist(allr2[allh], bins=np.linspace(-0.05, 0.6, 40), color=HDC, alpha=0.75,
        label="HD cell (median %.3f)" % np.median(allr2[allh]))
ax.set_xlabel("held-out pseudo-R²"); ax.set_ylabel("units")
ax.legend(fontsize=8, frameon=False)
ax.set_title("all animals", fontsize=9)
fig.tight_layout()
fig.savefig("fig08_glm.png", dpi=140)
plt.close(fig)
print("fig08")

# ================================================ fig 9: across-animal summary
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
names = list(R)
short = [s.split("_")[0].replace("sub-", "") for s in names]

ax = axes[0]
frac = [100 * R[s]["is_hd"].mean() for s in names]
fauth = [100 * R[s]["is_hd_author"].mean() for s in names]
xx = np.arange(len(names))
ax.bar(xx - 0.2, frac, 0.4, color=HDC, label="this analysis")
ax.bar(xx + 0.2, fauth, 0.4, color="#34495e", label="dataset label")
ax.set_xticks(xx); ax.set_xticklabels(short, rotation=60, fontsize=7)
ax.set_ylabel("% of units classified HD"); ax.legend(fontsize=7, frameon=False)
ax.set_title("HD cells per animal", fontsize=9)

ax = axes[1]
parts = [R[s]["mvl"][R[s]["is_hd"]] for s in names]
ax.boxplot(parts, showfliers=False,
           medianprops=dict(color=HDC), tick_labels=short)
ax.tick_params(axis="x", rotation=60, labelsize=7)
ax.set_ylabel("mean vector length (HD cells)")
ax.set_title("tuning strength", fontsize=9)

ax = axes[2]
de = [np.median(R[s]["decode_err"]) for s in names]
ax.bar(xx, de, color=HDC)
ax.axhline(90, color="k", ls=":", lw=1, label="chance")
ax.set_xticks(xx); ax.set_xticklabels(short, rotation=60, fontsize=7)
ax.set_ylabel("median decoding error (deg)"); ax.legend(fontsize=7, frameon=False)
ax.set_title("held-out head-direction decoding", fontsize=9)

ax = axes[3]
Rs = [R[s]["cross"]["R"] for s in names if R[s]["cross"] is not None]
rot = [np.degrees(R[s]["cross"]["rotation"]) for s in names if R[s]["cross"] is not None]
lab = [t for s, t in zip(names, short) if R[s]["cross"] is not None]
ax.scatter(rot, Rs, s=45, c=HDC)
for x_, y_, t in zip(rot, Rs, lab):
    ax.annotate(t, (x_, y_), fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.set_xlim(-10, 370); ax.set_ylim(0, 1.05)
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_xlabel("population rotation, square → triangle (deg)")
ax.set_ylabel("coherence of the rotation (R)")
ax.set_title("the map rotates as a whole\nbetween environments", fontsize=9)
fig.tight_layout()
fig.savefig("fig09_across_animals.png", dpi=140)
plt.close(fig)
print("fig09")
