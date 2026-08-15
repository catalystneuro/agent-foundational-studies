"""Generate all figures from the cached per-session results."""
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


def polar_close(x, y):
    return np.append(x, x[0]), np.append(y, y[0])


# =========================================================== fig 1: raw data
d = hd_lib.load_session(EX)
hd, units, sq = d["hd"], d["units"], d["epochs"]["wake_square"]
hd_idx = np.where(r["is_hd"])[0]
order = hd_idx[np.argsort(r["pref"][hd_idx])]

fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 2.2], width_ratios=[2.6, 2.6, 1.4],
                      hspace=0.45, wspace=0.55)
t0 = sq.start[0] + 600
win = nap.IntervalSet(t0, t0 + 40)

ax = fig.add_subplot(gs[0, :2])
h = hd.restrict(win)
ax.plot(h.t - t0, np.degrees(h.values), "k.", ms=1.5)
ax.set_ylabel("head\ndirection (deg)"); ax.set_ylim(0, 360); ax.set_yticks([0, 180, 360])
ax.set_title("Head direction, position and postsubicular spiking (%s)" % EX, fontsize=10)
ax.set_xticklabels([])

ax = fig.add_subplot(gs[1, :2])
p = d["position"].restrict(win)
ax.plot(p.t - t0, p["x"].values, lw=1, label="x")
ax.plot(p.t - t0, p["y"].values, lw=1, label="y")
ax.set_ylabel("position (cm)"); ax.legend(fontsize=7, ncol=2, frameon=False)
ax.set_xticklabels([])

ax = fig.add_subplot(gs[2, :2])
for k, u in enumerate(order):
    ts = units[int(u)].restrict(win).t - t0
    ax.plot(ts, np.full_like(ts, k), "|", ms=3.5, color=HDC, mew=0.7)
ax.set_ylabel("HD cells\n(sorted by preferred direction)")
ax.set_xlabel("time (s)")
axr = ax.twinx()
axr.plot(h.t - t0, np.degrees(h.values), "k.", ms=1.2, alpha=0.5)
axr.set_ylim(0, 360); axr.set_yticks([0, 180, 360]); axr.set_ylabel("head direction (deg)")
axr.spines["right"].set_visible(True)
ax.set_ylim(-1, len(order))

ax = fig.add_subplot(gs[:2, 2])
pos_sq = d["position"].restrict(sq)
hdi = hd.restrict(sq).interpolate(pos_sq)
sl = slice(None, None, 5)
sc = ax.scatter(pos_sq["x"].values[sl], pos_sq["y"].values[sl], c=np.degrees(hdi.values[sl]),
                cmap="hsv", s=0.7, vmin=0, vmax=360)
ax.set_aspect("equal"); ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
ax.set_title("trajectory coloured\nby head direction", fontsize=9)
plt.colorbar(sc, ax=ax, label="deg", fraction=0.05)

ax = fig.add_subplot(gs[2, 2])
occ = r["occupancy"] / r["occupancy"].sum()
ax.bar(np.degrees(C), occ * 100, width=360 / len(C), color="#34495e")
ax.set_xlabel("head direction (deg)"); ax.set_ylabel("% of time")
ax.set_title("directional occupancy", fontsize=9)
fig.savefig("fig01_raw_data.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("fig01")

# ============================================== fig 2: example tuning curves
best = hd_idx[np.argsort(-r["mvl"][hd_idx])][:8]
worst = np.where(~r["is_hd"])[0]
worst = worst[np.argsort(r["mvl"][worst])][:4]
sel = list(best) + list(worst)

fig, axes = plt.subplots(3, 4, figsize=(12, 9.5), subplot_kw={"projection": "polar"})
for ax, u in zip(axes.ravel(), sel):
    is_hd = r["is_hd"][u]
    col = HDC if is_hd else OTH
    x, y = polar_close(C, r["tc"][u])
    ax.plot(x, y, color=col, lw=1.8)
    ax.fill(x, y, color=col, alpha=0.25)
    for half, ls in [(r["tc1"][u], "--"), (r["tc2"][u], ":")]:
        xx, yy = polar_close(C, half)
        ax.plot(xx, yy, ls, color=col, lw=0.9, alpha=0.8)
    ax.set_title("unit %d — %s\nr=%.2f, %.2f bits/spk, %.1f Hz"
                 % (u, "HD cell" if is_hd else "not HD", r["mvl"][u], r["info"][u],
                    r["rates"][u]), fontsize=8, pad=16)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=7)
    ax.set_yticks([np.nanmax(r["tc"][u])])
    ax.set_yticklabels(["%.0f Hz" % np.nanmax(r["tc"][u])], fontsize=6)
    ax.set_rlabel_position(112)
    ax.tick_params(pad=0)
fig.suptitle("Head-direction tuning curves, wake / square arena (dashed & dotted: "
             "first and second half of the session)", fontsize=11, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig02_tuning_examples.png", dpi=140)
plt.close(fig)
print("fig02")

# ================================================ fig 3: significance testing
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
ax = axes[0]
ax.hist(r["null"].ravel(), bins=60, density=True, color=OTH, alpha=0.7,
        label="circular-shift null\n(%d cells x %d shuffles)" % r["null"].shape)
ax.hist(r["mvl"], bins=40, density=True, color=HDC, alpha=0.6, label="observed")
ax.axvline(r["thr"], color="k", ls="--", lw=1.2, label="99th pct of null = %.2f" % r["thr"])
ax.set_xlabel("mean vector length"); ax.set_ylabel("density")
ax.legend(fontsize=7, frameon=False)
ax.set_title("%s: directional tuning vs null" % EX, fontsize=9)

ax = axes[1]
ax.scatter(r["mvl"][~r["is_hd"]], r["info"][~r["is_hd"]], s=14, c=OTH, label="not HD")
ax.scatter(r["mvl"][r["is_hd"]], r["info"][r["is_hd"]], s=14, c=HDC, label="HD cell")
ax.axvline(r["thr"], color="k", ls="--", lw=1)
ax.set_xlabel("mean vector length"); ax.set_ylabel("directional information (bits/spike)")
ax.set_yscale("log"); ax.legend(fontsize=8, frameon=False)
ax.set_title("two independent tuning measures agree", fontsize=9)

ax = axes[2]
allm = np.concatenate([x["mvl"] for x in R.values()])
allh = np.concatenate([x["is_hd"] for x in R.values()])
bins = np.linspace(0, 1, 41)
ax.hist(allm[~allh], bins=bins, color=OTH, alpha=0.75, label="not HD (n=%d)" % (~allh).sum())
ax.hist(allm[allh], bins=bins, color=HDC, alpha=0.75, label="HD cell (n=%d)" % allh.sum())
ax.set_xlabel("mean vector length"); ax.set_ylabel("units")
ax.legend(fontsize=8, frameon=False)
ax.set_title("all 10 animals: %d units, %.0f%% classified HD"
             % (len(allm), 100 * allh.mean()), fontsize=9)
fig.tight_layout()
fig.savefig("fig03_significance.png", dpi=140)
plt.close(fig)
print("fig03")

# ======================================== fig 4: stability & cross-environment
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ax = axes[0]
alls = np.concatenate([x["stab"] for x in R.values()])
ax.hist(alls[~allh], bins=np.linspace(-1, 1, 41), color=OTH, alpha=0.75, label="not HD")
ax.hist(alls[allh], bins=np.linspace(-1, 1, 41), color=HDC, alpha=0.75, label="HD cell")
ax.set_xlabel("split-half tuning-curve correlation"); ax.set_ylabel("units")
ax.legend(fontsize=8, frameon=False)
ax.set_title("within-session stability (all animals)", fontsize=9)

ax = axes[1]
cr = r["cross"]
i = hd_idx
ax.scatter(np.degrees(r["pref"][i]), np.degrees(cr["pref_tri"][i]), s=18, c=HDC)
ax.set_xlabel("preferred direction, square (deg)")
ax.set_ylabel("preferred direction, triangle (deg)")
ax.set_xlim(0, 360); ax.set_ylim(0, 360)
ax.set_xticks([0, 90, 180, 270, 360]); ax.set_yticks([0, 90, 180, 270, 360])
ax.set_title("%s: same cells, two arenas\ncoherent rotation of %.0f°, R=%.2f"
             % (EX, np.degrees(cr["rotation"]), cr["R"]), fontsize=9)

ax = axes[2]
alld = np.concatenate([x["cross"]["dphi"] - x["cross"]["rotation"]
                       for x in R.values() if x["cross"] is not None])
alld = np.degrees(hd_lib.angdiff(alld, 0))
ax.hist(alld, bins=np.linspace(-180, 180, 73), color=HDC, alpha=0.85)
ax.set_xlabel("preferred-direction offset relative to the\npopulation rotation (deg)")
ax.set_ylabel("HD cells")
ax.set_xlim(-180, 180); ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_title("9 animals: the map rotates rigidly\n(circular SD = %.0f°)"
             % np.degrees(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * np.radians(alld))))))),
             fontsize=9)
fig.tight_layout()
fig.savefig("fig04_stability_crossenv.png", dpi=140)
plt.close(fig)
print("fig04")

# ============================================ fig 5: population tuning matrix
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2),
                         gridspec_kw={"width_ratios": [1.4, 1, 1]})
ax = axes[0]
norm = r["tc"][order] / r["tc"][order].max(1, keepdims=True)
im = ax.imshow(norm, aspect="auto", origin="lower", cmap="magma",
               extent=[0, 360, 0, len(order)])
ax.set_xlabel("head direction (deg)"); ax.set_ylabel("HD cell (sorted by preferred direction)")
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_title("%s: %d HD cells tile all directions" % (EX, len(order)), fontsize=9)
plt.colorbar(im, ax=ax, label="normalised rate", fraction=0.04)

ax = axes[1]
allp = np.concatenate([x["pref"][x["is_hd"]] for x in R.values()])
ax.hist(np.degrees(allp), bins=np.arange(0, 361, 15), color=HDC, alpha=0.85)
ax.set_xlabel("preferred direction (deg)"); ax.set_ylabel("HD cells")
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_title("preferred directions, all animals\n(n=%d, uniform: R=%.3f)"
             % (len(allp), np.abs(np.mean(np.exp(1j * allp)))), fontsize=9)

ax = axes[2]
w = np.array([np.degrees(hd_lib.circ_tuning_width(C, t)) for t in r["tc"][hd_idx]])
allw = np.concatenate([[np.degrees(hd_lib.circ_tuning_width(x["centers"], t))
                        for t in x["tc"][x["is_hd"]]] for x in R.values()])
ax.hist(allw, bins=30, color=HDC, alpha=0.85)
ax.axvline(np.median(allw), color="k", ls="--", lw=1.2,
           label="median %.0f°" % np.median(allw))
ax.set_xlabel("tuning width (deg, full width at half max)")
ax.set_ylabel("HD cells"); ax.legend(fontsize=8, frameon=False)
ax.set_title("tuning width, all animals", fontsize=9)
fig.tight_layout()
fig.savefig("fig05_population_tuning.png", dpi=140)
plt.close(fig)
print("fig05")
