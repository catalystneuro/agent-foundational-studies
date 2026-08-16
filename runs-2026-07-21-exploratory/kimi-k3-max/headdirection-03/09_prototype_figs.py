"""Single-session prototype figures for Mouse17-130128 (DANDI 000056)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap

from hd_analysis import (load_session, compute_head_direction, get_epochs,
                         analyze_session, circ_corr)

ASSET_ID = "4cc64fe0-7b1e-404c-8b86-fb5659292830"  # sub-Mouse17_ses-Mouse17-130128
SESSION = "Mouse17-130128"
FIGDIR = "figs_proto"
os.makedirs(FIGDIR, exist_ok=True)

nwb, io = load_session(ASSET_ID)
hd = compute_head_direction(nwb)
wake, rem, nrem = get_epochs(nwb)
print(f"wake {wake.tot_length():.0f}s | REM {rem.tot_length():.0f}s | "
      f"NREM {nrem.tot_length():.0f}s")

res = analyze_session(nwb, n_shuf=500, seed=0)

# ============================================================ fig 1: raw data
red = nwb["SubjectPosition/RedLED"]
blue = nwb["SubjectPosition/BlueLED"]
valid = (red.values[:, 0] > 0) & (blue.values[:, 0] > 0)

fig = plt.figure(figsize=(13, 4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.6, 1], wspace=0.3)

ax = fig.add_subplot(gs[0])
ax.plot(red.values[valid, 0], red.values[valid, 1], ".", ms=0.3,
        color="tab:red", alpha=0.5, label="red LED", rasterized=True)
ax.plot(blue.values[valid, 0], blue.values[valid, 1], ".", ms=0.3,
        color="tab:blue", alpha=0.5, label="blue LED", rasterized=True)
ax.set_aspect("equal")
ax.set_xlabel("x (camera pixels)")
ax.set_ylabel("y (camera pixels)")
ax.set_title("Head-tracker LED positions")
ax.legend(markerscale=10, loc="upper right", fontsize=8)

ax = fig.add_subplot(gs[1])
t0 = wake["start"][0]
snippet = hd.get(t0, t0 + 120)
ax.plot(snippet.t - t0, snippet.d, ".", ms=1.5, color="0.2", rasterized=True)
ax.set_xlabel("time in wake epoch (s)")
ax.set_ylabel("head direction (rad)")
ax.set_yticks([0, np.pi, 2 * np.pi], ["0", "$\\pi$", "$2\\pi$"])
ax.set_ylim(-0.15, 2 * np.pi + 0.15)
ax.set_title("Head direction over 120 s of exploration")

ax = fig.add_subplot(gs[2], projection="polar")
hd_w = hd.restrict(wake).values
hd_w = hd_w[~np.isnan(hd_w)]
occ, edges = np.histogram(hd_w, bins=60, range=(0, 2 * np.pi))
centers = (edges[:-1] + edges[1:]) / 2
ax.bar(centers, occ / 39.0625, width=2 * np.pi / 60, color="0.4")
ax.set_title("HD occupancy, wake (s)", pad=22)
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
ax.set_xticklabels(["0", "$\\pi/2$", "$\\pi$", "$3\\pi/2$"])
ax.set_rlabel_position(90)
ax.tick_params(axis="y", labelsize=7, pad=-2)
ax.yaxis.set_label_coords(-0.05, 0.5)

fig.savefig(f"{FIGDIR}/fig1_raw_head_direction.png", dpi=200)
plt.close(fig)
print("saved fig1")

# ============================================================ fig 2: ID scatter
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

ax = axes[0]
ok = ~np.isnan(res["mvl"])
ax.scatter(res["mvl"][ok & ~res["is_hd"]], res["null95"][ok & ~res["is_hd"]],
           c="0.55", s=25, label="not HD")
ax.scatter(res["mvl"][ok & res["is_hd"]], res["null95"][ok & res["is_hd"]],
           c="crimson", s=30, label="HD cell")
lim = [0, max(np.nanmax(res["mvl"]), np.nanmax(res["null95"])) * 1.08]
ax.plot(lim, lim, "k--", lw=1)
ax.axvline(0.3, color="0.4", ls=":", lw=1)
ax.text(0.305, lim[1] * 0.75, "MVL = 0.3 floor", rotation=90, fontsize=8,
        color="0.35")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("observed mean vector length")
ax.set_ylabel("95th pct of random-time null")
ax.set_title(f"HD identification ({res['is_hd'].sum()}/{ok.sum()} HD cells)")
ax.legend(loc="lower right", fontsize=9)

ax = axes[1]
ax.hist(res["mvl"][ok & ~res["is_hd"]], bins=np.arange(0, 1.01, 0.05),
        color="0.6", label="not HD")
ax.hist(res["mvl"][ok & res["is_hd"]], bins=np.arange(0, 1.01, 0.05),
        color="crimson", label="HD cell")
ax.set_xlabel("mean vector length")
ax.set_ylabel("unit count")
ax.set_title("Distribution of directional tuning strength")
ax.legend(fontsize=9)

fig.savefig(f"{FIGDIR}/fig2_hd_identification.png", dpi=200)
plt.close(fig)
print("saved fig2")

# ============================================================ fig 3: tuning curves
tc = res["tc_wake"]
theta = res["theta"]
theta_c = np.concatenate([theta, [theta[0] + 2 * np.pi]])

order = np.argsort(-np.nan_to_num(res["mvl"]))
n_show = 18
fig, axes = plt.subplots(3, 6, figsize=(13, 6.6),
                         subplot_kw=dict(projection="polar"))
for k in range(n_show):
    i = order[k]
    ax = axes[k // 6, k % 6]
    curve = tc[i]
    curve_c = np.concatenate([curve, [curve[0]]])
    col = "crimson" if res["is_hd"][i] else "0.5"
    ax.plot(theta_c, curve_c, color=col, lw=1.5)
    ax.fill(theta_c, curve_c, color=col, alpha=0.25)
    ax.set_title(f"unit {res['keys'][i]}   MVL={res['mvl'][i]:.2f}",
                 fontsize=9, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f"Wake head-direction tuning curves, {SESSION} "
             "(18 strongest; red = significant HD cell)", y=1.00)
fig.subplots_adjust(hspace=0.35, wspace=0.3)
fig.savefig(f"{FIGDIR}/fig3_tuning_curves.png", dpi=200)
plt.close(fig)
print("saved fig3")

# ============================================================ fig 4: cross-state correlations
# The mouse sleeps with its head in a narrow range of directions, so
# per-cell REM tuning curves are unreliable. Following Peyrache et al.
# (2015), we instead test whether the PAIRWISE correlation structure of
# the HD ensemble during sleep matches that during wake.
from hd_analysis import state_correlations

units_all = nwb["units"]
units_f2 = units_all[units_all.metadata["rate"].values > 0.1]
hd_keys = res["keys"][res["is_hd"]]
prefs_hd = res["pref_wake"][res["is_hd"]]
order = np.argsort(prefs_hd)

Cw = state_correlations(units_f2, hd_keys, wake)
Cr = state_correlations(units_f2, hd_keys, rem)
Cn = state_correlations(units_f2, hd_keys, nrem)

iu = np.triu_indices(len(hd_keys), k=1)
m_rem = ~np.isnan(Cw[iu]) & ~np.isnan(Cr[iu])
m_nrem = ~np.isnan(Cw[iu]) & ~np.isnan(Cn[iu])
r_rem = np.corrcoef(Cw[iu][m_rem], Cr[iu][m_rem])[0, 1]
r_nrem = np.corrcoef(Cw[iu][m_nrem], Cn[iu][m_nrem])[0, 1]
print(f"wake-REM corr-of-corrs: {r_rem:.3f} ({m_rem.sum()} pairs); "
      f"wake-NREM: {r_nrem:.3f} ({m_nrem.sum()} pairs)")

fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.32, wspace=0.35,
                      width_ratios=[1, 1, 1])

vmax = 0.6
for k, (C, name) in enumerate([(Cw, "wake"), (Cr, "REM sleep"),
                               (Cn, "NREM sleep")]):
    ax = fig.add_subplot(gs[0, k])
    Cs = C[np.ix_(order, order)]
    im = ax.imshow(Cs, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"{name}", fontsize=11)
    if k == 0:
        ax.set_ylabel("HD cells (sorted by wake pref. dir.)")
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(im, ax=[fig.axes[-3], fig.axes[-2], fig.axes[-1]],
             label="pairwise correlation", shrink=0.75, pad=0.02)

ax = fig.add_subplot(gs[1, 0])
ax.scatter(Cw[iu][m_rem], Cr[iu][m_rem], s=30, color="tab:red", alpha=0.8)
ax.plot([-0.4, 0.8], [-0.4, 0.8], "k--", lw=1)
ax.set_xlabel("wake correlation")
ax.set_ylabel("REM correlation")
ax.set_title(f"wake vs REM: r = {r_rem:.2f}")
ax.set_xlim(-0.45, 0.85)
ax.set_ylim(-0.45, 0.85)

ax = fig.add_subplot(gs[1, 1])
ax.scatter(Cw[iu][m_nrem], Cn[iu][m_nrem], s=30, color="tab:blue", alpha=0.8)
ax.plot([-0.4, 0.8], [-0.4, 0.8], "k--", lw=1)
ax.set_xlabel("wake correlation")
ax.set_ylabel("NREM correlation")
ax.set_title(f"wake vs NREM: r = {r_nrem:.2f}")
ax.set_xlim(-0.45, 0.85)
ax.set_ylim(-0.45, 0.85)

ax = fig.add_subplot(gs[1, 2])
# control: shuffle cell identity in REM, recompute corr-of-corrs
rng = np.random.default_rng(0)
shuf_r = []
for _ in range(200):
    p = rng.permutation(len(hd_keys))
    Crs = Cr[np.ix_(p, p)]
    mm = ~np.isnan(Cw[iu]) & ~np.isnan(Crs[iu])
    if mm.sum() > 5:
        shuf_r.append(np.corrcoef(Cw[iu][mm], Crs[iu][mm])[0, 1])
ax.hist(shuf_r, bins=20, color="0.6", label="label-shuffled null")
ax.axvline(r_rem, color="tab:red", lw=2, label=f"observed REM r={r_rem:.2f}")
ax.axvline(r_nrem, color="tab:blue", lw=2, label=f"observed NREM r={r_nrem:.2f}")
ax.set_xlabel("correlation of correlations")
ax.set_ylabel("count")
ax.set_title("Observed vs cell-label shuffle")
ax.legend(fontsize=8)

fig.suptitle(f"HD ensemble correlation structure is preserved in sleep, "
             f"{SESSION}", y=0.98)
fig.savefig(f"{FIGDIR}/fig4_wake_vs_sleep.png", dpi=200)
plt.close(fig)
print("saved fig4")

np.savez(f"{FIGDIR}/session_results_{SESSION}.npz",
         **{k: v for k, v in res.items() if v is not None})
print("done")
