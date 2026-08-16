"""Figures for the place-cell demonstration (DANDI 000044 Achilles)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.ndimage import gaussian_filter1d

np.random.seed(0)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "figure.dpi": 110})

OUT = "figures"
import os
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- load data
d = np.load("placefield_cache.npz")
e = np.load("si_cache.npz")
dd = np.load("direction_cache.npz")
c = np.load("classify_cache.npz")

bc = d["bc"]; occ_t = d["occ_t"]; counts_all = d["counts_all"]
spk_mat = d["spk_mat"]; lin_run = d["lin_run"]
t_run = d["t_run"]; t_lin = d["t_lin"]; lin_values = d["lin_values"]
x2, y2 = d["x2"], d["y2"]
unit_keys = [int(k) for k in d["unit_keys"]]
bouts = d["bouts"]; dirs = d["dirs"]

si_real = e["si_real"]; pvalues = e["pvalues"]; rates_sm = e["rates_sm"]
null_si = e["null_si"]
tc_pos = dd["tc_pos"]; tc_neg = dd["tc_neg"]
occ_pos = dd["occ_pos"]; occ_neg = dd["occ_neg"]
is_place = c["is_place"]; exc_mask = c["exc_mask"]; inh_mask = c["inh_mask"]
mean_rate = c["mean_rate"]; peak_rate = c["peak_rate"]; dir_corr = c["dir_corr"]

n_units = len(unit_keys)
nbins = len(bc)

# ============================================================= FIG 1: track
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ax = axes[0]
sc = ax.scatter(x2[::3], y2[::3], c=t_lin[::3], s=1, cmap="viridis",
                rasterized=True)
ax.set_title("2D trajectory (MazeEpoch, colored by time)")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
plt.colorbar(sc, ax=ax, label="time (s)", shrink=0.8)

ax = axes[1]
h, xe, ye = np.histogram2d(x2[~np.isnan(x2)], y2[~np.isnan(y2)], bins=100)
ax.imshow(h.T, origin="lower", aspect="auto",
          extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="hot", interpolation="bilinear")
ax.set_title("Occupancy map")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

ax = axes[2]
ax.plot(bc, occ_t, "o-", ms=3, lw=1)
ax.set_title("Linearized occupancy")
ax.set_xlabel("position along track (m)"); ax.set_ylabel("occupancy (s)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_track.png", bbox_inches="tight")
plt.close()

# ============================================================ FIG 2: raster
fig = plt.figure(figsize=(11, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.2], hspace=0.15)
# window: first 150 s of maze, show a subset of units
w0, w1 = 18100.0, 18300.0
mask_win = (t_run >= w0) & (t_run <= w1)
t_win = t_run[mask_win]
lin_win = lin_run[mask_win]

ax = fig.add_subplot(gs[0])
ax.plot(t_win, lin_win, "k-", lw=0.8)
ax.set_ylabel("lin pos (m)")
ax.set_xlim(w0, w1)

ax = fig.add_subplot(gs[1])
# pick 40 place cells spread across the track
sel = np.flatnonzero(is_place)[::max(1, is_place.sum() // 40)]
for j, k in enumerate(sel):
    spk_j = spk_mat[mask_win, k]
    tt = t_win[spk_j > 0]
    ax.plot(tt, np.full_like(tt, j), "|", ms=3, color="k", alpha=0.6)
ax.set_xlim(w0, w1)
ax.set_ylim(-1, len(sel))
ax.set_yticks([])
ax.set_ylabel("place cells")
ax.set_xlabel("time (s) during MazeEpoch")
ax.set_title("Position and spike raster (first 200 s of maze)")
plt.savefig(f"{OUT}/fig2_raster.png", bbox_inches="tight")
plt.close()

# ==================================================== FIG 3: example fields
fig, axes = plt.subplots(3, 4, figsize=(13, 8))
axes = axes.ravel()
pcs = np.flatnonzero(is_place)
# pick a spread: sort by field peak location
peak_loc = bc[np.argmax(rates_sm, axis=1)]
order = np.argsort(peak_loc[pcs])
for i, idx in enumerate(order[:12]):
    k = pcs[idx]
    ax = axes[i]
    ax.fill_between(bc, np.zeros_like(bc), rates_sm[k], color="tab:blue", alpha=0.5,
                    label="both dirs")
    ax.plot(bc, tc_pos[k], color="tab:green", lw=1, label="outbound")
    ax.plot(bc, tc_neg[k], color="tab:orange", lw=1, label="inbound")
    ax.set_title(f"unit {unit_keys[k]}, SI={si_real[k]:.2f} bits")
    ax.set_xlim(bc[0], bc[-1]); ax.set_yticks([])
    if i % 4 == 0:
        ax.set_ylabel("rate (Hz)")
axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("Example place fields (CA1, linear track)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_example_fields.png", bbox_inches="tight")
plt.close()

# ================================================== FIG 4: SI distributions
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
bins = np.linspace(0, 3.5, 60)
ax.hist(si_real[exc_mask], bins=bins, alpha=0.6, label="excitatory (n=%d)" % exc_mask.sum(),
        color="tab:blue")
ax.hist(si_real[inh_mask], bins=bins, alpha=0.6, label="inhibitory (n=%d)" % inh_mask.sum(),
        color="tab:red")
ax.set_xlabel("Skaggs spatial information (bits/spike)")
ax.set_ylabel("units")
ax.set_title("Spatial information by cell type")
ax.legend()

ax = axes[1]
# null distribution (pooled across units) vs real SI of place cells
null_pool = null_si.ravel()
ax.hist(null_pool, bins=bins, density=True, alpha=0.5, label="circular-shift null",
        color="gray")
ax.hist(si_real[is_place], bins=bins, density=True, alpha=0.6,
        label="place cells (n=%d)" % is_place.sum(), color="tab:blue")
ax.set_xlabel("Skaggs spatial information (bits/spike)")
ax.set_ylabel("density")
ax.set_title("Real vs shuffled (circular time-shift) SI")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_si.png", bbox_inches="tight")
plt.close()

# ===================================================== FIG 5: direction corr
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.scatter(si_real[exc_mask], dir_corr[exc_mask], s=12, alpha=0.6,
           color="tab:blue", label="excitatory")
ax.scatter(si_real[inh_mask], dir_corr[inh_mask], s=12, alpha=0.6,
           color="tab:red", label="inhibitory")
ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("SI (bits/spike)"); ax.set_ylabel("outbound–inbound map corr")
ax.set_title("Direction selectivity vs spatial information")
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(dir_corr[exc_mask], bins=np.linspace(-1, 1, 41), alpha=0.6,
        label="excitatory", color="tab:blue")
ax.hist(dir_corr[inh_mask], bins=np.linspace(-1, 1, 41), alpha=0.6,
        label="inhibitory", color="tab:red")
ax.set_xlabel("outbound–inbound map correlation")
ax.set_ylabel("units")
ax.set_title("Place field stability across run direction")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_direction.png", bbox_inches="tight")
plt.close()

# ============================================ FIG 6: population place field map
fig, ax = plt.subplots(figsize=(7, 9))
# normalize each place cell's field to peak 1, sort by peak location
sub = rates_sm[pcs]
norm = sub / np.maximum(sub.max(axis=1, keepdims=True), 1e-9)
order = np.argsort(peak_loc[pcs])
im = ax.imshow(norm[order], aspect="auto", origin="lower", cmap="magma",
               extent=[bc[0], bc[-1], 0, len(pcs)])
ax.set_yticks([])
ax.set_xlabel("position along track (m)")
ax.set_ylabel("place cells (sorted by field location)")
ax.set_title("Place field map (%d place cells)" % len(pcs))
plt.colorbar(im, ax=ax, label="normalized rate")
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_field_map.png", bbox_inches="tight")
plt.close()

# ================================================== FIG 7: classification stats
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
ax = axes[0]
frac = np.mean(pvalues[exc_mask] < 0.05)
ax.bar(["shuffle-sig", "place cells"], [frac * 100, 100 * is_place.sum() / exc_mask.sum()],
       color=["tab:gray", "tab:blue"])
ax.set_ylabel("% of excitatory units")
ax.set_title("Place cell fraction (excitatory)")

ax = axes[1]
ax.hist(si_real[is_place], bins=np.linspace(0, 3.5, 40), color="tab:blue", alpha=0.7)
ax.set_xlabel("SI (bits/spike)"); ax.set_ylabel("place cells")
ax.set_title("SI among place cells")

ax = axes[2]
ok = exc_mask & (mean_rate > 0.05)
ax.scatter(mean_rate[ok], peak_rate[ok], s=8, alpha=0.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("mean rate (Hz)"); ax.set_ylabel("peak rate (Hz)")
ax.set_title("Rate statistics (excitatory)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_classification.png", bbox_inches="tight")
plt.close()

print("figures written to", OUT)