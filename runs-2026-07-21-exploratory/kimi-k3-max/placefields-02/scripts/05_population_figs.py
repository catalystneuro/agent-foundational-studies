"""Figures 3-4: population rate-map tiling + place-cell statistics."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("cache/place_fields.npz")
keys = d["keys"]; cell_type = d["cell_type"]; rates = d["rates"]
rates_pos = d["rates_pos"]; rates_neg = d["rates_neg"]
si = d["si_real"]; p_si = d["p_si"]; mean_rate = d["mean_rate"]
peak_rate = d["peak_rate"]; is_place = d["is_place"]; dir_corr = d["dir_corr"]
centers = d["centers"]
exc = cell_type == "excitatory"

# ---------- Figure 3: population heatmaps sorted by peak ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 6.5),
                         gridspec_kw=dict(wspace=0.35, left=0.06, right=0.97,
                                          top=0.9, bottom=0.12))

def norm_maps(idx, R):
    M = R[idx].copy()
    M = M[~np.isnan(M).all(axis=1)]  # cells with no spikes in this condition
    pk = np.nanmax(M, axis=1, keepdims=True)
    pk[pk == 0] = np.nan
    M = M / pk
    M = M[~np.isnan(M).all(axis=1)]  # zero-peak rows became all-NaN
    order = np.argsort(np.nanargmax(M, axis=1))
    return M[order]

pc_exc = np.where(is_place & exc)[0]

for ax, R, title in [
    (axes[0], rates, "A  All runs (pooled)"),
    (axes[1], rates_pos, "B  Positive-direction runs"),
    (axes[2], rates_neg, "C  Negative-direction runs"),
]:
    M = norm_maps(pc_exc, R)
    im = ax.imshow(M, aspect="auto", cmap="viridis",
                   extent=[0, 1.6, 0, M.shape[0]], origin="lower",
                   vmin=0, vmax=1)
    ax.set_xlabel("track position (m)")
    ax.set_title(f"{title} — {M.shape[0]} place cells", fontsize=10, loc="left")
    if ax is axes[0]:
        ax.set_ylabel("place cells (sorted by peak)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="norm. rate")

fig.suptitle("Place fields tile the linear track — excitatory place cells, normalized rate maps",
             fontsize=12)
fig.savefig("figures/fig3_population_tiling.png", dpi=150)
print("saved figures/fig3_population_tiling.png")

# ---------- Figure 4: statistics ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5),
                         gridspec_kw=dict(hspace=0.4, wspace=0.3,
                                          left=0.08, right=0.96, top=0.92, bottom=0.09))

# A: SI distributions
ax = axes[0, 0]
bins = np.linspace(0, 3, 60)
ax.hist(si[exc & is_place], bins=bins, alpha=0.7, color="tab:blue",
        label=f"exc place cells (n={np.sum(exc & is_place)})")
ax.hist(si[exc & ~is_place], bins=bins, alpha=0.7, color="0.5",
        label=f"exc non-place (n={np.sum(exc & ~is_place)})")
ax.hist(si[~exc], bins=bins, alpha=0.7, color="tab:red",
        label=f"inhibitory (n={np.sum(~exc)})")
ax.set_xlabel("Skaggs spatial information (bits/spike)")
ax.set_ylabel("units")
ax.set_title("A  Spatial information by group", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8)

# B: p-value vs SI
ax = axes[0, 1]
for m, c, lab in [(exc & is_place, "tab:blue", "exc place"),
                  (exc & ~is_place, "0.5", "exc non-place"),
                  (~exc, "tab:red", "inhibitory")]:
    ax.scatter(si[m], p_si[m], s=12, alpha=0.7, c=c, label=lab)
ax.axhline(0.05, color="k", ls="--", lw=1)
ax.set_yscale("log")
ax.set_xlabel("Skaggs SI (bits/spike)")
ax.set_ylabel("shuffle p-value (log)")
ax.set_title("B  Shuffle significance vs SI", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8)

# C: peak rate vs mean rate with criteria
ax = axes[1, 0]
for m, c, lab in [(exc & is_place, "tab:blue", "exc place"),
                  (exc & ~is_place, "0.5", "exc non-place"),
                  (~exc, "tab:red", "inhibitory")]:
    ax.scatter(mean_rate[m], peak_rate[m], s=12, alpha=0.7, c=c, label=lab)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.axvline(0.1, color="k", ls="--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("mean in-run firing rate (Hz)")
ax.set_ylabel("peak place-field rate (Hz)")
ax.set_title("C  Rate criteria", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8)

# D: direction-map correlation
ax = axes[1, 1]
dc = dir_corr[exc & is_place]
dc = dc[~np.isnan(dc)]
ax.hist(dc, bins=np.linspace(-1, 1, 40), color="tab:blue", alpha=0.8)
ax.axvline(np.median(dc), color="k", ls="--", lw=1,
           label=f"median = {np.median(dc):.2f}")
ax.axvline(0, color="0.5", lw=0.5)
ax.set_xlabel("correlation between direction-specific rate maps")
ax.set_ylabel("place cells")
ax.set_title("D  Directionality of place fields", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=9)

fig.suptitle("Place-cell identification and statistics — 500 circular time-shifts per unit",
             fontsize=12)
fig.savefig("figures/fig4_place_cell_stats.png", dpi=150)
print("saved figures/fig4_place_cell_stats.png")
print(f"exc place {np.sum(exc&is_place)}/{np.sum(exc)}; "
      f"median SI place={np.median(si[exc&is_place]):.2f}, "
      f"non-place={np.median(si[exc&~is_place]):.2f}, inh={np.median(si[~exc]):.3f}")
print(f"median dir corr (place cells with both maps): {np.median(dc):.2f}, n={len(dc)}")
