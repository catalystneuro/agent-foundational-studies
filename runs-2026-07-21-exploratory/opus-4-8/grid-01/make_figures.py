"""Generate all figures for the grid-cell demonstration from results.pkl."""
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results.pkl", "rb") as f:
    R = pickle.load(f)

records = R["records"]
thr = R["threshold"]
BOX = R["BOX"]
grid = np.array([r["gridness"] for r in records], float)
order = np.argsort(-np.where(np.isfinite(grid), grid, -9))


def draw_examples(recs, sessions_by_path, fname, title):
    """Three-row panel: trajectory+spikes / rate map / autocorrelogram."""
    n = len(recs)
    fig, axes = plt.subplots(3, n, figsize=(2.7 * n, 8.2))
    if n == 1:
        axes = axes[:, None]
    for j, r in enumerate(recs):
        s = sessions_by_path[[k for k in sessions_by_path
                              if k.split("/")[-1].replace("_behavior+ecephys.nwb", "")
                              == r["session"]][0]]
        xy = s["xy"]
        # trajectory + spikes
        ax = axes[0, j]
        ax.plot(xy[:, 0], xy[:, 1], color="0.82", lw=0.35, zorder=1)
        ax.scatter(r["spike_x"], r["spike_y"], s=3, c="#d1352b", zorder=2, alpha=0.7)
        ax.set_aspect("equal")
        ax.set_xlim(BOX)
        ax.set_ylim(BOX)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{r['subject']}  {r['name']}", fontsize=9)
        # rate map
        ax = axes[1, j]
        rm = r["rate_map"]
        ax.imshow(rm.T, origin="lower", cmap="jet",
                  extent=[BOX[0], BOX[1], BOX[0], BOX[1]])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"peak {r['peak_rate']:.1f} Hz\nSI {r['spatial_info']:.2f} bits/spk",
                     fontsize=9)
        # autocorrelogram
        ax = axes[2, j]
        ax.imshow(r["autocorr"].T, origin="lower", cmap="jet", vmin=-0.5, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"gridness {r['gridness']:.2f}\nspacing {r['spacing_cm']:.0f} cm",
                     fontsize=9)
    for ax, lab in zip(axes[:, 0],
                       ["trajectory\n+ spikes", "firing-rate\nmap", "spatial\nautocorr."]):
        ax.set_ylabel(lab, fontsize=10, rotation=90, labelpad=8)
        ax.set_yticks([])
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print("saved", fname)


sessions_by_path = {s["path"]: s for s in R["sessions"]}

# ---- Figure 1: best example grid cells -------------------------------------
best = [records[i] for i in order[:6]]
draw_examples(best, sessions_by_path, "fig1_example_grid_cells.png",
              "Grid cells in medial entorhinal cortex (DANDI 000582, Sargolini 2006)")

# ---- Figure 2: population summary ------------------------------------------
spacing = np.array([r["spacing_cm"] for r in records], float)
si = np.array([r["spatial_info"] for r in records], float)
is_grid = grid > thr

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
# gridness distribution + shuffle null + threshold
ax[0].hist(R["pooled_null"], bins=30, density=True, color="0.7",
           alpha=0.8, label="shuffled null")
ax[0].hist(grid[np.isfinite(grid)], bins=20, density=True, color="#2c7fb8",
           alpha=0.65, label="observed cells")
ax[0].axvline(thr, color="crimson", ls="--", lw=2,
              label=f"95th pctile = {thr:.2f}")
ax[0].set_xlabel("gridness score")
ax[0].set_ylabel("density")
ax[0].set_title("Gridness vs. shuffled null")
ax[0].legend(fontsize=8)
# grid-cell spacing distribution
ax[1].hist(spacing[is_grid], bins=np.arange(20, 90, 5), color="#31a354",
           edgecolor="w")
ax[1].set_xlabel("grid spacing (cm)")
ax[1].set_ylabel("# grid cells")
ax[1].set_title(f"Grid spacing (n={int(is_grid.sum())} grid cells)\n"
                f"median {np.nanmedian(spacing[is_grid]):.0f} cm")
# spatial info vs gridness
ax[2].scatter(grid[~is_grid], si[~is_grid], s=22, c="0.6", label="non-grid")
ax[2].scatter(grid[is_grid], si[is_grid], s=28, c="#31a354",
              edgecolor="k", lw=0.4, label="grid")
ax[2].axvline(thr, color="crimson", ls="--", lw=1.5)
ax[2].set_xlabel("gridness score")
ax[2].set_ylabel("spatial information (bits/spike)")
ax[2].set_title("Spatial info vs. gridness")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig2_population_summary.png", dpi=130)
plt.close(fig)
print("saved fig2_population_summary.png")

# ---- Figure 3: gallery of all grid cells' autocorrelograms -----------------
gcells = [records[i] for i in order if is_grid[i]]
ncol = 6
nrow = int(np.ceil(len(gcells) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.2 * nrow))
axes = np.atleast_2d(axes)
for k in range(nrow * ncol):
    ax = axes.flat[k]
    ax.axis("off")
    if k < len(gcells):
        r = gcells[k]
        ax.imshow(r["autocorr"].T, origin="lower", cmap="jet", vmin=-0.5, vmax=1.0)
        ax.set_title(f"{r['name']} g={r['gridness']:.2f}\n{r['spacing_cm']:.0f}cm",
                     fontsize=8)
fig.suptitle(f"Spatial autocorrelograms of all {len(gcells)} identified grid cells",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig3_grid_cell_gallery.png", dpi=130)
plt.close(fig)
print("saved fig3_grid_cell_gallery.png")

# ---- Figure 4: one exemplar with shuffle null ------------------------------
ex = best[0]
fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
s = sessions_by_path[[k for k in sessions_by_path
                      if k.split("/")[-1].replace("_behavior+ecephys.nwb", "")
                      == ex["session"]][0]]
xy = s["xy"]
ax[0].plot(xy[:, 0], xy[:, 1], color="0.82", lw=0.35)
ax[0].scatter(ex["spike_x"], ex["spike_y"], s=4, c="#d1352b", alpha=0.7)
ax[0].set_aspect("equal")
ax[0].set_title(f"trajectory + spikes\n{ex['subject']} {ex['name']}")
ax[0].set_xticks([]); ax[0].set_yticks([])
im1 = ax[1].imshow(ex["rate_map"].T, origin="lower", cmap="jet",
                   extent=[BOX[0], BOX[1], BOX[0], BOX[1]])
ax[1].set_title(f"rate map (peak {ex['peak_rate']:.1f} Hz)")
ax[1].set_xticks([]); ax[1].set_yticks([])
plt.colorbar(im1, ax=ax[1], fraction=0.046)
im2 = ax[2].imshow(ex["autocorr"].T, origin="lower", cmap="jet", vmin=-0.5, vmax=1)
ax[2].set_title(f"autocorrelogram\ngridness {ex['gridness']:.2f}")
ax[2].set_xticks([]); ax[2].set_yticks([])
plt.colorbar(im2, ax=ax[2], fraction=0.046)
ax[3].hist(ex["null_gridness"][np.isfinite(ex["null_gridness"])], bins=12,
           color="0.7", label="shuffled")
ax[3].axvline(ex["gridness"], color="crimson", lw=2.5,
              label=f"observed {ex['gridness']:.2f}")
ax[3].set_xlabel("gridness")
ax[3].set_ylabel("# shuffles")
ax[3].set_title("cell vs. its own shuffle null")
ax[3].legend(fontsize=8)
fig.suptitle("Exemplar grid cell with hexagonal firing and shuffle control", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig4_exemplar_with_control.png", dpi=130)
plt.close(fig)
print("saved fig4_exemplar_with_control.png")

# ---- text summary ----------------------------------------------------------
print(f"\nTotal units: {len(records)}  |  grid cells: {int(is_grid.sum())} "
      f"({100*is_grid.mean():.0f}%)")
print(f"Median grid spacing: {np.nanmedian(spacing[is_grid]):.0f} cm")
print(f"Grid-cell median SI: {np.nanmedian(si[is_grid]):.2f} bits/spike")
