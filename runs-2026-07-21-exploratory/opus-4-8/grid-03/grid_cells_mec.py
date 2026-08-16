# %% [markdown]
# # Grid Cells in the Medial Entorhinal Cortex
#
# This notebook demonstrates **grid cells** in the medial entorhinal cortex (MEC)
# using real extracellular recordings from the DANDI Archive, dandiset
# [**000582**](https://dandiarchive.org/dandiset/000582) — *"Conjunctive
# Representation of Position, Direction, and Velocity in Entorhinal Cortex"*
# (Sargolini / Moser lab). Long-Evans rats foraged for food in a 1 x 1 m open
# arena while single units were recorded from the dorsocaudal MEC and the animal's
# position was tracked with an LED at ~50 Hz.
#
# A grid cell fires whenever the animal occupies any vertex of a triangular
# (hexagonal) lattice that tiles the environment. We demonstrate this by:
#
# 1. Building smoothed 2-D firing-rate maps for every recorded unit.
# 2. Computing the spatial autocorrelogram of each rate map, whose six-fold
#    symmetric ring of peaks is the signature of a hexagonal grid.
# 3. Quantifying hexagonality with the standard **gridness score** (rotational
#    symmetry of the autocorrelogram at 60/120 deg vs 30/90/150 deg).
# 4. Establishing a significance threshold by **shuffling** spike times, then
#    classifying grid cells across multiple sessions and animals.
#
# Data are streamed directly from the DANDI S3 bucket with `remfile` + a local
# disk cache; nothing is downloaded in full. All spatial units are centimeters.

# %%
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

import gridlib as G  # local module: streaming loaders + rate-map / gridness routines

np.random.seed(0)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# %% [markdown]
# ## 1. Select sessions
#
# We analyze one representative session per animal across several animals, plus a
# few extra sessions from the two best-sampled animals, to pool grid cells across
# subjects rather than relying on a single recording.

# %%
assets = G.list_assets()
SESSIONS = [
    "sub-10704/sub-10704_ses-20060402_behavior+ecephys.nwb",
    "sub-10704/sub-10704_ses-25060402_behavior+ecephys.nwb",
    "sub-10884/sub-10884_ses-01080402_behavior+ecephys.nwb",
    "sub-10884/sub-10884_ses-03080405_behavior+ecephys.nwb",
    "sub-10697/sub-10697_ses-02030402_behavior+ecephys.nwb",
    "sub-10073/sub-10073_ses-17010302_behavior+ecephys.nwb",
]
print(f"{len(SESSIONS)} sessions selected")

# %% [markdown]
# ## 2. Load and inspect one session
#
# Position is an LED trace in a 1 m box (values span roughly -50..+50 cm). We drop
# untracked (NaN) samples and, for rate maps, keep only samples where the animal is
# moving faster than 2.5 cm/s so that firing during immobility does not distort the
# spatial maps.

# %%
url0 = G.s3_url(assets[SESSIONS[0]])
t0, xy0, units0, meta0 = G.load_session(url0)
mask0, speed0 = G.speed_filter(t0, xy0)
print(f"subject {meta0['subject']}  duration {t0[-1]-t0[0]:.0f}s  "
      f"tracked samples {len(t0)}  units {len(units0)}")
print(f"median running speed {np.median(speed0):.1f} cm/s, "
      f"fraction moving {mask0.mean():.2f}")

# %%
# Raw behavior + spikes overview for the strongest grid cell in this session.
fig = plt.figure(figsize=(13, 4))
gs = GridSpec(1, 3, width_ratios=[1.1, 1.1, 1.3], wspace=0.35)

ax = fig.add_subplot(gs[0])
ax.plot(xy0[:, 0], xy0[:, 1], color="0.6", lw=0.4)
ax.set(title="Foraging trajectory", xlabel="x (cm)", ylabel="y (cm)")
ax.set_aspect("equal")

# example unit t2c1 (identified below as a grid cell)
ex = "t2c1"
sx = np.interp(units0[ex], t0, xy0[:, 0])
sy = np.interp(units0[ex], t0, xy0[:, 1])
ax = fig.add_subplot(gs[1])
ax.plot(xy0[:, 0], xy0[:, 1], color="0.85", lw=0.4)
ax.plot(sx, sy, ".", color="crimson", ms=2.5)
ax.set(title=f"Spikes of unit {ex} (n={len(units0[ex])})", xlabel="x (cm)")
ax.set_aspect("equal")

ax = fig.add_subplot(gs[2])
ax.plot(t0[:3000], speed0[:3000], lw=0.6)
ax.axhline(G.SPEED_THRESH, color="r", ls="--", lw=1, label=f"{G.SPEED_THRESH} cm/s")
ax.set(title="Running speed (first 60 s)", xlabel="time (s)", ylabel="speed (cm/s)")
ax.legend(fontsize=8)
fig.suptitle(f"Session {SESSIONS[0].split('/')[-1]}", y=1.02)
fig.savefig("fig1_behavior_overview.png", bbox_inches="tight")
plt.close(fig)
print("saved fig1_behavior_overview.png")

# %% [markdown]
# ## 3. Rate map, autocorrelogram, and gridness for the example cell
#
# The spatial autocorrelogram of a grid cell's rate map shows a central peak
# surrounded by a hexagonal ring of six peaks. The **gridness score** rotates the
# autocorrelogram's annular region and measures how much more it correlates with
# itself at 60 and 120 deg (grid-consistent) than at 30, 90, 150 deg.

# %%
rm = G.rate_map(units0[ex], t0, xy0, mask0)[0]
ac = G.spatial_autocorr(rm)
gscore, spacing = G.grid_score(ac)
print(f"unit {ex}: gridness = {gscore:.2f}, peak rate = {np.nanmax(rm):.1f} Hz")

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].plot(xy0[:, 0], xy0[:, 1], color="0.85", lw=0.4)
axs[0].plot(sx, sy, ".", color="crimson", ms=2.0)
axs[0].set(title=f"{ex}: trajectory + spikes"); axs[0].set_aspect("equal"); axs[0].axis("off")

im = axs[1].imshow(rm, origin="lower", cmap="jet")
axs[1].set(title=f"Firing-rate map (peak {np.nanmax(rm):.0f} Hz)"); axs[1].axis("off")
fig.colorbar(im, ax=axs[1], fraction=0.046, label="Hz")

im = axs[2].imshow(ac, origin="lower", cmap="jet", vmin=-1, vmax=1)
axs[2].set(title=f"Spatial autocorrelogram\ngridness = {gscore:.2f}"); axs[2].axis("off")
fig.colorbar(im, ax=axs[2], fraction=0.046, label="corr")
fig.tight_layout()
fig.savefig("fig2_example_grid_cell.png", bbox_inches="tight")
plt.close(fig)
print("saved fig2_example_grid_cell.png")

# %% [markdown]
# ## 4. Analyze all cells across all sessions
#
# For every unit we compute a rate map, its autocorrelogram, and a gridness score.
# We also run a spike-time **shuffle** control: each cell's spike train is circularly
# shifted by a random offset (>= 20 s) many times, and the gridness is recomputed.
# The 95th percentile of the pooled shuffle distribution is the significance
# threshold above which a cell is classified as a grid cell.

# %%
N_SHUFFLE = 50
MIN_SPIKES = 100
MIN_SHIFT = 20.0  # s

results = []          # per real cell
shuffle_scores = []   # pooled null distribution

for spath in tqdm(SESSIONS, desc="sessions"):
    url = G.s3_url(assets[spath])
    t, xy, units, meta = G.load_session(url)
    mask, _ = G.speed_filter(t, xy)
    T = t[-1] - t[0]
    for name, st in units.items():
        if len(st) < MIN_SPIKES:
            continue
        rmap = G.rate_map(st, t, xy, mask)[0]
        acorr = G.spatial_autocorr(rmap)
        gsc, sp = G.grid_score(acorr)
        results.append(dict(session=spath.split("/")[-1], subject=meta["subject"],
                            unit=name, n=len(st), peak=np.nanmax(rmap),
                            grid=gsc, spacing=sp, ratemap=rmap, acorr=acorr))
        # shuffle null
        for _ in range(N_SHUFFLE):
            shift = np.random.uniform(MIN_SHIFT, T - MIN_SHIFT)
            sh = t[0] + np.mod(st - t[0] + shift, T)
            g_sh = G.grid_score(G.spatial_autocorr(G.rate_map(sh, t, xy, mask)[0]))[0]
            if np.isfinite(g_sh):
                shuffle_scores.append(g_sh)

shuffle_scores = np.array(shuffle_scores)
GRID_THRESH = np.nanpercentile(shuffle_scores, 95)
real_grid = np.array([r["grid"] for r in results])
n_grid = np.sum(real_grid >= GRID_THRESH)
print(f"\n{len(results)} units analyzed across {len(SESSIONS)} sessions")
print(f"shuffle 95th-percentile gridness threshold = {GRID_THRESH:.2f}")
print(f"grid cells: {n_grid}/{len(results)} "
      f"({100*n_grid/len(results):.0f}%)")

# %% [markdown]
# ## 5. Gridness distribution vs the shuffle null
#
# Real cells populate a heavy right tail well beyond the shuffled null, and a clear
# subset exceeds the 95th-percentile threshold — these are the grid cells.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.linspace(-1.2, 1.6, 40)
ax.hist(shuffle_scores, bins=bins, density=True, color="0.7",
        alpha=0.8, label=f"shuffled null (n={len(shuffle_scores)})")
ax.hist(real_grid, bins=bins, density=True, histtype="step", color="navy",
        lw=2, label=f"real cells (n={len(results)})")
ax.axvline(GRID_THRESH, color="crimson", ls="--", lw=2,
           label=f"95th pct threshold = {GRID_THRESH:.2f}")
ax.set(xlabel="gridness score", ylabel="probability density",
       title="Grid cells exceed the spike-shuffle null")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("fig3_gridness_distribution.png", bbox_inches="tight")
plt.close(fig)
print("saved fig3_gridness_distribution.png")

# %% [markdown]
# ## 6. Gallery of the strongest grid cells
#
# Rate map (left of each pair) and spatial autocorrelogram (right) for the top grid
# cells, sorted by gridness. The hexagonal ring of six autocorrelation peaks is the
# defining signature of a grid cell.

# %%
top = sorted([r for r in results if r["grid"] >= GRID_THRESH],
             key=lambda r: -r["grid"])[:8]
ncol = 4
nrow = int(np.ceil(len(top) / ncol))
fig, axs = plt.subplots(nrow * 2, ncol, figsize=(3 * ncol, 3 * nrow))
for i, r in enumerate(top):
    c = i % ncol
    rblk = (i // ncol) * 2
    a1 = axs[rblk, c]
    a1.imshow(r["ratemap"], origin="lower", cmap="jet")
    a1.set_title(f"{r['subject']}/{r['unit']}\npeak {r['peak']:.0f} Hz", fontsize=8)
    a1.axis("off")
    a2 = axs[rblk + 1, c]
    a2.imshow(r["acorr"], origin="lower", cmap="jet", vmin=-1, vmax=1)
    a2.set_title(f"gridness {r['grid']:.2f}", fontsize=8)
    a2.axis("off")
for j in range(len(top), nrow * ncol):  # blank unused columns
    c = j % ncol; rblk = (j // ncol) * 2
    axs[rblk, c].axis("off"); axs[rblk + 1, c].axis("off")
fig.suptitle("Grid cells: firing-rate maps (top) and spatial autocorrelograms (bottom)",
             y=1.005)
fig.tight_layout()
fig.savefig("fig4_grid_cell_gallery.png", bbox_inches="tight")
plt.close(fig)
print("saved fig4_grid_cell_gallery.png")

# %% [markdown]
# ## 7. Summary table

# %%
print(f"{'subject':>9} {'session':>22} {'unit':>6} {'nspk':>6} "
      f"{'peakHz':>7} {'gridness':>9} {'grid?':>6}")
for r in sorted(results, key=lambda r: -r["grid"]):
    flag = "YES" if r["grid"] >= GRID_THRESH else ""
    print(f"{r['subject']:>9} {r['session'][:22]:>22} {r['unit']:>6} {r['n']:>6} "
          f"{r['peak']:>7.1f} {r['grid']:>9.2f} {flag:>6}")

# %% [markdown]
# ## Conclusion
#
# Across several animals and sessions from dandiset 000582, a substantial fraction
# of dorsocaudal MEC units fire in a periodic, hexagonally arranged set of fields
# that tiles the open arena. Their spatial autocorrelograms show the characteristic
# six-peak ring, and their gridness scores exceed a spike-time-shuffle null,
# confirming the classic **grid cell** phenotype in the medial entorhinal cortex.
