# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demonstrating Grid Cells in the Medial Entorhinal Cortex
#
# **Dataset:** DANDI dandiset [000582](https://dandiarchive.org/dandiset/000582),
# *"Conjunctive Representation of Position, Direction, and Velocity in Entorhinal
# Cortex"* (Sargolini et al., *Science*, 2006). Extracellular recordings from the
# dorsocaudal medial entorhinal cortex (MEC) of Long-Evans rats foraging in a
# 100 × 100 cm open field, with two-LED head tracking at 50 Hz.
#
# **Goal.** Grid cells fire whenever the animal occupies any vertex of a periodic
# triangular (hexagonal) lattice tiling the environment. We demonstrate this by
# computing, for each unit:
#
# 1. an occupancy-normalized 2-D firing-rate map,
# 2. the spatial autocorrelogram of that map (hexagonal symmetry is the signature
#    of a grid cell), and
# 3. the **gridness score** (Sargolini 2006), validated against a spike-time
#    shuffle null distribution.
#
# All data are streamed from the DANDI S3 store with `remfile` + local disk
# caching; nothing is downloaded in full. Analysis helpers live in `gridcells.py`.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from tqdm import tqdm

import gridcells as gc

BINS = 40                       # 40 x 40 spatial bins over the box
BOX = (-50.0, 50.0)             # box extent in cm
BIN_SIZE = (BOX[1] - BOX[0]) / BINS   # 2.5 cm per bin
SIGMA = 1.0                     # Gaussian smoothing (bins)
MIN_SPIKES = 100
N_SHUFFLE = 30

# %% [markdown]
# ## 1. Load and inspect one session
#
# Each NWB file stores head-tracking position under `processing/behavior/Position`
# (one `SpatialSeries` per LED; we average the LEDs to get the head centroid) and
# sorted spikes in the `units` table. We first validate a single rich session.

# %%
path = "sub-11265/sub-11265_ses-06020601_behavior+ecephys.nwb"
nwb = gc.load_session(path)
t, xy = gc.extract_position(nwb)
units = gc.extract_units(nwb)
print(f"duration = {t[-1] - t[0]:.0f} s, "
      f"position samples = {len(t)}, units = {len(units)}")
print(f"x range {xy[:,0].min():.0f}..{xy[:,0].max():.0f} cm, "
      f"y range {xy[:,1].min():.0f}..{xy[:,1].max():.0f} cm")

# %% [markdown]
# ### Validate the raw behavioral trajectory and spike coverage

# %%
occ, ex, ey = gc.occupancy_map(t, xy, BINS, BOX, sigma=SIGMA)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
ax[0].plot(xy[:, 0], xy[:, 1], lw=0.3, color="0.4")
ax[0].set_aspect("equal"); ax[0].set_title("Head trajectory (10 min)")
ax[0].set_xlabel("x (cm)"); ax[0].set_ylabel("y (cm)")
im = ax[1].imshow(occ.T, origin="lower", cmap="viridis",
                  extent=[BOX[0], BOX[1], BOX[0], BOX[1]])
ax[1].set_title("Occupancy (s)"); plt.colorbar(im, ax=ax[1], fraction=0.046)
ax[1].set_xlabel("x (cm)"); ax[1].set_ylabel("y (cm)")
fig.tight_layout(); fig.savefig("fig0_trajectory_occupancy.png", dpi=120)
print(f"occupancy total = {np.nansum(occ):.0f} s over "
      f"{np.isfinite(occ).sum()} visited bins")

# %% [markdown]
# The animal samples the whole box fairly uniformly, so occupancy normalization is
# well posed. Now compute rate map, autocorrelogram, and gridness for every unit.

# %%
for u in units:
    st = u["spike_times"]
    rm, ac, g, si = gc.analyze_unit(st, t, xy, occ, ex, BIN_SIZE, SIGMA)
    print(f"  {u['name']:6s}  gridness={g['gridness']:+.2f}  "
          f"spacing={g['spacing_cm']:.0f} cm  SI={si:.2f} bits/spk  "
          f"peak={np.nanmax(rm):.1f} Hz  n_spikes={len(st)}")

# %% [markdown]
# ## 2. The gridness score and the shuffle control
#
# A grid cell's autocorrelogram has a central peak surrounded by six peaks at
# ~60° spacing. The **gridness score** rotates an annular ring of the
# autocorrelogram by 30/60/90/120/150° and computes
# `min(r60, r120) − max(r30, r90, r150)`: hexagonal maps correlate strongly at
# 60°/120° and poorly at 30°/90°/150°, giving a positive score.
#
# To decide which scores are significant we build a null distribution by
# circularly shifting each cell's spike train by a random offset (destroying the
# spike–position relationship) and recomputing gridness. Cells whose observed
# gridness exceeds the 95th percentile of the pooled null are classified as grid
# cells.

# %% [markdown]
# ## 3. Pool the richest sessions
#
# The full analysis (`run_analysis.py`) processes the eight sessions with the most
# simultaneously recorded units, pools all units, runs the shuffle control, and
# caches everything to `results.pkl`. We load that cache here (run the script
# first if it is missing).

# %%
import os, pickle, subprocess
if not os.path.exists("results.pkl"):
    subprocess.run(["python", "run_analysis.py"], check=True)
with open("results.pkl", "rb") as f:
    R = pickle.load(f)
records = R["records"]
grid = np.array([r["gridness"] for r in records], float)
spacing = np.array([r["spacing_cm"] for r in records], float)
si = np.array([r["spatial_info"] for r in records], float)
is_grid = grid > R["threshold"]
print(f"pooled units: {len(records)}")
print(f"shuffle 95th-pctile gridness threshold: {R['threshold']:.2f}")
print(f"grid cells: {int(is_grid.sum())} / {len(records)} "
      f"({100 * is_grid.mean():.0f}%)")
print(f"median grid spacing: {np.nanmedian(spacing[is_grid]):.0f} cm")

# %% [markdown]
# ## 4. Figures
#
# `make_figures.py` renders the full figure set from the cached results:
#
# * `fig1_example_grid_cells.png` — six clearest grid cells (trajectory+spikes,
#   rate map, autocorrelogram).
# * `fig2_population_summary.png` — gridness vs. shuffle null, grid-spacing
#   distribution, spatial-info vs. gridness.
# * `fig3_grid_cell_gallery.png` — autocorrelograms of every identified grid cell.
# * `fig4_exemplar_with_control.png` — one exemplar with its own shuffle null.

# %%
subprocess.run(["python", "make_figures.py"], check=True)

# %% [markdown]
# ### Example grid cells
# ![example grid cells](fig1_example_grid_cells.png)
#
# ### Population summary
# ![population summary](fig2_population_summary.png)
#
# ### All identified grid cells
# ![gallery](fig3_grid_cell_gallery.png)
#
# ### Exemplar with shuffle control
# ![exemplar](fig4_exemplar_with_control.png)

# %% [markdown]
# ## Conclusion
#
# The spatial autocorrelograms show the hallmark six-fold (hexagonal) symmetry of
# grid cells, with grid spacings in the ~40–70 cm range expected for dorsocaudal
# MEC. A substantial fraction of MEC units exceed the shuffle-based gridness
# threshold, reproducing the central result of Sargolini et al. (2006): the MEC
# contains a population of cells that encode the animal's location through a
# periodic triangular lattice, independent of the specific environment.
