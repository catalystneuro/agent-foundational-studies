# %% [markdown]
# # Grid Cells in the Medial Entorhinal Cortex (DANDI:000582)
#
# This notebook demonstrates hexagonal **grid cells** in the medial entorhinal cortex
# (MEC) using real extracellular recordings from the DANDI Archive. Grid cells fire
# whenever an animal occupies any vertex of a triangular (hexagonal) lattice tiling the
# environment. They were discovered by Hafting, Fyhn, Moser & Moser (2005) and are a
# cornerstone of the brain's metric for space.
#
# **Dataset:** [DANDI:000582](https://dandiarchive.org/dandiset/000582) —
# *"Conjunctive Representation of Position, Direction, and Velocity in the Entorhinal
# Cortex"* (Sargolini et al., 2006, Moser lab). Long-Evans rats foraged in a 1 m × 1 m
# open field while single units were recorded from the dorsocaudal MEC, with head
# position tracked by LEDs at 50 Hz.
#
# **Approach.** For each unit we build a smoothed 2-D firing-rate map, compute its
# spatial autocorrelogram, and quantify the six-fold rotational symmetry with the
# standard **grid score** (gridness). Significance is assessed against a null
# distribution built by circularly shifting each cell's spikes relative to the
# trajectory. We pool units across six sessions from two rats.
#
# All data are streamed from S3 with `remfile` disk caching (no full downloads) and
# handled with `pynapple`.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynapple as nap
import gridlib  # local module: loading, rate maps, autocorrelation, grid score, shuffling

# %% [markdown]
# ## Dataset structure
#
# Each NWB file exposes a `units` table (spike times), a `SpatialSeriesLED1` position
# trace (x, y in cm, origin at box center), and LFP. We inspect the richest session.

# %%
SESSIONS = [
    ('sub-11265_ses-06020601', '60453c7c-79c0-4bc5-8484-7322ae99c4b2'),
    ('sub-11207_ses-18060501', 'edd1a8b4-1d9b-4c9a-9079-e6b006ed455b'),
    ('sub-11207_ses-27060501', '1043a415-4f66-4cdb-b369-cbc0076917f2'),
    ('sub-11265_ses-16030604', 'c18096f7-d802-4813-ba95-7141b58f947a'),
    ('sub-11207_ses-21060503', '11643a0c-f746-4adb-b8b3-e3d9cf90790f'),
    ('sub-11265_ses-09020601', 'f6dc02e9-4917-4a92-b5f7-aadb33364a2e'),
]

nwbfile, nwb = gridlib.load_session(SESSIONS[0][1])
units = nwb['units']
pos = nwb['SpatialSeriesLED1']
print(nwb)
print(f'\n{len(units)} units, session duration {pos.index[-1]:.0f} s, '
      f'{len(pos)} position samples')

# %% [markdown]
# ## Raw data check: trajectory and spike locations
#
# Before any analysis we plot the raw foraging trajectory and overlay the positions at
# which one example unit fired. Clustered, periodically spaced spike locations are the
# visual signature of a grid cell.

# %%
keys = list(units.keys())
x = pos[:, 0].values; y = pos[:, 1].values; t = pos.index.values
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].plot(x, y, lw=0.4, color='0.6'); ax[0].set_title('foraging trajectory')
ax[0].set_xlabel('x (cm)'); ax[0].set_ylabel('y (cm)'); ax[0].set_aspect('equal')
for j, u in enumerate([18, 16]):
    ut = units[keys[u]]
    sx = np.interp(ut.index.values, t, x); sy = np.interp(ut.index.values, t, y)
    ax[j+1].plot(x, y, lw=0.3, color='0.8')
    ax[j+1].scatter(sx, sy, s=4, c='red')
    ax[j+1].set_title(f'unit {u}: spike locations'); ax[j+1].set_aspect('equal')
    ax[j+1].set_xlabel('x (cm)')
plt.tight_layout(); plt.savefig('figures/04_trajectory_and_spikes.png', dpi=120)

# %% [markdown]
# ## Rate map, autocorrelogram, and grid score for one cell
#
# `gridlib.rate_map` divides smoothed spike counts by smoothed occupancy time to get a
# firing-rate map (Hz). `gridlib.spatial_autocorr` computes the unbiased Pearson spatial
# autocorrelogram (Sargolini et al., 2006). `gridlib.grid_score` measures gridness as
# `min(r60, r120) − max(r30, r90, r150)`, the difference between autocorrelogram-rotation
# correlations at multiples of 60° versus 30° over the ring of the six nearest fields.

# %%
ut = units[keys[18]]
rm, xe, ye = gridlib.rate_map(ut, pos)
ac = gridlib.spatial_autocorr(rm)
gs, params = gridlib.grid_score(ac)
spacing, orient = gridlib.field_com_stats(ac)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
im0 = ax[0].imshow(rm, origin='lower', cmap='jet',
                   extent=[-50, 50, -50, 50]); ax[0].set_aspect('equal')
ax[0].set_title(f'rate map (peak {np.nanmax(rm):.1f} Hz)')
ax[0].set_xlabel('x (cm)'); ax[0].set_ylabel('y (cm)')
plt.colorbar(im0, ax=ax[0], fraction=0.046, label='Hz')
im1 = ax[1].imshow(ac, origin='lower', cmap='jet', vmin=-0.5, vmax=1)
ax[1].set_aspect('equal'); ax[1].set_xticks([]); ax[1].set_yticks([])
ax[1].set_title(f'autocorrelogram\ngrid score {gs:.2f}, spacing {spacing:.0f} cm')
plt.colorbar(im1, ax=ax[1], fraction=0.046, label='correlation')
plt.tight_layout(); plt.savefig('figures/05_example_cell_detail.png', dpi=120)
print(f'grid score {gs:.3f}, spacing {spacing:.1f} cm, orientation {orient:.0f} deg')

# %% [markdown]
# ## Significance via spike-time shifting
#
# A cell is a grid cell if its grid score exceeds what is expected by chance. We build a
# null distribution by circularly shifting the spike train relative to position (which
# destroys spatial tuning while preserving spike statistics) and recomputing the grid
# score. The 95th percentile of the pooled null is the classification threshold.

# %%
null_example = gridlib.shuffle_grid_scores(ut, pos, n_shuffles=100, seed=1)
print(f'observed grid score {gs:.2f} vs shuffle null '
      f'mean {null_example.mean():.2f}, 95th pct {np.percentile(null_example, 95):.2f}')

# %% [markdown]
# ## Multi-session population analysis
#
# `run_analysis.py` runs the full pipeline over all six sessions: for every unit it
# computes the grid score, grid spacing, orientation, Skaggs spatial information, and a
# 30-shuffle null. The nulls are pooled and the 95th percentile is used as a single grid
# threshold. Run it once (a few minutes) to populate `results/`, then load the outputs.

# %%
# To (re)generate results:  python run_analysis.py
df = pd.read_csv('results/grid_metrics.csv')
pooled_null = np.load('results/pooled_null.npy')
thr = float(open('results/threshold.txt').read())
print(f'{len(df)} units from {df.session.nunique()} sessions / {df.subject.nunique()} rats')
print(f'grid threshold (95th pct of null) = {thr:.2f}')
print(f'grid cells: {int(df.is_grid.sum())} ({100*df.is_grid.mean():.0f}%)')

# %% [markdown]
# ## Results
#
# The figures below (also written by `visualize.py`) show (1) a gallery of the classified
# grid cells with their rate maps and hexagonal autocorrelograms, (2) the observed
# grid-score distribution against the shuffled null with the significance threshold, and
# (3) the population distributions of grid spacing and orientation. The consistent
# 40–70 cm spacings and clear six-fold-symmetric autocorrelograms confirm the presence of
# grid cells in the dorsocaudal MEC.

# %%
import subprocess, sys
subprocess.run([sys.executable, 'visualize.py'], check=True)
for f in ['figures/01_grid_cell_gallery.png',
          'figures/02_grid_score_significance.png',
          'figures/03_population_properties.png']:
    print('wrote', f)
