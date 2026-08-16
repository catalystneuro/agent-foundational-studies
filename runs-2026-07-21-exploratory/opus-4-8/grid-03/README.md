# Grid Cells in the Medial Entorhinal Cortex

This analysis demonstrates **grid cells** in the medial entorhinal cortex (MEC)
using real recordings from the DANDI Archive.

## Dataset

[**DANDI:000582**](https://dandiarchive.org/dandiset/000582) — *"Conjunctive
Representation of Position, Direction, and Velocity in Entorhinal Cortex"*
(Sargolini / Moser lab). Long-Evans rats foraged for food in a 1 x 1 m open arena
while single units were recorded from the dorsocaudal medial entorhinal cortex and
the animal's position was tracked with a head-mounted LED at ~50 Hz. Files are
streamed directly from S3 with `remfile` and a local disk cache; nothing is
downloaded in full. Six sessions across four animals (subjects 10073, 10697,
10704, 10884) were analyzed.

## What was analyzed

For every well-sampled unit (>= 100 spikes) we built a speed-filtered, smoothed 2-D
firing-rate map, computed its spatial autocorrelogram, and quantified hexagonal
periodicity with the standard **gridness score** (rotational symmetry of the
autocorrelogram at 60/120 deg relative to 30/90/150 deg). A spike-time shuffle
control (circular time shifts, 50 per cell) built a null distribution, and the 95th
percentile of that null (gridness = 0.44) was used as the grid-cell threshold.

## Key finding

A clear subpopulation of MEC units fires in a periodic, hexagonally arranged set of
fields that tiles the arena: **8 of 31 units (26%)** exceeded the shuffle-based
gridness threshold, matching the grid-cell fraction expected in dorsocaudal MEC.
The best cells (gridness up to 1.22) show textbook spatial autocorrelograms with a
central peak surrounded by a ring of six evenly spaced peaks at ~60 deg intervals —
the defining signature of a grid cell. The real-cell gridness distribution has a
heavy right tail well beyond the shuffled null, confirming that the hexagonal
structure is not an artifact of firing rate or trajectory coverage.

## Files

- `grid_cells_mec.py` — consolidated jupytext analysis script (runs end-to-end).
- `grid_cells_mec.ipynb` — executed Jupyter notebook version.
- `gridlib.py` — streaming loaders and rate-map / autocorrelogram / gridness routines.
- `fig1_behavior_overview.png` — trajectory, spike positions, and running speed.
- `fig2_example_grid_cell.png` — one grid cell: spikes, rate map, autocorrelogram.
- `fig3_gridness_distribution.png` — real vs shuffled gridness with threshold.
- `fig4_grid_cell_gallery.png` — rate maps + autocorrelograms of the top grid cells.

## Reproducing

```bash
pip install pynapple lindi remfile pynwb h5py tqdm matplotlib scipy dandi
python grid_cells_mec.py
```

All spatial units are centimeters (the NWB position is labeled "meters" but the
recorded values span the 1 m box in cm, roughly -50..+50).
