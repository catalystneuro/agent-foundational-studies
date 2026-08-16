# Grid Cells in the Medial Entorhinal Cortex (DANDI:000582)

## Dataset

[**DANDI:000582**](https://dandiarchive.org/dandiset/000582) — *"Conjunctive
Representation of Position, Direction, and Velocity in the Entorhinal Cortex"*
(Sargolini et al., 2006, Moser lab). Long-Evans rats foraged in a 1 m × 1 m open field
while single units were recorded from the dorsocaudal medial entorhinal cortex (MEC).
Head position was tracked by two LEDs at 50 Hz. Files were streamed directly from S3
with `remfile` disk caching (no full downloads) and handled with `pynapple`.

## What was analyzed

Grid cells fire whenever the animal occupies any vertex of a triangular (hexagonal)
lattice tiling the environment, providing a periodic metric for space (Hafting et al.,
2005). For every unit I built a smoothed 2-D firing-rate map (spike counts divided by
occupancy time), computed its unbiased Pearson spatial autocorrelogram, and quantified
the six-fold rotational symmetry with the standard **grid score**,
`min(r60, r120) − max(r30, r90, r150)`, over the ring of the six nearest autocorrelogram
fields. Statistical significance was assessed against a null distribution built by
circularly shifting each cell's spike train relative to the trajectory (which destroys
spatial tuning while preserving spike statistics); the 95th percentile of the pooled
null defined the grid-cell threshold. The analysis pooled **91 units across 6 sessions
from 2 rats**.

The pipeline is modular: `gridlib.py` (streaming/loading, rate maps, spatial
autocorrelation, grid score, shuffling), `run_analysis.py` (multi-session batch with the
shuffle test), `visualize.py` (population figures), and the consolidated jupytext
notebook `grid_cells_mec_dandi000582.py` / `.ipynb`.

## Key finding

**24 of 91 MEC units (26%) were classified as grid cells** (grid score above the
shuffle-derived threshold of 1.01), a fraction consistent with the known prevalence of
grid cells in dorsocaudal MEC. Their spatial autocorrelograms show the hallmark
hexagonal ring of six surrounding fields (`figures/01_grid_cell_gallery.png`,
`figures/05_example_cell_detail.png`), the best cells reaching grid scores up to 1.87.
Grid spacing ranged from ~43 to ~81 cm (median 54 cm) with a suggestion of two discrete
modules (~45 cm and ~68 cm), and grid orientations clustered near 20–30°, matching the
modular, orientation-aligned organization reported for MEC grids. The observed
grid-score distribution has a clear heavy right tail beyond the null
(`figures/02_grid_score_significance.png`), and grid cells were found in both rats,
confirming a robust demonstration of grid coding in the medial entorhinal cortex from
real DANDI data.

## Figures

- `00_validation_top_grid_cells.png` — prototyping panel (trajectory+spikes, rate map, autocorrelogram) for the richest session.
- `01_grid_cell_gallery.png` — rate maps and hexagonal autocorrelograms of the top classified grid cells.
- `02_grid_score_significance.png` — observed grid scores vs. shuffled null; ECDF with threshold.
- `03_population_properties.png` — grid spacing, orientation, and grid score vs. spatial information.
- `04_trajectory_and_spikes.png` — raw foraging path with example spike locations.
- `05_example_cell_detail.png` — annotated rate map and autocorrelogram for one grid cell.

## Reproduce

```bash
python run_analysis.py      # streams 6 sessions, computes metrics + shuffle null -> results/
python visualize.py         # writes population figures
# or run the full narrative end-to-end:
python grid_cells_mec_dandi000582.py
```
