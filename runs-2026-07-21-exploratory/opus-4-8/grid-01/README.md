# Grid Cells in the Medial Entorhinal Cortex

This analysis demonstrates **grid cells** in the medial entorhinal cortex (MEC)
using freely-moving rat recordings streamed from the DANDI Archive.

## Dataset

**DANDI dandiset [000582](https://dandiarchive.org/dandiset/000582)** —
*"Conjunctive Representation of Position, Direction, and Velocity in Entorhinal
Cortex"* (Sargolini et al., *Science*, 2006). Extracellular tetrode recordings
from the dorsocaudal MEC of Long-Evans rats foraging in a 100 × 100 cm open
field, with two-LED head tracking sampled at 50 Hz. NWB files were read directly
from the DANDI S3 store with `remfile` streaming plus local disk caching (no full
downloads). Eight sessions with the most simultaneously recorded units (subjects
11207 and 11265) were pooled, yielding **115 well-sampled units** (≥100 spikes).

## What was analyzed

For every unit we computed, entirely with NumPy/SciPy over Pynapple-loaded data:

1. an occupancy-normalized, Gaussian-smoothed **2-D firing-rate map** (2.5 cm bins);
2. the **spatial autocorrelogram** of that map (Pearson correlation at each spatial
   lag) — a hexagonal lattice of peaks is the defining signature of a grid cell;
3. the **gridness score** (Sargolini 2006), from rotating an annular ring of the
   autocorrelogram by 30/60/90/120/150° and taking `min(r60,r120) − max(r30,r90,r150)`;
4. **Skaggs spatial information** (bits/spike).

Significance was established with a **spike-time shuffle control**: each cell's
spikes were circularly shifted by a random offset (30 shuffles/cell), destroying
the spike–position relationship, and gridness was recomputed. The 95th percentile
of the pooled null distribution set the grid-cell threshold.

## Key finding

The spatial autocorrelograms of MEC units display the hallmark **six-fold
(hexagonal) symmetry** of grid cells — a central peak ringed by six peaks at ~60°
spacing that recurs periodically across the environment (see
`fig1_example_grid_cells.png` and `fig3_grid_cell_gallery.png`). Using the
shuffle-based threshold (gridness > 0.46, the 95th percentile of the null),
**22 of 115 units (19%) qualify as grid cells**, a fraction consistent with the
literature for dorsocaudal MEC. Their grid spacings span roughly **40–80 cm
(median 60 cm)** and they carry high spatial information (median 1.0 bits/spike),
while the observed gridness of the clearest cells (up to ~1.4) lies far above
their own shuffle nulls (`fig4_exemplar_with_control.png`). This reproduces the
central result of Sargolini et al. (2006): the MEC contains a population of cells
that encode the animal's location through a periodic triangular lattice.

## Files

| File | Description |
|------|-------------|
| `grid_cells_mec.py` | Consolidated jupytext script (percent format), runs end-to-end |
| `grid_cells_mec.ipynb` | Executed notebook version with embedded outputs |
| `gridcells.py` | Analysis helpers (loading, rate maps, autocorrelogram, gridness, shuffle) |
| `run_analysis.py` | Multi-session pipeline → caches `results.pkl` |
| `make_figures.py` | Renders all figures from `results.pkl` |
| `grid_cell_results.csv` | Per-unit metrics table |
| `fig0_trajectory_occupancy.png` | Raw trajectory and occupancy map (validation) |
| `fig1_example_grid_cells.png` | Six clearest grid cells: trajectory+spikes / rate map / autocorrelogram |
| `fig2_population_summary.png` | Gridness vs. null, spacing distribution, spatial-info vs. gridness |
| `fig3_grid_cell_gallery.png` | Autocorrelograms of all 22 identified grid cells |
| `fig4_exemplar_with_control.png` | One exemplar grid cell with its shuffle null |

## Reproduce

```bash
python run_analysis.py     # streams NWB, computes metrics + shuffle -> results.pkl (~8 min)
python make_figures.py     # renders all figures
# or run the whole story end-to-end:
jupytext --to notebook grid_cells_mec.py && \
  jupyter nbconvert --to notebook --execute --inplace grid_cells_mec.ipynb
```

Dependencies: `pynapple`, `pynwb`, `remfile`, `h5py`, `dandi`, `numpy`, `scipy`,
`matplotlib`, `tqdm`, `jupytext`, `nbconvert`.
