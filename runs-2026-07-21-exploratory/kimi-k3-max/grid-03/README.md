# Grid Cells in the Medial Entorhinal Cortex (DANDI 000582)

This analysis demonstrates grid cells in the medial entorhinal cortex (MEC)
using dandiset [000582](https://dandiarchive.org/dandiset/000582) from the
DANDI Archive: the NWB conversion of **Sargolini et al. (2006), Science**,
"Conjunctive Representation of Position, Direction, and Velocity in Entorhinal
Cortex" (Moser lab). The dataset holds 118 tetrode recording sessions from 15
Long Evans rats foraging in open-field boxes (1 m and 1.5 m), with 50 Hz 2D
position tracking and 620 sorted single units annotated by MEC layer. Files
are streamed from the archive with `remfile` (no full downloads) and accessed
through `pynapple`.

## Analysis

We follow the standard Moser-lab identification pipeline, implemented with the
`opexebo` package: for each unit we build an occupancy map and a
Gaussian-smoothed firing rate map (2.5 cm bins, sigma = 2 bins, running epochs
only, speed >= 2.5 cm/s), compute the spatial autocorrelation of the rate map,
and derive the grid score, which compares the autocorrelation annulus at 60/120
deg rotation (high for hexagonal structure) against 30/90/150 deg. Units with
an observed grid score above 0.3 are tested against 100 circular time-shift
shuffles of the spike train (shift >= 20 s, wrapping around the session), which
destroy the spike-position relationship while preserving the temporal
structure of the spike train and the behavioral statistics.

## Key findings

Of 615 units with at least 200 spikes (117 sessions contributed units), **189
are significant grid cells** (p < 0.05, shuffle test). Grid cells are
concentrated in the superficial layers: 41% of layer II and 44% of layer III
units pass the criterion, versus 17% in layer V and 21% in layer VI,
reproducing the laminar pattern of the original report. Significant cells have
a median grid score of 0.93 (max 1.44) and median grid spacing of 58 cm (IQR
47-70 cm), consistent with the dorsoventral spacing gradient of MEC. The
pooled shuffle null is centered near zero and cleanly separated from the
observed candidate distribution.

## Files

- `grid_cells_mec_dandi.py` — consolidated jupytext script, runs end-to-end
- `grid_cells_mec_dandi.ipynb` — executed Jupyter notebook (via jupytext/nbconvert)
- `grid_utils.py` — shared loading and analysis helpers
- `01_load_data.py`, `02_analyze_demo_session.py` — single-session prototypes
- `03_run_all_sessions.py` — batch analysis of all 118 sessions (8 workers)
- `04_visualize.py` — population figures from the results table
- `results/grid_scores.csv` — unit-level results (615 rows)
- `results/maps/*.npz` — per-session rate maps, autocorrelations, shuffle scores
- `fig_01_raw_data_validation.png` — position trace and spike locations
- `fig_02_demo_session_gallery.png` — rate maps and autocorrelations, one session
- `fig_03_example_grid_cells.png` — top significant grid cells across sessions
- `fig_04_grid_score_distributions.png` — observed scores vs shuffle null
- `fig_05_layer_breakdown.png` — grid-cell prevalence and scores by MEC layer
- `fig_06_grid_spacing.png` — grid spacing and orientation distributions

## Reproducing

```
python grid_cells_mec_dandi.py        # or run the .ipynb
```

The full batch (118 sessions, 100 shuffles for candidate units) takes a few
minutes with 8 parallel workers and caches to `results/`; delete that
directory to force a rerun. Dependencies: `pynapple`, `pynwb`, `remfile`,
`h5py`, `opexebo`, `dandi`, `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`.

Note: the NWB position metadata says "meters" but the values are centimeters
(the 1.5 m box spans +/-75); the analysis treats them as cm.
