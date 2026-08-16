# Grid Cells in the Medial Entorhinal Cortex (DANDI 000582)

## Dataset

This analysis uses DANDI Archive dataset
[000582](https://dandiarchive.org/dandiset/000582), *"Conjunctive Representation
of Position, Direction, and Velocity in Entorhinal Cortex"*, the data behind
Sargolini et al. (2006, *Science* 312:758-762) from the Moser lab at NTNU.
Fifteen Long Evans rats foraged in square or circular open-field arenas (1.0 or
1.5 m) while tetrode single units were recorded in the dorsocaudal medial
entorhinal cortex (MEC). Each of the 118 NWB files contains spike times for the
isolated units (620 units total, most with a histological layer assignment) and
50 Hz 2D position tracking of a head-stage LED. Files were streamed from the
archive with `remfile` plus a local disk cache; nothing was downloaded in full.

## Analysis

The consolidated script `grid_cells_mec_dandi.py` (jupytext format, also
provided as the executed notebook `grid_cells_mec_dandi.ipynb`) runs the full
pipeline end to end. Position samples with running speed below 2.5 cm/s were
excluded. For every unit we computed an occupancy-normalized firing-rate map
(2.5 cm bins, 5 cm Gaussian smoothing), its spatial autocorrelogram, and the
standard Moser-lab gridness score using the `opexebo` package: the minimum
correlation of the autocorrelogram with itself rotated by 60 and 120 degrees
minus the maximum correlation at 30, 90, and 150 degrees. Every unit with a
grid score above 0.3 and at least 200 spikes was validated with 100 circular
time-shift shuffles of its spike train, giving an empirical p-value per cell.

## Key Findings

Grid cells are robustly present in this dataset. Of 616 units with sufficient
spiking, 189 (31%) were significant grid cells at p < 0.05 against the shuffle
null, with a median grid score of 0.93 (maximum 1.44). As in the original
report, grid cells concentrate in the superficial layers: 45% of layer II and
43% of layer III units were significant, versus 17% and 20% in layers V and VI.
The median grid spacing was 60 cm (IQR 47-70 cm), consistent with the
dorsoventral location of the recordings, and the strongest cells show textbook
hexagonal autocorrelograms with six evenly spaced satellite fields.

## Outputs

- `grid_cells_mec_dandi.py`: consolidated jupytext script (runs end to end;
  per-session results are cached in `results/` so re-runs take about a minute)
- `grid_cells_mec_dandi.ipynb`: the same analysis as an executed Jupyter notebook
- `fig1_example_session_overview.png`: trajectory and speed distribution
- `fig2_example_session_units.png`: all 14 layer II units of one session:
  spike overlay, rate map, autocorrelogram, grid score
- `fig3_grid_cell_gallery.png`: the twelve strongest significant grid cells
- `fig4_population_grid_scores.png`: grid score distributions, layer
  breakdown, and shuffle validation
- `fig5_grid_spacing_orientation.png`: grid spacing and orientation of the
  significant cells
- `fig6_spatial_info_vs_grid.png`: Skaggs spatial information versus grid score

Supporting files: `grid_lib.py` and `run_population.py` (the parallel runner
used for the first full pass), `session_survey.json` (per-session contents),
and `results/*.pkl` (per-session per-unit results, including rate maps and
shuffle distributions).

## Requirements

`pynwb`, `h5py`, `remfile`, `opexebo`, `numpy`, `matplotlib`, `tqdm`,
`requests`, `jupytext` (for the notebook conversion).
