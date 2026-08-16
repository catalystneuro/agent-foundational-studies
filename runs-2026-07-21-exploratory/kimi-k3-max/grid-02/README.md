# Grid Cells in the Medial Entorhinal Cortex (DANDI 000582)

This analysis demonstrates grid cells in the medial entorhinal cortex using
dandiset [000582](https://dandiarchive.org/dandiset/000582) from the DANDI
Archive, the NWB conversion of Sargolini et al. (2006, Science), "Conjunctive
Representation of Position, Direction, and Velocity in Entorhinal Cortex"
(Moser lab). The dandiset contains 118 sessions from 15 Long Evans rats
foraging in square or circular arenas, with tetrode units annotated by MEC
layer and 50 Hz 2D position tracking. All data are streamed with remfile and
local disk caching; no files are downloaded in full.

The pipeline follows the standard grid cell analysis (Hafting et al., 2005):
firing rate maps at 2.5 cm resolution are Gaussian-smoothed, their spatial
autocorrelograms are computed, and the grid score measures the sixfold
symmetry of the autocorrelogram (opexebo package). Significance is tested per
unit against 100 circular time-shift shuffles of the spike train. In the
example session (`sub-11265_ses-16030604`, 14 layer II units), the best cells
show the defining hexagonal firing lattice directly in their spike positions,
with grid scores up to 1.16 (shuffle p = 0.002).

Across all 118 sessions, 197 of 617 units with at least 200 spikes (32%) are
significant grid cells (p < 0.05). Prevalence follows the known anatomical
gradient: 48% of layer II and 44% of layer III units are grid cells, compared
with 19% in layer V and 22% in layer VI. Grid spacing across significant cells
is broadly distributed with a median of 58 cm (IQR 46 to 71 cm), consistent
with the range of dorsoventral recording positions in the original study.

## Files

- `grid_cells_mec_analysis.py`: consolidated jupytext script; runs end-to-end
  (`python grid_cells_mec_analysis.py`, about 7 minutes from scratch on 8
  cores; cached intermediate results make re-runs take about 40 s)
- `grid_cells_mec_analysis.ipynb`: the same analysis as a Jupyter notebook
  (via `jupytext --to notebook`)
- `figures/fig01_raw_data_overview.png`: trajectory, running speed, and
  spike raster for the example session
- `figures/fig02_demo_session_ratemaps.png`: rate maps and autocorrelograms
  for all 14 units of the example session
- `figures/fig03_example_grid_cells.png`: the four best grid cells in detail
- `figures/fig04_shuffle_test.png`: circular time-shift shuffle distribution
  vs. the observed grid score
- `figures/fig05_population_summary.png`: grid score distribution, prevalence
  by layer, spacing distribution, and score vs. significance for 617 units
- `figures/fig06_population_examples.png`: the best grid cell from six
  different rats
- `population_analysis.py`: standalone batch script used for the full
  118-session run (same code as the population stage of the notebook)
- `assets.json`, `population_results.pkl`: cached asset listing and per-unit
  results

## Reproducing

Requires `pynapple`, `pynwb`, `h5py`, `remfile`, `opexebo`, `matplotlib`,
`requests`, `tqdm`. Run `python grid_cells_mec_analysis.py`. Delete
`population_results.pkl` to force recomputation of the population stage;
delete `assets.json` to re-query the DANDI API.
