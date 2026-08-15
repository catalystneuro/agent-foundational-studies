# Grid cells in the medial entorhinal cortex (DANDI:000582)

This analysis demonstrates grid cells in the rat medial entorhinal cortex (MEC) using
[DANDI:000582](https://dandiarchive.org/dandiset/000582), "Conjunctive Representation of
Position, Direction, and Velocity in Entorhinal Cortex" (Sargolini et al., *Science* 2006,
Moser lab). The dandiset holds 118 NWB files from 15 Long-Evans rats foraging in open
arenas, each with tetrode-sorted MEC units, two head-mounted tracking LEDs sampled at
50 Hz, and an annotation of the MEC layer the tetrode was in. All files were read by
streaming from S3 with `remfile` and a local disk cache; nothing was downloaded in full.
Spikes, position, head direction and running speed were handled as Pynapple objects
(`TsGroup`, `TsdFrame`, `Tsd`, `IntervalSet`), and rate maps were computed with
`nap.compute_tuning_curves`.

After surveying all 118 sessions, the analysis used the 1 x 1 m square-arena sessions
(the remaining twenty sessions used a much larger circular arena) and kept one session per
rat per recording day, giving 430 units from 84 sessions and 15 rats. For every unit,
occupancy-normalised rate maps (3 cm bins, 6 cm Gaussian smoothing, restricted to epochs
with running speed between 2.5 and 100 cm/s) were turned into Pearson spatial
autocorrelograms, from which we computed the gridness score
`min(r60, r120) - max(r30, r90, r150)`, grid spacing and orientation, Skaggs spatial
information, split-half map stability and head-direction tuning. Significance was
calibrated by circularly shifting each unit's spike times 100 times (100 shifts x 430
units = 43,000 surrogate maps) and taking the 95th percentile of the pooled shuffled
distribution as the threshold.

## Key finding

121 of 430 units (28%) exceeded the shuffled gridness threshold of 0.66, and their
gridness values form a distinct second mode near 1.0 that the shuffle distribution never
reaches (`fig04`). These cells fire in hexagonally periodic fields with a median spacing
of 55 cm (IQR 43-72 cm), autocorrelograms that are close to regular (median peak-distance
ratio 1.13), stable maps within a session (median split-half correlation 0.73 against 0.42
for the rest of the population) and higher spatial information (0.55 against 0.30
bits/spike). Grid cells recorded simultaneously share their grid orientation much more
closely than cells recorded in different sessions (median difference 3.8 vs 12.4 degrees),
which is what one expects if co-recorded cells belong to a common grid network.

The laminar pattern reproduces the main result of the original study. Grid cells occur in
every layer sampled (31% in layer II, 35% in layer III, 17% in layer V, 24% in layer VI),
but sharp head-direction tuning is essentially absent in layer II (1 of 17 units with
head-direction data) and common below it (63% in layer III, 88% in layer V, 74% in layer
VI). Conjunctive grid x head-direction cells, which have both a hexagonal map and a sharp
directional preference, were found only in the deeper layers (38 in layer III, 7 in layer
V, 3 in layer VI, none in layer II).

Caveats: the layer II sample is the smallest (45 units, and only 17 of them come from
sessions with the second tracking LED that head direction requires), so the layer II
percentages are the least well constrained; layers were taken from the file-level
histology annotation rather than per tetrode; and although sessions were deduplicated to
one per rat per day, a cell held across consecutive days could still enter the sample
twice.

## Files

| file | contents |
| --- | --- |
| `grid_cells_mec_dandi000582.py` | consolidated jupytext script, runs end-to-end (self-contained; about 10 minutes from scratch, then cached) |
| `grid_cells_mec_dandi000582.ipynb` | the same analysis as a Jupyter notebook |
| `gridlib.py` | analysis library used by the modular development scripts |
| `01_explore_session.py`, `02_survey_sessions.py`, `03_example_cells.py`, `04_population_analysis.py`, `05_figures.py` | modular pipeline: load and inspect, survey all sessions, prototype on one session, population analysis with shuffles, final figures |
| `fig01_raw_data_streams.png` | trajectory, tracked position, speed, spike raster, occupancy, sampling distributions |
| `fig02_example_grid_cells.png` | eight example grid cells (one per rat): trajectory with spikes, rate map, autocorrelogram |
| `fig03_gridness_method.png` | how gridness is measured, grid cell vs a band-like non-grid cell |
| `fig04_shuffle_classification.png` | observed vs shuffled gridness, stability and spatial information |
| `fig05_grid_geometry.png` | spacing, orientation, regularity, and orientation alignment of co-recorded cells |
| `fig06_layers_and_conjunctive.png` | grid, head-direction and conjunctive cells by MEC layer, with examples |
| `fig_check_raw_streams.png`, `fig_check_ratemaps.png`, `fig_check_session_all_units.png` | validation figures produced while building the pipeline |
| `unit_metrics.csv` | per-unit metrics for all 430 units |
| `shuffle_gridness.csv`, `shuffle_mvl.csv` | the 43,000 surrogate gridness and head-direction values |
| `session_survey.csv` | arena geometry, duration and unit count for all 118 sessions |
| `summary.json` | the headline numbers quoted above |

## Reproducing

```bash
pip install pynapple pynwb remfile h5py numpy scipy pandas matplotlib tqdm jupytext
python grid_cells_mec_dandi000582.py
```

The script downloads nothing up front: it lists the dandiset through the DANDI API and
streams byte ranges on demand into `/tmp/remfile_cache_000582`. Intermediate results are
written to CSV next to the script and reused on later runs.
