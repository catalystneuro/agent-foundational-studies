# Hippocampal Place Cells on a Linear Track (DANDI 000044)

This analysis demonstrates hippocampal place cells using a classic Buzsáki lab
session streamed from the DANDI Archive: `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`
from dandiset [000044](https://dandiarchive.org/dandiset/000044) ("Diversity in
neural firing dynamics supports both rigid and learned hippocampal sequences").
The session contains 137 single units recorded from CA1 tetrodes while a rat
ran back and forth on a 1.6 m linear track for about 34 minutes (~42 laps).
The file (8.7 GB) is accessed by HTTP range streaming with `remfile` and a
local disk cache; nothing is downloaded in full. Data are handled with
Pynapple, and model-based tuning curves are fit with NeMoS.

For each unit, spikes were binned by the animal's linearized track position
(50 bins of 3.2 cm), divided by occupancy, and smoothed to give firing rate
maps. Spatial tuning was quantified with Skaggs spatial information, and
significance was assessed per unit against 500 circular time-shift shuffles
of the spike train. Of 120 excitatory (putative pyramidal) units, 85 (71%)
passed the place cell criteria (shuffle p < 0.05, on-track rate > 0.1 Hz,
peak rate > 1 Hz). Sorted by peak position, their normalized rate maps tile
the full length of the track, the signature population code for position.
Most fields are direction-selective: the median correlation between
direction-specific rate maps is 0.35, and 59% of place cells have r < 0.5.
As an independent check, Poisson GLMs with a 12-function B-spline basis over
position reproduce the empirical rate maps, with peak positions agreeing at
r = 0.96 across the population (median Cohen pseudo-R^2 = 0.11, typical for
position-only models of noisy spike trains).

## Files

- `place_cells_achilles.py`: consolidated jupytext script that runs the full
  analysis end-to-end (streaming load, rate maps, shuffles, figures, GLMs).
- `place_cells_achilles.ipynb`: the same analysis as a Jupyter notebook.
- `06_placefields.py`, `07_figures.py`, `08_glm.py`: the modular pipeline
  scripts used to develop the analysis (kept for reference).
- `fig1_session_overview.png`: trajectory, linearized position, speed, and
  occupancy for the maze epoch.
- `fig2_example_place_cells.png`: six example place cells (spike locations,
  rate maps, direction-split rate maps).
- `fig3_top24_ratemaps.png`: rate maps of the 24 cells with highest spatial
  information.
- `fig4_population.png`: normalized rate maps of all 85 place cells sorted by
  peak position, overall and split by running direction.
- `fig5_spatial_info_stats.png`: spatial information distributions, observed
  vs shuffle, and rate vs spatial information.
- `fig6_directionality.png`: direction-selective example cells and the
  population distribution of directional consistency.
- `fig7_glm_tuning.png`: Poisson GLM tuning curves vs empirical rate maps.
- `fig8_glm_validation.png`: GLM vs empirical peak positions and goodness of
  fit across the population.
- `placefield_results.npz`, `glm_results.npz`: saved analysis results.

## Running

```
python place_cells_achilles.py
```

Requires `pynapple`, `pynwb`, `h5py`, `remfile`, `nemos`, `scipy`,
`matplotlib`, `tqdm`, `requests`. First run streams the needed data ranges
into `/tmp/remfile_cache_achilles`; later runs reuse the cache.
