# Hippocampal Place Cells on a Linear Track

This analysis demonstrates hippocampal place cells using a real recording streamed
from the [DANDI Archive](https://dandiarchive.org): dandiset
[000044](https://dandiarchive.org/dandiset/000044) (Buzsáki lab, "Diversity in neural
firing dynamics supports both rigid and learned hippocampal sequences"), session
`sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`. The session contains 137
sorted CA1 units (120 putative pyramidal cells, 17 interneurons, left and right CA1)
recorded while rat Achilles ran back and forth on a 1.6 m linear maze for about 35
minutes. The 8.7 GB NWB file is accessed by HTTP range streaming with `remfile` and a
local disk cache; no full download is needed.

## What was done

The pipeline restricts the data to the MazeEpoch, defines run epochs as bouts with a
valid linearized track position and speed above 5 cm/s (91 bouts, 238 s of running),
and computes occupancy-normalized firing-rate maps for every unit over 80 position
bins (2 cm, Gaussian smoothed with sigma = 4 cm). Spatial tuning is quantified with
Skaggs spatial information, and significance is assessed per unit against 200
circular time-shift shuffles of the spike train within the run epoch. Rate maps are
also computed separately for the two running directions, and model-based tuning
curves are fit with a Poisson GLM over a 12-function B-spline basis on position
(nemos).

## Key findings

83 of the 120 excitatory units (69%) pass the shuffle test (p < 0.05) and rate
criteria and are classified as place cells. Median spatial information is 0.94
bits/spike for excitatory units versus 0.01 bits/spike for inhibitory interneurons,
which is the classic separation between spatially tuned pyramidal cells and
untuned interneurons. Sorted by field location, the population rate maps tile the
full track in both running directions, and each run drives the sorted population in
sequence (Figure 1D). Many fields are strongly directional: the median correlation
between a cell's two direction-specific rate maps is 0.27, and a subpopulation fires
almost exclusively in one running direction. The Poisson GLM tuning curves closely
reproduce the empirical rate maps.

## Files

- `place_cells_achilles.py`: consolidated jupytext script, runs end-to-end
- `place_cells_achilles.ipynb`: the same analysis as a Jupyter notebook
- `fig1_raw_data.png`: trajectory, speed profile, example spikes on the position
  trace, and the sorted population raster showing place-cell sequences
- `fig2_behavior.png`: position occupancy per direction, speed distribution,
  position over the epoch
- `fig3_example_place_cells.png`: eight example place cells (spike locations on the
  maze and direction-split rate maps)
- `fig4_population_ratemaps.png`: normalized population rate maps sorted by peak,
  per running direction
- `fig5_spatial_info.png`: spatial information by cell type, observed versus shuffle
  threshold, and the place-cell classification
- `fig6_directionality.png`: direction selectivity of the place fields
- `fig7_glm_tuning.png`: Poisson GLM (B-spline) tuning curves versus empirical rate
  maps
- `01_load_data.py`, `02_preprocess.py`, `03_analyze_placefields.py`,
  `04_visualize.py`, `05_glm_nemos.py`: the modular development scripts; the
  consolidated notebook supersedes them
- `achilles_maze_cache.npz`, `achilles_placefields.npz`: intermediate arrays cached
  during development (not needed to run the notebook)

## Running it

```
python place_cells_achilles.py
```

requires `pynapple`, `pynwb`, `h5py`, `remfile`, `requests`, `numpy`, `scipy`,
`matplotlib`, `tqdm`, `nemos`, and `jax`. The first run streams position and spike
data through the disk cache at `/tmp/remfile_cache_placefields` (a few minutes);
later runs reuse the cache and finish in seconds.
