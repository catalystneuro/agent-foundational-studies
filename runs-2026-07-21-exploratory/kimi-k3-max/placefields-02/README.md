# Hippocampal Place Cells from the DANDI Archive

**Dataset:** DANDI dandiset [000044](https://dandiarchive.org/dandiset/000044) ("Diversity
in neural firing dynamics supports both rigid and learned hippocampal sequences", Buzsaki
lab), session `sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`
(8.7 GB, streamed with remfile + disk cache; no full download). The session contains 137
sorted CA1 units (120 excitatory, 17 inhibitory) and 39 Hz 2D + linearized position from a
34-minute run on a 1.6 m linear track.

**Analysis.** Run bouts (86 bouts, 245 s) were extracted from the linearized position
(>=1 s duration, >=0.3 m span, median speed >=0.15 m/s). For each unit we computed
occupancy-normalized rate maps (50 bins, 1.5-bin Gaussian smoothing), Skaggs spatial
information, and a shuffle p-value from 500 circular time shifts of the spike train on the
concatenated run-time axis. Place cells were defined by p < 0.05, mean in-run rate
> 0.1 Hz, and peak field rate > 1 Hz. As a model-based check, each place cell was also fit
with a Poisson GLM on 12 raised-cosine position basis functions (nemos, ridge 1e-4, JAX
float64).

**Key finding.** 86 of 120 excitatory units (72%) are significant place cells, and their
normalized rate maps tile the full length of the track in each travel direction
(Figure 3). The fields are strongly direction selective: the median correlation between a
cell's two direction-specific rate maps is only 0.22, and many cells fire exclusively in
one direction (Figure 2). Place cells carry a median 0.53 bits/spike of spatial
information versus 0.01 for inhibitory interneurons (Figure 4). The Poisson GLM reproduces
the binned tuning curves almost exactly (median map correlation 0.99; median pseudo-R2
0.086, typical for single-bin spike prediction), confirming that position alone accounts
for the spatial structure of the maps (Figure 5).

## Files

- `place_cells_dandi.py`: consolidated jupytext script, runs end-to-end in ~1-2 min
  (warm cache) and writes all figures
- `place_cells_dandi.ipynb`: the same analysis as a Jupyter notebook (via jupytext)
- `figures/fig1_session_overview.png`: trajectory, linearized position, speed, spike raster
- `figures/fig2_example_place_cells.png`: six example cells: spikes on trajectory +
  direction-specific tuning curves
- `figures/fig3_population_tiling.png`: normalized population rate maps sorted by peak,
  pooled and per direction
- `figures/fig4_place_cell_stats.png`: SI distributions, shuffle significance, rate
  criteria, directionality
- `figures/fig5_nemos_glm.png`: Poisson GLM fits vs binned tuning curves, pseudo-R2
  summary
- `scripts/`: modular development scripts (loading, figures, GLM) that the consolidated
  script was built from

## Reproducing

```bash
python place_cells_dandi.py        # or run the notebook
```

Requires pynapple 0.11, pynwb 4, remfile, h5py, scipy, matplotlib, nemos 0.2.6, jax,
tqdm. The S3 blob URL in the script is the public asset URL; if it ever stops resolving,
regenerate it from the DANDI API
(`GET /api/assets/{asset_id}/download/`, asset `c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d`).
