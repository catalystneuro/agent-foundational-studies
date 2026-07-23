# Hippocampal Place Cells on a Linear Track

This analysis demonstrates the classic hippocampal place cell phenomenon using
[DANDI:000044](https://dandiarchive.org/dandiset/000044), "Diversity in neural
firing dynamics supports both rigid and learned hippocampal sequences"
(Grosmark & Buzsaki, *Science* 2016). The dataset contains bilateral
silicon-probe recordings from dorsal CA1 in Long-Evans rats, obtained while
each animal ran on a novel maze for a water reward. Four sessions from four
different rats (Buddy, Achilles, Cicero, Gatsby), all using the same 1.6 m
linear-track configuration, were streamed directly from the DANDI S3 bucket
with `remfile` (no local download) and analyzed with `pynapple`.

For each session, spikes were restricted to running epochs (speed > 5 cm/s,
excluding the immobile bouts at the reward ports), position was reconstructed
as a continuous linear coordinate via PCA projection of the raw 2D LED
tracking (needed because the dataset's own linearized-position channel was
valid for only ~7% of samples), and 1D tuning curves were computed for every
unit. Units were classified as place cells if their Skaggs spatial
information (bits/spike) exceeded the 95th percentile of a circular-shift
shuffle null distribution (200 shuffles) and their mean running-epoch firing
rate exceeded 0.2 Hz. One data quirk worth flagging for reuse: the position
`SpatialSeries.rate` field in this dataset is mislabeled and actually stores
the sampling *period* (~0.0256 s, i.e. ~39 Hz), not a frequency; using it as a
literal rate stretches the position trace across ~41 days instead of the true
~40 minute session, which is caught and corrected in the notebook.

Across the four sessions, 50-70% of putative excitatory CA1 units showed
statistically significant spatial tuning, and their normalized place fields
collectively tiled the full extent of the track when sorted by peak location,
the classic population-level signature of place coding. Individual example
cells fired in tight, spatially restricted zones matching their tuning-curve
peaks, and the fraction of significant place cells was consistently higher
among putative excitatory (pyramidal) units than putative inhibitory
(interneuron) units, consistent with place coding being primarily a
principal-cell phenomenon in CA1. These results were reproducible across four
independent animals, indicating the phenomenon is a robust feature of
hippocampal activity during spatial navigation rather than an artifact of one
recording session.

## Files

- `place_cell_analysis.py` - consolidated jupytext (percent-format) analysis script, runs end-to-end
- `place_cell_analysis.ipynb` - the same analysis as an executed Jupyter notebook
- `fig1_raw_data_overview.png` - raw position, spike raster, speed, and LFP for one session excerpt
- `fig2_place_field_heatmap.png` - normalized place field map sorted by peak location, plus occupancy
- `fig3_example_place_cells.png` - top 6 place cells: spike locations on the trajectory and tuning curves
- `fig4_spatial_info_stats.png` - spatial information distributions, shuffle-test example, and cell-type comparison
- `fig5_population_summary.png` - place cell fraction and spatial information pooled across all 4 sessions/rats
