# Theta Phase Precession in Hippocampal Place Cells

This analysis demonstrates theta phase precession, the tendency of a
hippocampal place cell to fire at progressively earlier phases of the theta
oscillation as the animal crosses the cell's place field (O'Keefe and Recce
1993; Skaggs et al. 1996), using real data streamed from the DANDI Archive.

## Dataset

Dandiset [000044](https://dandiarchive.org/dandiset/000044), "Diversity in
neural firing dynamics supports observationally constrained network models of
hippocampal sharp-wave ripples" (Buzsaki lab). Session
`sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`: rat CA1
tetrode recordings (137 units, 120 classified excitatory) with a 128-channel
LFP at 1250 Hz, while the animal shuttles on a 1.6 m linear maze. The maze
epoch (18079.5-20147 s, ~34.5 min) contains 86 run bouts (44 and 42 per
direction, median speed 0.68 m/s) identified from the linearized position
series. The file is streamed with remfile plus a local disk cache; no full
download is needed.

## Analysis

Theta phase was extracted from the LFP channel with the strongest theta rhythm
(channel 117; theta/delta power ratio 3.1, spectral peak at 9.25 Hz) by 6-12 Hz
bandpass filtering and the Hilbert transform. Place fields were detected
separately for the two travel directions from direction-specific rate maps,
with spatial information (Skaggs et al. 1993) tested against 500 circular
shifts of the spike times relative to the trajectory. 120 cell-directions from
79 unique cells passed the criteria (peak rate >= 1 Hz, shuffle p < 0.05, >= 30
in-field spikes, field width 0.2-1.2 m). For each cell-direction, spikes inside
the field were assigned a field-progress coordinate (0 at entry, 1 at exit,
oriented by travel direction) and a theta phase, and precession was quantified
with the Kempter et al. (2012) circular-linear correlation. Sign and slope come
from a grid-search circular-linear regression, and significance from 1000
phase-permutation shuffles.

## Key finding

Phase precession is strong and widespread. 80 of 120 place cell-directions show
a significant negative phase-position correlation (phase-permutation p < 0.05)
versus only 5 significantly positive; the median signed circular-linear
correlation is -0.32 and the median slope is -0.52 theta cycles per field. The
effect is visible within single laps, not just in pooled data: across 2075
individual field traversals the median per-traversal correlation is -0.25, and
81% of the 113 cell-directions with enough traversals are negative on average.
The pooled spike-density heatmap over all precessing cells shows the canonical
diagonal band from late phases at field entry to early phases at field exit.

## Files

- `theta_phase_precession.py`: consolidated jupytext script that runs the full
  pipeline end-to-end (streaming load, preprocessing, place fields, precession,
  figures).
- `theta_phase_precession.ipynb`: the same script converted to a Jupyter
  notebook with jupytext.
- `achilles_loader.py`, `01_preprocess.py`, `02_place_fields.py`,
  `03_precession.py`, `04_figures.py`: the modular pipeline the consolidated
  script was assembled from.
- `fig01_data_overview.png`: linearized position, run bouts, theta reference
  channel snippet, and LFP spectrum with a clear ~9 Hz theta peak.
- `fig02_place_fields.png`: direction-specific rate maps sorted by peak
  position and the spatial-information distribution.
- `fig03_example_precession.png`: phase vs. field progress for six example
  cells with circular-linear fits.
- `fig04_single_traversals.png`: precession within six individual laps of one
  showcase cell.
- `fig05_population_stats.png`: population distributions of correlation, slope,
  significance, and per-traversal means.
- `fig06_pooled_population.png`: pooled phase-by-position spike-density
  heatmaps.
- `fig07_raw_traversal.png`: spikes of one cell plotted against the
  theta-filtered LFP during a single field traversal.
- `precession_results.json`: per-cell-direction statistics.

## Reproducing

```
python theta_phase_precession.py
jupytext --to ipynb theta_phase_precession.py
```

The script requires pynapple, pynwb, remfile, h5py, requests, scipy, tqdm, and
matplotlib, and takes about five minutes on a warm disk cache (longer on first
run, since remfile fetches the needed byte ranges from S3).
