# Theta Phase Precession in Hippocampal Place Cells (DANDI 000044)

This analysis demonstrates theta phase precession, the phenomenon in which a hippocampal
place cell fires at progressively earlier phases of the 6-12 Hz theta oscillation as a
rat runs through the cell's place field (O'Keefe & Recce, 1993).

## Dataset

DANDI Archive dandiset [000044](https://dandiarchive.org/dandiset/000044) (Buzsaki lab),
session `sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`: rat CA1
tetrode recording with 137 sorted units (120 excitatory, 17 inhibitory), a 128-channel
LFP at 1250 Hz, and 2D + linearized position on a 1.6 m linear maze. The maze epoch is
34.5 min with 86 run bouts (44 and 42 per direction, ~42 laps). The file is streamed
with remfile plus a local disk cache; no full download is needed.

## Pipeline

`theta_phase_precession.py` (jupytext; `theta_phase_precession.ipynb` is the converted
notebook) runs the whole analysis end to end:

1. **Behavior**: run bouts are taken as the contiguous valid stretches of the
   linearized position signal (NaN except during on-track runs), with a direction per
   bout from the sign of the position derivative.
2. **Theta phase**: the LFP reference channel (ch 117, chosen data-driven by
   theta/delta PSD ratio; clear 9.2 Hz peak during running) is bandpass filtered at
   6-12 Hz and the phase is the angle of the Hilbert analytic signal (0 = LFP peak).
3. **Place fields**: direction-specific smoothed rate maps (50 bins, 1.5-bin Gaussian);
   a cell-direction is a place field when its peak rate is >= 1 Hz, its Skaggs spatial
   information exceeds a 500-fold circular time-shift shuffle (p < 0.05), it has >= 30
   spikes, and its field (rate > 20% of peak) is 0.2-1.2 m wide. This yields 120 place
   cell-directions from 79 unique cells.
4. **Precession**: for each place cell-direction, spikes during runs through the field
   get a theta phase and a field-progress coordinate (0 at entry, 1 at exit, oriented by
   travel direction). The phase-position relationship is quantified with the
   circular-linear correlation of Kempter et al. (2012), signed by a circular-linear
   regression slope, with p-values from 1000 phase-permutation shuffles, and the same
   correlation is computed per single traversal.

## Key finding

Phase precession is clear and population-wide. 83 of 120 place cell-directions show a
significant phase-position correlation (p < 0.05), of which 79 are negative
(precessing) against only 4 positive; the median signed correlation across all 120 is
-0.32 and the median slope is -0.52 theta cycles per field. The effect is visible within single laps (81% of 113 cell-directions have
negative per-traversal means across 2075 traversals; median -0.25) and in the pooled
population density, which shows the classic diagonal band sweeping from ~600 deg at
field entry to ~360 deg at field exit.

## Files

- `theta_phase_precession.py` / `.ipynb`: consolidated end-to-end analysis
- `01_load_data.py`, `02_preprocess.py`, `03_place_fields.py`, `04_precession.py`,
  `05_figures.py`: modular development pipeline (same analysis, step by step)
- `fig01_data_overview.png`: position, run bouts, raw/filtered LFP, theta PSD
- `fig02_place_fields.png`: direction-specific rate maps sorted by peak + SI selection
- `fig03_example_precession.png`: phase vs field progress for six example cells
- `fig04_single_traversals.png`: per-lap precession for one showcase cell
- `fig05_population_stats.png`: correlation/slope distributions, shuffle p-values,
  per-traversal statistics
- `fig06_pooled_population.png`: pooled phase-position density (precessing cells and
  all place cells)
- `fig07_raw_traversal.png`: spikes against the theta-filtered LFP during one lap
- `precession_results.json`: per-cell-direction statistics
