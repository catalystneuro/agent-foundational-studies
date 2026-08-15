# Theta Phase Precession in Hippocampal CA1 Place Cells

## Dataset

[DANDI Archive Dandiset 000044](https://dandiarchive.org/dandiset/000044)
("Diversity in neural firing dynamics supports both rigid and learned hippocampal
sequences", Grosmark & Buzsaki, Buzsaki Lab, NYU), session
`sub-Achilles/sub-Achilles_ses-Achilles-10252013`. A rat runs back and forth on a
~1.6 m linear track for reward while a 128-channel silicon probe records local field
potentials and 137 spike-sorted CA1 units (120 putative excitatory, 17 putative
inhibitory), together with 2D video-tracked position. The NWB file was streamed
directly from S3 with `remfile` (disk-cached, no full download); only the
`MazeEpoch` (the ~34 min running session) was analyzed.

## Analysis

Because the file's pre-linearized position channel has long NaN gaps, a continuous
position/speed estimate was rebuilt from the raw 2D tracking, restricted to the
runway itself, and running epochs were split by direction of travel (place fields on
linear tracks are direction-selective). Direction-specific firing-rate maps were used
to detect 65-92 CA1 units with a single, well-isolated place field. A hippocampal LFP
channel with the strongest theta (6-10 Hz) vs. delta (2-4 Hz) power ratio was
selected, band-pass filtered, and its instantaneous theta phase extracted via the
Hilbert transform. For each place cell, the theta phase at every in-field spike
(restricted to the cell's preferred running direction) was regressed against
normalized position in the field using a circular-linear fit, with significance
assessed against a position-shuffled null distribution.

## Key Finding

Of 65 place cells with enough in-field spikes to test, 54 (83%) showed a
statistically significant (p < 0.05 vs. shuffle) **negative** phase-precession slope
(median -0.66 theta cycles per field traversal), while only 2 showed a significant
positive slope. This is the classic hippocampal theta phase precession signature
(O'Keefe & Recce, 1993): as the animal advances through a place cell's field, the
cell's spikes systematically shift to earlier phases of the local theta cycle. The
effect is visible both in individual example cells (`fig06_example_phase_precession.png`)
and at the population level (`fig07_population_summary.png`).

## Files

- `theta_phase_precession.py` - consolidated jupytext analysis script (runs end-to-end)
- `theta_phase_precession.ipynb` - executed Jupyter notebook version
- `fig01_position_speed.png` - runway position and running speed over the session
- `fig02_2d_trajectory.png` - raw 2D tracking trajectory
- `fig03_spike_raster.png` - example raw spike raster (30 CA1 units)
- `fig04_place_fields.png` - top 16 direction-specific place fields
- `fig05_theta_phase_extraction.png` - raw LFP, filtered theta, and extracted phase
- `fig06_example_phase_precession.png` - phase-vs-position precession for 6 example cells
- `fig07_population_summary.png` - population summary of precession slopes and significance
