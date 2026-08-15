# Grid Cells in the Medial Entorhinal Cortex

## Dataset

[DANDI:000582](https://dandiarchive.org/dandiset/000582), "Conjunctive
Representation of Position, Direction, and Velocity in Entorhinal Cortex,"
contains the original extracellular recordings from Sargolini et al.,
*Science* (2006). Rats foraged freely in a 100 x 100 cm open-field arena
while single units were recorded from each principal cell layer of the medial
entorhinal cortex (MEC), with position tracked via an LED on the headstage at
50 Hz. All NWB files were streamed directly from the DANDI S3 bucket with
`remfile` (disk-cached) rather than downloaded in full, and data access and
tuning-curve computation used Pynapple.

## Analysis

For each recorded unit we computed a smoothed 2D firing-rate map from the
tracked position and spike times, then its spatial autocorrelogram, and
scored the hexagonal periodicity of that autocorrelogram with the standard
rotational gridness score (correlating the annulus around the six nearest
peaks with 30/60/90/120/150-degree rotations of itself; Sargolini et al.
2006, Langston et al. 2010). To test whether each unit's gridness exceeded
chance, we built a per-unit null distribution from 100 circular time-shifts
of its spike train and classified a unit as a grid cell if its true gridness
score exceeded the 95th percentile of that null distribution. The analysis
was prototyped on a single session and then scaled to six sessions from six
different subjects (out of 118 sessions in the full dandiset).

## Key Finding

A substantial fraction of the recorded MEC units, most of them from layer II,
showed spatial autocorrelograms with clear six-fold hexagonal symmetry and
gridness scores that significantly exceeded their own shuffled null
distribution, reproducing the classic Sargolini et al. (2006) finding of grid
cells in MEC layer II. The clearest examples show textbook hexagonal firing
lattices covering the entire open field, visible directly in the raw spike
positions overlaid on the animal's trajectory as well as in the six-peaked
autocorrelogram.

## Files

- `grid_cells_mec.py` - jupytext (percent-format) script, runs end-to-end
- `grid_cells_mec.ipynb` - the same analysis as a Jupyter notebook
- `figures/` - all generated figures (PNG)
- `population_gridness_results.csv` - per-unit gridness scores, shuffle
  thresholds, and grid-cell classification across all analyzed sessions
