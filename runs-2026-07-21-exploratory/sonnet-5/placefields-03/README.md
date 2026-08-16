# Hippocampal Place Cells in Rat CA1

This analysis demonstrates the classic hippocampal place cell phenomenon using
real electrophysiology data from the DANDI Archive. The dataset is
[DANDI:000044](https://dandiarchive.org/dandiset/000044) ("Diversity in
neural firing dynamics supports both rigid and learned hippocampal
sequences," Grosmark & Buzsaki), which contains silicon-probe recordings from
dorsal CA1 of rats running back and forth on a 1.6 m linear track for water
reward at each end. We analyzed session
`sub-Buddy/sub-Buddy_ses-Buddy-06272013`, streaming the NWB file directly
from S3 with `remfile` and using `pynapple` for all spike and behavioral
analysis. One data-quality issue required a manual fix: the `SpatialSeries`
objects in this file store their sampling `rate` attribute inverted (the
value is actually the sampling period in seconds, not frequency in Hz); this
was confirmed by checking that inverting it makes the implied recording
duration match the MazeEpoch duration exactly, and the timestamps were
reconstructed accordingly before any further analysis.

We restricted analysis to the MazeEpoch and to periods of active locomotion
(speed > 5 cm/s, excluding immobility so that firing during sharp-wave
ripples does not contaminate the spatial tuning estimate). For each
putative excitatory CA1 unit with at least 50 spikes while running, we
computed a firing-rate tuning curve as a function of position along the
track and quantified spatial selectivity with Skaggs spatial information
(bits/spike). Significance was assessed against a null distribution built
from 200 circularly time-shifted versions of each spike train, which
preserves each neuron's overall firing statistics and the animal's behavior
while destroying their true temporal relationship. Of 42 candidate units, 38
(90%) showed significantly elevated spatial information relative to this
shuffled null (p < 0.05), the standard criterion for classifying a neuron as
a place cell. Individual place cells show the expected single, spatially
restricted firing field with spikes visibly clustered at one location when
plotted on the animal's trajectory, and as a population, sorting cells by
the location of their firing peak reveals that place fields tile the entire
length of the track. Firing rate maps computed over the full 2D maze
(including the reward zones at each end, which is why the maze shape looks
like a dumbbell) confirm that this spatial selectivity is not an artifact of
the 1D projection.

## Files

- `place_cells_analysis.py` — consolidated jupytext (percent format) analysis script, runs end-to-end from data loading through figure generation with no manual intervention.
- `place_cells_analysis.ipynb` — the same analysis as an executed Jupyter notebook.
- `fig1_raw_data_overview.png` — raw LFP snippet, spike raster, and behavioral trajectory.
- `fig2_speed_and_running_epochs.png` — running speed distribution and detected running epochs.
- `fig3_example_place_cells.png` — tuning curves and spike locations for 6 example place cells.
- `fig4_place_field_sequence.png` — population place field sequence, sorted by field location.
- `fig5_2d_place_field_maps.png` — 2D firing rate maps on the full maze for top place cells.
- `fig6_spatial_information_summary.png` — observed vs. shuffled spatial information and per-cell significance.
