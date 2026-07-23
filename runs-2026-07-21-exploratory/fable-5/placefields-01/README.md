# Hippocampal Place Cells in Rat CA1 (DANDI:000044)

## Dataset

This analysis uses [DANDI:000044](https://dandiarchive.org/dandiset/000044), the
Grosmark and Buzsáki (2016) hc-11 data set (*Diversity in neural firing dynamics
supports both rigid and learned hippocampal sequences*, Science 351:1440). It
contains eight bilateral silicon-probe recordings from dorsal CA1 in four rats
(Achilles, Buddy, Cicero, Gatsby). Each session is a pre-behaviour sleep period,
a period of running on a maze for water reward, and a post-behaviour sleep
period. Five sessions use a linear track (1.6 m or 2 m) and three use a circular
maze of about 2.9 m. Every file provides sorted units labelled as excitatory or
inhibitory, a linearised position signal, the raw two-dimensional tracking, and a
128-channel LFP. The files are 5 to 9 GB each and are streamed with `remfile`
using an on-disk chunk cache, so only the spike times, the position and short
snippets of LFP ever cross the network.

## What Was Analyzed

For each session I reconstructed the position stream, segmented it into
individual track traversals, and built occupancy-normalised firing rate maps for
every unit separately for each running direction, using 4 cm bins and a 6 cm
Gaussian smoothing kernel. Every unit was then tested against a null distribution
built by circularly shifting its own spike train within the concatenated running
epochs a thousand times, which preserves the spike count, the bursting structure
and the occupancy map while destroying the alignment between spiking and
position. Units that beat their own null at p < 0.01 in at least one direction,
and that fired enough spikes to make the map meaningful, were called place cells.
I then characterised the resulting fields, decoded position from the population
by Bayesian inference on held-out laps, and fitted Poisson GLMs with `nemos` to
check that the fields track position rather than the running-speed profile along
the track. Analysis and data access are both done with `pynapple`; the rate-map
code is verified to agree with `nap.compute_1d_tuning_curves` to within 3e-9 Hz.

## Key Finding

The defining properties of place cells are present and quantitatively
conventional in every session. In the primary session (rat Achilles, 1.6 m linear
track, 137 units) 82 of the 120 excitatory units, or 68%, have a significant
firing field; across all eight sessions the fraction ranges from 33% to 90%, for
a pooled total of 322 place cells carrying 412 fields. The median field is 32 cm
wide, which is a fifth of the track in the primary session and 15% of the
animal's own track when pooled across the three maze geometries, and its median
in-field peak rate is 7.3 Hz. Fields are reliable rather than incidental: a field
produces at least one spike on 87% of the passes through it (median), and rate
maps built from odd and even laps correlate at 0.94. Field peaks tile the whole
track with the usual
over-representation of the reward ends, and on the linear tracks the two
directions of travel carry essentially independent maps, with peak positions
correlating at only r = 0.18. Bayesian decoding from held-out laps recovers the
animal's position to a median error of 5.2 cm in the primary session, against a
chance level of 47 cm, and to between 3.9 and 12.6 cm across the full set. The
GLMs rule out the most obvious confound: a position-only model gains 0.63
bits/spike of held-out likelihood over a constant-rate model against 0.04
bits/spike for a speed-only model, position beats speed for 90% of place cells
pooled over sessions, and adding speed to a position model buys a median of
0.000 bits/spike.

Two methodological points came out of the analysis and are worth recording.
First, raw spatial information in bits per spike is not comparable across cells
with different spike counts. In this data set the excitatory units that *fail*
the place-cell test have a higher median raw spatial information (1.03
bits/spike) than the units that pass it (0.65 bits/spike), because sparse firing
produces noisy, spiky rate maps that score well on any map-shape statistic,
including sparsity and in-field to out-of-field rate contrast. Only the
comparison against each cell's own shuffle null, or a reliability measure such as
the fraction of laps on which the field is active, separates the two populations.
Second, the fast-spiking interneurons here are weakly but detectably spatially
modulated, so excluding them by cell type rather than by any map statistic is
what keeps the place-cell population clean.

## Files

| File | Contents |
| --- | --- |
| `place_cell_analysis.py` | Self-contained jupytext notebook (percent format) that runs the whole analysis end to end |
| `place_cell_analysis.ipynb` | The same notebook in `.ipynb` form |
| `fig00_behaviour_check.png` | Validation of the reconstructed position, lap segmentation and speed threshold |
| `fig01_session_overview.png` | Session structure, trajectory, spike raster during one traversal, simultaneous LFP |
| `fig02_example_place_cells.png` | Eight example cells: spike position lap by lap, and directional rate maps |
| `fig03_population_maps.png` | Population rate maps sorted by field position, and directional remapping |
| `fig04_spatial_information.png` | Observed spatial information against the circular-shift null |
| `fig05_field_properties.png` | Peak rate, width, lap-by-lap reliability, stability, track coverage |
| `fig06_decoding.png` | Bayesian decoding of held-out laps, confusion matrix, error against ensemble size |
| `fig07_glm_position_vs_speed.png` | Poisson GLM comparison of position, speed, and both |
| `fig08_across_sessions.png` | All eight sessions: place-cell fraction, spatial information, decoding, pooled fields |
| `session_summary.csv` | One row per session with the headline numbers |
| `place_fields_all_sessions.csv` | One row per (cell, direction) place field, pooled over sessions |
| `place_cell_stats_primary.csv`, `place_fields_primary.csv` | Per-unit and per-field tables for the primary session |

The development modules (`place_cell_lib.py`, `figures.py`, `run_all.py`) and the
assembly script (`build_notebook.py`) are also included. The notebook is
generated from those modules by `build_notebook.py`, so there is only one copy of
the analysis code and the notebook cannot drift from what was actually run.

## Reproducing

```bash
pip install pynapple nemos pynwb remfile h5py xarray tqdm matplotlib jupytext
python place_cell_analysis.py        # or open the .ipynb
```

The first run streams roughly 500 MB of spike times and behaviour into
`/tmp/remfile_cache` and takes about ten minutes. Subsequent runs read from the
cache and take about three.
