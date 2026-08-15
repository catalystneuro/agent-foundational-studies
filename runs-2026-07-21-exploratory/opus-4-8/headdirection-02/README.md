# Head Direction Cells in the Mouse Anterior Thalamus and Post-Subiculum

## Dataset

[DANDI dandiset 000056](https://dandiarchive.org/dandiset/000056), *"Internally
organized mechanisms of the head direction sense"* (Peyrache, Lacroix, Petersen
& Buzsáki). Extracellular recordings from the anterodorsal thalamic nucleus and
the post-subiculum of freely-moving mice foraging in an open field, with the
animal's head position tracked by two head-mounted LEDs. Files are streamed
directly from S3 with `remfile` plus a local disk cache; nothing is downloaded
in full.

## What was analyzed

Head direction cells fire selectively when the animal's head points in a
particular allocentric direction. The dataset does not store head azimuth
directly, so it was reconstructed as the angle of the vector connecting the two
head-mounted LEDs, after masking the `(-1, -1)` samples the tracker writes when
an LED is lost. Analysis was restricted to the awake foraging epochs (the
behavioral `states` interval set).

For each unit, occupancy-corrected tuning curves (firing rate vs. head
direction) were built with Pynapple's `compute_1d_tuning_curves`. Directional
selectivity was quantified by assigning every spike the animal's head direction
at that instant and computing the **mean resultant vector length** `R` and the
**preferred direction**. Significance was assessed with an occupancy-resampling null
(each spike is reassigned a head direction drawn from the animal's empirical
occupancy distribution, preserving the cell's spike count and the sampling of
directions the animal actually visited while destroying spike↔direction
alignment; the observed `R` is compared to this null over 500 repeats). A unit was
called an HD cell when `R > 0.3` and shuffle `p < 0.01`. The pipeline was
prototyped on one session and then run across five sessions from four animals,
pooling the results.

## Key finding

Every session yields a distinct population of sharply, reproducibly
direction-tuned cells. In the prototype session (Mouse12-120807) 18 of 77 units
qualified as HD cells, the strongest with a mean resultant vector length of
~0.87 (a near-perfectly directional cell). Across all five sessions from four
animals, 73 of 299 units (24%) were HD cells, and every session contained a
clear HD-cell population (10-42% of units). Plotting each spike against head
direction instead of wall-clock time collapses an HD cell's firing into a tight
band around its preferred direction, and the preferred directions of the pooled
HD-cell population tile the full circle, as expected of a system that represents
all head directions uniformly. This is the defining signature of head direction
cells, demonstrated directly from real DANDI Archive recordings.

## Files

- `head_direction_analysis.py` — consolidated jupytext analysis (runs end-to-end).
- `head_direction_analysis.ipynb` — notebook conversion of the same script.
- `session_summary.csv` — per-session unit / HD-cell counts.
- `fig1_raw_streams.png` — reconstructed head direction and spike rasters.
- `fig2_polar_tuning.png` — polar tuning curves of the strongest HD cells.
- `fig3_directional_raster.png` — spikes vs. head direction for the best HD cell.
- `fig4_population_summary.png` — `R` distribution and preferred-direction spread.
- `fig5_cross_session.png` — HD-cell counts and pooled `R` across sessions.
- `fig6_pooled_preferred_directions.png` — pooled HD-cell preferred directions.
