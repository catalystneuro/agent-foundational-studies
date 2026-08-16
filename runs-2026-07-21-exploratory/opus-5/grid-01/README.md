# Grid cells in the medial entorhinal cortex (DANDI:000582)

## Dataset

[DANDI:000582](https://dandiarchive.org/dandiset/000582), *Conjunctive
Representation of Position, Direction, and Velocity in Entorhinal Cortex*
(Sargolini et al., *Science* 2006, [doi:10.1126/science.1125572](https://doi.org/10.1126/science.1125572)).
118 NWB sessions from 15 Long-Evans rats, tetrode recordings in the dorsocaudal
medial entorhinal cortex while the animal foraged freely for scattered food.
Each file holds sorted spike times, position tracked at 50 Hz from one or two
head-mounted LEDs, and one LFP channel. All 118 sessions (1.9 GB) were streamed
from the DANDI S3 bucket with `remfile` and a local disk cache; nothing was
downloaded in full, and no data anywhere in this pipeline is simulated.

## What was analysed

Every one of the 620 sorted units was processed identically. Periods when the
rat was moving slower than 2.5 cm/s were discarded, a firing-rate map was built
on 3 cm bins and smoothed with a 4.5 cm Gaussian (spike positions obtained with
pynapple's `value_from`), and the Pearson spatial autocorrelogram of that map
was scored for six-fold rotational symmetry: gridness = min(r60, r120) −
max(r30, r90, r150), maximised over an expanding annulus. Significance came
from 100 circular time shifts of each cell's own spike train, run through the
identical pipeline; the 95th percentile of the pooled 62,000-sample shuffle
distribution (gridness 0.62) is the criterion for calling a cell a grid cell.
Grid spacing and orientation were read off the six autocorrelogram peaks
nearest the centre, and head-direction tuning was tested with the same shuffle
procedure applied to the mean vector length.

## Key finding

193 of 620 MEC units (31%) fire in a hexagonally periodic lattice that tiles
the whole enclosure, with gridness above the shuffle threshold. The observed
gridness distribution is visibly bimodal rather than a shifted copy of the null
distribution (`fig04`), grid cells appear in 14 of the 15 rats and in 85 of the
118 sessions, and their rate maps are stable within a session (median
split-half map correlation 0.70, versus 0.34 for the rest of the population,
p = 8e-31). Computing gridness separately in each half of the session gives
correlated values (r = 0.43), and 61% of the grid cells clear the threshold in
both halves independently. The proportion follows the known anatomy: 48% of units in layer II
are grid cells, 40% in layer III, and about 21% in layers V and VI
(chi-square p = 8e-07). In the ~1 m box the median grid spacing is 50 cm
(IQR 42-66 cm) and spacing grows with electrode depth below dura (r = 0.26,
p = 2e-03), the dorsoventral gradient. Grid cells recorded simultaneously share
their lattice orientation closely (median |Δorientation| 4.2° versus 15.0° for
randomly paired cells, p = 3e-45) and have similar spacing (ratio 1.16 versus
1.35), which is the module organisation of the grid map. Of the 134 grid cells
recorded with two tracking LEDs, 69 are also directionally tuned above the
head-direction shuffle threshold, the conjunctive position-by-direction coding
this dataset was collected to demonstrate.

## Files

| File | Contents |
| --- | --- |
| `grid_cells_mec_dandi000582.py` | consolidated jupytext script, runs end to end |
| `grid_cells_mec_dandi000582.ipynb` | the same analysis as an executed notebook |
| `gridlib.py` | loading, rate maps, autocorrelograms, gridness, shuffles |
| `scan_sessions.py`, `session_scan.json` | survey of all assets (units, duration, arena extent) |
| `01_load_and_validate.py` | single-session load and raw-data validation figure |
| `02_analyze_all_sessions.py` | all 118 sessions, all units, gridness + shuffles |
| `02b_hd_shuffles.py` | head-direction mean vector length + shuffle control |
| `03_figures.py` | all population figures and summary statistics |
| `unit_metrics.csv` | per-unit metrics for all 620 units |
| `summary.json` | the headline numbers quoted above |

| Figure | Contents |
| --- | --- |
| `fig01_data_validation.png` | position, occupancy, speed, LFP, spike raster |
| `fig02_example_grid_cells.png` | best grid cell from each of six rats |
| `fig03_gridness_method.png` | how gridness is computed and shuffle-tested |
| `fig04_population_statistics.png` | observed vs. shuffled gridness, per rat, per layer |
| `fig05_grid_geometry.png` | spacing, shared orientation, dorsoventral gradient |
| `fig06_conjunctive_cells.png` | grid × head-direction conjunctive coding |
| `fig07_grid_cell_gallery.png` | 24 highest-gridness cells |
| `fig08_single_cell_walkthrough.png` | single-cell walkthrough produced by the notebook |

Running order: `01_load_and_validate.py`, `02_analyze_all_sessions.py`
(about 6 minutes on 8 cores, writes `results.pkl`), `02b_hd_shuffles.py`,
`03_figures.py`. The consolidated script reuses `results.pkl` if it is present
and recomputes it otherwise.

## Caveats

The session description text in every NWB file says the rat ran in a 1 x 1 m
enclosure, but the tracked extent shows that 22 of the 118 sessions used a much
larger, roughly 2 m circular arena. Spacing is therefore reported separately
for the two groups, since a 1 m box truncates the largest spacings that can be
measured. The position `SpatialSeries` is labelled in meters while the values
are in centimetres, and the LFP is stored in raw acquisition units; the loader
treats them accordingly. The NWB file does not record which of the two LEDs
sits in front of the head, so a preferred head direction may be flipped by
180°, though the strength of the tuning is unaffected. Finally, the grid-cell
criterion is the pooled shuffle threshold used in the original literature; it
is a population-level 5% false-positive rate, and 63% of the cells it selects
also clear p < 0.01 against their own within-cell shuffle.
