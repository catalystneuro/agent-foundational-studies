# Grid cells in the medial entorhinal cortex (DANDI:000582)

This analysis demonstrates grid cells in real recordings from the DANDI Archive.
The dataset is [DANDI:000582](https://dandiarchive.org/dandiset/000582),
*Conjunctive Representation of Position, Direction, and Velocity in Entorhinal
Cortex* (Sargolini et al., Science 2006; Moser group, NTNU): tetrode recordings
from the dorsocaudal medial entorhinal cortex of 15 freely foraging Long-Evans
rats, with head-mounted LEDs tracked at 50 Hz. The NWB files were streamed from
the DANDI S3 mirror with `remfile` and an on-disk cache, and all spike and
tracking data were handled as Pynapple `TsGroup`, `TsdFrame` and `IntervalSet`
objects. The dandiset contains 118 sessions and 620 units; the analysis uses the
96 sessions recorded in the 1 x 1 m enclosure (468 units passing a 100-spike
minimum, 15 rats). The remaining 22 sessions have tracking coordinates spanning
about 190 cm and were excluded, because grid spacing in centimetres is only
interpretable when the spatial calibration is known and that second scale could
not be confirmed from the file metadata.

Each unit was given an occupancy-normalised, Gaussian-smoothed rate map on 2.5 cm
bins (validated to agree exactly with Pynapple's `compute_2d_tuning_curves`), a
NaN-aware spatial autocorrelogram, and the standard expanding-annulus gridness
score, `min(r60, r120) - max(r30, r90, r150)`. The score was calibrated against a
shuffling null: for every unit the whole spike train was circularly shifted 200
times by a random offset of at least 20 s within the session, which preserves
spike count and temporal structure while destroying the spike-to-position
relationship.

## Key finding

**133 of 468 MEC units (28.4%) exceed the 95th percentile of the shuffled
gridness distribution (threshold 0.67), roughly six times the 5% expected by
construction, and their rate maps show the hexagonal firing lattice that defines
a grid cell.** The pattern is not an artefact of sampling or of the animal's
trajectory: it survives a split-half test within each session (median map
correlation 0.71 for grid cells versus 0.40 for the rest, p = 3e-18), it vanishes
when the same spike train is shifted in time against the same trajectory, and it
is concentrated in the superficial layers (38% of layer III and 36% of layer II
units, versus 15% of layer V and 22% of layer VI). Grid cells were found in 14 of
the 15 rats and in 64 of the 96 sessions. Grid spacing has a median of 50 cm
(interquartile range 43 to 66 cm), grid orientation clusters at small angles to
the walls of the box, and peak in-field rates run from about 2 to 35 Hz.

Two further properties reproduce in the same data. Of the grid cells in sessions
that tracked both LEDs, 71% are also significantly tuned to head direction, the
conjunctive position-by-direction coding that the original study described (this
particular fraction is not an unbiased estimate for MEC, because the dandiset was
assembled to illustrate conjunctive and head-direction cells). And grid cells
recorded simultaneously in one animal share a grid orientation far more closely
than cells from different sessions (median absolute difference 3.4 degrees versus
11.6 degrees over 144 within-session pairs, Mann-Whitney p = 2e-27) while their
cross-correlograms have no central peak, so
they are offset from each other in spatial phase.

## Files

| file | contents |
| --- | --- |
| `grid_cells_mec.py` | consolidated, self-contained jupytext script (percent format) covering the whole analysis |
| `grid_cells_mec.ipynb` | the same script as a notebook |
| `grid_lib.py` | shared loading and analysis functions used by the modular scripts |
| `scan_sessions.py`, `scan_extent.py` | dandiset survey: units per session, enclosure size |
| `01_prototype_session.py` | single-session prototype used to develop the pipeline |
| `02_run_analysis.py` | population analysis with the shuffling control (multiprocess) |
| `02b_stability.py` | split-half stability control |
| `02c_validate_ratemap.py` | cross-check of the rate maps against Pynapple |
| `03_figures.py` | figure suite |
| `unit_metrics_classified.csv` | one row per unit: gridness, spacing, orientation, spatial information, head-direction tuning, stability, classification |
| `shuffle_gridness.npy` | 468 x 200 shuffled gridness scores |
| `maps.npz` | rate map, autocorrelogram and spike positions per unit |

## Figures

| figure | shows |
| --- | --- |
| `fig00_box_sizes.png` | the two enclosure sizes in the dandiset and the 1 m selection |
| `fig0a_worked_example.png` | one unit end to end: spikes on the path, rate map, autocorrelogram |
| `fig0b_shuffle_example.png` | the same unit after two random circular shifts |
| `fig01_raw_session.png` | raw behaviour, occupancy, speed, spike rasters, and a grid cell's firing rate with in-field epochs shaded |
| `fig02_example_grid_cells.png` | six grid cells from four rats and three layers |
| `fig03_gridness_population.png` | observed versus shuffled gridness, grid-cell fraction, layer breakdown, split-half stability |
| `fig04_grid_geometry.png` | spacing, orientation, spatial information, layer and depth relationships, firing rates |
| `fig05_conjunctive_cells.png` | gridness versus head-direction tuning, cell classification, polar tuning curves |
| `fig06_shuffle_control.png` | full shuffling control for one cell including its own null distribution |
| `fig07_grid_gallery.png` | rate maps and autocorrelograms of the 24 highest-gridness cells |
| `fig08_grid_modules.png` | orientation and spacing of simultaneously recorded grid cells, with cross-correlograms |

## Caveats

The gridness score is a single scalar and, as the unfiltered gallery in
`fig07_grid_gallery.png` shows, a small number of high-scoring units have
band-like rather than hexagonal autocorrelograms. Grid spacing is estimated from
the six autocorrelogram peaks nearest the centre, so it is least reliable for
cells whose spacing approaches the size of the box, where only a few fields fit
in the enclosure. The head-direction criterion uses the same shuffling procedure
as the gridness criterion and is permissive; the head-direction fractions should
be read as "significantly directionally modulated", not as a census of classical
head-direction cells.

## Reproducing

```
pip install pynapple pynwb remfile h5py numpy scipy pandas matplotlib tqdm jupytext
python grid_cells_mec.py          # or open grid_cells_mec.ipynb
```

The script reuses `unit_metrics.csv`, `shuffle_gridness.npy`, `maps.npz` and
`unit_stability.csv` if they are present and recomputes them otherwise, which
takes roughly half an hour single-threaded (the 93,600 shuffles dominate). The
multiprocess path in `02_run_analysis.py` does the same work in a few minutes.
