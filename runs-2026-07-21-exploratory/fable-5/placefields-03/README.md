# Hippocampal Place Cells in Rat CA1 (DANDI:000044)

This analysis demonstrates hippocampal place cells using real extracellular recordings streamed
from the DANDI Archive. Everything runs from a single script, `place_cells_dandi_000044.py`,
which is also provided as an executed notebook, `place_cells_dandi_000044.ipynb`.

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), "Diversity in neural firing dynamics
supports both rigid and learned hippocampal sequences", contributed by Andres D. Grosmark and
György Buzsáki (Buzsáki lab at NYU), with the NWB conversion by Cody Baker. The dandiset
contains eight sessions of bilateral silicon-probe recordings from dorsal
CA1 of four freely moving Long-Evans rats (Achilles, Buddy, Cicero and Gatsby). Each session is
a pre-run sleep epoch, a maze epoch in which the rat shuttles back and forth for water reward,
and a post-run sleep epoch. Spikes are already sorted, and each unit carries a `cell_type` label
of putative excitatory (pyramidal) or inhibitory (interneuron). Five of the eight sessions used
a linear track and three used a circular maze. Because the linearization used here assumes a
straight track, the analysis is scoped to the five linear-track sessions, which together
contribute 348 pyramidal cells and 82 interneurons.

Files are read over the network with LINDI plus a local chunk cache, so only the bytes actually
touched are transferred rather than the full multi-gigabyte session files.

## What Was Analyzed

Position was linearized by projecting the 2D tracking onto the principal axis of the track, laps
were segmented by direction of travel, and firing-rate maps were built over 5 cm position bins
using only samples where the rat ran faster than 10 cm/s. Spatial selectivity was quantified
with Skaggs information in bits per spike and tested against a circular-shift null that destroys
the spike-position relationship while preserving each train's own rate and burst statistics.
A unit was classified as a place cell if it was putatively excitatory, fired at least 50 spikes
in the running epochs, reached a peak rate of at least 1 Hz, exceeded the 95th percentile of its
own null, and had an odd-versus-even lap rate-map correlation above 0.5. The population code was
then tested two further ways: a Bayesian decoder reconstructing position from population spiking
on held-out traversals, and Poisson GLMs (NeMoS) comparing position, speed, and position-plus-speed
models by held-out McFadden pseudo-R².

Two properties of these particular NWB files needed explicit handling and are documented in the
script. The behavioural `SpatialSeries` objects store the sampling *period* (0.0256 s, or
39.06 Hz) in the `rate` field, so PyNWB reconstructs timestamps 1525 times too sparse and the
2068 s maze epoch appears to span 36 days. The files also ship a linearized position series, but
it is defined on only 7 to 32 % of frames depending on the session; the reconstruction used here
covers all frames and agrees with the file's own values where those exist to r = 1.0000.

## Key Finding

CA1 pyramidal cells fire in restricted, reliable portions of the track, and the population tiles
the environment well enough to decode the rat's position to within a few centimetres. In the
example session (Achilles_10252013), 97 of 120 pyramidal cells (81 %) qualified as place cells in
at least one direction of travel; pooled over all five sessions and four rats the figure is 232
of 348 (67 %). Median Skaggs information was about 0.79 bits/spike for pyramidal cells versus
0.032 bits/spike for simultaneously recorded interneurons (Mann-Whitney p = 9.4e-39 pooled),
which shows the selectivity is a property of the pyramidal population and not a trivial
consequence of the behaviour. Fields had a median half-maximum width of 35 cm, and rate maps
built from odd and even laps correlated at a median r of 0.80. Fields were strongly
direction-specific: sorting cells by their rightward field peak produces a clean diagonal band,
while the same ordering applied to leftward laps is essentially unstructured.

The population code was verified two ways. A Bayesian decoder reading 0.25 s of spiking from the
place-cell population recovered position on held-out traversals with a median error of 4.9 cm
against a chance error of 58 cm, and the result held in every session (5 to 11 cm observed versus
58 to 78 cm chance). Because speed covaries with position on a linear track, a speed-tuned
neuron could in principle mimic a place field, so Poisson GLMs were fitted to separate the two.
Position alone reached a median held-out pseudo-R² of 0.123 against 0.037 for speed alone, and
adding position to a speed-only model improved held-out likelihood for 100 of 107 units. The
spatial tuning is therefore genuine position coding rather than an artefact of the speed profile.

## Files

| File | Contents |
| --- | --- |
| `place_cells_dandi_000044.py` | Consolidated jupytext script, runs end to end |
| `place_cells_dandi_000044.ipynb` | The same analysis as an executed notebook |
| `fig01_behaviour.png` | Tracking, linearization check, occupancy, lap detection |
| `fig02_example_place_cells.png` | Six place cells: lap rasters and directional rate maps |
| `fig03_population.png` | Population rate maps sorted by field peak, plus raw spiking |
| `fig04_statistics.png` | Spatial information vs null, cell types, field geometry |
| `fig05_decoding.png` | Bayesian decoding of position on held-out traversals |
| `fig06_glm.png` | Poisson GLM comparison of position and speed models |
| `fig07_across_sessions.png` | The same measures across all five sessions |
| `session_summary.csv` | Per-session summary statistics |

The numbered scripts `01_load_data.py` through `10_test_glm.py`, `place_cell_lib.py`, and the
`fig_*.png` quality-control images are development scratch from building the pipeline. They are
not needed to reproduce the results; the consolidated script is self-contained.

## Running It

```bash
pip install pynapple lindi pynwb h5py nemos matplotlib scipy tqdm pandas jupytext
MPLBACKEND=Agg python place_cells_dandi_000044.py
```

The first run downloads roughly 400 MB into `./lindi_cache` and takes a few minutes; later runs
read from the cache. The shuffle test is seeded, so the reported numbers are reproducible.
