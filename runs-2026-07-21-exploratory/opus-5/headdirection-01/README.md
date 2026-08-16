# Head-direction cells in mouse postsubiculum (DANDI:000939)

This directory contains an end-to-end demonstration of head-direction (HD) cells
using real extracellular recordings streamed from the DANDI Archive. All
analysis is done with [Pynapple](https://pynapple.org); the NWB files are read
over S3 with `remfile` and a local disk cache, so the multi-GB raw traces in
those files are never transferred.

## Dataset

[DANDI:000939](https://dandiarchive.org/dandiset/000939), "Large-scale
recordings of head direction cells in mouse postsubiculum" (Duszkiewicz,
Peyrache and colleagues). Silicon-probe recordings from mouse postsubiculum
during open-field foraging in two differently shaped arenas, separated by
home-cage rest with scored sleep states. Each file provides spike times, tracked
head direction and position, epoch and sleep-state interval tables, and per-unit
labels including the authors' own `is_head_direction` flag.

Six sessions from six mice were analysed (subjects A3701, A3702, A3703, A3705,
A3706, A5505; 593 units in total). The single-session walkthrough uses
`sub-A3705/sub-A3705_ses-200306`.

## What was analysed

- Directional tuning curves, with significance assessed against a null built by
  circularly shifting the head-direction signal in time relative to the spikes,
  plus a mean-vector-length (MVL) effect-size criterion.
- Stability: tuning curves from interleaved 60 s blocks, and from the two
  arenas separately.
- Population structure: tuning curves sorted by preferred direction, pairwise
  spike-count correlation as a function of the offset between preferred
  directions, and the population activity bump over time.
- Bayesian decoding of head direction from population spiking in 200 ms bins,
  scored on held-out data, and applied to REM and NREM sleep.

## Key finding

Postsubicular units are strongly and reproducibly tuned to head direction. In
the example session, 62 of 117 units pass an MVL threshold of 0.3 on top of the
shuffle test (91% of units are statistically significant at p < 0.01, which is
why an effect-size floor is needed with more than an hour of data); this agrees
with the dataset's own published labels on 86% of units. Their tuning curves are
essentially identical across interleaved halves of the session (median
correlation 0.97), their preferred directions tile the full circle, and their
co-firing falls off monotonically with the difference between their preferred
directions (r = -0.63 across 1891 pairs). A Bayesian decoder trained on tuning
curves from the square arena recovers the animal's heading in the triangular
arena to a median absolute error of 13.6 degrees, with 85% of 200 ms bins within
30 degrees, against 114 degrees for a control that permutes which tuning curve
belongs to which cell. Across the six sessions, 44 to 70% of units are HD cells
(367 of 593), and held-out within-arena decoding error is 7 to 11 degrees in
every session.

The population behaves as a single rigid ring rather than a set of independent
cells. Between the two arenas the preferred directions rotate coherently by a
session-specific angle, ranging from 3 degrees in one mouse to 166 degrees in
another, with an interquartile range of only 7 to 11 degrees across cells within
a session. That rotation is exactly what breaks cross-arena decoding: the median
signed decoding error tracks minus the median rotation session by session, and
subtracting that single number per session restores the error to 10 to 14
degrees. The ring structure also outlives the behaviour. During sleep, when the
head is still, the same decoder reports a direction that keeps moving: in REM it
moves as smoothly as it does in waking (81% of consecutive 200 ms bins within
18 degrees of each other, versus 74% in wake and 9% for shuffled bins), while in
NREM it jumps between directions much faster. Pairwise correlations measured
during NREM remain organised by the waking preferred directions (r = -0.61),
which is the signature of an internally maintained attractor rather than a
stimulus-driven response.

## Files

| File | Purpose |
| --- | --- |
| `head_direction_cells_dandi.py` | Consolidated jupytext script; runs end to end |
| `head_direction_cells_dandi.ipynb` | The same, converted and executed |
| `hd_lib.py` | Loading and circular-statistics helpers |
| `01_inspect.py` | Raw NWB structure dump used during exploration |
| `02_validate_streams.py` | Per-stream validation and figure 1 |
| `03_single_session.py` | Single-session analysis, caches `single_session_results.pkl` |
| `04_figures.py` | Figures 2-6 from the cached results |
| `05_multi_session.py` | Six-session pipeline and figure 7 (`REPLOT=1` redraws from cache) |
| `fig01_data_validation.png` | Session structure, tracking, trajectory, occupancy, raster |
| `fig02_tuning_curves.png` | Polar tuning curves for tuned and untuned units |
| `fig03_population_statistics.png` | MVL and information vs shuffles, classification |
| `fig04_stability.png` | Split-half and cross-arena stability |
| `fig05_population_structure.png` | Sorted tuning curves, pairwise correlations, activity bump |
| `fig06_decoding.png` | Decoding accuracy, controls, and decoding during sleep |
| `fig07_multi_session.png` | Six sessions: HD fractions, decoding, ring rotation |
| `multi_session_summary.csv` | Per-session summary statistics |
| `multi_session_units.csv` | Per-unit statistics pooled across sessions |

## Reproducing

```bash
python head_direction_cells_dandi.py   # single-session walkthrough, figures 1-6
python 05_multi_session.py             # six sessions, figure 7
```

Requires `pynapple`, `pynwb`, `remfile`, `h5py`, `numpy`, `pandas`,
`matplotlib`, `tqdm` and `requests`. Streaming the seven session-loads used here
put 1.8 GB into `/tmp/remfile_cache_000939`, against about 165 GB of NWB files
on S3; later runs read from that cache.
Figures are written with the Agg backend and no interactive windows are opened.
