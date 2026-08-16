# Hippocampal place cells in rat CA1 (DANDI:000044)

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), *"Diversity in neural firing
dynamics supports both rigid and learned hippocampal sequences"* (Grosmark & Buzsáki,
Science 2016). Eight sessions from four Long-Evans rats, each combining bilateral
silicon-probe recordings from dorsal CA1 with video tracking. Every session is a
sleep / maze / sleep sandwich; the analysis uses the maze epoch, during which the animal
shuttles back and forth on a linear track for water reward at the ends. Spike sorting,
a pyramidal-cell / interneuron label, and a linearized position signal sampled at 39 Hz
come with the dataset and are taken as given. Five of the eight sessions use a linear
track (four 1.6 m, one 2 m) and are analyzed here; the other three use a circular maze,
whose topology would need separate treatment, and are excluded. The files are 5-9 GB
each, almost entirely raw LFP, so they are streamed with `remfile` and a local chunk
cache and only the spike times and behavior module are ever read.

## What was analyzed

The linearized position trace is defined only while the animal is on the track, so each
contiguous stretch of valid samples that spans at least half the track is one lap, and
the sign of the displacement gives the running direction. Rate maps are
occupancy-normalized firing rates in 40 spatial bins, computed separately for each
running direction over periods of running faster than 5 cm/s and lightly smoothed.
Spatial information is the Skaggs measure in bits per spike, tested against a null built
by circularly shifting each unit's own spike train within the running epochs 500 times;
a unit counts as a place cell in a direction if its spatial information exceeds the 99th
percentile of that null and its peak rate is at least 1 Hz. Split-half (odd versus even
lap) map correlation is computed as an independent check. Position is then decoded from
the population with pynapple's Bayesian decoder in 250 ms bins, five-fold
cross-validated over laps so that rate maps come only from the training laps. Finally a
Poisson GLM with position expanded in a 12-element B-spline basis is fit with NeMoS and
scored on held-out laps against a homogeneous-Poisson null. All computation uses
pynapple objects; the whole pipeline runs over every linear-track session.

## Key finding

CA1 pyramidal cells in this dataset are strongly and reliably spatially tuned. Across
the five sessions, 60-72% of the pyramidal cells active on the track have a significant
place field in a given running direction, and 73-90% have one in at least one direction.
The fields are compact and reproducible: median width at half maximum near 0.3 m on
tracks of 1.6-2 m, median peak rate around 5 Hz, and a median correlation near 0.85
between rate maps built from odd and from even laps. Peak positions cover the entire
track, with the usual over-representation of the two ends where the animal pauses to
drink, and the two running directions produce largely different maps in the same cells.

The population code is decodable, which is the strongest single check on the tuning-curve
description. Bayesian decoding of held-out laps recovers the animal's position to a
median absolute error of 6.8-11.9 cm, against a shuffled baseline of 44-58 cm, and the
error falls monotonically as more cells are included (39 cm with 2 cells, 14 cm with 10,
6.8 cm with the full 105-cell Achilles population). The GLM reaches the same conclusion
from the encoding side: its fitted tuning curves match the histogram rate maps almost
exactly (r = 0.997) and held-out pseudo-R² is clearly positive for the cells classified
as place cells (median 0.09-0.10, up to 0.56).

Two caveats. Spike sorting and the cell-type labels are used as published without
re-curation, so some units may be merges or splits. And because the animal pauses at the
reward zones, fields at the track ends could partly reflect reward or immobility rather
than position alone; the interior fields, which make up most of the population, do not
have this ambiguity.

## Files

| File | Contents |
| --- | --- |
| `hippocampal_place_cells_dandi000044.py` | Consolidated jupytext script (percent format), runs end to end |
| `hippocampal_place_cells_dandi000044.ipynb` | Same analysis as an executed notebook |
| `fig01_behavior.png` | Tracking, extracted laps, running speed, occupancy |
| `fig02_example_cells.png` | Four example place cells: spikes on the position trace, per-lap rasters, rate maps |
| `fig03_population_maps.png` | Peak-normalized rate maps sorted by field location, split-half stability |
| `fig04_spatial_information.png` | Spatial information against each unit's own shuffle null, field statistics |
| `fig05_decoding.png` | Bayesian decoding: posteriors on held-out laps, confusion matrix, error, population-size curve |
| `fig06_glm.png` | NeMoS Poisson GLM: fitted vs measured tuning, held-out pseudo-R² |
| `fig07_multisession.png` | Yields, field statistics and decoding across all five sessions |
| `fig08_all_session_maps.png` | Sorted population rate maps for every session |
| `session_summary.csv` | One row per session: unit counts, laps, place-cell fractions, decoding error |
| `all_unit_stats.csv` | One row per unit x direction: SI, null threshold, p, peak rate, width, sparsity, stability, classification |
| `single_session_unit_stats.csv` | Same for the prototype session (Achilles 10252013) |
| `pf_lib.py`, `decode_lib.py`, `07_multisession.py` | Modular version of the pipeline the notebook was built from; running `07_multisession.py` alone regenerates the two CSVs |
| `assets_000044.json` | Cached DANDI asset list |

## Environment

Python 3.12 with pynapple 0.11.2, nemos 0.2.6, pynwb, remfile, h5py, numpy, scipy,
pandas, matplotlib, tqdm. Figures are written with the Agg backend. The first run streams
about 1.6 GB of NWB chunks into `/tmp/remfile_cache_000044`, out of roughly 57 GB of
files; later runs read from that cache. End-to-end runtime is about 13 minutes once the
cache is warm.
