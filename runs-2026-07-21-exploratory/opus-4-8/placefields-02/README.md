# Hippocampal Place Cells in Rat CA1 (DANDI:000044)

This directory contains a self-contained demonstration of hippocampal place
cells built directly from archived data on the DANDI Archive. Everything is
streamed from the archive with LINDI, so no full file download is required even
though the source NWB files are 5 to 9 GB each.

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), "Diversity in neural
firing dynamics supports both rigid and learned hippocampal sequences" (Grosmark
and Buzsáki, *Science* 2016; the `hc-11` dataset). Eight bilateral
silicon-probe recordings from dorsal CA1 of four male Long-Evans rats. Each
session concatenates a long PRE sleep epoch in the home cage, a novel MAZE
running epoch in an unfamiliar room, and a long POST sleep epoch. Spikes are
already sorted, and every unit carries an `excitatory` or `inhibitory` label.

Only the MAZE epoch is used here. Five of the eight sessions ran the animal on a
linear platform (four on a 1.6 m track, one on a 2 m track); the three circular
platform sessions are excluded so that a single linear coordinate means the same
thing in every session. The analysis prototypes on
`sub-Achilles/ses-Achilles-10252013` and is then repeated unchanged on all five
linear sessions.

## What Was Analyzed

Running periods are extracted from the linearized position (speed above 5 cm/s,
position tracked, intervals shorter than 0.5 s dropped) and split by travel
direction. For each direction, occupancy-normalized firing-rate maps are built
in 2 cm bins with `pynapple.compute_tuning_curves` and lightly smoothed. Skaggs
spatial information is tested against a null distribution produced by circularly
shifting each spike train inside the concatenated run epochs, which destroys the
spike-position relationship while preserving spike count and fine-timescale
autocorrelation. A unit counts as a place cell in a direction if its information
exceeds the 95th percentile of its own null over 500 shuffles and its peak rate
is at least 1 Hz. Finally, position is decoded from the population with
pynapple's Bayesian decoder, cross-validated by estimating the rate maps on
odd-numbered traversals and decoding the even-numbered ones and vice versa.

## Key Finding

Across the five linear-track sessions from four rats, 248 of 302 dorsal-CA1
pyramidal cells that were active on the track (82%) had a statistically reliable
place field in at least one travel direction. Fields were compact (median width
at half maximum 24 cm on tracks of 1.6 to 2 m), the median spatial information of
a place cell was 0.65 bits per spike, and the population covered the whole track
with no large gaps. Fields were also strongly direction selective: sorting the
population by the location of the rightward field produces a clean diagonal for
rightward runs and an essentially unstructured map for leftward runs over the
same cells, and the median absolute directionality index was 0.38.

The population code is accurate enough to invert. A Bayesian decoder trained on
half of the traversals recovered the animal's position on the held-out half to a
median absolute error of 10 cm in 200 to 250 ms bins, against a chance level of
47 cm obtained by shuffling the pairing between decoded and true positions, and
the error fell monotonically as cells were added to the decoder. These are the
textbook signatures of the hippocampal place code, reproduced here from archived
data with a pipeline that streams the recordings rather than downloading them.

## Files

| File | Contents |
| --- | --- |
| `place_cells_dandi000044.py` | Jupytext (percent format) notebook that runs the whole analysis end to end and writes every figure |
| `place_cells_dandi000044.ipynb` | The same notebook in Jupyter format, executed, with outputs and inline figures |
| `place_cells_lib.py` | Loading, preprocessing, rate-map, statistics and shuffling helpers used by the notebook |
| `figures/fig01_session_overview.png` | Session structure, linearized position, velocity, and a 30 s raster with position overlaid |
| `figures/fig02_behavior_and_lfp.png` | 2-D trajectory colored by linear position, occupancy, speed distribution, raw CA1 LFP with its theta component and a simultaneous raster |
| `figures/fig03_example_place_cells.png` | Six example place cells: spikes by position and traversal, and the rate map per direction |
| `figures/fig04_population_maps.png` | Normalized rate maps for the whole population, sorted by rightward peak, plus the directionality index distribution |
| `figures/fig05_statistics.png` | Observed versus shuffled spatial information, p-value distributions, field width, peak and mean rate, sparsity |
| `figures/fig06_decoding.png` | Cross-validated Bayesian decoding: posterior over traversals, confusion matrix, error distribution, error versus population size |
| `figures/fig07_multisession.png` | The same statistics across all five sessions |
| `session_summary.csv`, `results_summary.json` | Per-session and pooled numbers |

`place_cells_dandi000044.py` imports `place_cells_lib.py`, so the two files need
to sit in the same directory. Running the notebook from scratch takes roughly
15 minutes, most of it spent streaming spike times for the four additional
sessions; LINDI chunks are cached under `/tmp/lindi_cache` (override with the
`LINDI_CACHE_DIR` environment variable), so re-runs are much faster.

## Requirements

`pynapple >= 0.11`, `lindi`, `pynwb`, `h5py`, `numpy`, `scipy`, `pandas`,
`matplotlib`, `xarray`, `tqdm`, `jupytext`.

## Notes on the Source Files

Two properties of this conversion are worth recording. The position
`TimeSeries` objects store the sampling *period* (0.0256 s) in the `rate` field
rather than the rate, so the timebase has to be reconstructed as
`starting_time + arange(n) * rate`; `n_samples * rate` reproduces the MAZE epoch
duration to within a millisecond, which is how the pipeline verifies it. The
linearized position is also left as NaN whenever the animal is off the track
proper, in the reward areas past either end, so roughly 13 to 32% of MAZE samples
carry a usable position depending on the session. That is the curation the
original authors applied and it is exactly the restriction a one-dimensional
place-field analysis wants.
