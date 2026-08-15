# Hippocampal place cells in rat dorsal CA1 (DANDI:000044)

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark & Buzsáki
(2016), *"Diversity in neural firing dynamics supports both rigid and learned
hippocampal sequences"*, Science 351:1440-1443 (the `hc-11` dataset). Four
Long-Evans rats (Achilles, Buddy, Cicero, Gatsby) with bilateral silicon probes
in dorsal CA1 ran back and forth on a novel linear track for water reward at the
two ends, flanked by pre- and post-run sleep. Each NWB file contains sorted
spike times labelled excitatory or inhibitory, 128-channel LFP at 1250 Hz, 2D
and linearized position at ~39 Hz, and epoch boundaries.

The eight sessions in the dandiset are not homogeneous: five used a linear track
(four at 1.6 m, one at 2 m) and three used a circular maze. The directional-pass
logic here assumes a track with two ends, so the three circular-maze sessions
(Achilles 11012013, Cicero 09102014, Gatsby 08282013) are excluded and reported
as excluded. The analysis therefore covers **five sessions from four rats, 231
excitatory CA1 units**.

Files are 5-9 GB each because they contain the full LFP, so nothing is
downloaded whole: `remfile` plus a local disk cache turns each S3 object into a
random-access file and `h5py` reads only the units table, the position series,
and the handful of LFP channels that are actually used.

## What was analysed

Position was restricted to on-track running (the linearized series is NaN
whenever the animal is in the end zones), differentiated within each contiguous
on-track segment, and split into rightward and leftward *passes*: stretches of
constant velocity sign above 0.10 m/s covering at least 0.6 m. Rate maps were
built per direction in 40 spatial bins with occupancy normalisation and mild
Gaussian smoothing.

A unit-direction was called a place field when all three held: Skaggs spatial
information above the 99th percentile of a 500-fold circular-shift null (spike
trains shifted within the maze epoch, which preserves burst structure but
destroys the position relationship), in-field peak rate ≥ 1 Hz, and split-half
stability (odd vs even passes) r > 0.3. Position was then decoded from the place
cells with a Poisson naive-Bayes decoder in 0.2 s bins, trained on odd passes and
evaluated on held-out even passes. Finally, theta phase was taken from the
Hilbert transform of the 6-10 Hz filtered LFP on the channel with the highest
theta/delta power ratio, and spike phase was regressed on the fraction of the
field traversed using the Kempter circular-linear correlation, for well-isolated
fields away from the reward zones.

## Key finding

CA1 pyramidal cells in these recordings are place cells by every standard
criterion. Of 231 excitatory units active while the animal ran, **104 (45%)
carried a significant, stable place field in at least one running direction**
(141 of 462 unit-direction pairs). Their spatial information was a median 1.10
bits/spike against a shuffled median of 0.50, fields were compact (median width
0.20 m at half of peak, median in-field peak rate 7.6 Hz, median sparsity 0.35),
and field peaks tiled the whole track with an enrichment at the two reward ends.
Fields were largely direction-selective: the rate maps of the two running
directions correlated only weakly (median r = 0.31 in the prototype session), and
sorting the leftward maps by the rightward peak order scrambles the diagonal
(Figure 3). The population code was accurate and redundant. A decoder trained on
half the passes localised the animal on the held-out passes to a median error of
**0.092 m against a chance level of 0.447 m**, and the error fell monotonically
from ~0.28 m with 2 cells to ~0.06 m with the full 34-cell ensemble.

The single-cell signature that separates hippocampal place coding from a generic
rate code was also present: across 31 well-isolated fields, spikes precessed to
earlier theta phases as the animal crossed the field, with a median
circular-linear ρ of −0.23 and a median slope of −0.56 theta cycles per field
traversal (20 fields significantly negative, 2 significantly positive; Wilcoxon
signed-rank against ρ ≥ 0, W = 51, p = 1.6 × 10⁻⁵).

Place-cell yield varied substantially by session, from 59% (Achilles 10252013,
97 units analysed) to 8% (Buddy 06272013, 25 units). The Buddy session is not a
failure of the pipeline so much as a limit on its statistical power: its units
fire sparsely while running (median 58 spikes per unit-direction against 108 for
Achilles), and with that few spikes the circular-shift null is broad enough that
genuinely tuned cells cannot clear the 99th percentile. Restricting the count to
unit-directions with at least 30 in-run spikes barely changes the pooled figure
(98 of 223 units, 44%), because the shortfall is concentrated in one session
rather than spread across the population. The Cicero 09172014 session (2 m
track, 18%) is limited the same way.

## Files

| file | contents |
| --- | --- |
| `place_cells_dandi000044.py` | jupytext (percent-format) script, runs end to end |
| `place_cells_dandi000044.ipynb` | the same analysis as an executed notebook |
| `fig01_behavior_and_raster.png` | 2D tracking, linearized position and passes, speed, spike rasters (real time and pass-aligned) |
| `fig02_example_place_cells.png` | six example place cells: spikes per pass and directional rate maps |
| `fig03_population_maps.png` | peak-normalised population rate maps for both directions; field-peak histogram |
| `fig04_spatial_information.png` | observed vs circular-shift spatial information; field width and peak-rate distributions |
| `fig05_directionality.png` | direction selectivity of place fields |
| `fig06_decoding.png` | Bayesian decoding posteriors, error distribution, error vs ensemble size |
| `fig07_theta_precession.png` | theta LFP and spectrum, example phase-precession plots, ρ distribution |
| `fig08_across_sessions.png` | place-cell yield, spatial information, field width, decoding and precession across all five sessions |
| `place_cell_metrics_all_sessions.csv` | per unit-direction metrics for every session |
| `decoding_all_sessions.csv`, `phase_precession_all_sessions.csv`, `session_summary.csv` | tabulated results |
| `results_all_sessions.pkl` | pickled per-session result dictionaries |

## Reproducing

```bash
python place_cells_dandi000044.py          # ~12 min cold, faster with a warm cache
```

Requires `pynapple`, `pynwb`, `remfile`, `h5py`, `numpy`, `scipy`, `pandas`,
`matplotlib`, `tqdm`. Streamed data is cached under `/tmp/remfile_cache_000044`.
All plotting uses the Agg backend and writes PNGs.
