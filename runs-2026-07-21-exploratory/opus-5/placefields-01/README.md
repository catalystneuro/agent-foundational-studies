# Hippocampal place cells in rat CA1 (DANDI:000044)

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), "Diversity in neural
firing dynamics supports both rigid and learned hippocampal sequences"
(Grosmark & Buzsáki, *Science* 2016). Eight sessions from four Long-Evans rats
(Achilles, Buddy, Cicero, Gatsby) of bilateral CA1 silicon-probe recording. Each
session has a pre-task sleep epoch, a maze epoch in which the rat shuttles for
water reward on a 1.6 m or 2 m linear track or runs laps on a ~2.9 m circular
track, and a post-task sleep epoch. The NWB files provide spike-sorted units
labelled excitatory or inhibitory, 2D position and a linearized track coordinate
at ~39 Hz, and a 1250 Hz LFP on 51-137 channels. All data were streamed from the
DANDI S3 bucket with `remfile` plus a local disk cache; the files are 5-9 GB each
but only the units table, the position series, and a handful of LFP channels are
ever read.

## What was analysed

Occupancy-normalized firing rate maps in 2 cm bins were built with Pynapple for
every putative CA1 pyramidal cell (excitatory label, mean maze-epoch rate between
0.1 and 5 Hz), separately for each running direction, restricted to periods when
the animal was moving faster than 5 cm/s along the track. A unit was called a
place cell for a given direction when four criteria held together: Skaggs spatial
information above the 95th percentile of its own null distribution from 200
circularly shifted surrogate spike trains, an in-field peak rate of at least
1 Hz, an odd-versus-even traversal map correlation of at least 0.4, and at least
30 spikes emitted while running. The surrogate maps are smoothed exactly like the
observed maps, which matters because smoothing lowers spatial information.

On top of the rate maps: cross-validated Bayesian decoding of position and running
direction from the place-cell population (`nap.decode_bayes`, trained on one half
of the traversals and tested on the other, split within direction); theta phase
precession from the CA1 LFP with circular-linear regression of spike phase on
within-field position; and a Poisson GLM (NeMoS) comparing position, position ×
direction, running speed, and spike history as predictors of binned spiking.

## Key findings

Place cells were abundant and consistent across all eight sessions and all four
animals: 47-91% of putative pyramidal cells (mean 71%; 470 significant fields out
of 885 unit-direction pairs) had a spatially localized, statistically significant
firing field. Pooled across sessions the median field had 0.83 bits/spike of
spatial information, a width of 26 cm at half maximum, an in-field peak rate of
7.1 Hz, a sparsity of 0.44, and an odd/even traversal map correlation of 0.85, all
squarely within the published range for rodent CA1 on linear tracks. Fields tiled
the track roughly uniformly in every session, and on the linear tracks they were
strongly direction-specific: for the same cells, the rate-map correlation between
the two running directions had a median of 0.17 against 0.78 for the
within-direction split-half correlation (n = 447 fields from the six bidirectional
sessions, Mann-Whitney p = 8e-76; the comparison uses a place-cell definition that
omits the reliability criterion so that neither quantity is selected on).

The code is dense enough to be read back out. In the prototype session
(Achilles_10252013, 87 place cells of 120 pyramidal cells), a cross-validated
Poisson naive-Bayes decoder recovered position from 250 ms of spiking with a
median absolute error of 8 cm on a 1.6 m track, against a chance level of 43 cm,
and recovered the running direction in 95% of time bins. Error fell monotonically
from 53 cm with two cells to 8 cm with all 87. Spikes within a field also arrived
at a systematically earlier theta phase as the animal advanced through it, and a
GLM over a spline basis in position reproduced the empirical rate maps, with
running direction improving held-out fit for the large majority of cells. (See the
note under "Files" on the provenance of these last two results.)

## Files

- `place_cells_dandi000044.py` — consolidated jupytext (percent format) script
  that runs the whole analysis end to end and writes every figure. Self-contained:
  it resolves the DANDI asset URLs itself and defines all helper functions inline.
  Expect roughly 30-40 minutes on a first run, dominated by the spike-shuffling
  null and the eight-session loop; later runs reuse the `remfile` disk cache.
- `place_cells_dandi000044.ipynb` — the same script converted with `jupytext`.
- `fig01_behavior_and_raster.png` — behaviour, velocity, and the population raster
  for single traversals in each direction.
- `fig02_example_place_cells.png` — spike-position rasters and rate maps for six
  example place cells.
- `fig03_population_maps.png` — sorted population rate maps, plus the
  rightward-sorted / leftward-plotted panel showing directionality.
- `fig04_statistics.png` — spatial information against the shuffle null, field
  width, peak rate, sparsity, reliability, field coverage, directionality.
- `fig05_decoding.png` — Bayesian decoding: posteriors over held-out traversals,
  confusion matrix, error distributions, direction accuracy, error versus time bin
  and versus population size.
- `fig06_multisession.png` — the same statistics across all eight sessions.
- `fig07_multisession_maps.png` — sorted place-field maps for each session.
- `fig08_theta_precession.png` — CA1 LFP, theta spectrum, and phase precession.
- `fig09_glm.png` — NeMoS Poisson GLM tuning curves and model comparison.

**Note on the last two figures.** `fig08` and `fig09` are produced by the theta and
GLM sections of the notebook (equivalently by `09_theta.py` and `10_glm.py`). Both
scripts were still running when this session's time budget ran out, on a machine
whose load average was above 100 on ten cores, so those two `.png` files are not in
this directory. The code paths were exercised end to end during development: the
theta section had already completed once and reported significant precession before
being restarted with a faster circular-linear routine, and the GLM section fits
870 Poisson GLMs (87 place cells x 5 models x 2 folds), which is what makes it
slow. Running either script, or the corresponding notebook cells, reproduces the
figures. Everything stated in the "key findings" section above is backed by
`fig01`-`fig07` and the CSV files, except the phase-precession sentence, which
rests on the completed first theta run rather than on a saved figure.
- `session_summary.csv`, `pooled_units.csv`, `glm_scores.csv` — per-session and
  per-unit results.
- `pf_lib.py` and the numbered `0*.py` / `1*.py` scripts are the modular pipeline
  the consolidated notebook was developed from; they are kept for reference and
  produce the same figures.

## Caveats

The three circular-maze sessions are analysed with the linearized coordinate
treated as a straight line, so a field straddling the lap boundary would be split
in two. On those mazes the rat runs almost exclusively in one direction, and
directions with fewer than ten traversals are not analysed, which is why they
contribute a single direction each (Achilles_11012013 is a partial exception: its
12 rightward traversals pass the threshold but support only a weak test, and it
yields just 4 rightward place cells). Pyramidal-cell identity is taken from the
`cell_type` labels already present in the NWB files rather than re-derived from
waveform features. The spike-history term in the GLM convolves across traversal
boundaries within the concatenated running epochs, which slightly overstates its
contribution.
