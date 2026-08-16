# Head-direction cells in mouse postsubiculum (DANDI:000939)

## Dataset

[DANDI:000939](https://dandiarchive.org/dandiset/000939), "Large-scale recordings of head
direction cells in mouse postsubiculum" (Duszkiewicz, Peyrache and colleagues). Each NWB file
is one animal: a 64-channel silicon probe in postsubiculum (PoSub) recorded across home-cage
sleep, free foraging in a square arena, and free foraging in a triangular arena, with head
direction tracked at 100 Hz and stored as an NWB `CompassDirection` series. The files also
carry the authors' own sleep-state scoring and their own `is_head_direction` unit flag.

This analysis uses ten sessions from ten different animals (all the sessions without
optogenetic manipulation among `sub-A37xx` and `sub-A55xx`), 998 units in total. The files are
20-30 GB each because they contain the raw broadband traces, which we never read: everything is
streamed with `remfile` plus a local byte-range cache, and only the spike times, head-direction
series, position and interval tables are pulled out (about 35 MB per session).

## What was analysed

Analysis is done with Pynapple; the encoding model uses NeMoS.

- Head-direction tuning curves during square-arena foraging, summarised by mean vector length,
  Skaggs directional information and full width at half maximum.
- A circular-shift null (200 random shifts of each spike train within the foraging epoch,
  200 per unit) and a conjunctive classification criterion: mean vector length above the 99th
  percentile of the pooled null for that session, per-cell shuffle p < 0.01, and split-half
  tuning-curve correlation above 0.5.
- Stability within a session (first vs second half) and across environments (square vs
  triangle), testing whether preferred directions move independently or as a rigid group.
- Cross-validated Bayesian decoding of head direction from the HD ensemble (Pynapple
  `decode_1d`, 200 ms bins, tuning curves estimated on the held-out half of the session), and
  decoding error as a function of ensemble size.
- Pairwise spike-count correlation structure of the HD ensemble during foraging, REM sleep and
  non-REM sleep.
- A spatial control: preferred directions recomputed separately within each of the four spatial
  quadrants of the arena.
- A Poisson GLM (NeMoS) with head direction projected onto ten cyclic B-spline basis functions
  as the only covariate, fit on one half of the foraging block and scored out of sample on the
  other.

## Key finding

Postsubicular neurons are strongly and sharply tuned to head direction: 662 of 998 units (66%)
across the ten animals pass the conjunctive criterion, with a median mean vector length of 0.76,
median directional information of 1.5 bits per spike and a median tuning width of 36 degrees,
against 0.09 and 0.04 bits per spike for the remaining units. This recovers 93% of the units the
dataset's authors independently flagged as head-direction cells, and the units flagged here but
not by them are almost entirely units whose waveform they left unclassified and which were
therefore never eligible for their label.

The more informative result is that the population behaves as one coherent internal variable
rather than as a collection of independent direction detectors. Preferred directions tile the
circle uniformly, and a Bayesian decoder trained on one half of a foraging session recovers the
animal's head direction on the held-out half to a median error of 9.1 degrees, improving
monotonically from roughly 60 degrees with two cells to 8 degrees with eighty. When the animal
is moved from the square to the triangular arena the map does not scatter and does not stay
fixed: it rotates by an animal-specific common angle, with individual preferred directions
following that rotation to within a circular standard deviation of about 19 degrees. And during
sleep, when no vestibular or visual heading signal is available, the pairwise correlation
structure of the same ensemble is largely preserved, correlating with the waking structure at
r = 0.83 in REM (9 animals) and r = 0.85 in non-REM (10 animals).

The GLM makes the single-cell claim quantitative: head direction alone accounts for a median
held-out pseudo-R² of 0.31 in HD cells against -0.002 in the rest of the population. A spatial
control rules out the obvious confound, since recomputing each cell's preferred direction
separately within the four spatial quadrants of the arena shifts it by a median of only
5 degrees, so the tuning is directional and not positional. Together these are the properties
expected of a continuous ring attractor whose state is maintained internally and is anchored,
rather than created, by sensory landmarks.

## Files

| File | Contents |
| --- | --- |
| `head_direction_cells_dandi000939.py` | Consolidated jupytext script, runs end to end |
| `head_direction_cells_dandi000939.ipynb` | Same, as a notebook |
| `fig01_raw_data.png` | Head direction, position, trajectory and a raster of HD cells sorted by preferred direction |
| `fig02_tuning_examples.png` | Polar tuning curves for eight HD cells and four untuned units, with split-half overlays |
| `fig03_significance.png` | Observed mean vector length against the circular-shift null; agreement between tuning measures; pooled distribution |
| `fig04_stability_crossenv.png` | Split-half stability; preferred directions square vs triangle; residuals after removing the population rotation |
| `fig05_population_tuning.png` | Sorted population tuning matrix; distribution of preferred directions; tuning widths |
| `fig06_decoding.png` | Population activity bump with true and decoded heading; held-out decoding accuracy; error vs ensemble size |
| `fig07_sleep_structure.png` | Pairwise correlation vs angular offset by brain state; wake vs REM scatter; preservation across states |
| `fig08_glm.png` | GLM fits against empirical tuning curves; held-out pseudo-R² vs tuning strength |
| `fig09_across_animals.png` | Per-animal summary of HD fraction, tuning strength, decoding error and cross-environment rotation |

Intermediate artefacts are written to `cache/` (extracted per-session arrays) and `results/`
(analysis and GLM output). Deleting either forces a recomputation; deleting `cache/` forces a
re-stream from DANDI. `assets_000939.json` maps session names to DANDI asset IDs and is fetched
from the DANDI API on first run.

The development scripts (`01_*` through `10_*`, `hd_lib.py`) are the incremental prototypes the
consolidated notebook was built from and are kept for provenance; they are not needed to
reproduce the results.
