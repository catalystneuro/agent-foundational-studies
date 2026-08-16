# A continuous ring attractor in the head-direction system, maintained during sleep

This analysis uses [DANDI:000939](https://dandiarchive.org/dandiset/000939), *Large-scale
recordings of head direction cells in mouse postsubiculum* (Duszkiewicz, Peyrache and
colleagues). Each session is a silicon-probe recording from the postsubiculum of one mouse,
alternating home-cage sleep with foraging in an open arena, and includes tracked head direction,
REM/NREM scoring, and a per-unit head-direction flag. Files are streamed from S3 with `remfile`
(disk-cached); nothing is downloaded in full. All analysis is done with pynapple, plus scikit-learn
(Isomap/PCA), Ripser (persistent homology) and NeMoS (GLM).

## What was analysed

Head-direction tuning curves were fitted during waking arena exploration, and the behaviour was
then discarded: every sleep analysis uses spike times only. For each brain state (waking, REM,
NREM) the same pipeline was applied to population vectors binned at 200 ms: Isomap embedding and
an annularity ("ring score"), persistent homology of the point cloud (does it have exactly one
hole?), tuning curves recomputed against the *internal* angle the population defines on its own
manifold, Bayesian decoding with the waking tuning curves, agreement between two disjoint halves
of the population decoding independently, the angular velocity of the decoded bump, pairwise
correlation structure, and a cross-validated Poisson GLM (NeMoS, cyclic B-spline basis) driven by
the internal angle. Every measure is compared against a per-cell circular shuffle that preserves
each cell's firing rate and temporal statistics and destroys only the coordination between cells.
One session (A3716) is worked through in detail; the pipeline is then repeated on all 19 sessions
(19 mice) with at least 20 well-tuned head-direction cells.

## Key finding

The one-dimensional ring structure of the head-direction population survives intact in sleep, when
there is no vestibular or visual reference to support it. In both REM and NREM the population
activity forms an annulus with a single long-lived H1 loop (median ring score 0.64 REM / 0.54 NREM
versus 0.39 / 0.22 for shuffles; longest H1 lifetime 0.74 / 0.54 versus 0.46 / 0.32), and the
coordinate along that ring is the waking head-direction map: recomputing each cell's tuning curve
against the purely internal manifold angle recovers its waking preferred direction (median
circular r = 0.76 in REM and 0.62 in NREM, versus ~0.1 for shuffles; ~0.88 in the example session).
A cross-validated GLM driven by that internal angle predicts held-out sleep spiking about as well
as measured head direction predicts waking spiking (median pseudo-R² 0.37 REM, 0.28 NREM, versus
0.38 for wake/measured HD). The activity is a single coherent bump that moves continuously: two
disjoint halves of the population decode the same angle (circular r 0.64 in REM, disagreement 15°,
versus ~0 and ~87° for shuffles), and the bump drifts at about 30°/s in REM, close to real head
speed, and about 255°/s in NREM.

The evidence is strongest in REM and in the sessions with the largest populations. In NREM the
moment-to-moment measures are noisier, for two reasons visible in the data: the bump moves roughly
five times faster than in REM, so a 200 ms bin smears it, and the population is repeatedly silenced
by cortical DOWN states. Consistent with a resolution limit rather than a missing ring, NREM
split-half agreement scales with the number of recorded cells (0.74 and 0.76 in the two largest
populations, scattering around zero below ~35 cells), while the geometric and ordering measures,
which pool over whole epochs, stay clearly above the shuffle in nearly every session. Because the
recordings are from postsubiculum, this shows that the ring is expressed there; it does not
localise where in the head-direction circuit it is generated.

## Files

| File | Contents |
|---|---|
| `ring_attractor_hd_sleep.py` | consolidated jupytext notebook (percent format), runs end to end |
| `ring_attractor_hd_sleep.ipynb` | the same notebook, executed |
| `hd_io.py` | streaming loader for DANDI:000939 into pynapple objects |
| `hd_core.py` | tuning curves, cell selection, embeddings, ring/topology metrics, decoding, shuffles |
| `hd_glm.py` | NeMoS Poisson GLM on an angular covariate, cross-validated pseudo-R² |
| `hd_figs.py` | figure-drawing routines |
| `sweep_sessions.py` | runs the pipeline on every usable session → `cross_session_metrics.csv` |
| `survey_sessions.py` | dandiset survey used to choose sessions → `session_survey.csv` |
| `figures/fig01..fig10*.png` | all figures |

Requirements: `pynapple`, `pynwb`, `remfile`, `h5py`, `scikit-learn`, `ripser`, `nemos`,
`matplotlib`, `pandas`, `tqdm`. Run with `python ring_attractor_hd_sleep.py` (headless, writes
PNGs) or open the notebook. The cross-session sweep takes roughly 10 minutes on a warm cache and
is skipped if `cross_session_metrics.csv` already exists.
