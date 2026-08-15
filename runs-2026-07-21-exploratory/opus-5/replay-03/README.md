# Decoding hippocampal replay during sharp-wave ripples

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark & Buzsáki (2016),
*Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences*
(the `hc-11` dataset). Each of the eight sessions records dorsal CA1 with a 128-site silicon
probe in a PRE-sleep → novel track → POST-sleep structure, and provides spike-sorted units
labelled excitatory or inhibitory, 1250 Hz LFP, tracked position, and scored brain states.
Everything is streamed from S3 with `remfile` and a chunk-level disk cache; no file is
downloaded in full. All spike-train, interval and position handling uses Pynapple.

The prototype session is `sub-Achilles/ses-Achilles-10252013` (137 units, 120 putative
pyramidal cells, 1.6 m linear track). The pipeline is then run unchanged on the five sessions
that use a straight track; the three circular-maze sessions are excluded because their
linearisation is an arc length rather than an affine function of x, which the position loader
asserts against.

## What was analysed

1. **Place fields.** Linear position is recovered for the whole maze epoch (the archived
   `LinearizedPosition` is masked to the authors' run epochs but is an exact affine function of
   the raw x coordinate, residual below 1e-15 m, so the same map is applied to every tracked
   on-track sample). Direction-specific tuning curves are computed in 4 cm bins over running
   periods above 5 cm/s. The decoding ensemble is every putative excitatory cell with a peak
   rate ≥ 1 Hz that fired ≥ 50 spikes while running (100/120 cells in the prototype session).
   A fixed Skaggs-information threshold was tried first and rejected: it left as few as 11 cells
   in one session and more than doubled the cross-validated decoding error there (19.6 cm
   against 10.5 cm). Spatial information is still computed and reported, but descriptively.
2. **Decoder validation.** The memoryless Bayesian decoder (Zhang et al., 1998) recovers the
   animal's actual position during running with a median error of **3.0 cm** in-sample and
   **4.1 cm** with templates fit on odd laps and tested on even laps.
3. **Ripples.** The pyramidal-layer channel is chosen as the one with the highest density of
   ripple-band transients, then band-passed at 140–250 Hz; events are envelope excursions above
   3 robust SD that reach 6 robust SD, lasting 30–300 ms, with REM excluded. The envelope is
   scaled by median and MAD rather than mean and SD, which matters: on one session a handful of
   artifacts reaching 20x the ripple amplitude inflated the SD enough that genuine ripples never
   crossed a 4 SD threshold, collapsing the detected rate to 0.016 Hz against roughly 0.25 Hz
   elsewhere. The prototype session yields 7219 POST and 7823 PRE ripples (0.43–0.49 Hz), median
   duration ~50 ms, and a ripple-triggered average oscillating near 170 Hz.
4. **Candidate replay events.** Population-burst events of the place-cell ensemble (smoothed
   multiunit rate above 3 SD, 100–500 ms) that contain a ripple peak and have ≥ 5 active cells.
5. **Decoding and testing.** Each event is decoded in 20 ms bins against both direction
   templates and scored by the posterior-weighted correlation between decoded position and time.
   Significance requires p < 0.025 against *both* a column-cycle shuffle and a time-bin
   permutation (500 each; Bonferroni for the two templates). A **cell-identity shuffle** of the
   place fields, which preserves every spike time and event boundary while destroying the spatial
   code, calibrates the empirical false-positive rate of the whole procedure.

## Key finding

During POST-sleep sharp-wave ripples the CA1 population re-expresses ordered spatial
trajectories from the track. The decoded posterior sweeps smoothly and monotonically across the
full 1.6 m track within 100–200 ms, in both the forward and the reverse direction, at a median
speed of about 5.8 m/s, roughly ten times the animal's actual running speed. Individual events
are visible directly in the raw spikes as a diagonal in the place-cell raster ordered by field
position (`figures/04_replay_examples.png`, `figures/06_replay_gallery.png`).

The effect is well above the pipeline's own false-positive rate. In the prototype session
11.9% of 2327 POST candidate events pass both shuffle tests, against 6.0% for the cell-identity
shuffled control (χ² p = 1.9e-12; |weighted correlation| Mann-Whitney p = 7.6e-09). PRE-sleep
ripples, recorded before the animal had ever run the track, are statistically indistinguishable
from that control on both measures (6.9%, χ² p = 0.20; |r| p = 0.56), and POST exceeds PRE on
both (χ² p = 3.8e-09; |r| p = 2.9e-09). That contrast is what ties the POST-sleep sequences to
the experience rather than to any standing structure in the ensemble.

Two caveats. The conjunction of two shuffle tests at p < 0.025 each is not exactly a 5% test,
which is why the cell-identity control rather than the nominal level is used as the chance line.
And the direction template that maximises |r| is chosen per event before testing, which is what
the Bonferroni correction across the two templates accounts for.

<!--MULTI-->

## Files

| file | contents |
| --- | --- |
| `hippocampal_replay_analysis.py` | consolidated jupytext script, runs end to end |
| `hippocampal_replay_analysis.ipynb` | the same as an executed notebook |
| `replaylib.py` | streaming/IO, position recovery, ripple detection, Bayesian decoder |
| `pipeline.py` | the analysis packaged as reusable per-session functions |
| `01_load_inspect.py` … `06_multi_session.py` | the staged prototype scripts |
| `figures/*.png` | all figures |
| `cache/*.csv`, `cache/*.npz` | per-event results and summaries |

Figures, in order: session overview and channel selection; place fields and decoder validation;
ripple detection; individual replay events; replay statistics; a gallery of decoded
trajectories; the cross-session summary.

## Reproducing

```bash
python 01_load_inspect.py      # inspect, choose the ripple channel
python 02_place_fields.py      # templates + decoder validation
python 03_ripples.py           # SWR detection in PRE and POST
python 04_replay_decode.py     # decoding, shuffles, controls
python 05_visualize_replay.py  # replay figures
python 06_multi_session.py     # all straight-track sessions (~30 min)
```

Requires `pynapple`, `pynwb`, `remfile`, `h5py`, `numpy`, `scipy`, `pandas`, `matplotlib`,
`tqdm`, `requests`, and `jupytext` for the notebook conversion.
