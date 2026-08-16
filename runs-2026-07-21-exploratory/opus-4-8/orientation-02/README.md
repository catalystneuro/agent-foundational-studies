# Orientation selectivity in the mouse visual system (DANDI:000021)

This analysis demonstrates orientation selectivity from public electrophysiology
data and asks where along the visual pathway it appears.

## Dataset

[DANDI:000021](https://dandiarchive.org/dandiset/000021), the Allen Institute
"Visual Coding - Neuropixels" release, Brain Observatory 1.1 stimulus set. Each
session is a head-fixed mouse watching a monitor while up to six Neuropixels
probes record simultaneously from visual thalamus, primary visual cortex, and five
higher visual areas. Session NWB files are streamed from the DANDI S3 bucket with
`remfile` and an on-disk block cache; nothing is downloaded in full. Two stimulus
blocks are used:

* **drifting gratings**, 2 s presentations, 8 directions x 5 temporal frequencies
  x 15 repeats, with interleaved blank sweeps that provide a matched baseline;
* **static gratings**, 0.25 s presentations, 6 orientations x 5 spatial
  frequencies x 4 phases, in a separate block later in the session.

Because the two blocks are separated in time and share no stimulus parameters
beyond orientation itself, the static gratings are a genuine out-of-sample test of
any orientation preference measured on the drifting gratings.

## What was analysed

Units were kept if they passed the quality metrics the Allen Institute recommends
for this dataset (ISI violations < 0.5, amplitude cutoff < 0.1, presence ratio >
0.9, SNR > 1) and if their peak channel was in a visual cortical area or in the
visual thalamus (LGd, LP). For each unit, the direction tuning curve was taken at
that unit's preferred temporal frequency with the blank-sweep baseline subtracted,
and summarised by the circular-variance gOSI, the classic
(R_pref - R_orth)/(R_pref + R_orth) OSI, and gDSI.

Three things were done to keep the comparison across brain areas honest:

* **A permutation test.** Direction labels are shuffled within temporal frequency
  and the whole pipeline, including the preferred-temporal-frequency selection, is
  repeated 1000 times per unit. This gives both a p-value and the noise floor of
  gOSI for that unit. The noise floor matters: gOSI is positively biased for units
  with few spikes, so raw values are not comparable across areas that differ in
  firing rate. All cross-area comparisons use the noise-corrected value.
* **Tuning at the preferred temporal frequency rather than pooled.** Cortical units
  are often tuned to low temporal frequencies and respond weakly at 15 Hz, so
  pooling all five frequencies dilutes cortical tuning while leaving high-pass
  thalamic units untouched. That would bias exactly the comparison of interest.
* **A trial-level responsiveness test.** A Welch t-test of the best condition's
  trials against the blank sweeps, plus a 1 Hz floor on the evoked response. Without
  it, units firing a couple of spikes per trial reach gOSI near 1 by chance, because
  one positive direction among zeros is maximally selective regardless of how few
  spikes produced it.

Beyond single-unit tuning: a cross-validated linear-discriminant decoder reads
orientation off population spike counts as a function of population size, and a
NeMoS Poisson GLM measures how much of each unit's single-trial spike count is
explained by orientation on held-out trials, using a cyclic B-spline basis on the
doubled direction angle so the basis is 180-degree periodic and cannot capture
direction tuning.

## Key finding

Eight sessions from eight mice were processed, giving 2691 quality-passing units
of which 1594 were visually responsive.

Orientation selectivity is unambiguous in mouse visual cortex. Sixty percent of
visually responsive cortical units are significantly orientation tuned by the
permutation test, the median classic OSI in V1 is 0.67, and the preference is
highly reproducible. Preferred orientation estimated from independent halves of
the drifting-grating repeats agrees at circular r = 0.92, and preferred
orientation measured on the entirely separate static-grating block, which shares
no parameter with the drifting gratings beyond orientation itself, agrees at
circular r = 0.83. Restricting to stationary trials does not change the picture:
selectivity there is in fact modestly higher than on running trials (median gOSI
0.50 vs 0.42), and the preferred orientation shifts by a median of 4.6 degrees, so
the tuning is not an artefact of locomotion-driven gain. Preferred orientations
are not uniformly distributed in cortex; there is a significant bias toward the
cardinal axes (Rayleigh R = 0.16 on doubled angles, p = 9e-10).

The comparison with thalamus is graded rather than all-or-none. LGd, the
first-order relay that feeds V1, is clearly less orientation selective than cortex
by every measure used here: median noise-corrected gOSI 0.096 vs 0.221
(p = 6e-5), 39% vs 60% of units significantly tuned, median GLM orientation
pseudo-R-squared 0.033 vs 0.082 (p = 3e-6), a visibly flatter cross-validated
population tuning curve, and lower population decoding accuracy at every
population size. But LGd selectivity is not zero: a substantial minority of LGd
units are significantly tuned and LGd populations decode orientation well above
chance. That is consistent with the published finding that mouse dLGN contains a
genuine orientation- and direction-selective subpopulation, and it means the
textbook claim that orientation selectivity first appears in cortex is best read
as a statement about degree rather than about presence and absence. LP, a
higher-order nucleus that receives heavy cortical input, sits between LGd and
cortex, which is what its connectivity predicts.

All reported numbers are in `stats_summary.txt`.

## Files

| file | contents |
| --- | --- |
| `orientation_selectivity_dandi.py` | consolidated jupytext notebook, runs end to end |
| `orientation_selectivity_dandi.ipynb` | the same, converted with jupytext |
| `oslib.py` | streaming/loading helpers and the tuning metrics |
| `pipeline.py` | `extract_session` and `analyze_session`, shared by the scripts and the notebook |
| `analyze_helpers.py` | the visual-responsiveness criterion, in one place |
| `sessions_000021.json` | session list with resolved S3 URLs, built from the DANDI API |
| `01_inspect.py` | prints the structure of one session |
| `02_extract.py` | CLI: cache per-trial firing rates for given sessions |
| `04_analyze.py` | CLI: per-unit tuning metrics for every cached session |
| `06_decode.py` | population decoding sweep |
| `07_glm.py` | NeMoS Poisson encoding GLM |
| `dev/` | the figure scripts used during development, superseded by the notebook |
| `fig01`-`fig07` `.png` | figures |
| `stats_summary.txt` | all reported statistics |
| `units.pkl`, `decoding_results.csv`, `glm_results.csv` | per-unit result tables |

### Figures

1. `fig01_raw_activity.png` - 30 s of the drifting-grating block: raster by area,
   population rate, running speed, stimulus epochs. An alignment check before any
   analysis.
2. `fig02_example_unit.png` - one V1 unit, raster and PSTH for all eight
   directions. Responds at two directions 180 degrees apart and is near baseline at
   the orthogonal orientation.
3. `fig03_polar_gallery.png` - polar tuning curves for twelve units. Raw firing
   rate with the spontaneous rate drawn as a reference circle, because a
   baseline-subtracted polar plot clipped at zero looks more dramatic than the data
   warrant.
4. `fig04_population.png` - selectivity by area, fraction significantly tuned,
   cortex vs LGd vs LP, and a cross-validated population tuning curve in which the
   preferred bin is chosen from held-out repeats so that noise alone cannot
   manufacture a peak.
5. `fig05_controls.png` - split-half reliability, drifting vs static gratings, and
   the locomotion control.
6. `fig06_preferred_orientation.png` - distribution of preferred orientations with
   a Rayleigh test on the doubled angles.
7. `fig07_decoding_glm.png` - decoding accuracy vs population size, the GLM
   orientation contribution, and the agreement between the descriptive and
   model-based measures.

## Reproducing

```
python 02_extract.py 715093703 719161530 721123822 732592105 \
                    742951821 750332458 760693773 762120172   # streams and caches
python 04_analyze.py                                 # per-unit tuning metrics
python 06_decode.py                                  # population decoding
python 07_glm.py                                     # NeMoS encoding GLM
python orientation_selectivity_dandi.py              # analysis and all figures
```

The notebook streams a few sessions itself if no cache is present. Streaming is
the slow step, roughly five minutes per session warm and considerably longer cold;
everything after that runs in a couple of minutes.

## Limitations

Unit counts are very unequal across areas because probe placement targeted cortex,
so LGd is the most sparsely sampled structure here and its confidence intervals
are correspondingly wide. Subtracting a per-unit permutation noise floor corrects
the leading rate-dependent bias in gOSI but is not a complete correction.
Receptive-field position relative to the monitor is not controlled, so units whose
receptive field falls near the screen edge see a truncated grating; this adds noise
to the tuning estimates in every area.

## Note on the working directory

An earlier interrupted attempt at this task left files in this directory. They
were moved to `_interrupted_attempt/` and `_prior_run_artifacts/` and none of them
are used here: every result, figure and number in this analysis was produced by the
code listed above. The one thing carried over is the `remfile` block cache in
`/tmp/remfile_cache`, which is a network cache of DANDI byte ranges and made the
streaming step faster.
