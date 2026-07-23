# Orientation Selectivity in Mouse Visual Cortex (DANDI:000021)

## Dataset

[DANDI:000021](https://dandiarchive.org/dandiset/000021), the Allen Institute
"Visual Coding - Neuropixels" dataset. Eight of the 32 session-level NWB files
were analysed, one per mouse:

```
715093703  750749662  751348571  754829445  755434585  756029989  763673393  799864342
```

Each session provides simultaneously recorded spike times from hundreds of
sorted units across primary visual cortex (VISp), five higher visual areas
(VISl, VISrl, VISal, VISpm, VISam) and the dorsal lateral geniculate nucleus
(LGd), together with the timing and parameters of every visual stimulus.
Files are streamed from S3 with `remfile` and a local disk cache; the 2 to 3 GB
session files are never downloaded in full.

## What was analysed

Two stimuli from the Brain Observatory battery were used. **Drifting gratings**
give 8 drift directions crossed with 5 temporal frequencies, 2 s per trial, 15
repeats per condition. **Static gratings** give 6 orientations crossed with 5
spatial frequencies and 4 phases, 0.25 s per trial. Orientation is the only
property the two stimuli share, so the second acts as an independent test of any
tuning found in the first.

Units were kept if they passed the Allen Institute's recommended quality cutoffs
and were visually responsive (peak condition rate above 1 Hz and above 1.5 times
the blank-screen rate), leaving 1346 of 2696 units. For each unit the pipeline
computes trial-by-trial firing rates, direction tuning curves, the global
orientation and direction selectivity indices (gOSI, gDSI) and their classical
counterparts (OSI, DSI), a 500-permutation shuffled-stimulus null for both
indices, a split-half estimate of preferred orientation, the same tuning
measured on static gratings, and a NeMoS Poisson-GLM model comparison between a
constant-rate model, a 180-degree-periodic model of the stimulus response and a
360-degree-periodic one, scored by 5-fold cross-validated held-out
log-likelihood.

## Key finding

Orientation selectivity is unambiguous in these recordings. Direction tuning
curves of responsive V1 units are bimodal with peaks 180 degrees apart and
troughs at the orthogonal orientation, which is the signature of tuning to the
axis of the grating rather than to its direction of motion. The median
noise-corrected gOSI in V1 is 0.219 against a shuffled-stimulus null of 0.038,
and 94% of responsive V1 units reach individual significance. The tuning is not
an artefact of noise or of the particular stimulus: preferred orientation
estimated from one random half of the drifting-grating trials agrees with the
estimate from the other half to a median of 2.9 degrees (chance is 45 degrees),
and preferred orientation measured with 2 s drifting gratings agrees with
preferred orientation measured with 0.25 s static gratings to a median of 11.4
degrees. The GLM comparison says the same thing in a different currency: adding
a 180-degree-periodic function of the stimulus to a constant-rate Poisson model
improves held-out prediction for 97% of responsive V1 units, against 11% when
the same procedure is run on shuffled stimulus labels.

Selectivity is graded across regions, and the ordering is identical in all eight
sessions: V1 (median noise-corrected gOSI 0.219, n = 288) is above the five
higher visual areas (0.159, n = 919), which are well above LGd (0.046, n = 139;
V1 versus LGd, Mann-Whitney p = 6e-31). LGd units are not completely untuned:
three quarters of them reach individual significance and their preferred
orientation partly survives the drifting-versus-static test (median 30 degrees
against 45 at chance), which is consistent with the reported population of
orientation-biased cells in mouse dLGN. Direction selectivity, in contrast, is
weak throughout. The noise-corrected gDSI is near zero for most units even where
gOSI is large, and in the GLM the direction term buys roughly a fortieth of the
held-out log-likelihood that the orientation term buys, although it is
detectable in about 60% of units.

## Files

| file | contents |
| --- | --- |
| `orientation_selectivity_dandi.py` | consolidated jupytext script, runs end to end |
| `orientation_selectivity_dandi.ipynb` | the same, converted and executed |
| `orientation_lib.py` | asset discovery, NWB streaming, trial rates, selectivity metrics, per-session pipeline |
| `glm_lib.py` | NeMoS Poisson-GLM model comparison |
| `figures.py` | all figure code |
| `01_screen_sessions.py` | surveys all 32 sessions for stimuli and unit yield, writes `session_screen.csv` |
| `02_run_all_sessions.py` | batch driver for the eight-session analysis, writes `results/` |
| `session_assets.csv` | cached DANDI asset list with resolved S3 URLs |
| `session_screen.csv` | per-session stimulus and unit-yield survey |
| `results/units_*.csv` | per-unit results, one file per session, plus the pooled table |
| `fig0*.png` | figures |

Figures 1 to 4 are single-unit examples from session 715093703; figures 5 to 8
pool all eight sessions.

1. `fig01_raw_activity.png` - population raster and rate during 12 consecutive gratings
2. `fig02_example_tuning_curves.png` - the 12 most selective V1 units
3. `fig03_raster_and_polar.png` - rasters, polar tuning and direction-by-temporal-frequency matrices
4. `fig04_glm_example_fits.png` - fitted GLMs on orientation- and direction-selective units
5. `fig05_population_selectivity.png` - selectivity against the shuffled null, by region
6. `fig06_tuning_reliability.png` - split-half and cross-stimulus agreement
7. `fig07_orientation_vs_direction.png` - gOSI versus gDSI, preferred orientations, GLM model selection
8. `fig08_multisession_summary.png` - per-session medians and pooled distributions

## Running it

```bash
pip install pynapple nemos pynwb remfile h5py requests tqdm matplotlib scikit-learn jupytext
python orientation_selectivity_dandi.py
```

Cached intermediates (`session_assets.csv`, `results/`, and `proto_units.csv`
plus `proto_raw.pkl` if kept) are reused if present; delete them to recompute
from the archive. `proto_raw.pkl` holds the raw spike trains of the prototype
session and is about 200 MB, so it is not committed here and will be rebuilt on
first run. A cold run takes roughly an hour, almost all of it streaming spike
times.

## Caveats

Firing rates were averaged over the whole presentation, so the analysis measures
mean rate and ignores the F1 component that distinguishes simple from complex
cells. Running speed and pupil size modulate gain throughout the mouse visual
system; because stimulus conditions are randomly interleaved this adds variance
rather than apparent orientation tuning, but it was not regressed out. Area
assignments come from the Allen CCF registration stored in the NWB electrode
table and are taken at face value, so units near a boundary may be mislabelled.
The eight sessions were chosen for having usable LGd yield, which is a selection
on probe placement rather than on any response property.
