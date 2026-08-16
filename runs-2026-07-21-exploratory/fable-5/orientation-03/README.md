# Orientation Selectivity in Mouse Visual Cortex from DANDI:000021

This analysis demonstrates orientation selectivity, the classic response property of visual
cortical neurons, using publicly archived Neuropixels recordings from the DANDI Archive.

## Dataset

[DANDI:000021](https://dandiarchive.org/dandiset/000021), "Allen Institute - Visual Coding -
Neuropixels (Brain Observatory 1.1 Stimulus Set)". Each session is a multi-probe Neuropixels
recording from an awake, head-fixed mouse passively viewing a fixed battery of visual stimuli.
Six sessions were analysed (`sub-699733573/ses-715093703`, `sub-703279277/ses-719161530`,
`sub-707296975/ses-721123822`, `sub-716813540/ses-739448407`, `sub-717038285/ses-732592105`,
`sub-718643564/ses-737581020`), streamed over HTTP with `remfile` and a local disk cache; no
file was downloaded in full. Two stimulus blocks were used: drifting gratings (8 directions in
45 degree steps, 5 temporal frequencies, 2 s per trial, 15 repeats per condition, plus blank
sweeps) and static gratings (6 orientations in 30 degree steps, 5 spatial frequencies, 4 phases,
0.25 s per trial).

After quality control (isolation quality, SNR, ISI violations, presence ratio, amplitude cutoff,
minimum firing rate) 1196 units in the six cortical visual areas VISp, VISl, VISrl, VISal, VISam
and VISpm entered the analysis. Per-trial firing rates were computed with pynapple
(`TsGroup.count` over an `IntervalSet` built from the stimulus table), tuning curves were taken
at each unit's preferred temporal frequency, and selectivity was quantified with the standard
vector-strength indices (gOSI, gDSI) and their ratio-based counterparts (OSI, DSI). Each unit's
gOSI was tested against a 500-permutation trial-shuffled null, and a Poisson GLM with a cyclic
B-spline basis over drift direction was fit with NeMoS (`PopulationGLM`) and scored by five-fold
cross-validated held-out log-likelihood against an intercept-only model.

## Key finding

Orientation selectivity is unambiguous in these data. Of the 1196 units, 758 (63 percent) were
significantly modulated by drifting-grating direction, and 633 of those (84 percent) had a gOSI
above their own trial-shuffled null at p < 0.01. The median gOSI among responsive units was
0.239 against a null of 0.066, and the median gDSI was only 0.087, so tuning is dominated by
orientation rather than by direction of motion. The clearest single statement of that is the
population-average tuning curve after aligning each unit to its preferred direction: the
response to the opposite direction of motion, that is, the same bar orientation moving the
other way, stays at 0.68 of the peak, whereas the response to the orthogonal orientation falls
to 0.17. In 96 percent of orientation-selective units the opposite-direction response exceeded
the orthogonal-orientation response, with a median ratio of 2.08. Preferred orientations were
not uniformly distributed: cardinal orientations were over-represented relative to obliques
(Rayleigh test on the fourth harmonic, z = 11.1, p = 1.4e-05; 58 percent of selective units
preferred an orientation within 22.5 degrees of horizontal or vertical).

Two independent controls support the result. Tuning curves built from alternating repeats of
each direction correlated with each other at a median r of about 0.9. More importantly, tuning
transferred across stimulus types: for the 583 units modulated by both stimuli, the preferred
orientation measured from 2 s drifting gratings and from 0.25 s static gratings (a different
block, different spatial frequencies and phases) differed by a median of only 10.9 degrees,
against 48.0 degrees for a unit-shuffled control, with 75 percent of units agreeing to within
22.5 degrees (Mann-Whitney p = 1.1e-73). The GLM analysis gives the same answer from a
model-based direction: 217 of 248 units in the example session predicted held-out spike counts
better with a direction-dependent rate than with a constant rate.

## Files

| File | Contents |
| --- | --- |
| `orientation_selectivity_dandi.py` | Consolidated jupytext script, runs end to end |
| `orientation_selectivity_dandi.ipynb` | Same analysis as an executed notebook |
| `orientation_lib.py` | Tuning-curve, selectivity-index, permutation and GLM computations |
| `dandi_io.py` | DANDI asset resolution and cached streaming access to NWB files |
| `fig01_raw_data_overview.png` | Population raster and running speed during drifting gratings |
| `fig02_example_unit.png` | Example V1 unit: raster, PSTHs and tuning curve |
| `fig03_polar_tuning.png` | Polar tuning curves for twelve units spanning a range of selectivity |
| `fig04_glm_fits.png` | NeMoS Poisson GLM fits, including an untuned negative control |
| `fig05_population_selectivity.png` | gOSI versus the shuffled null, gOSI versus gDSI, selectivity by area |
| `fig06_population_tuning.png` | Normalised tuning heatmaps and the aligned population curve |
| `fig07_preferred_orientation.png` | Distribution of preferred orientations and selectivity by area |
| `fig08_validation.png` | Split-half reliability and drifting versus static grating agreement |
| `summary_by_area.csv` | Per-area summary statistics |
| `drifting_grating_tuning_summary.csv`, `static_grating_tuning_summary.csv` | Per-unit results |
| `results/` | Cached per-session analysis output (pickle) |

## Running it

```
pip install pynapple nemos pynwb remfile h5py dandi tqdm matplotlib scikit-learn jupytext
python orientation_selectivity_dandi.py
```

The first run streams roughly 3 GB of byte ranges from the DANDI S3 bucket into
`/tmp/remfile_cache_000021` and takes about half an hour, most of it spent reading spike times
for the six sessions. Subsequent runs read from that cache and from `results/`, and complete in
a couple of minutes.

## Caveats

Firing rates are computed over the whole presentation without baseline subtraction, so units
with a purely transient onset response are under-weighted. Running speed modulates visual
cortical gain in mice and is not regressed out; because running bouts are not locked to grating
direction this adds trial to trial variance rather than creating apparent orientation tuning,
but it does depress the measured selectivity of some units. The permutation test on gOSI
establishes that a unit's response depends on the stimulus, not specifically that orientation
rather than direction is the relevant variable, which is why the opposite-versus-orthogonal
comparison and the drifting-versus-static cross-check are reported alongside it.
