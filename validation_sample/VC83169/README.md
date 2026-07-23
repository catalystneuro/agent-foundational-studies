# Orientation Selectivity in the Mouse Visual System

This analysis demonstrates orientation selectivity, the property first described by Hubel
and Wiesel in cat striate cortex, using public Neuropixels recordings from the DANDI
Archive. The specific claim being tested is the classical one: thalamic relay neurons that
feed primary visual cortex have centre-surround receptive fields and are largely
indifferent to the angle of a grating, whereas their cortical targets respond strongly to
one orientation and weakly to the orthogonal one.

## Dataset

[DANDI:000021](https://dandiarchive.org/dandiset/000021), the Allen Institute Visual Coding
Neuropixels dataset (Brain Observatory 1.1 stimulus set). Each session records
simultaneously from the dorsal lateral geniculate nucleus (LGd), the lateral posterior
nucleus (LP), primary visual cortex (VISp) and five higher visual cortical areas while a
head-fixed mouse views a fixed battery of visual stimuli. Two stimuli isolate orientation:
drifting gratings (2 s, 8 drift directions crossed with 5 temporal frequencies, 15 repeats)
and static gratings (0.25 s, 6 orientations crossed with 5 spatial frequencies and 4
phases).

All 32 sessions were surveyed for probe yield, and the six with the best simultaneous
LGd / VISp / higher-area coverage were analysed: sessions 715093703, 750749662, 755434585,
757970808, 763673393 and 799864342. Data are streamed from the DANDI S3 bucket with
`remfile`; nothing is downloaded in full. Because the spike times live in one contiguous
uncompressed HDF5 dataset, the loader looks up its byte offset and issues parallel HTTP
range requests for only the units that pass quality control. Analysis uses Pynapple for
data structures and epoching, and NeMoS for the Poisson GLM.

## What Was Analysed

For every one of the 2516 quality-control-passing units (2083 of which were visually
responsive) we measured the direction tuning curve at the unit's preferred temporal
frequency, then computed the global orientation selectivity index (gOSI, a vector strength
at the second harmonic), the classical two-point OSI, and the corresponding direction
indices. Significance was assessed with a permutation test that shuffles responses within
each temporal-frequency block, so the null hypothesis is specifically "no orientation or
direction tuning" while each unit's rate and temporal-frequency preference are preserved.
Four further analyses check that the result means what it appears to mean: cross-validation
of the preferred angle against the static gratings, a control for the cardinal-orientation
bias and for narrow- versus broad-spiking cell types, a cross-validated Poisson GLM that
puts running speed and temporal frequency in the model before adding direction, and a
Poisson naive-Bayes population decoder with the number of units matched across areas.

## Key Finding

Orientation selectivity appears between thalamus and cortex, and it appears sharply. Median
gOSI rises from 0.091 in LGd (n = 312) to 0.252 in VISp (n = 346), a roughly three-fold
increase (Mann-Whitney one-sided p = 3.8e-43), and the fraction of units with significant
tuning rises from 44.9% to 81.2% (chi-squared p = 6.9e-22). The ordering holds in every one
of the six mice individually. The preference is genuinely for orientation rather than for
something specific to moving stimuli: for tuned VISp units the preferred angle measured
from 2 s drifting gratings and from 250 ms static gratings differ by a median of only 9.7°,
against a chance value of 45°. The tuning is not an artefact of locomotion, since drift
direction still improves held-out prediction after running speed and temporal frequency are
in the GLM.

The population decoding makes the same point in a different currency, and the structure of
the decoder's errors is what identifies the code as an orientation code. Thirty
simultaneously recorded VISp units support 63% correct identification of which of eight
drift directions was shown, against 12.5% chance, compared with 38% for thirty LGd units.
The VISp decoder's mistakes are overwhelmingly 180° confusions: the rate of confusing a
direction with its opposite is 17 times the mean rate of the other six error types, so the
population reliably reports the axis of the grating while frequently losing which way it
moved. Collapsing the eight directions onto four orientations accordingly lifts VISp
accuracy to 90%, while LGd reaches only 52%. One caveat worth stating plainly is that LGd
is not perfectly untuned. A substantial minority of LGd units reach significance, and the
GLM's direction term helps there too; the thalamus in this dataset carries a weak
orientation bias, and the LGd-to-VISp difference is one of degree, though a large one.

## Files

| File | Contents |
| --- | --- |
| `orientation_selectivity_dandi.py` | Consolidated jupytext script, runs end to end |
| `orientation_selectivity_dandi.ipynb` | Same analysis as a notebook |
| `fig01_raw_overview.png` | Stimulus blocks, raw spike raster, population rate, locomotion |
| `fig02a_example_VISp.png` | Example orientation-selective V1 unit (gOSI 0.68) |
| `fig02b_example_LGd.png` | Example thalamic relay unit, strongly driven but untuned (gOSI 0.04) |
| `fig03a/b_population_tuning_*.png` | Population tuning curves, drifting and static gratings |
| `fig04_selectivity_by_area.png` | gOSI distributions and significant fractions by area |
| `fig05_drifting_vs_static.png` | Preferred orientation cross-validated across two stimuli |
| `fig06_controls.png` | Cardinal bias, cell types, and the rate-artefact control |
| `fig07_glm.png` | Cross-validated Poisson GLM against locomotion and temporal frequency |
| `fig08_decoding.png` | Population decoding and the 180° confusion structure |
| `unit_metrics.csv` | Per-unit metrics for all 2516 units |
| `summary_by_area.csv` | Per-area summary statistics |

## Reproducing

```bash
python orientation_selectivity_dandi.py
```

Requires `pynapple`, `pynwb`, `remfile`, `h5py`, `dandi`, `nemos`, `jax`, `scikit-learn`,
`scipy`, `pandas`, `matplotlib` and `tqdm`. The first run streams roughly 2 GB from S3 and
caches it under `~/.cache/dandi_000021`, after which re-runs take about fifteen minutes,
most of which is the per-unit GLM cross-validation. Set `OV_CACHE` to move the cache.
