# Spectrotemporal receptive fields in the auditory system, from DANDI

This analysis measures spectrotemporal receptive fields (STRFs) at two stages of
the auditory pathway using two publicly archived electrophysiology datasets on
the [DANDI Archive](https://dandiarchive.org). Both are streamed from S3 with
`remfile` and a local disk cache; nothing is downloaded in full, and no
simulated data is used anywhere.

* **[Dandiset 000986](https://dandiarchive.org/dandiset/000986)**, "Auditory
  cortex Neuropixels recordings and pupil diameter traces from mice during
  passive exposure to pure tones" (Lohse et al., preprint
  [10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)).
  Fifteen sessions from five awake head-fixed mice, 25 ms pure tones at 2, 4, 8,
  16 and 32 kHz and 60 dB SPL delivered about once every 0.8 s, with a
  simultaneous pupil trace. All 15 sessions and 1419 units above 0.5 Hz were
  analysed.
* **[Dandiset 001262](https://dandiarchive.org/dandiset/001262)**, "Single-unit
  auditory nerve fibre responses of young-adult and aging gerbils" (Huet et al.,
  [10.1038/s41597-024-03259-3](https://doi.org/10.1038/s41597-024-03259-3)).
  Each file is one Mongolian gerbil auditory-nerve fibre recorded with a glass
  electrode. The best-frequency protocol sweeps tone frequency on a grid of
  11-81 values centred near that fibre's characteristic frequency, five repeats
  each. A random sample of 260 of the 1160 files was processed; 255 had a usable
  frequency sweep.

## What was analysed

Both datasets present pure tones that are sparse in time and never overlap
within the receptive-field window. For that stimulus ensemble the
spike-triggered average of the stimulus spectrogram reduces exactly to the
frequency-conditioned PSTH minus the mean rate, so the STRF was estimated by
reverse correlation on a frequency-by-lag grid (5 ms bins out to 250 ms for
cortex; 1 ms bins out to 80 ms for the nerve). Each cortical STRF was z-scored
against a null built from 50 circular shifts of the same spike train, which
preserves the unit's rate and autocorrelation. The same cortical kernels were
then refitted as regularised Poisson GLMs in [NeMoS](https://nemos.readthedocs.io),
with a raised-cosine temporal basis, an explicit spike-history filter and
scoring on held-out data. Data access and all spike-train handling use
[Pynapple](https://pynapple.org).

## Key findings

Tone-evoked STRFs are present in the large majority of cortical units: 1207 of
1419 units (85%) had at least one STRF pixel reaching |z| >= 4 against the
shuffle null. The population STRF has the canonical cortical form, a transient
excitatory band at the best frequency with a median onset latency of 20 ms and a
median peak at 35 ms, decaying over a tail that outlasts the 25 ms tone by more
than 100 ms. Receptive fields are not purely excitatory: 386 units (27%) had
both significant excitation and significant suppression, and 83 had only
significant suppression, including units that combine 32 kHz excitation with
8-16 kHz suppression. Refitting the kernels as Poisson GLMs recovered the same
structure and improved the held-out log-likelihood for 22 of 24 units over a
spike-history-only model (median increase in McFadden pseudo-R² of 0.023 on
tone-window bins), so the STRF is a predictive model of the spike train and not
only a descriptive average.

At the auditory nerve the fine frequency grid resolves what the cortical dataset
cannot. The best frequency read off each fibre's receptive field agrees with the
value published with the dataset to a median of 0.009 octaves across 255 fibres
(r = 0.995 on log frequency), which validates the whole extraction path from
streamed NWB to receptive field. The population receptive field is a narrow
ridge, 0.35 octaves wide at half maximum, with a strong onset transient that
adapts to about half its peak within 40 ms and a firing rate that drops below
spontaneous after tone offset. Comparing the two stages qualitatively (species,
tone duration and level all differ), the nerve fibre tracks the tone envelope
and stops at offset, whereas the cortical response is transient with a long
tail. Finally, splitting the cortical tones by pupil diameter shows that arousal
leaves the STRF peak unchanged (median gain change -1%, p = 0.26) but increases
the late 60-150 ms component by about 20% (p = 7e-7, Wilcoxon signed-rank,
n = 696 units).

## Caveats

Dandiset 000986 samples only five frequencies, one octave apart, at a single
sound level, so cortical spectral tuning cannot be resolved more finely than an
octave and no level dependence can be measured. Dandiset 001262 gives five
repeats per frequency, so single-fibre maps are noisy and were smoothed for
display; each fibre also covers only its own 1-2 octave neighbourhood. Pupil
diameter covaries with locomotion in these recordings, so the arousal result is
an association with state rather than an isolated effect of arousal. The
cortical units carry no depth, layer or quality metadata in the archived files,
so no laminar analysis was attempted.

## Files

| file | contents |
|---|---|
| `strf_auditory_dandi.py` | consolidated jupytext (percent format) analysis, runs end to end |
| `strf_auditory_dandi.ipynb` | the same notebook, converted with jupytext |
| `strf_lib.py` | shared streaming and STRF helpers used by the modular scripts |
| `01_prototype_ac.py`, `02_prototype_anf.py` | single-session prototypes used during development |
| `03_ac_population.py` | cortical STRFs and shuffle nulls across all 15 sessions |
| `04_anf_population.py` | auditory-nerve receptive fields across 255 fibres |
| `05_glm_strf.py` | NeMoS Poisson-GLM STRFs with held-out scoring |
| `06_figures.py` | figure generation |
| `figures/*.png` | all figures |
| `ac_population.npz`, `anf_population.npz`, `glm_strf.npz` | cached intermediate results |

## Figures

| figure | contents |
|---|---|
| `fig1_ac_raw_data.png` | tone sequence, spike rasters and pupil trace for one cortical session |
| `fig1b_ac_raster_psth.png` | frequency-conditioned raster and PSTH for one unit |
| `fig2_ac_example_strfs.png` | eight single-unit cortical STRFs with significance outlines |
| `fig3_ac_population.png` | responsiveness, best frequency, latency, BF-aligned mean STRF, tuning and STRF sign across 15 sessions |
| `fig4_ac_glm_strf.png` | reverse-correlation versus GLM STRFs and held-out prediction |
| `fig5_anf_example_strfs.png` | four single-fibre auditory-nerve receptive fields |
| `fig6_anf_population.png` | BF validation, CF distribution, population receptive field, tuning width and adaptation |
| `fig7_nerve_vs_cortex.png` | receptive-field shape at the two stages |
| `fig8_ac_arousal.png` | STRF gain as a function of pupil-indexed arousal |

## Running

```bash
python strf_auditory_dandi.py            # writes figures/ and the .npz caches
jupytext --to notebook strf_auditory_dandi.py
```

A cold run takes roughly 25 minutes, most of it opening the 260 single-fibre
NWB files. Delete the `.npz` files to force recomputation. Requires `pynapple`,
`nemos`, `pynwb`, `remfile`, `h5py`, `numpy`, `scipy`, `matplotlib`, `tqdm` and
`jupytext`. Figures are written with the Agg backend and never displayed.
