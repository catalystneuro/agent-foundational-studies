# Auditory frequency tuning in mouse auditory cortex

This run demonstrates **frequency tuning** — the foundational property of
auditory cortex neurons that respond preferentially to a particular sound
frequency — using publicly available Neuropixels data from the DANDI Archive.

## Dataset

**DANDI:000986** — *Auditory cortex Neuropixels recordings and pupil diameter
traces from mice during passive exposure to pure tones*
(McCormick lab, University of Oregon; related preprint:
[doi:10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)).

A single session was used for prototyping:
`sub-LA11/sub-LA11_ses-1_behavior.nwb` — 235 sorted single units in
auditory cortex, 7,447 trials of passive 25 ms pure tones at five
frequencies (2, 4, 8, 16, 32 kHz), all at 60 dB SPL, ~1,500 reps per
frequency. Streamed directly from S3 with `remfile`'s on-disk cache.

## Analysis

The pipeline (`auditory_frequency_tuning.py`, also exported as
`auditory_frequency_tuning.ipynb`):

1. Streams the NWB file with `remfile`/`pynwb` and wraps it as a Pynapple
   `NWBFile`.
2. Sanity-checks stimulus locking with a per-trial raster
   (`fig01_raw_raster_with_tones.png`).
3. Builds tone-onset PSTHs per frequency for every unit and shows the
   population mean (`fig02_population_psth_by_frequency.png`).
4. Computes the per-trial evoked rate in a 5–60 ms post-onset window and
   tests frequency dependence with a one-way ANOVA across the five tone
   frequencies; "tone-driven" units are those with ANOVA p < 0.01 *and*
   mean evoked rate above their pre-stimulus baseline.
5. Plots tuning curves for the 12 most-tuned units
   (`fig03_example_tuning_curves.png`).
6. Summarises the population: ANOVA p-value distribution, best-frequency
   distribution, and tuning strength (`fig04_population_summary.png`).
7. Shows tonotopic organisation of single-unit responses by sorting
   driven units by best frequency and plotting per-frequency PSTH heatmaps
   (`fig05_tonotopic_population_psth.png`).
8. Aligns each driven unit's tuning curve to its own best frequency in
   octaves and averages across the population, recovering the canonical
   V-shaped tuning (`fig06_bf_aligned_mean_tuning.png`).

A per-unit summary CSV (`tuning_summary.csv`) is also written.

## Key finding

In one session of 235 units, **119 (51%)** were significantly modulated by
tone frequency (ANOVA p < 0.01, evoked > baseline). Best frequencies span
the full 2–32 kHz stimulus set, with a peak at 8 kHz (BF counts:
2 kHz: 21, 4 kHz: 33, 8 kHz: 50, 16 kHz: 11, 32 kHz: 4) — consistent with
the mouse audiogram having peak sensitivity in the few-kHz range. The
BF-aligned mean tuning curve forms a clean V centred on each unit's
preferred frequency, dropping to ~50% of the BF response within one
octave on either side. This is the cellular substrate of tonotopy in
auditory cortex.

## Files

| file | what |
| --- | --- |
| `auditory_frequency_tuning.py` | Final jupytext analysis script (runs end-to-end) |
| `auditory_frequency_tuning.ipynb` | Same, as a Jupyter notebook |
| `fig01_raw_raster_with_tones.png` | Raster of first 40 units, first 150 trials |
| `fig02_population_psth_by_frequency.png` | Population mean PSTH per stimulus frequency |
| `fig03_example_tuning_curves.png` | 12 most-tuned units, tuning curves with SEM |
| `fig04_population_summary.png` | ANOVA p, BF distribution, FSI distribution |
| `fig05_tonotopic_population_psth.png` | Per-frequency PSTH heatmaps, units sorted by BF |
| `fig06_bf_aligned_mean_tuning.png` | BF-aligned mean tuning curve across driven units |
| `tuning_summary.csv` | Per-unit baseline, ANOVA p, BF, FSI, per-frequency rates |

## Reproducing

```bash
pip install pynapple lindi remfile pynwb h5py tqdm matplotlib jupytext nbconvert scipy pandas
python auditory_frequency_tuning.py
```

The script downloads ~280 MB of NWB data on first run (cached at
`/tmp/remfile_cache_auditory`); the spike-counting pass over all 235
units × 7,447 trials takes ~8 minutes.
