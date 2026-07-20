# Auditory frequency tuning in mouse auditory cortex

This run demonstrates the canonical phenomenon of **frequency tuning** in primary
auditory cortex using a real Neuropixels dataset from the DANDI Archive.

## Dataset

- **DANDI:** [000986](https://dandiarchive.org/dandiset/000986) —
  *"Auditory cortex Neuropixels recordings and pupil diameter traces from mice
  during passive exposure to pure tones"* (Jo et al., McCormick lab,
  University of Oregon).
- **Companion preprint:** [doi:10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)
- **Session analysed:** `sub-LA11/sub-LA11_ses-1_behavior.nwb` (235 sorted units,
  7447 tone trials, 5 frequencies × ~1500 trials each).
- **Stimulus:** 25 ms pure-tone pips at 60 dB SPL, drawn pseudo-randomly from
  {2, 4, 8, 16, 32} kHz.
- **Access mode:** streamed from S3 with `remfile` + on-disk byte-range cache —
  no whole-file download.

## Pipeline

The single self-contained script `auditory_frequency_tuning.py` (also provided as
`auditory_frequency_tuning.ipynb`) does end-to-end:

1. Streams the NWB file with `remfile`/`pynwb`, wraps in `pynapple.NWBFile`.
2. Builds per-frequency tone-onset event sets and a global onset index.
3. Computes per-unit peri-stimulus time histograms (5 ms bins, –100 → +200 ms
   around onset) and a population PSTH.
4. Defines an evoked window (10–60 ms post-onset) and a baseline window
   (–100 → 0 ms), and computes per-unit, per-frequency tuning curves with
   per-unit z-scoring against pre-tone baseline.
5. Calls a unit "sound-responsive" if `peak z-score > 2` and identifies its
   best frequency (BF).
6. Generates raster, PSTH, single-unit tuning, population tuning heatmap, and
   BF-distribution figures.

## Key result

- **46 / 235 units (≈ 20 %)** cross the responsiveness threshold on this single
  session. (Most non-responsive units sit outside auditory cortex on the probe
  shank or have low firing rates.)
- The **population PSTH** (`fig02_population_psth.png`) shows a sharp,
  tone-locked excitatory transient peaking around 20–25 ms after tone onset,
  with frequency-dependent magnitude.
- **Single-unit tuning curves** (`fig03_example_tuning_curves.png`) show clear
  bandpass-style preferences at single octaves.
- The **population tuning heatmap** (`fig04_population_tuning_heatmap.png`),
  sorted by best frequency, reveals the textbook diagonal band — different
  units prefer different frequencies, exactly as expected from a tonotopically
  organised auditory cortex.
- Best frequencies span the full tested range (2–32 kHz), with most units
  preferring 2–8 kHz under these passive-listening conditions
  (`fig05_best_frequency_distribution.png`).
- A representative tuned unit (unit 64, BF 4 kHz) is shown trial-by-trial in
  `fig06_example_unit_raster.png` — spikes cluster tightly after tone onset
  on 4 kHz trials and essentially vanish at 32 kHz.

Together these results reproduce the canonical signature of auditory cortical
frequency tuning from a real DANDI dataset using a fully streaming,
reproducible pipeline.

## Files

| File | Purpose |
|------|---------|
| `auditory_frequency_tuning.py` | Jupytext source (percent-format, runs end-to-end) |
| `auditory_frequency_tuning.ipynb` | Notebook converted from the jupytext source |
| `fig01_raw_raster.png` | Sanity-check raster of sampled units |
| `fig02_population_psth.png` | Population PSTH per stimulus frequency |
| `fig03_example_tuning_curves.png` | Tuning curves of the 8 most strongly tuned units |
| `fig04_population_tuning_heatmap.png` | Z-scored tuning matrix, sorted by best frequency |
| `fig05_best_frequency_distribution.png` | Histogram of best frequencies across responsive units |
| `fig06_example_unit_raster.png` | Trial-level raster of an example tuned unit |
| `tuning_summary.csv` | Per-unit summary (baseline, peak z, BF, per-frequency rates) |

## Reproducing

```bash
pip install pynapple lindi remfile pynwb h5py tqdm matplotlib jupytext
python auditory_frequency_tuning.py
```

NWB file bytes are cached under `/tmp/remfile_cache` after the first run.
