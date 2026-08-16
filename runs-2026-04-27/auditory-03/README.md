# Auditory Frequency Tuning in Mouse Auditory Cortex (DANDI:000986)

This analysis demonstrates **auditory frequency tuning** — the canonical
finding that single neurons in primary auditory cortex respond preferentially
to specific tone frequencies — using publicly available Neuropixels recordings
from the DANDI Archive.

## Dataset

[**DANDI:000986**](https://dandiarchive.org/dandiset/000986) — "Auditory cortex
Neuropixels recordings and pupil diameter traces from mice during passive
exposure to pure tones" (Jo et al., 2024; McCormick lab, University of Oregon;
preprint: doi:10.1101/2024.04.04.588209).

- 5 head-fixed mice, multiple sessions each (we use 5 sessions, one per mouse).
- Neuropixels 1.0 recordings of auditory cortex (132–192 sorted units per session).
- Stimuli: 25 ms pure tones at **2, 4, 8, 16, 32 kHz** (one octave apart),
  60 dB SPL, ~1500 trials per frequency per session, randomly interleaved with
  silent inter-trial intervals (~0.8 s).
- NWB files were streamed directly from S3 with `remfile` + `DiskCache`; no full
  downloads.

## What was analyzed

For each session we:

1. Aligned spike trains to tone onset and computed peri-stimulus time
   histograms (PSTHs) per frequency.
2. Counted spikes in a 5–105 ms post-onset response window and a 100 ms
   pre-onset baseline window per trial, per unit.
3. Identified **tone-responsive units** with a paired t-test (evoked > baseline,
   p < 0.001), pooled across frequencies.
4. For each responsive unit, computed a **frequency tuning curve** (mean evoked
   rate per frequency) and assigned a **best frequency (BF)** as argmax of the
   driven (evoked − baseline) tuning curve.
5. Quantified tuning selectivity with **lifetime sparsity**.
6. Pooled units across all 5 sessions (576 units total) and compared best-
   frequency distributions across mice.

## Key findings

- **Robust onset responses.** Every tested frequency drove a clean ~10–40 ms
  onset transient in the population PSTH that returned to baseline within
  ~150 ms (`figures/02_population_psth_single_session.png`,
  `figures/05_population_psth_pooled.png`).
- **A large fraction of units are tone-responsive.** 31–49% of units per
  session passed the responsiveness criterion (253/576 pooled, 44%).
- **Single-unit frequency tuning is clear and varied.** Individual units have
  single-peaked tuning curves with preferred frequencies spanning the entire
  2–32 kHz range (`figures/03_unit_tuning_curves_single_session.png`,
  `figures/09_example_unit_rasters_and_tuning.png` — example unit with strong
  selectivity for 8 kHz where the raster goes from sparse at 2 kHz to a dense
  evoked carpet at 8 kHz).
- **Population organization.** Sorting responsive units by their best
  frequency produces the canonical diagonal in the tuning heatmap — exactly
  what is expected from a frequency-tuned, tonotopically-organized auditory
  cortex (`figures/06_tuning_heatmap_pooled.png`).
- **Best frequencies span the test range.** All five tested frequencies are
  represented as best frequencies in the population, with concentrations at
  8 and 16 kHz (consistent with mouse A1's emphasis on higher frequencies)
  and substantial representation at 2 kHz (`figures/07_best_frequency_distribution.png`).
- **Tuning is selective, not flat.** Median lifetime sparsity across responsive
  units is **0.57** (sparsity = 1 means tuned to a single frequency, 0 means
  uniform), with a median peak-to-trough driven-rate range of ~8 Hz
  (`figures/08_sparsity_distribution.png`).

## Files

- `auditory_frequency_tuning.py` — final consolidated jupytext script.
- `auditory_frequency_tuning.ipynb` — notebook version (converted via jupytext).
- `01_explore_session.py` — single-session prototyping script (early validation).
- `figures/*.png` — all generated figures.

## Reproducing

```bash
pip install pynapple lindi remfile pynwb h5py tqdm matplotlib scipy pandas dandi jupytext
python auditory_frequency_tuning.py
```

NWB files are streamed (~280 MB each) and cached locally in `/tmp/remfile_cache`.
First run downloads on demand; subsequent runs are fast.
