# Auditory Frequency Tuning in Mouse Auditory Cortex

**Dataset:** [DANDI 000986](https://dandiarchive.org/dandiset/000986) (version 0.251031.1939) —
"Auditory cortex Neuropixels recordings and pupil diameter traces from mice during passive
exposure to pure tones" (Jaramillo lab). 15 Neuropixels sessions from 5 mice; in each session,
25 ms pure tones at 60 dB SPL with frequencies of 2, 4, 8, 16, and 32 kHz were presented every
~0.8 s in random order (~1,480 trials per frequency per session) while the mouse was passive.

**Analysis:** NWB files were streamed from the archive with `remfile` (no full downloads) and
accessed with Pynapple. For every unit, spikes were counted in a 5–55 ms window after each tone
onset and baseline-subtracted using a 50 ms pre-tone window, yielding an evoked firing rate per
tone frequency. Tuning significance was assessed with a Kruskal–Wallis test of response-window
spike counts across frequencies (p < 0.01); each tuned unit's best frequency (BF) and a
selectivity index (1 − mean/max response) were extracted. As a regression-based complement, a
Poisson GLM with a 5-knot B-spline basis over log2 frequency (NeMoS) was fit to each tuned
unit's per-trial spike counts, giving smooth tuning curves and continuous BF estimates.

**Key finding:** Frequency tuning is widespread and sharp in these recordings. 85% of units
with a mean firing rate ≥ 0.5 Hz (1221/1431 units pooled over all 15 sessions; per-session
range 58–100%) were significantly tuned. Best frequencies tile the full 2–32 kHz range, with
the largest share of units preferring 8–16 kHz (median selectivity index 0.41). Single-trial
perievent rasters show that tuning reflects both excitation at the best frequency and
suppression at flanking frequencies. The GLM-based smooth tuning curves closely match the
empirical windowed-count estimates: best frequencies agree within one octave for 96% of tuned
units.

## Files

- `auditory_frequency_tuning.py` — consolidated jupytext analysis script (runs end-to-end;
  streams data from DANDI, caches per-session results in `cache/`)
- `auditory_frequency_tuning.ipynb` — the same analysis as a Jupyter notebook
- `figures/fig1_raw_data.png` — raw spike raster with tone onset markers
- `figures/fig2_perievent_raster_psth.png` — perievent raster and PSTH by frequency for an
  example tuned unit
- `figures/fig3_example_tuning_curves.png` — tuning curves of the 8 best-tuned units of the
  example session
- `figures/fig4_population_summary.png` — tuned fraction per session, p-value distribution,
  BF distribution, selectivity distribution
- `figures/fig5_tuning_heatmap.png` — normalized tuning curves of all 1221 tuned units sorted
  by best frequency
- `figures/fig6_nemos_glm.png` — Poisson-GLM smooth tuning curves and GLM-vs-empirical BF
  comparison
- `population_summary.json` — machine-readable summary statistics
- `cache/` — per-session spike-count matrices (regenerated if deleted)

## Running

```bash
python auditory_frequency_tuning.py   # ~25 min on first run (streaming + GLM fits)
```

Requires: pynapple, pynwb, h5py, remfile, nemos, scipy, matplotlib, tqdm.
