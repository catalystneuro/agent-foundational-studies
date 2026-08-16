# Auditory Frequency Tuning in Mouse Auditory Cortex (DANDI 000986)

This analysis demonstrates frequency tuning of single neurons in mouse auditory
cortex using data from the DANDI Archive. The dataset is dandiset
[000986](https://dandiarchive.org/dandiset/000986), "Auditory cortex Neuropixels
recordings and pupil diameter traces from mice during passive exposure to pure
tones" (Jaramillo lab). Mice passively heard 25 ms pure tones at 60 dB SPL at
five frequencies spaced one octave apart (2, 4, 8, 16, 32 kHz), presented in
random order roughly every 0.8 s, while Neuropixels probes recorded from
auditory cortex. All 15 sessions (5 mice, 1564 sorted units in total) were
analyzed. Data were streamed with LINDI rather than downloaded, using the
neurosift LINDI index for each NWB asset.

## What Was Analyzed

For every unit, spikes were counted in a 5-60 ms response window after each
tone onset and in a -50-0 ms baseline window before it. The tuning curve is
the mean evoked rate per frequency minus the baseline rate. Per unit we
computed the best frequency (BF), Vinje-Gallant sparseness of the tuning
curve, a Kruskal-Wallis test for frequency modulation of response-window
counts, a Wilcoxon signed-rank test for tone responsiveness (response versus
baseline windows), and a Poisson GLM (NeMoS `PopulationGLM`, 4-knot B-spline
basis over log2 frequency) fit to per-trial response counts, yielding a smooth
tuning curve and a per-unit McFadden pseudo-R² against an intercept-only null.
The full Poisson log-likelihood including the -log(y!) term is used for the
pseudo-R², because that term cancels in likelihood-ratio differences but not
in McFadden's ratio.

## Key Findings

Tones evoked a brisk, time-locked response peaking 10-30 ms after onset.
Across the population, 80.5% of units were tone-responsive (Wilcoxon p < 0.05)
and 86.6% were significantly modulated by tone frequency (Kruskal-Wallis
p < 0.05); 74.0% passed both criteria, with the pattern consistent across all
15 sessions (63-100% frequency-modulated per session). Single units show
peaked tuning curves whose best frequencies span the full 2-32 kHz range, and
the population tiles that range when units are sorted by best frequency
(Figure 3). Tuned units have much sharper (sparser) tuning than untuned units,
and the GLM's smooth best-frequency estimates agree with the raw estimates for
the large majority of tuned units (strong diagonal in Figure 5A), with
off-diagonal cases concentrated between 8 and 16 kHz where tuning is broadest.

## Files

- `auditory_frequency_tuning.py`: consolidated jupytext script that runs the
  full pipeline end-to-end (streaming, statistics, GLM, all figures).
- `auditory_frequency_tuning.ipynb`: the same script as a Jupyter notebook.
- `analysis_core.py`, `make_figures.py`: the modular development pipeline;
  `analysis_core.py` writes `results_all_sessions.npz`, which
  `make_figures.py` reads.
- `figures/`: six figures:
  - `fig1_stimulus_and_responses.png`: stimulus design, example-unit raster
    and PSTH, population PSTH for the example session.
  - `fig2_example_tuning_curves.png`: one example unit per best frequency,
    raw means ± SEM with the GLM fit overlaid.
  - `fig3_population_heatmap.png`: all 1564 units, normalized and sorted by
    best frequency.
  - `fig4_population_stats.png`: best-frequency distribution, tuning
    sharpness, tuning-strength measures, per-session consistency.
  - `fig5_glm_summary.png`: raw versus GLM best frequency and GLM
    explanatory power.
  - `fig6_rates_and_psth.png`: baseline versus peak evoked rates and the
    cross-session population PSTH.

## Running

```
python auditory_frequency_tuning.py        # end-to-end, ~5 minutes
jupytext --to ipynb auditory_frequency_tuning.py
```

Requirements: pynapple, lindi, pynwb, h5py, nemos, jax, scipy, matplotlib,
tqdm, jupytext. Network access is required for streaming from the DANDI
Archive via the neurosift LINDI service.
