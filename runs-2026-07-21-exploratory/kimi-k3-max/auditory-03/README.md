# Auditory Frequency Tuning in Mouse Auditory Cortex (DANDI 000986)

**Dataset:** [DANDI Archive dandiset 000986](https://dandiarchive.org/dandiset/000986),
*Auditory cortex Neuropixels recordings and pupil diameter traces from mice during
passive exposure to pure tones* (Jaramillo lab, published version 0.251031.1939).
Mice passively listen to short pure tones (20–25 ms, 60 dB SPL) presented every
~0.8 s at one of five frequencies (2, 4, 8, 16, 32 kHz) while Neuropixels probes
record in auditory cortex. The dandiset contains 15 sessions from 5 mice
(sub-LA3, LA8, LA9, LA11, LA12); all 15 were analyzed here.

**Analysis.** Data are streamed from the archive with `remfile` + disk caching
(no full downloads) and handled with Pynapple. For every unit with mean firing
rate ≥ 0.5 Hz, spikes are aligned to tone onsets (`nap.compute_perievent`) and
the evoked rate in a 5–55 ms response window is baseline-subtracted (−50–0 ms).
A tuning curve is the net evoked rate at each of the five frequencies;
significance is assessed with a Kruskal–Wallis test across frequencies
(p < 0.01). As a regression-based complement, a Poisson GLM (NeMoS) with a
4-function B-spline basis over log2 frequency is fit per unit to obtain smooth
model-based tuning curves, compared against the empirical curves with
McFadden's pseudo-R². Population metrics per unit include best frequency (BF),
tuning width at half maximum (the five frequencies are one octave apart), a
selectivity index, and response latency at BF (first 1 ms PSTH bin exceeding
20% of peak above baseline).

**Key findings.** Across all 15 sessions, 1426 units passed the rate criterion
and 1100 (77%) were significantly frequency tuned (Kruskal–Wallis across the
five frequencies, p < 0.01; per-session range 48–94%). Best frequencies tile
the full 2–32 kHz range and concentrate at 8–16 kHz (562 of 1100 tuned units),
consistent with the mouse hearing range and the tonotopic organization of
auditory cortex; the sorted population heatmap (fig7) shows the classic
diagonal band of peak responses with flanking suppression. Tuning is sharp:
87% of tuned units exceed half of their peak response at only one or two of
the five one-octave-spaced frequencies. Responses are fast (median latency
11 ms, IQR 3–18 ms). About 12% of tuned units (128/1100) are primarily
*suppressed* by tones, with their strongest modulation below baseline. The
Poisson GLM with a smooth B-spline basis over log2 frequency reproduces the
empirical tuning curves and best frequencies (fig5, fig6), confirming that the
per-frequency PSTH estimates are not a binning artifact.

## Files

- `auditory_frequency_tuning.py` — consolidated jupytext script, runs end-to-end
- `auditory_frequency_tuning.ipynb` — the same analysis as a Jupyter notebook
- `figures/` — all figures (PNG)
- `results/` — per-session CSVs of per-unit tuning metrics + `all_units.csv`
- `session_urls.json` — direct S3 URLs for the 15 sessions
- `explore_session.py`, `prototype.py`, `glm_tuning.py`, `run_all_sessions.py`,
  `population_figures.py` — development scripts (superseded by the consolidated
  notebook; kept for reference)

## Reproducing

```bash
python auditory_frequency_tuning.py          # ~25 min first run (streams ~3.7 GB total)
jupytext --to notebook auditory_frequency_tuning.py
```

The population pass caches per-session CSVs in `results/`, so re-runs skip
sessions that have already been processed. The remfile disk cache lives in
`/tmp/remfile_cache_auditory03`.
