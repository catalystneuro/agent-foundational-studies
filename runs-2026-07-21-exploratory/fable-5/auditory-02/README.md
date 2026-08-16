# Auditory frequency tuning in DANDI Archive recordings

This analysis demonstrates frequency tuning at two stages of the auditory pathway using
two dandisets, streamed from the DANDI S3 bucket with `remfile` (nothing is downloaded in
full). Analysis uses `pynapple` for time-series handling and `nemos` for the encoding
model.

* **DANDI:001262** - single auditory-nerve fibre recordings in Mongolian gerbils
  (Heeringa et al., *Scientific Data* 2024, doi:10.1038/s41597-024-03259-3). Each NWB file
  holds one fibre, with one row of the `units` table per stimulus sweep. A minority of
  fibres were probed with a full frequency x level grid of 50 ms tone bursts; most were
  probed with a fixed-level frequency sweep.
* **DANDI:000986** - Neuropixels recordings from mouse auditory cortex during passive
  presentation of 25 ms pure tones at 2, 4, 8, 16 and 32 kHz, 60 dB SPL, roughly 1500
  repeats of each frequency per session (15 sessions, 5 mice; doi:10.1101/2024.04.04.588209).

## Key finding

Frequency tuning is present and statistically clear at both stages, and the two stages
differ in the stimulus sets used to measure it. Single auditory-nerve fibres have narrow, V-shaped frequency
threshold curves: the tip of each curve defines a characteristic frequency, thresholds
rise steeply on both flanks, and the sharpness of tuning (Q10 = CF / bandwidth 10 dB above
the tip) increases with characteristic frequency, which is the signature of cochlear
filtering. Across 59 fibres with a measurable threshold curve, Q10 ranges from
1.4 to 7.6 (median 3.1, n = 21) and rises with CF
(log-log slope 0.38, r = 0.70, p = 0.0004). In the 395 fibres probed with a
fixed-level frequency sweep, 375 have a significantly
frequency-dependent response (p < 0.01) and the median half-maximum tuning width is
0.58 octaves (n = 205 measurable). Best frequencies span
0.3 to 16 kHz.

In auditory cortex, 1230 of 1426 units recorded
across 15 sessions have a tone-evoked firing rate that depends significantly on frequency
(permutation test, p < 0.01), their best frequencies span the full 2-32 kHz range tested,
and sorting the normalised tuning curves by best frequency produces the diagonal band that
is the population signature of frequency tuning. A cross-validated Bayesian decoder
recovers which of the five frequencies was played from 50 ms of population activity with
0.81 accuracy (chance 0.20; the same decoder trained on shuffled labels gives
0.20). A Poisson GLM with frequency-specific tone kernels and a spike-history
filter reproduces the same frequency preferences while accounting for the unit's own
spiking dynamics.

## Caveats

* Cortical best frequency here is an **iso-level** preference measured at a single level
  (60 dB SPL), so it mixes the unit's tuning with the mouse's audiogram. It is not the
  same quantity as the characteristic frequency of an auditory-nerve fibre, which is
  defined at threshold.
* Several cortical units respond strongly at 2 kHz even though mice are relatively
  insensitive there. Short (25 ms) low-frequency tones contain broadband onset energy, so
  part of the 2 kHz response is likely driven by the onset transient rather than by
  2 kHz per se.
* **We do not claim that cortical tuning is broader than peripheral tuning**, even though
  that is the textbook expectation. Both width estimates are biased downwards by their own
  measurement limits: the cortical tuning curve is sampled at only five octave-spaced
  frequencies, so widths below one octave cannot be resolved, and the nerve width is only
  measurable for fibres whose curve fell to half maximum inside the frequency range the
  experimenter chose, which drops the broadly tuned fibres from the sample. The two
  distributions in `fig11` are shown as descriptions of each dataset, not as a contrast.
  Testing that claim would require the same stimulus set at both stages.
* Bandwidth and Q10 estimates come from median-filtered tuning and threshold curves. Without
  that smoothing, a single noisy point next to the peak can truncate the width to an
  arbitrarily small value. Estimates for fibres with weak or bimodal responses remain
  unreliable, and fibres whose curve does not fall to half maximum inside the tested
  frequency range are reported as not measurable rather than being assigned a width.

## A dandiset we rejected

DANDI:001419 (mouse A1/A2 linear probes, 10 half-octave frequencies from 4 to 80 kHz, with
laminar electrode positions) was the first choice and would have been a better stimulus
set. Its `units/spike_times` are integer-valued, i.e. rounded to whole seconds, which makes
alignment to a 300 ms tone impossible. The same holds for the companion dandisets 001420
and 001421. The check is reproduced in the appendix of the notebook.

## Files

| File | Contents |
| --- | --- |
| `auditory_frequency_tuning.py` / `.ipynb` | consolidated end-to-end analysis (jupytext percent format) |
| `dandi_io.py` | streaming access and NWB parsing for both dandisets |
| `anf_analysis.py` | auditory-nerve response areas, threshold curves, CF, Q10, iso-level tuning |
| `cortex_analysis.py` | cortical PSTHs, tuning curves, significance tests, Bayesian decoding |
| `glm_analysis.py` | NeMoS Poisson GLM encoding model |
| `make_figures.py` | all figures |
| `run_anf_batch.py`, `run_cortex_batch.py` | batch runners that cache results to `results_*.pkl` |
| `figures/fig01`-`fig11` | figures, described below |

The working directory also contains files from an earlier, unrelated run that were already
present when this analysis started (`01_explore_session.py` through `05_glm_nemos.py`,
`aft_*.py`, `aud_io.py`, and the `fig0*_*.png` files in the top-level directory rather than
in `figures/`). They are not part of this analysis and were left untouched.

### Figures

1. `fig01_raw_data_cortex.png` - raw spike raster, population rate, pupil diameter and
   running speed with tones marked, as a check that all streams share one clock.
2. `fig02_raw_data_anf.png` - one fibre's sweeps sorted by level and by frequency, with the
   PSTH used to locate the 50 ms tone burst inside the 200 ms sweep.
3. `fig03_anf_response_areas.png` - frequency response areas and threshold curves for three
   fibres spanning the CF range.
4. `fig04_anf_tuning_curves.png` - fixed-level frequency tuning curves for eight fibres.
5. `fig05_anf_population.png` - threshold curves, thresholds at CF, Q10 versus CF, best
   frequency distribution and tuning widths across fibres.
6. `fig06_cortex_example_unit.png` - raster and PSTH by frequency for one cortical unit.
7. `fig07_cortex_tuning_curves.png` - eight single-unit cortical tuning curves.
8. `fig08_cortex_population.png` - normalised tuning sorted by best frequency, best
   frequency distribution, selectivity, per-session summary.
9. `fig09_cortex_decoding.png` - confusion matrix and per-session decoding accuracy against
   a shuffled-label control.
10. `fig10_glm_kernels.png` - GLM tone kernels, spike-history filter and model-based tuning.
11. `fig11_tuning_width.png` - tuning width in octaves in each dataset, with the
    resolution limits marked (the two distributions are not directly comparable).

## Reproducing

```bash
python run_anf_batch.py 400 results_anf.pkl   # auditory-nerve fibres
python run_cortex_batch.py 15                 # cortical sessions
python make_figures.py                        # all figures
```

The consolidated notebook runs the same pipeline end to end and reuses the cached
`results_*.pkl` files if they are present. First-time streaming of the cortical sessions
takes roughly 25 minutes; afterwards the local `remfile` cache makes re-runs much faster.
