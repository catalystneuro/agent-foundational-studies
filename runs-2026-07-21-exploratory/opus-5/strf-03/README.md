# Spectrotemporal receptive fields in mouse auditory cortex (DANDI:000986)

This analysis demonstrates spectrotemporal receptive fields (STRFs) in the auditory
system using [DANDI:000986](https://dandiarchive.org/dandiset/000986), "Auditory cortex
Neuropixels recordings and pupil diameter traces from mice during passive exposure to
pure tones" (Jo & McCormick, University of Oregon; related preprint
[doi:10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)). All 15
sessions from 5 mice were analysed, 1564 sorted units in total. Each session presents a
randomised sequence of 25 ms pure tones at 2, 4, 8, 16 and 32 kHz, 60 dB SPL, one tone
every 0.805 s, in ~25 min blocks that alternate with silence, giving roughly 1500
presentations per frequency per session. Data are streamed from the DANDI S3 bucket
with `remfile` plus a local disk cache; no file is downloaded in full. Spike trains,
tone onsets, pupil diameter and running speed are handled as pynapple objects
throughout, and the encoding models are fitted with NeMoS.

The STRF was estimated three ways and the estimates were checked against each other:
frequency-resolved peri-tone histograms (the receptive field directly in Hz), the
spike-triggered average of the binary stimulus spectrogram (classical reverse
correlation, computed in closed form and checked against both a literal discrete STA on
the 5 ms grid, r = 0.94, and pynapple's `compute_event_triggered_average`, r = 0.99
against the same reference), and the stimulus filter of a regularised Poisson GLM with
a nine-function raised-cosine lag
basis spanning 300 ms per frequency channel, trained on three tone blocks and tested on
a fourth. Tone responsiveness was tested by circularly shifting tone onsets within each
block (500 shuffles), which preserves both the spike trains and the tone sequence but
destroys the locking between them.

## Key finding

Auditory-cortex units in this dataset have clear, reproducible STRFs: 1072 of 1564
units (69%; 41-83% per session) are significantly locked to the tones, responses begin
a median of 18 ms after tone onset, and most units are driven by a narrow part of the
spectrum (median 1 of the 5 octave-spaced channels above 25% of peak). Averaging every
normalised receptive field after aligning it on its own best channel gives the canonical
shape: a short-latency excitatory peak at the best frequency that falls off within about
an octave on either side, followed by a longer, weaker tail and by suppression below
baseline at non-preferred frequencies. Of the responsive units, 878 are excitatory at
their best channel (median evoked increment 4.4 Hz, or 1.8 times the spontaneous rate),
and 194 are net-suppressed at every frequency.

The three estimators agree, which is the main internal check. The measured tone response
and the GLM filter pushed through the same 25 ms tone agree at r = 0.92 in log-gain
units across 24 fitted units, and the GLM predicts held-out spike trains well above the
mean-rate baseline (Poisson pseudo-R² = 0.053 ± 0.075 for the stimulus-only model,
rising to 0.090 ± 0.070 when a spike-history term is added), so these kernels are real
structure rather than noise fitted to the training data. A secondary analysis splitting
trials by pre-stimulus pupil diameter within each block found no significant change in
evoked gain at the best frequency (median difference +0.01 Hz, Wilcoxon p = 0.15,
n = 1072), so we report the arousal effect as null under this particular split.

Two limits of the stimulus are worth stating. It contains only five frequencies one
octave apart at a single 60 dB level, so the spectral axis of these STRFs is coarse and
nothing can be said about level dependence, bandwidth finer than an octave, or
combination sensitivity; a dense dynamic stimulus (a dynamic random chord or natural
sounds) would be needed for that. And the tone sequence is periodic, so the
reverse-correlation estimate is unbiased only for lags well inside the 0.805 s
inter-onset interval. The GLM does not rely on that assumption and returns the same
receptive field, which is why both are reported.

## Files

| File | Contents |
| --- | --- |
| `strf_auditory_cortex.py` | Consolidated jupytext script (percent format), runs end to end |
| `strf_auditory_cortex.ipynb` | The same analysis as an executed notebook |
| `strf_lib.py` | Streaming loader and the STRF / statistics routines |
| `01_explore_session.py` | Data loading and per-stream validation plots |
| `02_strf_single_session.py` | Reverse-correlation STRFs, implementation checks, shuffle test |
| `03_glm_nemos.py` | Poisson GLM fits (NeMoS/JAX, float64) and model comparison |
| `04_population_multisession.py` | All 15 sessions, population summary, pupil split |
| `proto_metrics.csv`, `population_metrics.csv` | Per-unit metrics (BF, latency, gain, p-value) |
| `proto_strf.npz`, `glm_results.npz`, `population.npz` | Cached intermediate results |

Figures:

| Figure | Contents |
| --- | --- |
| `fig01_raw_data.png` | 8 s of raw data: tones, spike raster, pupil, running speed |
| `fig02_session_structure.png` | Tone blocks vs silence, population rate, behaviour |
| `fig03_stimulus_matrix.png` | The binary stimulus spectrogram S(f, t) |
| `fig04_sta_validation.png` | Closed-form STA vs discrete STA vs pynapple's ETA |
| `fig05_example_unit_strf.png` | One unit: raster, STRF, STA, temporal profile, tuning |
| `fig06_strf_gallery.png` | 16 receptive fields ordered by best frequency |
| `fig07_responsiveness.png` | Shuffle-test p-values, null distribution, response strength |
| `fig08_glm_vs_revcorr.png` | Reverse-correlation STRFs next to GLM filters |
| `fig09_glm_quality.png` | Filter agreement, ridge sweep, model comparison, prediction |
| `fig10_population.png` | 1564 units: responsiveness, BF, latency, gain, arousal split |
| `fig11_bf_aligned.png` | BF-aligned mean STRF, population tuning, bandwidth |

## Reproducing

```bash
python 01_explore_session.py          # ~2 min
python 02_strf_single_session.py      # ~5 min
python 03_glm_nemos.py                # ~15 min (JAX, CPU)
python 04_population_multisession.py  # ~25 min (streams all 15 sessions)
jupytext --to notebook strf_auditory_cortex.py
jupyter nbconvert --to notebook --execute --inplace strf_auditory_cortex.ipynb
```

`strf_auditory_cortex.py` reuses the cached `.npz`/`.csv` files when they are present
and recomputes them from the streamed NWB files when they are not, so it runs end to end
either way. Requirements: `pynapple`, `nemos`, `pynwb`, `remfile`, `h5py`, `jax`,
`numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`, `jupytext`.
