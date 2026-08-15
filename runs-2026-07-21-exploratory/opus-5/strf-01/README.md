# Spectrotemporal receptive fields in the auditory nerve (DANDI:001262)

## Dataset

[DANDI:001262](https://dandiarchive.org/dandiset/001262), *Single-unit auditory nerve
fibre responses of young-adult and aging gerbils* (Heeringa & Köppl, *Scientific Data*
2024, [doi:10.1038/s41597-024-03259-3](https://doi.org/10.1038/s41597-024-03259-3)).
Each of the 1160 NWB files holds one isolated auditory-nerve fibre of a Mongolian
gerbil, with spike times stored per stimulus repetition and the acoustic waveform of
the stimulus stored alongside them. 143 of those files contain the frozen broadband
noise protocol (60 repeats of a 2.4 s token that holds two 1 s noise bursts, band
limited to roughly 0.6–12 kHz), and this analysis uses the 126 of them that also have
a tabulated pure-tone best frequency inside the noise passband, drawn from 21 gerbils
aged 125 to 1269 days. Files were streamed from the DANDI S3 bucket with `remfile` and
a local disk cache; nothing was downloaded in full.

## What was analyzed

The spectrotemporal receptive field is the linear filter that maps a time-frequency
representation of a sound onto a neuron's firing rate. For each fibre the stored noise
waveform was passed through a 34-band gammatone filterbank (500 Hz to 12 kHz), the
Hilbert envelope of each band was converted to dB and binned at 1.0035 ms (49 stimulus
samples, so that stimulus and spikes share exactly the same time base), and the STRF
was estimated by ridge-regularized reverse correlation of the repeat-averaged firing
rate on that cochleagram over lags of −10 to +25 ms. Spike trains, stimulus and repeat
structure were handled as `pynapple` `Ts`, `TsdFrame` and `IntervalSet` objects; the
same filter was then refitted as a Poisson GLM in `NeMoS`, with the temporal filter
expanded in a raised-cosine basis and an optional spike-history term. Because the noise
token is frozen, the only honest held-out data is a stretch of the token the filter has
not seen, so cross-validation splits contiguous time blocks indexed by position within
the repeating unit, and silence and the onset transient are excluded from both fitting
and scoring (including them roughly doubles the apparent prediction score, which is
reported for comparison). Prediction is judged against the split-half reliability of the
PSTH rather than against 1.

## Key finding

Reverse correlation on broadband noise recovers a clean, narrow, short-latency
receptive field in single auditory-nerve fibres, and the recovered filter agrees with
tuning measured independently from pure tones. Across the 126 fibres, the best
frequency of the STRF matched the tabulated tone best frequency with a median absolute
error of 0.10 octaves (81 % of fibres within a quarter octave; r = 0.88 between the two
in log frequency), which is close to the 0.14-octave resolution of the analysis
filterbank itself. The population-average STRF, aligned to each fibre's own best
frequency, is a single excitatory lobe confined to about a quarter octave, rising within
1 ms of the stimulus, peaking at 1–6 ms, and followed by weaker suppression over the
next 10 ms — the expected signature of a narrowly tuned cochlear filter followed by
synaptic adaptation. The estimates are causal and not artifacts of the fitting: the
largest weight at negative lag is a median 0.25 of the excitatory peak, and circularly
shifting or shuffling the PSTH before the same fit collapses the peak weight by a factor
of five to seven.

The filters also predict responses to stimulus the fit never saw. On held-out segments
of ongoing noise the median correlation between the predicted and measured PSTH was 0.23
against a median reliability ceiling of 0.79 (best fibre 0.65 against a ceiling of 0.89,
and 0.51 against 0.87 for the example fibre analyzed in detail); prediction improves
markedly with sound level, tracking the growth in response reliability. That the model
captures only about 30 % of the achievable correlation is itself informative, and the
last figure shows one reason: for low-best-frequency fibres a large part of the
millisecond-scale response structure is phase locking to the carrier, which a
band-envelope representation discards by construction. Averaging the raw pressure
waveform before each spike instead of its envelope recovers a ringing filter whose
spectrum peaks at 858 Hz in a fibre whose tone BF is 732 Hz, while the same measurement
in a 10.6 kHz fibre is flat.

## Files

| file | contents |
| --- | --- |
| `strf_auditory_nerve.py` / `.ipynb` | consolidated jupytext notebook, runs end to end |
| `anf_lib.py` | data access, cochleagram, STRF estimation and metrics (imported by the notebook) |
| `01_explore_fibre.py` | load one fibre, validate every data stream (figs 1–2) |
| `02_strf_single_fibre.py` | single-fibre STRF, level dependence, shift/shuffle controls (figs 3–5) |
| `04_glm_nemos.py` | Poisson GLM version of the same filter (fig 6) |
| `scan_all_tags.py` | index of which of the 1160 files ran which protocol (`all_tags.json`) |
| `05_population.py`, `05b_replot_population.py` | population fits and figures (figs 7–9), `population_strf.csv` |
| `06_revcor.py` | spike-triggered average of the pressure waveform (fig 10) |
| `make_notebook.py` | assembles the staged scripts into the consolidated notebook |

Figures:

- `fig01_raw_streams.png` — noise waveform, cochleagram, spike raster over 60 repeats, PSTH
- `fig02_stimulus_and_quality.png` — noise spectrum, filterbank drive, rate-vs-level, ISI distribution
- `fig03_single_fibre_strf.png` — STRF, its temporal and spectral profiles, held-out prediction
- `fig04_strf_vs_level.png` — the same fibre's STRF at four sound levels, one fixed penalty
- `fig05_strf_controls.png` — STRF from the measured, shifted and shuffled PSTH
- `fig06_glm_strf.png` — ridge vs NeMoS GLM filter, spike-history filter, test scores
- `fig07_strf_gallery.png` — twelve fibres' STRFs ordered by best frequency
- `fig08_population_metrics.png` — STRF BF vs tone BF, latency, bandwidth, prediction vs ceiling, causality
- `fig09_population_average_strf.png` — BF-aligned mean STRF and its marginal profiles
- `fig10_revcor.png` — waveform-triggered average at low, middle and high best frequency

## Caveats

The stimulus is a single frozen noise token per fibre, so cross-validation tests
generalization to unseen stretches of one noise process rather than to a new stimulus
ensemble, and the effective number of independent stimulus samples is set by the token
length rather than by the 60 repeats. Spectral width is measured at the resolution of
the analysis filterbank, and most fibres' STRFs occupy one or two bands, so the
bandwidth numbers are an upper bound on tuning sharpness rather than a tuning
measurement. Peak latency shows no systematic dependence on best frequency once the
group delay of the analysis filterbank is subtracted (+0.17 ms per octave); the raw
latencies do trend with frequency, but that trend is almost entirely the filterbank's.
Absolute sound level is not recoverable from the files: the four `NOISE_k` entries hold
the same waveform and differ only in an attenuator setting, so levels are reported as
the driven rates they produced. Finally, a note on dataset selection: DANDI:001419 was
tried first and abandoned because its spike times are rounded to whole seconds, which
makes any receptive-field analysis impossible; the same appears to hold for the sibling
dandisets 001420 and 001421.
