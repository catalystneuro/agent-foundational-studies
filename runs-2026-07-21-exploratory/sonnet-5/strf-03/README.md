# Spectrotemporal Receptive Fields in Mouse Auditory Cortex

**Dataset:** [DANDI:001419](https://dandiarchive.org/dandiset/001419), "Cortical Circuits for the integration of short-latency auditory pathways: Auditory cortex linear probe recording." Mice were implanted with a linear multi-electrode probe in primary auditory cortex (A1) and presented with pure-tone pips (10-18 log-spaced frequencies spanning 4-80 kHz, 300 ms duration, ~70 dB SPL, ~5 s inter-stimulus interval, pseudo-randomized order and timing). A subset of trials also delivered optogenetic photostimulation of a converging, ChrimsonR-expressing pathway; per the dataset's own documentation, those LED trials are excluded here and only control (LED-off) trials were analyzed. Data were streamed directly from DANDI (no local download) using `remfile` with disk caching, and loaded with `pynwb`/`pynapple`.

## Analysis

Spectrotemporal receptive fields (STRFs) were estimated by tone-pip reverse correlation: for each sorted "good"-quality unit, the peri-stimulus time histogram was computed separately for each tested frequency and the resulting histograms were stacked into a frequency x time-lag matrix. This was done for 266 units pooled across 8 sessions (8 mice). Each unit was tested for genuine spectral tuning with a Kruskal-Wallis test on per-trial spike counts (0-100 ms post-onset) across frequency conditions, and its best frequency (BF) and onset latency were read off the peak of that window.

## Key Finding

30 of 266 units (11.3%) showed statistically significant frequency tuning (p < 0.05), more than double the ~5% expected by chance under the significance threshold alone (binomial test, p = 3.3e-5). Tuned units had short onset latencies (median 40 ms, range 0-100 ms) consistent with known mouse A1 response timing, and best frequencies spanning the tested range. Individual example units show sharp, frequency-specific excitatory responses confined to a narrow band around their BF. Averaging the baseline-subtracted response of every tuned unit after aligning its frequency axis to its own BF (expressed in octaves) recovers the canonical spectrotemporal receptive field shape: a short-latency excitatory peak centered exactly at 0 octaves that falls off with spectral distance, closely matching the textbook description of an STRF.

## Files

- `strf_auditory_cortex.py` - consolidated jupytext analysis script (runs end-to-end)
- `strf_auditory_cortex.ipynb` - executed Jupyter notebook version
- `figures/01_raw_data_overview.png` - raw spike raster, tone-pip stimulus timeline, and population multi-unit activity for one example session
- `figures/02_example_strfs.png` - six example single-unit STRFs
- `figures/03_population_summary.png` - population tuning statistics, best-frequency and latency distributions, and the BF-aligned population-average STRF
