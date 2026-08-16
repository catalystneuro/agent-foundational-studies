# Spectrotemporal Receptive Fields in Mouse Auditory Cortex

This analysis uses [DANDI:000986](https://dandiarchive.org/dandiset/000986), "Auditory cortex
Neuropixels recordings and pupil diameter traces from mice during passive exposure to pure tones"
(Jaramillo lab). The session analyzed (`sub-LA11/sub-LA11_ses-1`) contains Neuropixels recordings
from 235 units in mouse auditory cortex during passive exposure to ~7,450 brief (25 ms) pure-tone
pips, pseudorandomly drawn from 5 log-spaced frequencies (2, 4, 8, 16, 32 kHz) and delivered at a
fixed sound level roughly every 0.8 s. The data were streamed directly from the DANDI S3 bucket
with `remfile` (no full download) and analyzed with Pynapple.

Because the tone frequency on each trial is drawn independently of the ongoing neural activity,
the tone-pip sequence acts as a valid stimulus for reverse correlation, even though it samples
only 5 discrete frequencies rather than a continuous spectrogram. For every unit we computed the
tone-triggered average firing rate as a function of frequency and time lag after tone onset
(z-scored against firing measured during interleaved spontaneous, no-stimulus blocks), which is
exactly the definition of a spectrotemporal receptive field (STRF). As an independent, model-based
cross-check, we also fit a Poisson GLM (NeMoS) per unit in which each frequency channel is a
binary event train convolved with log-spaced raised-cosine basis functions, and reconstructed the
GLM's spectrotemporal filter from the fitted basis weights.

## Key finding

18 of 235 recorded units (~8%) showed a tone-evoked response exceeding 3 standard deviations from
spontaneous baseline within 150 ms of tone onset. These units show canonical STRF structure: a
well-defined best frequency, a short-latency (10-40 ms) transient excitatory response, and in
several units a longer secondary excitatory component or spread of excitation to neighboring
octaves. Preferred frequencies were heterogeneous across the sampled population (most common in
the 2-8 kHz range but present up to 32 kHz), consistent with the coarse tonotopic organization
known to exist along a single linear Neuropixels probe track in mouse auditory cortex. The
independently fit NeMoS Poisson-GLM receptive fields closely reproduced the best frequency and
temporal profile of the non-parametric reverse-correlation STRFs for every example unit compared,
confirming that both methods recover the same underlying tuning.

## Files

- `spectrotemporal_receptive_fields.py` — jupytext (light format) analysis script, runs end-to-end
- `spectrotemporal_receptive_fields.ipynb` — same analysis as an executed Jupyter notebook
- `figures/01_raw_raster_and_stimulus.png` — raw spike raster with tone-pip stimulus timeline
- `figures/02_response_strength_distribution.png` — tone-evoked response strength across all 235 units
- `figures/03_strf_grid_top_units.png` — reverse-correlation STRFs for the 12 most tone-driven units
- `figures/04_population_summary.png` — best-frequency and response-latency distributions
- `figures/05_glm_vs_reverse_correlation.png` — reverse-correlation STRF vs. NeMoS GLM receptive field
