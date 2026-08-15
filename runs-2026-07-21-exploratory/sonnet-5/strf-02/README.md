# Spectrotemporal Receptive Fields in Mouse Auditory Cortex

This analysis uses [DANDI:000986](https://dandiarchive.org/dandiset/000986), "Auditory
cortex Neuropixels recordings and pupil diameter traces from mice during passive
exposure to pure tones" (Jo et al., McCormick Lab, University of Oregon). The dataset
contains 15 sessions from 5 mice in which a Neuropixels probe recorded spikes from
auditory cortex while the animal passively listened to brief (25 ms) pure tones at five
frequencies (2, 4, 8, 16, 32 kHz), presented in randomized order and with randomized
timing. Behavioral state (pupil diameter, running speed) was recorded simultaneously
but was not the focus of this analysis.

Because tone frequency and onset time are randomized, each neuron's spectrotemporal
receptive field (STRF) can be estimated by reverse correlation: for each frequency we
built a peristimulus time histogram of firing rate around tone onset, then stacked the
five histograms into a frequency-by-time-lag matrix. This is the standard random-tone
(reverse-correlation) approach to STRF estimation used in auditory electrophysiology.
All data access and spike-time manipulation used Pynapple; data were streamed directly
from the DANDI S3 bucket via `remfile` with disk caching, without downloading full
files.

The analysis started with one example session (`sub-LA11_ses-1`, 235 sorted units) to
validate the pipeline against raw spike rasters and behavioral traces, then scaled to
all 15 sessions across all 5 mice (1,564 units total). Tone responsiveness was assessed
per unit with a Wilcoxon signed-rank test comparing spike counts in a 50 ms window
after tone onset against a 50 ms pre-tone baseline, at each unit's best frequency. 891
of 1,564 units (57%) were significantly tone-responsive (p < 0.01). Example STRFs show
the expected structure for auditory cortex: a restricted frequency band and a
short-latency (10-30 ms), transient-to-sustained onset response. Pooling across all
responsive units and aligning each unit's STRF to its own best frequency produced a
grand-average STRF with a clear V-shaped tuning profile narrowing with time and a
median response latency around 15-20 ms, consistent with known mouse auditory cortical
response properties.

## Files

- `strf_analysis.py` — consolidated jupytext analysis script (markdown + code cells), runs end-to-end
- `strf_analysis.ipynb` — same analysis as an executed Jupyter notebook
- `raw_data_overview.png` — raw spike rasters, running speed, and pupil diameter for an example session
- `psth_to_strf_construction.png` — tone-triggered rasters and PSTHs for one unit across all 5 frequencies, the building blocks of an STRF
- `example_strfs.png` — example spectrotemporal receptive fields (frequency x time-lag) from the most strongly tone-driven units
- `population_summary.png` — population-level summary across all 15 sessions/5 mice: fraction of tone-responsive units, distribution of best frequency, distribution of response latency, and grand-average BF-aligned STRF
