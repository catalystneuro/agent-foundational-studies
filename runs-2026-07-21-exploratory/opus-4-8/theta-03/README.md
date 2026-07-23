# Theta phase entrainment and phase precession in hippocampal place cells

This analysis demonstrates two classical properties of hippocampal CA1 place cells,
theta phase entrainment and theta phase precession, using real electrophysiology from
the DANDI Archive.

## Dataset

[**DANDI:000044**](https://dandiarchive.org/dandiset/000044) — Grosmark, Long and
Buzsáki, *Recordings from hippocampal area CA1, PRE, during and POST novel spatial
learning* (the dataset behind Grosmark & Buzsáki, Science 2016, also distributed as
CRCNS hc-11). Bilateral silicon-probe recordings from dorsal CA1 of freely moving rats,
each session containing sorted units with cell-type labels, 128-channel LFP at 1250 Hz,
tracked position, and an epoch table separating pre-sleep, maze running and post-sleep.

Five of the eight sessions used a linear track (four 1.6 m, one 2 m) and are analysed
here: `Achilles-10252013`, `Cicero-09012014`, `Cicero-09172014`, `Gatsby-08022013` and
`Buddy-06272013`, from four rats. The other three sessions used a circular maze whose
linearized coordinate wraps around, which would require different field-detection logic,
and were skipped.

The NWB files are 5-9 GB each and are never downloaded in full. They are read over HTTP
with LINDI (which resolves into byte-range requests against the DANDI S3 bucket) and a
local cache. Because the LFP dataset is chunked one channel at a time, a single LFP
channel over an entire maze epoch costs only a few megabytes.

## What was analysed

The maze epoch was split into single track traversals (the linearized position is stored
only while the animal is on the track, so each contiguous block of finite samples is one
traversal), and the two running directions were treated separately because place fields
on a linear track are directional. For each session, the LFP channel with the largest
6-12 Hz / 1-4 Hz power ratio during running was selected, and theta phase was defined by
the Hilbert transform of the bandpassed signal, with 0 deg at the theta peak and 180 deg
at the trough. Directional rate maps were built with Pynapple, and place fields were
kept for excitatory units with a peak rate of at least 1 Hz, at least 0.3 bits/spike of
spatial information, and a contiguous region above 25% of the peak rate.

Phase locking was tested per field with a Rayleigh test, against a control in which each
spike train was jittered by a uniform offset of up to ±400 ms (preserving firing rate and
field position, destroying the relationship to the ongoing theta cycle). Precession was
fitted with the circular-linear regression of Kempter et al. (2012), regressing spike
phase on the animal's normalized position within the field oriented along the direction
of travel, so that a negative slope means phase advance. Significance came from a
permutation test with 1000 shuffles of the phase-position pairing.

## Key finding

Both phenomena are clearly present. Across the five sessions, 277 directional place
fields met the criteria. Of these, 199 (72%) showed significant phase locking to theta
(Rayleigh p < 0.05); the median mean resultant length was 0.256 against 0.093 for the
spike-jitter control, and 91% of individual fields were more strongly locked than their
own control (Wilcoxon signed rank p = 1e-42). Preferred phases were themselves clustered
across the population (mean 191 deg, R = 0.33, p = 3e-10), that is, close to the trough
of theta on the selected channel. The absolute value of the preferred phase depends on
recording depth, since the theta wave reverses across the CA1 layers, and one session
(`Buddy-06272013`) indeed sits about 150 deg away from the other four; it is the
clustering within a session, not the numerical value across sessions, that is the result.

Within a field, spike phase advanced systematically with position. Of the 199 fields with
enough in-field spikes to fit, 114 (57%) had a significant circular-linear correlation,
and 104 of those 114 (91%) had a negative slope. The median slope among significant
fields was −198 deg per field traversal, roughly half a theta cycle, with a median
circular-linear correlation of −0.32. The effect does not depend on the significance
threshold: pooling all 199 fitted fields, 82% have a negative slope and the slope
distribution is shifted well below zero (median −176 deg per field, Wilcoxon p = 9e-15).
Every one of the five sessions shows the same direction of effect, with per-session
median slopes between −140 and −180 deg per field.

## Figures

| file | content |
| --- | --- |
| `fig01_behaviour_and_spiking.png` | track traversals, running speed, and the place-cell sequence during one traversal |
| `fig02_theta_lfp.png` | raw and theta-filtered LFP with phase, running vs. rest spectra, channel selection, trough-triggered average |
| `fig03_place_fields.png` | directional rate maps for all fields, spatial information, example place fields |
| `fig04_theta_entrainment.png` | example spike-phase histograms, population preferred phases, locking strength vs. jitter control |
| `fig05_precession_examples.png` | phase vs. position for six single fields with the circular-linear fit |
| `fig06_precession_population.png` | slope and correlation distributions, pooled phase-position density, population phase advance |
| `fig07_across_sessions.png` | per-session precession fraction, slope, entrainment and slope distributions |

## Files

- `theta_precession_hippocampus.py` — consolidated jupytext script (percent format), runs end to end
- `theta_precession_hippocampus.ipynb` — the same as an executed Jupyter notebook
- `hc11_io.py` — streaming access to DANDI:000044 (LINDI), position / units / LFP loading
- `theta_analysis.py` — laps, speed, theta phase, circular statistics, circular-linear regression, place-field detection
- `run_session.py` — per-session pipeline, caches results to `results/`
- `make_figures.py` — all figures
- `run_all.py` — runs every session in the dandiset
- `results/*.csv`, `results/*.npz`, `all_sessions_fields.csv` — per-field statistics

Running the notebook from a cold cache takes roughly 30-45 minutes, most of it spent
fetching the LINDI indexes and scanning LFP channels; subsequent runs reuse the caches in
`/tmp/lindi_cache` and `results/`.

## References

- O'Keefe & Recce (1993), *Phase relationship between hippocampal place units and the EEG theta rhythm*, Hippocampus 3:317-330.
- Skaggs, McNaughton, Wilson & Barnes (1996), *Theta phase precession in hippocampal neuronal populations and the compression of temporal sequences*, Hippocampus 6:149-172.
- Kempter, Leibold, Buzsáki, Diba & Schmidt (2012), *Quantifying circular-linear associations: hippocampal phase precession*, J Neurosci Methods 207:113-124.
- Grosmark & Buzsáki (2016), *Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences*, Science 351:1440-1443.
