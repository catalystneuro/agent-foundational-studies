# Theta phase entrainment and phase precession in hippocampal CA1 place cells

This analysis demonstrates two hallmarks of the hippocampal temporal code using real
electrophysiology from the DANDI Archive: the locking of CA1 spikes to the local
field potential theta rhythm, and the systematic advance of place cell spike phase
across a place field.

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark & Buzsáki (2016),
*Diversity in neural firing dynamics supports both rigid and learned hippocampal
sequences*, Science 351:1440–1443. Bilateral 128-site silicon-probe recordings from
dorsal CA1 of freely moving Long-Evans rats, with 1250 Hz LFP, spike-sorted single
units labelled as excitatory or inhibitory, and video tracking on a linear maze.
Five of the eight sessions use a linear maze (Achilles 10/25/2013, Buddy 06/27/2013,
Cicero 09/01/2014 and 09/17/2014, Gatsby 08/02/2013) and all five are analyzed here;
the three circular-maze sessions are excluded because a periodic linearization
complicates the field-normalized regression. The NWB files are streamed from S3 with
`remfile` and an on-disk byte cache, so the roughly 8 GB per session never has to be
downloaded in full. All time-series handling is done with pynapple.

## What was analyzed

For each session, a theta reference channel was chosen as the LFP site with the
highest 6–10 Hz to 2–4 Hz power ratio during running, band-passed with a 4th-order
Butterworth filter, and converted to instantaneous phase by Hilbert transform. Laps
were extracted as contiguous blocks of valid linearized position, split by running
direction, and restricted to samples above 0.1 m/s. Directional firing-rate maps
were computed in 4 cm bins, and a unit was accepted as having a place field if its
peak rate exceeded 1 Hz, its Skaggs spatial information exceeded 0.4 bits/spike, the
contiguous region above 20% of peak was 12–90 cm wide, and it fired at least 50
spikes inside that region. Entrainment was quantified by mean resultant length and
the Rayleigh test. Precession was quantified by the circular–linear regression of
Kempter et al. (2012), fitting spike phase against position normalized to the field
and oriented along the direction of travel, with significance assessed by 500
phase-shuffle permutations of the statistic the fit maximizes.

## Key findings

Across the five sessions, 346 units had enough spikes during running to test, and
233 of them (67%) were significantly phase-locked to theta at p < 0.01: 156 of 266
excitatory cells (59%) and 77 of 80 inhibitory cells (96%). Interneurons locked more
tightly than pyramidal cells (median mean resultant length 0.213 versus 0.152,
Mann-Whitney U = 13419, p = 2.0e-4), which is the expected ordering given that an
interneuron fires on most theta cycles while a pyramidal cell fires only inside its
place field. The 155 accepted place fields (median width 0.52 m, median spatial
information 0.87 bits/spike) yielded 116 fields (75%) with significant phase
precession, and 86% of all fields had a negative regression slope (93% among the
significant ones; Wilcoxon signed-rank on the slopes, W = 1568, p = 6.3e-16). The
median slope was −0.54 theta cycles per field traversal, that is about 194° of phase
advance from field entry to field exit, in line with the classic reports of
O'Keefe & Recce (1993) and Skaggs et al. (1996). Pooling all significantly
precessing fields produces the canonical negative diagonal band in the
phase-versus-position density (`fig05`), and the observed circular–linear
correlations (median |ρ| = 0.255) lie well outside the phase-shuffled null.

The main caveat is that the phase reference is a single LFP channel picked by
theta/delta ratio with no independent depth calibration, so the absolute preferred
phases reported here are not directly comparable to studies referenced to the
pyramidal-layer trough. Relative phase, which is what both phenomena depend on, is
unaffected.

## Files

| File | Contents |
| --- | --- |
| `theta_precession_dandi000044.py` | Consolidated jupytext (percent format) script, runs end to end |
| `theta_precession_dandi000044.ipynb` | Same script converted to a notebook |
| `fig00_theta_channel_selection.png` | Theta/delta power ratio across the 128 recording sites |
| `fig01_raw_data_and_theta.png` | Position, raw and filtered LFP, theta phase, spike raster for one lap; LFP spectrum; all laps |
| `fig02_place_fields.png` | Directional rate maps for all accepted place fields, plus four examples |
| `fig03_theta_entrainment.png` | Phase convention, example units, pooled phase distributions, locking strength and significance, per-session reproducibility |
| `fig04_precession_examples.png` | Eight single place fields with their phase-versus-position scatter and fitted regression |
| `fig05_precession_population.png` | Pooled phase density, mean phase per position bin, slope distribution, observed versus shuffled correlation, per-session summary |
| `session_summary.csv` | Per-session counts and medians |
| `theta_entrainment.csv` | Per-unit entrainment statistics |
| `place_field_precession.csv` | Per-field place and precession statistics |

The development scripts used to build the pipeline (`01`–`07` plus the `theta_lib`,
`theta_analysis`, and `theta_figures` modules) are also kept in this directory; the
consolidated jupytext script is self-contained and does not import them.

## Running it

```bash
pip install pynapple pynwb remfile h5py matplotlib pandas scipy tqdm jupytext
python theta_precession_dandi000044.py
```

The first run streams roughly 1–2 GB of LFP and spike data from DANDI into
`/tmp/remfile_cache` and takes on the order of half an hour; subsequent runs read
from the cache and are much faster.

The script was re-run end to end from a clean directory as a reproducibility check.
It completed in about two minutes against a warm cache, with no errors or warnings,
and regenerated all six figures and all three CSV files byte for byte identically to
the versions committed here.
