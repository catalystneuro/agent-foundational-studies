# Theta phase entrainment and phase precession in hippocampal place cells

This analysis demonstrates two properties of dorsal CA1 pyramidal cells from real recordings in
the DANDI Archive: that their spikes are locked to the phase of the ongoing theta rhythm, and
that within a place field the preferred phase advances systematically as the animal runs through
the field.

## Dataset

[**DANDI:000044**](https://dandiarchive.org/dandiset/000044) — Grosmark, A. D. & Buzsáki, G.
(2016), *Diversity in neural firing dynamics supports both rigid and learned hippocampal
sequences*, Science 351:1440–1443 (the `hc-11` dataset). Bilateral silicon-probe recordings from
dorsal CA1 of freely moving rats shuttling back and forth on a 1.6 m linear track for water
reward. Each NWB file contains spike-sorted single units labelled excitatory or inhibitory, a
128-channel local field potential at 1250 Hz, and linearized track position at 39.06 Hz.

Four sessions were analysed, one from each of the four rats in the dandiset:
`Achilles-10252013`, `Gatsby-08022013`, `Buddy-06272013` and `Cicero-09012014`. The remaining
four assets were excluded: three are circular-maze sessions, on which "entering a field from the
near end" is not defined in the same way, and one (`Cicero-09172014`, a 2 m track) has only eight
complete rightward traversals, leaving spatial bins the animal never visited. The files are
6–9 GB each and were never downloaded; they were streamed over HTTP through a LINDI index with a
local byte cache, which for a single LFP channel over a 35-minute maze epoch transfers a few
megabytes.

## What was done

Analysis was restricted to running epochs (speed above 0.15 m/s) on the central 0.10–1.50 m of
the track, with rightward and leftward traversals kept separate. The reward ends are excluded
because the animal pauses there and theta gives way to sharp-wave ripples. The theta reference
channel was selected functionally in each session, by the largest 6–10 Hz to 2–4 Hz power ratio,
because all 128 electrodes are labelled `location = "unknown"` in this conversion. Theta phase is
the Hilbert phase of the 6–10 Hz band-pass filtered LFP, with 0° at the peak of the filtered wave.

Phase locking was quantified per unit with the mean resultant length and the Rayleigh test.
Place fields were taken as the contiguous region above 25% of the peak of a smoothed directional
rate map, subject to a peak rate of at least 2 Hz, spatial information of at least 0.4 bits/spike,
a width between 12 and 70 cm, and at least 50 in-field spikes. Precession was measured by the
Kempter et al. (2012) circular-linear regression of spike phase on normalized in-field position,
with significance required both from the asymptotic test on the circular-linear correlation and
from a 500-fold spike-phase permutation. Finally, Poisson GLMs were fitted with NeMoS to compare,
by five-fold cross-validated log-likelihood, a position-only model against position-plus-phase
(additive) and position × phase (interaction) models.

## Key findings

Both phenomena are present and replicate across all four animals. During running, 170 of 234 units
(73%) fired non-uniformly with respect to theta phase at p < 0.01, and interneurons were locked
roughly one and a half times as tightly as pyramidal cells (median mean resultant length 0.235
versus 0.157), the expected asymmetry between cells that follow the rhythm on nearly every cycle
and cells that participate in only a small fraction of cycles. Of 93 place fields, 63 (68%) showed
a significant circular-linear relation between spike phase and position within the field, and
**62 of those 63 (98%) had a negative slope**, with a median of −222° per field traversal and a
median circular-linear correlation of 0.32. Pooling the spikes of all precessing fields, mean
spike phase fell from 252° in the first third of the field to 101° in the last third. The effect
is visible on individual passes and not only in the pooled average: 78% of 701 single traversals
gave a negative slope. The GLM comparison separates the two phenomena directly — adding an
additive phase term to a position-only model improved cross-validated likelihood for 13 of 20
fields (phase locking), and adding the position × phase interaction improved it further for 19 of
20 (precession), producing joint rate maps with a tilted rather than horizontal ridge.

The single largest methodological trap is the direction convention. Because a leftward traversal
enters a place field at the *larger* track coordinate, running the regression in absolute track
coordinates yields slopes of opposite sign for the two running directions. An earlier version of
this analysis did exactly that and found 19 negative and 24 positive slopes, i.e. chance, before
the coordinate was re-expressed in the animal's direction of travel.

## Files

| File | Contents |
| --- | --- |
| `theta_phase_precession_dandi000044.py` | Consolidated jupytext script, runs end to end |
| `theta_phase_precession_dandi000044.ipynb` | The same analysis as an executed notebook |
| `fig01_raw_streams.png` | Position, traversals, raw and filtered LFP, extracted phase |
| `fig02_lfp_spectrum.png` | LFP power spectrum during running; theta channel selection |
| `fig03_place_fields.png` | Direction-specific place field maps and occupancy |
| `fig04_entrainment_examples.png` | Spike-phase histograms and polar plots for example units |
| `fig05_entrainment_population.png` | Phase-locking strength and preferred phase across the population |
| `fig06_precession_examples.png` | Phase versus in-field position for six place fields |
| `fig07_precession_population.png` | Slope and correlation distributions; pooled phase-position density |
| `fig08_single_pass.png` | Precession on individual traversals |
| `fig09_glm.png` | NeMoS GLM model comparison and joint position-phase rate maps |
| `fig10_multisession.png` | Replication across the four rats |
| `results_*.csv` | Per-unit entrainment, per-field precession, GLM scores, session summary |

Run with `python theta_phase_precession_dandi000044.py`. Requires `pynapple`, `nemos`, `lindi`,
`pynwb`, `scipy`, `scikit-learn`, `matplotlib` and `tqdm`. The first run streams roughly a
gigabyte into `./lindi_cache`; later runs reuse it.
