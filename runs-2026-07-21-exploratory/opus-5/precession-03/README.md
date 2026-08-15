# Theta phase precession in hippocampal place cells (DANDI:000044)

This directory contains an end-to-end analysis that demonstrates theta phase precession
in dorsal CA1 place cells using real data streamed from the DANDI Archive.

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044) version `0.250624.0426`,
*"Diversity in neural firing dynamics supports both rigid and learned hippocampal
sequences"* (Grosmark & Buzsáki, *Science* 2016). All eight sessions were analysed,
covering four Long-Evans rats (Achilles, Buddy, Cicero, Gatsby). Each session is a
bilateral 128-site silicon-probe recording of dorsal CA1 during a maze epoch of 30 to
70 minutes: four sessions on a 1.6 m linear track, one on a 2 m linear track, and three
on a circular maze of roughly 2.9 m circumference. Each NWB file provides spike-sorted
units labelled excitatory or inhibitory, a linearised position signal, and a 1250 Hz
local field potential. The files are 5 to 9 GB each; they are read over HTTP with
`remfile` plus a local disk cache, and only about 30 MB per session is actually
transferred (one LFP channel over the maze epoch, the spike times, and the position).

## What was analysed

For each session the pipeline selects a theta reference channel by its
theta-to-background power ratio, extracts theta phase with a zero-phase 6–10 Hz
Butterworth filter and the Hilbert transform, and segments the behaviour into
per-direction running epochs above 10 cm/s. Pynapple computes occupancy-normalised rate
maps in 4 cm bins for each running direction, and a place field is accepted when the
peak rate exceeds 1 Hz, the Skaggs spatial information exceeds 0.5 bits/spike, a
contiguous region above 25 % of the peak can be delimited whose width is 10–55 % of the
track, and at least 50 in-field spikes are distributed over at least 10 separate
traversals. For every accepted field, the theta phase of each in-field spike is
regressed on the fraction of the field the animal has traversed using the
circular-linear estimator of Kempter et al. (2012), and tested against 200 within-field
phase permutations.

## Key finding

181 place fields passed the criteria across the eight sessions. 136 of them (75 %) show
a phase-position relationship stronger than any of the 200 phase shuffles reached, and
94 % of those significant fields have a **negative** slope, meaning spike phase advances
as the animal crosses the field. The median field advances by 0.68 theta cycles, about
245°, from field entry to field exit, and pooling every spike from every significant
field traces a smooth 205° advance of the population circular mean across the normalised
field. Every one of the eight sessions has a negative median slope, ranging from −0.50
to −0.83 cycles.

Three controls argue that this reflects the physiological phenomenon rather than the
analysis. The effect is present within single traversals, where the animal crosses the
field once and nothing is averaged across passes (for the example unit, 97 % of
individually fitted passes have a negative slope). Permuting spike phases within a field
collapses the slope distribution onto zero. And both directions of travel precess with
the same sign, even though the linearised position coordinate runs opposite ways in the
two cases, which rules out the direction-dependent coordinate flip as the source of the
sign. The absolute phase at field entry is not interpretable here, because the theta
reference channel is chosen by signal quality rather than anatomically and the phase
offset between the pyramidal layer and the fissure is roughly 180°; the slope, which is
what precession is about, does not depend on that choice.

## Files

| file | contents |
| --- | --- |
| `theta_phase_precession.py` | consolidated jupytext script, runs end to end |
| `theta_phase_precession.ipynb` | the same notebook, executed, with figures embedded |
| `all_fields.csv` | one row per analysed place field: slope, offset, ρ, p, field geometry |
| `fig01_session_overview.png` | position, LFP spectrum, raw and filtered theta, place-cell raster within one run |
| `fig02_place_fields.png` | sorted rate maps, accepted fields, occupancy, both directions |
| `fig03_example_precession.png` | six individual fields: rate map and phase against in-field position |
| `fig04_single_pass.png` | precession within single traversals, and per-pass fits |
| `fig05_population.png` | slope and ρ distributions, pooled phase-position density, per-session breakdown |
| `fig06_controls.png` | shuffled null, phase occupancy, both running directions |
| `dev_scratch/` | the exploratory scripts used to build the pipeline |

Requirements: `pynapple`, `remfile`, `h5py`, `pynwb`, `scipy`, `pandas`, `matplotlib`,
`tqdm`, `requests`. The notebook takes roughly ten minutes on a cold cache and two to
three minutes once the LFP segments are cached in `/tmp/remfile_cache`.
