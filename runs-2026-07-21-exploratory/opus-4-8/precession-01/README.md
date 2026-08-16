# Theta Phase Precession in Hippocampal CA1 Place Cells

This analysis demonstrates theta phase precession, one of the most robust
temporal codes in the hippocampus, using real electrophysiology streamed from the
DANDI Archive.

## Dataset

**DANDI:000044** — *Diversity in neural firing dynamics supports both rigid and
learned hippocampal sequences* (Grosmark, Long & Buzsáki). Session
`sub-Achilles/ses-Achilles-10252013`. A rat runs back and forth on a 1.6 m linear
track (the "MAZE" epoch, flanked by pre- and post-run sleep). The session
provides 137 sorted CA1 units, linearized position at ~39 Hz, and a 128-channel
LFP at 1250 Hz. The ~9 GB NWB file is never downloaded in full; it is streamed
with `remfile` and only the MAZE-epoch spikes, position, and a single
strong-theta LFP channel (selected by theta/delta power ratio, channel 44) are
pulled and cached locally.

## What was analyzed

Restricting to running periods (speed > 0.1 m/s), track traversals were split
into rightward and leftward runs, and direction-specific 1-D place fields were
computed for the excitatory units. Place cells were selected by peak firing rate
(> 3 Hz), Skaggs spatial information (> 0.3 bits/spike), and having a single
contiguous field. For each place cell, the spikes falling inside its field were
paired with (i) the fraction of the field the animal had travelled and (ii) the
instantaneous theta phase (Hilbert transform of the 6–12 Hz band). Precession was
quantified per field with the Kempter et al. (2012) circular-linear regression,
with significance from a phase-shuffle permutation test.

## Key finding

CA1 place cells show robust theta phase precession in both running directions. Of
88 direction-specific place fields, 53 precessed significantly (negative slope,
p < 0.05) versus only 5 with a significant positive slope. The slope distribution
is strongly biased negative (median ≈ −184°/field, roughly half a theta cycle of
phase advance as the animal crosses a field), the circular-linear correlation is
negative for essentially every significant field (median ρ ≈ −0.24), and the
pooled phase-vs-position curve descends monotonically from field entry to exit.
This reproduces the classic O'Keefe & Recce (1993) result on real CA1 recordings
streamed directly from DANDI.

## Files

- `phase_precession_analysis.py` — consolidated jupytext script (percent format),
  runs end-to-end (streams and caches on first run).
- `phase_precession_analysis.ipynb` — executed notebook version.
- `load_data.py`, `preprocess.py`, `analyze.py`, `precession.py` — the modular
  development pipeline (loading, pynapple preprocessing, place fields, precession).
- `fig1_raw_data_validation.png` — position + speed, LFP with theta band, raster.
- `fig2_example_precession.png` — six example cells: place field (top) and
  spike theta-phase vs within-field position with the fitted regression (bottom).
- `fig3_population_summary.png` — slope distribution, ρ distribution, and the
  pooled phase-vs-position relationship across all significantly precessing cells.
- `achilles_maze_cache.npz` — cached MAZE-epoch data (regenerated if absent).

## Reproducing

```bash
pip install pynapple dandi remfile h5py pynwb scipy matplotlib jupytext
python phase_precession_analysis.py          # or open the .ipynb
```
