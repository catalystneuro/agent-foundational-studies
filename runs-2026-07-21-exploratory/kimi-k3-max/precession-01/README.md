# Theta Phase Precession in Hippocampal Place Cells

This analysis demonstrates theta phase precession, the tendency of hippocampal
place cells to fire at progressively earlier phases of the theta oscillation as
an animal runs through the place field (O'Keefe & Recce, 1993), using public
data streamed from the DANDI Archive.

## Dataset

Dandiset [000044](https://dandiarchive.org/dandiset/000044) ("Diversity in
neural firing dynamics supports both rigid and learned hippocampal sequences",
Buzsáki lab), session `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`.
The recording is rat CA1 (137 sorted units on tetrodes) with a 128-channel LFP
at 1250 Hz, while the animal runs back and forth on a 1.6 m linear maze
(maze epoch 18079.5-20147 s). The 8.7 GB file is accessed by streaming with
`remfile` plus a local disk cache; only the chunks read by the analysis are
fetched.

## Analysis

The pipeline (`theta_phase_precession.py`, runnable end-to-end, also provided
as an executed notebook) proceeds in four steps:

1. **Theta reference selection.** The electrodes table has no anatomical
   coordinates, so a reference channel is chosen data-driven by the
   theta (6-12 Hz) / delta (1-4 Hz) PSD ratio on a maze-epoch chunk
   (channel 117 wins, with a clear 9.5 Hz theta peak).
2. **Theta phase.** The maze-epoch LFP on that channel is bandpass-filtered at
   6-12 Hz and the instantaneous phase is taken from the Hilbert transform.
3. **Place fields.** Run epochs are bouts with speed above 10 cm/s on the
   track arm (98 bouts, 242 s), split by running direction. Per-direction
   tuning curves (50 bins, Gaussian smoothed) identify 94 place cells out of
   120 excitatory units (peak rate >= 1 Hz).
4. **Phase precession.** For each place cell with at least 30 in-field spikes
   (87 cells), spike theta phase is regressed on track position with
   circular-linear regression (resultant-length maximization over a slope
   grid; Kempter et al., 2012), and significance is assessed against 500
   phase-permutation shuffles per cell.

## Key finding

Phase precession is robust and highly significant across the population.
67 of 87 analyzed place cells show a significant circular-linear correlation
between theta phase and position (p < 0.05, shuffle test), 72 of 87 have
negative (precessing) slopes when referenced to field progress (binomial
p = 4.3e-10), and the median slope is -1.10 cycles/m (median R = 0.325 versus
0.236 in shuffles). The pooled phase-versus-field-progress plot shows the
classic descending wedge, and the effect is visible on single passes through
the field, replicating O'Keefe & Recce (1993) directly from archived data.

## Outputs

- `theta_phase_precession.py`: consolidated jupytext script (runs end-to-end)
- `theta_phase_precession.ipynb`: executed notebook version
- `fig_channel_selection.png`: theta/delta ratio per LFP channel and reference PSD
- `fig_theta_extraction.png`: raw and theta-filtered LFP, spike raster, phase trace
- `fig_place_fields.png`: per-direction population rate maps and behavior snippet
- `fig_precession_examples.png`: phase vs position for six example cells
- `fig_single_pass.png`: single-pass phase precession for one cell
- `fig_precession_population.png`: slope distribution, shuffle comparison, pooled wedge

Development scripts (`01_*.py` ... `06_*.py`) are the modular steps the
consolidated script was built from; `theta_phase_precession.py` is the
canonical deliverable.
