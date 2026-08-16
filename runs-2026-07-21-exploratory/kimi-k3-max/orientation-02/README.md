# Orientation Selectivity in Mouse Visual Cortex (DANDI 000021)

## Dataset

**DANDI Archive dandiset 000021** — Allen Institute Visual Coding, Neuropixels
(Brain Observatory 1.1 stimulus set), version 0.251116.2246. We analyzed one
session (`sub-699733573_ses-715093703.nwb`, session 715093703), streamed from
S3 with `remfile` plus a local disk cache (no full download). The session
contains 2779 sorted units recorded on 6 Neuropixels probes and the complete
stimulus tables. The drifting-gratings stimulus comprises 8 directions
(0–315° in 45° steps) × 5 temporal frequencies (1–15 Hz; spatial frequency
0.04 cpd, contrast 0.8), presented for 2 s with ~1 s inter-trial intervals,
plus interleaved blank (mean-luminance) sweeps. After excluding presentations
that overlap the session's `invalid_times`, 487 valid sweeps remained
(~54–62 per direction, 23 blank).

## Analysis

For every unit flagged `quality == 'good'` in VISp, VISl, VISpm, VISam, VISrl,
LGd, and CA1 (1186 units total), we computed per-trial firing rates during
each grating sweep, averaged them into an 8-point direction tuning curve,
subtracted the blank-sweep rate (clipped at 0), and computed the global
orientation selectivity index (gOSI, the 2θ resultant-vector magnitude), the
global direction selectivity index (gDSI), and the classic pref-vs-orthogonal
OSI. Significance was assessed per unit with a permutation test (1000 shuffles
of the direction labels, recomputing the identical gOSI statistic); a unit was
called orientation-selective at p < 0.05 with a peak blank-subtracted response
≥ 1 Hz. CA1 serves as a negative control. The full pipeline is in
`orientation_selectivity_analysis.py` (jupytext) / `.ipynb`, which runs
end-to-end and regenerates all figures and `orientation_results.csv`.

## Key findings

Orientation selectivity is widespread in the recorded visual areas but absent
outside them. The fraction of significantly orientation-selective units was
45% in primary visual cortex (VISp, 61/135), 38–49% across higher visual areas
(VISl, VISpm, VISam, VISrl), 29% in the visual thalamus (LGd), and only 4.5%
in hippocampal CA1 (17/376) — statistically indistinguishable from the 5%
false-positive rate of the permutation test, which validates the procedure.
Among selective units, gOSI almost always exceeded gDSI (points below the
diagonal in Fig. 3C), the classic signature of orientation- rather than
direction-selective cortical neurons, and the sorted population heatmap
(Fig. 4) shows the expected two-band structure of axis-tuned responses.
Preferred orientations of selective VISp units covered all four sampled axes,
with a modest, non-significant excess at 90° (11/14/24/12 units at
0/45/90/135°, chi² p ≈ 0.07).

## Files

- `orientation_selectivity_analysis.py` — consolidated jupytext script (runs end-to-end)
- `orientation_selectivity_analysis.ipynb` — executed notebook version
- `fig1_raw_data.png` — spike rasters and per-direction PSTHs for a tuned and an untuned VISp unit
- `fig2_example_tuning.png` — tuning curves (Cartesian + polar) for six example units
- `fig3_population.png` — fraction selective per area, gOSI distributions, gOSI-vs-gDSI, preferred-orientation histogram
- `fig4_tuning_heatmap.png` — normalized direction tuning of all selective VISp units, sorted by preference
- `fig5_permutation.png` — permutation null distributions vs observed gOSI for two example units
- `orientation_results.csv` — per-unit metrics (rates per direction, gOSI/OSI/gDSI/DSI, p-values, selectivity)
- `01–07_*.py` — incremental development scripts (loading, inspection, analysis, figures)
