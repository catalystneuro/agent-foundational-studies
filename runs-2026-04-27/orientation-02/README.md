# Orientation Selectivity in Mouse Visual Cortex

This analysis demonstrates **orientation and direction selectivity** in mouse
visual cortex using a publicly available Neuropixels recording from the DANDI
Archive.

## Dataset

- **Dandiset:** [DANDI:000021 — Allen Institute - Visual Coding - Neuropixels (Brain Observatory 1.1 Stimulus Set)](https://dandiarchive.org/dandiset/000021)
- **Session used:** `sub-707296975/sub-707296975_ses-721123822.nwb` (~1.7 GB),
  one Pvalb-Cre;Ai32 mouse recorded with 6 Neuropixels probes spanning V1
  (`VISp`) and higher visual areas (`VISl`, `VISal`, `VISrl`, `VISam`, `VISpm`).
- **Stimulus analyzed:** the `drifting_gratings_presentations` block —
  full-field sinusoidal gratings shown for 2 s at 8 directions
  (0/45/90/135/180/225/270/315°) × 5 temporal frequencies (1/2/4/8/15 Hz),
  contrast 0.8.
- **Streaming:** the file is streamed from S3 with `remfile.DiskCache` and
  read with `pynwb` + `pynapple`'s `NWBFile` adapter — no full download.

## What was analyzed

For every quality-passing unit recorded in mouse visual cortex (n = 197 across
6 areas), the script:

1. Computes the mean firing rate during each 2-s grating trial using
   `pynapple.TsGroup.count(ep=…)`.
2. Builds a per-unit orientation tuning curve (mean rate at each of the 8
   directions, averaged over all temporal frequencies × repeats).
3. Quantifies selectivity with the standard circular-statistics indices on
   raw firing rates (Allen Institute convention):
   - `OSI = |Σ rₖ exp(2iθₖ)| / Σ rₖ` (orientation selectivity)
   - `DSI = |Σ rₖ exp(iθₖ)| / Σ rₖ` (direction selectivity)
   - preferred direction = arg of the DSI vector
4. Tests significance with one-way ANOVA across orientations.
5. Visualizes example rasters, polar tuning curves, PSTHs at preferred vs.
   orthogonal vs. null directions, and population summaries.

## Key findings (single session, n = 197 visual-cortex units)

- **41.6% of units (82/197)** show statistically significant orientation tuning
  (ANOVA p < 0.01 *and* OSI > 0.2).
- **Most selective V1 example (unit 950908404):** OSI = 0.77, DSI = 0.07. The
  raster (`fig01`) and PSTH (`fig03`) show ~20 Hz sustained responses at the
  preferred direction (225°) and the collinear opposite direction (45°), with
  near-zero evoked response at the orthogonal orientation (315°) — the
  textbook signature of an orientation-tuned, *not* direction-tuned, V1 cell.
- **Population:** OSI distribution skews toward larger values than DSI
  (`fig04`, OSI > DSI for almost every unit), consistent with the well-known
  result that orientation tuning is more prevalent than direction tuning in
  mouse V1. Preferred directions are distributed across the full 360°
  (`fig04`, `fig05`).

## Files

| File | Description |
| --- | --- |
| `orientation_selectivity.py` | Jupytext source (paired with the .ipynb), runs end-to-end |
| `orientation_selectivity.ipynb` | Same content as a Jupyter notebook |
| `fig01_example_v1_raster.png` | Per-orientation spike raster for the most selective V1 unit |
| `fig02_polar_tuning_examples.png` | Polar tuning curves for 8 selective V1 units |
| `fig03_best_unit_psth.png` | Raster + PSTH at preferred / orthogonal / null direction |
| `fig04_population_summary.png` | OSI distribution, OSI-vs-DSI scatter, preferred-direction histogram |
| `fig05_tuning_heatmap.png` | Heatmap of normalized tuning curves sorted by preferred direction |
| `unit_summary.csv` | Per-unit OSI, DSI, preferred direction, ANOVA p-value, region |

## Reproducing

The script will stream the NWB file on first run, cache byte ranges in
`/tmp/remfile_cache`, and write all figures and `unit_summary.csv` next to the
script.

```bash
python orientation_selectivity.py
```

Dependencies: `pynapple`, `pynwb`, `h5py`, `remfile`, `numpy`, `pandas`,
`scipy`, `matplotlib`, `tqdm`, `jupytext`.
