# Orientation Selectivity in Mouse Visual Cortex

## Dataset

**DANDI [000021](https://dandiarchive.org/dandiset/000021)** — *Allen Institute Visual Coding — Neuropixels (Brain Observatory 1.1 Stimulus Set)*.
Single session: `sub-699733573_ses-715093703.nwb` (asset
`58703c97-c0a9-4736-b684-73c85c1a444a`). Six Neuropixels probes simultaneously
recorded ~2,800 single units across visual thalamus (LGd), primary visual
cortex (VISp), four higher visual areas (VISl, VISrl, VISam, VISpm), and
several non-visual regions, while an awake head-fixed mouse passively viewed
the Brain Observatory drifting-grating block (8 directions × 5 temporal
frequencies, 0.04 cyc/deg, 0.8 contrast, 2 s on / 1 s off, 15 reps each, plus
blanks). The NWB file was streamed directly from S3 with `remfile`, so no full
download was required.

## What is analyzed

This analysis demonstrates the textbook phenomenon of **orientation selectivity**
— neurons in early visual cortex respond preferentially to a specific stimulus
orientation. The pipeline:

1. **Streams** the NWB file with `remfile` + `pynwb`, wrapped in a `pynapple.NWBFile`.
2. Selects 810 `quality == "good"` single units whose peak channel was localized to
   one of `{VISp, VISl, VISrl, VISam, VISpm, LGd}`.
3. Builds a Pynapple `TsGroup` of spike times and uses `TsGroup.count(IntervalSet)`
   to compute trial-by-trial spike counts on each 2-second grating window.
4. Averages across repetitions to obtain per-unit tuning curves over 8 directions.
5. Quantifies tuning with three standard metrics:
   - **OSI** (orientation selectivity index, mod 180°): `(R_pref − R_orth) / (R_pref + R_orth)`
   - **DSI** (direction selectivity index): `(R_pref − R_null) / (R_pref + R_null)`
   - **gOSI** (1 − circular variance at 2θ): `|Σ rᵢ e^{i2θᵢ}| / Σ rᵢ`
6. Significance: 200-shuffle permutation test on trial-orientation labels for gOSI.

## Key findings

- **578 / 810 = 71.4 %** of recorded visual units have a gOSI exceeding the 95th
  percentile of trial-shuffled gOSI — i.e. their tuning is statistically significant.
- **VISp leads in significance**: 84 % of VISp units pass the permutation test
  (median gOSI ≈ 0.17, classical OSI ≈ 0.24), consistent with V1 being the
  primary site of orientation extraction.
- **Direction selectivity is weaker than orientation selectivity** in every area,
  as expected — orientation tuning saturates earlier in the hierarchy than
  direction tuning.
- The example V1 unit (#950912427) shows a clean unimodal tuning curve peaking at
  ~90° drift (vertical-grating-preferring), with classical OSI = 0.88 and
  gOSI = 0.63 — see `fig1_example_v1_unit.png`.
- Population heatmaps (`fig4`) show a clear diagonal stripe when units are sorted
  by preferred direction in every visual area, demonstrating that orientation
  tuning is a population-wide phenomenon rather than a property of a few cells.
- The polar histogram of preferred orientations (`fig5`) shows a slight
  cardinal-orientation bias in VISp/VISl that has been reported in mouse V1.

## Files

| File | Description |
|---|---|
| `orientation_selectivity_analysis.py` | Final consolidated jupytext-format analysis script (runs end-to-end) |
| `orientation_selectivity_analysis.ipynb` | Same content as a Jupyter notebook |
| `fig1_example_v1_unit.png` | Raster + linear + polar tuning for a high-OSI V1 unit |
| `fig2_top_units_per_area.png` | Polar tuning curves of the 6 most selective units per area |
| `fig3_selectivity_by_area.png` | Box+strip plots of OSI / DSI / gOSI across visual areas |
| `fig4_population_heatmap.png` | Per-area heatmap of z-scored tuning curves, sorted by preferred direction |
| `fig5_preferred_orientation_hist.png` | Polar histogram of preferred orientations (gOSI-significant units) |
| `unit_orientation_summary.csv` | Per-unit metrics (OSI, DSI, gOSI, preferred orientation, p-value) |
| `area_selectivity_summary.csv` | Median selectivity and significant-fraction per area |
| `01_explore.py`, `02_inspect_stimuli.py` | Prototype/inspection scripts used during development |

## Running

```bash
pip install pynapple lindi remfile pynwb h5py tqdm matplotlib jupytext
python orientation_selectivity_analysis.py
```

The script caches streamed S3 chunks in `/tmp/remfile_cache_orient`, so reruns
are fast.
