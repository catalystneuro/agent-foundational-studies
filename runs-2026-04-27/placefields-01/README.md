# Hippocampal place cells from DANDI 000044

Demonstration of classic CA1 place-cell coding on a 1.6 m linear track, using
a single session streamed from the DANDI Archive.

## Dataset

[DANDI 000044](https://dandiarchive.org/dandiset/000044/draft) — Grosmark, A.D.
& Buzsáki, G. (2016). *Diversity in neural firing dynamics supports both rigid
and learned hippocampal sequences.* **Science** 351:1440–1443.

* Subject: `sub-Buddy`, session `Buddy_06272013`
* Recording: bilateral silicon-probe ecephys in dorsal hippocampus (CA1),
  68 sorted units (48 putative pyramidal, 20 putative interneurons).
* Behaviour: rat shuttling for water reward on a 1.6 m linear track.
* Maze epoch analysed: 10717.9 – 13046.0 s (≈ 39 min).

The NWB file is streamed from S3 with `remfile` + on-disk caching — no full
download is performed.

## Pipeline

`place_cells_analysis.py` (jupytext, also exported as `place_cells_analysis.ipynb`)
runs end-to-end:

1. Stream NWB, identify the `MazeEpoch`, load 68 spike-time units.
2. Reconstruct the linearised-position time-series (the file's `rate` field
   stores the sampling *period*, not Hz — handled explicitly).
3. Group valid position samples into **continuous on-track segments**, then
   compute per-sample velocity and split each segment into right-bound and
   left-bound running sub-intervals (`|v| > 5 cm/s`).
4. For each putative pyramidal cell (excitatory, mean rate 0.1 – 10 Hz),
   compute 1-D firing-rate maps (64 bins × 2.5 cm, smoothed with a 5-cm
   Gaussian) separately for each running direction.
5. Score each cell with **Skaggs spatial information** (bits/spike). Cells
   with SI ≥ 0.5 bits/spike *and* peak rate ≥ 1 Hz in either direction are
   classified as place cells.
6. Visualise: example tuning curves, exemplar spike-on-track raster, sorted
   population rate-map heatmap, and place-field width / centre distributions.

## Key result

Of the 38 putative pyramidal CA1 cells passing the rate filter:

* **23 / 38 (60.5 %) classified as place cells** (SI ≥ 0.5 bits/spike, peak ≥ 1 Hz).
* Median place-field FWHM ≈ **25 cm** (mean ≈ 30 cm) — characteristic of CA1.
* The sorted population rate maps (`fig05_population.png` panels c–d) show
  the canonical **diagonal-stripe tiling**: place fields are distributed
  across the full 1.6 m track, with strong representation of both reward
  ends. The maps differ between left- and right-bound traversals,
  confirming the well-known direction-specificity of place fields on linear
  tracks.

These numbers match the literature on CA1 place coding in rats running linear
tracks for reward (e.g. O'Keefe & Recce 1993, Skaggs et al. 1996, Mehta et al.
2000, Grosmark & Buzsáki 2016), demonstrating place-cell firing directly from
publicly archived NWB data.

## Files

| File | Description |
|---|---|
| `place_cells_analysis.py` | End-to-end jupytext script |
| `place_cells_analysis.ipynb` | Same as `.ipynb` |
| `place_cell_metrics.csv` | Per-unit spatial info, peak rate, field width, classification |
| `fig01_position_overview.png` | Linear-track behaviour (raw position) |
| `fig02_behaviour.png` | Linearised position + velocity time-series |
| `fig03_top_place_cells.png` | Direction-specific tuning curves of the 12 most-spatially-informative cells |
| `fig04_exemplar_raster.png` | Spike-on-track raster + tuning curves for one exemplar place cell |
| `fig05_population.png` | SI distribution and sorted population rate-map heatmaps (the canonical diagonal stripe) |
| `fig06_field_widths.png` | Place-field FWHM and centre-position distributions |

## Reproducibility

Run-time on a laptop is dominated by streaming the relevant NWB chunks; with
the disk cache populated, the full pipeline executes in well under a minute.

```bash
pip install pynapple lindi remfile pynwb h5py tqdm matplotlib scipy dandi
python place_cells_analysis.py
```
