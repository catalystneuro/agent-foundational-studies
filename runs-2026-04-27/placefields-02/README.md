# Hippocampal place cells on a 1.6 m linear track (DANDI 000044)

## Dataset

[**DANDI:000044**](https://dandiarchive.org/dandiset/000044) — Grosmark, Long &
Buzsáki, *Diversity in neural firing dynamics supports both rigid and learned
hippocampal sequences*. Bilateral silicon-probe recordings from CA1 in four
Long-Evans rats. Each session contains a long PRE-sleep epoch, a ~45-min novel
running epoch on a 1.6 m linear maze, and a POST-sleep epoch.

This analysis uses the **Buddy 06/27/2013** session
(`sub-Buddy_ses-Buddy-06272013_behavior+ecephys.nwb`, ~5.2 GB) and streams it
on demand from the DANDI S3 bucket via `remfile` with a local disk cache —
nothing is downloaded in full.

## Pipeline

The single script [`place_cells_analysis.py`](place_cells_analysis.py)
(jupytext / `.ipynb` is also provided) runs end-to-end:

1. Stream the NWB file with `remfile` + `pynapple.NWBFile`.
2. Restrict to the maze epoch and reconstruct correct position timestamps
   (the file stores the sampling **period** in the `rate` slot, so Pynapple's
   default loader yields nonsensical timestamps; we fix that).
3. Build an `IntervalSet` from the contiguous runs of valid linearized-position
   samples — these correspond to traversals of the central linear arm at
   ~58 cm/s median speed; the rat's brief excursions onto the U-turn ends are
   automatically excluded.
4. Build per-sample (≈ 39 Hz) spike-count vectors, an occupancy histogram on
   50 bins along the track, and from these the per-cell **firing-rate maps**.
5. Compute **Skaggs spatial information** (bits/spike) for every pyramidal
   unit and a circular-shuffle null distribution by rolling the per-sample
   spike-count vector relative to the position vector (500 shuffles/cell).
6. Tag a unit as a place cell when its bits/spike exceeds the 99th-percentile
   shuffle and it has peak rate ≥ 1 Hz with ≥ 50 track-running spikes.
7. Visualize: behavior + raster, sorted population sequence, top tuning
   curves, bits/spike vs shuffle, spike-on-trajectory rasters.

## Key result

Out of 48 putative CA1 pyramidal cells, **18 (37.5%) are place cells** by the
above criterion. Their tuning curves tile the 1.6 m track and form the
classical diagonal "sequence" pattern when sorted by peak position
(`fig02_population_sequence.png`). Top examples have narrow fields with peak
rates of ~3–18 Hz and 1–3 bits/spike, well above the shuffle null
(`fig03_top_place_cells.png`, `fig04_spatial_info_distribution.png`). Spikes
plotted on top of the rat's trajectory show the location-locked firing
directly (`fig05_spikes_on_trajectory.png`). This reproduces the canonical
hippocampal place-cell phenomenon (O'Keefe & Dostrovsky 1971; Skaggs et al.
1993) on real DANDI data with a streaming-only workflow.

## Files

| File | Contents |
| --- | --- |
| `place_cells_analysis.py` | jupytext source — runs end-to-end |
| `place_cells_analysis.ipynb` | the same script as a Jupyter notebook |
| `place_cell_metrics.csv` | per-unit bits/spike, shuffle stats, peak rate, place-cell flag |
| `fig01_behavior_and_spikes.png` | maze trajectory, linearized position, full pyramidal raster |
| `fig02_population_sequence.png` | place-cell tuning curves sorted by peak position |
| `fig03_top_place_cells.png` | smoothed tuning curves of the top-9 place cells |
| `fig04_spatial_info_distribution.png` | bits/spike for real cells vs shuffle null |
| `fig05_spikes_on_trajectory.png` | spike-on-trajectory rasters for the top-4 place cells |

## Reproducing

```bash
pip install pynapple lindi remfile pynwb h5py tqdm matplotlib scipy pandas
python place_cells_analysis.py
```

The first run streams ~tens of MB into `/tmp/dandi_cache`; subsequent runs are
fast because the chunks are cached on disk.
