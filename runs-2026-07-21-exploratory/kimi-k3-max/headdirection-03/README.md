# Head Direction Cells in the Mouse Thalamus and Postsubiculum (DANDI 000056)

This analysis demonstrates head direction (HD) cells using data from the DANDI
Archive, dandiset [000056](https://dandiarchive.org/dandiset/000056):
Peyrache, Lacroix, Petersen & Buzsáki (2015), "Internally organized mechanisms
of the head direction sense", *Nature Neuroscience* 18:569-575. The dataset
contains extracellular recordings from the anterodorsal thalamus (ADn) and
postsubiculum (PoS) of freely moving mice, together with dual-LED head tracking
and scored sleep states. Files are streamed from DANDI with `remfile` (no full
downloads) and analyzed with pynapple.

## What was done

Head direction was computed as the angle of the vector between the two
head-mounted LEDs (tracking failures, marked by -1 sentinels, were excluded).
For each unit we computed the wake tuning curve and the mean vector length
(MVL) of the head directions at spike times. Because the animal's occupancy of
directions is non-uniform, significance was assessed against a random-time null
that redraws spike times uniformly from the wake epochs and therefore preserves
the occupancy distribution; a unit was called an HD cell when its MVL exceeded
the 95th percentile of this null (p < 0.05) and cleared an effect-size floor of
MVL > 0.3. To test whether the HD representation persists without movement, we
followed the paper's central analysis: pairwise spike-train correlations among
HD cells during wake were compared with the same pairs' correlations during REM
and NREM sleep (correlation of correlations), with a cell-label permutation
null.

## Key findings

In the prototype session (Mouse17-130128), 7 of 36 units showed strong,
significant single-peaked HD tuning, the classic signature of ADn/PoS head
direction cells, and the wake correlation structure of these seven cells was
almost perfectly preserved in both REM (r = 0.95) and NREM sleep (r = 0.95),
far outside the permutation null. Across nine sessions from five mice, 96 of
262 units were identified as HD cells, their preferred directions tiled the
full circle (with a mild non-uniformity, Rayleigh p = 0.01), and the
wake-sleep preservation of the ensemble correlation structure was positive in
all nine sessions and significant for both sleep states in seven of them
(permutation p < 0.05; the two exceptions were the sessions with the fewest
HD cells). Pooled across 572 REM and 577 NREM pairs, r = 0.91 for REM and
r = 0.92 for NREM. This reproduces the paper's headline result: the head
direction signal is internally organized by the circuit itself and does not
require movement or sensory input to maintain its structure.

## Contents

- `head_direction_cells_dandi.py`: consolidated jupytext script that runs the
  full analysis end to end (single-session deep dive, then the nine-session
  population analysis).
- `head_direction_cells_dandi.ipynb`: the same analysis as a Jupyter notebook.
- `figures/`: all figures (raw tracking and HD signal, HD-cell identification,
  tuning curves, sleep preservation, population summary).
- `results_multisession/`: per-session result arrays and the session summary
  table.
- `hd_analysis.py`, `01`-`16_*.py`: development scripts used to prototype the
  pipeline; the consolidated script is self-contained and does not import them.

## Running

```
python head_direction_cells_dandi.py
```

requires `pynapple`, `pynwb`, `remfile`, `h5py`, `numpy`, `pandas`,
`matplotlib`, `tqdm`. Streaming uses a disk cache at `/tmp/remfile_cache_hd`;
a full run takes roughly 15 minutes.
