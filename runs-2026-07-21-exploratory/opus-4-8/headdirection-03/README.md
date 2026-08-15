# Head Direction Cells in Mouse Postsubiculum

This analysis demonstrates head direction (HD) cells using real electrophysiology data
streamed from the DANDI Archive.

## Dataset

[DANDI dandiset 000939](https://dandiarchive.org/dandiset/000939), *"Large-scale
recordings of head direction cells in mouse postsubiculum"* (Ajabi, Peyrache and
colleagues). We used one session, `sub-A3701/sub-A3701_ses-191119` (a VGAT-Cre mouse),
which contains 102 well-isolated units (71 labelled as head-direction cells), a
head-direction angle tracked at 100 Hz, and two awake open-field foraging epochs in
geometrically distinct arenas (a square and a triangle) separated by home-cage rest. The
~30 GB file was never downloaded; it was read by streaming only the needed byte ranges
(spike times, the head-direction series, the epoch table) with `remfile` and an on-disk
cache. All analysis uses [Pynapple](https://pynapple.org).

## What was analyzed

For each unit we built a directional tuning curve (firing rate versus head direction, 60
angular bins) during square-arena foraging and quantified tuning with the mean resultant
vector length R. We then (1) contrasted labelled HD cells against the rest of the
population, (2) tested whether tuning is stable across the two arenas, and (3) decoded the
animal's head direction from HD-cell population spiking using Pynapple's Bayesian decoder
(`nap.decode_1d`), trained on the first half of the square epoch and tested on the held-out
second half.

## Key findings

The postsubicular population behaves as an internal neural compass, reproducing every
classic property of head direction cells. Labelled HD cells are sharply and unimodally
tuned to head direction (median R = 0.74, versus 0.13 for non-HD units), their preferred
directions tile the full circle, and firing rate falls off smoothly away from each cell's
preferred heading. Tuning is world-anchored rather than tied to arena geometry: between the
square and triangle arenas the entire HD ring rotates coherently by a single common offset
(here about 37 degrees, with circular concentration R = 0.96 across cells), and once that
one population rotation is removed the tuning curves are nearly identical (median per-cell
correlation rises from 0.23 to 0.94). This coherent rotation with preserved internal
structure is the signature of a ring-attractor code. Finally, the population is a readable
code: a Bayesian decoder reconstructs the true head direction from HD-cell spiking alone
with a median absolute error of about 11 degrees and a circular correlation of 0.91 on
held-out data.

## Files

- `head_direction_analysis.py` — consolidated jupytext script (runs end-to-end).
- `head_direction_analysis.ipynb` — executed notebook version.
- `fig1_raw_hd_trace_with_spikes.png` — spikes of example HD cells overlaid on the raw
  head-direction trace, showing firing clusters at each cell's preferred heading.
- `fig2_tuning_curves_polar.png` — polar tuning curves for the 12 most tuned HD cells.
- `fig3_population_summary.png` — tuning-strength distribution (HD vs non-HD), preferred-direction rose, tuning strength vs peak rate.
- `fig4_stability_across_arenas.png` — coherent ring rotation between arenas and preserved relative tuning.
- `fig5_population_decoding.png` — Bayesian decoding of head direction vs ground truth and the error distribution.

## Reproducing

```bash
python head_direction_analysis.py
# or regenerate the executed notebook:
jupytext --to notebook --execute head_direction_analysis.py
```

Requires `pynapple`, `pynwb`, `h5py`, `remfile`, `matplotlib`, `scipy`, `tqdm`.
