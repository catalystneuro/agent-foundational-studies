# Head Direction Cells in the Anterior Thalamus / Postsubiculum

## Dataset

[DANDI:000056](https://dandiarchive.org/dandiset/000056), "Internally organized
mechanisms of the head direction sense" (Peyrache lab). Mice were implanted with
silicon probes targeting the anterior thalamic nucleus (ADn) and postsubiculum
(PoSub) and recorded extracellularly while freely foraging in an open arena, with
two head-mounted LEDs (red and blue) tracked to reconstruct instantaneous head
direction. The analysis uses a single session,
`sub-Mouse24/sub-Mouse24_ses-Mouse24-131213` (the smallest session in the
dandiset, ~1.3 GB), streamed directly from the DANDI S3 bucket with `remfile`
rather than downloaded in full.

## Analysis

Spikes, LED positions, and behavioral-state epochs (Awake / Non-REM / REM) were
loaded with Pynapple. Head direction was computed as the angle of the vector from
the blue LED to the red LED, restricted to waking behavior. Circular tuning
curves were computed for each of the 21 active units (one unit had zero spikes
for the whole session and was excluded), and directional selectivity was
quantified with the Rayleigh mean resultant vector length, tested for
significance against 500 circular shuffles of the head-direction trace. Units
passing a shuffle-test threshold of p < 0.01 were treated as head-direction
cells. Their tuning curves, fit on the first half of the waking epochs, were
then used to Bayesian-decode head direction on the second (held-out) half of the
session, to test whether the population as a whole carries a coherent
representation of heading.

## Key Finding

9 of 21 units showed statistically significant circular tuning to head
direction (shuffle test, p < 0.01), with resultant vector lengths ranging from
weak (R ≈ 0.1) to very strong (R ≈ 0.8) and preferred directions tiling the
full 360-degree range (see `figures/02_polar_tuning_curves.png` and
`figures/04_population_heatmap.png`). Bayesian decoding using only these 9
units reconstructed the animal's instantaneous heading on held-out data with a
median absolute error of 25.3 degrees and a circular mean resultant length of
0.76 for the error distribution, well above what uniform-random guessing would
produce (`figures/05_decoding.png`). The decoder tracked both slow drifts and
large, rapid head turns in the held-out data, confirming that this subset of
units collectively encodes a continuously updated representation of head
direction, the defining property of head direction cells.

## Files

- `head_direction_analysis.py` — jupytext (percent-format) analysis script, runs
  end-to-end with no manual intervention.
- `head_direction_analysis.ipynb` — the same analysis as an executed Jupyter
  notebook.
- `figures/01_raw_data_overview.png` — spike raster, LED trajectory, and
  head-direction trace for an example 60 s awake window.
- `figures/02_polar_tuning_curves.png` — polar tuning curves for the 9
  significant head-direction cells.
- `figures/03_R_vs_pvalue.png` — tuning strength (R) vs. shuffle-test p-value
  for all units.
- `figures/04_population_heatmap.png` — normalized firing rate vs. head
  direction for all units, sorted by preferred direction.
- `figures/05_decoding.png` — Bayesian-decoded vs. true head direction on
  held-out data, and the decoding error distribution.
