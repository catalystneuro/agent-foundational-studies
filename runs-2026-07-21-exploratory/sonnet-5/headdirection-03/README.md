# Head Direction Cells in ADn/PoS (DANDI:000056)

This analysis demonstrates the classic head-direction (HD) cell phenomenon using data
from [DANDI:000056](https://dandiarchive.org/dandiset/000056), "Internally organized
mechanisms of the head direction sense" (Peyrache & Buzsaki lab). The dataset contains
silicon-probe recordings from the antero-dorsal thalamic nucleus (ADn) and the
post-subiculum (PoS) of freely foraging mice, with two head-mounted LEDs tracked
overhead to reconstruct instantaneous heading. We used session
`sub-Mouse24/sub-Mouse24_ses-Mouse24-131213_behavior+ecephys.nwb`, streamed directly
from S3 with `remfile` (no full download), and analyzed it with Pynapple.

Head direction was computed as the angle of the vector between the two tracked LEDs,
restricted to the longest continuous "Awake" epoch (the ~41-minute open-field foraging
session). For each of the 22 recorded units we computed an occupancy-normalized
tuning curve as a function of head direction, quantified tuning sharpness with the
Rayleigh vector length, and verified stability by correlating tuning curves computed
on independent first/second halves of the session. Three units (of 22) met both a
sharpness criterion (Rayleigh r > 0.3) and a stability criterion (split-half
correlation > 0.5), clearly separated from the rest of the population, which showed
flat or unstable tuning. These three cells had distinct preferred directions spanning
the full 360 degrees. Using only this small ensemble, Bayesian population decoding
(`pynapple.decode_bayes`) reconstructed the animal's held-out heading with a median
absolute circular error of about 48 degrees, versus the roughly 90 degrees expected by
chance, confirming that the identified units carry a genuine, temporally stable
directional signal consistent with the head-direction cell phenomenon originally
described in ADn and PoS.

## Files

- `head_direction_analysis.py` — jupytext-formatted analysis script (source of truth)
- `head_direction_analysis.ipynb` — executed notebook, converted from the script
- `fig01_raw_data_overview.png` — behavioral states, raw LED tracking, raw spike raster
- `fig02_head_direction_trace.png` — computed head direction over a 60 s window
- `fig03_polar_tuning_curves.png` — polar tuning curves for all 22 units
- `fig04_tuning_strength_summary.png` — Rayleigh vector / split-half reliability summary
- `fig05_ensemble_raster_vs_heading.png` — HD-cell ensemble raster vs. instantaneous heading
- `fig06_decoding_performance.png` — Bayesian-decoded vs. actual heading, error distribution
