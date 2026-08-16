# Reach direction and velocity tuning in macaque motor cortex

This analysis demonstrates directional and velocity tuning of motor-cortical neurons using two
datasets streamed from the DANDI Archive. The primary dataset is
[DANDI:000128](https://dandiarchive.org/dandiset/000128) (`MC_Maze`), a delayed centre-out reaching
session from macaque Jenkins containing 182 sorted units, 2295 trials, and hand position and
velocity sampled at 1 kHz. The replication dataset is
[DANDI:000129](https://dandiarchive.org/dandiset/000129) (`MC_RTT`), a self-paced random-target
reaching session from macaque Indy with 130 sorted units and 649 s of continuous finger kinematics.
Both come from the Neural Latents Benchmark release of previously published recordings. Files are
read over the network with `remfile` and a local disk cache, so nothing is downloaded in full, and
all data handling and most of the analysis go through pynapple. The generalized linear models are
fitted with NeMoS.

The main finding is that firing rate in motor cortex is tuned to the reach *velocity vector*, not
merely to reach direction. Of the 182 MC_Maze units, 150 (82%) show significant directional
modulation of firing rate near movement onset under a permutation test, and a cosine describes that
modulation well, with a median R² of 0.72 against direction-binned mean rates. The crucial
additional result is that this directional signal is scaled by hand speed. Sorting every 50 ms bin
by hand speed and by the angle between hand direction and each unit's own preferred direction, the
normalised population rate rises from 1.34 to 1.59 times the mean as speed increases for movements
toward the preferred direction, while it falls from 0.78 to 0.57 for movements away from it. If
units coded direction alone these curves would be flat in speed. A nested set of cross-validated
Poisson GLMs makes the same point quantitatively: a general velocity model (a two-dimensional
spline over vx and vy) reaches a median pseudo-R² of 0.040, roughly double the 0.020 of a
direction-only model, and beats it in 96% of tuned units. Speed on its own is a poor predictor
(0.003), so it is the joint dependence on direction and speed, which is to say velocity, that
matters.

The timing is consistent with motor cortex driving the movement rather than reporting it. The
trial-based directional signal is strongest for a 300 ms window centred on movement onset, which is
about 125 ms before the hand reaches peak speed, and cross-validated population decoding of hand
velocity peaks when neural activity is advanced by 100 ms. Hand velocity can be decoded from the
population with an R² of 0.55 during movement, and the classic Georgopoulos population vector, which
fits nothing beyond the preferred directions, recovers reach direction to a median error of 26°.
Every one of these results replicates in MC_RTT, a different animal performing a different task:
87 of 130 units (67%) are velocity tuned under a circular-shift permutation test, the decoding lag
again peaks at +100 ms, velocity decodes at R² = 0.63 with a median direction error of 19°, and the
normalised rate again rises with speed toward the preferred direction (1.08 to 1.60) while staying
flat away from it (0.71 to 0.72).

Two results are reported as negative or weak rather than smoothed over. Although 100 units are
direction tuned during the delay period, their delay-period preferred directions do not predict
their movement-period preferred directions (circular correlation -0.22, with 43% of units within 45°
against a chance level of 25%). And the length of the population vector, sometimes claimed to track
movement speed, correlates with hand speed only weakly here (r = 0.25), even though the fitted ridge
decoder recovers speed considerably better (r = 0.40).

One data caveat is worth flagging. The MC_Maze NWB file describes two Utah arrays in its electrode
table, one in M1 and one in PMd, but every unit's electrode reference resolves into the PMd block,
with only 87 distinct electrode rows for 182 units. That is not consistent with the documented
dual-array recording, so the unit-to-area mapping in this file cannot be trusted and no analysis
here splits units by area. All 182 units are treated as one motor-cortical population.

## Files

| File | Contents |
|---|---|
| `reach_direction_and_velocity_tuning.py` | Consolidated jupytext script, runs end to end |
| `reach_direction_and_velocity_tuning.ipynb` | The same analysis as an executed notebook |
| `fig01_data_overview.png` | Raw data validation: reach paths, speed profiles, velocity traces, raster |
| `fig02_direction_tuning_examples.png` | Four example units: direction-conditioned PSTHs and polar cosine fits |
| `fig03_direction_population.png` | Population direction tuning, window scan, delay versus movement PDs |
| `fig04_velocity_tuning.png` | Lag curves, 2D velocity rate maps, speed-by-relative-direction analysis |
| `fig05_glm_model_comparison.png` | Nested Poisson GLMs fitted with NeMoS |
| `fig06_population_decoding.png` | Population vector and ridge decoding of hand velocity |
| `fig07_rtt_replication.png` | Replication in DANDI:000129 MC_RTT |

The numbered scripts `01_` through `07_` are the staged versions used during development, with
`reach_lib.py` holding the shared loading helpers. The consolidated script is self-contained and
does not depend on them.

## Running it

```
pip install pynapple nemos lindi remfile pynwb h5py scikit-learn scipy matplotlib tqdm jupytext
python reach_direction_and_velocity_tuning.py
```

The full run takes roughly fifteen minutes, most of it in the GLM fitting and the two lag scans.
Streamed byte ranges are cached under `/tmp/remfile_cache`, so repeat runs are much faster.

## Limitations

The GLMs model instantaneous kinematics only and include no spike-history term, so a unit's own
refractoriness and bursting are unmodelled. Direction sampling in MC_Maze is uneven, which leaves
two of twelve 30° bins too sparse to use and probably contributes to the non-uniform distribution
of preferred directions. Single-unit R² values on 50 ms Poisson counts are small in absolute terms
throughout, which is expected at that resolution; the lag curves are interpreted from their shape
and peak rather than their height. Most importantly, all of this is correlational. Tuning to
velocity here is a statement about what firing rate predicts, not a claim that velocity is the
variable the cortex explicitly represents.
