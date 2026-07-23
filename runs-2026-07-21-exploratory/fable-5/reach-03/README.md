# Reach direction and velocity tuning in macaque motor cortex

This analysis demonstrates directional and velocity tuning of motor-cortical
neurons during arm reaching, using data streamed from the DANDI Archive.

## Data

The primary dataset is **DANDI 000128 (MC_Maze)**: monkey Jenkins performing a
delayed reaching task, with 2295 trials, 182 sorted units from Utah arrays in M1
and PMd, and hand kinematics sampled at 1 kHz. The 789 barrier-free trials are
straight, near-center-out reaches and are used for the trial-based direction
analysis; the full session is used for the continuous velocity analysis. The
findings are replicated in **DANDI 000129 (MC_RTT)**: monkey Indy performing a
self-paced random-target task, 130 M1 units, 649 s of continuous reaching. Both
dandisets were released with the Neural Latents Benchmark (Pei et al., 2021).
Files are read over the network with `remfile` and a local disk cache; nothing
is downloaded in full. Data handling and tuning-curve computation use
`pynapple`, and the encoding models are fit with `nemos`.

## What was analysed and what was found

Reach direction was defined per trial as the angle of hand displacement over the
400 ms after movement onset, and a cosine tuning curve was fit to each unit's
firing rate in a movement window, with significance from a 1000-fold permutation
test. 152 of 182 units (84%) are significantly directionally tuned during
movement and 105 (58%) already during the delay period, before the hand moves;
preferred directions tile the workspace close to uniformly. The preparatory and
movement preferred directions are related but far from identical (median shift
67°, against 90° expected by chance), so delay activity is not a scaled preview
of the movement response.

Velocity tuning was then measured continuously: spikes binned at 20 ms, smoothed,
and regressed on the hand velocity sampled at a variable lag. The fit peaks when
neural activity leads the movement by 80 ms, the expected sign and magnitude for
a motor command, and the preferred direction recovered this way agrees with the
trial-based estimate (median difference 31°, against 90° by chance). The key
result is that speed acts as a *direction-dependent gain* rather than an
independent drive: firing rate rises with speed for movements toward the
preferred direction (mean slope +3.0 Hz per m/s) and falls for movements away
from it (-1.4 Hz per m/s), with the preferred-direction slope steeper in 138 of
159 units. Nested Poisson GLMs fit with NeMoS confirm this on held-out data: median
cross-validated pseudo-R² rises from 0.004 for a speed-only model to 0.015 for
direction alone, 0.022 for direction plus speed, and 0.029 for their
interaction, with the interaction beating the direction-only model in 157 of 159
units. The fitted surfaces also show that the depth of direction tuning grows
with speed and then saturates above roughly 500 mm/s rather than growing without
bound. The population is informative enough that ridge regression decodes hand velocity
with cross-validated R² of 0.56 (vx) and 0.55 (vy), recovering movement
direction to within 45° in 86% of moving time bins. Every effect reproduces in
MC_RTT: the same +80 ms lag, the same rising-at-PD versus flat-opposite speed
profile, and a comparable spread of preferred directions in a different animal
performing a structurally different task.

Two honest caveats. Single-unit R² for the continuous velocity model is small
(median 0.02) because restricting the analysis to movement bins removes the
largest source of rate variance, the difference between rest and movement; and
per-unit optimal lags are broadly distributed even though the population estimate
is sharp, so 80 ms is a population property rather than a per-neuron one. Also,
although the MC_Maze electrode table lists 96 PMd and 96 M1 channels, all 182
units index into the first 96 rows, so the array of origin cannot be recovered
from the file. No M1-versus-PMd comparison is attempted anywhere in this
analysis.

## Files

| File | Contents |
| --- | --- |
| `reach_direction_velocity_tuning.py` | Consolidated jupytext notebook, runs end to end |
| `reach_direction_velocity_tuning.ipynb` | The same notebook, executed |
| `reachlib.py` | Shared loading, tuning-curve, and rate-map helpers |
| `01_load_and_validate.py` … `05_decoding_and_replication.py` | Development stages |
| `fig01`–`fig09` `.png` | All figures |

Figures, in order: raw kinematics and spiking; reach geometry and speed profiles;
example units with rasters, direction-resolved PSTHs, and polar tuning curves;
population direction-tuning summary; continuous velocity fields and lag analysis;
speed gain at and opposite the preferred direction; nested GLM comparison with
GLM tuning surfaces; population decoding of hand velocity; MC_RTT replication.

## Running it

```
pip install pynapple nemos pynwb remfile h5py tqdm matplotlib jupytext
python reach_direction_velocity_tuning.py
```

The first run streams roughly 270 MB of kinematics and caches them under
`cache/`; later runs start from the cache. (The two large raw-kinematics caches
were deleted after the final run to keep this directory small; they regenerate
automatically. The small derived caches under `cache/` are the intermediate
results of the staged scripts.) The GLM cross-validation is the slow step, about
20 minutes on a laptop, and the whole notebook takes roughly 35 minutes end to
end.
