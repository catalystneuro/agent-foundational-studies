# Reach direction and velocity tuning in monkey motor cortex

This analysis demonstrates directional and velocity tuning of motor cortical neurons using
real data from the DANDI Archive.

## Dataset

[DANDI:000128](https://dandiarchive.org/dandiset/000128) — *MC_Maze*, the Neural Latents
Benchmark release of the Churchland / Kaufman / Shenoy maze-reaching recordings. The
analysis uses one session, the full training file from monkey Jenkins
(`sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb`, recorded 2009-09-25):
2295 successful reaches, 182 sorted units recorded simultaneously from two Utah arrays in
primary motor cortex (M1, 90 units) and dorsal premotor cortex (PMd, 92 units), and hand
position and velocity sampled at 1 kHz. The file is streamed from the DANDI S3 bucket with
`remfile` and a local disk cache; nothing is downloaded in full. The session is held as
Pynapple objects (`TsGroup`, `TsdFrame`, `IntervalSet`), with spike binning and epoch
restriction done through Pynapple; the GLMs use NeMoS.

The maze task suits this question because the barriers force curved reaches, so the
direction the hand actually travels is not simply a function of which target was cued.
Reach direction and speed are therefore always taken from the *measured* hand velocity,
never from the target position.

## What was analysed

Three converging analyses of the same session:

1. **Trial-based direction tuning.** Spike counts in a window from 100 ms before to 150 ms
   after movement onset were fitted with a cosine tuning curve over the measured reach
   direction, with significance from a 1000-fold permutation test on the modulation depth.
   Trials were then split into speed terciles and the cosine refitted within each, which
   separates a pure direction code (three identical curves) from a velocity code (same
   preferred direction, larger amplitude at higher speed).
2. **Continuous velocity tuning.** Spikes binned at 20 ms across the whole session were
   paired with the mean hand velocity a fixed lag later, and regressed on the velocity
   vector. Sweeping that lag measures how far the neural signal leads the hand, and binning
   the rate over the $(v_x, v_y)$ plane gives a model-free picture of the velocity field.
3. **Poisson GLMs and population decoding.** Four nested encoding models (direction only,
   linear velocity, velocity plus speed, and a 5×5 B-spline velocity field) were fitted in
   NeMoS and scored by held-out McFadden pseudo-$R^2$ over five contiguous time blocks. A
   ridge decoder then reconstructed hand velocity from the population, and the classical
   Georgopoulos population vector was computed for comparison.

## Key findings

**Units are cosine-tuned to reach direction.** 154 of 182 units (85%) show significant
directional modulation by permutation test at p < 0.01 (M1 70/90, PMd 84/92), with a median
modulation depth of 1.3 Hz. Preferred directions are spread around the full circle rather
than clustering on a few target locations, and the preferred directions recovered from the
trial-averaged analysis and from the independent continuous 20 ms-bin analysis agree to a
median of 15°.

**The tuning is a function of velocity, not direction alone.** Splitting trials by speed
leaves the preferred direction essentially where it was (median shift 32°, against 90° if
the two estimates were unrelated) while increasing the amplitude of the cosine from 1.37 to
1.73 Hz between the slow and fast terciles (Wilcoxon p = 2 × 10⁻⁵); the direction-independent
baseline does not move. The same structure appears model-free in the two-dimensional rate
maps, where firing rate rises smoothly and monotonically with speed along a unit's preferred
direction and stays flat or falls in the opposite direction. In the cross-validated GLM
comparison, a model given both direction and speed beats a direction-only model by roughly
a factor of four in pseudo-$R^2$ (0.0103 versus 0.0025, p = 5 × 10⁻²³, improving 150 of 182
units), and a fully nonlinear velocity field does slightly better still (0.0119). Notably,
a GLM whose *log* rate is linear in $(v_x, v_y)$ does not beat the direction-only model:
the exponential link makes it predict exponential growth with speed, which the data do not
show. The velocity dependence is real but saturating.

**The population encodes velocity, about 100 ms ahead of the hand.** Sweeping the lag
between spikes and kinematics puts the peak of the velocity-model fit at a neural lead of
100 ms. A ridge decoder using 200 ms of population history reconstructs hand velocity on
held-out data with R² = 0.53 for both components, recovers reach direction to a median error
of 17°, and tracks hand speed at r = 0.82. The Georgopoulos population vector recovers
direction about as well (median error 15°, 95% of trials within 45°), but its length is a
poor readout of speed (r = 0.07), which is what one would expect from a vote-based readout
that normalises away each unit's gain.

## Files

| file | contents |
|---|---|
| `reach_direction_velocity_tuning.py` | consolidated jupytext script, runs end to end |
| `reach_direction_velocity_tuning.ipynb` | the same, as a notebook |
| `fig01_raw_data.png` | hand position, velocity, speed and the full raster over eight trials |
| `fig02_behavior.png` | reach trajectories, direction coverage, speed distribution |
| `fig03_example_tuning.png` | polar tuning curves with cosine fits, split by speed |
| `fig04_population_direction.png` | population direction tuning and the speed-scaling test |
| `fig05_velocity_maps.png` | rate maps over the $(v_x, v_y)$ plane and speed tuning curves |
| `fig06_velocity_population.png` | lag scan, preferred-direction agreement, velocity gains |
| `fig07_glm_decoding.png` | GLM model comparison and population decoding |
| `results_direction_tuning.csv` | per-unit cosine fit, preferred direction, permutation p |
| `results_velocity_model.csv` | per-unit velocity gain, preferred direction, best lag |
| `results_glm_pseudo_r2.csv` | per-unit cross-validated pseudo-$R^2$ for the four GLMs |
| `common.py`, `01_`–`04_*.py` | the modular development pipeline the notebook was built from |

The consolidated script caches the extracted arrays in `mc_maze_cache.pkl` on first run, so
re-running it does not re-stream the NWB file. With that cache in place a full run takes
about 5 minutes, most of which is the cross-validated GLM fitting; the first run adds the
time to stream what it needs out of the 0.69 GB asset.

## Caveats

The analysis covers a single session from a single animal, so the population-level numbers
describe this recording rather than the species. The M1 versus PMd labels rest on
reconstructing the unit-to-electrode mapping by hand: `units/electrodes` in this file stores
within-array channel numbers rather than global electrode rows, and PyNWB does not resolve
the `DynamicTableRegion` correctly, so the array boundary is recovered by detecting the
single reset in the channel sequence. That reconstruction is documented in the notebook and
was checked against the file, but none of the direction or velocity conclusions depend on
it. Single-bin $R^2$ values for the 20 ms continuous fits are small in absolute terms
because most bins contain zero or one spike; the cross-validated GLM pseudo-$R^2$ is the
meaningful goodness-of-fit measure, and it is reported alongside.

Three smaller points about how the held-out numbers were obtained. The ridge penalty for
the velocity decoder is picked from a four-point grid by the same held-out folds that the
reported $R^2 = 0.53$ comes from, so that figure carries a little selection optimism. The
three smallest of the four penalties (10, 10², 10³) give held-out $R^2$ within 0.005 of
each other and only the largest (10⁴) is clearly worse at 0.47, so the choice barely
matters here, but it is still not a clean nested estimate. The GLM design matrices are
standardised using the mean and standard deviation of the whole session rather than of
each training fold, which leaks feature scaling (not the spike counts) across the split.
The Georgopoulos population vector is computed with preferred directions fitted on the
same trials it is then evaluated on, so its 15° median error is an in-sample number and
is not directly comparable to the decoder's cross-validated 17°; the point of that
comparison is the contrast in *speed* readout (r = 0.82 versus r = 0.08), which does not
depend on the difference.
