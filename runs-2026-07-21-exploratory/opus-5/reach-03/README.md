# Reach direction and velocity tuning in macaque motor cortex

This analysis demonstrates the two classic tuning properties of primate motor cortex during
reaching, using only real data streamed from the DANDI Archive.

## Data

Six sessions from three experiments released through the Neural Latents Benchmark, all read over
HTTP with `remfile` plus a local disk cache (no file is downloaded in full):

| dandiset | experiment | subject | area | task | trials |
|---|---|---|---|---|---|
| [000128](https://dandiarchive.org/dandiset/000128) | MC_Maze, full | Jenkins | motor cortex | delayed center-out and maze reaching, 2009-09-25 | 2295 |
| [000138](https://dandiarchive.org/dandiset/000138) | MC_Maze, large | Jenkins | motor cortex | same task, 2009-10-06 | 500 |
| [000139](https://dandiarchive.org/dandiset/000139) | MC_Maze, medium | Jenkins | motor cortex | same task, 2009-09-29 | 250 |
| [000140](https://dandiarchive.org/dandiset/000140) | MC_Maze, small | Jenkins | motor cortex | same task, 2009-09-28 | 100 |
| [000129](https://dandiarchive.org/dandiset/000129) | MC_RTT | Indy | M1 | self-paced random-target reaching, 2017-02-02 | continuous |
| [000127](https://dandiarchive.org/dandiset/000127) | Area2_Bump | Han | somatosensory area 2 | center-out with mechanical bumps, 2017-12-07 | 826 |

MC_Maze (000128) is the primary session: 182 sorted units and 1 kHz hand kinematics from monkey
Jenkins performing delayed reaches, 789 of them straight and barrier free. Area 2 is included
deliberately as a control, because it carries proprioceptive feedback and should therefore follow
the hand where motor cortex leads it.

## What was analysed

Direction tuning was measured by regressing each unit's single-trial firing rate over the first
350 ms of movement on the cosine and sine of the reach direction, with reach direction taken from
the actual hand displacement rather than the nominal target. Significance came from a permutation
test on the modulation depth. Velocity tuning was measured on 20 ms bins of a 30 ms Gaussian rate
across all successful trials, including the curved maze reaches, which sweep out a far wider range
of velocities than the straight ones. The neural lead time was estimated by holding the spike
window fixed and resampling hand velocity at offsets from -300 to +300 ms. NeMoS Poisson GLMs
compared three nested feature sets (direction only, velocity vector, and a 6 x 6 B-spline surface
over the velocity plane) on held-out trials, and a ridge decoder reconstructed hand velocity from
the population. Pynapple carried the data handling throughout: `TsGroup` binning, `IntervalSet`
epoching, gap-safe interpolation for the lag sweep, and `compute_tuning_curves` for the
two-dimensional velocity tuning maps.

## Key findings

130 of 150 motor cortical units are significantly cosine tuned for reach direction during movement
(permutation p < 0.01, median single-trial R² = 0.11, maximum 0.59), with preferred directions
spread around the whole circle (resultant length 0.16). The directional modulation is graded by
speed rather than all-or-none: the median modulation along a unit's preferred direction rises from
-0.03 Hz in the slowest quartile of hand speeds (14 mm/s) to 1.34 Hz in the fastest (736 mm/s), and
adding a speed-scaled term to a direction-only model improves the cross-validated fit in 123 of 150
units. Single-unit encoding of velocity is modest in absolute terms (median R² = 0.07, best unit
0.51), which is the honest number for a static velocity model of an area whose response is
dominated by condition-independent and preparatory components, but the population is emphatic: a
linear readout of all 150 units recovers hand velocity on held-out single trials with R² = 0.62 and
0.52 for the two components, and R² = 0.94 on condition-averaged velocity. Decoding accuracy rises
monotonically with the number of units, from essentially zero for one unit to 0.57 for all 150.

The timing result is the sharpest cross-dataset test. Velocity decoding peaks at a neural lead of
+75 to +100 ms in the three well-sampled motor cortex maze sessions and at +100 ms in the
independent M1 random-target session from a different monkey seven years later, but at -75 ms in
somatosensory area 2: motor cortex precedes the hand and proprioceptive cortex follows it, exactly
the sign reversal the anatomy predicts. Two results are worth flagging as caveats. The smallest
maze session (100 trials) has a flat, double-peaked lag curve whose per-fold optimal lead spans the
entire range tested, so its nominally negative lead should not be read as a real difference. And a
control that was not anticipated: split-half estimates of a unit's movement-epoch preferred
direction agree almost perfectly (R = 0.96, 98% within 45°), yet the delay-period preferred
direction of the same unit is essentially unrelated to its movement preferred direction (R = 0.13,
31% within 45°). Both epochs are directionally tuned, but they are not tuned the same way.

## Files

- `reach_direction_velocity_tuning.py` — consolidated jupytext script (percent format), runs end to end
- `reach_direction_velocity_tuning.ipynb` — the same notebook with all outputs, executed
- `dandi_io.py` — DANDI asset resolution, NWB streaming, unit conversion, local caching
- `analysis.py` — cosine fits, permutation tests, lag sweep, speed gain, condition averaging
- `01_inspect_raw.py` … `05_cross_session.py` — the staged pipeline the notebook was built from
- `fig01_raw_data_qc.png` — hand paths, speed profiles, velocity traces, raster, population rate
- `fig02_direction_tuning.png` — PSTHs by direction, polar tuning curves, population summary
- `fig03_velocity_tuning.png` — 2D velocity tuning maps, speed scaling, lag sweep, PD agreement
- `fig04_glm_decoding.png` — NeMoS GLM model comparison, fitted rate surfaces, held-out decoding
- `fig05_cross_session.png` — the six-session comparison including the area 2 sign reversal
- `results_*.npz` — intermediate results from the staged scripts

Running either the notebook or the staged scripts recreates a `cache/` directory of extracted
arrays on first use; the full pipeline takes about six minutes once the NWB chunks are cached
locally, and about fifteen on a cold cache.

Requires `pynapple`, `nemos`, `pynwb`, `remfile`, `h5py`, `scikit-learn`, `scipy`, `matplotlib`,
`tqdm`, and `jupytext` for the notebook conversion.
