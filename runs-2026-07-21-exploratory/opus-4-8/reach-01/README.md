# Reach direction and velocity tuning in macaque motor cortex

## Datasets

Two dandisets are used, both released through the Neural Latents Benchmark, both streamed from the
DANDI S3 bucket through LINDI with a local chunk cache. Neither file is downloaded in full.

**Primary: [DANDI:000128](https://dandiarchive.org/dandiset/000128)**, *MC_Maze: macaque primary
motor and dorsal premotor cortex spiking activity during delayed reaching* (Churchland & Kaufman).
The analysis uses the single training asset, `sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb`
(691 MB). It contains 182 sorted units from Utah arrays in M1 and PMd, 2295 successful trials of a
delayed center-out maze task, and cursor position, hand position and hand velocity sampled at 1 kHz
within trials. On 789 trials the workspace is empty and the reach is straight; on the remaining
1506 trials virtual barriers force a curved path to the same set of targets. That mix partially
decorrelates where the target is from which way the hand is actually moving at any given instant.
Analysis was restricted to the 115 units firing above 1 Hz.

**Replication: [DANDI:000129](https://dandiarchive.org/dandiset/000129)**, *MC_RTT: macaque motor
cortex spiking activity during self-paced reaching* (Makin, O'Doherty, Cardoso & Sabes), asset
`sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb` (50 MB). A different monkey, a different
laboratory, and a different task: 130 M1 units and 649 s of continuous recording while the animal
makes self-paced reaches to targets that appear at random locations, 548 targets in all at a median
of 1.1 s apart. 94 units fire above 1 Hz. This dataset was added specifically because its random
target placement breaks a confound that the center-out task cannot break, as described below.

## What was analysed

Pynapple is used for data access, interval handling and binning throughout, and NeMoS for the
Poisson encoding models.

1. **Trial-level reach-direction tuning (MC_Maze).** Per-trial firing rates in a pre-target
   baseline window, the instructed delay, and two movement windows, fitted with an ordinary
   least-squares cosine $r = b_0 + b_1\cos\theta + b_2\sin\theta$ and tested with an F test on the
   two cosine terms. The pre-target window acts as a negative control.
2. **Instantaneous velocity tuning (both datasets).** Spikes and kinematics binned at 20 ms, then
   firing rate as a function of the direction of the instantaneous hand velocity, as a 2-D map over
   $(v_x, v_y)$, and as a function of speed within each unit's preferred direction. The pairing lag
   between spikes and kinematics was swept from -400 to +500 ms.
3. **The position/velocity confound (both datasets).** A cross-validated ridge regression from a
   2-D spline expansion of hand position to hand velocity, which measures directly how much a
   position-only encoding model can get for free in each task.
4. **Poisson GLM model comparison (NeMoS `PopulationGLM`, Ridge; both datasets).** Nested encoding
   models built from a cyclic B-spline basis on movement direction, an M-spline basis on speed, and
   a tensor product of B-splines on position, scored by held-out Poisson deviance explained under
   5-fold cross-validation grouped by trial (MC_Maze) or by 5 s block (MC_RTT).
5. **Decoding (both datasets).** Cross-validated ridge regression from lagged population counts to
   instantaneous hand velocity, and, for MC_Maze, multinomial logistic decoding of the discrete
   reach-direction bin from single-epoch trial rates.
6. **A block-shift null for the per-bin significance (both datasets).** The F test used for the
   velocity-direction tuning treats 20 ms bins as independent samples, which they are not. The
   neural series is circularly shifted against the kinematics by at least 40 s, which preserves
   the autocorrelation of both signals while destroying the true pairing, and each unit is scored
   against 100 such shifts.

## Key findings

Direction tuning is present, strong, and cosine-shaped. In MC_Maze, 93% of units (107/115) are
significantly cosine-tuned for reach direction in the peri-movement window against 3% in the
pre-target baseline window, and 83% are already tuned during the instructed delay, before any
movement. The signal is recoverable at the population level: a 7-way decoder reaches 22% accuracy
on baseline activity, 59% on delay-period activity, and 81% during movement, against 14% chance.
Delay-period and movement-epoch preferred directions of the same unit are only weakly related
(circular r = 0.15), so the preferred direction of a unit in this cortex is epoch-specific rather
than a fixed property.

The tuning follows the hand, not the goal. When the barriers force a curved path to the same
targets, preferred directions shift by a median of 27 degrees relative to the straight-reach fits,
and the cosine fit is visibly worse (median $R^2$ 0.052 for curved reaches against 0.095 for
straight ones). Binning at 20 ms and pairing spikes with instantaneous velocity makes the point
directly: 111 of 115 MC_Maze units are tuned for the direction of the instantaneous velocity
vector, and the population-median tuning is strongest when spikes are paired with the velocity
about 100 ms in the future, the classic M1 lead time. The population median is the honest number
here: individual units peak over a wide range (interquartile range -100 to +240 ms across the
well-modulated half), so 100 ms describes the population and not a value each unit shares.
Speed matters too but weakly and with saturation, the population
mean rate in each unit's preferred direction roughly doubling from about 4 Hz at rest to about
8.5 Hz at the highest speeds. Hand velocity reconstructs moment by moment from the population by
simple linear decoding, with cross-validated $R^2$ of 0.59 for $v_x$ and 0.52 for $v_y$ and a
median instantaneous direction error of 21 degrees.

Every one of those velocity results replicates in MC_RTT. 75 of 94 units are tuned for the
direction of the instantaneous velocity, the tuning peaks at a neural lead of 80 ms rather than
100 ms, rate in the preferred direction grows from 4.6 Hz at rest to 10.7 Hz at the fastest speeds
with the same saturating shape, and velocity decodes with $R^2$ of 0.40 and 0.41 and a median
direction error of 20 degrees.

The replication also settles a question the primary dataset raised but could not answer. In
MC_Maze the position encoding model scored a median held-out deviance explained of 0.022 against
0.024 for the velocity model, which reads as evidence for a position code. It is not. In a
center-out task hand position predicts hand velocity with a cross-validated $R^2$ of 0.49 for $v_x$
and 0.33 for $v_y$, because every reach starts at the same centre and runs outward, so a position
model can recover the velocity code without encoding velocity. In MC_RTT, where the targets are
random, position predicts velocity with $R^2$ of essentially zero, and there the position model
collapses to 0.002 while the velocity model holds at 0.015 and wins for 77 of 94 units. The most
economical reading is that the apparent position tuning in the center-out task was task geometry
rather than neural coding. It is worth being explicit that this is a comparison across two
datasets that differ in more than target placement (different monkey, different array, different
laboratory, and a session about a tenth as long), so the inference rests on the measured mediator,
the position-to-velocity $R^2$ of 0.49 against 0.00, rather than on the dataset contrast alone. A
within-animal test, in which target placement was randomised for part of a session, would settle
it more firmly than anything available here.

The per-bin significance counts were checked rather than assumed. Firing rate and hand velocity are
both strongly autocorrelated within a reach, so an F test over tens of thousands of 20 ms bins has
many fewer independent samples than it thinks and its p-values are anticonservative. Against a null
built by circularly shifting the neural series against the kinematics, which preserves both
autocorrelations exactly, MC_Maze returns the same 111 of 115 units and the median unit's fit is
about 12 times its own null 99.9th percentile; MC_RTT returns 80 of 94, slightly more than the
nominal test gives, at about 3 times its null. Autocorrelation alone cannot produce tuning at the
observed scale.

One difference between the datasets is worth reporting rather than smoothing over. Adding speed to
the direction model improves the held-out fit for 114 of 115 MC_Maze units but for only 42 of 94
MC_RTT units, even though the MC_RTT population median still rises from 0.012 to 0.015. MC_RTT
contributes roughly a fifth as many above-threshold bins as MC_Maze, so the 40-feature direction by
speed model overfits the weakly modulated units there. The direction contribution replicates
cleanly; the speed contribution is less firmly established in the second dataset.

## Files

| file | contents |
|---|---|
| `reach_direction_and_velocity_tuning.py` | consolidated jupytext (percent format) script, runs end to end |
| `reach_direction_and_velocity_tuning.ipynb` | the same analysis as an executed Jupyter notebook |
| `fig00_raw_session.png` | population raster (units sorted by preferred velocity direction) with simultaneous cursor position, hand velocity and speed for 8 consecutive trials |
| `fig01_behavior_overview.png` | reach trajectories for straight and maze trials, speed profile aligned to movement onset, distribution of reach directions |
| `fig02_example_units_direction.png` | rasters, direction-conditioned PSTHs and polar cosine fits for three example units |
| `fig03_velocity_tuning.png` | velocity-direction tuning curves, 2-D $(v_x, v_y)$ rate maps, lag sweep, speed tuning, distribution of preferred velocity directions |
| `fig04_population_direction.png` | fraction tuned and modulation depth by epoch, preferred-direction distributions, delay vs movement PD, straight vs curved PD shift, direction decoding |
| `fig05_glm_and_decoding.png` | NeMoS GLM model comparison and cross-validated velocity decoding |
| `fig06_replication_mc_rtt.png` | MC_RTT replication: example tuning curves and velocity maps, lag and speed tuning against MC_Maze, the position/velocity confound in both tasks, GLM comparison in both tasks, velocity decoding |
| `fig07_shuffle_control.png` | per-unit observed cosine $R^2$ against the block-shift null, and the ratio of each unit's fit to its own null, for both datasets |

The numbered scripts `01_*.py` through `24_*.py` are the modular development pipeline (loading,
preprocessing, tuning analyses, GLM, decoding, figures, and the MC_RTT replication); the
consolidated script reproduces all of it in one pass. Intermediate results are cached under
`cache/`, which is populated automatically on the first run.

Running the consolidated script from scratch takes roughly 30 to 45 minutes, dominated by the two
GLM cross-validation sections. Requirements: `pynapple`, `nemos`, `lindi`, `pynwb`, `h5py`,
`scikit-learn`, `scipy`, `matplotlib`, `tqdm`, `jupytext`.

## References

- Georgopoulos, Kalaska, Caminiti & Massey (1982). On the relations between the direction of
  two-dimensional arm movements and cell discharge in primate motor cortex. *J. Neurosci.*
- Moran & Schwartz (1999). Motor cortical representation of speed and direction during reaching.
  *J. Neurophysiol.*
- Churchland, Cunningham, Kaufman, Ryu & Shenoy (2010). Cortical preparatory activity:
  representation of movement or first cog in a dynamical machine? *Neuron.*
- Makin, O'Doherty, Cardoso & Sabes (2018). Superior arm-movement decoding from cortex with a new,
  unsupervised-learning algorithm. *J. Neural Eng.*
- Pei et al. (2021). Neural Latents Benchmark '21. https://neurallatents.github.io
