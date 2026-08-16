# Reach Direction and Velocity Tuning in Primary Motor Cortex

This analysis demonstrates directional and velocity (speed) tuning of single-neuron
firing rates during reaching movements, using data from
[Dandiset 000129](https://dandiarchive.org/dandiset/000129) ("MC_RTT") on the DANDI
Archive. The dataset contains a session of 130 sorted single/multi-unit recordings
from a multi-electrode Utah array implanted in macaque primary motor cortex (M1),
recorded while the animal performed a self-paced, continuous "random target" reaching
task: a cursor controlled by fingertip position was moved between targets placed
randomly on an 8x8 grid, with no inter-trial delay. The NWB file (streamed directly
from DANDI S3 storage with `remfile`, no full download) provides continuous 1 kHz
fingertip position, fingertip velocity, cursor position, and target position alongside
the spike times.

Using fingertip velocity as the behavioral variable, we (1) built nonparametric
firing-rate tuning curves against movement direction and speed with Pynapple, (2)
fit the classic linear ("cosine") velocity tuning model for every unit simultaneously
and assessed significance with a circular-shift shuffle test, and (3) fit Poisson GLM
encoding models with NeMoS, comparing a simple linear-velocity model against a more
flexible model built from cyclic B-spline (direction) and B-spline (speed) basis
functions, evaluated on held-out data.

## Key Finding

92 of 130 recorded units (71%) showed firing rates significantly modulated by
movement velocity beyond what would be expected by chance (p < 0.01, circular-shift
shuffle test). Individual units display clear cosine-shaped tuning to movement
direction together with a positive, roughly monotonic dependence of firing rate on
movement speed, consistent with the classic linear velocity tuning model described by
Georgopoulos and colleagues for motor cortex. The population of preferred directions
spans the full range of movement directions, as expected for a population encoding an
arbitrary reach direction. NeMoS Poisson GLMs fit directly to spike counts recover
matching tuning curves on held-out data, and the flexible spline-based encoding model
matches or exceeds the two-parameter cosine model's held-out pseudo-R² for most
example units, occasionally revealing tuning shapes that deviate mildly from a pure
cosine.

## Files

- `reach_direction_velocity_tuning.py` — consolidated analysis script (jupytext
  format with markdown cells), runs end-to-end from data streaming through final
  figures.
- `reach_direction_velocity_tuning.ipynb` — the same analysis as an executed Jupyter
  notebook.
- `fig01`–`fig07` `.png` — figures produced by the analysis (raw data validation,
  behavioral overview, example tuning curves with cosine fits, population summary,
  2D velocity tuning maps, and NeMoS GLM comparisons).
