# Head-direction cells in the mouse postsubiculum (DANDI:000939)

This analysis demonstrates head-direction cells using data streamed from the DANDI
Archive. The dataset is **DANDI:000939**, "Large-scale recordings of head direction
cells in mouse postsubiculum" (Peyrache Lab, McGill University; Duszkiewicz et al.,
*Nature Neuroscience* 2024, doi:10.1038/s41593-024-01588-5). It contains 31
silicon-probe sessions from freely moving mice foraging in an open field, with head
direction tracked at 100 Hz, and 2691 sorted postsubicular units. The NWB files are
18 to 30 GB each because they include the raw broadband traces, so everything here
uses `remfile` range requests with a local disk cache and reads only the spike times
and the behavioural series. No data were downloaded in full and nothing is simulated.

Directional tuning curves and Bayesian decoding come from Pynapple, and the GLM
section uses NeMoS with a cyclic B-spline basis over head direction. Cells were
classified as direction-modulated by a circular-shift test (p < 0.01 against 1000
shifts of the spike train relative to head direction, with shifts under 20 s
excluded) combined with a split-half tuning-curve correlation above 0.5. The mean
vector length at every possible shift is computed with one FFT per unit, since it can
be written as a ratio of two circular cross-correlations; the zero-shift value was
checked against a direct recomputation of the tuning curve.

## Key finding

Across all 31 sessions, 1989 of 2691 units (74%) are significantly direction-modulated,
and 1493 of them have a mean vector length above 0.3, the range usually described as a
head-direction cell. The proportion is high because the test asks whether tuning is
present at all rather than whether it is strong: with 30 to 45 minutes of foraging,
even a broadly tuned interneuron reaches significance. Restricted to excitatory cells,
the classification agrees with the labels published with the dataset in 94% of cases
(median across sessions), and in the walk-through session it produced no false
positives against those labels. The tuning is not an artefact of the trajectory or of
burst structure: it survives circular shifts, reproduces across halves of a session,
and holds up in a Poisson GLM that competes head direction against the cell's own
spike history, where adding history on top of direction improves held-out fit only
modestly. Preferred directions tile the circle, pairwise tuning similarity falls off
with the angle between preferred directions and crosses zero near 70 to 80 degrees,
and the median tuning width at half maximum is about 40 degrees.

Two results make the case that these cells encode an internal compass rather than the
sensory scene. First, when the animal is moved to a triangular arena, every cell's
preferred direction shifts by roughly the same angle: the median coherence of the
per-cell rotations is 0.98 over the 20 sessions with two environments, with a median
residual shift of 4.5 degrees after removing the common rotation. Second, a naive
Bayesian decoder whose tuning curves are estimated on the first half of a session
recovers head direction in the held-out second half to a median error of 9.4 degrees
across sessions (chance is 90 degrees), and about 30 cells are enough to saturate that
accuracy.

## Files

| File | Contents |
| --- | --- |
| `head_direction_dandi000939.py` | Final consolidated analysis, jupytext percent format, runs end to end |
| `head_direction_dandi000939.ipynb` | The same analysis as an executed notebook |
| `fig01_raw_data_validation.png` | Head-direction trace, trajectory, occupancy, spikes plotted at the animal's heading, example polar tuning curves |
| `fig02_hd_cell_classification.png` | Circular-shift null, tuning strength vs chance, classification criteria, comparison with the published labels |
| `fig03_population_structure.png` | Sorted tuning-curve heat map, preferred directions, pairwise tuning similarity |
| `fig04_cross_environment.png` | Square vs triangle arena: coherent rotation of preferred directions |
| `fig05_decoding.png` | Bayesian decoding of head direction on held-out data, error distributions, accuracy vs population size |
| `fig06_glm.png` | NeMoS Poisson GLM fits, held-out pseudo-R², head direction vs spike history, tuning widths |
| `fig07_cross_session.png` | All 31 sessions: yield, pooled tuning strength, preferred directions, decoding, rotation coherence, label agreement |
| `all_units.csv`, `session_summary.csv` | Per-unit and per-session results |
| `stats_example_session.csv`, `glm_example_session.csv` | Walk-through session tables |
| `hd_io.py`, `hd_analysis.py`, `01`-`05_*.py` | Modular development pipeline the notebook was consolidated from |

The notebook is self-contained and does not import the development modules. Running it
takes roughly 15 minutes on a warm cache, most of which is the per-unit GLM fitting and
the pass over all 31 sessions.

## Caveats

Head direction is measured from head-mounted LEDs, so it is the orientation of the
headstage rather than of the gaze, and tracking error broadens tuning curves and sets a
floor on the decoding error. The cell counts depend on the two thresholds; the mean
vector length distribution in `fig07` is the more informative summary. Angular head
velocity is not controlled for, and postsubicular cells can be modulated by turning
speed as well as by direction, so part of the residual decoding error is likely
systematic lag rather than noise. Sessions with optogenetic manipulation contribute
only their pre-stimulation foraging epoch.
