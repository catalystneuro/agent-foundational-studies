# The Head-Direction System as a Continuous Ring Attractor, Maintained During Sleep

## Dataset

[DANDI:000056](https://dandiarchive.org/dandiset/000056), *Internally organized
mechanisms of the head direction sense* (Peyrache, Lacroix, Petersen & Buzsáki,
CC-BY-4.0). Extracellular silicon-probe recordings from the anterior thalamus and
post-subiculum of freely moving mice, with open-field foraging interleaved with long
sleep sessions in a rest box and sleep states already scored as awake, REM or non-REM.
Six sessions from five animals were analysed (303 sorted units, 104 head-direction
cells, 3.5 h of foraging, 1.7 h of REM and 18.6 h of non-REM). Files were streamed
from S3 with LINDI and cached locally; nothing was downloaded in full. All analysis
uses Pynapple for data access, epoching, tuning curves, decoding and correlograms.

Head direction is not stored in these files and was reconstructed from the two
head-mounted LEDs. The open-field blocks were identified from the spatial extent
covered, and because the map can rotate between visits hours apart, tuning curves were
estimated from a single visit rather than pooled.

## What Was Analysed

The question is not whether these cells are directionally tuned, which is already
known, but whether the population is organised as a ring attractor and whether that
organisation is intrinsic. Two predictions were tested. First, that the set of
population states is one-dimensional and closed rather than filling the space
available to *N* independently tuned cells. Second, that the structure survives into
sleep, when the animal is motionless and has no sensory evidence about its heading.

The main lines of evidence are the pairwise correlation structure as a function of the
offset between preferred directions; internal tuning curves, in which an angle decoded
from half the head-direction cells is used to score the other half; Isomap embeddings
of the population state with a cell-identity-shuffled control; Bayesian decoding of
the internal angle and its step-size and autocorrelation; and cross-correlograms of
similarly and oppositely tuned pairs, which use spike times directly and involve no
decoding.

## Key Finding

The population behaves as a continuous ring attractor, and the ring is maintained
internally during sleep. Sorting head-direction cells by preferred direction turns the
pairwise correlation matrix into a band along the diagonal that wraps around at the
corners, positive out to roughly one tuning width and negative beyond it. The same
matrix computed during REM and non-REM matches the wake matrix pair by pair, with
r = 0.91 to 0.98 across all six sessions. Isomap returns a closed ring whose angular
coordinate tracks head direction (ring-angle match 0.93 in wake, 0.86 in REM and 0.69
in non-REM, against 0.02 to 0.13 for the cell-shuffled control), with residual
variance that drops sharply at two embedding dimensions and then stops improving, and
with a radial spread roughly half that of the shuffled control. Cells scored against
an angle decoded entirely from other cells still fire at their wake preferred
direction with close to their wake tuning shape while the animal is asleep (shape
correlation 0.79 in REM and 0.85 in non-REM, median across sessions), and predicting a
held-out cell from another cell's tuning curve at the same decoded angle gives
essentially zero, which rules out a common population-rate fluctuation as the
explanation. The decoded angle moves in small steps rather than jumping, and at 40 ms
resolution it can be watched sweeping continuously around the ring during non-REM,
passing through 0/360 without a break.

What changes between states is the timescale, not the structure. Cross-correlograms of
similarly tuned pairs have a half-width of 1.5 to 1.8 s in wake and 1.2 to 2.9 s in
REM, but only 0.08 to 0.12 s in non-REM, so the same trajectory on the same ring runs
about fifteen times faster during slow-wave sleep. Two other departures are also
consistent with the model rather than against it. The map sometimes rotates as a
rigid whole, by about 50 to 120 degrees between visits to the open field hours apart
and, in one session, by 90 degrees within a single visit, with a residual scatter of
only 2 to 20 degrees around the common rotation. And during non-REM down states the
population falls silent, so the ring coordinate is briefly undefined; those bins were
excluded from the manifold analyses. Both are what one expects of an internally
maintained ring that has lost its anchoring to the world rather than lost its shape.

The two sessions with only seven head-direction cells (Mouse24-131216 and
Mouse17-130204) undersample the circle, and the measures that need full coverage
(absolute decoding, ring geometry, internal tuning shape) are correspondingly weaker
there. The correlation-structure result, which does not require full coverage, is at
its strongest in exactly those sessions.

## Files

| File | Contents |
| --- | --- |
| `hd_ring_attractor_sleep.py` | Self-contained jupytext script, runs end to end |
| `hd_ring_attractor_sleep.ipynb` | The same notebook with outputs, executed |
| `session_summary.csv` | Every summary statistic, one row per session |
| `fig01_dataset_overview.png` | Session structure, trajectory, head direction, raster sorted by preferred direction |
| `fig02_tuning_curves.png` | Polar tuning curves and the shuffle-based HD-cell criterion |
| `fig03_tuning_stability.png` | Within-visit stability, rigid rotation, cross-validated decoding error |
| `fig04_correlation_structure.png` | Correlation matrices and offset profiles in wake, REM and non-REM |
| `fig05_internal_tuning.png` | Internal tuning curves and split-half prediction of held-out cells |
| `fig06_ring_manifold.png` | Isomap rings with cell-shuffled controls |
| `fig07_dimensionality.png` | Residual variance, PCA spectrum, radial spread, ring-angle match |
| `fig08_bump_dynamics.png` | Decoded posteriors, step sizes, angular autocorrelation |
| `fig09_sleep_sweeps.png` | 40 ms decoding inside the longest REM and non-REM bouts |
| `fig10_timescale.png` | Cross-correlograms and the non-REM time compression |
| `fig11_multisession_summary.png` | All six sessions on every measure |

## Reproducing

```bash
python hd_ring_attractor_sleep.py          # or open the notebook and run all cells
```

Requires `pynapple`, `lindi`, `pynwb`, `h5py`, `scikit-learn`, `scipy`, `pandas`,
`matplotlib` and `tqdm`. The first run streams roughly 1 GB of chunks from the DANDI
S3 bucket into `cache/lindi` and takes about 20 minutes; later runs work from the
cache and take about 5 minutes. Figures are written with the Agg backend and are never
displayed.
