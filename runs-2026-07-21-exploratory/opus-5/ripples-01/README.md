# Sharp-wave ripples and hippocampal replay in DANDI:000044

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark & Buzsáki (2016),
*Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences*
(Science 351:1440). Eight bilateral silicon-probe recordings (128 sites, 1250 Hz LFP plus
spike-sorted units, 51-137 units per session) from rat dorsal CA1 in four Long-Evans rats.
Each session runs PRE rest/sleep in the familiar home cage, then ~35-90 min of running for
water reward on a **novel** maze in a novel room, then POST rest/sleep back in the home
cage. Files were read by streaming with `remfile` and a local disk cache; no file was
downloaded whole.

## What was analysed

Sharp-wave ripples were detected on the CA1 pyramidal-layer channel with the largest
ripple-band spectral bump during non-REM sleep (130-250 Hz band-pass, Hilbert envelope,
z-scored against non-REM; peak ≥ 5 SD, boundaries at 2 SD, 20-200 ms, broadband-artefact
rejection). Place fields were estimated with `pynapple.compute_tuning_curves` over 4 cm
bins from running periods on the maze, separately for each running direction. Spiking
inside each candidate ripple was then decoded in 20 ms bins with a memoryless Bayesian
decoder against both directional templates and scored by the posterior-weighted correlation
between decoded position and time. An event counts as replay only if it beats **three**
shuffles at p < 0.05: a column-cycle shuffle of the posterior, a place-field identity
permutation followed by re-decoding, and a permutation of the time-bin order.

PRE sleep is a built-in negative control. It was recorded before the animal had ever
entered the novel maze, so no genuine replay of that maze's template can exist in it, and
whatever rate the test returns there is the rate it returns on data that cannot contain
the effect.

## Key finding

The detected events have every property that defines a sharp-wave ripple. Pooled over the
eight sessions they occur at 0.31-0.40 Hz in non-REM sleep (0.08 Hz in the one outlier
session, Buddy, which has very little scored non-REM), at essentially zero in REM, last a
median of 46-54 ms, oscillate at a median of 157-165 Hz, and confine their power increase
to 140-200 Hz within about ±30 ms of the peak. Across the depth of a shank the ripple
envelope is largest at the cell layer while the accompanying slow wave reverses polarity a
few hundred micrometres deeper, which is the stratum-radiatum current sink that gives the
sharp wave its name. CA1 pyramidal cells triple their firing rate inside ripples (median
3.4×, 95% of cells more than doubling) and interneurons roughly double theirs.

Decoding inside those ripples recovers place-cell sequences that sweep across the track at
about ten times the animal's running speed (median 9.6 m/s against a peak running speed
near 1 m/s). In the prototype session (Achilles_10252013) 6.2% of POST-sleep ripples carry
a significant sequence against 1.9% of PRE-sleep ripples (χ² = 32.6, p = 1.2 × 10⁻⁸), and
34% of awake ripples recorded while the animal sat still on the maze itself. Pooled over
the five linear-maze sessions the same ordering holds: 2.7% in POST (100/3730), 1.6% in
PRE (78/4734, χ² = 10.3, p = 1.3 × 10⁻³), and 10.9% on the maze (97/893). The awake
on-maze events are biased towards reverse order (20 reverse against 13 forward in the
prototype session) while the sleep events are not, matching the split reported for
reward-site versus offline replay.

The honest caveat is that the sleep effect is a difference between small percentages, and
it is not uniform: POST exceeds PRE in three of the five linear-maze sessions, is slightly
lower in one (Cicero_09172014, 1.3% against 1.8%), and drops to zero in Buddy, which
contributed only 20 POST candidate events against 175 in PRE. The PRE rate
of 1.6% sitting below the nominal 5% is the sign that the three-shuffle conjunction is
conservative rather than permissive; the p-value panel in `figures/06_replay_summary.png`
shows that the field-identity and time-bin nulls are close to uniform in PRE while the
commonly used column-cycle shuffle on its own is clearly liberal, which is why all three
are required.

## Files

| file | contents |
|---|---|
| `sharp_wave_ripples_replay.py` | consolidated jupytext analysis, runs end to end |
| `sharp_wave_ripples_replay.ipynb` | the same notebook, executed |
| `swr_lib.py` | streaming NWB access, LFP reading, filtering helpers |
| `analysis_lib.py` | ripple detection, place fields, Bayesian decoder and shuffles |
| `multi_session.py` | runs the pipeline over all eight sessions (`python multi_session.py`) |
| `figures/*.png` | all figures |
| `ripples.csv`, `replay_events.csv` | per-event tables for the prototype session |
| `multi_session/*.csv` | per-session ripple, rate and replay tables |

### Figures

| figure | shows |
|---|---|
| `01_session_overview.png` | population rate, raster and hypnogram across the whole session |
| `01_maze_position.png` | linearised position and speed on the novel maze |
| `01_channel_survey.png` | ripple-band spectral bump across the 128 sites |
| `02_ripple_triggered_average.png` | ripple-triggered LFP, filtered trace and wavelet spectrogram |
| `02_ripple_properties.png` | duration, frequency, amplitude, interval and rate by state |
| `03_laminar_profile.png` | depth profile of the ripple and its sharp wave |
| `03_example_ripples.png` | single events with the concurrent spike raster |
| `03_ripple_spiking.png` | ripple-triggered firing rates by cell type |
| `04_place_fields.png` | occupancy, spatial information and the place-field matrix |
| `04_example_place_cells.png` | individual fields and spike-position rasters |
| `06_replay_examples.png` | decoded posteriors and rasters for six significant replay events |
| `06_replay_summary.png` | replay rates, scores, direction, speed and null behaviour |
| `07_multi_session.png` | ripple and replay statistics across all eight sessions |

## Reproducing

```bash
python multi_session.py            # ~30 min, mostly transfer; writes multi_session/
python sharp_wave_ripples_replay.py  # regenerates every figure
```

Requires `pynapple`, `pynwb`, `remfile`, `h5py`, `scipy`, `pandas`, `matplotlib`, `tqdm`.
The LFP cache lives in `/tmp/remfile_cache_000044` and grows to roughly 13 GB across the
eight sessions; set `REMFILE_CACHE` to move it.

## Scope note

Replay is scored only for the five linear-maze sessions. Three sessions used a circular
maze, where the linearised coordinate wraps and a weighted correlation against a straight
line is the wrong statistic; those sessions contribute to the ripple statistics only.
