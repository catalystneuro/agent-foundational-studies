# Theta phase entrainment and phase precession of CA1 place cells (DANDI:000044)

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), "Diversity in neural
firing dynamics supports both rigid and learned hippocampal sequences"
(Grosmark & Buzsáki 2016, *Science*), better known as the `hc-11` data set.
Eight sessions from four rats (Achilles, Cicero, Gatsby, Buddy) with bilateral
silicon-probe recordings from dorsal CA1 while the animal ran for water reward,
each session flanked by pre- and post-run sleep. Each NWB file provides sorted
spike times with an excitatory/inhibitory label, a 128-channel LFP at 1250 Hz,
and 2D position tracking at 39 Hz.

Five of the eight sessions use a linear runway (1.6 m or 2 m) and three use a
circular ring track, which is not apparent from the epoch labels (all read
`MazeEpoch`) and only becomes clear on plotting the raw trajectories
(`fig07_maze_geometry.png`). The linearization used here projects the 2D track
onto its principal axis, which folds a ring back on itself, so the three
ring-track sessions (Achilles_11012013, Cicero_09102014, Gatsby_08282013) are
excluded and the five linear-track sessions are analysed. Data were streamed
from the archive with LINDI and a local cache; the session files are 5-9 GB each
and roughly 200 MB per session was actually read (one LFP channel per shank over
the maze epoch, plus spike times and position).

## What was analysed

The maze epoch was restricted to periods of running faster than 5 cm/s, split
into rightward and leftward traversals because CA1 fields on a linear track are
direction-selective. The LFP channel with the largest theta/delta power ratio
during the maze epoch was used as the phase reference; it was bandpass-filtered
at 6-10 Hz and the Hilbert phase taken at every spike time.

- **Entrainment** was quantified with the Rayleigh test on each unit's running
  spike phases, together with the mean resultant length and preferred phase.
- **Place fields** were defined per direction as contiguous runs of the rate map
  above 20% of the peak, 15 cm to 1 m wide, with a peak rate of at least 1 Hz
  and at least 0.5 bits/spike of Skaggs spatial information.
- **Precession** was quantified with the circular-linear regression of Kempter
  et al. (2012), regressing theta phase on the fraction of the field already
  traversed. A within-field phase permutation provides the null.

## Key finding

Both phenomena are clearly present. Across the five linear-track sessions,
299 of 369 (81%) of CA1 units with at least 100 running spikes fired
non-uniformly across the theta cycle (Rayleigh p < 0.01), with interneurons more
strongly locked than pyramidal cells, and the pyramidal population shared a
common preferred phase visible as a clear modulation of the pooled spike-phase
histogram. Within place fields, theta phase fell systematically as the animal
advanced: 114 of 223 (51%) of fields reached significance individually
(p < 0.05 with a negative slope), 83% of all fields had a negative
slope, and the median sweep was -177 degrees of theta phase per field
traversal, in line with the classic reports of roughly half a cycle to a full
cycle. Individual fields reach circular-linear correlations of -0.4 to -0.6, and
pooling spikes from every field in every session reproduces the canonical
precession band running from late to early phase across the normalised field.

The permutation control makes clear that this is not an artefact of the
regression: permuting phases within a field gives a null distribution of
correlations tightly centred on zero, while the observed distribution is shifted
well to the negative side. One methodological caveat is worth stating: fields
were taken from the pooled direction-specific rate map rather than lap by lap,
and spikes were pooled across laps, so lap-to-lap variability in where a field
begins adds phase jitter. The reported correlations are therefore, if anything,
a lower bound on the strength of precession within single traversals.

## Files

| File | Contents |
| --- | --- |
| `theta_phase_precession_hc11.py` | Consolidated jupytext script, runs end to end |
| `theta_phase_precession_hc11.ipynb` | The same as an executed notebook |
| `hc11.py` | Shared streaming / preprocessing / circular-statistics helpers |
| `analysis_core.py` | Per-session pipeline (`analyze_session`, `precession`) |
| `01_load_data.py` | Initial inspection of one NWB file |
| `02_preprocess.py` | LFP channel selection, position, speed, run epochs |
| `03_analyze_theta.py` | Place fields, phase locking, precession for one session |
| `04_visualize.py` | Figures 2-5 |
| `05_multi_session.py` | Repeats the pipeline over the five linear-track sessions (figure 6) |
| `fig01_preprocessing.png` | Behaviour, LFP spectra per shank, theta filter output |
| `fig02_place_fields.png` | Occupancy, sorted field maps per direction, spatial information |
| `fig03_theta_entrainment.png` | Raw LFP with place-cell raster, phase histograms, population locking |
| `fig04_precession_examples.png` | Six single fields with rate map and phase-position regression |
| `fig05_precession_population.png` | Pooled phase-position density, slopes, shuffle control |
| `fig06_across_sessions.png` | Both effects reproduced in all five linear-track sessions |
| `fig07_maze_geometry.png` | Raw 2D trajectories showing which sessions are ring tracks |

## Reproducing

```
pip install pynapple lindi pynwb h5py matplotlib scipy tqdm jupytext
python theta_phase_precession_hc11.py
```

Set `RUN_ALL_SESSIONS = False` in the script to analyse only the prototyping
session (Achilles_10252013) and skip the cross-session replication.

## Notes on this NWB conversion

Two quirks of the conversion were worked around and are worth recording:

1. `SpatialSeries.rate` holds the sampling *period* (0.0256 s) rather than a
   rate in Hz, so timestamps have to be rebuilt explicitly.
2. The stored `LinearizedPosition` series is NaN for about 87% of the maze
   epoch, being filled in only during identified traversals. The 2D series is
   ~79% valid, so position was linearized here by projecting the 2D track onto
   its principal axis. Where both are defined the two agree at r = 0.99999.

Separately, differentiating position with pynapple's `smooth` / `derivative` on
a series whose time support has gaps bridges those gaps and manufactures
apparent speeds above 2 m/s. Velocity is therefore computed within each tracked
interval separately.
