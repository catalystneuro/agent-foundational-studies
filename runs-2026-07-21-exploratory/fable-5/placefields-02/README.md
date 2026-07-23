# Hippocampal Place Cells in DANDI:000044

This analysis demonstrates hippocampal place cells using real extracellular recordings streamed
from the DANDI Archive. Nothing is simulated: every spike time and every position sample is read
from the archive at analysis time.

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), from Grosmark and Buzsáki (2016),
"Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences"
(Science 351:1440-1443). The dandiset contains eight bilateral silicon-probe recordings from
dorsal hippocampus (CA1) in four Long-Evans rats. Each session consists of a pre-run sleep epoch,
a novel-track running epoch, and a post-run sleep epoch. Five sessions use a straight track
(1.6 m or 2 m) and three use a closed circular track of roughly 2.85 m. Spikes are already sorted
and each unit carries an `excitatory` or `inhibitory` label. Files are streamed with LINDI so that
only the chunks actually read are fetched, and all analysis is done with pynapple.

## What Was Analyzed

For every excitatory unit in all eight sessions we built spatial rate maps in 4 cm bins over the
running periods, split by direction of travel on the straight tracks. Spatial selectivity was
tested by comparing each unit's Skaggs spatial information against a null built by circularly
shifting that same unit's spike train along a compressed time axis containing only the running
epochs. Because spatial information is biased upward at low spike counts, each unit is judged
against its own null rather than a global threshold. We then measured field width, peak rate and
field position, quantified within-session reliability by correlating odd-lap and even-lap rate
maps, quantified direction selectivity by correlating each cell's rightward and leftward maps,
and finally decoded position with pynapple's `decode_bayes` using tuning curves fitted on odd laps
and evaluated on held-out even laps.

## Key Finding

Of 562 excitatory CA1 units, 374 (67%) fire in a restricted portion of the track with spatial
information exceeding their own circular-shift null at p < 0.01. Median spatial information is
0.70 bits per spike against a null median near 0.20, median field width is 0.28 m at half maximum,
and median in-field peak rate is 6.4 Hz. The fields are a stable property of the cell rather than
an artifact of a few traversals: odd-lap and even-lap rate maps correlate at a median r of 0.91,
and the cell ordering derived from odd laps still produces a clean diagonal when applied to
held-out even laps. On the straight tracks the fields are strongly direction-specific, with
rightward and leftward maps of the same cell correlating at a median r of only 0.29, which
reproduces the classic directionality of linear-track place fields.

The population code is strong enough to read out directly. A naive Bayes decoder fitted on odd
laps and tested on held-out even laps localizes the animal to a median error of 4.8 cm on a 1.6 m
track, against roughly 45 cm when cell identities are permuted, with an R² of 0.95 for rightward
runs and 0.87 for leftward runs.

## Files

| File | Contents |
| --- | --- |
| `place_cells_dandi_000044.py` | Consolidated jupytext script, self-contained, runs end to end in about 30 s once the stream cache is warm |
| `place_cells_dandi_000044.ipynb` | The same notebook, already executed with outputs |
| `pf_lib.py` | Shared loading and place-field library used by the modular scripts |
| `01_load_data.py` | NWB structure inspection for one session |
| `02_validate_ratemaps.py` | Cross-check of the fast rate maps against `nap.compute_tuning_curves` |
| `03_place_fields.py` | Single-session place fields, shuffles and figures 3 to 6 |
| `04_decoding.py` | Cross-validated Bayesian decoding, figure 7 |
| `05_all_sessions.py` | All eight sessions pooled, figure 8 |
| `figures/*.png` | All eight figures |

Figures, in order: behaviour validation; running epochs and a raw spike raster; individual place
cells shown as spikes plotted at the animal's position on each traversal; population rate maps
with cross-validated ordering; spatial information against the shuffled null; field properties and
direction selectivity; decoded position on held-out laps; and replication across all eight sessions.

## Reproducing

```bash
pip install pynapple pynwb lindi h5py matplotlib scipy tqdm jupytext
python place_cells_dandi_000044.py                     # writes figures/*.png
jupytext --to notebook --execute place_cells_dandi_000044.py
```

## Notes on Method

Two details are worth flagging for anyone reusing this code.

The position `SpatialSeries` in this dandiset stores `rate = 0.02560`, but that number is the
sampling period in seconds (39.06 Hz), not a frequency. Taking it at face value as a rate would
spread 80762 samples across 36 days rather than 34 minutes. The notebook rebuilds the time base as
`starting_time + arange(n) * rate` and asserts that the result falls inside the session's
`MazeEpoch`.

The shuffle test needs a thousand rate maps per direction, so the notebook uses its own rate-map
implementation rather than calling pynapple in a loop. Section 6 of the notebook checks that
implementation against `nap.compute_tuning_curves` with smoothing switched off: the rate maps
correlate at 0.9997 with a median absolute difference of 3e-4 Hz, occupancy totals agree exactly,
and the resulting spatial information values correlate at 0.9996. Note that pynapple reports its
`occupancy` attribute in samples rather than seconds, so it is divided by the sampling rate before
comparison.

The three circular-track sessions are handled separately from the five straight-track sessions.
On a closed loop the animal runs essentially one way and position is a circular variable, so only
the dominant running direction is analyzed and the smoothing and field-width measurements wrap
around the ends. Position increments are also unwrapped onto the shortest arc before
differentiating, since otherwise the wrap point produces a spurious full-track-length velocity
spike on every lap.

Caveats: units come from the published spike sorting and the `excitatory` label is taken at face
value; place-cell counts depend on the inclusion thresholds for peak rate, spike count and alpha,
though the qualitative picture is not sensitive to them; and the reward sites at the ends of the
straight tracks mean that occupancy and field density are not uniform along the track.
