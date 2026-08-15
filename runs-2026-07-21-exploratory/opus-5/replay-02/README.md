# Hippocampal replay during sharp-wave ripples

Decoding the spatial trajectory represented during sharp-wave ripple (SWR) events in rat dorsal CA1,
using real recordings streamed from the DANDI Archive.

## Dataset

[**DANDI:000044**](https://dandiarchive.org/dandiset/000044) — Grosmark, A.D. and Buzsáki, G. (2016),
*Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences*,
Science 351, 1440–1443. Bilateral silicon-probe recordings from dorsal CA1 in freely moving rats.
Each session runs several hours of sleep (PRE), ~35 min of running on a track (MAZE), and several
more hours of sleep (POST), and provides spike-sorted units with excitatory/inhibitory labels,
linearised position during track traversals, 128-channel LFP at 1250 Hz, and curated
Awake / Non-REM / REM state annotations.

Four sessions from three rats were analysed. `Achilles_10252013` (1.6 m linear maze) is used for the
detailed walk-through; `Achilles_11012013`, `Cicero_09102014` and `Gatsby_08282013` (2.8–2.9 m
circular mazes) serve as replications. Files are 5–9 GB each and are read by streaming byte ranges
with `remfile` plus a local disk cache; nothing is downloaded in full.

## What was analysed

1. **Place fields.** Direction-specific tuning curves along the linearised track, built with Pynapple
   from the traversals where tracking is defined. Cells with peak rate > 1 Hz and > 0.5 bits/spike of
   spatial information in at least one direction were kept.
2. **Decoder validation.** A flat-prior Bayesian decoder was trained on every second traversal and
   used to decode the animal's real position on the held-out laps.
3. **Ripple detection.** The channel with the highest 140–230 Hz power relative to a 300–500 Hz noise
   band was selected from all 128, and its ripple envelope was z-scored against Non-REM sleep. Events
   crossed 1 SD, peaked above 4 SD, and lasted 20–250 ms.
4. **Candidate replay events.** Multi-unit population bursts (50–600 ms, peaking above 3 SD of the
   smoothed pyramidal-population rate) that overlap a detected ripple.
5. **Replay scoring.** Position was decoded in 20 ms bins inside each event, once per direction
   template, and scored by the weighted correlation between decoded position and time. Significance
   was assessed against two shuffles at p < 0.05 each: a place-field identity shuffle (the
   unit → place-field assignment is permuted) and a posterior column-cycle shuffle (each time bin's
   posterior is circularly shifted in position). A calibration control repeated the entire analysis
   with a single permuted template, which carries no genuine position code.

## Key finding

The decoder is accurate on behaviour: on held-out running laps it recovers the animal's position to a
median error of 5.4 cm on the 160 cm linear track (6.5–8.4 cm across the other sessions), against a
chance level near 50 cm. Applied to the population bursts that accompany sharp-wave ripples in
slow-wave sleep, the same decoder produces posteriors that repeatedly sweep smoothly across a large
part of the track within 100–200 ms (`figures/04_replay_examples.png`). In the walk-through session
the median replayed trajectory covers 79 cm of the 160 cm track at a virtual speed of 6.3 m/s, about
ten times the animal's own mean running speed of 0.64 m/s, and forward and reverse sweeps occur in
comparable numbers. That is hippocampal replay: the spike sequence inside a single ~100 ms ripple
recapitulates a spatial trajectory the animal ran while awake, compressed roughly tenfold.

Statistically, POSTRUNFRAC of ripple-associated bursts in the sleep following track running beat both
shuffle nulls, against PREFRAC in the sleep that preceded it and CTRLFRAC for the permuted-template
control (pooled across the four sessions: POST vs control PPOST, POST vs PRE PPP). The control
landing on the nominal 5% shows the test is calibrated, so the excess in POST sleep comes from the
correspondence between specific cells and specific places rather than from the coarse firing
statistics of ripples. Replay is a minority of events, which is expected: short bursts with few
active cells carry too little information to score, and the detection rate rises steeply with the
number of participating cells (`figures/05_replay_statistics.png`, "detection vs. event size").

## Files

| file | contents |
| --- | --- |
| `hippocampal_replay_analysis.py` | consolidated jupytext script, runs end to end |
| `hippocampal_replay_analysis.ipynb` | the same as a notebook |
| `common.py`, `pipeline.py` | streaming loader and reusable analysis functions |
| `01_load_data.py` … `07_cross_session.py` | the modular staged pipeline |
| `figures/01_raw_data_overview.png` | session structure, position, spike raster, raw LFP |
| `figures/02_ripple_detection.png` | channel selection, example ripple, ripple-triggered averages |
| `figures/03_place_fields.png` | place fields, spatial information, cross-validated decoding |
| `figures/04_replay_examples.png` | twelve replay events: posterior and place-field-ordered raster |
| `figures/05_replay_statistics.png` | replay scores, significance, speed, direction, extent |
| `figures/06_cross_session_summary.png` | replication across the four sessions |
| `cache/` | per-session results (`.pkl`) and the cross-session summary table |

## Reproducing

```bash
pip install pynapple pynwb remfile h5py matplotlib pandas scipy tqdm requests jupytext
python hippocampal_replay_analysis.py          # writes figures/ and cache/
jupytext --to notebook hippocampal_replay_analysis.py
```

The first run streams roughly 1–2 GB per session into `/tmp/remfile_cache` (override with
`REMFILE_CACHE`) and takes about ten minutes per session; subsequent runs read from `cache/`.
Plotting is headless (`matplotlib.use("Agg")`); no figure windows are opened.
