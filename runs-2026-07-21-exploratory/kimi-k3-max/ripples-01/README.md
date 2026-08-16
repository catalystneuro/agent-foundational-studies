# Sharp-Wave Ripples and Replay in Rat Hippocampus

**Dataset**: DANDI Archive dandiset [000044](https://dandiarchive.org/dandiset/000044)
("Diversity in neural firing dynamics supports both rigid and learned hippocampal
sequences", Buzsaki lab), session
`sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`. A rat ran ~42 laps on a
1.6 m linear track (MazeEpoch) between ~5 h of pre-sleep (PRE) and ~4 h of
post-sleep (POST). The NWB file contains 137 sorted CA1 units (120 excitatory, 17
inhibitory), 128-channel LFP at 1250 Hz, linearized track position at 39 Hz, and
sleep-state labels (Awake / Non-REM / REM). Data were streamed with `remfile`
(HTTP range requests + local disk cache); no full download.

## What was analyzed

1. **Ripple detection.** The LFP channel with the strongest ripple content was
   chosen data-driven (channel 117; highest ripple-band power x envelope
   peakiness x ripple/delta ratio during Non-REM). Sharp-wave ripples were
   detected in POST Non-REM sleep by bandpassing at 100-250 Hz, taking the
   smoothed Hilbert envelope, and applying a mean+4SD peak threshold with
   mean+1SD boundaries (30-500 ms duration, events < 30 ms apart merged).
2. **Place fields.** Run bouts on the track (86 bouts, 245 s) were used to build
   direction-pooled rate maps (50 bins, 1.5-bin Gaussian) for the 120 excitatory
   units. Spatial tuning was assessed with Skaggs spatial information against a
   circular time-shift shuffle (200 shifts on a concatenated run-bout time
   axis): 88/120 units are significant place cells (p<0.05, peak >= 1 Hz).
3. **Replay.** Each POST ripple was divided into 20 ms bins and position was
   Bayesian-decoded from place-cell spikes under a Poisson model. Sequential
   structure was scored with the posterior-weighted correlation between time
   and position, tested per event against 500 cell-identity shuffles and 500
   within-event time-bin shuffles. As a control, PRE-sleep ripples were
   detected and decoded identically with the maze template.

## Key findings

- 3083 sharp-wave ripples in POST Non-REM (0.58 per min; median duration 48 ms).
  The event-locked average shows the classic SWR anatomy: a slow sharp wave with
  a ~150 Hz oscillation locked to the envelope peak. Pyramidal cells and
  interneurons both increase firing around the ripple peak (z ~ 23 and ~ 26),
  interneurons slightly earlier.
- Of 328 POST ripples long and active enough to decode (>= 100 ms, >= 5 place
  cells), **17.4% show significant replay** (cell-identity shuffle p<0.05;
  18.6% under the bin-shift shuffle; 15.5% under both), far above the 5% chance
  level. Median |weighted correlation| is 0.39 vs 0.28 in the null. Significant
  events split roughly evenly between forward (30) and reverse (27)
  trajectories, as expected for sleep replay.
- **Experience dependence**: PRE-sleep ripples decoded with the same maze
  template sit at chance (5.6% significant, median |wc| 0.25), while POST is
  strongly enriched (Mann-Whitney p = 2.9e-12; chi-square p = 8.1e-10),
  showing the sequences reflect the preceding track experience.

## Files

- `ripple_replay_analysis.py` — consolidated jupytext script (runs end-to-end)
- `ripple_replay_analysis.ipynb` — the same analysis as a Jupyter notebook
- `01_load_inspect.py`, `02a_select_channel.py`, `02b_detect_ripples.py`,
  `03_place_fields.py`, `04_replay.py`, `05_peri_ripple_showcase.py`,
  `06_pre_vs_post.py`, `common.py` — modular development scripts
- Figures: `fig_channel_selection.png`, `fig_channel_traces.png`,
  `fig_ripple_detection_example.png`, `fig_envelope_distribution.png`,
  `fig_ripple_properties.png`, `fig_ripple_perievent.png`,
  `fig_placefields_examples.png`, `fig_placefields_population.png`,
  `fig_placefields_si.png`, `fig_replay_examples.png`, `fig_replay_stats.png`,
  `fig_replay_direction.png`, `fig_peri_ripple_psth.png`,
  `fig_replay_showcase.png`, `fig_pre_vs_post.png`

## Reproducing

```
python ripple_replay_analysis.py          # ~10-15 min (streams ~60 MB of LFP chunks)
jupytext --to notebook ripple_replay_analysis.py
jupyter nbconvert --to notebook --execute ripple_replay_analysis.ipynb
```

Requires: pynapple, pynwb, h5py, remfile, scipy, matplotlib, tqdm, requests.
