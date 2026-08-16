# Hippocampal Replay During Sharp-Wave Ripples (DANDI 000044)

This analysis demonstrates hippocampal replay: during sharp-wave ripple (SWR)
events in sleep, the CA1 population re-expresses compressed spatial
trajectories of a recently experienced environment. All data are real
experimental recordings streamed from the DANDI Archive; no synthetic data is
used anywhere in the pipeline.

## Dataset

We use dandiset [000044](https://dandiarchive.org/dandiset/000044) from the
Buzsaki lab, session `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`
(8.7 GB, streamed with remfile plus a local disk cache, so no full download is
needed). The session contains 137 sorted CA1 units from tetrodes, 128-channel
LFP at 1250 Hz, the animal's position on a 1.6 m linear maze (~39 Hz, with a
linearized 0-1.6 m coordinate), PRE/Maze/POST epoch boundaries, and
Awake/Non-REM/REM sleep-state labels. The maze epoch is 34.5 min and is
flanked by 2.94 h (PRE) and 1.48 h (POST) of Non-REM sleep.

## Analysis

The pipeline (`hippocampal_replay_analysis.py`, a jupytext script that runs
end-to-end, and the executed `hippocampal_replay_analysis.ipynb`) proceeds in
four steps:

1. **Place-field templates.** Run bouts are contiguous valid stretches of the
   linearized position (merged over < 0.3 s gaps, kept if >= 1 s long, > 0.3 m
   span, median speed > 0.15 m/s), giving 86 bouts and 245 s of running.
   Rate maps (50 position bins, 1.5-bin Gaussian smoothing of counts and
   occupancy) are computed for all units, and spatial information (Skaggs,
   bits/spike) is tested against 500 circular shifts of spike times along the
   concatenated bout clock. 88 of 120 excitatory units pass as place cells
   (rate > 0.1 Hz, peak >= 1 Hz, shuffle p < 0.05).
2. **SWR detection.** The ripple reference channel is chosen data-driven
   (ripple-band envelope peakiness times ripple/delta power on a POST Non-REM
   block; channel 117 wins, consistent with a CA1 pyramidal-layer site). The
   100-250 Hz Hilbert envelope (4 ms smoothing) is thresholded at mean + 4 SD
   (peaks) and mean + 1 SD (edges), crossings < 30 ms apart are merged, and
   events of 30-500 ms fully inside Non-REM are kept: 8394 SWRs (5440 PRE at
   30.9/min, 2954 POST at 33.3/min, median duration 60 ms). The peri-SWR
   excitatory multiunit burst (z ~ 45 at the peak) confirms these are bona
   fide sharp-wave ripples.
3. **Bayesian decoding.** Position is decoded from place-cell spiking in
   20 ms bins under a Poisson model, and each event is scored with the
   weighted correlation between bin time and decoded position (weights equal
   to posterior mass). Significance comes from 200 per-event cell-ID shuffles
   of the tuning curves. 2233 events with >= 5 non-empty bins and >= 5 active
   place cells are decodable (1596 PRE, 637 POST).
4. **Statistics.** Replay prevalence is compared between PRE and POST sleep
   with a Fisher exact test on the fraction of significant events and a
   Mann-Whitney test on |weighted correlation|.

## Key Finding

SWRs during POST-maze sleep replay the track significantly above chance, and
PRE-maze sleep does not. 10.7% of decodable POST events are significant
(68/637; 40 forward, 28 reverse), versus 4.8% in PRE (76/1596), which is the
5% false-positive rate expected under the null (Fisher exact p = 8.5e-07;
Mann-Whitney on |w|, POST > PRE, p = 3.2e-10; median |w| 0.337 POST vs 0.253
PRE). Example events show clean posterior trajectories sweeping the full
1.6 m track within ~120 ms, roughly 100x compressed relative to real running
speed. This replicates the classic replay result: sleep SWRs after, but not
before, a spatial experience reactivate compressed trajectories of that
experience.

## Outputs

- `hippocampal_replay_analysis.py`: consolidated jupytext pipeline (runs
  end-to-end with `python hippocampal_replay_analysis.py`, ~4 min with a warm
  remfile cache).
- `hippocampal_replay_analysis.ipynb`: executed notebook version.
- `fig1_session_and_position.png`: session layout, laps, occupancy, speed.
- `fig2_place_fields.png`: rate maps of the 88 place cells and SI shuffle.
- `fig3_swr_detection.png`: ripple-band detection validation, SWR rates and
  durations, peri-SWR population burst.
- `fig4_example_replay_events.png`: rasters and decoded posteriors for the
  strongest forward and reverse replay events.
- `fig5_replay_statistics.png`: |weighted correlation| vs shuffle null,
  significant fractions PRE vs POST, direction split, duration dependence.
- `replay_results.npz`: rate maps, SWR table, per-event scores and shuffle
  nulls; `summary.json`: headline numbers.

## Methods notes

Spike positions are looked up by mapping spike times onto the nearest
bout-position sample; the SI shuffle circularly shifts spike times along the
concatenated bout clock rather than over wall-clock maze time, so shuffled
spikes never land in reward-platform pauses. The cell-ID shuffle used for
replay significance leaves each event's spike trains untouched and permutes
only the tuning-curve assignment, which preserves any structure explained by
co-firing rates alone. The Butterworth filter is applied in second-order
sections form, which is required for numerical stability at a 100 Hz cutoff
relative to the 1250 Hz sampling rate.
