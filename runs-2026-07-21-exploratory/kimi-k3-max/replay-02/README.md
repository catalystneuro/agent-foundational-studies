# Hippocampal replay: decoding spatial trajectories during sharp-wave ripples

**Dataset.** DANDI Archive dandiset [000044](https://dandiarchive.org/dandiset/000044)
(Buzsáki lab, "Diversity in neural firing dynamics across brain states"), session
`sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb` (8.7 GB, streamed with
remfile + local disk cache; nothing downloaded wholesale). A rat ran ~42 laps on a
1.6 m linear track (MazeEpoch, 18079.5-20147 s) between PRE and POST sleep epochs,
with 137 sorted CA1 units (120 excitatory, 17 inhibitory), 128-channel LFP at
1250 Hz, position tracking at 39 Hz, and a sleep-state table (Awake/Non-REM/REM).

**Analysis.** Place fields were built from run bouts on the linearized track
position (86 bouts, 245 s; bouts are contiguous valid stretches merged over <0.3 s
gaps, >=1 s, spanning >0.3 m, median speed >0.15 m/s). 92/120 excitatory units
qualified as place cells (Skaggs spatial information p<0.05 against 500 circular
time shifts on the concatenated run-bout axis, mean rate >=0.1 Hz, peak >=1 Hz).
The ripple channel was chosen data-driven (peakiness of the 100-250 Hz envelope
times ripple/delta power on a POST Non-REM chunk), landing on channel 117, the CA1
pyramidal-layer channel. SWRs were detected from the smoothed Hilbert envelope of
the 100-250 Hz signal (edges at mean+1SD, peak >mean+4SD of Non-REM samples, merge
<30 ms, keep 30-500 ms fully inside Non-REM): 5389 PRE and 2926 POST events
(~31-33 per minute of Non-REM, median 59 ms). Each SWR was binned into 20 ms
windows and decoded against the place-field template with a Poisson Bayesian
decoder; events needed >=5 active place cells and >=5 non-empty bins. Replay was
scored per event as the posterior-mass-weighted correlation between time and
decoded position, with significance from 500 cell-ID shuffles per event.

**Key finding.** SWRs during POST sleep replay spatial trajectories of the track
at well above chance levels, while PRE sleep (before the animal ever ran on the
track) sits at chance. 65/477 decodable POST events (13.6%) were significant at
p<0.05 versus 56/1228 (4.6%) in PRE (Fisher exact p=7.0e-10; Mann-Whitney on
|score| p=1.0e-09). Significant POST events include both forward (40) and reverse
(25) trajectories that sweep much of the 1.6 m track within a single ~100-250 ms
ripple, an effective speed of ~6-16 m/s against the ~0.5 m/s the animal actually
ran.
This is the classic signature of hippocampal replay: experience-dependent,
compressed reactivation of spatial trajectories during sharp-wave ripples.

## Files

- `replay_analysis.py`: consolidated end-to-end analysis (jupytext format)
- `replay_analysis.ipynb`: same analysis as a Jupyter notebook
- `figures/fig01_behavior.png`: position traces, run bouts, example place-cell raster
- `figures/fig02_place_fields.png`: place-field template, example rate maps, SI distributions
- `figures/fig03_ripple_detection.png`: example SWRs, envelope thresholds, duration distribution
- `figures/fig04_example_replays.png`: decoded posteriors for the strongest forward and reverse events
- `figures/fig05_replay_stats.png`: score distributions and PRE/POST significance summary
- `01_load_data.py`, `02_prep_cache.py`, `03_placefields.py`, `04_ripples.py`,
  `05_replay.py`: modular development scripts (same pipeline, with local caching)
- `cache/`: intermediate arrays produced by the modular scripts

## Reproduce

```
python replay_analysis.py          # ~5-10 min with a warm remfile cache
jupytext --to ipynb replay_analysis.py
```

Requirements: pynapple 0.11, pynwb 4, remfile, h5py, scipy, matplotlib, tqdm, jupytext.
