# Sharp-wave ripples and hippocampal replay in CA1

This analysis demonstrates two hallmark hippocampal phenomena, and the link
between them, in a single publicly available recording: **sharp-wave ripples
(SWRs)** in the CA1 local field potential, and **replay** of spatial
trajectories by place-cell populations during those ripples.

## Dataset

- **DANDI:000044** — Grosmark & Buzsáki (2016), *"Diversity in neural firing
  dynamics supports both rigid and learned hippocampal sequences."*
- Session `sub-Achilles_ses-Achilles-10252013`: bilateral CA1 silicon-probe
  recording with LFP (128 channels, 1250 Hz), 137 sorted units (120 excitatory
  / 17 inhibitory), and linearized position on a 1.6 m linear track.
- The session has three epochs: **PRE** sleep, a ~34 min **MAZE** running epoch,
  and **POST** sleep, with hypnogram states (Awake / non-REM / REM).
- The file is streamed directly from the DANDI S3 bucket with `remfile`
  (disk-cached). Nothing is downloaded in full; only one LFP channel and the
  spike/behavior tables are read.

## What was analyzed

1. **Ripple detection** (`fig2_ripples.png`). One CA1 channel was band-passed at
   150-250 Hz, its Hilbert envelope smoothed and z-scored, and threshold-crossing
   events extracted (peak > 4 SD, 20-250 ms). 5,548 ripples were detected across
   MAZE + POST.
2. **Place fields** (`fig3_placefields.png`). Using `pynapple`, 1-D place fields
   were computed for pyramidal cells during sustained running (> 8 cm/s) on the
   track. 114 / 120 pyramidal cells qualified as place cells (peak > 1 Hz,
   spatial information > 0.3 bits/spike).
3. **Bayesian decoding and replay** (`fig4_replay.png`). A memoryless Bayesian
   decoder built from the running place fields was first validated on running
   data, then applied to ripple-associated population-burst events in POST
   sleep, decoding position in 20 ms bins. Each event was scored by the weighted
   correlation between decoded position and time, with significance from a
   within-event time-bin shuffle.

## Key findings

Sharp-wave ripples showed their textbook signature: a sharp negative LFP
deflection carrying a ~180 Hz oscillation (median intra-ripple frequency 181 Hz),
lasting a median of 47 ms. Their occurrence was strongly brain-state dependent,
with a rate of 31 ripples/min in non-REM sleep, 16/min during awake immobility,
and essentially zero in REM. The place-cell population tiled the whole track
(with the reward ends over-represented, a documented feature of this dataset)
and supported Bayesian position decoding to a median error of 16 cm on a 160 cm
track.

During POST-sleep ripples the decoded position swept smoothly across the track
within individual ripples, reactivating running trajectories compressed into a
few tens of milliseconds. These trajectories were significant against a time-bin
shuffle in 12% of decodable events, more than twice the 5% chance level, and
occurred in both forward and reverse directions. Together these results
reproduce, in real data, the canonical picture of hippocampal replay: during
sharp-wave ripples in rest and sleep, place-cell assemblies replay the paths the
animal took while awake.

## Files

- `swr_replay_analysis.py` — consolidated jupytext script (runs end-to-end, ~90 s).
- `swr_replay_analysis.ipynb` — executed notebook version.
- `swr_common.py` and `stage1`–`stage4` scripts — modular development pipeline.
- `fig1_overview.png` … `fig4_replay.png` — figures.

Run with: `python swr_replay_analysis.py` (requires `pynapple`, `pynwb`,
`lindi`/`remfile`, `h5py`, `scipy`, `matplotlib`).
