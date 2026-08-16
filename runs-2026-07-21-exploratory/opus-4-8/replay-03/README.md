# Hippocampal Replay During Sharp-Wave Ripples

This analysis demonstrates hippocampal **replay**: during sharp-wave ripple (SWR)
events of quiet rest and sleep, ensembles of CA1 place cells re-express the
spatial trajectories the animal traversed while awake, compressed into tens of
milliseconds. The demonstration decodes the represented position directly from
spikes and shows that the decoded location sweeps coherently across the track
during individual ripples.

## Dataset

- **DANDI:000044** — Grosmark, Long & Buzsáki (2016), *"Diversity in neural firing
  dynamics supports both rigid and learned hippocampal sequences."*
- **Session:** `sub-Achilles/sub-Achilles_ses-Achilles-10252013` — a bilateral CA1
  silicon-probe recording from a rat that ran back and forth on a 1.6 m linear
  track (MAZE epoch, ~34 min) flanked by long rest/sleep epochs (PRE ~5 h, POST
  ~4 h). The file provides 137 sorted units (120 putative pyramidal), a 128-channel
  LFP at 1250 Hz, a linearized position signal, epoch boundaries, and scored
  behavioural states. The file was streamed from S3 with `remfile` disk caching;
  no full download was needed.

## What was analyzed

1. **Place fields.** Using on-track running periods (>5 cm/s), 1-D rate maps were
   built for all pyramidal cells over the 1.6 m track. Cells with a peak rate ≥ 1 Hz
   (89 cells) form the decoding ensemble; their fields tile the whole track
   (`fig1_place_fields.png`).
2. **Ripple detection.** The CA1 channel with the strongest 150–250 Hz power was
   band-pass filtered and its Hilbert envelope thresholded (5 SD onset, 2 SD
   extension, 15–450 ms). This yielded **4378 ripples** in POST, median duration
   ~49 ms, with textbook sharp-wave + ripple morphology (`fig2_ripples.png`).
3. **Bayesian decoding.** For each ripple with ≥ 5 active place cells and ≥ 80 ms
   duration (696 events), position was decoded in 20 ms bins with a memoryless
   Bayesian decoder (flat spatial prior).
4. **Replay scoring.** Each decoded posterior was scored by its posterior-weighted
   correlation between position and time (a line-like-sweep measure) and compared
   to a within-event column-cycle shuffle null (500 shuffles/event).

## Key finding

Ripples reactivate spatial trajectories far above chance. **202 of 696 events
(29%) were significant replays at p < 0.05** — roughly six times the 5% false-
positive rate expected under the shuffle null — and the effect holds at p < 0.01
(~17% of events vs 1% chance). The mean absolute weighted correlation of real
ripples (0.42) clearly exceeds the shuffle mean (0.26). Both **forward** (121) and
**reverse** (81) replays occur, matching the known coexistence of both directions
during awake/sleep SWRs. The example events (`fig3_example_replays.png`) show the
decoded posterior sweeping smoothly and monotonically across the full 1.6 m track
within 100–160 ms, i.e. a compressed re-expression of the running trajectory that
is the defining signature of hippocampal replay. Population statistics are
summarized in `fig4_population_stats.png`.

## Files

- `replay_analysis.py` — consolidated jupytext script (markdown + code cells),
  runs end-to-end.
- `replay_analysis.ipynb` — executed notebook with embedded outputs.
- `replay_lib.py` — streaming loader and analysis helpers (ripple detection,
  weighted-correlation scoring).
- `fig1_place_fields.png`, `fig2_ripples.png`, `fig3_example_replays.png`,
  `fig4_population_stats.png` — figures.
- `replay_scores.csv` — per-event weighted correlation, p-value, and cell counts.

## Reproducing

```bash
pip install pynapple pynwb lindi remfile h5py dandi tqdm scipy matplotlib jupytext
python replay_analysis.py            # runs the full pipeline, writes figures
jupytext --to notebook replay_analysis.py   # regenerate the notebook
```
