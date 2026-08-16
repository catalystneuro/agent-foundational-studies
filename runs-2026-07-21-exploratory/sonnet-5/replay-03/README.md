# Hippocampal Replay: Decoding Spatial Trajectories During Sharp-Wave Ripples

## Dataset

[DANDI Dandiset 000044](https://dandiarchive.org/dandiset/000044), "Diversity in neural
firing dynamics supports both rigid and learned hippocampal sequences" (Grosmark &
Buzsaki, *Science* 2016). Session `sub-Buddy_ses-Buddy-06272013` (streamed directly
from DANDI S3, no local download) is a bilateral CA1 silicon-probe recording (128
channels / 16 shanks, 68 sorted units) from a Long-Evans rat during a long home-cage
sleep/rest recording (`PREEpoch`), a run on a 1.6 m linear track rewarded at both ends
(`MazeEpoch`), and a second sleep/rest recording (`POSTEpoch`). The NWB file provides
LFP (1250 Hz), 2D tracked position, sorted units with cell-type/location labels, and a
sleep-scoring table (`Awake` / `Non-REM`).

## Analysis

CA1 place fields were computed from spiking activity during track running (speed >
0.05 m/s), yielding 28 place cells that tile the length of the track. Sharp-wave
ripples were detected from the CA1 LFP (150-250 Hz band, z-scored envelope threshold)
during Non-REM sleep bouts of both `PRE` and `POST`. For each ripple with sufficient
population spiking, a memoryless Bayesian decoder (place fields learned during
running, uniform prior) reconstructed a posterior probability distribution over track
position in 20 ms bins across the event. Each event's decoded trajectory was scored
with the weighted correlation between decoded position and time-within-event, tested
against a within-event column-cycle shuffle null (500 shuffles) to flag events with
significant sequential structure (p < 0.05).

A real implementation bug was caught and fixed during development: naively computing
running speed/position across the temporal gaps left by excluding off-track (reward
well) samples produced spurious "teleport" position estimates that collapsed nearly
every cell's place field onto the same track location. Splitting the position data
into contiguous on-track bouts (breaking wherever consecutive samples are more than
0.1 s apart) fixed this and produced the expected place field map (see
`fig02_place_fields.png`).

## Key finding

Ripples detected in `POST` sleep (after the animal had run the track) showed
sequential position decoding significantly more often (15.4%, 8/52 qualifying events)
than ripples detected in `PRE` sleep, before the animal had ever experienced the track
(9.3%, 19/204 qualifying events) — both above the 5% expected by chance under the
shuffle null. This experience-dependent elevation in decodable sequential structure is
the classic signature of hippocampal replay: population activity during sharp-wave
ripples reconstructs spatial trajectories related to the just-explored environment.
The non-zero rate of significant sequences in `PRE` is consistent with Grosmark &
Buzsaki (2016), who describe a subset of "rigid" CA1 sequences that pre-exist
experience alongside a "flexible" subset that emerges only after running the track;
the `POST`-over-`PRE` elevation recovered here matches the flexible component being
layered on top of pre-existing structure.

## Files

- `hippocampal_replay_analysis.py` — consolidated jupytext (percent-format) analysis script, runs end-to-end.
- `hippocampal_replay_analysis.ipynb` — executed Jupyter notebook (same analysis, with outputs).
- `fig01`-`fig06` — figures (behavior, place fields, example ripple, ripple duration distribution, example decoded replay events, POST vs PRE replay score summary).
- `replay_scores_post.csv`, `replay_scores_pre.csv` — per-ripple decoding results (weighted-correlation score, shuffle p-value, significance) for all qualifying events.
