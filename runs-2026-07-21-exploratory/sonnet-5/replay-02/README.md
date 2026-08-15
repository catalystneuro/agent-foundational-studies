# Decoding Hippocampal Replay During Sharp-Wave Ripples

**Dataset**: [DANDI:000447](https://dandiarchive.org/dandiset/000447) ("Novel-familiar-novel
WTrack (CA1-PFC)", Frank/Jadhav labs), session `sub-JDS-NFN-AM2`, streamed directly from
DANDI S3 storage (no local download). The session contains simultaneous CA1 and
prefrontal cortex tetrode recordings (LFP + spike-sorted units) and video-tracked
position from a rat performing a W-track spatial alternation task across three
behavioral epochs on different physical mazes. All analysis uses epoch 1, a classic
three-arm W-track with 49 completed trials, 32 CA1 units, and 30 CA1 LFP channels.

**Analysis**: We first built 2D place fields for CA1 units from position and spiking
during running (speed > 4 cm/s), selecting 26/32 units as place cells by peak rate,
Skaggs spatial information, and spike count. We then detected candidate sharp-wave
ripples (SWRs) from the CA1 LFP ripple-band (150-250 Hz) envelope, restricting both the
z-scoring baseline and the candidate search to immobile bouts (speed < 4 cm/s) - an
unrestricted, whole-recording baseline turned out to be dominated by movement-related
broadband artifacts (high ripple-band power samples were *less* likely to be immobile
than chance), so gating on immobility first was necessary to get physiologically
plausible events (~1 Hz during immobile time, consistent with the literature). For the
23 events with enough spiking to constrain a decode (≥4 active place cells, ≥6 spikes),
we performed Bayesian decoding of 2D position from population spike counts
(`pynapple.decode_bayes`, 10 ms bins) using the place fields as the encoding model, then
scored each event's decoded trajectory with a weighted position-time correlation tested
against a 500-fold within-event time-bin shuffle.

**Key finding**: 7 of 23 candidate SWR events (30%) showed decoded position sequences
that swept through space significantly more than expected by chance (p < 0.05 vs. the
shuffle null; ~5% expected under the null), directly demonstrating hippocampal replay:
the SWR-associated reactivation of coherent spatial trajectories from the place-cell
code learned during behavior. Individual examples illustrate two replay patterns
reported in the literature for this same task and dataset family (Karlsson & Frank,
2009; Gillespie et al., 2021): *local* replay, where the decoded posterior stays
concentrated near the animal's actual (stationary) position, and *remote* replay, where
the decoded trajectory sweeps through a distant, currently-unvisited part of the track.

## Files

- `hippocampal_replay_decoding.py` - jupytext (percent format) source script, runs end-to-end
- `hippocampal_replay_decoding.ipynb` - executed notebook version
- `fig01_raw_data_overview.png` - trajectory, speed, and CA1 raster overview
- `fig02_place_fields.png` - CA1 place fields on the W-track
- `fig03_ripple_detection.png` - example detected SWR events during immobility
- `fig04_example_replay_local.png` - example locally-anchored replay event
- `fig05_example_replay_remote.png` - example remote replay event
- `fig06_population_summary.png` - replay score distributions and significance summary
- `replay_scores.csv` - per-event trajectory scores, shuffle statistics, and p-values
