# Decoding hippocampal replay during sharp-wave ripples

This analysis demonstrates hippocampal replay by decoding the spatial trajectory that CA1
place cells represent during sharp-wave ripple events. It uses session
`sub-Achilles_ses-Achilles-10252013` from [DANDI:000044](https://dandiarchive.org/dandiset/000044)
(Grosmark & Buzsáki, *Science* 2016), in which a rat runs back and forth on a 1.6 m linear
track while roughly 120 CA1 pyramidal cells and the local field potential are recorded with
bilateral silicon probes. The track run (the Maze epoch) is flanked by PRE and POST rest
periods that include slow-wave sleep. All data are streamed directly from the DANDI S3 bucket
with `remfile` and analyzed with Pynapple; nothing is downloaded in full.

The pipeline builds direction-specific place fields from the running periods, detects
sharp-wave ripples from a CA1 channel by band-pass filtering at 150 to 250 Hz and thresholding
the z-scored ripple envelope during immobility, and validates a memoryless Bayesian position
decoder against the animal's true position during running (median error 4.9 cm, correlation
0.96 on the 1.6 m track). The same decoder is then applied inside each ripple at 20 ms
resolution. During ripples the decoded posterior sweeps smoothly across the track, both in the
forward and the reverse direction, and these trajectories are scored by the weighted
correlation between decoded position and elapsed time.

## Key finding

Sharp-wave ripples contain compressed replays of the linear-track trajectory. Individual
example events show clean posteriors moving across the full track in 80 to 140 ms with weighted
correlations near 0.97. Across the population, the observed trajectory scores have a heavy tail
well beyond a position-shuffled null, and 14% of PRE-sleep candidate events and 21% of
POST-sleep candidate events qualify as significant replay at p < 0.05 (both far above the 5%
chance level). Replay is significantly more frequent after the track experience than before it
(two-proportion z = 6.6, p = 5e-11), the expected signature of experience-dependent
reactivation. The decoded trajectories advance at a median virtual speed of about 12 m/s,
roughly 23 times the animal's actual running speed, which is the temporal compression that
defines hippocampal replay.

## Files

- `replay_analysis.py`: consolidated jupytext script (runs end to end in about 80 s, most of
  which is streaming one LFP channel on the first pass).
- `replay_analysis.ipynb`: executed notebook version.
- `fig_placefields.png`: direction-specific place fields tiling the track.
- `fig_ripple_example.png`: one ripple showing raw LFP, ripple-band trace, and pyramidal raster.
- `fig_ripple_mua.png`: ripple-triggered population firing (6.5x increase at ripple peak).
- `fig_decoder_validation.png`: decoder recovering true position during running.
- `fig_replay_examples.png`: six example replay trajectories from POST-sleep ripples.
- `fig_replay_summary.png`: trajectory scores vs shuffle, PRE vs POST replay rate, replay speed.
