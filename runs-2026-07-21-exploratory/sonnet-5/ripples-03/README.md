# Sharp-Wave Ripples and Replay in Hippocampal CA1

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), "Diversity in neural firing
dynamics supports both rigid and learned hippocampal sequences" (Grosmark & Buzsaki,
*Science* 2016). We use the `sub-Buddy` session (`Buddy-06272013`), a bilateral
128-channel silicon-probe recording from dorsal CA1 consisting of a long PRE home-cage
sleep epoch, a ~40-minute MAZE epoch in which the rat runs on a novel 1.6 m linear track
for water reward, and a long POST home-cage sleep epoch. The NWB file is streamed
directly from the DANDI S3 bucket with `remfile`/`h5py` and accessed with `pynapple`; no
full-file download is required.

## Analysis

The notebook (`sharp_wave_ripples_and_replay.py`, jupytext-paired with
`sharp_wave_ripples_and_replay.ipynb`) does the following:

1. Selects the LFP channel with maximal 100-250 Hz ripple-band power and detects discrete
   sharp-wave ripple events during NREM sleep of the POST epoch with a standard
   threshold-crossing/Hilbert-envelope detector (488 events, ~0.24 Hz during NREM, median
   duration 58 ms).
2. Builds place fields for CA1 excitatory units from the MAZE running epoch (own
   linearization of the 2D LED position, since the dataset's precomputed linearized
   position is >90% missing) and validates the resulting tuning curves with a held-out
   Bayesian decoding test (r = 0.64 between true and decoded position on unseen running
   bouts).
3. Decodes population activity during each ripple event with `pynapple.decode_bayes` and
   shows example decoded trajectories, alongside a rigorous, shuffle-controlled
   significance test (rank-order correlation between spike order and place-field order)
   at both the single-event and population level.
4. Complements the sequential-replay test with a decode-free, population-level
   reactivation measure (explained variance, Kudrimoti/Pavlides-style): whether the
   pairwise firing correlation structure among place cells during RUN is preferentially
   reinstated during POST sleep versus PRE sleep.

## Key Finding

Sharp-wave ripples were detected robustly and with physiologically expected properties.
Individual ripple events, however, did not show statistically significant sequential
replay by rank-order correlation (488 ripples, 431 scored; fraction of individually
significant events = 3.2%, at chance level; population-level permutation p = 0.92) — an
honestly reported negative result, most likely reflecting that this single 40-minute
session yields only ~34 usable place cells, well below the ensemble sizes (typically
hundreds of simultaneously recorded cells) needed to reliably detect single-event replay
sequences. At the population level, however, a complementary decode-free measure showed
directional reactivation: the explained-variance statistic comparing RUN, PRE, and POST
pairwise correlation structure was higher for RUN->POST than for the RUN->PRE control in
92% of bootstrap resamples over cell pairs (EV = 0.029 vs. REV = 0.001), consistent with
experience-dependent reactivation of the track-running population code during subsequent
sleep, even where exact single-event trajectory decoding was underpowered.
