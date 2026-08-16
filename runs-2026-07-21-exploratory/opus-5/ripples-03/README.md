# Sharp-Wave Ripples and Replay in Hippocampal CA1

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark and Buzsáki (2016),
*Science* 351:1440. Four rats ran back and forth on a linear track between two sleep
sessions in the home cage. Each NWB file holds roughly nine hours of 128-channel CA1 LFP
at 1250 Hz, spike-sorted units labelled excitatory or inhibitory, the linearized position
on the track, and manually scored brain states (Awake, non-REM, REM). Files are about 9 GB
each and are never downloaded whole. They are streamed with `remfile` over a local disk
cache; because the LFP dataset is chunked one channel per chunk, reading a single channel
for a whole session costs about 90 MB. All time-series handling (interval sets,
restriction, tuning curves, perievent alignment, wavelet transform, Bayesian decoding)
uses [Pynapple](https://pynapple.org).

The analysis prototypes on `Achilles-10252013`, the session with the largest unit yield,
and then repeats the entire pipeline on `Gatsby-08022013`, `Cicero-09172014` and
`Buddy-06272013`. `Achilles-11012013` is excluded because its maze is circular rather than
linear, which would require a different linearization.

## What Was Analyzed

Two separate things, which turn out to have quite different evidential strength.

The first is the oscillation itself. Ripples were detected on the channel with the largest
130 to 250 Hz envelope during post-task non-REM sleep, using a standard threshold rule
(peak above 4 SD, edges at 2 SD, 20 to 200 ms, locomotion excluded), and then checked
against six criteria the detector was not tuned to satisfy: the ripple-triggered average
of the raw LFP, the mean wavelet spectrum, the duration and intra-ripple frequency
distributions, the pyramidal and interneuron peri-ripple firing profiles, the dependence
on brain state, and re-detection on an electrode from a different shank.

The second is the content of those ripples. Direction-specific place-field templates were
built from running periods on the track, validated by decoding the animal's real position,
and then used to decode ripple-associated population bursts in 20 ms bins. Each event was
scored by the weighted correlation between decoded position and time and tested against
two posterior shuffles (a per-bin circular position shift and a time-bin permutation, 400
draws each). Two controls back this up: permuting which cell owns which place field, and a
pairwise explained-variance measure (EV/REV) that involves no decoding at all.

## Key Finding

In `Achilles-10252013` the detected events are sharp-wave ripples by every standard
criterion. There are 14,625 of them over 9.7 hours, with a median duration of 42 ms and a
median intra-ripple frequency of 162 Hz, riding on a clear slow sharp wave. They occur at
0.61 Hz in non-REM sleep, 0.29 Hz during quiet waking, and 0.0014 Hz in REM, which is the
classic state dependence and cannot be produced by an amplitude artifact. Both cell classes
fire more during ripples, interneurons far more strongly than pyramidal cells, and 89% of
the events are recovered independently from a different shank. This part of the result
reproduces almost exactly in all four sessions: rate, duration, intra-ripple frequency and
the non-REM to REM contrast are nearly identical across animals.

The spikes inside those ripples carry the track. With 106 place cells decoding the
animal's real position to a median error of 4.7 cm (chance is 46.8 cm), about 15% of
ripple-associated population bursts in post-task sleep carry a statistically significant
trajectory, against about 6% in pre-task sleep (Fisher odds ratio 2.8, p = 3e-11). The
trajectories run both forward and reverse and sweep the 1.6 m track in 200 to 300 ms,
about seven times faster than the animal ran it, and the decoding-free EV/REV statistic
shows the same asymmetry (0.088 against 0.001 over 5565 cell pairs).

Two things temper this. The cell-identity shuffle, which permutes which cell owns which
place field and is the strongest available control, comes out at about 9% rather than the
nominal 5%. The per-event test is therefore liberal, roughly half of the significant POST
events are what a pipeline with no true cell-to-field correspondence would produce anyway,
and the meaningful comparison is 15% against 9% (p = 0.002) rather than 15% against 5%.
The excess is still there, but it is smaller than the headline number suggests. Second,
the replay result is carried by one session. Tested
individually, Achilles is significant (odds ratio 2.8, p = 1e-11), Cicero is marginal
(1.6, p = 0.05), and Gatsby (1.07) and Buddy (1.3) are not. The Cochran-Mantel-Haenszel
odds ratio stratified by session is 1.9 at p = 2e-9, but its homogeneity test rejects at
p = 0.002, which is the formal statement that no single pooled number describes these four
sessions. The most likely reason is ensemble size: Achilles contributes 106 place cells,
the other three contribute 30 to 44, and a weighted-correlation measure over 20 ms bins
needs enough simultaneously active fields to tell a trajectory from a scattered posterior.
That is a limitation of the measurement rather than evidence against replay, and the
pairwise EV/REV statistic, which needs far fewer cells and shows EV greater than REV in
all four sessions, is the better-supported cross-session claim.

One caveat on exact numbers. The per-event p-values come from a 400-draw shuffle, so
counts move by a few events between runs, and the cell-identity shuffle rate itself is
estimated from only 400 events and carries a standard error of about 1.4 percentage
points. Percentages in this README are quoted to the nearest point for that reason;
`results_summary.txt` holds the exact counts from the run that produced the committed
figures. Taking the ripple physiology and the EV/REV reactivation together, the strongest
defensible statement from these four sessions is that sharp-wave ripples are present and
well characterized in all of them, that ripple-associated reactivation of run-period
co-firing is present in all of them, and that full sequential replay is demonstrated
convincingly in the one session with a large enough place-cell ensemble to measure it.

## Files

| File | Contents |
| --- | --- |
| `sharp_wave_ripples_and_replay.py` | Consolidated jupytext analysis, runs end to end |
| `sharp_wave_ripples_and_replay.ipynb` | The same notebook with executed outputs |
| `swr_utils.py` | Streaming loaders for DANDI:000044 (remfile + disk cache) |
| `pipeline.py` | Ripple detection, place fields, replay scoring, EV/REV |
| `01_*.py` through `07_*.py` | Prototyping scripts, kept for provenance |
| `fig01_channel_selection.png` | Ripple-band power across the 128 sites |
| `fig02_raw_streams.png` | Raw LFP, filtered LFP, position, spike raster |
| `fig03_ripple_characterization.png` | The six validation checks on the detected events |
| `fig04_example_ripples.png` | Six individual ripples with spikes |
| `fig05_place_fields.png` | Direction-specific place fields and spatial information |
| `fig06_replay_examples.png` | Six replay events: LFP, posterior, ordered raster |
| `fig07_replay_summary.png` | Decoder validation, PRE/POST/shuffle, speed, EV/REV |
| `fig08_multi_session.png` | The same measures across four sessions |
| `multi_session_summary.csv` | One row per session of every summary statistic |
| `multi_session_events.csv` | Every scored candidate event across the four sessions |
| `results_summary.txt` | The headline numbers as printed by the notebook |

## Running It

```bash
pip install pynapple pynwb remfile h5py lindi xarray statsmodels tqdm matplotlib jupytext
export MPLBACKEND=Agg
jupytext --to notebook --output sharp_wave_ripples_and_replay.ipynb sharp_wave_ripples_and_replay.py
jupyter nbconvert --to notebook --execute --inplace sharp_wave_ripples_and_replay.ipynb
```

The four-session loop is cached in `multi_session_summary.csv` and
`multi_session_events.csv`. Delete those two files to recompute it, which takes about half
an hour with a warm `remfile` cache and considerably longer without one.
