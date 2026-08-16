# Sharp-wave ripples and hippocampal replay in DANDI:000044

Streaming analysis of rat dorsal CA1 recordings from the DANDI Archive,
demonstrating sharp-wave ripples (SWRs) and the replay of running trajectories
inside them.

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), "Diversity in neural
firing dynamics supports both rigid and learned hippocampal sequences"
(Grosmark & Buzsáki, *Science* 2016; the CRCNS `hc-11` data set). Eight sessions
from four rats, each consisting of a pre-task sleep epoch, a maze epoch on a
novel track, and a post-task sleep epoch. Every file contains spike-sorted units
labelled excitatory or inhibitory, 128-channel LFP at 1250 Hz, position tracking,
and manually scored sleep states.

The primary analysis uses `sub-Achilles/sub-Achilles_ses-Achilles-10252013`
(1.6 m linear track, 120 pyramidal cells, 10 h session); the pipeline is then run
unchanged on all five linear-track sessions (the three circular-maze sessions are
excluded because they would need a different linearization and direction split).

Files are 5–9 GB each and are never downloaded in full: they are streamed from S3
with `remfile` plus a local disk cache. The LFP dataset is chunked one channel
per chunk, so reading a single channel for a whole session transfers well under
100 MB.

## What was analyzed

Sharp-wave ripples were detected from a single automatically selected CA1
pyramidal-layer LFP channel and characterized (waveform, spectrum, duration,
frequency, associated population firing, dependence on sleep state and running
speed). Direction-specific place fields were then built from the running laps
and used as the template for Bayesian decoding of the population activity inside
every ripple, testing whether the spikes in an event trace out a spatial
trajectory. Finally, the same conclusion was checked without any template using
the explained-variance measure on pairwise spike-count correlations. The whole
pipeline was run on all five linear-track sessions of the dandiset.

## Key finding

The detected events have every property expected of sharp-wave ripples. In the
primary session, 10,109 events were detected at 17.4/min, with a median duration
of 51 ms and a median intra-event frequency of 164 Hz. The ripple-triggered
average shows a positive deflection in the pyramidal layer against a ~1 mV
negative sharp wave in stratum radiatum, pyramidal population firing rises about
fivefold at the ripple peak, and the event rate is strongly state-dependent:
24.4/min in non-REM sleep, 13.2/min during quiet wakefulness, 0.03/min in REM,
and 0.19/min while the animal runs. These numbers are consistent across all five
sessions (51–57 ms, 156–164 Hz, 18–30/min in non-REM, under 1/min in REM).

Decoding the population activity inside those events against the place-field
template recovers replay. The decoder itself is accurate (3.0 cm median error
when tested against the animal's true position on running laps, r = 0.98). Of
the 5,461 decodable events in the primary session, 13.7% of those in post-task
sleep carry a spatial trajectory significant against both a column-cycle and a
field-identity shuffle, compared with 7.1% in pre-task sleep (odds ratio 2.08,
Fisher p = 2.8 × 10⁻¹⁵); awake ripples during immobility on the track are the
richest of all at 40.6%. The trajectories sweep the full 1.6 m track inside a
decoding window of 100–185 ms (10th to 90th percentile; the 100 ms lower edge is
the minimum window the analysis imposes), roughly twenty times faster than the
several seconds the animal took to run the same track, and they run in both
directions (59% forward, 41% reverse). Post-task sleep exceeded pre-task sleep in
all five sessions (sign test p = 0.031, Wilcoxon p = 0.062; pooled 10.2% vs 6.1%,
odds ratio 1.73, Fisher p = 1.4 × 10⁻¹⁸). Five sessions is a small enough paired
sample that the sign test cannot go below p = 0.031 even when every session
agrees, so the pooled event-level comparison is doing most of the statistical
work here. The template-free measure agrees: the pairwise
correlation structure of running is reinstated in post-task sleep far more than
in the time-reversed control (mean explained variance 4.2% vs 0.5% reverse; 6.0%
vs 0.8% in the primary session). That ordering holds in four of the five
sessions; in Cicero_09172014, the session with the weakest replay signal, the
two measures are both under 1% and the reverse control is nominally the larger.

A caveat worth stating plainly: pre-task sleep events also pass the sequence test
slightly above the nominal 5% rate. That is consistent with the "preplay"
literature, but it is equally consistent with the shuffles being slightly
liberal, so the defensible claim is the paired comparison rather than the
absolute pre-task rate.

## Files

| file | contents |
| --- | --- |
| `swr_replay_analysis.py` / `.ipynb` | consolidated end-to-end analysis (jupytext percent format) |
| `dandi_io.py` | DANDI asset resolution and streaming NWB access |
| `swr_lib.py` | ripple detection, position linearization, Bayesian decoding primitives |
| `pipeline.py` | one-call per-session pipeline and the multi-session driver |
| `01_load_inspect.py` … `07_multisession.py` | the development pipeline, step by step |
| `fig01`–`fig07` `.png` | figures (see below) |
| `multisession_summary.csv` | per-session summary statistics |
| `cache_events_*.csv`, `cache_summary_*.csv` | per-session decoded-event tables and summaries |
| `place_field_stats_Achilles_10252013.csv` | per-cell place-field statistics for the primary session |
| `ripples_*.npz`, `place_fields_*.npz`, `replay_events_*.csv`, `ripple_channel.npy` | intermediate results written by the step-by-step development scripts |

Figures:

- `fig01_session_overview.png`: session structure, sleep states, ripple-channel
  selection, raw and filtered LFP around a ripple
- `fig02_ripple_detection.png`: example events, ripple-triggered averages,
  power spectra, duration / frequency / amplitude distributions, population
  firing, state and speed dependence
- `fig03_place_fields.png`: direction-specific place fields, spatial
  information, place-cell selection, sequential activation during a traversal
- `fig04_replay_examples.png`: four post-sleep replay events (LFP, raster,
  decoded posterior)
- `fig05_replay_summary.png`: decoder validation, replay fraction by epoch,
  sequence-score distributions, forward vs reverse, time course
- `fig06_reactivation.png`: template-free explained-variance analysis
- `fig07_multisession.png`: all five linear-track sessions

## Method summary

**Ripple detection.** The channel with the largest ripple-band envelope standard
deviation during post-task non-REM sleep is selected automatically. The signal is
band-pass filtered at 130–250 Hz, the Hilbert envelope is smoothed with an 8 ms
Gaussian and z-scored against the immobility distribution, and events are periods
above 2 SD containing a peak above 5 SD and lasting 20–200 ms.

**Place fields.** Position is linearized by projecting the 2-D tracking onto the
track axis; the projection is calibrated against the linearized series shipped in
the file (r > 0.9999) and then applied to every tracked sample, which recovers
the immobile periods at the reward wells that the shipped series leaves as NaN.
Tuning curves are computed in 50 bins separately for each running direction. A
cell is a place cell if peak rate ≥ 1 Hz, Skaggs information ≥ 0.4 bits/spike,
and split-half tuning stability ≥ 0.3.

**Replay decoding.** Ripples outside running epochs are candidate events, padded
to at least 100 ms, binned at 20 ms, and required to have ≥ 5 active place cells
and ≥ 5 time bins. Each event is decoded with a flat-prior Poisson Bayesian
decoder against both direction templates; the score is the posterior-weighted
correlation between decoded position and time, and the direction with the larger
|r| is assigned. Significance requires p < 0.05 against **both** a column-cycle
shuffle (each time bin's posterior circularly shifted independently) and a
field-identity shuffle (place fields randomly reassigned among cells), 500
iterations each.

**Reactivation.** As a template-free cross-check, the Kudrimoti-style explained
variance is computed from pairwise spike-count correlations (100 ms bins) among
place cells: EV is the partial correlation between RUN and POST controlling for
PRE, and reverse EV swaps PRE and POST.

## Reproducing

```bash
pip install pynapple pynwb remfile h5py numpy scipy pandas matplotlib tqdm jupytext
python swr_replay_analysis.py          # or open the .ipynb
```

The first run streams roughly 1 GB per session into `/tmp` (remfile cache plus
one cached LFP channel per session) and takes on the order of ten minutes per
session; subsequent runs read the cached per-session CSVs.
