# Decoding hippocampal replay during sharp-wave ripples

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark, Long and Buzsáki
(2016), *Diversity in neural firing dynamics supports both rigid and learned
hippocampal sequences*. Bilateral silicon-probe recordings from dorsal CA1 in freely
moving Long-Evans rats. This analysis uses one session,
`sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb` (8.7 GB),
streamed with `remfile` plus a local disk cache; only the spike times, the tracking,
and one of the 128 LFP channels are ever transferred. The session has three epochs:
about 5 h of PRE rest/sleep in the home cage, 35 min of running on a **novel** 1.6 m
linear platform, and about 4 h of POST rest/sleep. Because the maze is novel, PRE is a
within-session control: any apparent replay of the maze during PRE is a false positive
of the pipeline, measured on the same cells and the same event detector.

## What was analysed

Direction-specific place fields were built from 84 maze traversals for the 91 of 120
pyramidal cells that met a peak-rate and spike-count criterion (median spatial
information 0.85 bits/spike). The rightward and leftward field sets were stacked into a
single 100-state space so that one decoder represents both position and running
direction. Sharp-wave ripples were detected on the channel with the strongest
150-250 Hz band (7476 events in the session, median duration 54 ms; 0.20 Hz in PRE
non-REM and 0.40 Hz in POST non-REM). Candidate replay events were the place-cell
population bursts that contain a detected ripple (5209 events, median 175 ms). Inside
each event a posterior over position was decoded in 20 ms bins with `pynapple`'s
`decode_bayes`, and scored by the posterior-weighted correlation between decoded
position and time. Significance required beating two within-event nulls (column-cycle
shuffle and time-bin permutation) at p < 0.05, with the same best-of-two-directions
maximum applied to both nulls. The false-positive rate of the whole pipeline was
measured separately for PRE and POST by permuting which place field belongs to which
cell and re-running everything.

## Key finding

The decoder is sound: cross-validated decoding of real running position from these
place fields has a median error of 5.2 cm on a 1.6 m track (chance 49.8 cm) and
recovers the running direction in 95.7% of 250 ms bins. Applied inside ripples, it
finds replay only where experience makes replay possible. In PRE sleep 6.2% of SWR
events are called significant replay of the maze, and the cell-identity shuffle on the
same PRE events returns 6.2% as well, so PRE sits exactly on the noise floor (binomial
p = 0.50) as it should for a track the animal has never seen. The identical analysis on
POST sleep gives 8.4% against a 5.35% identity-shuffle floor (p = 2e-6), exceeds PRE
directly (Fisher p = 0.016), and shows higher per-event sequence quality than PRE
(Mann-Whitney on |weighted correlation|, p = 1e-4). Awake ripples recorded on the track
itself are strongest, with 20.7% of events containing a significant trajectory
(p = 5e-13). Since the ripple rate is also more than twice as high in POST as in PRE,
the rate of significant replay events per minute of sleep is about 2.5x higher after
the maze than before it.

The replayed trajectories look like compressed running. Significant events sweep across
the track at a median of 4.6 m/s, roughly eight times the animal's 0.56 m/s running
speed, and the individual posteriors (`fig05`) show the characteristic clean diagonal
band rather than a scatter of positions. One caveat: reverse-going trajectories
outnumber forward-going ones in every epoch, including PRE, where the content is at
chance. That asymmetry is therefore a property of the analysis, most likely the
non-uniform distribution of field peaks combined with the stereotyped time course of a
population burst, and should not be read as a biological forward/reverse bias. This is
also a single session from one animal; the PRE/POST contrast is well controlled within
the session, but the size of the effect should not be generalised from n = 1.

## Files

| file | contents |
| --- | --- |
| `hippocampal_replay_analysis.py` | jupytext (percent format) script, runs end to end in about 4 min |
| `hippocampal_replay_analysis.ipynb` | the same analysis as an executed notebook |
| `fig01_session_overview.png` | epochs, sleep scoring, full-session raster, tracking, linearised position |
| `fig02_place_fields.png` | direction-specific place fields, spatial information, example fields |
| `fig03_decoder_validation.png` | cross-validated decoding of real running position |
| `fig04_ripple_detection.png` | channel selection, example ripples, ripple-triggered average, spectra, event statistics |
| `fig05_example_replay_events.png` | eight decoded trajectories with the underlying spike rasters |
| `fig06_replay_summary.png` | population summary across epochs with the cell-identity controls |
| `replay_events.csv` | one row per scored event: epoch, duration, direction, weighted correlation, both p-values, slope |
| `replay_summary.csv`, `statistics.txt` | per-epoch summary and the statistical tests |

Running the script requires `pynapple`, `pynwb`, `h5py`, `remfile`, `xarray`, `scipy`,
`pandas`, `matplotlib` and `tqdm`. It streams from the DANDI S3 bucket and caches into
`$REMFILE_CACHE` (default `/tmp/remfile_cache_000044`). Changing `S3_URL` to another
asset of dandiset 000044 runs the same analysis on any of the other seven sessions.
