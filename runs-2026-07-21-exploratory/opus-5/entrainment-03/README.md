# Theta phase entrainment of hippocampal CA1 neurons

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark & Buzsáki (2016), the
`hc-11` recordings. Bilateral CA1 silicon-probe recordings from rats running back and forth
on a linear track for water reward, with sleep sessions before and after the maze. Each NWB
file contains sorted units labelled as putative pyramidal cells or interneurons, a
128-channel LFP `ElectricalSeries` at 1250 Hz, tracked position, and hand-scored REM /
non-REM sleep intervals. Five sessions from four rats were analysed (Achilles ×2, Cicero,
Gatsby, Buddy), giving 430 units with enough spikes to test.

Files are streamed from S3 with `remfile` plus a disk cache; nothing is downloaded in full.
The LFP dataset is 43M × 128 `int16` chunked as (170221, 1), so reading a single channel is
cheap. All time-series handling and the tuning-curve computations use Pynapple.

## What was analysed

For every unit, the distribution of LFP theta phases at which it spikes. Theta phase is the
Hilbert phase of the 6–10 Hz bandpass-filtered LFP, taken from the channel with the highest
6–10 Hz to 1–4 Hz power ratio in that session, with 0° defined as the peak of the filtered
signal. Each running interval is filtered separately so that phase is not smeared across
the gaps between traversals. Locking is quantified by the mean resultant length (MRL) and
by pairwise phase consistency (PPC, which is not biased upward by low spike counts), and
tested both with a Rayleigh test under Benjamini–Hochberg FDR correction within session and
against a ±0.5 s spike-jitter null that preserves each unit's spike count and slow rate
profile while destroying the theta relationship.

## Key finding

Across 430 CA1 units, **87% (372/430) are significantly locked to the theta phase during
running** (Rayleigh, FDR q < 0.05 within session), and 374/430 also exceed the spike-jitter
null at p < 0.05. The effect replicates in every session individually, at 82% to 98% of units.
Putative interneurons are more strongly locked than pyramidal cells (median MRL 0.20 vs
0.12; 96% vs 84% significant) and fire near the trough of the pyramidal-layer LFP theta
(mean preferred phase 178°, where 0° is the filtered LFP peak), while pyramidal cells fire
later in the cycle (227°). Both classes spread their preferred phases broadly around those
means, so the population tiles the theta cycle rather than firing in one synchronous burst.
Theta phase precession provides independent support that the phase assignment is
behaviourally meaningful: of 24 direction-specific place fields tested in the prototype
session, 13 show a significant circular-linear correlation between position in the field
and theta phase, 12 of them with a negative slope, median -0.58 theta cycles per traversal.

One caveat is worth stating because it cuts against a naive reading of the result. Bandpass
filtering followed by a Hilbert transform assigns a phase to any signal, including one with
no oscillation in the band, so a high MRL is not by itself evidence of a rhythm. In this
data the median MRL during non-REM sleep is statistically indistinguishable from running
(Wilcoxon signed-rank p = 0.42) even though the non-REM LFP has no theta peak in its power
spectrum. What does separate the states is the rhythmicity of each unit's spike-triggered
LFP average, which shows repeating theta-period side lobes during running and REM and only
a single transient during non-REM. Measured as the theta/delta band-power ratio of each
unit's own STA waveform, the median is 14.5 during running and 35.7 during REM but 0.69
during non-REM (Wilcoxon signed-rank p ≈ 1e-21 for running vs non-REM). The entrainment claim above rests on that
control together with the precession result, not on the MRL alone.

## Files

| File | Contents |
| --- | --- |
| `theta_phase_entrainment.py` | Consolidated jupytext script, runs end to end |
| `theta_phase_entrainment.ipynb` | Same, as a notebook |
| `theta_utils.py` | Loading, filtering, circular statistics, shared by all scripts |
| `01_load_inspect.py` | Stage 1: load one session, validate every data stream |
| `02_theta_channel.py` | Stage 2: theta channel selection and phase validation |
| `03_phase_locking.py` | Stage 3: per-unit phase locking, single session |
| `04_visualize.py` | Stage 4: figures 4–7 |
| `05_multisession.py` | Stage 5: all five sessions, pooled |
| `06_phase_precession.py` | Stage 6: phase precession control |
| `phase_locking_single_session.csv` | Per-unit statistics for Achilles-10252013, three brain states |
| `phase_locking_all_sessions.csv` | Per-unit statistics for all 430 units |
| `phase_precession.csv` | Per-field circular-linear fits |

### Figures

| Figure | Contents |
| --- | --- |
| `fig01_raw_data_overview.png` | Raw LFP, theta filter and envelope, spike raster, speed |
| `fig02_behavior_validation.png` | Tracked position, track traversals, speed distribution |
| `fig03_theta_channel_and_spectra.png` | Channel selection, power spectra by state, peak-triggered LFP average |
| `fig04_theta_cycle_entrainment.png` | Spikes tracking the theta cycle; phase tuning of every locked unit |
| `fig05_example_units.png` | Spike-phase histograms of six example units |
| `fig06_population_summary.png` | Locking strength, jitter null, preferred phases, PPC |
| `fig07_state_comparison.png` | Brain-state control: MRL vs spike-triggered rhythmicity |
| `fig08_multisession_summary.png` | Replication across five sessions |
| `fig09_phase_precession.png` | Phase precession in individual place fields and pooled |

## Reproducing

```bash
python theta_phase_entrainment.py
```

The consolidated script reuses `phase_locking_all_sessions.csv` and `phase_precession.csv`
when they are present; delete them to force a full recomputation, which re-runs the
five-session sweep (the 128-channel scan per session is the slow step) but not the
precession permutation test, which stays in `06_phase_precession.py`. Requires `pynapple`,
`pynwb`, `remfile`, `h5py`, `scipy`, `pandas`, `matplotlib` and `tqdm`.
