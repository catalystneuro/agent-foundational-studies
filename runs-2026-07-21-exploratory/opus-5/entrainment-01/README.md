# Theta phase entrainment of hippocampal CA1 neurons

## Dataset

[DANDI 000044](https://dandiarchive.org/dandiset/000044), Grosmark and Buzsáki
(2016), *Diversity in neural firing dynamics supports both rigid and learned
hippocampal sequences*, Science 351:1440. Bilateral silicon-probe recordings
from dorsal CA1 of freely moving rats. Each session contains a pre-task sleep
block, a run on a 1.6 m linear track, and a post-task sleep block. Every NWB
file carries a 1250 Hz LFP `ElectricalSeries`, spike-sorted units labelled
`excitatory` or `inhibitory`, position tracking, and scored sleep states.

Four sessions from three animals were analysed: `Achilles-10252013`,
`Achilles-11012013`, `Cicero-09012014`, `Gatsby-08022013`. The files are 5-9 GB
each and were never downloaded in full; the LFP dataset is chunked one channel
at a time, so streaming with `remfile` plus a local disk cache reads only the
bytes for the single channel used for phase estimation. All time-series handling
and analysis uses Pynapple.

## What was analysed

For each session, the LFP channel with the largest 6-10 Hz over 1-4 Hz power
ratio during running was selected, band-pass filtered at 6-10 Hz, and converted
to an instantaneous phase with the Hilbert transform (phase 0 = peak of the
filtered cycle, π = trough). The theta phase at every spike of every unit was
read out with Pynapple, and locking was summarised per unit by the mean
resultant length (MRL) of those phases together with the preferred phase. The
significance of each unit's locking was assessed against a circular-shift null
that rotates the phase series relative to the spike train, preserving spike
count, burst structure, and the marginal phase distribution; the Rayleigh test
is reported alongside but is anti-conservative for spike trains. The analysis
was repeated in REM sleep, in non-REM sleep, in 3 Hz bands from 1.5 to 41.5 Hz,
and split by theta amplitude.

## Key finding

Hippocampal CA1 neurons are strongly entrained by the theta rhythm during
locomotion. Across four sessions, 307 of 377 units (81%) with at least 100
spikes during running fired at a significantly non-uniform theta phase by the
circular-shift test, and 265 also survived a Bonferroni-corrected Rayleigh test.
Putative interneurons were entrained substantially more strongly than putative
pyramidal cells (median MRL 0.173 versus 0.102, Mann-Whitney p = 2.3e-10) and
were locked almost without exception (97% versus 78% of units). The entrainment
is band-specific and amplitude-dependent: repeating the analysis across
frequency bands gives a clear peak in the theta band during running and REM but
a monotonically decreasing, peakless curve in non-REM, and 69% of units are more
strongly locked on high-amplitude than on low-amplitude theta cycles (Wilcoxon
p = 2.1e-17). Units that lock in both running and REM tend to keep a similar
preferred phase across the two states, though the relationship is loose (n = 238,
circular concentration of the run-to-REM phase difference 0.32, median offset
+17°, Rayleigh z = 24).

Non-REM sleep is a partial rather than a clean negative control, and the
discrepancy is informative. The non-REM power spectrum has no theta peak, yet
band-passing the signal at 6-10 Hz still produces a phase, and nearly every unit
appears "locked" to it. The population map (`fig05`) shows why: in non-REM every
unit prefers almost the same phase, the signature of synchronous population
bursts (sharp-wave ripples) creating a transient that the band-pass filter turns
into a consistent phase estimate. During running and REM the preferred phases
instead tile the cycle, which is what genuine entrainment of individual cells to
an ongoing oscillation looks like. After matching spike counts across states,
median MRL is 0.139 in REM, 0.117 during running, and 0.095 in non-REM. Two
caveats: the absolute preferred phase depends on where the channel sits relative
to the CA1 layers, since theta reverses phase across the pyramidal layer and the
fissure, so only within-session phase structure should be compared; and locking
is measured here irrespective of position on the track, so it averages over any
phase precession within place fields and therefore understates pyramidal-cell
locking.

## Files

| File | Contents |
| --- | --- |
| `theta_phase_entrainment.py` | Consolidated jupytext notebook, self-contained, runs end to end |
| `theta_phase_entrainment.ipynb` | The same notebook, executed |
| `hc11_io.py` | Streaming NWB access helpers for DANDI 000044 |
| `theta.py` | Filtering, phase, circular statistics, circular-shift null |
| `compute_session.py` | Per-session pipeline with on-disk caching |
| `figures.py` | Figure builders |
| `01_load_data.py`, `02_theta_signal.py`, `03_analyze_phase_locking.py`, `04_make_figures.py` | Stepwise development pipeline |
| `phase_locking_all_sessions.csv` | One row per unit per state: MRL, preferred phase, p-values, spike counts |
| `fig01`-`fig08` `.png` | Figures |

### Figures

- `fig01_raw_streams.png` — raw LFP, position, speed, and spike raster
- `fig02_theta_signal.png` — theta extraction and per-state power spectra
- `fig02b_theta_validation.png` — running-versus-immobility validation of the channel choice
- `fig03_spike_phase_excerpt.png` — spikes of the most locked units against single theta cycles
- `fig04_example_units.png` — phase histograms, polar plots, and shuffle tests for two example units
- `fig05_population_map.png` — phase-resolved firing of every unit in each state
- `fig06_population_stats.png` — pooled MRL distributions, significance, preferred phases, run-versus-REM
- `fig07_frequency_specificity.png` — locking across frequency bands, by theta amplitude, and by state
- `fig08_across_sessions.png` — replication across sessions and animals

## Running it

```
pip install pynapple pynwb h5py remfile tqdm matplotlib pandas scipy jupytext
python theta_phase_entrainment.py          # or run the notebook
```

The first run streams roughly 100-200 MB per session and takes about 5 minutes
per session; results are cached to `nb_cache_<session>.pkl`, so later runs are
immediate.
