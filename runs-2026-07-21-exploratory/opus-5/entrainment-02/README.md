# Theta phase entrainment of hippocampal neurons (DANDI:000059)

This analysis demonstrates theta phase entrainment of hippocampal neurons using
[DANDI:000059](https://dandiarchive.org/dandiset/000059), "Cooling of Medial Septum
Reveals Theta Phase Lag Coordination of Hippocampal Cell Assemblies" (Petersen &
Buzsáki, *Neuron* 2020). Rats run a spatial alternation maze while silicon probes
record the hippocampus, and on a subset of trials the medial septum is cooled through
an implanted Peltier device, which slows the theta rhythm without abolishing it. Five
sessions from three subjects (MS13, MS21, MS22) were analysed. Each session is
distributed as two NWB assets, a raw-ephys file that also carries a 1250 Hz LFP series
and a processed file with spike-sorted units, position, running speed, septal
temperature and trial intervals; both were streamed from S3 with `remfile` and a local
disk cache, and their shared clock was verified before use (last spike time matches the
LFP duration to within 0.05 s). All analysis is done with pynapple, and the GLM control
with NeMoS.

For every curated unit, the theta phase (5-11 Hz, Hilbert) of the LFP on the channel
the original authors flagged as the theta reference was read at the time of each spike
during locomotion, and the mean resultant length (MRL) and preferred phase of that
circular distribution were computed. Chance level was set per unit by circularly
shifting its own spike train within the running epochs 200 times, which preserves the
unit's spike count and inter-spike-interval structure while destroying its alignment to
the LFP. Phase 0 is the peak of the band-passed LFP and 180° its trough.

## Key finding

**310 of 483 units (64%) are individually phase-locked to theta**, with a median MRL of
0.073 against a median chance level of 0.025, and the fraction of locked units is
between 50% and 69% in every session. Locking is not an artefact of spike-count or
burst structure: it is present in the spike-triggered LFP average (theta-rhythmic and
many times larger than the circular-shift band for locked units, flat for the weakest
ones), the preferred phase repeats across interleaved subsets of each session (median
|Δphase| 23° for the better-locked half of units), and a Poisson GLM that already
contains running speed and the unit's own spike history still gains held-out
log-likelihood when a cyclic basis over theta phase is added (31 of the 40 tested units
improve, the gain per spike tracks each unit's MRL at r = 0.89, and the nine that do
not improve are the weakly locked ones). Putative interneurons are
more strongly entrained than putative pyramidal cells (bias-corrected MRL 0.095 vs
0.060, Mann-Whitney p = 2e-6) and prefer a phase 62° earlier in the cycle (permutation
p = 4e-4), the classic separation between the two cell classes.

The cooling manipulation confirms that the measurement tracks the oscillation rather
than a fixed clock in the recording. Cooling the septum slowed theta by 0.66 Hz on
average (8.12 to 7.44 Hz, in all five sessions) and reduced theta amplitude to 82% of
baseline, yet phase locking not only survived but strengthened: with the comparison
matched on both observation duration and spike count, the median MRL rose from 0.082 to
0.117 and 72% of units moved above unity (Wilcoxon p = 9e-24), while preferred phases
were largely preserved (median |Δphase| 22°) and firing rates changed little (median
ratio 0.90). Spikes therefore follow theta to its new frequency, which is what
entrainment to the oscillation, rather than incidental correlation with it, predicts.

## Files

| File | Contents |
| --- | --- |
| `theta_phase_entrainment.py` / `.ipynb` | Consolidated jupytext notebook that runs the whole analysis end to end |
| `theta_lib.py` | Streaming/loading from DANDI, filtering, epoch definitions, circular statistics |
| `analysis.py` | Per-session phase-locking pipeline, circular-shift shuffles, cell-type classification |
| `scan_sessions.py` | Session screening that produced the five-session selection |
| `01_load_data.py` | Single-session load and data-stream validation (figs 1-2) |
| `02_run_all_sessions.py` | Runs all sessions, caches per-unit statistics to `results/` |
| `03_population.py` | Population figures and summary statistics (figs 3-5) |
| `04_examples.py` | Spike-triggered averages and split-half stability (fig 6) |
| `05_cooling.py` | Matched comparison of cooled versus normal theta (fig 7) |
| `06_glm.py` | NeMoS Poisson GLM control for speed and spike history (fig 8) |

Figures:

- `fig01_raw_data_validation.png`: LFP, band-passed theta, extracted phase, spike raster, running speed, septal temperature
- `fig02_power_spectrum.png`: theta peak during running versus rest; theta amplitude versus speed
- `fig03_example_units.png`: spike-phase histograms for strongly, moderately and non-locked units
- `fig04_population_summary.png`: MRL versus chance, per-unit tests, cell-type comparison, per-session consistency
- `fig05_population_heatmap.png`: all locked units sorted by preferred phase, over two theta cycles
- `fig06_sta_and_stability.png`: spike-triggered LFP averages with shuffle bands; odd/even split-half consistency
- `fig07_cooling.png`: theta frequency and amplitude under cooling; matched locking comparison
- `fig08_glm.png`: held-out log-likelihood gain from the theta-phase term

## Notes and caveats

Running was defined as smoothed speed above 5 cm/s; on this maze the animals are
almost never immobile, so the no-theta contrast in figure 2 comes from the post-maze
rest period rather than from within-maze immobility. Cell types were assigned from
firing rate and autocorrelogram burstiness because this dandiset does not distribute
spike waveforms, so the pyramidal/interneuron labels are putative. The MRL test is
conservative: some units that fail it still show spike-triggered LFP averages above
their own shuffle band, so 64% should be read as a lower bound on the fraction of
entrained units. Phase is referenced to a single channel per session, so preferred
phases are comparable within a session but only approximately across sessions, since
the reference channel's depth relative to the CA1 layers is not identical in each.
