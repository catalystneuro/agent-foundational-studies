# Theta Phase Entrainment of Hippocampal Neurons

## Dataset

[DANDI 000044](https://dandiarchive.org/dandiset/000044), "Diversity in
neural firing dynamics supports both rigid and learned hippocampal sequences"
(Grosmark & Buzsaki, NYU). We used subject **Achilles**, session
`Achilles_10252013`: bilateral CA1 silicon-probe recordings (137 units, 120
putative pyramidal cells and 17 putative interneurons) recorded while the rat
shuttled back and forth on a 1.6 m linear track for reward. The NWB file was
streamed directly from S3 with `remfile` (disk-cached locally, no full
download), and Pynapple was used for all time-series handling and analysis.

## Analysis

The LFP from a CA1 pyramidal-layer channel was band-pass filtered at 6-10 Hz
and the Hilbert transform used to extract instantaneous theta phase and
amplitude. Running epochs were identified from the linearized position
(sustained locomotion above 3 cm/s), since theta is a locomotion-associated
rhythm and is weak or absent during immobility. For every unit, each spike
fired during running was assigned the theta phase at which it occurred, and
a Rayleigh test for circular non-uniformity was used to test whether spike
phases cluster around a preferred phase, with the mean resultant length
(MRL) as a continuous measure of entrainment strength. Two phase-free
cross-checks were added to confirm the result independently of the
Hilbert-phase pipeline: the spike-triggered average of the raw (unfiltered)
LFP, and each unit's own spike-time autocorrelogram, both of which should
show theta-frequency (~125 ms) rhythmicity for entrained cells. Finally,
entrainment strength was compared between putative pyramidal cells and
interneurons, and theta-band amplitude was related to running speed as a
sanity check that the effect is tied to genuine locomotion-driven theta
rather than an artifact of the filtering pipeline.

## Key Findings

Theta phase entrainment was clearly present in CA1 during running: 57% of
120 pyramidal cells and 100% of 17 interneurons showed significant phase
locking (Rayleigh p < 0.001), with interneurons showing roughly double the
mean phase-locking strength of pyramidal cells (mean MRL 0.24 vs. 0.12),
consistent with the well-established finding that hippocampal interneurons
are more strongly and consistently theta-entrained than principal cells
(Csicsvari et al., 1999). The phase-free cross-checks corroborated this
directly: the spike-triggered average LFP for entrained example units shows
several clean cycles of theta-frequency oscillation around the spike, and
their spike-time autocorrelograms show a trough at zero lag flanked by
side peaks spaced roughly 125-150 ms apart (~7-8 Hz), the signature of
theta-paced spiking, whereas a non-entrained pyramidal cell's autocorrelogram
shows only a sharp refractory-period peak near zero lag with no longer-range
rhythmicity. Theta-band LFP amplitude also increased systematically with
running speed (Pearson r = 0.31 across ~10,000 speed samples), confirming
that the oscillation being analyzed is the genuine locomotion-associated
theta rhythm.

## Files

- `theta_phase_entrainment_analysis.py` — consolidated jupytext analysis script (runs end-to-end)
- `theta_phase_entrainment_analysis.ipynb` — same analysis as an executed Jupyter notebook
- `fig1_raw_lfp_theta_filter.png` — raw vs. theta-band-filtered LFP
- `fig2_running_epochs.png` — linearized position/velocity with running epochs used for the analysis
- `fig3_polar_phase_histograms.png` — spike-theta-phase polar histograms for example units (strongly entrained pyramidal cell, non-entrained pyramidal cell, strongly entrained interneuron)
- `fig4_population_entrainment_summary.png` — population summary: MRL distributions by cell type, fraction significantly phase-locked, MRL vs. firing rate, and preferred-phase polar scatter
- `fig5_sta_and_autocorrelogram.png` — phase-free confirmation: spike-triggered average LFP and spike-time autocorrelograms for the example units
- `fig6_theta_amplitude_vs_speed.png` — theta-band LFP amplitude vs. running speed
