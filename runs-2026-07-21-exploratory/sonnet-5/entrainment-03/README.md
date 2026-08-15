# Theta Phase Entrainment of Hippocampal Neurons

## Dataset

[DANDI 000044](https://dandiarchive.org/dandiset/000044), "Diversity in
neural firing dynamics supports both rigid and learned hippocampal
sequences" (Grosmark & Buzsaki, NYU). We used subject **Gatsby**, session
`Gatsby_08022013`: bilateral CA1 silicon-probe recordings (80 sorted units:
25 in left CA1, 55 in right CA1; 66 putative excitatory/pyramidal cells and
14 putative inhibitory interneurons) recorded while the rat shuttled back
and forth on a 1.6 m linear track for reward. The NWB file (~7.9 GB) was
streamed directly from S3 with `remfile` (disk-cached locally, no full
download), and Pynapple was used for all time-series handling and analysis.

## Analysis

An LFP channel was selected objectively by scanning 14 channels spread
across the probe and picking the one with the largest fraction of spectral
power in the 6-10 Hz theta band; the winning channel showed a clean,
consistent ~7.6 Hz oscillation. That channel was band-pass filtered at 6-10
Hz and the Hilbert transform used to extract instantaneous theta phase and
amplitude. Running epochs were identified from the linearized position
(sustained locomotion above 3 cm/s), since theta is a locomotion-associated
rhythm and is weak or absent during immobility; because the video tracker
intermittently lost the animal (gaps up to ~80 s), position was first split
into contiguous tracked bouts and speed/running detection was computed
per-bout so that untracked gaps could never be misread as running. For
every unit, each spike fired during running was assigned the theta phase
at which it occurred, and a Rayleigh test for circular non-uniformity was
used to test whether spike phases cluster around a preferred phase, with
the mean resultant length (MRL) as a continuous measure of entrainment
strength. Two phase-free cross-checks were added to confirm the result
independently of the Hilbert-phase pipeline: the spike-triggered average of
the raw (unfiltered) LFP, and each unit's own spike-time autocorrelogram,
both of which should show theta-frequency (~125-150 ms) rhythmicity for
entrained cells. Finally, entrainment strength was compared between
putative pyramidal cells and interneurons, and theta-band LFP amplitude was
related to running speed as a sanity check that the effect is tied to
genuine locomotion-driven theta rather than an artifact of the filtering
pipeline.

## Key Findings

Theta phase entrainment was clearly present in CA1 during running: 44.6%
of 65 pyramidal cells with usable spike counts and 100% of 14 interneurons
showed significant phase locking (Rayleigh p < 0.001), with the two groups
showing similar mean phase-locking strength (mean MRL 0.24 for pyramidal
cells vs. 0.22 for interneurons) but a much higher fraction of interneurons
reaching significance, consistent with interneurons firing at higher rates
and thus having more statistical power to detect a real but comparably
sized phase preference, and consistent with the broader finding that
hippocampal interneurons are more consistently theta-entrained than
principal cells (Csicsvari et al., 1999). The phase-free cross-checks
corroborated this directly: the spike-triggered average LFP for the
entrained example units shows several clean cycles of theta-frequency
oscillation around the spike, and their spike-time autocorrelograms show a
central peak flanked by side peaks spaced roughly 125-150 ms apart (~7-8
Hz), the signature of theta-paced spiking, whereas a non-entrained
pyramidal cell's autocorrelogram is flat and noisy with no clear central
peak or side-peak structure. Theta-band LFP amplitude also increased
systematically with running speed (Pearson r = 0.11 across ~460,000 speed
samples, and a clear monotonic rise visible once binned by speed),
confirming that the oscillation being analyzed is the genuine
locomotion-associated theta rhythm rather than a filtering artifact.

## Files

- `theta_phase_entrainment_analysis.py` — consolidated jupytext analysis script (runs end-to-end)
- `theta_phase_entrainment_analysis.ipynb` — same analysis as an executed Jupyter notebook
- `fig1_raw_lfp_theta_filter.png` — raw vs. theta-band-filtered LFP and instantaneous phase
- `fig2_running_epochs.png` — linearized position/speed (gap-broken to show real tracker dropouts) with detected running epochs
- `fig2b_running_epochs_zoom.png` — zoomed-in view of a single track traversal and its detected running epoch
- `fig3_polar_phase_histograms.png` — spike-theta-phase polar histograms for example units (strongly entrained pyramidal cell, non-entrained pyramidal cell, strongly entrained interneuron)
- `fig4_population_entrainment_summary.png` — population summary: MRL distributions by cell type, fraction significantly phase-locked, MRL vs. firing rate, and preferred-phase polar scatter
- `fig5_sta_and_autocorrelogram.png` — phase-free confirmation: spike-triggered average LFP and spike-time autocorrelograms for the example units
- `fig6_theta_amplitude_vs_speed.png` — theta-band LFP amplitude vs. running speed
