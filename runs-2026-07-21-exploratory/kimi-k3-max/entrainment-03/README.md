# Theta Phase Entrainment of Hippocampal CA1 Neurons

## Dataset

DANDI Archive dandiset [000044](https://dandiarchive.org/dandiset/000044) ("Diversity in neural firing dynamics...", Buzsaki lab), session `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb` (version 0.250624.0426). The recording is rat CA1 with tetrodes during behavior on a 1.6 m linear maze: 137 sorted units (120 putative excitatory, 17 putative inhibitory; 72 right CA1, 65 left CA1) and a 128-channel LFP at 1250 Hz. The file is streamed with remfile plus a local disk cache; nothing is downloaded in full.

## Analysis

The goal is to demonstrate theta phase entrainment: the tendency of CA1 neurons to fire at preferred phases of the ongoing 4-12 Hz theta rhythm during active exploration. The pipeline (see `theta_phase_entrainment.py`, runnable end-to-end, and the converted `theta_phase_entrainment.ipynb`):

1. Pick a theta reference channel. Because the units' `shank_id` metadata does not map onto the LFP electrode groups, a single session-wide reference is used: the channel with the highest theta/delta power ratio on a 200 s maze-epoch chunk (channel 117, theta peak 9.25 Hz).
2. Bandpass filter the maze-epoch LFP at 4-12 Hz and take the angle of the Hilbert transform as instantaneous theta phase (0 = LFP peak).
3. Restrict to running. Speed comes from the 2D position series (see note below on a timestamp quirk); bouts with smoothed speed above 10 cm/s, merged across gaps under 0.5 s, give 209 bouts totaling 1137 s of the 2067 s maze epoch.
4. For each of the 137 units (all had at least 50 spikes in bouts), interpolate theta phase at spike times and compute the mean resultant length (MRL), preferred phase, and a Rayleigh p-value. A random-time control redraws the same number of spikes at random times inside the bouts, preserving spike count and the marginal phase distribution.

## Key Finding

Theta entrainment is widespread and cell-type specific. 104 of 137 units (76%) are significantly phase locked (Rayleigh p < 0.01): 88/120 excitatory and 16/17 inhibitory. Inhibitory interneurons lock much more strongly (median MRL 0.259) than pyramidal cells (median MRL 0.109; Mann-Whitney p = 5.7e-5), while the random-time control sits near 0.02, so the locking reflects genuine spike-phase coupling rather than structure of the theta waveform itself. Preferred phases are tightly clustered: pyramidal cells pool at -139 degrees relative to the reference LFP peak (descending phase, approaching the trough) and interneurons at +123 degrees. Since the layer of the reference channel is unknown, absolute phase values are convention-dependent; the robust results are the prevalence, strength, and cell-type specificity of the entrainment. Locking strength is only weakly related to firing rate, so theta organizes spike timing even for sparse place cells.

## Notes

- Data quirk: the position SpatialSeries stores the sampling period (0.0256 s) in the NWB `rate` field, so naive timestamps are spaced 39 s apart. The pipeline reconstructs correct timestamps from the true 39.06 Hz rate. Speed is interpolated only across tracking dropouts shorter than 0.5 s; longer dropouts are treated as stationary.
- Figures: `fig_theta_channel_selection.png`, `fig_raw_lfp_theta.png`, `fig_speed_running_bouts.png`, `fig_example_phase_histograms.png`, `fig_mrl_distributions.png`, `fig_preferred_phase_distribution.png`, `fig_mrl_vs_rate.png`. Per-unit statistics are in `unit_phase_locking.csv`.
- Intermediate pipeline scripts (`01`-`04`) were used for development; the consolidated jupytext script is self-contained.
