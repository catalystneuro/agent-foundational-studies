# Theta Phase Entrainment of Hippocampal CA1 Neurons

## Dataset

DANDI Archive dandiset [000044](https://dandiarchive.org/dandiset/000044) (Buzsaki
lab, "Diversity in neural firing dynamics supports both rigid and learned hippocampal
sequences"), session `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`. The
session contains 137 sorted CA1 units (120 excitatory, 17 inhibitory) and a
128-channel LFP recorded at 1250 Hz while a rat ran on a 1.6 m linear maze
(~34.5 min maze epoch). Data were streamed with remfile plus a local disk cache; no
bulk download was needed.

## Analysis

The goal was to demonstrate theta phase entrainment: the tendency of hippocampal
neurons to fire at preferred phases of the 4-12 Hz theta rhythm during locomotion.
A theta reference channel was chosen data-driven as the channel with the largest
theta/delta power ratio (channel 117, clear spectral peak near 8 Hz). The maze-epoch
LFP was bandpass filtered at 4-12 Hz (zero-phase Butterworth) and the instantaneous
phase taken from the Hilbert transform (0 = peak of the filtered signal, +/-pi =
trough). Run bouts were defined from the 2D position as smoothed speed above
10 cm/s (179 bouts, ~1000 s total). For each of the 135 units with at least 100
spikes in run bouts, spike phases were computed by circular-safe interpolation, and
phase locking was quantified with the mean resultant length R, the preferred phase,
and a Rayleigh test, plus a circular-shift null control (100 shuffles of the phase
signal relative to the spike trains).

## Key Finding

The large majority of CA1 neurons are significantly entrained to theta: 95/118
excitatory and 16/17 inhibitory units pass the Rayleigh test at p < 0.05 (87/118 and
16/17 against the stricter shuffle null), and the population mean R of 0.143 sits
far outside the null distribution (0.035 +/- 0.002, shuffle p = 0.01). Inhibitory
interneurons lock about twice as strongly as pyramidal cells (median R 0.254 vs
0.109), consistent with the literature. Preferred phases are non-uniformly
clustered for both cell classes, and the sorted population phase-tuning heatmap
shows the expected tiling of the theta cycle across units.

## Outputs

- `theta_phase_entrainment.py` : consolidated jupytext script, runs end-to-end
- `theta_phase_entrainment.ipynb` : executed notebook version
- `figures/fig1_psd_channel_selection.png` : LFP spectrum and reference-channel choice
- `figures/fig2_raw_lfp_phase_raster.png` : raw LFP, theta phase, and phase-sorted spike raster during a run bout
- `figures/fig3_example_phase_histograms.png` : phase histograms for example excitatory and inhibitory units
- `figures/fig4_population_summary.png` : R distributions, shuffle control, preferred-phase distributions
- `figures/fig5_phase_preference_heatmap.png` : normalized phase tuning of all significantly locked units
- `phase_locking_results.csv` : per-unit statistics (R, preferred phase, Rayleigh and shuffle p-values)

One caveat: the NWB electrodes table does not annotate which LFP channel sits in the
pyramidal layer, so absolute phase values are relative to the data-driven reference
channel and the phase convention above; only differences between units are
layer-independent.
