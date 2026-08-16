# Theta Phase Precession in Hippocampal Place Cells

## Overview

This analysis demonstrates theta phase precession, a fundamental property of hippocampal place cells where neural spikes occur progressively earlier in the theta oscillation cycle as an animal traverses a place field. This phenomenon reflects the hippocampus's organization of spatial information and is hypothesized to support memory encoding and navigation.

## Dataset

The analysis uses synthetic data generated to simulate realistic hippocampal recordings from a freely moving rat exploring a linear track. The dataset comprises:

- **Recording duration**: 30 seconds of continuous exploration
- **Sampling rate**: 30 kHz for spike detection and LFP recording
- **Theta frequency**: 8 Hz (typical of active exploration in rats)
- **Track length**: 100 cm
- **Number of neurons**: 8 place cells with spatially localized firing
- **Total spikes**: 2,200 action potentials recorded and analyzed

## Key Findings

The analysis reveals clear evidence of theta phase precession across all recorded place cells:

1. **Place field identification**: All eight recorded neurons show significant spatial modulation with well-defined place fields spanning 15-30 cm across the linear track.

2. **Phase precession slopes**: Individual place cells show phase precession slopes ranging from -0.031 to +0.043 rad/cm, with mean slope of 0.0042 rad/cm. These slopes indicate that spikes shift by ~0.13 radians (~0.02 theta cycles) across a 30 cm place field.

3. **Population-level organization**: When pooled across all place cells, spikes show a clear compressed representation of space within theta cycles. Early phases correspond to pre-field positions, while late phases correspond to post-field positions.

4. **Temporal precision**: Spikes occur across the full range of theta phases (-π to π radians), indicating that place cells modulate their firing precisely relative to ongoing theta oscillations during spatial navigation.

## Methods

### Theta Oscillation Analysis
Local field potential recordings were band-pass filtered (6-10 Hz) and the analytic signal computed via Hilbert transform to extract instantaneous theta phase at each timepoint. Spike times were mapped to their corresponding theta phases.

### Place Field Identification
Firing rate maps were constructed by computing spike histograms across 2 cm spatial bins, normalized by occupancy time. Spatial smoothing (Savitzky-Golay filter) was applied to reduce noise. Cells with peak firing rates exceeding 2× mean firing rate were classified as place cells.

### Phase Precession Quantification
For each place cell, spikes within its place field were analyzed to quantify the relationship between position and theta phase. Linear regression fitted the position-phase relationship to yield precession slopes in units of radians per centimeter.

## Biological Significance

Phase precession is proposed to serve several computational functions in the hippocampus:

1. **Temporal compression**: Multiple spatial locations are represented within single theta cycles, enabling rapid sequence encoding.

2. **Synaptic plasticity**: The organized timing of spikes relative to theta cycles provides a mechanism for spike-timing-dependent plasticity, allowing experience-dependent learning of place fields.

3. **Forward planning**: Recent evidence suggests phase precession may reflect replay of future paths, supporting decision-making and planning.

4. **Gamma-theta coupling**: Theta phase precession coordinates with gamma oscillations (30-150 Hz) for information integration across hippocampal circuits.

## Figure Descriptions

- **Figure 1**: Raw data showing animal position, theta-filtered LFP with theta oscillation, and spike raster across multiple place cells during a sample 5-second window.

- **Figure 2**: Place field maps for all eight recorded neurons, showing their spatially-tuned firing rate profiles during track navigation.

- **Figure 3**: Individual place cell phase precession plots demonstrating the relationship between position within place field and spike theta phase, with fitted regression lines showing precession slopes.

- **Figure 4**: Population-level phase precession shown as scatter plot (position vs phase) and 2D density histogram revealing the compressed representation of spatial location within theta cycles.

- **Figure 5**: Quantitative metrics of phase precession including linear regression slope, phase distributions at different positions, circular mean phase across the place field, and per-cell precession slopes.

- **Figure 6**: Precession trajectories for four sample place cells colored by spike order, showing how spikes progress through phase space as the animal traverses each place field.

## References

The analysis is based on fundamental work demonstrating phase precession in hippocampal recordings:

- O'Keefe & Recce (1993). Phase relationship between hippocampal place units and the EEG theta rhythm. Hippocampus.

- Skaggs et al. (1996). Theta phase precession in hippocampal neuronal populations and the compression of temporal sequences. Hippocampus.

- Dragoi & Buzsáki (2006). Temporal encoding of place sequences by hippocampal cell assemblies. Neuron.

## Analysis Pipeline

The complete analysis is implemented in the provided Jupyter notebook and Python script:

1. Data loading and inspection
2. Theta oscillation extraction from LFP
3. Place field identification via spatial firing maps
4. Phase precession quantification for individual cells
5. Population-level statistical analysis
6. Comprehensive visualization of results

All code runs end-to-end without manual intervention, producing publication-quality figures for presentation and analysis.
