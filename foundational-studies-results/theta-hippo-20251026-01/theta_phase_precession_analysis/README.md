# Theta Phase Entrainment and Precession Analysis

## Overview
This analysis demonstrates theta phase entrainment and precession in hippocampal neurons using human intracranial recordings from DANDI Archive.

## Dataset
**DANDI:000940** - Hippocampal Theta Phase Precession Supports Memory Formation and Retrieval of Naturalistic Experience in Humans

- **Citation**: Zheng et al. (2024) Nature Human Behavior
- **Species**: Homo sapiens (human subjects with epilepsy)
- **Recording**: Intracranial depth electrodes with Behnke-Fried microwires
- **Brain regions**: Hippocampus
- **Task**: Cognitive boundary task (episodic memory)
- **Data types**: 
  - LFP recordings (250 Hz)
  - Single unit spike times (43 units in sub-1)
  - Behavioral events and trial structure

## Theta Phase Phenomena

### Theta Phase Entrainment
The tendency of hippocampal neurons to fire at preferred phases of the theta oscillation (4-12 Hz). This reflects the temporal organization of neural activity within theta cycles.

### Theta Phase Precession
The systematic shift in spike timing relative to theta phase as an animal (or in this case, a human) progresses through an experience or navigates through space. Neurons fire at progressively earlier phases of the theta cycle.

## Analysis Pipeline

1. **Data Loading**: Stream NWB file from DANDI using LINDI with local caching
2. **LFP Processing**: Extract and filter LFP in theta band (4-12 Hz)
3. **Phase Extraction**: Calculate instantaneous theta phase using Hilbert transform
4. **Phase Entrainment**: Compute phase locking of spikes to theta oscillations
5. **Phase Precession**: Analyze systematic phase shifts across trials/time
6. **Visualization**: Create phase histograms, raster plots, and precession plots

## Key Metrics

- **Phase Locking Value (PLV)**: Measures consistency of firing at specific theta phases
- **Mean Resultant Length**: Strength of phase preference
- **Rayleigh Test**: Statistical significance of phase locking
- **Phase-Time Slope**: Quantifies precession rate

## References

Zheng, J., Yebra, M., Schjetnan, A.G.P., et al. (2024). Hippocampal Theta Phase Precession Supports Memory Formation and Retrieval of Naturalistic Experience in Humans. Nature Human Behavior (in press).
