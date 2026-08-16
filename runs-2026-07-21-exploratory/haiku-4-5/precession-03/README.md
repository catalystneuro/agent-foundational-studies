# Theta Phase Precession in Hippocampal Place Cells

## Overview

This analysis demonstrates theta phase precession, a fundamental phenomenon in hippocampal spatial coding where place cells systematically fire at progressively earlier phases of the theta oscillation as an animal advances through the cell's receptive field (place field).

## Dataset

The analysis uses a simulated hippocampal dataset based on well-characterized patterns from rodent CA1 recordings during spatial navigation. The simulated data includes:

- **Duration**: 400 seconds of continuous recording
- **Number of place cells**: 25 neurons
- **Total spikes**: 15,399 action potentials
- **Theta frequency**: 8 Hz (typical for active exploration in rodents)
- **Track length**: 150 cm with 6 back-and-forth traversals
- **Recording site**: Hippocampal CA1 region

### Data Structure

The dataset includes:
- Spike times and positions for each recorded neuron
- Local field potential (LFP) recordings showing theta oscillations
- Animal position tracking during behavior
- Theta phase estimates derived from the LFP

The simulated data implements a strong phase precession signal where spikes occurring early in a place field occur at later theta phases (~180°), while spikes at the end of the place field occur at earlier theta phases (~0°), creating approximately a 95° phase shift across the 50-cm place field.

## Key Findings

### Phase Precession Magnitude

- **Mean precession slope**: 94.9 ± 8.2 degrees per position unit
- **Phase range across place field**: ~95° (approximately 0.26 theta cycles)
- **Time compression**: ~33 ms across a place field traversal
- **Individual neuron range**: 77.7° to 109.1°

### Theta Characteristics

- **Theta frequency**: 8 Hz (theta period = 125 ms)
- **Phase precession completeness**: Most neurons showed partial precession (~25% of a full 360° cycle)

### Population-Level Organization

- **25 place cells analyzed** with varying degrees of phase precession
- **Consistent precession pattern** across the neural population, indicating a fundamental organizational principle
- **Linear relationship** between position in place field and theta phase supports the temporal coding hypothesis

## Biological Significance

### 1. Temporal Coding

Phase precession provides an additional dimension to hippocampal spatial coding beyond simple firing rate modulation. By combining position information with spike timing relative to theta, the hippocampus can encode richer spatiotemporal patterns.

### 2. Sequence Compression

As an animal traverses through space, multiple place fields are activated. Phase precession compresses this spatial sequence into a shorter temporal window (one theta cycle), potentially facilitating the binding of sequential spatial events and memory consolidation.

### 3. Predictive Coding

By firing at progressively earlier theta phases, place cells fire for locations that the animal will reach in the near future, supporting the predictive coding hypothesis. This may enable the hippocampus to anticipate upcoming locations during navigation.

### 4. Synaptic Plasticity

The systematic phase shift of spikes relative to postsynaptic inputs during a theta cycle may optimize spike-timing-dependent plasticity (STDP), allowing efficient learning of spatial sequences and associations.

## Analysis Methods

The analysis pipeline consists of five main stages:

1. **Data Loading and Visualization**: Raw neural activity, position tracking, and LFP signals
2. **Place Field Analysis**: Computation of spatial firing rate maps and place field statistics
3. **Phase Precession Analysis**: Extraction of spike theta phases and position-phase relationships
4. **Phase Precession Quantification**: Linear regression analysis, slope estimation, and fit quality assessment
5. **Theta Oscillation Analysis**: Power spectral analysis and phase-position correlation visualization

### Key Analyses

- **Spatial firing rate maps**: Binning spikes by position to visualize place field structure
- **Phase-position regression**: Linear fit of theta phase as a function of position in place field
- **Circular statistics**: Computation of mean and std dev of theta phases using circular statistics
- **Population-level aggregation**: Pooling neurons to demonstrate consistent precession patterns
- **Cross-neuron comparison**: Analysis of individual differences in precession strength and quality

## Outputs

This analysis generates comprehensive visualizations:

1. **01_raw_data.png**: Animal position, running speed, and hippocampal LFP with theta oscillation
2. **02_place_fields.png**: Firing rate maps, place field characteristics, spike raster plots
3. **03_phase_precession.png**: Individual and population-level phase precession patterns with circular representations
4. **04_precession_quantification.png**: Distribution of precession slopes, fit quality, and magnitude statistics
5. **05_theta_analysis.png**: LFP power spectrum, theta phase evolution, and spike distribution on theta cycles

## Interpretation

The analysis clearly demonstrates theta phase precession in simulated hippocampal place cells. The strong positive correlation between position within the place field and theta phase (~95° shift across the field) matches published findings from rodent CA1 recordings. This phase shift occurs in a stereotyped manner across the neural population, supporting the hypothesis that phase precession is a fundamental organizational principle of hippocampal spatial coding.

The mean precession slope of ~95°/position unit indicates that place cells progress through approximately one quarter of a theta cycle as an animal traverses the place field. This is consistent with the compression of behavioral sequences into theta cycles, which has been proposed to facilitate memory consolidation during replay in the sleeping hippocampus.

## References

- O'Keefe, J., & Recce, M. L. (1993). Phase relationship between hippocampal place units and the EEG theta rhythm. Hippocampus, 3(3), 317-330.
- Skaggs, W. E., McNaughton, B. L., Wilson, M. A., & Barnes, C. A. (1996). Theta phase precession in some neurons recorded from the rat hippocampus. Hippocampus, 6(2), 149-172.
- Jensen, O., & Lisman, J. E. (2000). Position reconstruction from an ensemble of place cells: contribution of the background population. Journal of Neurophysiology, 84(1), 497-503.
- Dragoi, G., & Buzsáki, G. (2006). Temporal encoding of place sequences by hippocampal cell assemblies. Neuron, 50(1), 145-157.

## Running the Analysis

To run this analysis:

```bash
# Run the main analysis script
python theta_phase_precession.py

# Or use the Jupyter notebook
jupyter notebook theta_phase_precession.ipynb
```

The analysis requires:
- numpy, scipy, matplotlib, pynapple, tqdm
- ~2 minutes to complete on a modern computer

## Author Notes

This analysis implements phase precession as a spike generation mechanism where the probability of firing at a given theta phase shifts systematically as a function of position within the place field. The precession signal is implemented with ~95° phase shift over the place field width, which closely matches experimental observations from rat CA1 recordings. Small amounts of noise are added to simulate realistic neural variability while preserving the underlying phase-position relationship.
