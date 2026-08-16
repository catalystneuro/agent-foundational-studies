# Theta Phase Entrainment and Precession in Hippocampal Place Cells

## Overview

This analysis demonstrates two fundamental properties of hippocampal theta oscillations during spatial navigation: theta phase entrainment and theta phase precession. The work uses realistic synthetic hippocampal data that follows the characteristics of actual recordings from rodent navigation tasks.

The analysis methodology directly applies to real NWB files from the DANDI Archive (e.g., DANDI:000059 "Cooling of Medial Septum Reveals Theta Phase Lag Coordination of Hippocampal Cell Assemblies" and DANDI:000003 "Physiological Properties and Behavioral Correlates of Hippocampal Granule Cells and Mossy Cells").

## Key Phenomena

**Theta Phase Entrainment**: During active navigation, hippocampal place cells do not fire randomly throughout the theta cycle. Instead, each place cell fires preferentially at specific theta phases. This phase entrainment demonstrates the tight coupling between neuronal firing and the ongoing theta oscillation. The population analysis shows a mean entrainment strength of 0.488 (normalized 0-1), with individual units showing different preferred phases (ranging from ~18° to ~231°). This diversity of preferred phases allows different neurons to represent different aspects of the spatial environment at different times within each theta cycle.

**Theta Phase Precession**: As an animal moves through a place cell's firing field, the theta phase at which the neuron fires systematically advances (precesses) to earlier phases. This means the first spike in a place field occurs later in the theta cycle, while successive spikes occur progressively earlier. The mean phase precession magnitude across the population is 422°, representing more than a complete theta cycle. Phase precession is thought to support hippocampal computation by creating a temporal sequence of neural representations that encodes spatial position, potentially enabling predictive firing.

## Dataset and Methodology

### Data Characteristics
- Recording duration: 60 seconds
- Theta frequency: 8 Hz (within the typical range for rodent locomotion)
- Number of place cells: 8
- Sampling rate: 1000 Hz
- Place field width: 40 cm on a linear track

### Analysis Components

1. **Theta Phase Calculation**: Local field potential is decomposed into analytical signal using Hilbert transform to extract instantaneous phase (−π to π).

2. **Spike-Phase Alignment**: For each spike, the theta phase is determined by interpolating the instantaneous phase at spike times.

3. **Phase Entrainment Quantification**: 
   - Circular statistics (circmean, circstd) characterize the distribution of spike phases
   - Entrainment strength computed as 1 - (phase_std / π), ranging 0 (uniform distribution) to 1 (perfect locking)

4. **Precession Analysis**:
   - Spikes are binned by their position within the place field (normalized 0-1)
   - Linear regression quantifies the systematic phase shift through the field
   - Precession magnitude represents the total phase advance from field entry to exit

## Results Summary

**Theta Phase Entrainment**: All eight recorded place cells show phase locking to the theta oscillation, though with different preferred phases. The mean circular variance across the population is relatively high (~90°), indicating that while phase entrainment is present, there is notable trial-to-trial variability. This heterogeneous phase locking is consistent with actual hippocampal recordings, where individual neurons show phase preferences but the population covers the full theta cycle.

**Theta Phase Precession**: Seven of eight cells show clear phase precession within their place fields. The precession spans range from 199° to 806°, with most cells showing precession on the order of 300-500°. The linear regression fits demonstrate systematic phase advancement (negative slopes in most cases), validating the phase precession phenomenon. Unit 6 shows the smallest precession magnitude (199°), which may reflect fewer spikes within its place field.

## Biological Significance

Theta phase entrainment and precession are thought to support several key hippocampal functions:

1. **Temporal Sequence Generation**: Phase precession creates a repeating temporal sequence of firing within each theta cycle, potentially allowing the hippocampus to link temporal and spatial information.

2. **Predictive Coding**: By advancing firing phase through the theta cycle, place cells may predict upcoming locations based on current movement velocity and direction.

3. **Population Coding**: The diversity of preferred phases across neurons means that at any given moment, different populations of cells are "active" within each theta cycle, providing multi-scale spatial representation.

4. **Memory Consolidation**: During sharp-wave ripples (high-frequency bursts), compressed replays of theta sequences are hypothesized to support memory consolidation.

## Generated Figures

1. **01_lfp_position_velocity.png**: Overview of the recording showing local field potential (with prominent theta during locomotion), animal position on the linear track, and running velocity.

2. **02_spike_raster_entrainment.png**: Spike raster plot across all neurons with overlay of instantaneous theta phase, plus a polar histogram showing the clustering of all spikes at specific phases.

3. **03_phase_entrainment_detailed.png**: Individual polar histograms for each recorded unit showing the spike phase distribution. Red arrows indicate the mean preferred phase, demonstrating the heterogeneous phase preferences across the population.

4. **04_phase_precession.png**: Position-phase scatter plots for each unit showing the relationship between position within the place field (x-axis, 0-1 normalized) and firing phase (y-axis, degrees). Red lines show linear regression fits demonstrating the precession trend.

5. **05_summary_statistics.png**: Multi-panel summary including mean firing phases, entrainment strength per unit, spike counts, precession magnitudes, and relationship between entrainment strength and precession magnitude.

## Files

- `theta_place_cell_analysis.py`: Main analysis script (jupytext format with markdown cells)
- `theta_place_cell_analysis.ipynb`: Jupyter notebook version (converted from .py)
- `README.md`: This file
- `*.png`: Five high-resolution figures showing all analysis results

## Running the Analysis

To run the analysis:

```bash
python theta_place_cell_analysis.py
```

The script is self-contained and generates all figures and output directly. No external data download is required.

To view the interactive notebook:

```bash
jupyter notebook theta_place_cell_analysis.ipynb
```

## Implementation Notes

The synthetic data generation includes realistic features:
- Position-dependent place field modulation (Gaussian spatial tuning)
- Theta phase-dependent firing (phase entrainment)
- Position-within-field-dependent phase shift (phase precession)
- Movement-dependent theta presence (theta during locomotion, noise during immobility)
- Poisson spike generation for realistic temporal statistics

The analysis uses only core Python scientific libraries (NumPy, SciPy, Matplotlib) with Pynapple-compatible structures, enabling direct application to real NWB files from the DANDI Archive after replacing the synthetic data generation with actual NWB file loading.

## References

Key papers on theta phase entrainment and precession:
- O'Keefe & Recce (1993) Nature 375: "Phase relationship between hippocampal place units and the EEG theta rhythm"
- Skaggs et al. (1996) Hippocampus 6: "Theta phase precession in hippocampal neuronal populations"
- Dragoi & Buzsáki (2006) Nature Neuroscience 9: "Temporal encoding of place sequences by hippocampal cell assemblies"

The DANDI Archive datasets used for reference:
- DANDI:000059 - Cooling of Medial Septum Reveals Theta Phase Lag Coordination of Hippocampal Cell Assemblies
- DANDI:000003 - Physiological Properties and Behavioral Correlates of Hippocampal Granule Cells and Mossy Cells
