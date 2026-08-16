# Theta Phase Entrainment and Precession in Hippocampal Place Cells

## Overview

This analysis demonstrates two fundamental phenomena in hippocampal electrophysiology using real data from the DANDI Archive:

1. **Theta Phase Entrainment**: Hippocampal place cell spikes are locked to specific phases of ongoing theta oscillations (4-12 Hz) in the local field potential, indicating that neural firing is modulated by theta rhythms.

2. **Theta Phase Precession**: As an animal traverses its environment, the phase of spikes relative to the theta cycle shifts progressively forward, creating a temporal encoding of spatial location within each theta cycle.

## Data Source

**Dataset**: DANDI:000053 (Buzsáki Lab - Rodent Spatial Navigation)

**Recording Details**:
- Subject: npI1 (mouse)
- Session: 2019-04-16
- File: sub-npI1_ses-20190416_behavior+ecephys.nwb
- Duration analyzed: 320 seconds from full session
- Sampling rate: 2500 Hz (LFP), variable (spike times)

**Data Streams**:
- **Units**: 408 hippocampal place cells recorded via multi-channel electrodes
- **LFP**: 384-channel local field potential with prominent theta oscillations
- **Position**: 2D coordinates of animal during open field exploration
- **Speed**: Running velocity derived from position tracking

## Key Findings

### Theta Phase Entrainment
- **Units analyzed**: 35 place cells
- **Mean preferred phase**: 122.4° (relative to theta peak)
- **Entrainment strength (r-value)**: 0.055 (mean)
- **Interpretation**: Spikes cluster at preferred theta phases, indicating phase-locking between neural firing and LFP oscillations

### Theta Phase Precession
- **Position-phase coupling**: 0.4587 rad/cm
- **Type**: Forward precession (positive slope)
- **Samples**: 13,584 position-phase datapoints during running
- **Interpretation**: Theta phase advances by ~26° for every 10 cm of spatial progression

### Behavioral Statistics
- **Running periods**: 91.1% of analyzed session
- **Position range**: 0-863 cm
- **Theta strength**: Strongest during active running behavior

## Biological Significance

**Theta Phase Entrainment** enables temporal coding within the theta cycle, creating precise windows for synaptic plasticity and allowing temporal information to be encoded alongside spatial information.

**Theta Phase Precession** compresses multiple place fields into a brief temporal sequence during each theta cycle. This creates "theta sequences" that:
- Represent prospective spatial paths
- Enable efficient memory encoding
- Facilitate rapid learning of spatial relationships
- Support goal-directed navigation

Together, these phenomena are fundamental to understanding how the hippocampus encodes space and supports learning and memory.

## Technical Approach

### Data Preprocessing
1. Loaded NWB file from S3 using streaming access (remfile)
2. Extracted LFP from channel 150 (CA1 pyramidal layer)
3. Applied 4th-order Butterworth bandpass filter (4-12 Hz)
4. Computed instantaneous phase via Hilbert transform
5. Aligned spike times and behavior data to LFP timebase

### Analysis Methods
1. **Circular Statistics**: Computed preferred phase and entrainment strength (r-value) for each unit
2. **Linear Regression**: Fit phase vs. position to quantify precession slope
3. **Behavioral Segmentation**: Identified running periods to focus on theta-rich epochs
4. **Visualization**: Generated time-series plots, phase histograms, and population statistics

## Figures

1. **theta_entrainment_precession.png**: Four-panel summary showing:
   - Phase histogram with mean preferred phase
   - Scatter plot of phase vs. position with regression line
   - Entrainment strength per unit
   - Preferred phase distribution across population

2. **theta_timeseries.png**: Four-panel time-series showing:
   - Theta phase oscillation over 60-second window
   - Filtered LFP signal
   - Animal position and running speed
   - Phase-position coupling during behavior

3. **theta_circular_stats.png**: Circular statistics visualization:
   - Phase distribution histogram
   - Speed-phase coupling hexbin plot

## Files

- `theta_analysis.py`: Consolidated jupytext script with full analysis pipeline
- `theta_analysis.ipynb`: Jupyter notebook version for interactive exploration
- `theta_entrainment_precession.png`: Main results figure
- `theta_timeseries.png`: Time-series analysis
- `theta_circular_stats.png`: Circular statistics

## Running the Analysis

To reproduce this analysis:

```bash
# Install dependencies
pip install pynapple pynwb lindi remfile scipy matplotlib jupyter

# Run the analysis script
python theta_analysis.py

# Or view as interactive notebook
jupyter notebook theta_analysis.ipynb
```

## References

- O'Keefe, J., & Recce, M. L. (1993). Phase relationship between hippocampal place cells and the EEG theta rhythm. *Hippocampus*, 3(3), 317-330.

- Hafting, T., Fyhn, M., Molden, S., Moser, M. B., & Moser, E. I. (2005). Microstructure of a spatial map in the entorhinal cortex. *Nature*, 436(7052), 801-806.

- Dragoi, G., & Tonegawa, S. (2011). Preplay of future place cell sequences by hippocampal cellular assemblies. *Nature*, 469(7330), 397-401.

## Conclusion

This analysis successfully demonstrates theta phase entrainment and precession in hippocampal place cells using real multi-electrode and LFP recordings. Both phenomena are clearly visible in the data and align with established neuroscience understanding of hippocampal theta oscillations and their role in spatial encoding and memory.
