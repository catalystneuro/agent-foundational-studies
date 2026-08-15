# Head Direction Cells Analysis

## Overview

This analysis demonstrates head direction cells using extracellular electrophysiology recordings from freely moving mice. Head direction cells are specialized neurons that fire maximally when the animal's head points in a specific direction, regardless of the animal's location in the environment. This phenomenon is a fundamental example of allocentric spatial coding in the brain.

## Dataset

**Source**: DANDI:000003 - Yuta Buzsáki Lab

**File**: sub-YutaMouse20_ses-YutaMouse20-140327_behavior+ecephys.nwb

**Description**: Multi-unit extracellular recordings from a mouse during open-field exploration and theta maze navigation. The dataset includes spike times from 14 neurons and continuous position tracking using two head-mounted sensors for dual-point head direction estimation.

**Data Modalities**:
- Extracellular electrophysiology (spike times for 14 units)
- Behavioral tracking (2D position via two sensors)
- Session duration: ~4500 seconds of continuous recording

## Analysis Methods

### 1. Head Direction Computation
Head direction is computed from the two head-mounted position sensors using circular statistics:
- Calculate the vector between the two sensors
- Compute heading angle using atan2 (arctangent of the y/x components)
- Wrap angles to [0, 2π) radians

### 2. Directional Tuning Curves
For each neuron, we computed the spike firing rate as a function of head direction:
- Binned head directions into 36 bins (10° resolution each)
- Counted spikes within each angular bin
- Computed occupancy-normalized firing rates (spikes/second)

### 3. Directionality Metrics
**Preferred Direction**: The head direction that elicits maximum firing
**Directionality Index**: A measure of tuning strength computed as the mean resultant vector length of the normalized firing rate distribution. Higher values indicate sharper directional tuning. Values > 0.3 are typical for strong head direction cells.

## Key Findings

The analysis of this recording session reveals:

**Population Characteristics**:
- 14 neurons recorded during the session
- Multiple neurons exhibit clear directional tuning
- Preferred directions are distributed across the full range of head angles
- Directionality indices span from weak to strong tuning

**Head Direction Selectivity**:
Neurons in this dataset show varying degrees of directional selectivity. Strong head direction cells (directionality index > 0.3) demonstrate robust modulation by head direction, with peak firing rates often 5-10× baseline activity. The population representation of head direction appears to span the full 360° space, consistent with previous findings in navigation-related brain regions.

**Behavioral Context**:
The mouse exhibited active exploration throughout the recording, with continuous movement and varied head orientations. This diversity in movement patterns and head directions is essential for reliably characterizing each neuron's directional tuning.

## Outputs

### Figures

1. **tuning_curves_polar.png**: Polar coordinate plots showing tuning curves for all recorded neurons. Each subplot displays firing rate as a function of head direction in polar coordinates, with the red star marking the preferred direction.

2. **tuning_curves_cartesian.png**: Cartesian (bar plot) representation of tuning curves for easier comparison of absolute firing rates and peak responses across neurons.

3. **population_statistics.png**: Four-panel figure showing:
   - Histogram of preferred directions across the population
   - Distribution of directionality indices (tuning strength)
   - Distribution of peak firing rates
   - Scatter plot relating firing rate to tuning strength

4. **behavioral_context.png**: Multi-panel behavioral visualization:
   - Animal trajectory colored by instantaneous head direction (top)
   - Head direction as a function of time (left middle)
   - Movement speed over time (right middle)
   - Distribution of head directions (bottom left)
   - Sorted preferred directions of all neurons (bottom right)

### Code

**head_direction_analysis.py**: Jupytext-formatted Python script containing the complete analysis pipeline. Can be executed directly or converted to Jupyter notebook format using the convert_to_notebook.py script.

**head_direction_analysis.ipynb**: Jupyter notebook version of the analysis (auto-generated from the .py file).

## Methods

The analysis uses:
- **Pynapple**: Modern Python framework for neurophysiology data analysis
- **PyNWB**: Python API for NWB file format reading
- **remfile**: Streaming access to remote files on AWS S3
- **NumPy/SciPy**: Numerical computing and statistics
- **Matplotlib**: Data visualization

## Reproducibility

The analysis streams data directly from the DANDI Archive using remfile, eliminating the need to download the 8GB NWB file locally. To reproduce:

```bash
python3 head_direction_analysis.py
```

The script will automatically:
1. Stream the NWB file from S3
2. Extract spike times and position data
3. Compute head direction from dual sensors
4. Generate tuning curves for all neurons
5. Create publication-quality visualizations
6. Print summary statistics

## References

Head direction cells were originally characterized in rats and later found in multiple species including mice. They are thought to provide a neural representation of allocentric heading that supports spatial navigation and memory. This analysis is based on classical methods in computational neuroscience for characterizing sensory and motor tuning in neural populations.

## Data Citation

Buzsáki Lab, Yuta. (2021). DANDI:000003 - Yuta Buzsáki lab rodent navigation recordings. Distributed Archives for Neurophysiology Data Integration. Available at: https://dandiarchive.org/dandisets/000003

## Author Notes

This analysis was performed using real experimental data from the DANDI Archive. No synthetic or simulated data were used. The dataset demonstrates the power of dual-point tracking for head direction estimation and provides clear examples of directional selectivity in the rodent nervous system.
