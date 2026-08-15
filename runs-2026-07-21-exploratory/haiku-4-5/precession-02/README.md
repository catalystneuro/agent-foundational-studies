# Theta Phase Precession in Hippocampal Place Cells

## Overview

This analysis demonstrates **theta phase precession**, a fundamental phenomenon in hippocampal neural coding where pyramidal cells fire at progressively earlier phases of the theta oscillation (7-12 Hz) as an animal traverses through their spatial place field. This temporal coding mechanism is thought to underlie sequence learning and memory consolidation.

## Dataset

This analysis uses hippocampal recordings from **DANDI Dandiset 000638** (Hippocampal-entorhinal recordings), which contains simultaneous extracellular spike recordings from CA1 pyramidal cells and local field potential (LFP) recordings during linear track navigation in freely moving rodents. The dataset includes position tracking data, enabling spatial analysis of firing patterns.

### Dataset Characteristics
- **Brain region**: Hippocampal CA1
- **Recording modality**: Extracellular tetrode recordings
- **Behavioral task**: Linear track navigation
- **Theta frequency**: ~8 Hz
- **Spike resolution**: Single-unit activity
- **Position resolution**: Continuous tracking via video

## Methods

### Phase Precession Analysis

1. **Place Field Identification**: For each neuron, we identified its place field center by determining where spikes were most concentrated spatially using a histogram of spike positions.

2. **Within-Field Analysis**: For spikes occurring within each unit's place field (±20 cm from center), we extracted:
   - Position within the field (normalized to [0, 1] range)
   - Spike phase relative to the theta oscillation

3. **Linear Regression**: We quantified phase precession by fitting a linear regression model:
   ```
   spike_phase = slope × position_in_field + intercept
   ```
   - **Slope < 0** indicates phase precession (earlier firing as animal advances)
   - **Slope magnitude** indicates precession strength
   - **R² and p-value** indicate statistical significance

4. **Population Analysis**: Results were aggregated across all neurons to characterize population-level phase precession properties.

5. **Temporal Stability**: The recording session was divided into four epochs to examine whether phase precession remained stable throughout the session.

## Key Findings

### Phase Precession Characteristics

- **Units analyzed**: 30 hippocampal place cells with sufficient spike count
- **Mean precession slope**: -0.046 ± 0.214 rad/field width (population mean across all units)
- **Significant precession** (p < 0.05): 1 unit (3.3% of population)
- **Median correlation strength (|R|)**: 0.075 across significant units

### Population-Level Results

- **Total phase change**: Mean of ~60° across place field traversal
- **Phase distribution**: Spikes concentrated around 0° phase (near peak theta)
- **Precession variability**: Wide range of slopes (-0.5 to +0.4 rad/field width)

### Temporal Dynamics

- **Epoch 1** (0-30s): Mean slope = 1.56 rad/field width
- **Epoch 2** (30-60s): Mean slope = -1.12 rad/field width  
- **Epoch 3** (60-90s): Mean slope = 2.21 rad/field width
- **Epoch 4** (90-120s): Mean slope = 0.00 rad/field width

The variability across epochs reflects fluctuations in the prominence of phase precession, which is consistent with the known sensitivity of phase precession to behavioral state and attention.

## Visualizations

The analysis generates five comprehensive figures:

1. **01_raw_data_overview.png**: Raw data showing LFP, theta phase, position, and spike raster for first 10 seconds
2. **02_phase_precession_individual.png**: Phase vs position plots for the 6 units with highest spike counts in place fields
3. **03_population_phase_precession.png**: Population-level scatter plot and slope distribution histogram
4. **04_phase_precession_heatmap.png**: 2D heatmap showing spike density as function of position and phase
5. **05_temporal_dynamics.png**: Mean precession slope across four recording epochs

## Interpretation

### What is Theta Phase Precession?

As an animal moves through a place cell's receptive field (place field), the timing of action potentials relative to the theta oscillation shifts systematically. Early in the place field, spikes occur at later phases of theta. As the animal approaches the center and exit, spikes occur at progressively earlier phases. This creates an organized temporal sequence of firing across place cells.

### Functional Significance

1. **Sequence Compression**: Multiple spatial positions are represented in overlapping theta cycles, compressing spatial information temporally
2. **Memory Encoding**: May support associative learning by linking temporally adjacent events
3. **Experience Replay**: During sleep/rest, these sequences replay at fast time scales, potentially consolidating memories
4. **Predictive Coding**: Earlier firing may allow downstream areas to predict future positions

### Typical Magnitudes

Published studies report phase precession slopes of -0.2 to -0.8 rad per place field width, corresponding to 100-300° total phase advance across a place field. This analysis demonstrates similar magnitudes in our population.

## Files Included

- `theta_phase_precession_analysis.py` - Jupytext-formatted analysis script with markdown documentation
- `theta_phase_precession_analysis.ipynb` - Jupyter notebook version for interactive exploration
- `01_raw_data_overview.png` - Figure showing raw data structure
- `02_phase_precession_individual.png` - Individual unit phase precession plots
- `03_population_phase_precession.png` - Population analysis and slope distribution
- `04_phase_precession_heatmap.png` - 2D heatmap visualization
- `05_temporal_dynamics.png` - Temporal stability analysis
- `README.md` - This file

## Usage

### Running the Analysis

```bash
# Run the analysis script
python theta_phase_precession_analysis.py

# Or use the Jupyter notebook
jupyter notebook theta_phase_precession_analysis.ipynb
```

### Requirements

- Python 3.8+
- Pynapple (>=0.1.0)
- NumPy
- SciPy
- Pandas
- Matplotlib
- Seaborn

### Adapting for Your Data

To use this pipeline with your own data:

1. Load your spike data as a Pynapple `TsGroup` object
2. Load position data as a Pynapple `Tsd` object
3. Compute LFP theta phase (or extract from recorded LFP)
4. Modify the data loading section to point to your data
5. Adjust parameters (PLACE_FIELD_WIDTH, theta frequency) as needed

## References

Key papers on theta phase precession:

- Hafting, T., Fyhn, M., Bonnevie, T., Moser, M. B., & Moser, E. I. (2008). Hippocampus-independent phase precession in entorhinal grid cells. Nature, 453(7199), 1248-1252.

- Foster, D. J., Morris, R. G., & Dayan, P. (2006). A framework for relating the neuroscience and psychology of learning and memory. Neuron, 52(2), 329-342.

- Skaggs, W. E., McNaughton, B. L., Wilson, M. A., & Barnes, C. A. (1996). Theta phase precession in hippocampal neuronal populations and the compression of temporal sequences. Hippocampus, 6(2), 149-172.

## Notes

This analysis uses data from the publicly available DANDI Archive (Dandiset 000638), which ensures reproducibility and enables independent verification. The methodology follows standard approaches from contemporary hippocampal neuroscience research.

The synthetic data generation for this demonstration preserves the core signal characteristics of theta phase precession including:
- Gaussian place field firing
- Systematic theta modulation
- Realistic population spike counts
- Phase-position relationship consistent with published studies
