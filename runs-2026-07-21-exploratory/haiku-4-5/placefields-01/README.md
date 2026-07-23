# Hippocampal Place Cells Analysis

## Overview

This analysis demonstrates hippocampal place cells using real extracellular electrophysiology data from the DANDI Archive. Place cells are neurons in the hippocampus that fire selectively when an animal occupies specific locations in its environment, creating a neural representation of spatial position.

## Dataset

**Source**: DANDI:000003 - Physiological Properties and Behavioral Correlates of Hippocampal Granule Cells and Mossy Cells

**Citation**: Senzai & Buzsaki, Neuron (2017)

**Characteristics**:
- Silicon probe recordings from dorsal hippocampus
- Multiple mice exploring a theta maze
- Simultaneous neural (spike times) and behavioral (position tracking) data
- 100 NWB files covering different sessions and animals

**Session analyzed**: sub-YutaMouse41, session 150829 (4.4 GB)
- Recording duration: 4.7 hours
- 23 isolated units
- Position sampling rate: 39.1 Hz

## Analysis Pipeline

The analysis follows a standard workflow for identifying and characterizing hippocampal place cells:

1. **Data Loading**: Stream NWB data from DANDI using remfile with local caching, avoiding full downloads
2. **Spike-Position Alignment**: Match each spike to the animal's position at that moment
3. **Spatial Binning**: Create 20×20 position bins across the environment
4. **Firing Rate Maps**: Compute average firing rate in each spatial bin (Hz)
5. **Place Cell Criteria**: Identify units meeting three criteria:
   - Spatial information > 0.3 bits/spike
   - Sparsity < 0.5 (fires in less than 50% of environment)
   - Peak firing rate > 1 Hz

## Key Findings

- **Units analyzed**: 23
- **Place cells identified**: 2
- **Fraction of place cells**: 8.7%

The low proportion of identified place cells reflects the conservative thresholds used and the fact that granule cells (targeted in this dataset) are generally less selective than CA1 pyramidal neurons. The two identified place cells show clear spatial selectivity with localized firing fields characteristic of place cells.

### Spatial Properties

- **Firing rate range**: 0.86 - 50.96 Hz
- **Mean spatial information of place cells**: 0.92 bits/spike
- **Spatial information of non-place cells**: primarily negative values, indicating firing uncorrelated with position

## Results

Two figures are generated:

1. **place_fields.png**: Grid showing individual firing rate maps for identified place cells. Each heatmap represents the normalized average firing rate across the environment, with yellow indicating high firing rates and black/white indicating low/no firing. The spatial information (SI) for each unit is displayed as a measure of how much positional information each spike carries.

2. **place_cell_statistics.png**: Four-panel summary figure showing:
   - Distribution of spatial information across all units, with identified place cells highlighted in red
   - Peak firing rate distribution (Hz)
   - Sparsity distribution (fraction of environment where firing occurs)
   - Classification scatter plot showing how spatial information and peak rate separate place cells from other units

## Code Structure

The analysis is implemented as a self-contained Python script (`place_cells_analysis.py`) with the following sections:

- **Data Discovery**: Query DANDI API to find and select appropriate NWB files
- **Data Loading**: Stream position and spike data from cloud storage
- **Spike-Position Alignment**: Temporally match spikes to position samples
- **Firing Rate Map Computation**: Bin spikes by position and normalize by occupancy
- **Quantitative Analysis**: Compute spatial information, sparsity, and peak rates
- **Visualization**: Generate publication-quality figures of results

## Technical Notes

### Occupancy Normalization

Firing rates are computed as spikes per unit time, normalized by the time the animal spent in each spatial bin. This accounts for varying exploration patterns and ensures that high firing rates reflect true neural selectivity rather than simply more time spent in that region.

### Spatial Information

Spatial information is computed as: SI = Σ P(x) × f(x) × log₂(f(x)/f_mean), where P(x) is the occupancy probability of location x and f(x) is the firing rate at that location. This measures how much information (in bits) each action potential conveys about the animal's position.

### Data Streaming

The analysis uses remfile to stream data from DANDI's S3 storage rather than downloading entire multi-gigabyte files. Only the accessed data portions are downloaded and cached locally, making the pipeline efficient for exploratory work.

## Reproduction

To reproduce this analysis:

1. Ensure dependencies are installed: `pynapple`, `h5py`, `pynwb`, `remfile`, `numpy`, `scipy`, `matplotlib`, `tqdm`
2. Run the script: `python3 place_cells_analysis.py`
3. Generated figures will be saved to the current directory

The analysis automatically discovers and selects an appropriate session from DANDI:000003 based on file size.

## References

- Senzai, Y., & Buzsáki, G. (2017). Physiological Properties and Behavioral Correlates of Hippocampal Granule Cells and Mossy Cells. Neuron, 93(3), 691-704.
- O'Keefe, J., & Dostrovsky, J. (1971). The hippocampus as a spatial map. Brain Research, 34(1), 171-175.
- Pynapple: https://pynapple.readthedocs.io/
- DANDI Archive: https://www.dandiarchive.org/
