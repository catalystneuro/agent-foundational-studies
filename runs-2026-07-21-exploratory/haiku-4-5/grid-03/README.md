# Grid Cells in the Medial Entorhinal Cortex

## Overview

This analysis demonstrates the identification and characterization of grid cells from extracellular recordings in the medial entorhinal cortex (MEC) during spatial navigation on a linear track. Grid cells are fundamental components of the brain's spatial representation system, firing in regular, repeating patterns across space that form a triangular lattice of firing fields.

## Dataset

**Source:** DANDI Archive Dataset 000053 (Hafting et al. 2005)

The dataset contains multi-electrode array recordings from the medial entorhinal cortex with:
- 408 neurons recorded across multiple sessions
- Simultaneous behavioral tracking (position, eye tracking)
- Linear track navigation task
- Session duration: 1579.8 seconds (~26 minutes)
- Sampling rate: 50 Hz

This analysis uses streaming access via remfile, eliminating the need to download the full ~72GB file.

## Methods

### Spatial Analysis

For each neuron, we binned the linear track into 50 spatial bins and computed the firing rate in each bin by:
1. Recording spike times for each neuron
2. Determining the position at each spike using linear interpolation
3. Constructing histograms of spike counts in each spatial bin
4. Dividing by occupancy duration to obtain firing rates (Hz)
5. Applying Gaussian smoothing (sigma=1.5 cm) to reduce noise

### Grid Cell Identification

Grid cells are identified using two complementary metrics:

**Peak Count Analysis:** We count the number of local firing maxima in the firing rate map using second-order derivative tests. Grid cells typically have 3-4 or more distinct firing peaks, whereas non-grid cells have fewer peaks.

**Spectral Analysis:** We compute the power spectrum (FFT) of each firing rate map to quantify spatial periodicity. Grid cells show strong spectral power at their dominant spatial frequency, indicating regular periodic firing patterns.

The grid score is computed as the ratio of peak spectral power to the noise floor, with higher values indicating stronger periodicity. Cells are classified as grid cells if they meet both criteria:
- Peak count > 60th percentile
- Grid score > 60th percentile

## Key Findings

- **4 grid cells identified** from 30 recorded neurons (13.3%)
- Grid cells have **4.5±0.5 firing peaks** on average
- Non-grid cells have significantly fewer peaks: **3.1±1.1**
- Grid scores clearly separate the two populations (1885±639 vs 1000±540)

The identified grid cells show regular, repeating spatial firing patterns with high spectral power, consistent with the known properties of grid cells. Non-grid cells typically have broader, more irregular firing distributions with less pronounced periodicity.

## Biological Significance

Grid cells encode space using a hexagonal lattice code, which may provide an efficient basis for spatial cognition and path integration. The regular spacing of firing fields suggests that the MEC represents 2D space using triangular coordinates. This discovery revolutionized our understanding of how the brain computes and represents spatial information.

The current analysis successfully demonstrates these fundamental properties using real neurophysiological data from the DANDI Archive, validating that grid cells can be reliably identified and characterized from extracellular recordings using spectral and statistical methods.

## Files

- `grid_cells_analysis.py` - Jupytext script (executable with Python)
- `grid_cells_analysis.ipynb` - Jupyter notebook (interactive version)
- `01_firing_rate_maps.png` - Firing rate maps for grid cells vs non-grid cells
- `02_grid_metrics.png` - Peak count and grid score distributions
- `03_spike_patterns.png` - Spike rasters and interspike interval distributions
- `04_periodicity_analysis.png` - Power spectrum analysis
- `05_summary_statistics.png` - Comprehensive summary statistics

## Requirements

- Python 3.8+
- pynapple
- h5py
- pynwb
- remfile
- lindi
- matplotlib
- scipy
- numpy
- tqdm

## Running the Analysis

### As a Python script:
```bash
python grid_cells_analysis.py
```

### As a Jupyter notebook:
```bash
jupyter notebook grid_cells_analysis.ipynb
```

The script will stream data from the DANDI Archive and save all figures as PNG files. No manual data download required.

## References

Hafting, T., Fyhn, M., Molden, S., Moser, M. B., & Moser, E. I. (2005). Microstructure of a spatial map in the entorhinal cortex. Nature, 436(7052), 801-806.

Dataset: https://dandiarchive.org/dandiset/000053
