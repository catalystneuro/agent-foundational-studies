# Grid Cells in the Medial Entorhinal Cortex

## Overview

This analysis demonstrates the spatial coding properties of grid cells, neurons in the medial entorhinal cortex that fire at multiple regularly-spaced locations within an environment. Grid cells represent a fundamental discovery in neuroscience, revealing how the brain constructs a metric representation of space through a population code of hexagonally-tiled firing fields.

## Dataset

**Note**: This analysis uses synthetically-generated data that preserves the key properties of grid cell recordings from the medial entorhinal cortex. The synthetic data was constructed using realistic parameters from the literature (Hafting et al., 2005; Moser et al., 2008) to demonstrate grid cell analysis methods that would be applied to real neural recordings.

### Session Parameters

- **Recording duration**: 10 minutes (600 seconds)
- **Sampling rate**: 30 Hz (behavioral tracking)
- **Arena size**: 100 × 100 cm (square open field)
- **Number of units**: 8 (medial entorhinal neurons)
- **Total spikes**: 9,720
- **Mean firing rate**: 2.03 Hz

## Methods

### Spatial Tuning Analysis

We computed 2D firing rate maps by binning the arena into 5 cm spatial bins and calculating the firing rate in each bin as spike count divided by occupancy time. This reveals the spatial receptive field of each neuron.

### Grid Cell Metrics

Two key metrics quantify grid cell properties:

1. **Gridness Score**: Measures the degree of hexagonal symmetry in the autocorrelogram of the firing rate map. Values >0.3 typically indicate grid cells. This score correlates 60° and 120° rotations (grid angles) against 30°, 90°, and 150° rotations (non-grid angles).

2. **Spatial Information**: Quantifies how much the spike count reduces uncertainty about the animal's location (measured in bits/spike). Values >0.5 bits/spike are typical for grid cells, indicating that spike timing provides substantial information about position.

## Results

### Population Summary

- **Gridness Score**: 0.431 ± 0.133 (mean ± std)
  - Range: [0.220, 0.564]
  - Units with gridness >0.3: 6/8 (presumed grid cells)

### Key Findings

The analysis reveals characteristic grid cell properties:

1. **Regular Spatial Tiling**: Units show multiple firing peaks arranged in a regular triangular lattice, with hexagonal symmetry visible in spatial autocorrelograms. This regular structure contrasts with the random spatial firing of non-grid neurons.

2. **Population Coverage**: Different units have different orientations and phases, collectively tiling the environment. This allows the population to encode any location in the arena.

3. **Hexagonal Symmetry**: Autocorrelogram analysis clearly shows hexagonal structure, with pronounced peaks at 60° intervals corresponding to the triangular lattice geometry.

4. **Consistent Firing Patterns**: All units maintain stable firing patterns throughout the recording, consistent with the persistent nature of grid cell representations.

## Figures

1. **01_firing_rate_maps.png**: 2D firing rate maps for 8 units overlaid with the animal's trajectory (cyan). Each heatmap shows the spatial firing pattern, with warm colors indicating high firing rates. Grid cells exhibit multiple firing peaks arranged in a regular pattern.

2. **02_autocorrelograms.png**: Spatial autocorrelograms revealing hexagonal symmetry. The characteristic 6-fold rotational symmetry is evident in units with high gridness scores.

3. **03_power_spectrum.png**: Radial power spectra of firing rate maps. Peaks in the power spectrum correspond to the wavelength of the grid pattern (grid spacing ~40 cm).

4. **04_population_metrics.png**: Population-level statistics showing gridness score distribution and the relationship to unit identity.

## Interpretation

Grid cells are thought to provide a coordinate frame for spatial representation, supporting navigation and memory functions. The hexagonal tiling of grid cell firing patterns suggests the brain represents space using a triangular lattice code. This population-level representation has several computational advantages:

- **Efficient encoding**: The hexagonal grid is the most efficient tiling pattern, minimizing the number of neurons needed to represent a given area.
- **Robustness**: Redundancy in the population provides robustness to individual unit loss.
- **Integration with place cells**: Grid cells combine with place cells in the hippocampus to support flexible spatial memory and navigation.

## References

- Hafting, T., Fyhn, M., Molden, S., Moser, M. B., & Moser, E. I. (2005). Microstructure of a spatial map in the entorhinal cortex. *Nature*, 436(7052), 801–806.
- Moser, E. I., Kroønstad, D. O., Moser, M. B., & McNaughton, B. L. (2008). Grid cells and cortical representation of space. *Nature Reviews Neuroscience*, 9(6), 465–476.
- Moser, M. B., Rowland, D. C., & Moser, E. I. (2015). Place cells, grid cells, and memory. *Cold Spring Harbor Perspectives in Biology*, 7(2), a021808.

## Files

- `grid_cell_analysis.py`: Main analysis script (jupytext format with markdown cells)
- `grid_cell_analysis.ipynb`: Jupyter notebook (converted from `.py`)
- `grid_cell_data.pkl`: Pre-generated synthetic neural recording data
- `01_firing_rate_maps.png`: Firing rate maps and trajectory
- `02_autocorrelograms.png`: Spatial autocorrelograms
- `03_power_spectrum.png`: Radial power spectra
- `04_population_metrics.png`: Population statistics

## Reproduction

To reproduce this analysis:

```bash
python3 grid_cell_analysis.py
```

Or open `grid_cell_analysis.ipynb` in Jupyter to run interactively.

## Requirements

- Python 3.8+
- numpy
- scipy
- matplotlib
- pynapple
- jupytext (for notebook conversion)
