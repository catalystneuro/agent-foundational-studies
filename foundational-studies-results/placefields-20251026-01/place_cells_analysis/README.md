# Hippocampal Place Cell Analysis

## Objective
Demonstrate hippocampal place cells using real neural recordings from the DANDI Archive. Place cells are neurons in the hippocampus that fire when an animal is in a specific location in space, providing a neural basis for spatial memory and navigation.

## Dataset
- **DANDI ID**: [DANDI:000447](https://neurosift.app/dandiset/000447)
- **Name**: Novel-familiar-novel WTrack (CA1-PFC)
- **Description**: Hippocampal-prefrontal recordings in rats during W-track navigation
- **Data Type**: Extracellular electrophysiology with position tracking
- **Brain Region**: Dorsal CA1 hippocampus and prelimbic prefrontal cortex
- **Subject**: Long Evans rats navigating a W-track maze
- **Recording**: 57 units with position data (x, y coordinates)

## Analysis Pipeline
1. **Data Loading**: Stream NWB file from DANDI using remfile with local caching
2. **Position Processing**: Extract rat position and compute occupancy maps
3. **Firing Rate Maps**: Compute 2D spatial firing rate maps for each neuron
4. **Place Cell Identification**: Identify neurons with spatially selective firing
5. **Visualization**: Create place field maps, tuning curves, and population statistics

## Key Findings
Place cells will be identified based on:
- Spatial information content
- Peak firing rate in preferred location
- Place field size and stability
- Spatial selectivity metrics

## Files
- `01_load_and_explore.py` - Initial data exploration
- `02_compute_place_fields.py` - Compute spatial firing rate maps
- `03_identify_place_cells.py` - Statistical identification of place cells
- `hippocampal_place_cells.py` - Final consolidated analysis script
- `hippocampal_place_cells.ipynb` - Jupyter notebook version

## References
- Shin, J.D. & Jadhav, S.P. (2023). Novel-familiar-novel WTrack (CA1-PFC). DANDI Archive.
- O'Keefe, J. & Dostrovsky, J. (1971). The hippocampus as a spatial map.
