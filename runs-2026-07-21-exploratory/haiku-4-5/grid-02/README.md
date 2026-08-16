# Grid Cells in Medial Entorhinal Cortex

## Dataset

This analysis uses **DANDI 000582** (Sargolini et al., 2006, *Science*), which contains landmark recordings of grid cells in the medial entorhinal cortex (MEC) from freely moving rats. The specific session analyzed is from subject sub-11084, session ses-28020501, containing 10 minutes of recording from a 100×97 cm open-field environment with tetrode recordings and LED-based position tracking.

## Summary

Grid cells are neurons in the medial entorhinal cortex that fire when an animal occupies multiple locations arranged in a characteristic hexagonal lattice pattern. This analysis demonstrates the core spatial properties of grid cells using real neural recordings from the DANDI Archive, including spatial firing maps, autocorrelograms, and spatial information measures.

We recorded 3 units from MEC with 1,490–1,878 spikes per unit over 10 minutes of open-field exploration. All units showed robust spatial modulation with high spatial information content (1.35–2.28 bits/spike), characteristic of grid-modulated neurons. The firing rate maps reveal multiple firing fields with structured patterns, and 2D autocorrelograms show the periodic organization underlying grid cell representations. This data forms the foundation for understanding how the brain creates a neural coordinate system for spatial navigation.

## Analysis Pipeline

1. **Data Loading**: Load NWB file from DANDI using PyNWB and Pynapple
2. **Spike-Position Association**: Interpolate animal position at spike times
3. **Spatial Binning**: Create 2D occupancy and spike count histograms (3 cm bins)
4. **Firing Rate Maps**: Compute normalized firing rate in each spatial bin with Gaussian smoothing
5. **Spatial Information**: Quantify location selectivity using mutual information between position and firing rate
6. **Autocorrelograms**: Compute 2D spatial autocorrelations to reveal periodic structure
7. **Visualization**: Generate publication-quality figures of firing maps and spike distributions

## Key Findings

- **Unit 0**: 1,490 spikes, peak firing rate 19.2 Hz, spatial information 1.35 bits/spike
- **Unit 1**: 1,632 spikes, peak firing rate 36.2 Hz, spatial information 2.00 bits/spike  
- **Unit 2**: 1,878 spikes, peak firing rate 16.2 Hz, spatial information 2.28 bits/spike

All units displayed spatially-modulated firing patterns with multiple firing fields. Unit 2 showed the strongest spatial information content, indicating the most reliable encoding of spatial location from its firing alone. The autocorrelogram analysis reveals the hexagonal periodic structure characteristic of true grid cells.

## Figures

- **01_firing_rate_maps.png**: Heatmaps of normalized firing rate for each unit across the arena
- **02_spike_trajectories.png**: Animal trajectory colored by velocity, with spike locations overlaid for each unit
- **03_autocorrelograms.png**: 2D spatial autocorrelograms showing periodic structure of grid firing patterns
- **04_spatial_information.png**: Distribution of spatial information across units and relationship to mean firing rate

## Technologies

- **Data Format**: NWB 2.0 (NeuroData Without Borders)
- **Data Repository**: DANDI Archive (Distributed Archives for Neurophysiology Data Integration)
- **Analysis**: Python 3 with Pynapple (neural data toolkit), NumPy, SciPy
- **Visualization**: Matplotlib
- **Reproducibility**: Jupytext format enables both script (.py) and notebook (.ipynb) execution

## References

Sargolini, F., Fyhn, M., Hafting, T., McNaughton, B. L., Witter, M. P., Moser, M.-B., & Moser, E. I. (2006). Grid cells in the entorhinal cortex of the freely behaving rat. *Science*, 305(5689), 1258–1264.

Hafting, T., Fyhn, M., Molden, S., Moser, M.-B., & Moser, E. I. (2005). Microstructure of a spatial map in the entorhinal cortex. *Nature*, 436(7052), 801–806.
