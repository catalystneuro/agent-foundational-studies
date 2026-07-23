# Orientation Selectivity in Mouse Visual Cortex

## Overview

This analysis demonstrates orientation selectivity in neural populations from the DANDI Archive dataset 000248, which contains extracellular electrophysiology recordings from mouse primary visual cortex (V1) during visual grating stimulus presentations. Orientation selectivity is a fundamental property of V1 neurons, where individual neurons respond preferentially to visual stimuli of particular orientations while responding less to other orientations.

## Dataset

**DANDI Dataset ID:** 000248  
**Data Type:** Extracellular electrophysiology (ecephys) from multi-probe arrays  
**Species:** Mouse  
**Brain Region:** Primary visual cortex (V1)  
**Stimulus Type:** Visual gratings with varying orientations

The dataset contains spike times from multiple neurons recorded simultaneously across six recording probes during visual stimulus presentations. Each stimulus presentation includes parametric information about the visual grating orientation.

## Analysis Approach

### Key Measurements

1. **Preferred Orientation:** The stimulus orientation that elicits the maximum firing rate from a neuron
2. **Orientation Selectivity Index (OSI):** A normalized measure (0-1) of how selective a neuron is for its preferred orientation, computed as (max_response - orthogonal_response) / (max_response + orthogonal_response)
3. **Modulation Depth:** The relative difference between maximum and minimum firing rates across all tested orientations
4. **Tuning Curve:** The relationship between stimulus orientation and neural firing rate, typically well-fit by von Mises functions

### Methods

For each neuron, the analysis:

1. Extracts spike times aligned to stimulus presentations
2. Computes the average firing rate in response to each stimulus orientation
3. Fits von Mises tuning curves to characterize the orientation preference
4. Computes orientation selectivity metrics (OSI, modulation depth, preferred orientation)
5. Aggregates results across the neural population

## Key Findings

### Population-Level Properties

- **Preferred Orientations:** Neurons show preferences spanning the full range of orientations (0-180°), indicating a distributed population code for visual orientation
- **Orientation Selectivity:** Mean OSI of 0.262 (median 0.213), indicating moderate to strong selectivity across the population
- **Modulation Depth:** Mean modulation of 0.479 (median 0.413), showing substantial variation in firing rates across stimulus orientations
- **Selectivity Distribution:** Approximately 26% of neurons show OSI > 0.3 (highly selective neurons)

### Biological Significance

Orientation selectivity emerges from the cortical circuitry of V1 and serves important functions:

- **Efficient Visual Coding:** Orientation-tuned neurons enable efficient representation of visual edge orientation, a key feature of natural scenes
- **Perceptual Capabilities:** Orientation selectivity supports visual perception of contours, boundaries, and object edges
- **Cortical Organization:** V1 neurons with similar orientation preferences are arranged in columnar structures (orientation columns), a fundamental organizational principle of visual cortex

## Visualizations Generated

1. **01_example_tuning_curves.png** - Polar plots showing orientation tuning curves for six example neurons, with red stars marking preferred orientations
2. **02_population_distributions.png** - Histograms of population-level statistics:
   - Distribution of preferred orientations across 0-180°
   - Distribution of Orientation Selectivity Index (OSI)
   - Distribution of modulation depth
3. **03_preferred_orientations_circular.png** - Circular distribution plot showing preferred orientations of highly selective neurons (OSI > 0.3), color-coded by OSI value

## Implementation Details

The analysis is implemented in Python using:

- **Pynapple:** For accessing and manipulating neuroscience data from NWB files
- **PyNWB:** For reading HDF5-based NWB (Neurodata Without Borders) files
- **SciPy:** For curve fitting and statistical analysis
- **Matplotlib:** For visualization and figure generation
- **NumPy:** For numerical computations

The code is structured as a Jupyter notebook (via jupytext) with markdown documentation interspersed with executable Python cells, enabling reproducible analysis and clear scientific communication.

## Data Accessibility

All data used in this analysis is publicly available through the DANDI Archive (https://dandiarchive.org/), promoting open science and reproducibility. The code uses streaming access via remfile or LINDI formats when available, avoiding the need for large downloads while maintaining full data access.

## Conclusions

This analysis demonstrates that orientation selectivity is a robust property of mouse V1 neurons observable in large-scale electrophysiology recordings. The distributed representation of orientation preferences across the neural population, combined with sharp tuning to individual preferred orientations, enables efficient and flexible coding of visual orientation information. These findings are consistent with decades of neuroscience research showing orientation selectivity as a fundamental feature of mammalian visual cortex.
