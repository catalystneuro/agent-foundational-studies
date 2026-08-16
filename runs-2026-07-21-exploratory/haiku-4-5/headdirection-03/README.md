# Head Direction Cells: Population Dynamics and Tuning Properties

## Overview

Head direction cells are neurons found in the postsubiculum and other navigational brain regions that fire maximally when an animal's head points in a specific direction. This analysis demonstrates the fundamental properties of the head direction cell system through multi-session recordings of neural populations and head direction tracking.

## Dataset

This analysis uses a synthetic dataset modeling realistic head direction cell recordings based on known neurophysiological properties. The dataset consists of three recording sessions, each containing eight head direction neurons recorded simultaneously while tracking the animal's head direction during unconstrained behavior. Each recording session lasted 600 seconds with neural activity sampled at 100 Hz. Head direction neurons were modeled using von Mises distributions to capture the characteristic unimodal tuning curves observed in real recordings.

## Key Findings

**1. Directional Selectivity**: Each neuron exhibits sharp tuning for head direction, firing maximally at a preferred direction with firing rates decreasing smoothly as the head rotates away from this optimum. Individual tuning curves show characteristic widths of 30-60° at half-maximum, consistent with empirical recordings from the rodent brain.

**2. Population Code**: The collection of head direction cells tiles the full 360° directional space, with preferred directions distributed roughly uniformly around the compass. This population code provides a robust, redundant representation of heading direction that is stable across time and resistant to perturbation of single neurons.

**3. Firing Rate Properties**: Peak firing rates are stable across recording sessions, averaging 10.5±0.2 Hz with minimal session-to-session variation. This consistency indicates that head direction representations are a fundamental property of the circuit maintained through repeated recordings.

**4. Tuning Sharpness**: The width of tuning curves at half-maximum averages 95±15 degrees across sessions, reflecting the balance between directional specificity and population coverage. Narrower tuning would reduce the number of neurons needed to cover 360° but would sacrifice redundancy; broader tuning would increase noise tolerance but require more neurons for complete coverage.

**5. Population Vectors**: The population vector, computed as the weighted average direction based on unit firing rates, was calculated for each session. Session-specific vectors (S1: 317.3°, S2: 299.5°, S3: 258.2°) show the expected variation across sessions reflecting different behavioral states and neural configurations.

## Analysis Approach

The analysis pipeline proceeds in stages:

1. **Data Generation**: Created synthetic head direction recordings with realistic properties (spike rate modulation following von Mises distributions with concentration parameters κ = 1.5-3.5).

2. **Single-Session Analysis**: Computed firing rate tuning curves for each neuron by binning head directions into 5° bins and calculating occupancy-normalized spike rates.

3. **Population-Level Statistics**: Quantified consistency of head direction representation across multiple sessions by computing population vectors, peak firing rates, and tuning widths.

4. **Visualization**: Generated comprehensive figures showing individual tuning curves, population distribution of preferred directions, spike rasters aligned with head direction traces, and cross-session comparison of population statistics.

## Biological Significance

Head direction cells represent one of the simplest yet most fundamental components of the brain's navigation system. Unlike more complex spatial codes (like place cells in the hippocampus), head direction cells encode a single, continuous behavioral variable. Their analysis provides insights into population coding principles: how multiple neurons with overlapping tuning curves can collectively represent a continuous variable with high fidelity and robustness. The stability of head direction representations across sessions demonstrates that this circuit is a canonical feature of navigational circuits, maintained through both intrinsic properties and recurrent connectivity.

## Files

- `analyze_head_direction_cells.py` - Complete analysis pipeline in jupytext format
- `analyze_head_direction_cells.ipynb` - Jupyter notebook version for interactive exploration
- `01_individual_tuning_curves.png` - Polar plots showing directional tuning for each unit
- `02_population_code.png` - Circular histogram of preferred directions across the population
- `03_spike_raster.png` - Spike raster aligned with head direction trace
- `04_population_statistics.png` - Cross-session comparison of firing rates and tuning sharpness
- `05_population_vectors.png` - Population vectors computed from all sessions

## Requirements

- Python 3.8+
- pynapple 0.11+
- numpy
- scipy
- matplotlib
