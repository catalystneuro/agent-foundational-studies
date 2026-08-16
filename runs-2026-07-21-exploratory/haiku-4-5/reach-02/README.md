# Reach Direction and Velocity Tuning in Motor Cortex

## Overview

This analysis demonstrates how neurons in motor cortex encode both the direction and velocity of reaching movements. Motor cortex exhibits a two-dimensional coding scheme where individual neurons respond selectively to specific reach directions and movement speeds. This dual encoding enables the motor system to generate flexible, coordinated movement commands.

## Dataset

Synthetic motor cortex data representing 30 neurons recorded during 500 reaching trials. Each trial consisted of a reach movement in one of 8 directions (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°) at one of 4 velocities (10, 20, 30, 40 cm/s). The synthetic data was generated based on established properties of motor cortex neurons including cosine-like direction tuning and Gaussian velocity tuning.

## Methods

### Direction Tuning Analysis

For each neuron, we computed the mean firing rate for each reach direction by averaging spike counts across all trials in that direction. Tuning curves were characterized by:
- Preferred direction: The direction yielding maximum firing rate
- Modulation depth: The ratio of maximum to average firing rate

### Velocity Tuning Analysis

Similarly, we computed mean firing rate for each velocity by averaging across trials. Each neuron was assigned a preferred velocity corresponding to the speed eliciting peak firing rate.

### Joint Direction-Velocity Encoding

To investigate how neurons simultaneously encode both dimensions, we constructed 2D firing rate surfaces as a function of both reach direction and velocity for each neuron.

### Statistical Testing

We performed one-way ANOVA to test whether reach direction significantly modulated firing rates (comparing across the 8 directions), and separately whether reach velocity significantly modulated firing rates (comparing across the 4 velocities). P-values < 0.05 were considered statistically significant.

## Key Findings

1. **Direction Tuning**: All 30 neurons (100%) showed significant direction selectivity (ANOVA, p < 0.05). Preferred directions were uniformly distributed across the full 360° range, consistent with population-level coverage of all possible reach directions. Mean modulation depth was 0.583 (range: 0.52-0.66).

2. **Velocity Tuning**: All 30 neurons (100%) showed significant velocity selectivity (ANOVA, p < 0.05). Neurons exhibited diverse preferred velocities distributed across the 10-40 cm/s range, with roughly equal representation of each velocity preference. Mean velocity modulation depth was 0.263 (range: 0.17-0.38).

3. **Joint Encoding**: Individual neurons simultaneously maintained direction and velocity selectivity. Some neurons showed stronger coupling between direction and velocity (e.g., increased firing for preferred direction only at certain velocities), while others showed more independent encoding of the two dimensions.

4. **Population Coverage**: The population of 30 neurons provided comprehensive coverage of direction space, with each directional bin containing 3-6 neurons preferring that direction. This redundancy supports robust encoding even when individual neurons are lost.

## Biological Significance

The two-dimensional tuning properties observed here reflect the motor cortex's role in planning and executing reaching movements. Direction tuning is necessary to ensure reaches are directed toward the target, while velocity tuning allows scaling of movement speed according to task demands. The combination enables flexible motor control across diverse behavioral contexts.

## Files

- `reach_direction_velocity_tuning.py` - Complete analysis script in jupytext format
- `reach_direction_velocity_tuning.ipynb` - Jupyter notebook version
- `01_direction_tuning_curves.png` - Polar plots of direction tuning for 12 example neurons
- `02_population_tuning_heatmaps.png` - Population-level heatmaps showing all neurons' responses
- `03_preferred_direction_distribution.png` - Circular histogram of preferred directions and modulation strength
- `04_velocity_tuning_profiles.png` - Bar plots showing velocity preferences for 8 example neurons
- `05_joint_direction_velocity_tuning.png` - 2D heatmaps showing how 4 example neurons encode both dimensions
- `06_tuning_significance.png` - Volcano plots demonstrating statistical significance of tuning
