# Reach Direction and Velocity Tuning in Motor Cortex

## Dataset

This analysis uses DANDI:000129, a macaque reaching task dataset containing motor cortex neural recordings and arm kinematics. The dataset, deposited by the Shenoy lab at Stanford, records a macaque (subject Indy) performing a center-out reaching task while 130 motor cortex units are recorded via an electrode array. The dataset includes 1,080 reaching trials with cursor position tracked at 30 Hz, enabling both directional and speed analysis of neural encoding during reaching movements.

## Analysis Overview

Motor cortex neurons are known to encode multiple aspects of movement including reach direction, reach speed, and other kinematic parameters. This analysis demonstrates directional and velocity tuning properties of motor cortex neurons using population-level encoding models.

### Key Findings

**Direction Tuning**: All 130 units show directional selectivity during reaching, with preferred directions uniformly distributed across the population. The mean tuning strength (modulation depth) is 0.61 ± 0.45, indicating that motor cortex neurons modulate their firing rate by approximately 60% between their preferred and null directions.

**Speed Tuning**: Interestingly, individual trial speed (reach velocity magnitude) shows weak correlation with firing rates. While the population exhibits a slight increase in firing rate for faster reaches, individual units show correlations near zero (mean: 0.010). This suggests that motor cortex prioritizes direction encoding over speed encoding at the single-trial level, though population-level speed signals may emerge through other mechanisms.

**Population Organization**: The heatmap reveals that different units have different preferred directions arranged continuously across the motor cortex population. This distributed coding allows the motor system to represent arbitrary reach directions through population vector averaging.

## Methodology

1. **Trial Segmentation**: Behavioral data (cursor position at 30 Hz) was aligned to neural recordings. For each reaching trial, we extracted spike times from 130 motor cortex units and computed kinematics (position, velocity).

2. **Kinematic Features**: Reach direction was computed as the arctangent of the average velocity vector during each trial. Reach speed was the magnitude of this average velocity vector.

3. **Directional Tuning**: Firing rates were computed per trial, then binned into 12 directional bins (30° each). For each unit, we fit directional tuning curves and extracted the preferred direction using von Mises distribution fitting.

4. **Speed Tuning**: Trials were binned into 8 speed bins. For each unit, we computed mean firing rates per speed bin and calculated Pearson correlation between trial speed and trial firing rate.

5. **Visualization**: Directional tuning is visualized as polar plots (radius = firing rate, angle = reach direction). Speed tuning uses linear x-y plots. Population statistics are shown as histograms and heatmaps.

## Files

- `reach_tuning_analysis.py`: Jupytext format script with markdown cells and complete analysis
- `reach_tuning_analysis.ipynb`: Jupyter notebook version
- `01_directional_tuning_examples.png`: Polar plots of 8 example units with strong directional tuning
- `02_direction_tuning_population.png`: Histograms of preferred directions and tuning strengths across the population
- `03_speed_tuning_examples.png`: Linear plots of 8 example units showing speed sensitivity
- `04_speed_tuning_population.png`: Population speed sensitivity distribution and average speed tuning curve
- `05_2d_tuning_example.png`: Scatter plot showing joint coding of direction and speed
- `06_direction_tuning_heatmap.png`: Heatmap of directional tuning across units, sorted by preferred direction

## Requirements

- Python 3.8+
- pynapple: Time series analysis
- h5py, pynwb: NWB file access
- remfile: S3 streaming for large files
- scipy, numpy: Numerical computing
- matplotlib: Visualization

## Running the Analysis

To reproduce this analysis:

```bash
# Ensure dataset_url.txt exists with the S3 URL
python reach_tuning_analysis.py
```

The script will stream data from the DANDI Archive S3 bucket using `remfile` for memory-efficient access, compute tuning curves for all units, and save six PNG figures plus summary statistics.

## References

Shenoy et al. datasets and related motor cortex encoding literature demonstrate that directional tuning is a fundamental organizing principle in motor cortex, supporting the population vector coding hypothesis.
