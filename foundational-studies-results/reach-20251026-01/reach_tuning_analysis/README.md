# Reach Direction and Velocity Tuning Analysis

## Dataset
- **DANDI Archive**: [DANDI:000688](https://neurosift.app/dandiset/000688)
- **Title**: Long-term recordings of motor and premotor cortical spiking activity during reaching in monkeys
- **Description**: Electrophysiology and behavioral data from macaques performing center-out reaching tasks
- **Data**: Neural spiking activity from M1/PMd, cursor position and velocity

## Analysis Goal
Demonstrate how motor cortex neurons exhibit:
1. **Direction tuning**: Neurons fire preferentially for specific reach directions
2. **Velocity tuning**: Neural activity correlates with movement velocity

## Session Information
- Subject: Monkey C
- Session: CO-20131003 (center-out task)
- 71 recorded units
- 193 trials (169 valid reaching trials)
- Recording duration: 663 seconds

## Key Results

### Direction Tuning
- **87.3%** of neurons (62/71) show significant direction selectivity (modulation depth > 0.3)
- Mean modulation depth: 0.567 ± 0.229
- Preferred directions distributed across all 8 reach directions
- Neurons exhibit characteristic cosine-like tuning curves

### Velocity Tuning
- **69.0%** of neurons (49/71) show significant correlation with movement speed (p < 0.01)
- Mean correlation: 0.107 ± 0.094
- Both positive and negative velocity correlations observed
- Neural activity tracks moment-to-moment changes in speed

## Files

### Analysis Scripts
- `01_load_and_explore.py` - Initial data exploration and visualization
- `02_direction_tuning.py` - Direction selectivity analysis
- `03_velocity_tuning.py` - Velocity correlation analysis
- `reach_direction_velocity_tuning.py` - Final consolidated analysis script
- `reach_direction_velocity_tuning.ipynb` - Jupyter notebook version

### Generated Figures
- `fig_01_cursor_trajectories.png` - Reach trajectories and speed profiles
- `fig_02_spike_rasters.png` - Example neural raster plots
- `fig_03_direction_tuning_curves.png` - Direction tuning curves (from script 2)
- `fig_04_population_direction_tuning.png` - Population direction statistics (from script 2)
- `fig_05_velocity_tuning_examples.png` - Velocity tuning examples (from script 3)
- `fig_06_population_velocity_tuning.png` - Population velocity statistics (from script 3)
- `fig_07_single_trial_velocity_tuning.png` - Single trial time series (from script 3)
- `fig_direction_tuning_curves.png` - Direction tuning curves (final)
- `fig_population_direction_tuning.png` - Population direction statistics (final)
- `fig_velocity_tuning_examples.png` - Velocity tuning examples (final)
- `fig_population_velocity_tuning.png` - Population velocity statistics (final)

## Running the Analysis

To run the complete analysis:
```bash
cd reach_tuning_analysis
python reach_direction_velocity_tuning.py
```

Or open the Jupyter notebook:
```bash
jupyter notebook reach_direction_velocity_tuning.ipynb
```

## Dependencies
- h5py
- pynwb
- remfile
- pynapple
- numpy
- matplotlib
- scipy
