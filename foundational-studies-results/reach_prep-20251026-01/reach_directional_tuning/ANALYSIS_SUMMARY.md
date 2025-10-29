# Directional Tuning During Reach Planning - Analysis Summary

## Dataset
- **DANDI Archive ID**: [DANDI:000140](https://neurosift.app/dandiset/000140)
- **Dataset**: MC_Maze_Small - macaque primary motor and dorsal premotor cortex spiking activity during delayed reaching
- **Subject**: Jenkins (Macaca mulatta, Male)
- **Recording Areas**: Primary motor cortex (M1) and dorsal premotor cortex (PMd)
- **Institution**: Stanford University, Shenoy Lab

## Task Description
Center-out delayed reaching task where:
1. **Target appears** - indicating reach direction
2. **Delay period** - monkey must wait (planning phase, ~0.6 seconds on average)
3. **Go cue** - signal to begin movement
4. **Movement execution** - monkey reaches to target

## Analysis Approach
- **Analysis Window**: From 100ms after target onset to 100ms before go cue
- **This captures the planning period** when the monkey knows where to reach but hasn't started moving yet
- **Key Question**: Do neurons show directional tuning during planning?

## Key Results

### Dataset Statistics
- **Total trials**: 100
- **Unique reach directions**: 9 (ranging from -145° to 161°)
- **Total neurons**: 142
- **Average delay period**: 0.599 seconds (range: 0.014 to 0.999 s)

### Neural Tuning Results
- **Well-tuned neurons**: 12 / 142 (8.5%)
  - Criteria: p < 0.01 (ANOVA) and tuning strength > 1.5
- **Tuning strength**: 
  - Mean: 13.99
  - Median: 9.16
  - Range: 2.24 to 47.28
- **Preferred directions**: Distributed across all reach directions (mean: 39.7°, std: 106.3°)

### Scientific Findings
1. **Directional tuning is present during planning** - Neurons fire at different rates for different planned reach directions
2. **Tuning occurs before movement** - Neural activity encodes movement direction during the delay period, before movement begins
3. **Population code** - Different neurons prefer different directions, providing a distributed representation
4. **Strong tuning in subset** - While only 8.5% meet strict criteria, many show moderate tuning

## Generated Visualizations

1. **trial_distribution.png** - Shows the spatial layout of targets and distribution of reach directions
2. **tuning_curves_examples.png** - Polar plots of 6 well-tuned neurons showing firing rate vs. direction
3. **raster_plots.png** - Spike rasters for the best-tuned neuron across all directions
4. **population_analysis.png** - Population-level statistics including preferred direction distribution

## Files Created

### Analysis Scripts
- `01_inspect_data.py` - Initial data exploration and validation
- `02_directional_tuning_analysis.py` - Complete tuning analysis
- `reach_directional_tuning_analysis.py` - Final consolidated script (jupytext format)
- `reach_directional_tuning_analysis.ipynb` - Jupyter notebook version

### Documentation
- `README.md` - Project overview and methodology
- `ANALYSIS_SUMMARY.md` - This summary document

### Figures
- All PNG files showing analysis results and visualizations

## Conclusions

This analysis successfully demonstrates that:

1. **Motor planning involves directional coding** - Neurons in M1 and PMd encode the planned movement direction during the delay period, before movement execution

2. **Tuning strength varies** - Some neurons show very strong directional preferences (up to 47x difference between preferred and anti-preferred directions)

3. **Population representation** - The population of neurons collectively represents reach direction through their distributed preferred directions

4. **Planning vs. execution** - The presence of directional tuning during the delay period (planning) demonstrates that motor cortex is not just involved in movement execution, but also in movement planning

This analysis provides clear evidence for directional tuning during reach planning, a fundamental property of motor and premotor cortical neurons that enables the brain to prepare movements before executing them.

## Reproducibility

All analysis can be reproduced by running:
```bash
cd reach_directional_tuning
python reach_directional_tuning_analysis.py
```

Or by opening the Jupyter notebook:
```bash
jupyter notebook reach_directional_tuning_analysis.ipynb
```

The analysis uses streaming data access via LINDI, so no large downloads are required.
