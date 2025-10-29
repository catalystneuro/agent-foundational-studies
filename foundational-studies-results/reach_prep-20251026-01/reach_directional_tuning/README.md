# Directional Tuning During Reach Planning

## Overview
This analysis demonstrates directional tuning in motor and premotor cortical neurons during the planning phase of reaching movements. The analysis uses data from [DANDI:000140](https://neurosift.app/dandiset/000140), which contains neural recordings from macaque motor cortex (M1) and dorsal premotor cortex (PMd) during a delayed reaching task.

## Dataset Information
- **Dataset**: MC_Maze_Small - macaque primary motor and dorsal premotor cortex spiking activity during delayed reaching
- **Subject**: Jenkins (Macaca mulatta)
- **Recording areas**: Primary motor cortex (M1) and dorsal premotor cortex (PMd)
- **Task**: Center-out delayed reaching with maze barriers
- **Number of units**: 142 neurons
- **Number of trials**: 100 (training set)

## Scientific Question
Do neurons in motor and premotor cortex show directional tuning during the delay period (planning phase) before movement execution?

## Key Timing Events
- **Target On**: Visual cue indicating the reach target location
- **Go Cue**: Signal to initiate the reach movement
- **Movement Onset**: When the monkey begins reaching
- **Delay Period**: Time between target presentation and go cue (the planning period)

During the delay period, the monkey knows where to reach but must wait for the go cue. This is the planning phase where we expect to see directional tuning emerge.

## Analysis Steps
1. Load NWB file with neural spiking data and behavioral information
2. Extract reach target directions from trial data
3. Calculate firing rates during the delay period for each neuron and direction
4. Compute directional tuning curves
5. Visualize tuning properties (preferred directions, tuning strength)
6. Generate raster plots and population analyses

## Expected Results
- Many neurons should show significant directional tuning during the delay period
- Neurons will have "preferred directions" where they fire most strongly
- Tuning curves will show cosine-like shapes (characteristic of motor cortex)
- Population activity should represent the planned movement direction

## Files
- `README.md`: This file
- Analysis scripts (to be created)
- Generated figures showing tuning curves and neural responses
