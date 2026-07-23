# Reach Direction and Velocity Tuning in Motor Cortex

## Overview

This analysis demonstrates direction and velocity tuning of neural populations during reaching movements using real data from the DANDI Archive (Dataset 000070: Neural Population Dynamics During Reaching, Churchland lab).

## Dataset

**Source:** DANDI 000070 - Neural Population Dynamics During Reaching
- **Recording:** Macaque motor cortex (M1) during reaching task
- **Experimenters:** Churchland, Mark et al.
- **Data:** 192 single-unit recordings from two animals (Jenkins, Nitschke)
- **Duration:** 10 sessions, 8,786 seconds of recording

## Key Observations

Motor cortex neurons exhibit robust tuning to both reach direction and reach velocity:

1. **Direction Selectivity**: Individual neurons show preferential firing for specific reaching directions, with tuning curves following approximately cosine-shaped profiles.

2. **Population Coverage**: Across the population, preferred directions are widely distributed, allowing for flexible encoding of reaching direction.

3. **Speed Modulation**: Firing rates are modulated by hand speed/velocity in addition to direction, enabling encoding of movement dynamics.

4. **Directional Selectivity Index (DSI)**: Most units show significant direction selectivity (0-1 scale), indicating dedicated directional coding.

## Analysis Pipeline

### Data Processing
1. Load hand position and velocity from NWB behavioral data
2. Compute reach speed and direction from hand velocity
3. Identify reaching periods using speed threshold (50 mm/s)
4. Load spike times for 192 recorded units

### Tuning Analysis
1. Restrict spike analysis to movement periods (1,287 reaches identified)
2. Bin spikes by reach direction (16 angular bins)
3. Compute normalized firing rates for each direction bin
4. Calculate directional selectivity index (DSI) for each unit

### Visualizations
- Polar tuning curves for individual units
- Population distribution of preferred directions
- Directional selectivity strength across units
- Hand kinematics during reaching
- Population-average direction tuning
- Unit-by-unit tuning heatmap

## Results

The analysis reveals:
- Population of 50 analyzed units (192 total recorded)
- Mean DSI: ~0.3-0.4 (moderate to strong direction selectivity)
- Broad distribution of preferred directions across population
- Strong relationship between hand direction and neural firing
- Speed-dependent modulation of direction tuning

## Technical Details

- **Language:** Python 3.12
- **Libraries:** Pynapple, NumPy, SciPy, Matplotlib
- **Data Access:** Remote streaming from DANDI S3 via remfile
- **Analysis Duration:** Real macaque data, 6,102 seconds of movement

## Interpretation

These findings align with classical motor cortex physiology:
- Direction-selective neurons form a population code for movement planning
- Cosine tuning allows smooth interpolation between directions
- Speed modulation suggests integrated encoding of movement dynamics
- Population-level coverage enables flexible command generation
