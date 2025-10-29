# Orientation Selectivity in Visual Cortex

## Overview
This analysis demonstrates orientation selectivity in mouse primary visual cortex (V1) using data from the DANDI Archive. Orientation selectivity is a fundamental property of visual neurons where they respond preferentially to visual stimuli (typically gratings) presented at specific orientations.

## Dataset
**DANDI:000168** - "Simultaneous loose seal cell-attached recordings and two-photon imaging of GCaMP8 expressing mouse V1 neurons with drifting gratings visual stimuli"

- **Source**: Svoboda Lab, HHMI
- **Recording technique**: Two-photon calcium imaging + electrophysiology
- **Brain region**: Primary visual cortex (V1), Layer 2/3
- **Stimulus**: Full-field, high-contrast drifting gratings in 8 directions
- **Subject**: Mouse (AAV-infected L2/3 pyramidal neurons expressing GCaMP variants)

## Scientific Background
Orientation selectivity is one of the most well-studied properties of visual cortex neurons, first described by Hubel and Wiesel in the 1960s. V1 neurons respond maximally to edges or gratings at specific orientations, forming the basis for visual processing of shapes and contours.

## Analysis Goals
1. Load NWB files containing neural responses to oriented gratings
2. Extract spike times and stimulus information
3. Compute orientation tuning curves for individual neurons
4. Identify orientation-selective neurons
5. Visualize population-level orientation selectivity

## Expected Results
- Neurons should show tuning curves with peaks at preferred orientations
- Orientation selectivity index (OSI) distributions
- Example neurons with clear orientation preferences
- Population statistics on orientation tuning
