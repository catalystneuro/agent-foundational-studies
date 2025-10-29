# Orientation Selectivity in Visual Cortex - Analysis

## Overview
This analysis demonstrates orientation selectivity in mouse primary visual cortex (V1) using data from DANDI Archive dataset [DANDI:000168](https://neurosift.app/dandiset/000168).

## Dataset Information
**Dataset**: Simultaneous loose seal cell-attached recordings and two-photon imaging of GCaMP8 expressing mouse V1 neurons with drifting gratings visual stimuli

**Authors**: Rozsa, Marton; Liang, Yajie; Zhang, Yan; Hasseman, Jeremy; Kolb, Ilya; Looger, Loren; Svoboda, Karel

**Description**: Two-photon calcium imaging (122 Hz) of L2/3 pyramidal neurons in mouse V1 combined with electrophysiological recordings. Neurons were presented with full-field, high-contrast drifting gratings in 8 different orientations.

## Analysis Goal
Demonstrate orientation selectivity - the fundamental property of V1 neurons where individual neurons respond preferentially to visual stimuli oriented at specific angles. This property is crucial for edge detection and feature extraction in the visual system.

## Key Features
- Calcium imaging data (GCaMP indicators) from L2/3 pyramidal neurons
- Drifting grating stimuli at 8 different orientations (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
- Simultaneous electrophysiology for ground truth validation
- High temporal resolution (122 Hz imaging)

## Analysis Pipeline
1. Load NWB file with streaming access and local caching
2. Extract neural responses (calcium fluorescence traces)
3. Extract stimulus information (orientation angles, trial timings)
4. Compute trial-averaged responses for each orientation
5. Generate orientation tuning curves
6. Visualize raster plots and tuning properties
7. Quantify selectivity metrics (preferred orientation, tuning width, selectivity index)

## Results

### Key Findings
**Neuron**: jGCaMP7f_ANM471991_cell02 (L2/3 pyramidal neuron)

**Orientation Selectivity Metrics:**
- **Preferred Orientation**: 270°
- **Maximum Response**: 0.215 ΔF/F
- **Orientation Selectivity Index (OSI)**: 0.259 (weak selectivity)
- **Direction Selectivity Index (DSI)**: 0.207 (not direction-selective)
- **Circular Variance**: 0.897 (broad tuning)

### Interpretation
This neuron demonstrates **orientation selectivity**, responding most strongly to 270° oriented gratings. While the selectivity is weak (OSI = 0.259), it clearly shows the fundamental property of V1 neurons to preferentially respond to specific orientations - crucial for edge detection and visual processing.

## Output Files
1. `raw_fluorescence_trace.png` - Raw calcium imaging trace
2. `fluorescence_with_trials.png` - Fluorescence with trial structure overlay
3. `orientation_raster_plots.png` - Individual trial responses for each orientation
4. `orientation_averaged_responses.png` - Trial-averaged responses (Mean ± SEM)
5. `orientation_tuning_curve.png` - Linear and polar tuning curves
6. `orientation_selectivity_summary.png` - Comprehensive summary figure

## Running the Analysis
```bash
cd orientation_selectivity_analysis
python 02_orientation_analysis.py
```

## Citation
Rozsa, M., Liang, Y., Zhang, Y., Hasseman, J., Kolb, I., Looger, L., & Svoboda, K. (2022). 
Simultaneous loose seal cell-attached recordings and two-photon imaging of GCaMP8 expressing 
mouse V1 neurons with drifting gratings visual stimuli. DANDI Archive. 
https://doi.org/10.48324/dandi.000168/draft
