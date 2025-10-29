# Theta Phase Precession Analysis

## Overview
This analysis demonstrates theta phase precession in hippocampal place cells using data from DANDI Archive dataset [DANDI:000059](https://neurosift.app/dandiset/000059).

## Dataset
**Title:** Cooling of Medial Septum Reveals Theta Phase Lag Coordination of Hippocampal Cell Assemblies

**Citation:** Petersen, Peter; Buzsáki, György (2025)

**Description:** This dataset contains hippocampal recordings from rats navigating a spatial maze. The study investigates how medial septum cooling affects theta oscillations and hippocampal place cell activity. The data includes:
- 409 hippocampal units (place cells)
- Position tracking during maze navigation
- LFP recordings for theta oscillation analysis
- Trial information with maze conditions

## Phenomenon: Theta Phase Precession
Theta phase precession is a fundamental property of hippocampal place cells where:
1. As an animal traverses a place field, the neuron fires at progressively earlier phases of the theta oscillation
2. This creates a temporal code where position within the place field is encoded by theta phase
3. The phase-position relationship is approximately linear, with spikes occurring earlier in the theta cycle as the animal progresses through the field

## Analysis Goals
1. Identify place cells with well-defined spatial receptive fields
2. Extract theta oscillations from LFP
3. Compute theta phase for each spike
4. Demonstrate the relationship between position in the place field and theta phase
5. Visualize phase precession for example neurons

## Files
- `01_inspect_data.py` - Initial data exploration
- `02_identify_place_cells.py` - Identify neurons with spatial tuning
- `03_theta_phase_precession.py` - Main analysis demonstrating phase precession
- `theta_phase_precession_analysis.py` - Consolidated jupytext script

## Requirements
- lindi
- pynwb
- pynapple
- numpy
- scipy
- matplotlib
- tqdm
