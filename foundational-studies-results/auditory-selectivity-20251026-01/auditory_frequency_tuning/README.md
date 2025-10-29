# Auditory Frequency Tuning Analysis

## Overview
This analysis demonstrates auditory frequency tuning in mouse primary auditory cortex (A1) using data from [DANDI:001419](https://neurosift.app/dandiset/001419).

## Dataset Description
- **Dataset**: Cortical Circuits for the integration of short-latency auditory pathways: Auditory cortex linear probe recording
- **Recording**: Linear probe recordings from mouse A1 during pure tone presentations
- **Species**: Mus musculus (Mouse)
- **Data Type**: Spike-sorted single/multi-unit activity
- **Stimuli**: Pure tones at different frequencies (3-30 kHz range)

## Phenomenon
Auditory frequency tuning refers to the selective response of neurons to specific sound frequencies. Neurons in the auditory cortex exhibit characteristic frequency preferences, with each neuron showing maximal firing rates at its "best frequency" and reduced responses at other frequencies. This creates tuning curves that reveal the tonotopic organization of the auditory cortex.

## Analysis Goals
1. Load neural spike data and stimulus information from NWB file
2. Calculate firing rates for each unit in response to different tone frequencies
3. Generate frequency tuning curves showing neural responses across frequencies
4. Visualize raster plots of spike responses to different frequencies
5. Identify units with strong frequency selectivity

## Data Structure
- **Units**: 124 spike-sorted units with spike times
- **Trials**: 180 trials with frequency and intensity labels
- **Stimulus**: Pure tone presentations with onset times
- **Electrodes**: 64-channel linear probe spanning cortical depths
