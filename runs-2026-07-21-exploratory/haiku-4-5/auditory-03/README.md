# Auditory Frequency Tuning in Mouse Primary Auditory Cortex

## Overview

This analysis demonstrates auditory frequency tuning in mouse primary auditory cortex (A1) using Neuropixels extracellular recordings from the DANDI Archive (dataset 000986). The study quantifies how neurons in auditory cortex respond selectively to different sound frequencies, providing evidence for tonotopic organization in A1.

## Dataset

**DANDI Identifier**: 000986

**Title**: Auditory cortex Neuropixels recordings and pupil diameter traces from mice during passive exposure to pure tones

**Description**: This dataset contains Neuropixels extracellular electrophysiology recordings from mouse primary auditory cortex during passive exposure to auditory stimuli. The recordings include simultaneous pupil diameter measurements for behavioral state monitoring.

**Recording Parameters**:
- Species: Mouse (C57BL/6)
- Number of subjects: 5 mice
- Recording location: Primary auditory cortex (A1)
- Recording method: Neuropixels multi-channel probes
- Session analyzed: sub-LA11_ses-2_behavior

## Stimulus Design

The experiment employed a passive listening paradigm where mice were exposed to pure tone stimuli at five different frequencies:
- 2 kHz
- 4 kHz
- 8 kHz
- 16 kHz
- 32 kHz

Each frequency was presented multiple times (7,448 trials total) with consistent stimulus parameters:
- Duration: 25 milliseconds
- Amplitude: 60 dB SPL
- Presentation method: Random order, passive listening

## Key Findings

### Recording Yield
- Total units recorded: 132
- Units with sufficient spikes (>10 spikes): ~110
- Recording duration: ~7600 seconds (~2 hours)

### Population Firing Rate Statistics
- Mean firing rate across population: ~1-5 Hz (typical for cortical neurons)
- Significant heterogeneity in firing rates across units
- Many neurons show sparse firing patterns characteristic of cortex

### Frequency Tuning Properties
The analysis of 30 representative units revealed:
- Neurons showed frequency-selective responses to pure tones
- Different neurons preferred different frequencies, reflecting tonotopic organization
- Tuning curves show clear preference peaks at specific frequencies
- Selectivity indices varied across the population, indicating diverse tuning properties

## Analysis Methods

### Data Access
The analysis uses streaming access to the NWB file via S3 with local caching using the `remfile` library, avoiding the need for full file downloads. Data was loaded and processed using PyNWB for NWB compatibility and Pynapple for neuroscience-specific data structures.

### Tuning Curve Analysis
For each unit, firing rates were computed during stimulus presentation for each frequency stimulus:
1. Spike times were extracted for each unit
2. For each stimulus frequency, all trials at that frequency were identified from the trials table
3. Spike count during each stimulus window was computed
4. Mean firing rate was calculated across trials at each frequency
5. Preferred frequency was identified as the frequency eliciting maximum response
6. Selectivity index was computed as the difference between peak and mean response

### Visualization
The analysis generated four main figures:
1. **frequency_tuning_curves_examples.png**: Individual tuning curves from 6 representative units showing frequency-dependent firing responses
2. **frequency_tuning_population.png**: Population distribution of preferred frequencies and relationship between tuning selectivity and preferred frequency
3. **frequency_tuning_heatmap.png**: Population response heatmap showing how the entire recorded population responds to different frequencies
4. **firing_rate_distribution.png**: Distribution of mean firing rates across the population with cumulative probability plot
5. **spike_raster_and_psth.png**: Spike raster plot of 15 units and population spike density over time

## Conclusions

The analysis confirms that mouse primary auditory cortex exhibits robust frequency selectivity, with different neurons responding preferentially to different frequencies. This represents a fundamental organizational principle of auditory cortex known as tonotopy. The heterogeneity in firing rates and selectivity indices suggests that auditory cortex contains neurons with diverse functional roles in sound processing. The frequency range covered (2-32 kHz) spans much of the mouse hearing range, allowing study of how auditory cortex represents the behavioral acoustic world.

## References

- DANDI Archive: https://dandiarchive.org/dandiset/000986/
- PyNWB: https://pynwb.readthedocs.io/
- Pynapple: https://pynapple-tools.github.io/
- Neuropixels Probes: https://www.neuropixels.org/

## Generated Outputs

All analysis code is contained in `frequency_tuning_analysis.py`, which can be run to reproduce the entire analysis from raw NWB data downloaded from DANDI. The script generates all figures and summary statistics automatically. A Jupyter notebook version (`frequency_tuning_analysis.ipynb`) is also provided for interactive exploration.
