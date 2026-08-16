# Auditory Frequency Tuning Analysis

## Dataset

This analysis demonstrates auditory frequency tuning using data from the DANDI Archive dandiset **000986**: "Arousal-Dependent Modulation of Auditory Cortex Ensemble Dynamics" (Papadopoulos et al., 2025, *Neuron*). The dataset contains recordings from mouse primary auditory cortex (A1) during passive exposure to auditory stimuli, recorded using Neuropixels probes.

Specific session analyzed:
- Subject: LA11
- Session: 2
- Recording region: Primary auditory cortex (A1)
- Neural data: 132 single units
- Stimulus trials: 7,448 pure tone presentations
- Frequency range: 2–32 kHz (five frequency levels: 2, 4, 8, 16, 32 kHz)

## Analysis Overview

Auditory neurons encode sound frequency through frequency tuning: each neuron responds most strongly to a preferred frequency and responds less to distant frequencies. This analysis characterizes how mouse auditory cortex neurons represent the acoustic frequency spectrum.

The pipeline extracts spike times during stimulus presentations and computes firing rate responses to each frequency. For each unit, we identify its preferred frequency (where firing rate is maximal), quantify tuning breadth (frequency selectivity using Q10), and visualize population-level tuning patterns.

## Key Findings

1. **Responsive population**: 101 of 132 units (77%) showed significant responses (>1 Hz mean firing rate) to auditory stimuli, indicating robust sound-evoked activity throughout the recorded population.

2. **Distributed frequency representation**: Preferred frequencies are distributed across the stimulus range, with 37.6% preferring 2 kHz, 25.7% preferring 8 kHz, and smaller populations at 4, 16, and 32 kHz. This skewed distribution toward lower frequencies reflects the natural distribution of behaviorally relevant sounds.

3. **Frequency selectivity**: Tuning breadth values (Q10 metric) ranged from ~0.1 to 2.0, indicating variable frequency selectivity across units. Most units showed moderate selectivity, responding to multiple frequency octaves.

4. **Population encoding**: The population tuning curve—computed by averaging responses across all units—shows relatively broad frequency tuning, with peak response around 8 kHz and maintained responsiveness across the full frequency range.

5. **Response magnitudes**: Mean firing rates ranged from 0.08 to 72.1 Hz, with a population average of 6.68 Hz. Individual units achieved peak firing rates up to 107 Hz at their preferred frequencies, demonstrating strong frequency-dependent modulation.

## Methods

Analysis window: 0–100 ms after stimulus onset (stimulus duration: 25 ms)
- Firing rate computed as mean spike count per trial divided by analysis window duration
- Preferred frequency identified as the frequency with maximum mean firing rate
- Tuning breadth (Q10) computed as preferred frequency divided by bandwidth at half-maximum response
- Population statistics computed using trial-by-trial spike responses across all units

## Outputs

- `frequency_tuning_properties.png`: Five-panel summary figure showing distribution of preferred frequencies, response magnitudes, tuning selectivity, population tuning curve, and heatmap of top-responsive units
- `individual_tuning_curves.png`: 4×4 grid of individual unit tuning curves for representative neurons with different preferred frequencies
- `analyze_frequency_tuning.py`: Reproducible jupytext script with embedded markdown documentation
- `analyze_frequency_tuning.ipynb`: Jupyter notebook version of the analysis script

## Reproducibility

The analysis uses streaming access via remfile to the DANDI Archive S3 bucket, so the entire dataset is accessed on-the-fly without requiring local downloads. All dependencies (pynapple, pynwb, remfile, matplotlib) are standard Python neurophysiology packages.

To run: `python3 analyze_frequency_tuning.py`

## Citation

If using this analysis, please cite the original data source:

Papadopoulos et al. (2025). "Modulation of metastable ensemble dynamics explains the inverted-U relationship between tone discriminability and arousal in auditory cortex." *Neuron*, 113(4), 609-623.e4. https://doi.org/10.1016/j.neuron.2025.01.020

DANDI Dataset: https://dandiarchive.org/dandiset/000986
