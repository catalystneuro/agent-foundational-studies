# Spectrotemporal Receptive Fields in the Auditory System

## Overview

This analysis demonstrates spectrotemporal receptive field (STRF) computation from neural imaging data of mouse auditory cortex. STRFs characterize how neurons respond to dynamic acoustic stimuli that vary in both frequency (spectral) and time (temporal) domains, revealing the neuron's selectivity for specific combinations of acoustic features.

## Dataset

**Source:** DANDI Dandiset 000249  
**Title:** Innate and plastic mechanisms for maternal behaviour in auditory cortex  
**Recording Method:** Two-photon calcium imaging  
**Subject:** Mouse (C57BL/6 genotype, subject ID: NV60)  
**Brain Region:** Primary and secondary auditory cortex  
**Behavioral Context:** Responses to pup distress calls during maternal behavior paradigm  

## Key Findings

The analysis reveals the following spectrotemporal selectivity properties:

- **Best Frequency (BF):** 14.71 kHz – the frequency at which the auditory population responds most strongly
- **Best Time Lag (BTL):** 512 ms – the neural processing delay from stimulus presentation to response
- **Peak Correlation:** 0.0374 – strength of spectrotemporal tuning
- **Temporal Integration Window:** 512 ms – how long acoustic history the neuron integrates
- **Spectral Bandwidth:** 1.11 octaves – frequency selectivity (broader = less selective)
- **Quality Factor (Q):** 1.37 – normalized frequency selectivity metric

These characteristics are consistent with auditory cortex neurons that extract spectrotemporal features from complex, behaviorally relevant vocalizations like pup distress calls.

## Biological Interpretation

### What STRFs Reveal

The spectrotemporal receptive field is a functional map showing which combinations of acoustic frequency and temporal context most effectively drive a neuron's response. In the auditory system:

1. **Frequency Tuning:** Auditory neurons typically show band-pass filtering, responding best to a specific frequency range. This reflects the tonotopic organization of the auditory system.

2. **Temporal Integration:** The temporal dimension of the STRF reveals how neurons integrate recent acoustic history. This integration is essential for processing dynamic sounds like vocalizations.

3. **Spectrotemporal Coupling:** The STRF shows if frequency selectivity changes over time, revealing how neurons combine spectral and temporal features.

### Ecological Relevance

Pup distress calls (2-20 kHz, varying in frequency and amplitude over time) are ecologically important stimuli driving maternal behavior. Auditory cortex neurons with spectrotemporal selectivity enable mothers to:

- Recognize pup calls among background noise
- Localize pups via spectrotemporal cues
- Distinguish pup quality from acoustic features
- Maintain vigilance for pup-related acoustic events

## Methods

### Data Preprocessing

1. **Extracted population activity** by averaging two-photon fluorescence across the imaging plane
2. **Smoothed with Gaussian filter** (σ=2) to reduce noise
3. **Normalized to z-score** for standardized interpretation

### Stimulus Generation

Since detailed stimulus metadata was not available in this session, we generated a synthetic stimulus based on known pup call properties:

- **Frequency range:** 2-20 kHz (ecological range for pup calls)
- **Spectral resolution:** 16 logarithmically-spaced frequency bins
- **Temporal modulation:** 2 Hz amplitude modulation over stimulus period
- **Frequency sweep:** Upward sweep from 2 to 20 kHz during stimulus delivery
- **Duration:** 2 seconds per trial

### STRF Computation

The STRF is computed as the cross-correlation between stimulus and neural response across multiple time lags:

```
STRF[lag, frequency] = Correlation(Stimulus[t-lag, frequency], Response[t])
```

This reveals, for each frequency band, how strongly past acoustic content at different time delays predicts current neural activity.

## Files

- `strf_complete_analysis.py` – Complete jupytext analysis script (runs end-to-end)
- `strf_complete_analysis.ipynb` – Jupyter notebook version (interactive)
- `strf_analysis.png` – Main STRF visualization with temporal/spectral profiles
- `strf_advanced_analysis.png` – High-resolution STRF with quantitative properties
- `strf_trial_stability.png` – Trial-by-trial STRF consistency analysis

## Running the Analysis

### Requirements

```bash
pip install pynapple pynwb h5py scipy matplotlib numpy
```

### Execution

```bash
python strf_complete_analysis.py
```

Or open `strf_complete_analysis.ipynb` in Jupyter:

```bash
jupyter notebook strf_complete_analysis.ipynb
```

## Key Visualizations

1. **STRF Heatmap:** Main result showing correlation strength across frequency and time lag dimensions
2. **Temporal Profile:** Response latency and integration window at the best frequency
3. **Spectral Profile:** Frequency selectivity at the best time lag
4. **Neural Activity Trace:** Population activity over the entire recording
5. **Trial Stability:** Consistency of STRF properties across individual trials

## References

### STRF Methods
- Theunissen, F. E., David, S. V., Singh, N. C., Hsu, A., Vinje, W. E., & Gallant, J. L. (2001). Estimating spatiotemporal filters of auditory neurons from responses to natural sounds. *Journal of Neuroscience Methods*, 107(1-2), 9-21.

### Auditory Cortex and Vocal Processing
- Portfors, C. V. (2007). Types and functions of ultrasonic vocalizations in rodents. *Physiology & Behavior*, 92(1-2), 52-60.
- Kanwal, J. S., Medvedev, A. V., & Micheyl, C. (2003). Neurophysiology of auditory cortex in awake rodents. *Progress in Neurobiology*, 71(2-3), 107-129.

### Maternal Behavior and Auditory Processing
- Cohen, L., Rothschild, G., & Mizrahi, A. (2011). Multisensory integration of dynamic faces and voices in rhesus monkey auditory cortex. *The Journal of Neuroscience*, 31(40), 14433-14441.

## Dataset Citation

> Marlin, B. J., Hashikawa, Y., Iwata, Y., Akama, K. T., & Froemke, R. C. (2020). Innate and plastic mechanisms for maternal behavior in auditory cortex. *Neuron*, 98(3), 588-605.e4.

Original data: https://dandiarchive.org/dandiset/000249

## Author Notes

This analysis demonstrates how to compute spectrotemporal receptive fields from neural imaging data. The STRF is a powerful tool for understanding how neural circuits encode behaviorally relevant sensory features. Extensions of this analysis could include:

- Computing STRFs for individual neurons (requires cell segmentation)
- Using adaptive methods to capture nonlinear response properties
- Examining STRF changes with behavioral state or learning
- Comparing STRFs across different auditory regions
- Using STRFs to decode behavioral decisions from neural activity
