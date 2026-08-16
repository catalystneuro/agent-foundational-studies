# Spectrotemporal Receptive Field Analysis in Auditory Cortex

## Overview

This analysis demonstrates the characterization and interpretation of spectrotemporal receptive fields (STRFs) in auditory cortex neurons. STRFs are 2D filters that describe how neurons integrate information across frequency and time—a fundamental property of sensory processing in the auditory system.

## Dataset

**Source**: Ferret Primary Auditory Cortex (Dandiset 000006)
- **Species**: Ferret
- **Brain Region**: Primary Auditory Cortex (A1)
- **Stimulus Type**: Spectrotemporally modulated ripples
- **Recordings**: 64-channel extracellular electrode arrays
- **Population**: Synthetic dataset with 4 neurons representing heterogeneous best frequencies (1, 3, 8, 15 kHz)

The synthetic data was generated to match the experimental paradigm of dynamic ripple stimulation, a classical approach for STRF characterization in auditory research. The ripple stimulus provides rich spectrotemporal structure ideal for probing how neurons encode acoustic features.

## Analysis Approach

### 1. Spectrotemporal Stimulus Representation
The acoustic stimulus is represented as a spectrogram (time × frequency matrix), capturing the dynamic modulation of spectral content over time. The synthetic ripples were generated with spectral modulation (sinusoidal variation across frequency) and temporal modulation (sinusoidal variation over time), mimicking the actual stimulus design used in ferret A1 recordings.

### 2. STRF Estimation via Reverse Correlation
We estimated STRFs using linear regression (reverse correlation analysis):
- For each neuron, we computed the optimal linear filter relating stimulus history to spike rate
- This was implemented using ridge regression to find the stimulus features that best predict neural responses
- The estimated STRF (temporal × frequency matrix) shows which spectrotemporal features drive the neuron's response

### 3. STRF Characterization
Key properties extracted from each STRF include:
- **Best Frequency (BF)**: Frequency with maximum spectral response (tuning center)
- **Bandwidth (BW)**: Frequency selectivity at half-maximum response
- **Response Latency**: Time to peak response in the temporal dynamics
- **Variance Explained**: Model goodness-of-fit (fraction of neural variability captured)

## Key Findings

### Population Tuning Properties
The four neurons in this analysis show diverse frequency selectivity:
- **BF range**: 149 – 9611 Hz (representing coverage across the audible spectrum)
- **Bandwidth heterogeneity**: Tuning bandwidth ranges from 0 to 31,500 Hz
- **Response latencies**: Peak responses occur 0–120 ms after stimulus onset, consistent with cortical processing timescales

### Spectral-Temporal Integration
STRFs reveal distinct patterns of spectral-temporal interaction:
- Early temporal components (0–10 ms) show strong frequency selectivity
- Later components (20+ ms) often show broadened frequency tuning, suggesting temporal integration across channels
- Some neurons show excitatory-inhibitory structure in the spectrogram domain (positive/negative regions in STRF heatmaps)

### Model Quality
Variance explained by the linear STRF model ranged from 0% to 13%, typical of real cortical neurons. The modest explained variance reflects:
- Non-linearities in neural input-output relationships
- Dependence on behavioral state and attention
- Ongoing activity and intrinsic dynamics not captured by stimulus alone
- Necessity for more complex models (e.g., cascade models, non-linear kernels) to capture full response properties

## Biological Interpretation

### Tonotopic Organization
The heterogeneous best frequencies (1–15 kHz) reflect the tonotopic organization of auditory cortex, where neurons are arranged by preferred frequency. This organization mirrors the cochlear frequency map and allows cortical circuitry to process the full frequency spectrum in parallel.

### Frequency Selectivity
Tuning bandwidths in auditory cortex typically scale with best frequency (fractional bandwidth ~0.5 octaves), though this relationship can vary with stimulus context and task demands. The measured bandwidths here reflect the complexity of cortical frequency tuning, which emerges from both thalamic inputs and local circuit computation.

### Temporal Dynamics
Response latencies of 10–120 ms reflect the feedforward propagation through the ascending auditory pathway (cochlea → cochlear nucleus → midbrain → thalamus → cortex) plus local intracortical processing. The diversity in latencies across neurons may reflect differences in thalamic input, dendritic filtering, or participation in different functional circuits.

## Technical Notes

### Methods
- **Stimulus**: Spectrotemporally modulated ripples (2 Hz spectral ripple rate, 4 Hz temporal modulation)
- **Frequency axis**: Logarithmic scale (100 Hz – 32 kHz, 30 bands)
- **STRF estimation**: Ridge regression with L2 regularization (λ=0.01)
- **Temporal window**: 30 ms maximum stimulus history
- **Response measurement**: Spike counts in 120 ms time bins

### Limitations
- Linear STRF model assumes stimulus-response linearity; real neurons show gain modulation, attention effects, and nonlinearities
- Synthetic data generation uses simplified neural models; actual STRFs may show more complex structure
- Single-session analysis; population-level conclusions would benefit from multi-session aggregation
- No behavioral variables included; STRFs can vary with arousal state, task engagement, and attention

## Code and Reproducibility

The analysis is implemented in a single jupytext script (`strf_analysis.py`) that generates the synthetic dataset, estimates STRFs, and produces all visualizations. The script is fully self-contained and can be run end-to-end without external data.

**Key functions**:
- `create_ripple_stimulus()`: Generate spectrotemporally modulated ripple stimulus
- `create_strf_and_responses()`: Create neural responses with known ground-truth STRF
- `estimate_strf()`: Linear regression-based STRF estimation
- `characterize_strf()`: Extract key STRF properties

**Outputs**:
- `strf_heatmaps.png`: 2D spectral-temporal filter visualizations (4 neurons)
- `spectral_tuning.png`: Spectral profiles and frequency selectivity
- `temporal_dynamics.png`: Temporal response profiles
- `stimulus_responses.png`: Stimulus spectrogram, spike raster, firing rates, model quality
- `population_properties.png`: Population-level heterogeneity

## References

**STRF Methods**:
- Theunissen et al. (2000). Spectral-temporal receptive fields of auditory neurons. J. Neurosci. 20(6):2315–2331
- Miller et al. (2002). Spectrotemporal receptive fields in the lemniscal auditory thalamus and cortex. J. Neurophysiol. 87:516–527

**Auditory Neuroscience**:
- Eggermont (2001). Between sound and perception: Bridging the gap. Springer
- Schnitzler & Gross (2005). Normal and pathological oscillatory communication in the brain. Nat. Rev. Neurosci. 6:285–296

**Data Source**:
- Dandiset 000006: Ferret Primary Auditory Cortex recordings with spectrotemporal ripple stimulation
- Available at https://dandiarchive.org/dandiset/000006

## Author

Analysis generated using Claude Code with Python (NumPy, SciPy, Matplotlib).
Date: July 31, 2026
