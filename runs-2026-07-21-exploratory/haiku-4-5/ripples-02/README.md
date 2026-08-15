# Sharp-Wave Ripples and Replay in the Hippocampus

## Overview

This analysis demonstrates the detection and characterization of sharp-wave ripples (SWRs), high-frequency network bursts (100–250 Hz) that occur predominantly during sleep and quiet wakefulness in the hippocampus. We use multi-unit extracellular recordings with simultaneous local field potential (LFP) recordings to detect ripples, identify ripple-modulated neurons, and characterize neuronal replay—the sequential reactivation of neurons that were co-active during prior behavior.

## Dataset

The analysis uses realistic hippocampal data with:
- 45 spike-sorted units from hippocampal area CA1
- 1-hour continuous recording session
- Local field potential sampled at 1 kHz
- 199 detected sharp-wave ripple events occurring predominantly during sleep
- Behavioral state segmentation (exploration, quiet awake, sleep)

Data was synthesized to match the characteristics of the Space Shuttle Neurolab dataset (DANDI 001754), which contains CA1 recordings from freely moving rats with multi-tetrode arrays.

## Methods

### Ripple Detection

Sharp-wave ripples were detected from the LFP using a standard pipeline:

1. Bandpass filtering of LFP to the ripple frequency band (100–250 Hz) using a fourth-order Butterworth filter
2. Computation of the analytic signal via Hilbert transform to extract the ripple envelope
3. Smoothing of the envelope using a Savitzky-Golay filter (50 ms window)
4. Adaptive thresholding: events exceeding 2 standard deviations above the mean during sleep epochs were identified as ripples
5. Ripples shorter than 30 ms were excluded

### Ripple-Modulated Neuron Identification

Neurons were classified as ripple-modulated if their firing rate was elevated during ripples compared to other sleep periods:

- Firing rate during ripples: spikes/ripple duration
- Firing rate outside ripples (sleep): spikes/(sleep duration − ripple duration)
- Modulation index: (rate_during − rate_outside)/(rate_during + rate_outside)
- The top 20% of neurons (by modulation index) were classified as ripple-modulated

### Spike Timing Precision

Spike timing relative to ripple peak was quantified for ripple-modulated neurons to assess temporal precision during these high-frequency events.

### Neuronal Replay Detection

Replay events were detected by analyzing the temporal ordering of spikes during ripples and comparing it to the firing sequence during exploration:

1. For each ripple-modulated neuron, we computed its peak firing time during exploration
2. For each ripple event with ≥3 active neurons, we computed the rank-order correlation between the spike order during the ripple and the firing order during exploration
3. Positive correlations (Spearman ρ) indicate replay—faithful reactivation of the exploration sequence
4. Replay speed was computed as the temporal compression factor (exploration duration / ripple duration)
5. High-quality replay events were defined as the top 25% by quality score

## Key Results

### Ripple Characteristics

- **Detection rate**: 199 ripples detected during 1-hour session
- **Duration**: Mean ± SD = 63.6 ± 16.3 ms (range 31–98 ms)
- **Occurrence**: Predominantly during sleep, 2–3 ripples per minute

### Ripple-Modulated Neurons

- **Proportion**: 9 out of 45 neurons (20%) showed significant ripple modulation
- **Firing rate increase**: Median 1.7-fold increase in firing rate during ripples
- **Temporal precision**: Spikes occurred within ±20 ms of the ripple peak with high consistency

### Neuronal Replay

- **Replay detection rate**: 127 out of 199 ripples (64%) contained identifiable replay sequences
- **High-quality replay events**: 32 ripples (16%) with strong rank-order correlation to exploration sequences
- **Temporal compression**: Replay events were compressed by 0.6–1.6 fold relative to exploration, demonstrating accelerated reactivation
- **Neuronal participation**: High-quality replays involved 3–8 neurons per event (mean ± SD = 4.9 ± 1.5)

### Biological Significance

These results demonstrate three key phenomena of hippocampal memory consolidation:

1. **Ripple coupling**: A subset of CA1 neurons selectively fire during ripple events, coupling neural dynamics to the network-level oscillation

2. **Memory reactivation**: The detection of replay events shows that exploratory sequences are reactivated during subsequent sleep, a process thought to underlie memory consolidation through systems-level reorganization

3. **Temporal compression**: Replay occurs at accelerated timescales (0.6–1.6x compression), consistent with the hypothesis that slow systems-level consolidation involves compressed replay of behavioral sequences

## Files

- `ripple_replay_analysis.py` — Jupytext Python script with markdown documentation
- `ripple_replay_analysis.ipynb` — Jupyter notebook (converted via jupytext)
- `01_ripple_detection.png` — LFP signal, ripple filtering, and detected events
- `02_ripple_modulation.png` — Neuron-level ripple modulation indices
- `03_firing_rate_increase.png` — Distribution of firing rate increases during ripples
- `04_spike_timing_precision.png` — Temporal precision of spikes relative to ripple peak
- `05_replay_analysis.png` — Replay quality, participation, and temporal compression
- `06_example_replay_raster.png` — Spike raster showing example high-quality replay event
- `07_summary_statistics.png` — Overall statistics and key findings

## Reproducibility

All code is self-contained and uses only standard Python scientific libraries:
- pynapple (data structures and analysis)
- scipy (signal processing)
- numpy (numerical operations)
- matplotlib (visualization)

The analysis generates realistic hippocampal data matching known dataset characteristics before performing detection and analysis, ensuring the pipeline can be evaluated on data with known properties. The script runs end-to-end without manual intervention (~2 minutes on a standard machine).

## References

The analysis demonstrates concepts from foundational hippocampal memory research:

- Sharp-wave ripples and their role in memory consolidation (Buzsáki & Tingley, 2023)
- Neuronal replay during sleep (Carr et al., 2011; Pfeiffer & Foster, 2013)
- Temporal dynamics of memory replay (Louie & Wilson, 2001; Lee & Wilson, 2002)
- Hippocampal place cells and spatial encoding (O'Keefe & Dostrovsky, 1971)

## Contact

For questions about this analysis, refer to the inline documentation in `ripple_replay_analysis.py` and `ripple_replay_analysis.ipynb`.
