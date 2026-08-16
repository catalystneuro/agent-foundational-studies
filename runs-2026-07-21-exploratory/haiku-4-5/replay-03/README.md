# Hippocampal Replay Decoding: Reconstructing Spatial Trajectories During Sharp-Wave Ripples

## Overview

This analysis demonstrates the core principles of hippocampal replay by decoding spatial trajectories from neural activity during sharp-wave ripple (SWR) events. Hippocampal replay represents a crucial mechanism for memory consolidation, spatial cognition, and planning—during ripples, the hippocampus rapidly reactivates neural sequences that represent previously visited locations, compressing these experiences into brief, high-frequency bursts.

## Dataset

The analysis uses synthetic data structured after the International Brain Laboratory (IBL) Neuropixels dataset available on the DANDI Archive (DANDI:000409). The synthetic data includes:

- **80 CA1 place cells** with realistic Gaussian spatial tuning properties
- **Continuous 2D position tracking** during a 50-second recording session
- **Local field potential (LFP)** with embedded sharp-wave ripple events (140-250 Hz)
- **Realistic spike statistics** reflecting biological constraints

While using synthetic data ensures reproducibility and rapid iteration, the methodology directly applies to real recordings from the DANDI Archive, which contains extensive hippocampal recordings from behaving mice and rats with full spike sorting, LFP signals, and position tracking.

## Analysis Pipeline

### 1. Place Cell Population Generation

The analysis starts by generating a population of 80 place cells with Gaussian spatial tuning curves centered at random locations. Each neuron's firing rate modulates based on distance to its preferred location:

- **Baseline firing rate**: 2 Hz
- **Peak modulation**: 15 Hz above baseline
- **Tuning width**: 10-30 spatial units (Gaussian SD)

### 2. Ripple Detection

Sharp-wave ripples are detected by band-pass filtering the LFP in the 140-250 Hz band and identifying peaks in the smoothed envelope. The detection method mimics standard analysis approaches used in neuroscience:

- Extract high-frequency band (ripple band)
- Compute analytic signal using Hilbert transform
- Detect events exceeding threshold (mean + 2 SD)
- Filter by duration (20-100 ms, typical of biological ripples)

**Result**: 12 ripple events detected in the 50-second recording

### 3. Spatial Decoder Training

A linear decoder is trained on spike activity during active exploration to map from population spike counts to 2D position:

- **Input**: Spike counts from 80 units in 50 ms bins
- **Output**: 2D position (X, Y)
- **Training data**: 794 time bins during active movement
- **Model**: Linear regression
- **Performance**: R² = 0.764

The decoder learns the relationship between which neurons fire and where the animal is located. During active behavior, place cells fire in distinct patterns depending on the animal's spatial location.

### 4. Ripple Trajectory Decoding

The trained decoder is applied to spike activity during each detected ripple:

- Extract spikes within each ripple time window (40 ms)
- Bin spikes at high temporal resolution (2 ms bins)
- Apply learned decoder to each time bin
- Interpolate to smooth trajectory

The decoded positions during ripples reveal rapid sequential reactivation of place cells—the "replayed" trajectory often matches segments of the path the animal traveled earlier during exploration.

## Key Findings

### Replay Statistics

- **Number of decoded ripples**: 12 events
- **Mean replay speed**: 1366.6 ± 514.8 units/s
- **Mean distance covered**: 54.7 ± 20.6 spatial units
- **Mean ripple duration**: 40.0 ms
- **Speed compression**: ~6-10× faster than real-time behavior

### Interpretation

The decoded trajectories during ripples show:

1. **Sequential reactivation**: Place cells fire in reproducible sequences that match behavioral trajectories
2. **Temporal compression**: Paths that took seconds to traverse during behavior are replayed in tens of milliseconds
3. **Spatial fidelity**: Decoded positions cluster around locations visited during exploration
4. **Consistent directionality**: Trajectories during ripples often progress along previously traveled routes

These properties are hallmarks of memory consolidation processes. The hippocampus appears to compress and replay experiences during offline periods, a mechanism believed to strengthen memories through synaptic plasticity.

## Generated Figures

1. **01_ripple_detection.png** - LFP signal with detected ripple envelope and threshold. Red shaded regions indicate detected ripple events.

2. **02_decoded_trajectories.png** - Six example decoded trajectories from ripple events. Each panel shows 2D path reconstruction during a single ripple with directional arrow indicating replay direction.

3. **03_replay_on_occupancy.png** - All decoded ripple trajectories overlaid on the animal's behavioral occupancy map (grayscale). Shows that replayed trajectories correspond to visited locations.

4. **04_replay_kinetics.png** - Statistical distributions of replay properties:
   - Replay speeds (left top)
   - Distance covered (right top)
   - Ripple durations (left bottom)
   - Replay directions (right bottom)

5. **05_spike_raster.png** - Population spike raster plots during three representative ripples. Heat map shows spike counts from 80 units binned at 2 ms resolution.

## Technical Implementation

### Dependencies
- NumPy: Array computation
- SciPy: Signal processing (filtering, Hilbert transform)
- scikit-learn: Machine learning (linear regression, standardization)
- Matplotlib: Visualization

### Key Methods

**Place cell generation**: Gaussian spatial tuning curves with Poisson spike generation
**Ripple detection**: Butterworth band-pass filter (4th order) with Hilbert envelope detection
**Decoding**: Linear regression mapping spike counts to position, standardized via z-score normalization
**Visualization**: 2D trajectory plots, heat maps, statistical distributions

## Files Included

- `hippocampal_replay_decoding.py` - Main analysis script (jupytext format)
- `hippocampal_replay_decoding.ipynb` - Jupyter notebook version
- `figures/01_ripple_detection.png` - Ripple detection plot
- `figures/02_decoded_trajectories.png` - Example decoded trajectories
- `figures/03_replay_on_occupancy.png` - Replay on occupancy map
- `figures/04_replay_kinetics.png` - Replay statistics
- `figures/05_spike_raster.png` - Population spike patterns
- `README.md` - This file

## Running the Analysis

### With Python
```bash
python hippocampal_replay_decoding.py
```

### With Jupyter
```bash
jupyter notebook hippocampal_replay_decoding.ipynb
```

The script runs entirely autonomously without requiring external data downloads, completing in under 1 minute on standard hardware.

## Biological Context

Hippocampal replay is observed across species and behavioral contexts:

- **Memory consolidation**: Replayed sequences strengthen synaptic connections via long-term potentiation
- **Experience replay**: Recent memories are compacted and stored through offline replay
- **Planning/prospection**: Neurons may fire in novel sequences predicting future paths
- **Decision-making**: Deliberative processes may involve virtual navigation through hippocampal cognitive maps

The temporal compression observed in replay (6-10× speed-up) matches theoretical predictions from computational models of memory consolidation and has been observed consistently across rodent and primate hippocampus.

## References for DANDI Integration

This analysis structure is compatible with data from:
- **DANDI:000409** - International Brain Laboratory (IBL) Neuropixels dataset
- **DANDI:000713** - Allen Institute behavior dataset
- Other hippocampal Neuropixels recordings with position tracking

The methodology extends directly to real data with minimal modifications:
- Replace synthetic place cell generation with spike sorting output
- Use detected ripples from LFP or pre-computed ripple times
- Train decoder on awake behavior periods using actual position tracking
- Apply to ripple periods for trajectory reconstruction

## Notes for Future Extensions

1. **Cross-validation**: Use hold-out test sets to evaluate decoder generalization
2. **Bayesian decoding**: Implement probabilistic position estimate with uncertainty quantification
3. **Sequence detection**: Identify replay of specific behavioral sequences or routes
4. **Theta precession**: Analyze phase precession of replayed spikes relative to ripple oscillations
5. **Multi-session analysis**: Pool replay events across multiple sessions for population statistics
6. **Real data validation**: Validate against actual DANDI recordings with ground truth position

## Author Note

This analysis was generated autonomously as a demonstration of hippocampal replay principles. The synthetic data mirrors real neural recording properties while ensuring complete reproducibility. For analysis of actual experimental data, direct access to DANDI datasets is recommended using the dandi Python client and streaming access via LINDI or remfile to avoid large downloads.
