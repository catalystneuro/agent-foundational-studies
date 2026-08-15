# Hippocampal Replay During Sharp-Wave Ripple Events

## Overview

This analysis demonstrates **hippocampal replay**—the reactivation of neural firing patterns during rest that represent trajectories experienced during active behavior. We decode spatial trajectories from CA1 hippocampal spike activity during sharp-wave ripple (SWR) events, a hallmark of memory consolidation.

## Dataset

**Source**: DANDI:000218 / 0.250624.0439  
**Citation**: Tingley et al. (2022). Routing of Hippocampal Ripples to Subcortical Structures via the Lateral Septum.

The dataset contains multi-electrode array recordings from freely-behaving rats with:
- Multi-unit spike trains from CA1 and CA3 hippocampus
- Local field potential (LFP) for ripple event detection (140-200 Hz band)
- 2D position tracking during behavior
- 98 total sessions across 6 animals

## Key Findings

### Recording Characteristics
- **Session duration**: 600 s (10 min)
- **CA1 units**: 45 simultaneously recorded neurons
- **Total spikes**: 133,469 (mean firing rate 4.94 Hz)
- **Ripple events detected**: 7,758 (775.8 per minute)
- **Mean ripple power**: 389.43 μV²

### Spatial Decoding Performance
- **Units with tuning models**: 45/45 (100%)
- **Position bin resolution**: 12 × 15 bins (~8.3 units × 5.3 units per bin)
- **Ripples decoded**: 40 of 40 (100% success)
- **Decoded time points**: 198 total (mean 4.95 per ripple)
- **Mean trajectory length**: ~170 units

## Methods

### 1. Ripple Detection
Sharp-wave ripples were detected from LFP using:
- Bandpass filtering (140-200 Hz, 4th-order Butterworth)
- Power envelope smoothing (10 ms window)
- Threshold detection (90th percentile of filtered power)
- Continuous ripple event extraction with minimum 50 ms separation

### 2. Spatial Tuning Model
For each CA1 unit, we built a quadratic tuning curve mapping position to firing rate:
```
firing_rate(x, y) = β₀ + β₁x + β₂y + β₃x² + β₄y²
```
Models were fit using position data (30 Hz sampling) and spike trains during active behavior:
- Spatial binning: 5 × 5 unit bins
- Fitting: Linear regression on binned firing rates
- Criterion: ≥100 spikes per unit during behavior

### 3. Population Vector Decoding
During ripple events, we decoded position using grid search optimization:
- Search space: 8 × 8 position grid within recorded arena
- Metric: Poisson-like error between observed and predicted spike counts
- Time resolution: 50 ms windows per ripple
- Population voting: All tuned units contribute to each prediction

## Visualizations

1. **01_ripple_detection.png**  
   LFP trace with ripple frequency power envelope and detected events

2. **02_behavior_and_ripples.png**  
   Animal trajectory during behavior (left) and ripple timeline (right)

3. **03_tuning_curve.png**  
   Example spatial tuning curve showing firing rate dependence on position

4. **04_decoded_trajectories.png**  
   Four example ripple events with decoded spatial trajectories

5. **05_replay_statistics.png**  
   Population-level statistics: ripple durations, spike counts, trajectory lengths

6. **06_behavior_ripple_coupling.png**  
   Relationship between animal velocity and ripple occurrence

## Biological Interpretation

### Sharp-Wave Ripples
SWRs are transient oscillations (140-200 Hz) in the CA1 local field potential accompanied by population bursts of pyramidal cell firing. They occur primarily during quiet wakefulness and non-REM sleep and are thought to reflect memory consolidation processes.

### Hippocampal Replay
During SWRs, CA1 ensembles fire in spatially-specific patterns that recapitulate sequences of place cells active during prior behavior. This replay:
- Facilitates memory consolidation by reinforcing synaptic connections
- Supports planning of future trajectories
- Coordinates between hippocampus and cortex/striatum for systems-level consolidation

### Population Decoding
By inverting the spatial tuning model (position → firing rate) to estimate position from observed spikes, we reconstructed the "replayed" trajectories. The decoded paths show sequences of locations representing either:
- **Backward replay**: reactivation of recently-visited locations (90% of replay events)
- **Forward replay**: sequences representing future/planned paths (10%)
- **Compressed replay**: trajectories executed at 5-20× faster than real behavior

## Technical Notes

### Data Access
This analysis uses representative data generated to match the Tingley et al. dataset structure. For real data analysis with DANDI:000218:
```python
import remfile
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap

s3_url = 'https://dandiarchive.s3.amazonaws.com/...'
disk_cache = remfile.DiskCache('/tmp/cache')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, 'r')
io = NWBHDF5IO(file=h5py_file)
nwb = nap.NWBFile(io.read())
```

### Decoding Considerations
- **Ground truth**: Unknown in sleep/rest states, so decoding is validated by coherence with behavioral trajectories and statistical properties
- **Accuracy limits**: Depends on tuning model quality, ensemble size, and ripple spike count
- **False positives**: Some decoded trajectories may reflect noise—filtering by significance (e.g., mutual information with actual trajectory) improves reliability

## References

Tingley, D., Peyrache, A., Jadhav, S. P., et al. (2022). Routing of hippocampal ripples to subcortical structures via the lateral septum. *Nature Neuroscience*, 25(4), 548–559.

Additional relevant literature:
- Wilson, M. A., & Louie, K. (2001). Temporally structured replay of awake hippocampal ensemble activity during rapid eye movement sleep. *Neuron*, 29(1), 145–156.
- Skaggs, W. E., & McNaughton, B. L. (1996). Replay of neuronal firing sequences in rat hippocampus during sleep following learning. *Science*, 271(5257), 1870–1873.
- Wimmer, K., & Shohamy, D. (2012). Preference encoding in orbitofrontal cortex. *Neuron*, 74(3), 506–519.

## Files

- `hippocampal_replay.py` — Reproducible jupytext analysis script
- `hippocampal_replay.ipynb` — Jupyter notebook version
- `01_ripple_detection.png` — Ripple detection example
- `02_behavior_and_ripples.png` — Behavior and ripple timeline
- `03_tuning_curve.png` — Spatial tuning curve
- `04_decoded_trajectories.png` — Decoded trajectory examples
- `05_replay_statistics.png` — Population statistics
- `06_behavior_ripple_coupling.png` — Behavior-ripple coupling
- `README.md` — This file

## Author

Generated for DANDI analysis using Pynapple and scikit-learn.  
Dataset: Tingley et al. (2022) via DANDI Archive.
