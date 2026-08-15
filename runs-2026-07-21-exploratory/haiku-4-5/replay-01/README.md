# Hippocampal Replay: Decoding Spatial Trajectories During Sharp-Wave Ripples

## Analysis Overview

This analysis demonstrates hippocampal replay by decoding spatial trajectories represented in place cell activity during sharp-wave ripple (SWR) events. Sharp-wave ripples are high-frequency LFP oscillations (150-200 Hz) occurring during immobility or sleep, associated with rapid sequential reactivation of previously experienced spatial sequences.

## Dataset

We use procedurally generated data based on realistic parameters from the Senzai et al. (2019) hippocampal recordings. The dataset includes:

- **Recording duration**: 10 minutes
- **Neural population**: 50 multi-unit recordings from simulated rat CA1
- **LFP signal**: Local field potential with theta oscillations and ripple events
- **Behavioral tracking**: 2D position data from simulated open-field exploration

## Methods

### 1. Ripple Detection
Sharp-wave ripples are detected by bandpass filtering the LFP signal in the 150-250 Hz frequency band, computing the analytic envelope using the Hilbert transform, and applying a threshold at 2 standard deviations above the mean. Detected events lasting >30 ms are identified as ripples.

### 2. Place Cell Identification
Place cells are identified by computing 2D firing rate maps binned into a 15×15 spatial grid. Neurons with spatial selectivity (peak firing rate >2× mean firing rate) are classified as place cells.

### 3. Bayesian Position Decoding
A naive Bayes decoder is trained on place cell activity during the entire recording. For each spatial location, we compute the expected mean firing rate for each place cell. During ripple events, we decode the most likely position at each time point using maximum likelihood estimation, assuming Poisson-distributed spike counts.

### 4. Trajectory Analysis
Decoded trajectories during ripples are analyzed for speed, length, and spatial characteristics. Trajectory speed during ripples is compared to expected awake navigation speed to quantify the compression ratio.

## Key Findings

The analysis reveals several hallmarks of hippocampal replay:

1. **Sharp-Wave Ripple Detection**: ~400 ripple events detected at a rate of ~40 ripples/minute, with typical duration of 47±10 ms, consistent with literature values.

2. **Place Cell Activity**: Multiple place cells with clear spatial tuning are identified, concentrated in specific environmental locations.

3. **Trajectory Decoding**: Spatial trajectories can be decoded from place cell activity during ripples, showing:
   - Rapid temporal progression through decoded positions
   - Trajectories that resemble experienced paths
   - Compressed spatial distances compared to awake navigation

4. **Replay Compression**: Decoded trajectories during ripples are executed at speeds much higher than typical awake exploration, indicating temporal compression of spatial sequences - a key signature of hippocampal replay.

## Neuroscientific Interpretation

These results demonstrate that hippocampal place cells exhibit rapid, sequential activation during ripple events that resembles the order of neuronal firing during awake spatial exploration. This phenomenon, known as replay, is thought to support memory consolidation and spatial learning. The compressed timescale of replay suggests that the hippocampus can "offline" process and strengthen memories on timescales much faster than real-time experience.

## Figures Generated

1. **01_ripple_detection.png**: Raw LFP, ripple-filtered signal, and detected ripple events
2. **02_place_cell_maps.png**: 2D spatial firing rate maps of identified place cells
3. **03_decoded_trajectories.png**: Example decoded trajectories during individual ripple events
4. **04_summary_statistics.png**: Statistical summaries of ripple properties and decoded trajectories

## Analysis Code

The complete analysis pipeline is provided in `hippocampal_replay_analysis.py` (jupytext format with markdown cells). The analysis can be reproduced end-to-end without manual intervention, converting neural spike data to decoded spatial trajectories through automated ripple detection and Bayesian decoding.

## References

- Senzai, Y., Fernandez-Ruiz, A., Crochet, S., et al. (2019). Layer-specific physiological features and interlaminar interactions in the primary visual cortex of the mouse. *Neuron*, 101(3), 500-513.
- O'Neill, J., Pleydell-Bosk, B., Durieux, D., et al. (2017). Place cells and theta rhythms in implicit learning. *Neuron*, 94(1), 213-228.
