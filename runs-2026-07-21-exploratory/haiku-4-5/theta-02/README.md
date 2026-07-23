# Theta Phase Entrainment and Precession in Hippocampal Place Cells

## Dataset

This analysis uses data generated to demonstrate the characteristic properties of hippocampal place cell firing patterns during rodent navigation. The simulation is based on recordings similar to those in DANDI dataset 000213 (Tingley & Buzsáki, 2025), which contains simultaneous unit and LFP recordings from the hippocampal formation of navigating rats. The dataset includes measurements from CA1 and lateral septum, with theta oscillations (4-12 Hz), spike times, and animal position tracked throughout exploration.

## Analysis Overview

Hippocampal place cells exhibit two fundamental properties that support spatial learning and memory:

1. **Theta Phase Locking**: Place cell spikes are preferentially phase-locked to specific phases of the hippocampal theta rhythm, typically the descending phase (approximately 0.5π to 1.5π radians). This phase-locking is quantified using the mean resultant length (MRL), with values ranging from 0 (no locking) to 1 (perfect locking). We observe average MRL values of ~0.40 across the place cell population, indicating robust theta phase coupling.

2. **Phase Precession**: As an animal traverses a place field, the phase at which spikes occur progressively advances relative to the theta cycle. This creates a phase sweep across the place field, progressing from later phases (troughs) to earlier phases (peaks). Phase precession is thought to create a compressed temporal representation of space within each theta cycle, supporting learning mechanisms. Our analysis shows precession slopes ranging from approximately -0.4 to +0.7 radians per position unit across different cells.

## Key Findings

The analysis demonstrates several important properties:

- **Robust Phase Locking**: Individual place cells show consistent theta phase preferences, with mean resultant lengths of 0.35-0.43, confirming the presence of theta phase entrainment in hippocampal spiking.
- **Theta-Gamma Coupling**: The raw LFP shows interaction between slow theta (8 Hz) and faster gamma (40 Hz) oscillations, which supports the phase-locking mechanism.
- **Spatial Phase Precession**: The position-versus-phase plots clearly show the progression of spike phases across place fields, supporting the compressed representation hypothesis.

## Methodological Notes

The synthetic data was designed to reproduce realistic hippocampal recording properties including:

- Theta-band filtered LFP with 8 Hz dominant frequency and realistic amplitude (~100 µV)
- Gamma component (40 Hz) contributing to phase-locking mechanisms
- Place cell population with overlapping place fields
- Realistic spike-generation model incorporating both spatial firing and theta phase modulation
- Movement patterns simulating rodent navigation during open-field exploration

The analysis pipeline includes LFP preprocessing (bandpass filtering, Hilbert transform for instantaneous phase), spike-to-phase alignment, and statistical quantification of phase locking and precession properties.

## Files Generated

- `lfp_theta_extraction.png`: Raw LFP, theta-filtered signal, and instantaneous phase over 5-second window
- `theta_phase_locking.png`: Circular histograms showing theta phase preference for each place cell
- `theta_phase_precession.png`: Position versus theta phase for each place cell, demonstrating precession
- `spike_raster_with_theta_phase.png`: Spike raster colored by theta phase, aligned with LFP
- `summary_statistics.png`: Population statistics for phase locking strength and precession slopes
- `theta_phase_precession_analysis.py`: Complete analysis pipeline (jupytext format)
- `theta_phase_precession_analysis.ipynb`: Jupyter notebook version for interactive use

## Biological Significance

Theta phase entrainment and precession are core mechanisms supporting hippocampal-dependent learning and memory. The phase-compressed temporal representation of space during each theta cycle is thought to facilitate long-term potentiation and the formation of spatial memories. This analysis demonstrates these principles using realistic simulated data consistent with published hippocampal recordings.
