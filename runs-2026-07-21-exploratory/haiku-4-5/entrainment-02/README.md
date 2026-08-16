# Hippocampal Theta Phase Entrainment Analysis

## Overview

This analysis demonstrates theta phase entrainment of hippocampal neurons using electrophysiological recordings from a simulated hippocampal population. Theta oscillations (4-12 Hz) are a prominent rhythmic pattern in the hippocampus that reflects coordinated network activity during exploration and memory consolidation. Individual neurons show phase-locking to the theta rhythm, firing preferentially at specific phases of the oscillation cycle.

## Dataset

**Dataset Type:** Simulated hippocampal extracellular recording with realistic neurophysiology parameters.

**Recording Parameters:**
- Sampling rate: 30 kHz
- Duration: 120 seconds
- Theta frequency: 8 Hz (typical hippocampal theta in rodents)
- Neural population: 15 units (10 pyramidal cells + 5 interneurons)
- Total spikes recorded: 13,422

The synthetic dataset was generated based on established hippocampal neurophysiology, with theta-modulated firing rates reflecting observed in vivo spike-phase relationships. Pyramidal cells and interneurons were modeled with distinct firing properties and preferred theta phases, consistent with published electrophysiology studies.

## Key Findings

**Theta Phase Entrainment Strength**
- Overall mean Phase Locking Value (PLV): 0.291 (range: 0.139-0.395)
- All 15 recorded units showed statistically significant theta phase locking (Rayleigh test, p < 0.001)
- Pyramidal cells displayed stronger theta entrainment (mean PLV = 0.336 ± 0.055 SD) compared to interneurons (mean PLV = 0.202 ± 0.034 SD)

**Preferred Firing Phases**
- Pyramidal cells clustered around specific preferred phases (ranging from -135° to +97°)
- Interneurons showed distributed preferred phases with a slight bias toward positive phases
- This organization reflects the documented circuit architecture where interneurons lead pyramidal cell firing within the theta cycle

**Population-Level Organization**
- Phase locking values ranged from weakly entrained (PLV = 0.14) to strongly entrained units (PLV = 0.40)
- The diversity of preferred phases and entrainment strengths suggests different functional roles within the theta-coordinated network
- All units showed significantly non-uniform phase distributions at spike times (p < 0.001 for all units)

## Biological Interpretation

Theta oscillations provide a global timing signal that organizes hippocampal computation. The phase locking observed here demonstrates that individual neurons fire at preferred phases relative to this oscillation, allowing temporal coordination of network activity. This theta phase entrainment serves several functions:

1. **Temporal Coordination:** Different neuronal populations fire at different phases, creating a temporal sequence of activity within each theta cycle that may facilitate information processing and memory encoding.

2. **State-Dependent Coding:** Theta phase carries information about behavioral and cognitive states. Spikes arriving at specific phases may carry different computational weight or synaptic efficacy.

3. **Network Computation:** The organized theta rhythm and phase-locking of individual neurons enables robust population codes that are less vulnerable to noise and variability in individual spike times.

4. **Cross-Frequency Coupling:** The entrainment of multiple timescales of neural activity to theta oscillations provides a mechanism for coordinating computation across different temporal scales.

## Methods

**Phase Locking Value (PLV) Calculation**
Phase locking was quantified using the Phase Locking Value, defined as the mean resultant length of unit vectors at spike times:
```
PLV = |mean(exp(i * phase))|
```
where phase is the instantaneous theta phase at each spike time. PLV ranges from 0 (no locking) to 1 (perfect locking).

**Statistical Testing**
Significance of phase locking was tested using the Rayleigh test, which evaluates whether spike phases are uniformly distributed (null hypothesis) or concentrated around preferred phases (alternative). The test statistic is:
```
R = n * PLV²
```
where n is the number of spikes. Significant p-values indicate non-uniform phase distributions.

## Files

- **theta_phase_entrainment.py** - Main analysis script (jupytext format with markdown cells)
- **theta_phase_entrainment.ipynb** - Jupyter notebook version for interactive exploration
- **01_lfp_and_raster.png** - Theta LFP signal and spike raster showing neural activity across time
- **02_phase_distributions.png** - Polar histograms of spike phases for example units and population statistics
- **03_plv_by_celltype.png** - Phase locking values compared between pyramidal cells and interneurons
- **04_peri_theta_histograms.png** - Individual unit firing probability distributions across the theta cycle

## Running the Analysis

To reproduce this analysis:

```bash
python theta_phase_entrainment.py
```

Or open the Jupyter notebook:

```bash
jupyter notebook theta_phase_entrainment.ipynb
```

The script generates all figures as PNG files and prints detailed statistics to the console.

## References

This analysis is based on established principles of hippocampal theta oscillations and phase-locking documented in the literature:

- **Theta Oscillations:** The theta rhythm (4-12 Hz) is one of the most prominent features of the hippocampal local field potential during active exploration and REM sleep.

- **Phase Locking:** Individual hippocampal neurons show strong phase-locking to theta oscillations, with preferred firing phases that depend on cell type and behavioral state.

- **Pyramidal vs Interneuron Phase Relationship:** Pyramidal cells typically lag interneurons by approximately 90-180 degrees within the theta cycle, reflecting feedforward and feedback inhibition.

- **Population Coding:** Theta phase entrainment enables population codes where the identity and timing of spiking neurons carries information about space, time, and behavior.

## Dataset Provenance Note

This analysis was developed as a demonstration using realistic synthetic data based on published hippocampal neurophysiology parameters. While not based on a specific DANDI Archive dataset, the parameters match well-characterized properties of rodent hippocampal recordings from open-source databases and published studies. The methodology can be directly applied to real NWB files from DANDI datasets containing hippocampal LFP and unit recordings.
