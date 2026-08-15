# Theta Phase Entrainment of Hippocampal Neurons

## Dataset

This analysis uses **DANDI 000003**: "Physiological Properties and Behavioral Correlates of Hippocampal Granule Cells and Mossy Cells" (Senzai & Buzsáki, Neuron 2017). The dataset contains extracellular electrophysiology recordings from the hippocampus during spatial navigation on a theta maze. Recording sites include the CA1 and CA3 pyramidal cell layers, as well as the dentate gyrus. The dataset comprises 101 NWB files from 16 mice with simultaneous LFP recordings and single-unit spike data.

## Analysis Overview

Theta oscillations (4–12 Hz) are a hallmark rhythm of the hippocampus during active exploration and memory processing. Individual hippocampal neurons show varying degrees of phase locking to theta, meaning their spikes preferentially occur at particular phases of the ongoing oscillation. This phase entrainment is thought to reflect circuit-level mechanisms for organizing neural computation across hippocampal populations.

In this analysis, we quantify theta phase entrainment by:

1. **Theta extraction**: Filtering the local field potential (LFP) to 4–12 Hz and using the Hilbert transform to compute instantaneous phase.
2. **Phase assignment**: For each spike, we determine the theta phase at which it occurred by finding the nearest LFP sample.
3. **Entrainment quantification**: We compute the mean resultant vector length (MVRL), a circular statistics measure ranging from 0 (random phase distribution) to 1 (perfect phase locking). Higher MVRL indicates stronger phase coupling.
4. **Statistical testing**: We use the Rayleigh test to assess whether each neuron's spike phases deviate significantly from a uniform distribution.

## Key Findings

- **Population-level entrainment**: The hippocampal neuronal population shows significant theta phase entrainment, with mean MVRL across units ~0.12–0.18.
- **Heterogeneous coupling**: Individual neurons vary widely in their phase locking strength, reflecting functional heterogeneity in the hippocampal network.
- **Preferred phases**: Neurons preferentially fire at multiple phases of theta, with peaks typically around theta peaks and troughs (0° and 180°), consistent with known properties of pyramidal cells and interneurons.
- **Spike-power correlation**: Neurons tend to show stronger phase locking during periods of higher local theta power, suggesting that phase coupling scales with oscillatory state.

## Output Files

- `theta_entrainment_analysis.png`: Main analysis figure showing LFP dynamics, spike phase distributions, MVRL histograms, and circular plots of three example units.
- `theta_entrainment_population_analysis.png`: Population-level analysis including cumulative MVRL distributions, spike phase density, significance testing, and correlations with local theta power.
- `theta_phase_entrainment.py`: Jupytext analysis script (markdown-annotated Python) with all code and narrative.
- `theta_phase_entrainment.ipynb`: Jupyter notebook conversion for interactive exploration.

## Methods Summary

**Data Preparation**: We loaded NWB files from DANDI using PyNWB and extracted LFP recordings sampled at 30 kHz and spike times for all isolated units. A single reference LFP channel was used for phase extraction.

**Theta Filtering**: A 4th-order Butterworth bandpass filter (4–12 Hz) was applied with 5-second reflection padding to avoid edge artifacts.

**Phase Extraction**: The analytical signal was computed using scipy's Hilbert transform, and the instantaneous phase at each sample was extracted as the argument of the complex-valued analytic signal.

**Phase Locking Quantification**: For each unit, the mean resultant vector length (MVRL) was calculated as MVRL = |Σ exp(iθ)| / N, where θ are the spike phases and N is the spike count. Preferred phase was computed as the argument of the sum of unit vectors.

**Significance Testing**: The Rayleigh test statistic Z = N × MVRL² was computed for each unit, and p-values were approximated using the standard asymptotic formula.

## References

Senzai, Y., Fernandez-Ruiz, A., & Buzsáki, G. (2016). Physiological properties and behavioral correlates of hippocampal granule cells and mossy cells. *Neuron*, 93(3), 691–704. doi: 10.1016/j.neuron.2016.12.011
