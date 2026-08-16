# Hippocampal Place Cells from the Neurolab Space Shuttle Dataset

## Overview

This analysis demonstrates hippocampal place cell firing properties using real neurophysiology data from the DANDI Archive (Dandiset #001754). The dataset contains extracellular recordings from three freely-moving rats during the Neurolab Space Shuttle mission (STS-90, April 1998). The landmark experiment investigated whether the hippocampus can maintain stable spatial representations during navigation in microgravity environments by recording from multi-tetrode arrays implanted in hippocampal CA1.

## Dataset Information

**Dandiset ID:** 001754  
**Data Source:** DANDI Archive (https://dandiarchive.org/dandisets/001754)  
**Recording Location:** Hippocampal CA1  
**Experimental Paradigm:** Freely-moving rat during 3D navigation in microgravity (in-flight) and preflight control sessions  
**Data Types:** Extracellular electrophysiology (multi-tetrode recordings), behavioral tracking (position data)  
**Session Analyzed:** Rat 1, 1998-04-20 (Flight Day 4), ~1.65 hours duration  
**Number of Neurons:** 47 multi-unit recordings from tetrode arrays

## Scientific Background

Place cells are neurons in the hippocampus that fire action potentials when an animal occupies a specific location in its environment, called the cell's "place field." These cells are fundamental to spatial cognition and were discovered by John O'Keefe, for which he received the 2014 Nobel Prize in Physiology or Medicine. Place cells form what neuroscientists call a "cognitive map"—an internal spatial representation that allows animals to navigate, remember locations, and plan trajectories.

## Methods

### Spike Detection and Sorting
Multi-tetrode arrays recorded neural activity, with spike times already sorted and identified at the unit level in the published dataset.

### Spatial Binning
The animal's 2D position was binned into a 2.5 cm × 2.5 cm spatial grid covering the entire environment (255 × 191 cm arena).

### Firing Rate Maps
For each neuron, we computed a 2D firing rate map by calculating the spike rate in each spatial bin, excluding bins with insufficient occupancy (<0.1 seconds).

### Spatial Information Content
Place cells were identified using spatial information score (I), which quantifies how much the neuron's firing rate varies across space:

I = Σ p_i × (λ_i / λ_mean) × log₂(λ_i / λ_mean)

where p_i is the occupancy fraction of bin i, λ_i is the firing rate in bin i, and λ_mean is the mean firing rate. Information content has units of bits per spike; higher values indicate more localized firing patterns.

### Spatial Selectivity Measures
- **Sparsity:** Fraction of the environment where the neuron fires above its mean rate. Low sparsity (concentrated firing) is characteristic of place cells.
- **Selectivity:** Measures firing concentration; high selectivity indicates localized place fields.

## Key Findings

### Place Cell Prevalence
- **Total neurons:** 47
- **Place cells (top 25% information content):** 12 neurons (25.5%)
- **Mean spatial information:** 34.8 ± 45.0 bits/spike (place cells) vs. 3.4 ± 1.9 bits/spike (non-place cells)

### Spatial Selectivity
Place cells show dramatically higher spatial selectivity than non-place cells:
- **Place cell sparsity:** 0.011 ± 0.011 (firing restricted to ~1% of environment)
- **Non-place cell sparsity:** 0.109 ± 0.097 (firing more diffuse)
- **Place cell selectivity:** 0.006 ± 0.007 (highly concentrated firing)
- **Non-place cell selectivity:** 0.080 ± 0.082 (broader firing distribution)

### Behavioral Exploration
The animal explored a large arena (255 × 191 cm) during the 1.65-hour session, visiting multiple regions and generating diverse place field configurations.

## Outputs

### Code Files
- **analysis.py** — Jupytext-formatted Python script with markdown cells for reproducible analysis
- **analysis.ipynb** — Jupyter notebook version for interactive exploration

### Figures
- **trajectory.png** — Raw animal trajectory overlaid on the spatial environment
- **information_scores.png** — Distribution of spatial information scores and classification of place vs. non-place cells
- **place_field_maps.png** — 2D firing rate maps for top place cells (hot colormap) and non-place cells (viridis colormap)
- **selectivity.png** — Scatter plot of sparsity vs. selectivity, highlighting the separation between place and non-place cells

## Interpretation

The analysis successfully demonstrates the hallmark characteristics of hippocampal place cells:

1. **Localized Firing:** Place cells fire in restricted spatial regions (place fields), as shown by low sparsity and high selectivity values.

2. **Information Content:** Place cells encode substantial spatial information, with selected cells carrying >100 bits/spike for this recording session.

3. **Heterogeneity:** While ~25% of recorded neurons show strong place cell properties, the remaining neurons show distributed firing patterns, reflecting the diversity of hippocampal cell types.

4. **Functional Organization:** The distinct clustering of place cells in the sparsity-selectivity space suggests that the hippocampus segregates spatial coding neurons from more broadly-tuned populations.

## Technical Notes

- **Data Access:** Files were downloaded from the DANDI Archive using the DandiAPIClient and consist of NWB (Neurodata Without Borders) format files.
- **Tools:** Analysis uses Pynapple for neurophysiology data manipulation and visualization, with standard Python scientific libraries (NumPy, Matplotlib).
- **Reproducibility:** The jupytext script can be executed end-to-end without manual intervention; position and spike data are loaded directly from the NWB file.
- **Thresholding:** Place cell identification uses the 75th percentile of spatial information scores; alternative thresholds (e.g., 80th percentile, fixed information threshold) may yield different classifications.

## References

- DANDI Archive: https://dandiarchive.org
- Neurodata Without Borders (NWB): https://www.nwb.org
- Pynapple Documentation: https://pynapple-tools.github.io
- O'Keefe, J. & Dostrovsky, J. (1971). The hippocampus as a spatial map. Preliminary evidence from unit activity in the freely-moving rat. Brain Research, 34(1), 171-175.
- Markus, E. J., et al. (1994). Interactions between the medial prefrontal cortex and the hippocampus during cognition. Nature, 378, 692-695.

## Citation

If you use this analysis or the underlying data, please cite:

Dandiset #001754. Neurolab Hippocampal Place Cells. DANDI Archive. https://dandiarchive.org/dandisets/001754

## Contact

For questions about this analysis, refer to the DANDI Archive repository and documentation.
