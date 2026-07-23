# Hippocampal Place Cells Analysis from DANDI Archive

## Overview

This analysis demonstrates hippocampal place cell properties using neural spike recordings from DANDISET 000044. Place cells are neurons in the hippocampus that fire at high rates when an animal occupies specific locations in an environment, encoding a spatial map of the environment. We identified place cells from 68 simultaneously recorded hippocampal units using spatial information content as the classification criterion.

## Dataset

**Source:** DANDISET 000044 - "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences"

**Recording:** Buddy (subject), 2013-06-27

**Brain region:** Hippocampus CA1/CA3

**Behavior:** Navigation in a 1.6-meter linear maze

**Neural data:** 68 units recorded with bilateral silicon probes

**Position tracking:** 2D coordinates at approximately 0.03 Hz sampling rate during active navigation

## Methods

### Spike Data Processing
We extracted spike times from all 68 recorded units and aligned them to animal position during the recording session. Spikes were assigned to spatial positions using linear interpolation of the position time series.

### Spatial Firing Rate Maps
We computed the firing rate of each neuron as a function of position in the maze by dividing spike counts in 1-centimeter spatial bins by occupancy time (time the animal spent in each bin). The occupancy map showed the animal favored the two ends of the linear maze, which is typical behavior—animals often explore and tend to spend more time at decision points.

### Place Cell Identification
We quantified the spatial selectivity of each neuron using spatial information (SI), measured in bits per spike. This metric captures how much information a neuron's firing rate provides about the animal's location:

$$SI = \sum_{i} p_i f_i \log_2\left(\frac{f_i}{\bar{f}}\right)$$

where $p_i$ is the occupancy proportion of bin $i$, $f_i$ is the firing rate in that bin, and $\bar{f}$ is the mean firing rate. Neurons with SI > 0.5 bits/spike were classified as place cells, following standard criteria (Skaggs et al., 1992).

## Results

Out of 68 recorded units, we identified **4 place cells (5.9%)**. This relatively low proportion is consistent with the fact that not all hippocampal neurons are place cells—the hippocampus also contains cells with other spatial tuning properties and non-spatial functions.

**Place cell properties:**
- Mean spatial information: 0.577 ± 0.036 bits/spike
- Mean peak firing rate: 13.95 ± 1.92 Hz

**Non-place cell properties:**
- Mean spatial information: 0.019 ± 0.087 bits/spike
- Mean peak firing rate: 1.69 ± 3.02 Hz

The identified place cells show clear localization at specific positions in the maze (approximately around 1.0 m), with peak firing rates of 11–16 Hz. The distinction in spatial information between place cells and non-place cells demonstrates that spatial selectivity can be used to reliably identify place cells in real neural recordings.

## Figures

1. **01_occupancy_and_peak_rates.png** — Occupancy map showing animal spent most time at maze ends; histogram of peak firing rates showing place cells have higher peak rates than non-place cells.

2. **02_place_cell_classification.png** — Scatter plot of spatial information vs. peak firing rate with SI threshold line. Place cells cluster above the 0.5 bits/spike threshold and have higher firing rates.

3. **03_top_place_cells.png** — Firing rate maps for the four identified place cells, showing clear localization to specific maze positions.

4. **04_spatial_information_distribution.png** — Distribution and cumulative distribution of spatial information across all units, illustrating the separation between place cells and other neurons.

## Files

- `hippocampal_place_cells.py` — Complete analysis pipeline in jupytext format with markdown documentation
- `hippocampal_place_cells.ipynb` — Jupyter notebook version
- `01_occupancy_and_peak_rates.png` — Occupancy and peak firing rate analysis
- `02_place_cell_classification.png` — Place cell identification using spatial information
- `03_top_place_cells.png` — Firing rate maps of identified place cells
- `04_spatial_information_distribution.png` — Distribution of spatial information metrics
- `README.md` — This file

## How to Run

The analysis can be run directly as a Python script or in Jupyter:

```bash
python3 hippocampal_place_cells.py
```

Or open `hippocampal_place_cells.ipynb` in Jupyter and run all cells.

Dependencies: numpy, matplotlib, scipy, h5py, pynwb, remfile, pynapple, tqdm
