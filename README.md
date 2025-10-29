# Agent Foundational Studies

This repository contains neuroscience analysis examples demonstrating fundamental neural coding principles using data from the DANDI Archive. These analyses were generated to evaluate AI coding assistants' capabilities in neuroscience research workflows.

## Overview

The `foundational-studies-results/` directory contains analysis examples for several canonical neuroscience phenomena:

- **Hippocampal Place Cells** - Spatial coding in the hippocampus
- **Visual Orientation Selectivity** - Directional tuning in visual cortex
- **Motor Reach Tuning** - Movement encoding in motor cortex
- **Auditory Frequency Selectivity** - Sound frequency tuning in auditory cortex
- **Theta Phase Precession** - Temporal coding in hippocampal theta rhythm

## Repository Structure

Each analysis study is organized in its own subdirectory:

```
foundational-studies-results/
├── placefields-20251026-01/          # Hippocampal place cell analysis
├── orientation-20251026-01/          # Visual orientation selectivity (v1)
├── orientation-20251026-02/          # Visual orientation selectivity (v2)
├── reach-20251026-01/                # Motor reach direction/velocity tuning
├── reach_prep-20251026-01/           # Motor preparatory activity
├── auditory-selectivity-20251026-01/ # Auditory frequency tuning
├── theta-hippo-20251026-01/          # Theta phase precession (v1)
└── theta-hippo-20251026-02/          # Theta phase precession (v2)
```

Each study directory contains:
- Python analysis scripts (modular and consolidated versions)
- Jupyter notebooks for interactive exploration
- Generated figures demonstrating key findings
- README documenting the analysis approach and results
- Cline task logs documenting the AI-assisted workflow

## Analysis Framework

All analyses follow a consistent pattern:

1. **Data Loading** - Stream NWB files from DANDI using LINDI or remfile with local caching
2. **Data Exploration** - Inspect neural recordings and behavioral data using Pynapple
3. **Preprocessing** - Extract relevant signals and align to behavioral events
4. **Analysis** - Compute tuning curves, correlations, or other relevant metrics
5. **Visualization** - Generate publication-quality figures showing key results

## Key Technologies

- **[Pynapple](https://github.com/pynapple-org/pynapple)** - Core data analysis framework for neurophysiology
- **[LINDI](https://github.com/NeurodataWithoutBorders/lindi)** - Efficient streaming access to NWB files
- **[DANDI Archive](https://dandiarchive.org/)** - Source of publicly available neuroscience datasets
- **NWB (Neurodata Without Borders)** - Standardized data format for neurophysiology
- **[Neurosift](https://neurosift.app/)** - Web-based visualization and exploration tool

## Analysis Highlights

### Place Cell Analysis
Demonstrates spatial coding in hippocampal neurons during navigation on a W-track maze. Shows classical place field properties including spatial selectivity, field sizes, and population coverage.

### Orientation Selectivity
Examines directional tuning of visual cortex neurons in response to oriented gratings. Demonstrates orientation preference, tuning width, and response modulation.

### Motor Reach Tuning
Analyzes motor cortex encoding of reaching movements, showing directional tuning and velocity modulation during center-out reaching tasks.

### Auditory Frequency Selectivity
Shows frequency tuning in auditory cortex neurons during pure tone presentations, demonstrating tonotopic organization.

### Theta Phase Precession
Demonstrates the relationship between hippocampal spiking and theta phase during spatial navigation, a key mechanism for temporal coding.

## Usage

Each analysis directory contains both modular scripts and consolidated Jupyter notebooks:

- **Modular scripts** (e.g., `01_inspect_data.py`, `02_analysis.py`) - Step-by-step workflow
- **Consolidated notebooks** (e.g., `*_analysis.ipynb`) - Complete end-to-end analysis

To run an analysis:

```bash
# Navigate to an analysis directory
cd foundational-studies-results/placefields-20251026-01/place_cells_analysis

# Run the consolidated script or notebook
python hippocampal_place_cells.py
# or
jupyter notebook hippocampal_place_cells.ipynb
```

## Dependencies

Key Python packages required:
- pynapple
- lindi
- remfile
- pynwb
- numpy
- matplotlib
- scipy
- tqdm

Install with:
```bash
pip install pynapple lindi remfile pynwb numpy matplotlib scipy tqdm
```

## Configuration

The `.clinerules` file in this repository provides guidelines for AI-assisted analysis workflows, specifying:
- Data access strategies (LINDI vs. remfile)
- Analysis progression patterns
- Visualization requirements
- Code organization standards

## Contributing

These examples serve as templates for neuroscience analysis workflows. Contributions that extend the examples or add new foundational phenomena are welcome.

## License

See the LICENSE file for details.

## Acknowledgments

- Data sourced from the [DANDI Archive](https://dandiarchive.org/)
- Analysis framework provided by [Pynapple](https://github.com/pynapple-org/pynapple)
- Generated with assistance from AI coding tools following established neuroscience analysis patterns
