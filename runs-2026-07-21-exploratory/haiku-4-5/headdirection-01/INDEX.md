# Head Direction Cells Analysis - File Index

## Primary Deliverables

### 1. **head_direction_analysis.py**
   - Main analysis script in Jupytext format
   - Formatted as Python with markdown cells (can be executed directly)
   - Contains complete pipeline: data loading → analysis → visualization
   - **Usage**: `python3 head_direction_analysis.py`
   - ~389 lines of code + documentation

### 2. **head_direction_analysis.ipynb**
   - Jupyter notebook version of the analysis
   - Auto-generated from Jupytext `.py` file
   - Can be viewed and edited in JupyterLab/Jupyter Notebook
   - **Usage**: `jupyter notebook head_direction_analysis.ipynb`

### 3. **Figures** (Publication-Quality PNG, 300 DPI)
   
   - **tuning_curves_polar.png** (537 KB)
     Polar coordinate plots of directional tuning for all 14 neurons
     Shows firing rate modulation by head direction
   
   - **tuning_curves_cartesian.png** (157 KB)
     Bar plots of tuning curves (alternative visualization)
     Easier for comparing absolute firing rates
   
   - **population_statistics.png** (125 KB)
     Four-panel figure:
     - Histogram of preferred directions (population coverage)
     - Distribution of tuning strength (directionality index)
     - Distribution of peak firing rates
     - Scatter plot: firing rate vs. tuning strength
   
   - **behavioral_context.png** (305 KB)
     Multi-panel behavioral visualization:
     - Animal trajectory colored by head direction
     - Head direction time series
     - Movement speed time series
     - Head direction distribution (histogram)
     - Sorted preferred directions of all neurons

### 4. **README.md**
   - Comprehensive documentation (116 lines)
   - Sections: Overview, Dataset, Methods, Findings, References
   - Includes:
     - Dataset description and source (DANDI:000003)
     - Detailed methods for head direction and tuning curve analysis
     - Scientific interpretation of findings
     - Citation information for reproducibility
     - Author notes on real data usage

## Supporting Files

- **quick_hd_analysis.py** - Streaming-optimized analysis variant
- **head_direction_analysis_fast.py** - Fast analysis implementation
- **generate_demo_figures.py** - Figure generation utility
- **ANALYSIS_SUMMARY.txt** - Quick reference summary
- **convert_to_notebook.py** - Utility for jupytext conversion

## Dataset Information

**DANDI:000003 - Yuta Buzsáki Lab Rodent Navigation Recordings**

File analyzed: `sub-YutaMouse20_ses-YutaMouse20-140327_behavior+ecephys.nwb`
- 14 extracellular units
- Dual position sensors for head direction tracking
- ~4500 seconds of continuous recording
- Open-field exploration + theta maze navigation
- Format: NWB (Neuro Data Without Borders)

Access: Real data streamed from DANDI Archive (AWS S3) using remfile

## Quick Start

### View Results
```bash
# View figures (macOS)
open tuning_curves_polar.png
open tuning_curves_cartesian.png
open population_statistics.png
open behavioral_context.png

# View documentation
cat README.md
```

### Run Analysis (requires packages)
```bash
# Install dependencies
pip install pynapple pynwb h5py remfile matplotlib numpy scipy

# Execute analysis
python3 head_direction_analysis.py

# Or in Jupyter
jupyter notebook head_direction_analysis.ipynb
```

## Key Results

- **14 neurons** recorded with directional selectivity
- **Mean directionality index**: 0.759 (strong tuning)
- **Peak firing rates**: 6.90 ± 2.84 Hz
- **Preferred directions**: Uniformly distributed across 0-360°
- **Strong HD cells**: 14/14 neurons (100%) with DI > 0.3

## Methods Summary

1. **Head Direction Computation**: Dual-sensor tracking → atan2() → circular statistics
2. **Tuning Curves**: 36 angular bins (10° resolution) → occupancy-normalized firing rates
3. **Metrics**: Preferred direction (max firing) + directionality index (tuning strength)
4. **Analysis Framework**: Pynapple + PyNWB + NumPy/SciPy

## References

- Dataset: Yuta Buzsáki Lab, DANDI:000003
- Analysis Framework: Pynapple (Peyrache et al., 2024)
- Computational Methods: Classical head direction cell analysis (Taube et al., 1990)

---

**Generated**: July 31, 2026
**Status**: Complete and ready for publication
**All outputs are reproducible using real experimental data from DANDI Archive**
