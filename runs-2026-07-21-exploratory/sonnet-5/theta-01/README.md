# Theta Phase Entrainment and Precession in Hippocampal Place Cells

## Dataset

[DANDI 000044](https://dandiarchive.org/dandiset/000044), "Diversity in neural
firing dynamics supports both rigid and learned hippocampal sequences"
(Grosmark & Buzsaki, NYU). We used subject **Achilles**, session
`Achilles_10252013`: bilateral CA1 silicon-probe recordings (137 units, 120
putative pyramidal cells and 17 interneurons across left and right CA1) while
the rat shuttled back and forth on a 1.6 m linear track for reward at both
ends. The NWB file was streamed directly from S3 with `remfile` (disk-cached
locally, no full download), and Pynapple was used for all time-series handling
and analysis.

## Analysis

The LFP from a CA1 pyramidal-layer channel was band-pass filtered at 6-10 Hz
and the Hilbert transform used to extract instantaneous theta phase. Running
epochs were split by direction from the linearized position, and place fields
for rightward runs were computed for all pyramidal cells, screened for
well-formed single fields (adequate spatial information, field width, and
spike count) after finding that naive occupancy-normalized rate estimates at
the very ends of the track were badly inflated by brief immobility-related
population bursts (most likely sharp-wave-ripple activity at the reward
ports) — this was fixed by requiring genuine track-spanning runs and
restricting place-field analysis to the interior of the track.

Two classic theta-related phenomena were then tested:

1. **Phase entrainment**: a Rayleigh test on spike theta-phase for every
   pyramidal cell during active running.
2. **Phase precession**: for each screened place cell, a circular-linear
   regression (Kempter et al., 2012) of theta phase against normalized
   within-field position, computed per lap and also pooled across cells.

## Key Findings

About 55% of the 120 pyramidal cells tested show significant theta phase
locking (Rayleigh p < 0.001), with mean resultant lengths mostly in the
0.1-0.5 range, consistent with the well-established theta entrainment of CA1
pyramidal firing. For phase precession, individual place cells showed
variable and often statistically weak single-cell relationships (a known
feature of real data with limited per-cell sampling — tens to a few hundred
field-crossing spikes over 20-40 laps), but pooling normalized field position
and spike phase across the 14 screened place cells revealed a clear, highly
significant negative relationship (slope ≈ -0.56 cycles per field, circular-
linear permutation p ≈ 0.0005): theta phase advances to earlier values as the
rat crosses through a cell's place field, reproducing the classic phase
precession phenomenon described by O'Keefe & Recce (1993).

## Files

- `theta_phase_precession_analysis.py` — consolidated jupytext analysis script (runs end-to-end)
- `theta_phase_precession_analysis.ipynb` — same analysis as an executed Jupyter notebook
- `circstats.py` — circular statistics helpers (Rayleigh test, circular-linear regression) used by the main script
- `fig1_raw_lfp_theta_filter.png` — raw vs. theta-band-filtered LFP
- `fig2_position_and_running.png` — linearized position/velocity with direction-classified running epochs
- `fig3_place_fields.png` — place fields of screened pyramidal cells
- `fig4_theta_entrainment.png` — example polar phase histograms and population summary of theta phase locking
- `fig5_precession_examples.png` — single-cell phase precession scatter plots with circular-linear fits
- `fig6_precession_population.png` — pooled population phase precession across screened place cells
