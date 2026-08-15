# Theta Phase Precession in Hippocampal Place Cells

## Dataset

[DANDI:000059](https://dandiarchive.org/dandiset/000059), "Cooling of Medial Septum Reveals Theta
Phase Lag Coordination of Hippocampal Cell Assemblies" (Petersen & Buzsaki, *Neuron* 2020, Buzsaki
lab). Rats ran laps on a peanut-shaped (figure-8-like) maze while extracellular activity was
recorded from dorsal CA1 with silicon probes, alongside a dedicated theta-reference LFP channel and
tracked (x, y) position. We used session `sub-MS10/Peter-MS10-170317-153237`, restricted to the
pre-cooling baseline epoch (before septal cooling perturbed theta), and to the 'Right'-condition
trials, which are near-complete laps around one loop of the maze. Data were streamed directly from
S3 with `remfile` (no full download): units, position, and trial metadata came from the processed
`behavior+ecephys` NWB file, and the wideband LFP for the theta-reference electrode was streamed in
20-second blocks from the raw `ecephys` NWB file and decimated from 20 kHz to 1250 Hz.

## Analysis

Maze position was linearized as the unwrapped angle around the loop centroid, reset to zero at each
lap's start, giving a monotonically increasing "distance traveled" coordinate per lap. From the 395
spike-sorted units, 13 fired at >0.5 Hz during runs, and 7 of those met place-cell criteria (peak
rate >2 Hz, spatial information >0.5 bits/spike) from 1D occupancy-normalized tuning curves computed
with Pynapple. Theta phase was extracted from the LFP with a zero-phase 5-11 Hz Butterworth filter
(chosen from the running-epoch power spectrum, which shows a clear theta peak near 7-9 Hz on top of
the 1/f background) followed by a Hilbert transform. For each place cell, in-field spikes (linearized
position within the tuning curve's field, in the running epochs) were assigned a linearized position
and a theta phase, and tested for phase precession with the circular-linear regression of Kempter et
al. (2012), with significance from a position-shuffle permutation test on the resulting circular-linear
correlation coefficient.

## Key Finding

One place cell (unit 471) shows a clean, textbook example of theta phase precession: as the rat
advances through the cell's place field, spike theta phase decreases from roughly 520° down to
roughly 60° (slope ≈ -221°/rad of linearized position), giving a significant circular-linear
correlation (rho = 0.53, p = 0.010, permutation test, n = 32 in-field spikes). Three additional
candidate place cells show negative precession slopes in the same direction but do not reach
significance individually (p = 0.25-0.39), consistent with the limited sampling of a single session
(18 laps). This reproduces, in one real recording from DANDI:000059, the classic phase-precession
phenomenon originally described by O'Keefe & Recce (1993): the systematic advance of spike timing
relative to the local theta rhythm as an animal crosses a place field.

## Files

- `theta_phase_precession_analysis.py` — consolidated, end-to-end jupytext script (source of truth)
- `theta_phase_precession_analysis.ipynb` — executed notebook version
- `01_load_data.py`, `02_preprocess_and_placefields.py`, `03_theta_phase_precession.py`,
  `04_visualize_placefields.py` — modular prototyping scripts used to develop the pipeline
- `figures/` — all output figures (maze trajectory, theta validation, per-cell precession plots,
  population summary, place-field tuning curves, exemplar spike map)
