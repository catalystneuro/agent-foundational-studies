# Theta Phase Precession in Hippocampal Place Cells

Demonstration of theta phase precession (O'Keefe & Recce, 1993) with real
extracellular recordings from the DANDI Archive, streamed directly from S3
without downloading the full NWB file.

## Dataset

- **DANDI dandiset 000044** ("Diversity in neural firing dynamics within the
  hippocampal formation", Buzsaki lab, Rutgers).
- **Session**: `sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`
  (8.7 GB NWB; rat CA1 tetrode recording on a 1.6 m linear maze).
- **Access**: remote byte-range streaming via `remfile` with an on-disk cache
  (`/tmp/remfile_cache`). Only the LFP theta channel (~87 MB) and the spike /
  position / epoch objects were actually pulled over the network, well within a
  2 GB cache budget.

## What was analyzed

1. **Theta LFP**. The 128-channel LFP (1.25 kHz) was scanned during a 200 s
   window of the maze epoch; each channel was scored by the theta/delta PSD
   ratio (6-10 Hz vs 3-5 Hz). The strongest-theta channel was band-pass
   filtered (3rd-order Butterworth, sosfiltfilt) and its instantaneous phase
   taken from the analytic signal (Hilbert transform).
2. **Run bouts**. Run bouts were defined from the linearized position series
   (NaN when the rat sits at the reward platforms): contiguous valid stretches,
   gaps < 0.3 s merged, requiring >= 1 s duration, >= 0.3 m span, and median
   |speed| > 0.15 m/s, labelled by travel direction.
3. **Place cells**. Per-direction tuning curves (50 bins, Gaussian-smoothed).
   A cell-direction counted as a place cell when peak rate >= 1 Hz and its
   Skaggs spatial information was significant (p < 0.05) against a circular
   time-shift shuffle of the cell's own spike train.
4. **Theta phase precession**. For each place field (>= 30 in-field spikes;
   field delimited by the 20%-of-peak rate contour; progress coordinate
   oriented by travel direction), a circular-linear regression of spike theta
   phase on field progress was fit by maximizing the mean resultant length
   R(k) = |mean(exp(i(phi - 2 pi k x)))| over k in [-6, 6] cycles/field.
   Significance was assessed with a phase-permutation shuffle (500 draws).
   A cell-direction precesses when the fitted slope is significantly negative
   (phase advancing to earlier theta phases as the animal crosses the field).
5. **Per-lap check**. The identical correlation was computed within single
   traversals of each field (>= 4 spikes covering >= 25% of the field) to
   confirm precession is present lap-by-lap, not only in the pooled scatter.

## Key finding

Theta phase precession is robustly present: ~55% of the detected
cell-directions showed a significant negative phase-progress relationship
(e.g. 79 negative vs 5 positive), with a median circular-linear correlation of
rho ~ -0.3 and a median slope of about -0.5 cycles per field width
(approaching the canonical ~-2.5 radians per 100 cm / -1 cycle per theta
cycle). The precession is also visible within single lap traversals (81% of
laps with a negative correlation), and the pooled phase-progress scatter shows
the classic diagonal band when each cell's phase offset is removed.

## Outputs

- `final.py` — consolidated jupytext script (markdown cells + code) that runs
  end-to-end; the source notebook.
- `final.ipynb` — the same pipeline as a Jupyter notebook (converted with
  jupytext).
- `figures/` — all figures as PNG:
  - `fig01_raw_overview.png` — trajectory geometry, position validity, example spikes
  - `fig02_theta_lfp.png` — theta-channel PSD and raw/filtered LFP trace
  - `fig03_run_bouts.png` — run-bout detection, durations, speeds
  - `fig04_example_cells.png` — tuning curves and lap-colored phase-progress for the
    strongest precession cells
  - `fig05_population.png` — slope/rho distributions and significance counts
  - `fig06_pooled_progress.png` — pooled phase vs progress with circular means
  - `fig07_perlap.png` — per-lap correlation distribution
- `results_summary.json` — machine-readable summary of the analysis.
- `run.log` — full execution log.