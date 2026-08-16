# Theta Phase Precession in Hippocampal Place Cells

## Dataset

[DANDI:000044](https://dandiarchive.org/dandiset/000044), "Diversity in neural
firing dynamics supports both rigid and learned hippocampal sequences"
(Grosmark & Buzsaki, NYU). Four rats (Buddy, Gatsby, Cicero, Achilles) were
implanted with silicon probes spanning bilateral dorsal CA1 and ran back and
forth on a 1.6-2 m linear track for water reward. Each session's NWB file
contains a 128-channel local field potential (1250 Hz), spike times for
sorted units annotated as excitatory or inhibitory, 2D and linearized
position on the track, and PRE/MAZE/POST epoch boundaries. Analysis used one
representative session per rat, streamed directly from the DANDI S3 bucket
with `remfile` (only the MAZE-epoch slice of one LFP channel, the spike
tables, and the position tables were fetched, not the full 5-9 GB files).

## Analysis

For each session, the LFP channel with the strongest theta (6-10 Hz) to delta
(2-4 Hz) power ratio was selected as the theta reference, band-pass filtered,
and Hilbert-transformed to obtain instantaneous theta phase. Running velocity
was computed from the linearized position to split each session into outbound
and inbound traversals of the track. For every excitatory unit and direction,
an occupancy-normalized firing-rate map was built and a single place field was
detected as the contiguous, monotonically descending region around the peak
rate. For spikes fired inside the field while running, we paired each spike's
theta phase with the animal's normalized position within the field and fit a
circular-linear regression (Kempter et al., 2012) to obtain a precession slope
and a circular-linear correlation coefficient, with significance assessed by
a 1000-iteration position-label permutation test.

## Key finding

Across 135 qualifying place fields (excitatory units x running direction)
pooled over all four rats, spike theta-phase and within-field position were
reliably correlated: the median circular-linear correlation was |rho| = 0.31,
76% of fields had a negative phase-position slope (the sign expected for
phase precession, median -0.57 theta cycles per field traversal), and 68%
of fields (92/135) were individually significant by permutation test, with
61% both significant and negatively sloped. This held consistently within
each of the four rats despite differences in theta reference channel, track
length, and recording session. Pooling spikes across all significant fields,
after centering each field's phase on its own fit, produces a clear diagonal
band of spike density descending from theta peak to trough as normalized
position advances through the field, the textbook signature of hippocampal
theta phase precession (O'Keefe & Recce, 1993; Skaggs et al., 1996).

## Files

- `theta_phase_precession_analysis.py` - consolidated jupytext script, runs
  end-to-end from DANDI streaming through final figures
- `theta_phase_precession_analysis.ipynb` - the same analysis as an executed
  Jupyter notebook
- `figures/01_raw_data_overview.png` - raw LFP, theta-filtered trace, theta
  phase, and linearized position validation plots
- `figures/02_example_place_fields_and_precession.png` - rate maps and
  phase-vs-position scatter plots for the four most significant example
  place fields
- `figures/03_population_summary.png` - correlation and slope distributions,
  per-rat reliability, and pooled phase-precession density plot across all
  135 place fields
