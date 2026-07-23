# Theta Phase Entrainment and Precession in Hippocampal Place Cells

This analysis uses [DANDI:000044](https://dandiarchive.org/dandiset/000044) ("Diversity
in neural firing dynamics supports both rigid and learned hippocampal sequences",
Grosmark & Buzsaki), specifically session
`sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`. The session
contains simultaneous 128-channel CA1 LFP (1250 Hz) and 137 spike-sorted units (120
excitatory, 17 inhibitory) recorded while a rat ran back and forth on a 1.6 m linear
track. Data was streamed directly from the DANDI S3 bucket with `remfile` (no full
download) and analyzed with Pynapple.

The pipeline first selects an LFP channel with the strongest theta (6-10 Hz) to delta
(2-4 Hz) power ratio, bandpass-filters it and extracts the instantaneous theta phase
via the Hilbert transform. Track traversals were identified from gaps in position
tracking (the animal is untracked at the reward wells) and split into outbound and
inbound runs; place fields for excitatory units were computed as position tuning
curves on outbound runs and ranked by spatial information. For theta phase
entrainment, spike phases across all place cells were tested for circular uniformity
with a Rayleigh test: 63 of 89 place cells (71%) showed significant phase locking.
For phase precession, spikes fired inside each cell's place field were related to
normalized in-field position with a circular-linear regression (Kempter et al., 2012),
with significance assessed by a permutation test shuffling spike-position
assignments. Of 61 candidate place cells with a well-defined field, 28 showed a
significant negative circular-linear slope (spike phase decreasing as the rat crosses
the field, the classic O'Keefe & Recce 1993 precession signature) versus only 1 with
a significant positive slope, confirming phase precession as a population-level
phenomenon in this session.

## Files

- `theta_phase_precession.py` — jupytext (percent format) source script, runs
  end-to-end with no manual intervention.
- `theta_phase_precession.ipynb` — the same analysis as an executed Jupyter notebook.
- `fig1_raw_data_overview.png` — tracked position and example spike raster across
  track traversals.
- `fig2_theta_extraction.png` — raw vs. theta-filtered LFP, instantaneous theta
  phase, and the LFP power spectrum during running.
- `fig3_place_fields.png` — population place-field map (sorted by field location)
  and example single-cell tuning curves.
- `fig4_theta_entrainment.png` — example spike-phase polar histograms and population
  summary of theta phase-locking strength.
- `fig5_phase_precession_examples.png` — example phase-vs-position scatter plots
  with fitted circular-linear regression lines.
- `fig6_precession_summary.png` — population summary of circular-linear slopes and
  correlations across candidate place cells.
