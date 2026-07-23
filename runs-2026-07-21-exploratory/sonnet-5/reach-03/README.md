# Reach Direction and Velocity Tuning in Motor Cortex

**Dataset**: [DANDI Dandiset 000070](https://dandiarchive.org/dandiset/000070), "Neural population
dynamics during reaching" (Churchland lab, Stanford). We analyze one session,
`sub-Jenkins/sub-Jenkins_ses-20090912_behavior+ecephys.nwb`, streamed directly from the DANDI
S3 bucket with `remfile` (no full download). The session contains 192 sorted single units from
two 96-channel Utah arrays (96 in primary motor cortex, M1, and 96 in dorsal premotor cortex,
PMd), hand position sampled at 1 kHz, and 1588 trials of a center-out-like reaching task with
virtual barriers ("maze" task) that force straight or curved reaches to one of several targets.

## Analysis

Reach-direction tuning was demonstrated in two complementary ways. First, using the subset of
348 trials with no barriers and a single target (classic straight center-out reaches to one of
9 peripheral locations), we fit the cosine tuning model of Georgopoulos et al. (1982) to each
unit's mean firing rate during the movement epoch. Second, using continuous hand-velocity
(computed from hand position with Pynapple's `derivative`) across all 1573 successful reaches,
including curved obstacle trials, we computed Pynapple tuning curves relating instantaneous
firing rate to both movement direction and movement speed.

## Key finding

A substantial fraction of units in both M1 and PMd show clear cosine-shaped direction tuning
(up to R² = 0.91 for the best-fit unit), with preferred directions spread broadly around the
circle, consistent with a distributed population code for reach direction rather than a small
set of direction-specific cells. Independently, the continuous-kinematics analysis shows that
firing rate in many units (116 of 192 with |r| > 0.7) is very strongly and often near-monotonically
correlated with instantaneous hand speed (up to r = 0.98), reproducing the classic finding that
motor and premotor cortical activity co-varies with movement velocity in addition to movement
direction.

## Files

- `reach_tuning_analysis.py` — jupytext (light format) analysis script, runs end-to-end
- `reach_tuning_analysis.ipynb` — same analysis as an executed Jupyter notebook
- `fig01`–`fig07` `.png` — figures (raw data validation, task structure, discrete cosine-tuning
  polar plots and population summary, continuous velocity direction/speed tuning curves and
  population summary)
