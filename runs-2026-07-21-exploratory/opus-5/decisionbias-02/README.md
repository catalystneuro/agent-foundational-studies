# Decoding an upcoming decision bias before stimulus onset

## Dataset

DANDI [000409](https://dandiarchive.org/dandiset/000409), the International Brain
Laboratory Brain Wide Map. Mice perform a two-alternative visual contrast
discrimination task while Neuropixels probes record from distributed brain
areas. After an initial unbiased block, the probability that the Gabor appears
on the left alternates in uncued blocks between 0.8 and 0.2. The animal has to
infer the current prior from experience, and that inferred prior biases the
upcoming decision.

Files were streamed from the archive with `remfile` and cached on disk; none
were downloaded in full. 44 processed sessions were scanned and 20 were analysed
(7,631 units, 12,283 trials in biased blocks). Sessions were selected on
data-quality criteria and then stratified by the *behavioural* bias quartile, so
nothing about the neural data entered the selection. Eight of the 44 scanned
sessions were rejected because their `probability_left` alternates on every
trial rather than following the IBL block schedule; those files have a
meaningless block variable and, consistently, no measurable behavioural bias.

## What was analysed

The bias is present in the behaviour: on trials with |contrast| ≤ 6.25 % the
probability of choosing right is 0.368 higher in right-prior blocks than in
left-prior blocks (t = 10.3, p = 3 × 10⁻⁹ across 20 sessions, `fig01`).

The question is whether that bias can be read out of neural activity *before*
the stimulus. Spike counts of all units in the 400 ms window ending at Gabor
onset were fed to an L2-regularised logistic regression, cross-validated with
contiguous folds in trial order. The window is movement-free by construction:
the IBL trial engine requires the wheel to be still for 400 to 700 ms before the
stimulus appears (`fig02`). Because the block is strongly autocorrelated in
trial order, significance was assessed against **pseudo-sessions**, that is,
surrogate block sequences drawn from the IBL block-length distribution, which
have the same temporal statistics as the real task but no relation to the
recording.

## Key finding

The prior block is decodable from pre-stimulus population activity above the
pseudo-session null: mean cross-validated AUC 0.554 versus a null mean of 0.511
(group permutation p = 0.008; 5 of 20 sessions individually p < 0.05, combined
Stouffer Z = 2.84, `fig04`). The signal is not a last-moment anticipation of the
stimulus: an earlier window, 1000 to 600 ms before onset, gives AUC 0.544
(p = 0.016), and the time-resolved sweep is above the null across the whole
pre-stimulus interval (mean AUC 0.543 versus null 0.506, `fig03`). Put in terms
of the choice rather than its cause, the upcoming choice on trials that carry
almost no sensory evidence is also decodable from that window: AUC 0.536 for
|contrast| ≤ 6.25 % (p = 0.027) and 0.565 on zero-contrast trials alone
(p = 0.029, `fig05`). Across session × region populations the effect is
strongest in striatum (Z = 2.8), midbrain (Z = 2.4) and thalamus (Z = 2.1), and
absent in hippocampus and the remaining cortical grouping.

Three controls support the interpretation. A negative-control target that the
animal cannot anticipate, the contrast of the upcoming stimulus, decodes at
chance from the identical pipeline (mean AUC 0.501, `fig06`), so the
cross-validation and null model are not manufacturing significance. Pre-stimulus
wheel speed, whisker-pad motion energy and pupil diameter do not differ between
blocks (all p > 0.1), although as a feature set they do decode the block
slightly above chance (AUC 0.529), which is close enough to the neural value
that movement and arousal cannot be dismissed entirely. Single-neuron encoding
is weak: in a Poisson `PopulationGLM` (NeMoS) that regresses pre-stimulus counts
on block identity while absorbing slow drift with a B-spline basis over trial
index, only 2.9 % of 7,612 units exceed the pseudo-session threshold against
2.5 % expected (`fig08`), so this is a small, distributed population signal
rather than strong coding in individual cells.

Two results run against the strongest possible version of the claim, and are
reported as they came out. First, the animal's own recent behaviour predicts the
block far better than the neural data do (AUC 0.906 from previous choice,
previous reward, previous stimulus side and a running average of the last ten
choices), and adding the pre-stimulus neural read-out to that model does not
improve it (combined Z = 1.1, `fig06`). Second, the trial-by-trial fluctuation
of the read-out does not predict the animal's choice on low-evidence trials once
block identity, previous choice and contrast are in the model (t = −0.21,
p = 0.84, `fig07`), and the correlation across sessions between behavioural bias
magnitude and decoding accuracy is not significant (r = 0.10, p = 0.67,
`fig09`). So the demonstration is that the upcoming bias is present in
pre-stimulus activity above a stringent null, not that this particular linear
read-out is what the animal uses.

## Files

| File | Contents |
| --- | --- |
| `decision_bias_prestimulus_decoding.py` / `.ipynb` | consolidated end-to-end analysis |
| `ibl_common.py` | streaming NWB access, trial and unit tables, spike counting |
| `decoding.py` | cross-validation and the pseudo-session / circular-shift nulls |
| `01_scan_sessions.py` | session scan, block-integrity check, session selection |
| `02_extract.py` | per-session feature extraction (`run_extract.sh` supervises it) |
| `03_analyze.py` | block, choice, control and trial-wise analyses |
| `05_glm_encoding.py` | NeMoS Poisson `PopulationGLM` single-neuron encoding |
| `06_figures.py` | all figures |
| `results/` | result tables, including `group_tests.csv` |
| `session_data/` | extracted spike-count tensors and trial tables |

## Figures

1. `fig01_task_and_behavioural_bias.png` — psychometric shift between blocks, per-session bias, block structure
2. `fig02_raw_activity_and_alignment.png` — raw spiking, wheel trace, example unit split by block
3. `fig03_time_resolved_block_decoding.png` — decoding accuracy against time relative to onset
4. `fig04_prestimulus_block_decoding.png` — per-session and per-region block decoding
5. `fig05_upcoming_choice_decoding.png` — upcoming choice on low-evidence trials
6. `fig06_controls.png` — movement, history, and the negative control
7. `fig07_trialwise_bias_readout.png` — trial-wise read-out and psychometric split
8. `fig08_glm_single_unit_encoding.png` — NeMoS GLM block coefficients
9. `fig09_brain_behaviour_correlation.png` — behavioural bias against decoding accuracy
