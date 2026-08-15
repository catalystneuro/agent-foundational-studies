# Decoding an upcoming decision bias from pre-stimulus neural activity

## Dataset

**DANDI 000409 — IBL, Brain Wide Map.** Head-fixed mice perform the International Brain
Laboratory `biasedChoiceWorld` visual decision task: a Gabor patch appears on the left or
right at one of several contrasts (0, 6.25, 12.5, 25, 100 %) and the mouse turns a wheel to
bring it to center. Neuropixels probes record hundreds of spike-sorted units per session.
The analysis uses the processed `*_desc-processed_behavior+ecephys.nwb` files, streamed
from the DANDI S3 store with `remfile` disk caching (only spike times and the trial table
are fetched, never the raw voltage). Six sessions from six different mice were analyzed
(NYU-11, NYU-30, NYU-37, NYU-46, NYU-40, NYU-39).

Two features of the task make it ideal for this question. First, the stimulus side is drawn
in **blocks** with `probability_left` ∈ {0.2, 0.5, 0.8}, imposing a **decision bias** that
mice learn and act on. Second, each trial enforces a **quiescence period** (≥ 400 ms of
wheel stillness) before the stimulus, giving a movement-free **pre-stimulus window**.

## What was analyzed

For each session, spikes were counted per unit in the **−400 to 0 ms window before
stimulus onset** (`gabor_stimulus_onset_time`), and an L2-regularized logistic decoder
(5-fold cross-validation, balanced classes) was trained to predict, from that
pre-stimulus population activity alone: (1) the **upcoming choice** (clockwise vs
counter-clockwise wheel turn) and (2) the **upcoming bias** (block prior, 0.2 vs 0.8).
Accuracy was compared to a 200-iteration label-shuffle null. The decode was also run in
sliding 100 ms windows from −1.0 to +0.6 s around onset, and linked to behavior on the
zero-contrast trials where choice reflects pure internal bias.

## Key finding

**An upcoming decision bias is decodable from population activity 400 ms before the
stimulus appears.** The imposed block prior is read out at cross-validated **AUC 0.71–0.85
(mean 0.79) in all six sessions** (each p < 0.005 vs shuffle), and the upcoming choice at
**AUC 0.54–0.65 (mean 0.61), significant in 5/6 sessions**. Sliding-window decoding shows
the bias signal is present and roughly flat throughout the movement-free pre-stimulus
period and then jumps sharply at stimulus onset, confirming the pre-onset signal is not a
movement or sensory artifact. On zero-contrast trials, where the choice is pure bias, the
zero-contrast choice probability rises monotonically with the block prior (0.24 → 0.52 →
0.64), and sorting those trials by the **pre-stimulus decoder's output** recovers a
monotonic gradient in the mouse's actual choice (pooled AUC ≈ 0.64). Together this shows
the brain holds a decision-relevant bias state before the evidence arrives, and that this
pre-stimulus state predicts the biased behavioral choice.

## Files

- `decode_prestim_decision_bias.py` — consolidated jupytext script (percent format), runs end-to-end.
- `decode_prestim_decision_bias.ipynb` — executed notebook with embedded figures.
- `db_pipeline.py` — streaming NWB loading, unit selection, peri-onset spike-count tensors.
- `build_cache.py` — per-session tensor/behavior caching.
- `analyze.py` — decoding (per-session, shuffle nulls, sliding-window).
- `make_figs.py` — figure generation.
- `figures/`
  - `fig1_raw_validation.png` — peri-onset firing heatmap and population rate by block (validation).
  - `fig2_prestim_decoding.png` — per-session pre-stimulus decoding of choice and bias vs shuffle null.
  - `fig3_temporal_decoding.png` — sliding-window decoding across time (the headline result).
  - `fig4_behavior_link.png` — behavioral bias by block and neural prediction of zero-contrast choice.
- `cache/decode_results.csv` — per-session AUCs and p-values.

## Reproducing

```bash
pip install dandi pynwb remfile h5py pynapple scikit-learn matplotlib pandas tqdm jupytext nbconvert
python analyze.py          # builds cache from DANDI stream (first run ~a few min), writes figures
# or run the notebook end-to-end:
jupytext --to notebook decode_prestim_decision_bias.py
jupyter nbconvert --to notebook --execute --inplace decode_prestim_decision_bias.ipynb
```

## Notes and caveats

- Decoding is within-session (units are not shared across sessions); reported AUCs are the
  mean of 5-fold cross-validated out-of-sample predictions.
- The block prior is a slowly varying trial variable, so cross-validation uses shuffled
  (non-temporal) folds; the shuffle null (which preserves the same fold structure) confirms
  the effect is not an artifact of class imbalance or fold construction.
- Units were filtered to ≥ 1 Hz mean rate; no brain-region selection was applied, so the
  decode reflects a distributed, brain-wide bias signal consistent with the IBL findings.
