# Decoding an upcoming decision bias from pre-stimulus neural activity

**Dataset.** DANDI Archive dandiset [000149](https://dandiarchive.org/dandiset/000149), "IBL ephys data" (International Brain Laboratory). All four sessions in the dandiset were used (591, 794, 303, and 388 sorted units; 520 to 1213 trials per session). In this task a mouse reports whether a visual grating appeared on the left or right by turning a wheel. Trials are organized in blocks in which the stimulus appears on the left with probability 0.8, 0.5, or 0.2, and this block prior biases the animals' choices. Data were streamed with LINDI (no bulk download); spike times and trial tables are cached locally as `session_*.npz` on first run.

**Analysis.** We asked whether the choice an animal is about to make can be decoded from population activity in a pre-stimulus window, [-0.4, 0) s relative to stimulus onset, where no sensory evidence is available (the go cue is simultaneous with stimulus onset, the previous trial's feedback ends at least 2.4 s earlier, and the median reaction time is about 0.15 s after onset, so the window contains neither stimulus responses nor movement). For each session we counted spikes of all quality-controlled units (`ks2_label == "good"`) in this window and decoded the upcoming choice with logistic regression under 10-fold stratified cross-validation, scored by balanced accuracy against 100 label permutations. We then (a) measured the time course of decoding in sliding 100 ms windows, (b) repeated the decoding on 0%-contrast trials only, where the stimulus is uninformative and choices are pure expressions of bias, (c) compared against within-block and previous-choice stratified nulls, and (d) localized the signal with single-unit ROC analysis and per-region decoding.

**Key finding.** The upcoming choice is decodable from pre-stimulus activity above chance in all four sessions (balanced accuracy 0.52, 0.60, 0.53, 0.59 against nulls centered at 0.50; individually significant in three sessions, Stouffer combined p = 1.1e-08). Because the stimulus has not yet appeared, this predictability is a readout of the animal's upcoming decision bias. The behavioral bias itself is large: at 0% contrast, the probability of choosing left differs by 0.31 to 0.74 between the 0.8-left and 0.2-left blocks. On 0%-contrast trials the pre-stimulus decoder reaches 0.64 in the best session (p = 0.04). Time-resolved decoding shows choice information present hundreds of milliseconds before stimulus onset and growing sharply after it. Stratified nulls show that the signal has two components: a slow block-level bias state (present in all sessions) and, in the two strongest sessions, trial-resolution choice information that survives within-block and previous-choice shuffling (p = 0.01 to 0.04). About 12% of single units are individually choice-selective before stimulus onset, and the signal is distributed across regions (caudoputamen, midbrain reticular nucleus, lateral posterior thalamus, and visual areas, depending on probe placement).

## Files

- `decision_bias_prestim_decoding.py`: consolidated jupytext script, runs end-to-end (streams from DANDI on first run, then uses the local npz caches).
- `decision_bias_prestim_decoding.ipynb`: the same analysis as a Jupyter notebook.
- `figs/fig1_behavior_psychometric.png`: psychometric curves per block prior per session.
- `figs/fig2_prestim_choice_decoding.png`: pre-stimulus choice decoding per session vs shuffle nulls.
- `figs/fig3_timecourse_decoding.png`: time-resolved decoding relative to stimulus onset.
- `figs/fig4_zero_contrast_decoding.png`: decoding on 0%-contrast trials.
- `figs/fig5_control_nulls.png`: real accuracy against trial, within-block, and previous-choice nulls.
- `figs/fig6_single_unit_selectivity.png`: PSTHs of the most choice-selective units, split by upcoming choice.
- `figs/fig6b_auc_distribution.png`: distribution of single-unit pre-stimulus choice AUCs vs null.
- `figs/fig7_region_decoding.png`: per-region pre-stimulus choice decoding.
- `cache_session.py`, `analyze_prototype.py`, `benchmark_sessions.py`, `controls.py`, `within_block_null.py`: development scripts (superseded by the consolidated script).

## Reproducing

```
python decision_bias_prestim_decoding.py   # ~5 min with caches present; first run streams ~300 MB of table data
jupytext --to ipynb decision_bias_prestim_decoding.py
```

Requires `pynwb`, `lindi`, `pynapple`, `scikit-learn`, `scipy`, `matplotlib`, `tqdm`.
