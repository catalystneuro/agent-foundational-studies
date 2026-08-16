# Decoding upcoming decision bias from pre-stimulus neural activity

**Dataset:** [DANDI:000149](https://dandiarchive.org/dandiset/000149), International
Brain Laboratory (IBL) ephys data. Four Neuropixels sessions (one per subject),
each with spike-sorted units (106–429 KS2-good units after quality filtering) and a
trial table (choice, signed contrast, block prior, stimulus onset, first movement).
Files are streamed with LINDI; no bulk downloads.

**Task and manipulation.** Mice report the side of a visual grating by turning a
wheel. Within blocks of trials the stimulus appears on the left with probability
0.2, 0.5, or 0.8 (the *block prior*), which biases the animals' choices. On
0%-contrast trials no visual evidence is shown, so the choice there is a pure
readout of the animal's bias. An enforced ~0.4–0.7 s pre-stimulus quiescence period
provides a movement-free window immediately before stimulus onset; we decode from
spike counts in the window [-0.4, 0] s relative to stimulus onset.

**Analysis.** Cross-validated LDA (Ledoit-Wolf shrinkage; balanced accuracy;
permutation nulls, 200 shuffles) decodes, from pre-stimulus population activity:
(i) the upcoming choice on all trials, (ii) the upcoming choice on 0%-contrast
trials, (iii) the block prior (sustained bias state), and (iv) the previous trial's
choice (control). We additionally test whether the neural axis that encodes the
block prior generalizes to predict choice on 0%-contrast trials (cross-decoding),
measure the sliding-window time course of choice information, and quantify
single-unit choice selectivity (auROC with per-unit permutation tests).

**Key findings.** *(numbers from the four sessions; see `decoding_summary.csv`,
`cross_decoding.csv`)*

- The behavioral bias is strong: on 0%-contrast trials, P(choose left) tracks the
  block prior (session means 0.29 / 0.63 / 0.85 for p(left) = 0.2 / 0.5 / 0.8).
- The upcoming choice is decodable from pre-stimulus activity on all trials in
  4/4 sessions (balanced accuracy 0.534–0.597, permutation p ≤ 0.025 per session;
  Fisher combined p < 0.0001).
- On 0%-contrast trials, where choice cannot be stimulus-driven, the upcoming
  choice is decodable at the group level (accuracies 0.510–0.636; Fisher combined
  p = 0.010; individually significant in 2/4 sessions).
- The block prior (sustained bias state) is decodable in 4/4 sessions
  (accuracies 0.563–0.701, p = 0.005 per session; Fisher combined p < 0.0001).
- The block-bias neural axis generalizes: a decoder trained to distinguish
  left- vs right-biased blocks (on stimulus-present trials only) predicts the
  upcoming choice on held-out 0%-contrast trials (accuracies 0.546–0.690; Fisher
  combined p = 0.0017), linking the sustained bias state to moment-to-moment
  biased choices.
- The previous trial's choice is not decodable from the same window (all
  p > 0.27), and the sliding-window analysis shows choice information present
  hundreds of ms before stimulus onset and rising sharply after it.
- 12–19% of single units are significantly choice-selective before stimulus onset
  on all trials (per-unit permutation tests, vs 5% expected by chance), and
  34–53% after stimulus onset; on 0%-contrast trials the significant pre-stimulus
  units are most numerous in caudoputamen (dorsal striatum).

**Conclusion.** An upcoming decision bias is represented in pre-stimulus population
activity: before any sensory evidence appears, neural activity carries both the
sustained block-wise bias and, on ambiguous trials, the specific choice the animal
is about to make.

## Files

- `decision_bias_analysis.py`: consolidated jupytext script, runs end-to-end
  (streams the four sessions, caches small arrays to `cache_*.npz`, produces all
  figures and CSVs; ~30 min total, most of it the sliding-window analysis).
- `decision_bias_analysis.ipynb`: the same analysis as a Jupyter notebook.
- `fig1_behavior.png` … `fig6_single_units.png`: figures.
- `decoding_summary.csv`, `cross_decoding.csv`, `single_unit_auroc.csv`: results.
- `decoding_lib.py`, `cache_session.py`, `prototype_*.py`, `test_*.py`,
  `sliding_window.py`, `single_units.py`: development/prototype scripts.

## Reproducing

```
python decision_bias_analysis.py          # end-to-end (streams from DANDI once)
jupytext --to notebook decision_bias_analysis.py -o decision_bias_analysis.ipynb
```

Requires: pynwb, lindi, numpy, pandas, scikit-learn, scipy, matplotlib, tqdm,
jupytext.
