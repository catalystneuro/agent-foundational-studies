# Decoding an upcoming decision bias from pre-stimulus neural activity

**Dataset:** DANDI Archive dandiset [000149](https://dandiarchive.org/dandiset/000149),
"IBL ephys data" (International Brain Laboratory). All 4 sessions were analyzed
(subjects 92130c1b, 70bf8cbd, 9bebfe0b, c6e8125f). Each session contains
Neuropixels spike-sorted units (303–794 per session, 156–500 after the
Kilosort "good" filter) and a trials table for the IBL contrast-detection task:
a grating appears on the left or right (or at 0% contrast) after an enforced
0.4–0.7 s pre-stimulus quiescence period, and the mouse reports the side by
turning a wheel. Blocks of trials carry a prior P(left) ∈ {0.2, 0.5, 0.8} that
biases choice. Data were streamed with LINDI (no bulk download).

**Analysis:** For each session, spikes of good units were counted in the
pre-stimulus window [-0.4, 0] s relative to stimulus onset (a movement-free
period by task design; the ~1% of trials with premature movement were
excluded). The upcoming choice (left vs right) was decoded with an
L2-regularized logistic regression under repeated stratified 5-fold
cross-validation, and significance was assessed against 200-replicate shuffle
nulls, both global (labels permuted across all trials) and within-block (labels
permuted within contiguous block runs). The signal was further characterized
with time-resolved decoding in sliding 250 ms windows, single-unit choice AUCs
with shuffle-based p-values, and decoding restricted to bias-dominated trial
subsets (0%-contrast trials; unbiased-block trials), training on all trials and
scoring only the subset within each test fold.

**Key finding:** The upcoming choice is decodable from pre-stimulus population
activity in 3 of 4 sessions (balanced accuracy 0.52–0.59; Stouffer combined
p ≈ 1.3e-5 against the global trial-shuffle null), before any sensory evidence
is available. Three controls identify the signal as a slow decision-bias state:
(i) it is attenuated by the within-block shuffle control, which preserves
block-level choice statistics; (ii) it is strongest on 0%-contrast trials, where
choice is purely bias-driven (s4 balanced accuracy 0.66, p = 0.01; combined
p ≈ 0.006); and (iii) it is weak within unbiased P(left) = 0.5 blocks. About
10% of single units are individually choice-selective before stimulus onset
(uncorrected p < 0.05 vs a 5% null expectation), distributed across visual,
thalamic, and midbrain regions. In short, the animal's upcoming decision bias
is present in population activity hundreds of milliseconds before the stimulus
appears.

## Files

- `decision_bias_prestimulus_decoding.py` — consolidated jupytext script; runs
  the full pipeline end-to-end (`python decision_bias_prestimulus_decoding.py`,
  a few minutes with warm cache, ~15 min cold).
- `decision_bias_prestimulus_decoding.ipynb` — the same pipeline as a Jupyter
  notebook (via `jupytext --to notebook`).
- `fig1_task_behavior.png` — task schematic, psychometric curves by block
  prior, block structure.
- `fig2_example_neural.png` — population raster and example pre-stimulus
  choice-selective units.
- `fig3_timecourse.png` — time-resolved choice decoding around stimulus onset.
- `fig4_prestim_decoding.png` — pre-stimulus decoding vs global and
  within-block shuffle nulls.
- `fig5_single_units.png` — single-unit choice AUC distribution and regional
  breakdown.
- `fig6_bias_controls.png` — decoding on 0%-contrast and unbiased-block
  subsets.
- `summary_stats.csv` — per-session statistics.
- `ibl_common.py`, `run_session.py`, `make_figures.py` — modular development
  scripts (the consolidated script is self-contained).
- `results_s1..s4.npz` — per-session analysis outputs; `cache/` — pickled
  session data for fast re-runs.
