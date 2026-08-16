# Head Direction Cells in the Mouse Anterodorsal Thalamus (DANDI 000056)

This analysis demonstrates head direction (HD) cells using data from the
[DANDI Archive](https://dandiarchive.org), dandiset
[000056](https://dandiarchive.org/dandiset/000056): Peyrache, Lacroix, Petersen &
Buzsáki (2015), *"Internally organized mechanisms of the head direction sense"*
(Nature Neuroscience 18:569–575). The dataset contains extracellular recordings
from the anterodorsal nucleus of the thalamus (ADn) and postsubiculum of freely
moving mice, dual-LED head tracking at ~39 Hz, and manually scored behavioral
states (Awake / REM / Non-REM). Five sessions from five mice were analyzed
(Mouse17-130128, Mouse20-130514, Mouse24-131213, Mouse25-140123,
Mouse28-140310), streamed with `remfile` + disk caching; no files were
downloaded in full.

## What was analyzed

Head direction was reconstructed as the angle of the red-minus-blue LED
difference vector (tracking failures marked by -1 sentinels were masked to
NaN). For every unit, an occupancy-corrected HD tuning curve (60 bins) and the
mean vector length (MVL) of spike angles were computed over wake epochs. A unit
was classified as an HD cell when its MVL exceeded an occupancy-matched
random-time resampling null (1,000 shuffles, p < 0.05) and an effect-size floor
of MVL > 0.3. The random-time null is required because wake is fragmented into
short epochs of nearly constant heading, which invalidates the usual circular
time-shift shuffle. Analyses were run with Pynapple; in addition, a Poisson GLM
with a cyclic B-spline basis over HD (NeMoS) was fit as an encoding model, and
head direction was decoded from the HD-cell population with Bayesian decoding
(100 ms bins). Finally, pairwise HD-cell correlations were compared between
wake and sleep states, reproducing the paper's central result.

## Key findings

- **53 of 159 units (33%) across the five sessions are HD cells** (per session:
  7/39, 6/17, 6/22, 14/34, 20/47), with median MVL 0.57. Their preferred
  directions tile the circle without significant clustering (resultant
  R = 0.21, uniformity p = 0.09).
- A Poisson GLM with an 8-knot cyclic B-spline basis on HD reproduces the
  empirical tuning curves closely (per-cell McFadden pseudo-R² 0.16–0.31 for
  six of seven prototype HD cells),
  and Bayesian decoding of heading from the population is accurate: median
  absolute error ~23° on informative 100 ms bins (chance ~ 90°), with 58% of
  bins within 30° of the true heading (prototype session, 7 HD cells).
- **Pairwise correlation structure among HD cells is preserved from wake into
  sleep**: correlating the wake and sleep pairwise-correlation vectors across
  cell pairs gives r = 0.92 (REM) and r = 0.90 (Non-REM) pooled over 313 pairs
  from five sessions (permutation test, p ≤ 1e-4 for both). This reproduces
  the signature finding of Peyrache et al. (2015) and supports an internally
  organized, attractor-like head direction network.

## Files

- `head_direction_cells.py` — consolidated jupytext script; runs end-to-end
  (streams data, recomputes everything, writes all figures)
- `head_direction_cells.ipynb` — the same analysis as a Jupyter notebook
  (converted with jupytext)
- `figures/` — all figures (PNG):
  - `01_raw_tracking.png` — LED tracking snippet, HD trace, wake HD occupancy
  - `02_tuning_curves.png` — polar tuning curves of the HD cells, sorted by
    preferred direction
  - `03_mvl_vs_null.png` — observed MVL vs occupancy-matched null
  - `04_raster.png` — HD trace and sorted HD-cell raster over 60 s
  - `05_glm_tuning.png` — NeMoS GLM tuning curves vs empirical, rate snippet
  - `06_decoding.png` — Bayesian-decoded vs actual HD and error distribution
  - `07_state_correlations.png` — HD-cell correlation matrices by state
  - `08_wake_sleep_scatter.png` — wake vs sleep pairwise correlations
    (prototype session)
  - `09_multisession_summary.png` — per-session HD-cell counts, pooled MVL
    distribution, preferred-direction histogram
  - `10_pooled_wake_sleep.png` — wake vs sleep correlations pooled across
    sessions
- `hd_utils.py`, `prototype_*.py`, `multisession.py`, `aggregate_test.py` —
  development scripts (superseded by the consolidated script)
- `cache/` — remfile disk cache and per-session result caches

## Reproducing

```bash
python head_direction_cells.py          # end-to-end run (streams from DANDI)
jupytext --to ipynb head_direction_cells.py
```

Requirements: pynapple 0.11, nemos 0.2.6, pynwb, h5py, remfile, numpy, scipy,
matplotlib, xarray, tqdm, jupytext.
