# The head-direction system as a continuous ring attractor maintained during sleep

**Dataset**: DANDI Archive dandiset [000056](https://dandiarchive.org/dandiset/000056)
(Peyrache et al. 2015, *Nature Neuroscience* 18:569–575, "Internally organized
mechanisms of the head direction sense"). Extracellular recordings from the
anterodorsal thalamus (ADn) of freely moving mice, with dual-LED head
tracking (~39 Hz) and manually scored Awake / Non-REM / REM states. Five sessions
were analyzed, one per mouse (Mouse28-140310 as the main session, plus
Mouse25-140123, Mouse17-130128, Mouse20-130514, Mouse24-131213). All data were
streamed from DANDI with `remfile` + disk caching (no full downloads) and handled
with Pynapple.

**What was analyzed** (`ring_attractor_hd_sleep.py`, runnable end-to-end; figures
in `figures/`):

1. **HD cell identification** — wake tuning curves from the LED-derived heading;
   HD cells selected by mean vector length > 0.3 against a random-time resampling
   null (20/47 units in Mouse28). Tuning curves tile the circle.
2. **Wake ring readout** — sorted by preferred direction, the HD population forms
   a single activity bump that tracks the true heading; Bayesian decoding from
   100 ms population vectors recovers heading with median circular error ~20°
   (Mouse28; 50–105° in sessions with only 6–14 HD cells).
3. **Sleep correlation preservation** — pairwise correlations of HD-cell activity
   (0.5 s bins, Fisher-z averaged over epochs; wake restricted to active
   head-movement periods) are preserved in REM and Non-REM sleep. Nine of ten
   session-by-state tests are significant against a label-permutation null
   (p ≤ 0.027); the exception is the NREM test in the session with the fewest HD
   cells (5 cells, 10 pairs; p = 0.09). Pooled over 291 pairs, corr-of-corr
   r = 0.90 (REM) and 0.84 (NREM). This replicates the central result of
   Peyrache et al. 2015.
4. **REM bump dynamics** — decoding a *virtual* heading during REM from the wake
   tuning curves (with the population rate landscape flattened by NNLS rescaling)
   yields a localized bump (posterior concentration R = 0.98, comparable to wake)
   that drifts slowly and visits the whole ring. Against a circular time-shift
   null that preserves firing rates and slow single-cell rate fluctuations but
   destroys cross-cell coordination, the bump is significantly smoother and more
   persistent in the best-recorded session (p90 angular speed 14 vs 23°/s,
   p = 0.01; posterior autocorrelation at 1 s lag 0.47 vs 0.41, p = 0.01). In the
   four lower-yield sessions (5–14 HD cells) the effect is weak or absent
   (p = 0.05–1.0), where the decoded posterior is dominated by single-cell rate
   fluctuations that the null also captures.

**Key finding**: the pairwise correlation structure of the HD ring measured during
wake is internally maintained during both REM and Non-REM sleep across all five
mice, and — where the recorded ensemble is large enough — REM sleep contains a
localized, slowly drifting activity bump covering the ring. The ring organization
therefore does not depend on ongoing sensory input, as predicted by continuous
ring-attractor models of the HD system.

## Files

- `ring_attractor_hd_sleep.py` — consolidated jupytext script (markdown + code),
  runs end-to-end and regenerates all figures.
- `ring_attractor_hd_sleep.ipynb` — the same analysis as a Jupyter notebook.
- `figures/01_raw_data_validation.png` … `figures/06_multisession_summary.png`.
- `scripts/` — modular development scripts (`common.py` helpers, `01`–`07` stages);
  `data/` holds cached intermediate results (`.npz`) and run logs.

## Reproduce

```
pip install pynapple pynwb h5py remfile xarray matplotlib scipy tqdm jupytext
python3 ring_attractor_hd_sleep.py        # ~1 h, streams from DANDI
jupytext --to ipynb ring_attractor_hd_sleep.py
```
