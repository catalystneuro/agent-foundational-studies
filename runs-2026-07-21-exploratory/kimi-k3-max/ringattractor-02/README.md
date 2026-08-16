# The head-direction system is a ring attractor whose structure is internally maintained during sleep

**Dataset**: [DANDI 000056](https://dandiarchive.org/dandiset/000056), published version `0.250624.0430`: Peyrache, Lacroix, Petersen & Buzsáki (2015), *Nature Neuroscience*, "Internally organized mechanisms of the head direction sense". Extracellular units from the anterodorsal thalamic nucleus (ADn) and postsubiculum (PoSub), dual-LED head tracking, and scored behavioral states (Awake / REM / Non-REM). Five sessions were analyzed, one per mouse (Mouse17-130128, Mouse20-130514, Mouse24-131213, Mouse25-140123, Mouse28-140310), streamed from the archive with `remfile` plus a local disk cache; no file was downloaded in full.

**Analysis**. Head direction was reconstructed from the two LEDs (tracking failures are `-1` sentinels and were masked). HD cells were identified from wake tuning curves using the mean vector length of spike angles against a random-time resampling null (500 shuffles) with an effect-size floor of MVL > 0.3, yielding 53 HD cells of 159 units (7/39, 6/17, 6/22, 14/34, 20/47 per session). Pairwise Pearson correlations of 100 ms spike counts were computed separately in Awake, REM, and Non-REM epochs, and a virtual head direction was decoded from population activity with a Poisson Bayesian decoder (cross-checked against `pynapple.decode_bayes`, median agreement 6.4 deg).

**Key finding**. The HD system behaves as a continuous ring attractor whose one-dimensional structure is internally maintained during sleep. During wakefulness, pairwise correlations sorted by preferred direction show the circulant signature of a bump on a ring: correlation falls with angular distance between preferred directions and goes negative near 180 deg, with wrap-around at the 0/360 deg boundary. This structure is preserved essentially intact in both sleep states: the pooled correlation between wake and sleep pairwise-correlation matrices is r = 0.92 (REM) and r = 0.90 (Non-REM) over 313 pairs, and per-session correlation-of-correlations (means 0.88 REM, 0.78 Non-REM) sit far outside a label-permutation null (99th percentile 0.52-0.60; p = 0.005 and 0.008). Because the sleeping animal receives no directional sensory input, this preservation demonstrates that the ring is generated and held by the network itself. Decoding makes the same point dynamically: during sleep the population still forms a single localized bump (posterior concentration R ~ 0.94-0.97, as high as in wake) whose position drifts coherently around the ring. REM drift (~92 deg/s) is close to the actual head speed during wake exploration (~115 deg/s), Non-REM drift is faster (~270 deg/s), and shuffling time bins within the same sleep epochs makes the decoded angle 3-8x jumpier (~710-770 deg/s), showing that the continuity is a property of the activity, not of the decoder. Wake decoding validates the method (median error 15 deg). These results reproduce the central findings of Peyrache et al. (2015) from the same data.

## Files

- `ring_attractor_hd_sleep.py`: consolidated jupytext script; runs the full pipeline end-to-end (~20-40 min on a cold cache)
- `ring_attractor_hd_sleep.ipynb`: the same analysis as a Jupyter notebook
- `figures/`: all figures:
  - `fig_raw_raster_wake.png`: HD trace and sorted HD-cell raster during exploration: the bump tracks the head
  - `fig_tuning_polar_M28.png`: wake tuning curves of the 20 HD cells of Mouse28-140310
  - `fig_hdcell_selection.png`: preferred directions across sessions and the MVL selection statistic
  - `fig_corrmatrices_sorted.png`: correlation matrices sorted by preferred direction, per session and state
  - `fig_ring_profile_and_preservation.png`: correlation vs angular distance per state; wake vs sleep scatter
  - `fig_corr_of_corr_null.png`: observed structure preservation vs label-permutation null
  - `fig_decode_wake_validation.png`: decoded vs actual HD during wake
  - `fig_decode_rem.png` / `fig_decode_nrem.png`: decoded virtual HD during sleep: the bump drifts around the ring
  - `fig_decode_stats.png`: drift continuity (vs time-bin shuffle) and bump concentration by state
- `dev/`: development scripts and per-session caches used while building the pipeline

## Reproducing

```
python ring_attractor_hd_sleep.py        # or open the .ipynb
```

Requires `pynapple`, `pynwb`, `remfile`, `h5py`, `matplotlib`, `tqdm`. Streaming cache: `/tmp/remfile_cache_hd`.
