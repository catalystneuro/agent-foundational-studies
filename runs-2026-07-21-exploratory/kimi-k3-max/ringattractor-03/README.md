# The head-direction system as a continuous ring attractor maintained during sleep

**Dataset**: DANDI Archive dandiset [000056](https://dandiarchive.org/dandiset/000056) (published version `0.250624.0430`) — Peyrache, Lacroix, Petersen & Buzsáki (2015), *"Internally organized mechanisms of the head direction sense"*, Nature Neuroscience 18:569–575. Extracellular units from the anterodorsal thalamic nucleus (ADn) of freely moving mice, dual-LED head tracking at 39 Hz, and manually scored brain states (Awake / Non-REM / REM). NWB files are streamed with `remfile` + disk caching; all analysis uses Pynapple.

**What was analyzed**: Five sessions, one per mouse (Mouse17-130128, Mouse20-130514, Mouse24-131213, Mouse25-140123, Mouse28-140310), with Mouse28-140310 (20 HD cells) as the prototype. Head direction was computed from the two LEDs (tracking failures are `-1` sentinels, masked to NaN). HD cells were identified by the mean vector length (MVL) of spike angles during wakefulness against a random-time resampling null that inherits the true angular occupancy, with an effect-size floor of MVL > 0.3 (52/147 units passed across the five sessions). Wake tuning curves were used for Bayesian decoding of head direction at 100 ms resolution, and pairwise spike-count correlations (100 ms bins) were computed separately within Awake, REM, and Non-REM epochs.

**Key findings**: The ADn population forms a single bump of activity on the ring of preferred directions that tracks the measured heading during wakefulness (median decode error ~14°). During REM sleep, with the animal immobile and no sensory heading input, the same bump persists and drifts continuously around the ring — the decoded internal heading rotates smoothly (median angular speed ~120°/s versus ~660°/s after shuffling time bins within REM epochs), completing multiple full revolutions in a single REM episode. The pairwise correlation structure predicted by a ring (positive between cells with nearby preferred directions, negative near 180° separation) is preserved in both sleep states: correlation-of-correlations between wake and sleep matrices range 0.77–0.96 for REM and 0.47–0.95 for Non-REM across the five sessions (label-permutation null, all p ≤ 0.05; pooled pair-level r = 0.92 for REM and 0.90 for NREM over 300+ pairs; 53/147 units passed the HD-cell criteria). These are the defining signatures of a continuous ring attractor whose one-dimensional structure is internally maintained without sensory drive, reproducing the central result of the original publication.

## Files

- `ring_attractor_analysis.py` — consolidated jupytext script; runs end-to-end (`python ring_attractor_analysis.py`)
- `ring_attractor_analysis.ipynb` — same analysis as a Jupyter notebook (via `jupytext`)
- `fig1_session_overview.png` — full-session head direction with sleep-state shading and spike rasters
- `fig2_tuning_curves.png` — polar tuning curves of all HD cells in the prototype session + MVL selection
- `fig3_wake_bump_decode.png` — population bump tracking actual HD; wake decoding validation
- `fig4_rem_bump_speed.png` — bump drifting during a REM episode; angular-speed distribution vs time-bin shuffle
- `fig5_corr_matrices.png` — pairwise correlation matrices per state (sorted by preferred direction) and correlation vs angular distance
- `fig6_panel.png` — five-session replication: HD-cell yield, corr-of-corr with permutation nulls, pooled topology curves, pair-level scatter
- `01_inspect.py`, `02_hd_cells.py`, `03_ring_attractor.py`, `04_panel.py` — incremental development scripts (superseded by the consolidated script)

## Reproducing

```bash
pip install pynapple pynwb remfile h5py matplotlib scipy tqdm jupytext
python ring_attractor_analysis.py          # ~15 min with a warm remfile disk cache
jupytext --to notebook ring_attractor_analysis.py
```

Streaming uses a disk cache at `/tmp/remfile_cache_hd`; the first run downloads only the data chunks actually read.
