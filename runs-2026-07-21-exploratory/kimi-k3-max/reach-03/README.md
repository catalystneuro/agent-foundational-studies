# Reach direction and velocity tuning in macaque motor cortex

**Dataset**: [DANDI 000128](https://dandiarchive.org/dandiset/000128) (version 0.220113.0400) — MC_Maze from the Neural Latents Benchmark '21 (Churchland lab). Extracellular recordings from primary motor cortex (M1) and dorsal premotor cortex (PMd) of a macaque ("Jenkins") performing a delayed center-out reaching task in which virtual barriers force curved reach trajectories. The file `sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb` (690 MB) is streamed with `remfile` + disk cache; it contains 182 sorted units, 2295 successful trials, and 1 kHz hand position/velocity. (The NWB file labels kinematics as meters/m/s, but values are on a millimeter scale; we report mm and mm/s.)

**Analysis** (`reach_tuning_analysis.py`, jupytext; `reach_tuning_analysis.ipynb` is the converted notebook): reach direction per trial is the angle from the hand position at movement onset to the cued target. Peri-movement firing rates (−50 to +400 ms around movement onset) are fit with cosine tuning (preferred direction, modulation depth), and significance is assessed with an ANOVA across direction bins plus a shuffle test on the rate-weighted resultant length. Velocity tuning is quantified continuously during movement epochs with pynapple tuning curves over hand speed and velocity direction, and with a spike-rate × speed cross-correlation over ±300 ms lags. Finally, per-unit Poisson GLMs (nemos) predict 10 ms spike counts from [cos θ_velocity, sin θ_velocity, speed], trained on the dataset's own `train` trials and scored on held-out `val` trials.

**Key findings**: 71% of units (130/182) are significantly direction-tuned during movement execution and 43% during the delay period, with preferred directions tiling the workspace. Firing rates increase with hand speed and the rate–speed cross-correlation peaks at positive lags (median +35 ms across all units, +30 ms among the 109 units with peak r ≥ 0.1), i.e. neural activity leads the kinematics. Preferred directions estimated from instantaneous velocity-direction tuning agree with reach-direction PDs (circular correlation 0.33, difference distribution peaked at 0). GLMs confirm that velocity direction and speed each carry unique, decodable information about spiking on held-out trials: 97% of units with rates ≥ 2 Hz beat the null model, and the direction + speed model scores best (median pseudo-R² 0.0105 vs 0.0043 for either alone).

## Figures

| File | Content |
|---|---|
| `fig1_behavior.png` | Reach trajectories colored by direction, speed profiles, direction distribution |
| `fig2_raw_data.png` | Raster of 60 units with hand velocity/speed over two example reaches |
| `fig3_example_direction_tuning.png` | Cosine direction tuning of 4 example units (Cartesian + polar) |
| `fig4_psth_by_direction.png` | Rasters and PETHs by direction bin for 2 example units |
| `fig5_population_direction.png` | Population PD distribution, PD stability, modulation depths, population PETH |
| `fig6_speed_tuning.png` | Speed tuning curves, velocity-direction PD agreement, rate–speed cross-correlation and lags |
| `fig7_glm_encoding.png` | GLM example prediction, direction-vs-speed contributions, model comparison |

## Reproducing

```
python reach_tuning_analysis.py          # end-to-end, ~15 min (GLM fits dominate)
jupytext --to notebook reach_tuning_analysis.py
```

Requires: pynapple, pynwb, h5py, remfile, nemos, scipy, matplotlib, tqdm, requests, pandas. Data are streamed from DANDI; downloaded chunks are cached in `/tmp/remfile_cache_mcmaze`.
