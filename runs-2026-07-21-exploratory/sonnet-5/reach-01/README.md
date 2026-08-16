# Reach Direction and Velocity Tuning in Primary Motor Cortex

This analysis uses Dandiset [000129](https://dandiarchive.org/dandiset/000129) ("MC_RTT: Macaque
motor cortex spiking activity during self-paced reaching"), contributed by the Sabes lab
(O'Doherty, Cardoso, Makin et al.) as part of the Neural Latents Benchmark. The session
(`sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb`, streamed directly from the DANDI S3 bucket
with `remfile`) contains 130 sorted units recorded from a multi-electrode array in primary motor
cortex (M1) of a rhesus macaque, together with 1 kHz cursor kinematics, while the animal made
continuous, self-paced reaches to targets that appeared at random locations on a grid with no
inter-trial delay (649 s of recording, 545 target acquisitions).

## Analysis

Cursor velocity was computed as the smoothed numerical derivative of cursor position, from which
instantaneous movement direction and speed were derived. Two complementary, independent methods
were used to characterize tuning:

1. **Continuous kinematic tuning**: firing rate was related to instantaneous movement direction
   (during speed-thresholded movement epochs) and to instantaneous speed (quantile-binned to
   handle the right-skewed speed distribution) using `pynapple`'s `compute_tuning_curves`.
   Direction tuning curves were fit with a cosine function (Georgopoulos et al., 1982) to extract
   each unit's preferred direction (PD) and tuning strength (R²).
2. **Discrete reach-based tuning**: the session was independently segmented into 545 individual
   reaches, each defined by a target acquisition event, with reach direction computed
   geometrically from the cursor position at target onset to the new target location. Per-reach
   firing rates (0-600 ms post-onset) were binned by direction octant to derive a second,
   independent estimate of preferred direction for comparison.

## Key Findings

Directional tuning was strong and widespread: 83/130 units (64%) had a cosine-fit R² > 0.3, and
59/130 (45%) exceeded R² > 0.5, with several units reaching R² > 0.9 (e.g., unit 301, R²=0.91,
PD=27°). Speed tuning was similarly pervasive, with 99/130 units (76%) showing |Spearman ρ| > 0.5
between firing rate and speed; most of these were positively tuned (85 of 99), consistent with
the classic M1 finding that firing rate scales with movement speed (Moran & Schwartz, 1999),
though a substantial minority (14 of 99) showed the opposite, negatively-tuned relationship. The
two independent direction-tuning estimates (continuous velocity-based vs. discrete per-reach)
agreed well: across the 83 well-tuned units, the median angular offset between the two methods'
preferred-direction estimates was only 18°, and a peri-reach raster/PSTH for the most strongly
tuned unit showed a clear ~2.5x modulation of peak firing rate across reach-direction quadrants,
confirming that the tuning curves reflect real, direction-selective spiking rather than an
artifact of the binning procedure. Preferred directions were not uniformly distributed across the
population but clustered into a few preferred sectors (visible in the rose histogram and the
blocky, rather than perfectly diagonal, structure of the sorted tuning-curve heatmap).

## Files

- `reach_direction_velocity_tuning.py` — consolidated jupytext analysis script (markdown + code cells)
- `reach_direction_velocity_tuning.ipynb` — executed Jupyter notebook version
- `fig1_raw_data_validation.png` — raw cursor trajectory and population spike raster
- `fig2_kinematics.png` — derived speed and direction traces
- `fig3_direction_tuning_examples.png` — polar tuning curves, top 6 direction-tuned units
- `fig4_direction_population_summary.png` — sorted tuning-curve heatmap and PD rose histogram
- `fig5_speed_tuning_examples.png` — example positively- and negatively-speed-tuned units
- `fig6_speed_population_summary.png` — population histogram of speed-rate correlations
- `fig7_discrete_vs_continuous_validation.png` — cross-method preferred-direction agreement
- `fig8_raster_by_direction.png` — peri-reach raster/PSTH by direction quadrant for one example unit
