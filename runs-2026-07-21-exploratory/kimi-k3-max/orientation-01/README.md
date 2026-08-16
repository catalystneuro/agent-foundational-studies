# Orientation Selectivity in Mouse Visual Cortex (DANDI 000021)

## Dataset

[DANDI:000021](https://dandiarchive.org/dandiset/000021) — Allen Institute Visual
Coding, Neuropixels (Brain Observatory 1.1 stimulus set). Mice passively viewed
drifting gratings (8 directions x 5 temporal frequencies, 2 s presentations,
80% contrast) and static gratings (6 orientations x 5 spatial frequencies x 4
phases, 0.25 s presentations) while spikes were recorded with six Neuropixels
probes. Three sessions were analyzed (715093703, 719161530, 721123822),
streamed from the archive with LINDI so only the needed chunks are downloaded
and cached locally. Units were kept if their quality label was "good" and their
peak channel mapped to a visual cortical area (VISp, VISl, VISpm, VISam, VISrl)
through the electrodes table, giving 1389 units in total.

## Analysis

For every unit, the mean firing rate was computed for each grating direction
(drifting) or orientation (static). Orientation selectivity was quantified with
the vector-based global orientation selectivity index (gOSI) and direction
selectivity with gDSI; significance was assessed per unit with a permutation
test that shuffles direction labels across sweeps (500 shuffles). The pipeline
is implemented with Pynapple/NWB data structures plus numpy and is fully
reproduced by the consolidated script `orientation_selectivity_analysis.py`
(jupytext format; `orientation_selectivity_analysis.ipynb` is the same analysis
as a notebook).

## Key finding

Orientation selectivity is widespread and robust: 65% of visual-cortex units
(904/1389) are significantly orientation-selective for drifting gratings
(permutation p < 0.01), with similar fractions across VISp (62%) and the higher
visual areas VISl (68%), VISpm (72%), VISam (61%) and VISrl (65%). Preferred
orientations tile all four sampled orientations without a strong bias, and most
selective units respond similarly to both drift directions of their preferred
orientation (low gDSI), i.e. they are orientation- rather than
direction-selective, as expected for mouse visual cortex. The preference
estimated from drifting gratings replicates on the independent static-grating
block: among units selective in both, 82% agree on preferred orientation within
+/-30 deg. Example units show the classic signature, e.g. VISp unit 950930609
fires ~23 Hz at 135/315 deg and ~0 Hz at the orthogonal orientations
(gOSI = 0.75, p = 0.002).

## Outputs

- `orientation_selectivity_analysis.py` — consolidated jupytext script, runs end-to-end
- `orientation_selectivity_analysis.ipynb` — the same analysis as a Jupyter notebook
- `fig1_raw_data_overview.png` — VISp population raster with grating-direction color code
- `fig2_example_unit_psth.png` — perievent raster/PSTH of an example selective unit
- `fig3_example_tuning_curves.png` — direction tuning curves across the selectivity range
- `fig4_population_summary.png` — gOSI/gDSI distributions, preferred orientations, per-area fractions
- `fig5_static_gratings.png` — replication of orientation preference with static gratings
- `metrics_all_sessions.csv` — per-unit selectivity metrics for all 1389 units
- `01_load_data.py`, `01b_inspect_electrodes.py`, `02_prototype.py`,
  `03_multisession.py`, `03b_save_rates.py`, `04_figures.py` — development
  scripts showing the incremental build-up of the pipeline
