# Orientation Selectivity in Mouse Visual Cortex

This analysis demonstrates orientation selectivity, the classic response property of visual cortex neurons first described by Hubel and Wiesel, using real data from the DANDI Archive.

## Dataset

[DANDI:000021](https://dandiarchive.org/dandiset/000021), the Allen Institute *Visual Coding: Neuropixels* dataset (Brain Observatory 1.1). Three sessions were streamed with LINDI (no full downloads): `ses-715093703`, `ses-719161530`, and `ses-721123822`. In each session, a mouse viewed full-field drifting gratings (8 drift directions x 5 temporal frequencies, 2 s presentations at contrast 0.8, plus blank sweeps for spontaneous rate) while spiking was recorded from Neuropixels probes. Analysis was restricted to well-isolated (`quality == "good"`) units whose peak channel mapped to a visual cortical area (VISp, VISl, VISpm, VISam, VISrl), giving 1389 units across the three sessions.

## Analysis

For each unit, spikes were counted in every grating sweep and converted to firing rates. Sweeps overlapping the session's `invalid_times` intervals (flagged by the Allen SDK) were excluded; this affected 141 of 628 sweeps in session 715093703 and none in the other two sessions. Direction tuning curves were computed at each unit's preferred temporal frequency (the one with the strongest mean response). Orientation selectivity was quantified with the global orientation selectivity index (gOSI, the normalized vector sum at twice the angle, i.e. 1 minus circular variance) and direction selectivity with gDSI. Significance was assessed with a permutation test (500 shuffles of direction labels within each temporal frequency, p < 0.01); the null distribution recomputes the full statistic, including preferred-TF selection, so the test is properly matched to the analysis.

## Key Finding

622 of 1389 visual-cortex units (45%) were significantly orientation-selective, with consistent fractions across sessions (42%, 45%, 49%) and across all five visual areas (38% in VISrl to 48% in VISl). Median gOSI was 0.20 across all units and 0.30 among significant ones. Most selective units respond to both drift directions of the same orientation axis (high gOSI, low gDSI), the classic orientation-selective but direction-insensitive signature of V1, and preferred orientations tile the full 0 to 180 degree range (visible as the two diagonal bands, 180 degrees apart, in the sorted population heatmap).

## Files

- `orientation_selectivity_analysis.py`: consolidated jupytext script, runs end-to-end
- `orientation_selectivity_analysis.ipynb`: the same analysis as a Jupyter notebook
- `fig1_example_raster_psth.png`: spike raster, per-direction PSTH, and tuning curve for an example VISp unit
- `fig2_example_tuning_curves.png`: example tuning curves (cartesian and polar) from each visual area
- `fig3_population_stats.png`: gOSI distribution, preferred-orientation histogram, gOSI vs gDSI, per-area fractions
- `fig4_population_heatmap.png`: normalized tuning of all 622 significant units, sorted by preferred orientation
- `fig5_cross_session.png`: per-session consistency of unit counts, selective fraction, and gOSI
- `orientation_selectivity_results.npz`: pooled per-unit results

## Running

```
python orientation_selectivity_analysis.py
```

Requires `pynapple`, `pynwb`, `lindi`, `h5py`, `numpy`, `matplotlib`, `tqdm`. Data are streamed from the DANDI Archive via LINDI with local caching, so an internet connection is needed on first run.
