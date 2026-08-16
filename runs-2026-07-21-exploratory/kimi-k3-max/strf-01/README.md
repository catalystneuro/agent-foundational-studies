# Spectrotemporal Receptive Fields in the Gerbil Auditory Nerve

## Overview

This analysis demonstrates spectrotemporal receptive fields (STRFs) in the auditory system using dandiset [001262](https://dandiarchive.org/dandiset/001262/0.241205.0959) from the DANDI Archive: *Single-unit auditory nerve fibre responses of young-adult and aging gerbils* (Heeringa et al., 2024, [Sci Data](https://doi.org/10.1038/s41597-024-03259-3)). Each NWB file in the dandiset holds the responses of one single auditory nerve fiber to a battery of acoustic stimuli: tone pips at dozens of frequencies around the fiber's characteristic frequency (CF), clicks, rate-level functions, Schroeder-phase complexes, and 60 repeats of an identical 2.4 s frozen-noise token. All data are streamed from the archive's S3 store with `remfile` behind a local disk cache; nothing is downloaded in full. Fiber selection was based on a catalog scan of all 1160 files in the dandiset: 59 fibers with at least 15 tone frequencies and at least 1500 spikes, stratified across log CF and preferring files that also contain the click and noise protocols.

The analysis follows the original formulation of the STRF by Aertsen and Johannesma (1981). For each fiber, spike times are aligned to tone onset and the per-frequency PSTHs are stacked into a frequency by time firing-rate map, with the spontaneous rate subtracted. The example fiber shows the classic structure: a compact excitatory field centered on CF whose onset arrives earlier for higher frequencies, the signature of the cochlear traveling wave. A Poisson GLM encoding model fit with NeMoS recovers the same field as a regularized regression (predicted-map correlation 0.54, McFadden pseudo-R2 0.17), which validates the linear-nonlinear encoding picture while illustrating why boxcar tone pips constrain only the integral of the temporal kernel. The frozen-noise repeats show that the fiber reproduces the same millisecond-precision spike pattern on every presentation of an unknown broadband sound (split-half correlation 0.84), the property that makes STRF estimation with naturalistic stimuli possible. Across the 59-fiber population (CFs from 450 Hz to 14.5 kHz), our CF estimates match the reported best frequencies with r = 0.99, response latency at CF decreases with CF (1.7 to 13.2 ms, log-log slope -0.16), tuning bandwidth in octaves narrows with CF, and the CF-aligned average over fibers yields the canonical STRF shape: a single excitatory subfield at CF peaking 7 to 9 ms after tone onset.

## Files

- `strf_auditory_nerve.py`: the consolidated analysis as a jupytext script (percent format); runs end to end without manual intervention.
- `strf_auditory_nerve.ipynb`: the same analysis converted with jupytext and executed with nbconvert.
- `fig1_raw_data.png`: raw electrode trace with extracted spike times, and a raster of all tone-pip sweeps of the example fiber.
- `fig2_single_fiber_strf.png`: spectrotemporal response field, tuning curve, and latency versus frequency for the example fiber.
- `fig3_glm_encoding.png`: PSTH-based map versus GLM-predicted map, with the cross-section at CF.
- `fig4_temporal_precision.png`: click PSTH, frozen-noise raster, noise-token PSTH with split-half correlation, and a zoom of independent repeat halves.
- `fig5_population_gallery.png`: spectrotemporal response fields of 8 fibers ordered by CF.
- `fig6_population_summary.png`: CF validation against reported BF, latency versus CF, bandwidth versus CF, CF-aligned average STRF, spontaneous rate by age group, and driven rate versus stimulus level.
- `population_metrics.csv`: per-fiber metrics for all 59 analyzed fibers.

## Running

```
python3 strf_auditory_nerve.py
```

Requires `pynapple`, `nemos`, `remfile`, `h5py`, `pynwb`, `pandas`, `scipy`, `matplotlib`, and `tqdm`. Network access to the DANDI S3 store is required on first run; subsequent runs reuse the disk cache at `/tmp/remfile_cache_strf`. To convert and execute the notebook:

```
jupytext --to notebook strf_auditory_nerve.py
jupyter nbconvert --to notebook --execute --inplace strf_auditory_nerve.ipynb
```
