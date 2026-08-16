# Spectrotemporal Receptive Fields in the Auditory System

This analysis demonstrates spectrotemporal receptive fields (STRFs) at two stages
of the auditory system using real data streamed from the DANDI Archive. An STRF
maps a neuron's response strength over stimulus frequency and time relative to
sound onset. Because both datasets used here present discrete tone pips (and do not
store continuous stimulus waveforms), the STRF is estimated as the
baseline-subtracted per-frequency PSTH, the tone-pip map of Aertsen & Johannesma
(1981), which is the spike-triggered average of the one-hot tone ensemble.

## Datasets

- **DANDI:001262** (Heeringa et al. 2024, *Scientific Data*): single auditory
  nerve fibers in gerbil, one fiber per NWB file. The `BF` protocol is a tone-pip
  frequency scan (50 ms tones, 13-69 closely spaced frequencies around the fiber's
  characteristic frequency, 5 repetitions). We analyze 85 fibers from 47 animals,
  sampled across the 104 gerbils in the dataset.
- **DANDI:000986** (Jaramillo lab): Neuropixels recordings from mouse auditory
  cortex during passive pure-tone listening (25 ms tones at 2/4/8/16/32 kHz,
  ~1500 trials per frequency). We analyze one session (sub-LA8 ses-1, 192 units).

## What was analyzed

`strf_auditory_system.py` (jupytext; `strf_auditory_system.ipynb` is the converted
notebook) runs end-to-end:

1. **Single-fiber deep dive**: raster, STRF map, tuning curve, and PSTH at CF for
   one auditory nerve fiber (`fig1_single_fiber_strf.png`).
2. **Nerve population**: STRFs for 85 fibers; gallery across the tonotopic axis
   (`fig2_an_strf_gallery.png`); population summary (`fig3_an_population.png`).
3. **Cortical STRFs**: per-unit STRF maps for 192 cortical units, with
   tone-responsive (Wilcoxon) and frequency-modulated (Kruskal-Wallis) selection;
   gallery spanning excitatory STRFs with different dominant frequencies and
   suppressive STRFs, plus the BF-aligned population average
   (`fig4_cortical_strf_gallery.png`).
4. **Periphery vs cortex**: side-by-side STRFs and onset-latency distributions
   (`fig5_periphery_vs_cortex.png`).

## Key findings

- Auditory nerve STRFs are compact excitatory ridges at the fiber's characteristic
  frequency, beginning a few milliseconds after tone onset and sustained for the
  tone duration. The STRF-based CF reproduces the published per-fiber best
  frequency almost exactly (r = 0.994 in log space, n = 85), validating the
  estimate. Onset latency decreases with CF (in a level-matched 40-60 dB subset,
  r = -0.41, p = 0.013), the signature of the cochlear traveling wave, and tuning
  bandwidth in octaves narrows with CF (median ~0.46 octaves).
- Cortical STRFs show the same excitatory BF hotspot but with longer onset
  latency (median 17 ms vs 4.5 ms in the nerve; Mann-Whitney p < 1e-10) and more
  transient dynamics, often with suppression following the onset response,
  consistent with the additional synaptic stages between the nerve and cortex.

## Reproducing

```
python strf_auditory_system.py        # streams data via remfile, writes figs
jupytext --to notebook strf_auditory_system.py
jupyter nbconvert --to notebook --execute strf_auditory_system.ipynb
```

Dependencies: requests, h5py, remfile, numpy, pandas, scipy, matplotlib, tqdm.
Data are streamed from S3 with a local disk cache (`/tmp/remfile_cache_strf`); no
full file downloads are needed. Runtime is roughly 5 minutes, dominated by
streaming the ~90 auditory nerve files.
