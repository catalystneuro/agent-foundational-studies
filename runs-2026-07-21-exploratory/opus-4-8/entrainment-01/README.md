# Theta phase entrainment of hippocampal CA1 neurons

This analysis demonstrates **theta phase entrainment**, the tendency of hippocampal
neurons to fire at a preferred phase of the local theta rhythm rather than uniformly
across the cycle, using publicly available data from the DANDI Archive.

## Dataset

[**DANDI:000044**](https://dandiarchive.org/dandiset/000044) — Grosmark, Long & Buzsáki
(2016), *"Diversity in neural firing dynamics supports both rigid and learned hippocampal
sequences."* The dandiset contains bilateral silicon-probe recordings from dorsal CA1 in
rats running back and forth on a 1.6 m linear maze. Each NWB file provides the local field
potential (LFP, 1250 Hz), spike-sorted single units labelled as excitatory (pyramidal) or
inhibitory (interneuron), linearized position, and scored brain states. Two sessions were
analyzed: `sub-Achilles/Achilles-10252013` and `sub-Cicero/Cicero-09012014`. Data were
streamed directly from the DANDI S3 bucket with `remfile` plus a local disk cache; nothing
was downloaded in full.

## What was analyzed

For each session the pipeline (1) selects the CA1 LFP channel with the strongest theta by
theta/delta power ratio, (2) defines RUN epochs from the animal's speed on the maze, (3)
extracts the instantaneous theta phase from the Hilbert transform of the 6–12 Hz
band-passed LFP, and (4) reads the theta phase at every spike during RUN. Each neuron's
locking is summarized by the mean resultant length (MRL, 0 = uniform, 1 = perfect
locking), its preferred phase, and the Rayleigh test for non-uniformity. Pyramidal cells
and interneurons are compared, and the population result is confirmed on the second animal.

## Key finding

CA1 neurons are strongly and consistently entrained to the theta rhythm during running.
Pooling both sessions, 87% of pyramidal cells and 100% of interneurons were significantly
phase-locked (Rayleigh p < 0.05). Interneurons were locked more strongly than pyramidal
cells (median MRL ≈ 0.19 vs ≈ 0.11), and each cell type had a distinct preferred phase
(pyramidal near the theta trough, interneurons on the descending flank). Individual cells
reach MRL values of 0.3–0.5 with Rayleigh p-values below 1e-50. This reproduces the
canonical hippocampal finding that principal cells and interneurons discharge at preferred,
cell-type-specific phases of theta.

## Files

- `theta_phase_entrainment.py` — consolidated jupytext script (runs end-to-end).
- `theta_phase_entrainment.ipynb` — notebook conversion of the script.
- `fig1_lfp_theta_phase.png` — raw LFP, 6–12 Hz theta, and instantaneous phase (validation).
- `fig2_entrained_raster.png` — spikes of an entrained interneuron colored by theta phase.
- `fig3_example_polar.png` — polar phase-tuning curves for example pyramidal cells and interneurons.
- `fig4_population_phase_hist.png` — pooled phase histograms per cell type over two theta cycles.
- `fig5_population_summary.png` — MRL distributions, fraction locked, and preferred phases across both sessions.

## Reproducing

```bash
python theta_phase_entrainment.py     # regenerates all five figures
```

Requires `pynapple`, `pynwb`, `remfile`, `h5py`, `dandi`, `scipy`, `numpy`, `matplotlib`.
