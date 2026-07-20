# Theta phase entrainment and precession in hippocampal place cells

This analysis demonstrates two classical signatures of CA1 hippocampal coding from a single
real recording streamed from the DANDI Archive:

1. **Theta phase entrainment** — pyramidal cells fire preferentially near a consistent phase
   of the 6–12 Hz hippocampal theta rhythm during locomotion.
2. **Theta phase precession** — within a place field, spikes occur at progressively earlier
   theta phases as the rat traverses the field (O'Keefe & Recce, 1993; Skaggs et al., 1996).

## Dataset

[**DANDI:000044**](https://dandiarchive.org/dandiset/000044) — Grosmark, Long & Buzsáki (2016),
*"Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences."*
Bilateral 128-channel silicon-probe recordings from dorsal CA1 in Long-Evans rats running on
a 1.6 m linear track.

Session analysed: `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`. 137 sorted units
(120 excitatory, 17 inhibitory) and 1.25 kHz LFP across the full 34.5 min maze epoch.
Streaming via `remfile` with disk caching — no full file download.

## Pipeline

1. Stream the NWB file, isolate the maze epoch (18079.5 – 20147.0 s).
2. Reconstruct linearised position timestamps (the SpatialSeries `rate` field actually stores
   the sampling **interval** in this file). Drop NaN tracking samples.
3. Compute speed; threshold at 0.20 m/s to define running epochs (and exclude reward
   lingering); split into left→right and right→left runs by net displacement.
4. Pick the CA1 LFP channel with the strongest theta/delta ratio (channel 95;
   theta/delta = 2.7), band-pass filter 6–12 Hz, Hilbert transform → instantaneous theta
   phase + amplitude.
5. Compute 1-D place fields (2 cm bins, Gaussian smoothing) per direction; classify place
   cells by peak rate ≥ 2 Hz, mean rate ≤ 5 Hz, and Skaggs spatial information ≥ 0.5
   bits/spike.
6. Sample theta phase at every spike during running for each unit; per-unit Rayleigh test
   and mean resultant length (MRL).
7. For each (place cell × direction) pair, restrict to in-field spikes, normalise position
   in field (flipping for right→left runs so the rat enters at 0 and exits at 1), and fit a
   circular–linear regression (Kempter) to obtain precession slope and ρ.

## Key findings

- **38 left→right place cells and 31 right→left place cells** with well-defined fields
  tiling the track (`fig02_place_field_population.png`).
- **70% (72/103) of pyramidal cells with ≥200 running spikes are significantly theta-entrained**
  (Rayleigh p < 0.001). Mean MRL = 0.13. Preferred phases cluster in the upper half of the
  theta cycle (`fig03_theta_entrainment_population.png`).
- **80% (49/61) of place cell × direction pairs show negative phase-precession slopes**
  (median = **−151°/field**, median circular–linear ρ = **−0.21**), the canonical signature of
  theta phase precession (`fig04_phase_precession_examples.png`,
  `fig05_precession_population.png`).
- **Single-traversal rasters** for the strongest example cell (unit 50) show the spike-phase
  gradient explicitly: spikes early in field traversals occur at late theta phases and shift
  toward earlier phases as the rat passes through the field
  (`fig06_example_cell_summary.png`).

## Files

| File | Contents |
| --- | --- |
| `theta_phase_precession.py` | End-to-end jupytext analysis script (percent format) |
| `theta_phase_precession.ipynb` | Same analysis as a Jupyter notebook |
| `fig01_raw_lfp_theta_position.png` | Example LFP, theta filter, phase, position, raster |
| `fig02_place_field_population.png` | Population place-field heat-maps (both directions) |
| `fig03_theta_entrainment_population.png` | Polar plot of preferred phases + MRL distribution |
| `fig04_phase_precession_examples.png` | 12 strongest precession examples |
| `fig05_precession_population.png` | Slope and ρ population histograms |
| `fig06_example_cell_summary.png` | 4-panel summary of the strongest example cell |

## Reproducing

```bash
pip install pynapple lindi remfile pynwb h5py tqdm matplotlib scipy jupytext nbconvert dandi
python theta_phase_precession.py
```

The script streams ~10 GB of LFP-channel data on demand (only 1 channel × 34 min is fully
read), with `remfile` caching to `/tmp/remfile_cache`.
